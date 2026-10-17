"""OpenAI-compatible recording proxy: Pi -> (this) -> engine.

Agentic RL needs the exact tokens the model produced, but a harness like Pi only speaks
chat messages. This proxy:

1. renders Pi's messages with the model's own chat template,
2. CHAINS turns: on turn t+1 the prompt is "tokens of turn t (exactly as sampled) + newly
   rendered tool results", so the whole episode becomes ONE token sequence. Training then
   costs one forward pass per attempt instead of one per turn, and never re-tokenises text
   the model wrote,
3. picks the LoRA adapter (model "solver" or "maker"),
4. applies the overthinking bias only inside <think>,
5. parses Qwen's XML tool calls back into OpenAI JSON for Pi,
6. records everything per rollout (Pi sends the rollout id as its API key).

The loss mask marks which tokens the model generated (1) versus prompt/tool output (0).
"""
from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engines.base import Engine, GenParams, GenResult
from .tokenizer import normalize_messages, parse_tool_calls, split_completion, thinking_open


@dataclass
class Turn:
    prompt_len: int
    gen: GenResult
    chained: bool
    reasoning: str = ""
    content: str = ""
    tool_calls: list = field(default_factory=list)


class Rollout:
    def __init__(self, rid: str, role: str, max_turns: int):
        self.rid, self.role, self.max_turns = rid, role, max_turns
        self.turns: list[Turn] = []
        self.requests = 0
        self.seq_ids: list[int] | None = None   # current chained sequence
        self.seq_mask: list[int] | None = None
        self.segments: list[tuple[list[int], list[int]]] = []
        self.truncated = False                  # hit turn or context limit
        self.lock = threading.Lock()

    def sequences(self) -> list[tuple[list[int], list[int]]]:
        segs = list(self.segments)
        if self.seq_ids is not None:
            segs.append((self.seq_ids, self.seq_mask))
        return segs

    @property
    def think_tokens(self) -> int:
        return sum(t.gen.think_tokens for t in self.turns)

    @property
    def gen_tokens(self) -> int:
        return sum(len(t.gen.token_ids) for t in self.turns)


class RecordingProxy:
    def __init__(self, tok, engine: Engine, params: dict[str, GenParams], max_ctx: int = 16384,
                 host: str = "127.0.0.1", port: int = 8282, min_new_tokens: int = 256):
        self.tok, self.engine, self.params = tok, engine, params
        self.max_ctx, self.min_new_tokens = max_ctx, min_new_tokens
        self.host, self.port = host, port
        self.rollouts: dict[str, Rollout] = {}
        self.lock = threading.Lock()
        self.httpd: ThreadingHTTPServer | None = None

    # -------------------------------------------------------------- rollout bookkeeping
    def begin(self, role: str, max_turns: int = 1, rid: str | None = None) -> str:
        rid = rid or uuid.uuid4().hex
        with self.lock:
            self.rollouts[rid] = Rollout(rid, role, max_turns)
        return rid

    def end(self, rid: str) -> Rollout:
        with self.lock:
            return self.rollouts.pop(rid)

    # -------------------------------------------------------------- the core
    def _final(self, text: str) -> dict:
        return {"message": {"role": "assistant", "content": text}, "finish_reason": "stop",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    def chat(self, rid: str, messages: list[dict], tools: list | None = None,
             max_tokens: int | None = None) -> dict:
        ro = self.rollouts.get(rid)
        if ro is None:
            raise KeyError(f"unknown rollout {rid}")
        with ro.lock:
            return self._chat_locked(ro, messages, tools, max_tokens)

    def _chat_locked(self, ro: Rollout, messages, tools, max_tokens) -> dict:
        tok = self.tok
        ro.requests += 1
        if ro.requests > ro.max_turns:
            ro.truncated = True
            return self._final("Turn limit reached; stopping here.")
        msgs = normalize_messages(messages)
        full_text = tok.render_text(msgs, tools, add_generation_prompt=True)

        prompt_ids, chained, delta_mask_prefix = None, False, []
        last_asst = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None)
        if ro.seq_ids is not None and last_asst is not None:
            core = tok.render_text(msgs[:last_asst + 1], tools, add_generation_prompt=False).rstrip("\n")
            if core.endswith(tok.im_end_text) and full_text.startswith(core):
                base, base_mask = list(ro.seq_ids), list(ro.seq_mask)
                if base[-1] not in tok.stop_ids:          # previous turn was cut off at max tokens
                    base.append(tok.im_end)
                    base_mask.append(0)
                delta = tok.encode(full_text[len(core):])
                prompt_ids = base + delta
                delta_mask_prefix = base_mask + [0] * len(delta)
                chained = True
        if prompt_ids is None:
            prompt_ids = tok.encode(full_text)
            delta_mask_prefix = [0] * len(prompt_ids)

        room = self.max_ctx - len(prompt_ids)
        if room < self.min_new_tokens:
            ro.truncated = True
            return self._final("Context limit reached; stopping here.")

        params = copy.copy(self.params[ro.role])
        params.max_new_tokens = min(params.max_new_tokens, room, max_tokens or 10**9)
        think_end = thinking_open(tok, prompt_ids)   # which tag closes the thinking (model-specific)
        gen = self.engine.generate(prompt_ids, params, think_end)

        # record (and chain) the sequence
        if chained:
            ro.seq_ids = prompt_ids + gen.token_ids
            ro.seq_mask = delta_mask_prefix + gen.loss_mask
        else:
            if ro.seq_ids is not None:
                ro.segments.append((ro.seq_ids, ro.seq_mask))
            ro.seq_ids = prompt_ids + gen.token_ids
            ro.seq_mask = delta_mask_prefix + gen.loss_mask

        reasoning, answer = split_completion(tok, gen.token_ids, think_end)
        content, calls = parse_tool_calls(answer, tools) if tools else (answer, [])
        ro.turns.append(Turn(len(prompt_ids), gen, chained, reasoning, content, calls))
        finish = "tool_calls" if calls else ("length" if gen.finish_reason == "length" else "stop")
        msg = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
        if calls:
            msg["tool_calls"] = calls
        return {"message": msg, "finish_reason": finish,
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(gen.token_ids),
                          "total_tokens": len(prompt_ids) + len(gen.token_ids)}}

    # -------------------------------------------------------------- HTTP server
    def start_http(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code: int, obj: dict):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._json(200, {"object": "list", "data": [{"id": r, "object": "model"} for r in proxy.params]})
                elif self.path.rstrip("/").endswith("/health"):
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._json(404, {"error": "not found"})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                rid = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                model = body.get("model", "solver")
                if rid not in proxy.rollouts:  # manual use (e.g. curl): open an ad-hoc rollout
                    rid = proxy.begin(model if model in proxy.params else "solver", max_turns=10**6,
                                      rid=rid or None)
                max_tok = body.get("max_completion_tokens") or body.get("max_tokens")
                if body.get("stream"):
                    self._stream(rid, body, model, max_tok)
                else:
                    try:
                        res = proxy.chat(rid, body.get("messages", []), body.get("tools"), max_tok)
                    except Exception as e:  # report instead of hanging the client
                        return self._json(500, {"error": {"message": str(e)}})
                    self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion",
                                     "created": int(time.time()), "model": model,
                                     "choices": [{"index": 0, "message": res["message"],
                                                  "finish_reason": res["finish_reason"]}],
                                     "usage": res["usage"]})

            def _stream(self, rid, body, model, max_tok):
                cid = "chatcmpl-" + uuid.uuid4().hex[:16]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                wlock = threading.Lock()

                def send(obj):
                    with wlock:
                        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                        self.wfile.flush()

                def chunk(delta, finish=None):
                    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

                send(chunk({"role": "assistant"}))
                done = threading.Event()

                def heartbeat():  # SSE comments keep slow CPU generations from timing out
                    while not done.wait(15):
                        try:
                            with wlock:
                                self.wfile.write(b": keep-alive\n\n")
                                self.wfile.flush()
                        except OSError:
                            return
                threading.Thread(target=heartbeat, daemon=True).start()
                try:
                    res = proxy.chat(rid, body.get("messages", []), body.get("tools"), max_tok)
                except Exception as e:
                    res = {"message": {"role": "assistant", "content": f"[proxy error: {e}]"},
                           "finish_reason": "stop", "usage": {}}
                finally:
                    done.set()
                msg = res["message"]
                if msg.get("reasoning_content"):
                    send(chunk({"reasoning_content": msg["reasoning_content"]}))
                if msg.get("content"):
                    send(chunk({"content": msg["content"]}))
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    send(chunk({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                                "function": tc["function"]}]}))
                send(chunk({}, res["finish_reason"]))
                if res.get("usage"):
                    send({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                          "model": model, "choices": [], "usage": res["usage"]})
                with wlock:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                self.close_connection = True

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop_http(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"
