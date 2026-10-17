"""Chat templating, thinking-token bookkeeping and tool-call parsing.

Everything that turns messages into token ids (and model output back into messages)
lives here, so the proxy, the trainer and the engines all agree on the exact tokens.
In RL this matters a lot: if you re-tokenise text differently from what the model
actually sampled, you train on tokens the model never produced.
"""
from __future__ import annotations

import json
import re
import uuid

TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
IFM_CALL_RE = re.compile(r"<ifm\|tool_call>\s*([^\s<]+)\s*(.*?)</ifm\|tool_call>", re.S)
IFM_ARG_RE = re.compile(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*(?:<ifm\|arg_type>.*?</ifm\|arg_type>\s*)?"
                        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>", re.S)
TOOL_START_MARKERS = ("<tool_call>", "<ifm|tool_calls>", "<ifm|tool_call>")

# Chat formats differ per model family: end-of-turn token and thinking tags. The first format
# whose end-of-turn marker is a single token in the vocabulary wins.
CHAT_FORMATS = [
    {"name": "qwen", "eot": "<|im_end|>", "think": [("<think>", "</think>")]},
    {"name": "k2_horizon", "eot": "<|ifm|im_end|>",
     # reasoning_effort high / medium / low opens <ifm|think> / <ifm|think_fast> / <ifm|think_faster>
     "think": [("<ifm|think>", "</ifm|think>"), ("<ifm|think_fast>", "</ifm|think_fast>"),
               ("<ifm|think_faster>", "</ifm|think_faster>")]},
]


class ChatTokenizer:
    """Wraps a Hugging Face tokenizer (the one shipped with the trained model)."""

    def __init__(self, path: str, enable_thinking: bool = True, reasoning_effort: str | None = None,
                 trust_remote_code: bool = False):
        from transformers import AutoTokenizer
        # trust_remote_code: models with their own code (K2-Horizon) would otherwise stop and ask y/N
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        fmt = next((f for f in CHAT_FORMATS if self._single_id(f["eot"]) is not None), None)
        if fmt is None:
            raise ValueError(f"unknown chat format for {path}: add its end-of-turn/thinking tokens to CHAT_FORMATS")
        self.format = fmt["name"]
        self.im_end_text = fmt["eot"]
        self.im_end = self._single_id(fmt["eot"])
        self.think_pairs = {}  # start-token id -> matching end-token id
        for start, end in fmt["think"]:
            a, b = self._single_id(start), self._single_id(end)
            if a is not None and b is not None:
                self.think_pairs[a] = b
        self.think_start, self.think_end = next(iter(self.think_pairs.items()), (None, None))
        self.stop_ids = {i for i in (self.im_end, self.tok.eos_token_id) if i is not None}

    def _single_id(self, text: str):
        ids = self.tok.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def render_text(self, messages: list[dict], tools: list | None, add_generation_prompt: bool) -> str:
        kwargs = {"enable_thinking": self.enable_thinking}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self.tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                            add_generation_prompt=add_generation_prompt, **kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


class MockTokenizer:
    """Character-level stand-in with a Qwen-like template, for the dry run and tests."""

    SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>"]
    BASE = 0x110000  # above every unicode code point

    def __init__(self, enable_thinking: bool = True):
        self.enable_thinking = enable_thinking
        self.sid = {s: self.BASE + i for i, s in enumerate(self.SPECIALS)}
        self.rev = {v: k for k, v in self.sid.items()}
        self.format = "qwen"
        self.think_start = self.sid["<think>"]
        self.think_end = self.sid["</think>"]
        self.think_pairs = {self.think_start: self.think_end}
        self.im_end = self.sid["<|im_end|>"]
        self.im_end_text = "<|im_end|>"
        self.stop_ids = {self.im_end}
        self._split = re.compile("(" + "|".join(re.escape(s) for s in self.SPECIALS) + ")")

    @property
    def vocab_size(self) -> int:
        return self.BASE + len(self.SPECIALS)

    def render_text(self, messages, tools, add_generation_prompt):
        out = []
        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        if tools:
            sys_parts.insert(0, "# Tools\n" + "\n".join(json.dumps(t) for t in tools))
        if sys_parts:
            out.append("<|im_start|>system\n" + "\n\n".join(sys_parts) + "<|im_end|>\n")
        last_user = max([i for i, m in enumerate(messages) if m["role"] == "user"], default=-1)
        for i, m in enumerate(messages):
            if m["role"] == "user":
                out.append("<|im_start|>user\n" + m["content"] + "<|im_end|>\n")
            elif m["role"] == "assistant":
                body = m.get("content") or ""
                if i > last_user:
                    body = "<think>\n" + (m.get("reasoning_content") or "").strip() + "\n</think>\n\n" + body
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in fn["arguments"].items())
                    body += f"<tool_call>\n<function={fn['name']}>\n{params}</function>\n</tool_call>"
                out.append("<|im_start|>assistant\n" + body + "<|im_end|>\n")
            elif m["role"] == "tool":
                out.append("<|im_start|>user\n<tool_response>\n" + str(m.get("content")) + "\n</tool_response><|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n" + ("<think>\n" if self.enable_thinking else "<think>\n\n</think>\n\n"))
        return "".join(out)

    def encode(self, text):
        ids = []
        for part in self._split.split(text):
            if part in self.sid:
                ids.append(self.sid[part])
            else:
                ids.extend(ord(c) for c in part)
        return ids

    def decode(self, ids):
        return "".join(self.rev.get(i, chr(i) if i < self.BASE else "") for i in ids)


# ------------------------------------------------------------------ helpers

def thinking_open(tok, prompt_ids: list[int]) -> int | None:
    """If the prompt ends inside an unfinished thinking block, return the token id that will close
    it (e.g. </think>); otherwise None (generation starts with the answer)."""
    ends = set(tok.think_pairs.values())
    for t in reversed(prompt_ids[-64:]):
        if t in tok.think_pairs:
            return tok.think_pairs[t]
        if t in ends:
            return None
    return None


def normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages -> what Qwen / K2-Horizon chat templates expect."""
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
            m["content"] = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        elif content is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", tc))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"raw": args}
                fn["arguments"] = args if isinstance(args, dict) else {"value": args}
                calls.append({"type": "function", "id": tc.get("id"), "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def _param_types(tools: list | None) -> dict:
    types = {}
    for t in tools or []:
        fn = t.get("function", t)
        props = (fn.get("parameters") or {}).get("properties") or {}
        types[fn.get("name")] = {k: (v.get("type") if isinstance(v, dict) else None) for k, v in props.items()}
    return types


def _coerce(value: str, typ):
    if typ in (None, "string") or (isinstance(typ, list) and "string" in typ):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_tool_calls(text: str, tools: list | None) -> tuple[str, list[dict]]:
    """Split assistant text into (content, OpenAI-style tool_calls)."""
    types = _param_types(tools)
    calls = []
    for name, body in IFM_CALL_RE.findall(text):  # K2-Horizon: <ifm|tool_call>name <ifm|arg_key>..
        args = {k.strip(): _coerce(v, types.get(name, {}).get(k.strip())) for k, v in IFM_ARG_RE.findall(body)}
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    for block in TOOL_BLOCK_RE.findall(text):
        fm = FUNC_RE.search(block)
        if fm:  # Qwen3.5 / Qwen3-Coder XML format
            name, body = fm.group(1), fm.group(2)
            args = {k: _coerce(v, types.get(name, {}).get(k)) for k, v in PARAM_RE.findall(body)}
        else:   # Qwen2.5/3 JSON format: {"name": ..., "arguments": {...}}
            try:
                d = json.loads(block)
                name, args = d.get("name"), d.get("arguments") or {}
            except json.JSONDecodeError:
                continue
        if name:
            calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args)}})
    starts = [i for i in (text.find(m) for m in TOOL_START_MARKERS) if i != -1]
    content = (text[:min(starts)] if starts else text).strip()
    return content, calls


def split_completion(tok, completion_ids: list[int], think_end: int | None) -> tuple[str, str]:
    """Return (reasoning_text, answer_text) with stop tokens removed. `think_end` is the token that
    closes the thinking block the generation started in (None = it started with the answer)."""
    ids = [t for t in completion_ids if t not in tok.stop_ids]
    if think_end is not None and think_end in ids:
        k = ids.index(think_end)
        return tok.decode(ids[:k]).strip(), tok.decode(ids[k + 1:]).strip()
    if think_end is not None:
        return tok.decode(ids).strip(), ""  # never closed its thinking
    return "", tok.decode(ids).strip()
