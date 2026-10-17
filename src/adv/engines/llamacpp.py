"""Rollouts with llama.cpp: the quantised GGUF base + two hot-swappable LoRA adapters.

Why llama.cpp for sampling? It runs the GGUF you chose (UD-IQ4_XS for 27B), it is fast on
CPU, and it can hold several LoRA adapters at once and pick one PER REQUEST
("lora": [{"id": 0, "scale": 1.0}, ...]). After each training step the PEFT adapters are
converted to GGUF and the server restarts with them.

Note: the sampler (IQ4_XS GGUF) and the learner (NF4 safetensors) are two different
quantisations of the same weights, so the sampled tokens come from a very slightly different
policy than the one being trained. This "sampler/learner mismatch" also exists in big RL
systems (vLLM vs FSDP); at one update per batch it is small and we ignore it.

Thinking-only bias is done in two passes:
  pass 1: generate with logit_bias until "</think>" (or the think budget runs out)
  pass 2: continue from prompt + pass-1 tokens WITHOUT bias (prompt cache makes it cheap)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .base import Engine, GenParams, GenResult, count_markers


def convert_adapter_to_gguf(peft_dir: str, base_model_dir: str, out_file: str, convert_script: str,
                            python: str | None = None, gguf_py: str | None = None) -> str:
    env = dict(os.environ)
    if gguf_py:
        env["PYTHONPATH"] = gguf_py + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [python or sys.executable, convert_script, "--base", base_model_dir, "--outtype", "f16",
           "--outfile", out_file, peft_dir]
    # stdin=DEVNULL: for models with custom code (K2-Horizon) the converter asks "run the custom code? [y/N]";
    # with no input it takes the default and continues instead of waiting forever.
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not Path(out_file).exists():
        raise RuntimeError(f"LoRA->GGUF conversion failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return out_file


class LlamaCppEngine(Engine):
    def __init__(self, tok, server_exe: str, gguf_path: str, port: int = 8181, ctx_per_slot: int = 8192,
                 parallel: int = 4, gpu_layers: str = "auto", threads: int | None = None,
                 log_file: str | None = None, extra_args: list[str] | None = None,
                 converter: dict | None = None, adapter_gguf_dir: str | None = None,
                 swap_during_training: bool = False, startup_timeout: float = 900, spec_args: list[str] | None = None):
        self.tok = tok
        self.server_exe = server_exe
        self.gguf_path = gguf_path
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.ctx_per_slot = ctx_per_slot
        self.max_parallel = parallel
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.log_file = log_file
        self.extra_args = extra_args or []
        self.spec_args = spec_args or []      # KV type + speculative flags chosen by rollout_tune
        self.converter = converter or {}
        self.adapter_gguf_dir = Path(adapter_gguf_dir) if adapter_gguf_dir else None
        self.swap_during_training = swap_during_training
        self.startup_timeout = startup_timeout
        self.adapter_files: dict[str, str] = {}   # name -> gguf path
        self.adapter_ids: dict[str, int] = {}     # name -> server id
        self.proc: subprocess.Popen | None = None
        self.session = requests.Session()

    # ------------------------------------------------------------ server lifecycle
    def _cmd(self) -> list[str]:
        cmd = [self.server_exe, "-m", self.gguf_path, "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(self.ctx_per_slot * self.max_parallel), "-np", str(self.max_parallel),
               "-ngl", str(self.gpu_layers), "--no-webui", "--ctx-checkpoints", "32",
               "--checkpoint-min-step", "0"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        if self.log_file:
            cmd += ["--log-file", self.log_file]
        files = [self.adapter_files[n] for n in ("maker", "solver") if n in self.adapter_files]
        if files:  # one comma-separated --lora (repeating the flag keeps only the last one)
            cmd += ["--lora", ",".join(files), "--lora-init-without-apply"]
        return cmd + self.extra_args + self.spec_args

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        out = open(self.log_file + ".stdout", "ab") if self.log_file else subprocess.DEVNULL
        self.proc = subprocess.Popen(self._cmd(), stdout=out, stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.startup_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_file}")
            try:
                if self.session.get(self.url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        else:
            raise RuntimeError("llama-server did not become healthy in time")
        self._read_adapter_ids()

    def _read_adapter_ids(self):
        self.adapter_ids = {}
        if not self.adapter_files:
            return
        listed = self.session.get(self.url + "/lora-adapters", timeout=10).json()
        by_path = {os.path.normcase(os.path.abspath(a["path"])): a["id"] for a in listed}
        for name, path in self.adapter_files.items():
            key = os.path.normcase(os.path.abspath(path))
            if key not in by_path:
                raise RuntimeError(f"adapter {name} not reported by llama-server: {listed}")
            self.adapter_ids[name] = by_path[key]

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def pause(self):
        if self.swap_during_training:
            self.stop()

    def resume(self):
        self.start()

    def sync_adapters(self, adapter_dirs: dict[str, str], restart: bool = True):
        """Convert PEFT adapters to GGUF and (re)start the server with them (or leave it stopped)."""
        assert self.adapter_gguf_dir is not None
        self.adapter_gguf_dir.mkdir(parents=True, exist_ok=True)
        new_files = {}

        def convert(name, peft_dir):
            out = str(self.adapter_gguf_dir / f"{name}.gguf")
            convert_adapter_to_gguf(peft_dir, self.converter["base_model_dir"], out + ".tmp",
                                    self.converter["script"], self.converter.get("python"),
                                    self.converter.get("gguf_py"))
            return name, (out + ".tmp", out)

        # both adapters at once: each conversion is a separate Python process (mostly start-up time)
        with ThreadPoolExecutor(max_workers=len(adapter_dirs) or 1) as ex:
            for name, files in ex.map(lambda kv: convert(*kv), adapter_dirs.items()):
                new_files[name] = files
        self.stop()  # the server holds the old files open (Windows cannot replace them otherwise)
        for name, (tmp, out) in new_files.items():
            os.replace(tmp, out)
            self.adapter_files[name] = out
        if restart:
            self.start()

    # ------------------------------------------------------------ generation
    def _complete(self, prompt_ids, n_predict, params: GenParams, bias: dict, stop: list[str]) -> dict:
        body = {
            "prompt": prompt_ids, "n_predict": int(n_predict), "temperature": params.temperature,
            "top_p": params.top_p, "top_k": params.top_k, "min_p": 0.0, "return_tokens": True,
            "cache_prompt": True, "stop": stop,
            "logit_bias": [[int(t), float(b)] for t, b in bias.items()],
        }
        if params.seed is not None:
            body["seed"] = params.seed
        if self.adapter_ids:
            body["lora"] = [{"id": i, "scale": 1.0 if name == params.adapter else 0.0}
                            for name, i in self.adapter_ids.items()]
        r = self.session.post(self.url + "/completion", json=body, timeout=3600)
        r.raise_for_status()
        return r.json()

    def generate(self, prompt_ids, params: GenParams, think_end: int | None) -> GenResult:
        tok = self.tok
        out, mask = [], []
        think_tokens, forced, markers = 0, False, 0
        if think_end is not None:
            budget = max(1, min(params.think_budget, params.max_new_tokens))
            r = self._complete(prompt_ids, budget, params, params.think_bias, [tok.decode([think_end])])
            toks = list(r.get("tokens") or [])
            stop_type = r.get("stop_type")
            if think_end in toks:                     # closed naturally (stop string not matched)
                toks = toks[:toks.index(think_end) + 1]
                out, mask = toks, [1] * len(toks)
            elif stop_type == "word":                 # closed naturally, server cut the stop word
                out, mask = toks + [think_end], [1] * (len(toks) + 1)
            elif stop_type == "eos":                  # ended the whole answer inside <think>
                out, mask = toks + [tok.im_end], [1] * (len(toks) + 1)
                return GenResult(out, mask, "stop", len(toks), False, count_markers(toks, params.think_bias))
            else:                                     # budget exhausted: force the end of thinking
                out, mask = toks + [think_end], [1] * len(toks) + [0]
                forced = True
            think_tokens = len(out) - 1
            markers = count_markers(out, params.think_bias)
        remaining = params.max_new_tokens - len(out)
        if remaining <= 0:
            return GenResult(out, mask, "length", think_tokens, forced, markers)
        r = self._complete(prompt_ids + out, remaining, params, {}, [])
        toks = list(r.get("tokens") or [])
        finish = "stop" if r.get("stop_type") in ("eos", "word") else "length"
        if finish == "stop" and (not toks or toks[-1] not in tok.stop_ids):
            toks.append(tok.im_end)  # the server omits the end-of-turn token it sampled
        out += toks
        mask += [1] * len(toks)
        return GenResult(out, mask, finish, think_tokens, forced, markers)
