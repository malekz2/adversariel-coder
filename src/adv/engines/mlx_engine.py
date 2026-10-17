"""Rollouts with MLX on Apple Silicon, straight from the models being trained.

On the Mac the Maker and the Solver are two different models (each with its own LoRA adapter),
and both live in this process: sampling uses the current adapters directly, so there is
nothing to convert or sync after a training step, and no sampler/learner mismatch.

MLX generates one sequence at a time here (a lock serialises callers); Pi attempts still
overlap their tool calls and tests with other attempts' generation.
"""
from __future__ import annotations

import threading

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from .base import Engine, GenParams, GenResult, count_markers


class ThinkingOnlyBiasMLX:
    """MLX version of engines.hf.ThinkingOnlyBias: bias marker tokens while thinking, force the
    closing tag when the think budget is spent, and leave the answer untouched."""

    def __init__(self, prompt_len: int, think_end: int | None, bias: dict, budget: int):
        self.prompt_len = prompt_len
        self.think_end = think_end
        self.active = think_end is not None
        self.budget = budget
        self.forced_at: int | None = None
        self.ids = mx.array(list(bias.keys()), dtype=mx.int32) if bias else None
        self.vals = mx.array(list(bias.values()), dtype=mx.float32) if bias else None

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if not self.active:
            return logits
        n = tokens.size - self.prompt_len            # tokens generated so far
        if n > 0 and int(tokens[-1].item()) == self.think_end:
            self.active = False                      # thinking is over: no more bias
            return logits
        if n >= self.budget:
            hot = mx.arange(logits.shape[-1]) == self.think_end
            self.forced_at = n
            self.active = False
            return mx.where(hot, mx.zeros_like(logits), mx.full(logits.shape, -mx.inf, dtype=logits.dtype))
        if self.ids is not None:
            logits[:, self.ids] += self.vals.astype(logits.dtype)
        return logits


class MLXEngine(Engine):
    def __init__(self, toks: dict, policy, max_parallel: int = 4, guard=None):
        self.guard = guard        # safety.MemoryGuard: stop before the Mac runs out of memory
        self.toks = toks          # role -> ChatTokenizer
        self.policy = policy      # MLXPolicy: role -> model with LoRA layers
        self.max_parallel = max_parallel
        self.lock = threading.Lock()

    def generate(self, prompt_ids, params: GenParams, think_end: int | None) -> GenResult:
        tok = self.toks[params.adapter]
        model = self.policy.models[params.adapter]
        budget = min(params.think_budget, params.max_new_tokens)
        if self.guard:
            self.guard.check(f"{params.adapter} generation", force=True)
        with self.lock:
            if params.seed is not None:
                mx.random.seed(params.seed)
            proc = ThinkingOnlyBiasMLX(len(prompt_ids), think_end, params.think_bias, budget)
            sampler = make_sampler(temp=params.temperature, top_p=params.top_p, top_k=params.top_k)
            out: list[int] = []
            for token, _ in generate_step(mx.array(prompt_ids), model, max_tokens=params.max_new_tokens,
                                          sampler=sampler, logits_processors=[proc]):
                out.append(int(token))
                if out[-1] in tok.stop_ids:
                    break
                if self.guard and len(out) % 32 == 0:
                    self.guard.check(f"{params.adapter} generation")
        mask = [1] * len(out)
        if proc.forced_at is not None and proc.forced_at < len(mask):
            mask[proc.forced_at] = 0
        finish = "stop" if out and out[-1] in tok.stop_ids else "length"
        think = 0
        if think_end is not None:
            think = out.index(think_end) if think_end in out else len(out)
        return GenResult(out, mask, finish, think, proc.forced_at is not None,
                         count_markers(out[:think], params.think_bias))
