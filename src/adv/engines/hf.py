"""Rollouts with Hugging Face `generate` on the training model itself.

Slower than llama.cpp, but it needs nothing extra and samples from EXACTLY the weights being
trained (no sampler/learner mismatch). Used automatically if the llama.cpp LoRA path does
not work for this architecture, and as the reference implementation of the
thinking-only bias, which here is a single-pass logits processor.
"""
from __future__ import annotations

import threading

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .base import Engine, GenParams, GenResult, count_markers


class ThinkingOnlyBias(LogitsProcessor):
    """Adds the marker bias while inside <think>; forces </think> when the budget is spent."""

    def __init__(self, prompt_len: int, think_end: int, bias: dict, budget: int, active: bool, device):
        self.prompt_len = prompt_len
        self.think_end = think_end
        self.active = active and think_end is not None
        self.budget = budget
        self.forced_at: int | None = None
        self.ids = torch.tensor(list(bias.keys()), dtype=torch.long, device=device)
        self.vals = torch.tensor(list(bias.values()), dtype=torch.float32, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.active:
            return scores
        gen = input_ids[0, self.prompt_len:]
        if gen.numel() and bool((gen == self.think_end).any()):
            self.active = False          # thinking is over: no more bias
            return scores
        if gen.numel() >= self.budget:
            scores = torch.full_like(scores, float("-inf"))
            scores[:, self.think_end] = 0.0
            self.forced_at = int(gen.numel())
            self.active = False
            return scores
        if self.ids.numel():
            scores = scores.clone()
            scores[:, self.ids] += self.vals.to(scores.dtype)
        return scores


class HFEngine(Engine):
    max_parallel = 1  # one sequence at a time; the lock serialises concurrent Pi attempts

    def __init__(self, tok, policy):
        self.tok = tok
        self.policy = policy
        self.lock = threading.Lock()

    def generate(self, prompt_ids, params: GenParams, in_thinking: bool) -> GenResult:
        tok, model = self.tok, self.policy.model
        with self.lock, torch.no_grad():
            self.policy.use_adapter(params.adapter)
            was_training = model.training
            model.eval()
            device = self.policy.device
            inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            proc = ThinkingOnlyBias(len(prompt_ids), tok.think_end, params.think_bias,
                                    min(params.think_budget, params.max_new_tokens), in_thinking, device)
            gen_kwargs = dict(max_new_tokens=params.max_new_tokens, logits_processor=LogitsProcessorList([proc]),
                              eos_token_id=sorted(tok.stop_ids), pad_token_id=sorted(tok.stop_ids)[0],
                              use_cache=True)
            if params.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=params.temperature, top_p=params.top_p,
                                  top_k=params.top_k)
            else:
                gen_kwargs.update(do_sample=False)
            if params.seed is not None:
                torch.manual_seed(params.seed)
            out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), **gen_kwargs)
            if was_training:
                model.train()
        gen = out[0, len(prompt_ids):].tolist()
        for i, t in enumerate(gen):          # drop padding after the end-of-turn token
            if t in tok.stop_ids:
                gen = gen[:i + 1]
                break
        mask = [1] * len(gen)
        if proc.forced_at is not None and proc.forced_at < len(mask):
            mask[proc.forced_at] = 0
        finish = "stop" if gen and gen[-1] in tok.stop_ids else "length"
        think = 0
        if in_thinking:
            think = gen.index(tok.think_end) if tok.think_end in gen else len(gen)
        return GenResult(gen, mask, finish, think, proc.forced_at is not None,
                         count_markers(gen[:think], params.think_bias))
