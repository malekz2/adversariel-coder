"""Two models, one LoRA adapter each, trained with GRPO in MLX (Apple Silicon).

Same idea and same maths as policy.Policy, but here the Maker and the Solver are DIFFERENT
models (e.g. MiniCPM5-2B and K2-Horizon-3.7B, both 4-bit MLX). Each keeps its quantised
weights frozen and trains only its own LoRA layers (mlx-lm's LoRALinear on top of
QuantizedLinear = QLoRA). The loss is the clipped GRPO surrogate from grpo.policy_loss,
rewritten in MLX.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .grpo import Sample

LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _chunk_logprobs(head, h: mx.array, tgt: mx.array) -> mx.array:
    logits = head(h).astype(mx.float32)
    return mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)


class MLXPolicy:
    def __init__(self, paths: dict[str, str], lora_cfg: dict, optim_cfg: dict, trust_remote_code: bool = False,
                 grad_checkpoint: bool = True, logit_chunk: int = 512, guard=None, mlx_memory_fraction: float = 0.6):
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from .safety import limit_mlx_memory
        self.paths, self.logit_chunk, self.grad_ckpt, self.guard = paths, logit_chunk, grad_checkpoint, guard
        self.memory = limit_mlx_memory(mlx_memory_fraction)  # before any weights are loaded
        r = int(lora_cfg.get("r", 16))
        alpha = float(lora_cfg.get("alpha", 32))
        # mlx-lm takes the LoRA multiplier directly: alpha/sqrt(r) = rsLoRA, alpha/r = classic LoRA
        self.lora = {"rank": r, "dropout": float(lora_cfg.get("dropout", 0.0)),
                     "scale": alpha / (r ** 0.5) if lora_cfg.get("use_rslora", True) else alpha / r}
        self.lr = {"solver": float(optim_cfg.get("lr_solver", 1e-5)), "maker": float(optim_cfg.get("lr_maker", 1e-5))}
        self.weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        self.models, self.opt, self.keys = {}, {}, {}
        for role, path in paths.items():
            if guard:
                guard.check(f"loading the {role} model", force=True)
            model, _ = load(path, tokenizer_config={"trust_remote_code": trust_remote_code},
                            trust_remote_code=trust_remote_code)
            model.freeze()
            present = {k for k, _ in model.layers[0].named_modules()}
            keys = [k for k in LORA_KEYS if k in present]
            linear_to_lora_layers(model, len(model.layers), {**self.lora, "keys": keys})
            self.models[role], self.keys[role] = model, keys
            self.opt[role] = self._make_optimizer(role)
        self.device = "mps"

    def _make_optimizer(self, role):
        return optim.AdamW(learning_rate=self.lr[role], betas=[0.9, 0.99], weight_decay=self.weight_decay)

    # ------------------------------------------------------------ interface shared with policy.Policy
    def num_trainable(self) -> dict:
        return {r: sum(v.size for _, v in tree_flatten(m.trainable_parameters())) for r, m in self.models.items()}

    def use_adapter(self, name: str):
        pass  # each role is its own model

    def offload(self) -> bool:
        return False  # unified memory: nothing to move

    def onload(self):
        pass

    # ------------------------------------------------------------ log-probs
    @staticmethod
    def _head(model):
        head = getattr(model, "lm_head", None)
        return head if head is not None else model.model.embed_tokens.as_linear

    def token_logprobs(self, model, input_ids: list[int], start: int) -> mx.array:
        """log p(token_i | tokens_<i) for i in [start, len), chunked over the vocabulary-sized logits."""
        x = mx.array([input_ids])
        h = model.model(x)[0]
        hs, tgt = h[start - 1:len(input_ids) - 1], x[0, start:]
        head = self._head(model)
        parts = []
        for c in range(0, hs.shape[0], self.logit_chunk):
            t = tgt[c:c + self.logit_chunk]  # bound here, not passed in: checkpoint differentiates every input
            parts.append(mx.checkpoint(lambda hh, t=t: _chunk_logprobs(head, hh, t))(hs[c:c + self.logit_chunk]))
        return mx.concatenate(parts)

    @contextmanager
    def _checkpointing(self, model):
        """Gradient checkpointing only while training (it would slow down every generated token)."""
        if not self.grad_ckpt:
            yield
            return
        from mlx_lm.tuner.trainer import grad_checkpoint
        cls = type(model.layers[0])
        original = cls.__call__
        grad_checkpoint(model.layers[0])
        try:
            yield
        finally:
            cls.__call__ = original

    # ------------------------------------------------------------ GRPO update
    def train_step(self, role: str, samples: list[Sample], grad_clip: float = 1.0, max_seq_len: int = 8192,
                   ppo_epochs: int = 1, clip_eps: float = 0.2, kl_beta: float = 0.0, lr_scale: float = 1.0) -> dict:
        model, opt = self.models[role], self.opt[role]
        opt.learning_rate = self.lr[role] * lr_scale
        live = [s for s in samples if s.advantage != 0 and any(s.loss_mask)]
        usable = [s for s in live if len(s.input_ids) <= max_seq_len]
        stats = {"samples": len(usable), "skipped_long": len(live) - len(usable), "loss": 0.0, "grad_norm": 0.0,
                 "tokens": 0, "clip_frac": 0.0, "kl": 0.0}
        if not usable:
            return stats
        total = sum(sum(s.loss_mask) for s in usable)
        stats["tokens"] = total

        def loss_fn(model, ids, start, mask, adv):
            lp = self.token_logprobs(model, ids, start)
            ratio = mx.exp(lp - mx.stop_gradient(lp))   # one update per batch: on-policy, ratio == 1
            per_tok = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
            return (per_tok * mask).sum() / total

        value_and_grad = nn.value_and_grad(model, loss_fn)
        grads, loss_total = None, 0.0
        with self._checkpointing(model):
            for s in usable:
                if self.guard:
                    self.guard.check(f"{role} training", force=True)
                st = s.loss_mask.index(1)
                mask = mx.array(s.loss_mask[st:], dtype=mx.float32)
                loss, g = value_and_grad(model, s.input_ids, st, mask, s.advantage)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                mx.eval(grads, loss)                     # bound memory: one sequence at a time
                loss_total += loss.item()
        grads, norm = optim.clip_grad_norm(grads, grad_clip)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
        mx.clear_cache()
        stats.update(loss=loss_total, grad_norm=float(norm.item()))
        return stats

    # ------------------------------------------------------------ persistence
    def save(self, ckpt_dir: Path, with_optimizer: bool = True) -> dict[str, str]:
        out = {}
        for role, model in self.models.items():
            d = Path(ckpt_dir) / role
            d.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(d / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            (d / "adapter_config.json").write_text(json.dumps(
                {**self.lora, "keys": self.keys[role], "base_model": self.paths[role]}, indent=2), encoding="utf-8")
            if with_optimizer and self.opt[role].state:
                state = {k: v for k, v in tree_flatten(self.opt[role].state) if isinstance(v, mx.array)}
                mx.save_safetensors(str(d / "optimizer.safetensors"), state)
            out[role] = str(d)
        return out

    def load(self, ckpt_dir: Path, names=("solver", "maker"), with_optimizer: bool = True):
        for role in names:
            d = Path(ckpt_dir) / role
            model = self.models[role]
            model.load_weights(str(d / "adapters.safetensors"), strict=False)
            opt_file = d / "optimizer.safetensors"
            if with_optimizer and opt_file.exists():
                self.opt[role].state = tree_unflatten(list(mx.load(str(opt_file)).items()))
            elif not with_optimizer:  # fresh optimizer moments after a reset
                self.opt[role] = self._make_optimizer(role)
            mx.eval(model.parameters())
