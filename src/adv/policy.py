"""One frozen base model + two LoRA adapters ("maker" and "solver").

Sharing the base halves memory compared with two 27B models, and turning the adapters off
gives the original model for free (handy as a KL reference). Only the adapters train.

LoRA in one line: instead of updating a big weight W, learn a low-rank change W + (alpha/r)*B@A
with A: r x d_in, B: d_out x r. B starts at zero, so training starts exactly at the base
model. rsLoRA scales by alpha/sqrt(r) instead, which keeps updates stable as r grows.
QLoRA = the same, but the frozen W is stored in 4-bit NF4 (needs an NVIDIA GPU via bitsandbytes).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .grpo import Sample, policy_loss

LORA_TARGET_REGEX = (r"^(?!.*(?:mtp|visual|vision)).*layers\.\d+\."
                     r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
                     r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
                     r"|mlp\.(?:gate_proj|up_proj|down_proj))$")
MLP_ONLY_REGEX = r"^(?!.*(?:mtp|visual|vision)).*layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$"


def _chunk_logprobs(h: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logits = (h @ weight.t()).float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


class Policy:
    def __init__(self, model_dir: str, lora_cfg: dict, optim_cfg: dict, quant_4bit: bool = False,
                 dtype: str = "bfloat16", device: str = "cpu", gradient_checkpointing: bool = True,
                 logit_chunk: int = 512, trust_remote_code: bool = False, optimizer: str = "adamw",
                 offload_activations: bool = False):
        from peft import LoraConfig, get_peft_model
        self.device = torch.device(device)
        self.home = self.device            # where the weights live while training
        self.logit_chunk = logit_chunk
        self.optimizer_name = optimizer
        self.offload_activations = offload_activations and self.device.type == "cuda"
        self.model_dir = model_dir
        torch_dtype = getattr(torch, dtype)
        kwargs = {"dtype": torch_dtype, "low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
        if quant_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16)
        if self.device.type == "cuda":
            kwargs["device_map"] = {"": self.device.index or 0}
        base = self._load(model_dir, kwargs)
        if self.device.type != "cuda":
            base.to(self.device)
        base.config.use_cache = False
        # Not peft.prepare_model_for_kbit_training: it upcasts every 16-bit weight to fp32, including the
        # embedding and output layers, which for a 250k vocabulary costs GBs of VRAM (5 GB on K2-Horizon-3.7B).
        if gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()

        target = lora_cfg.get("target_modules", "all")
        target = {"all": LORA_TARGET_REGEX, "mlp_only": MLP_ONLY_REGEX}.get(target, target)
        lcfg = LoraConfig(r=lora_cfg.get("r", 16), lora_alpha=lora_cfg.get("alpha", 32),
                          lora_dropout=lora_cfg.get("dropout", 0.0), use_rslora=lora_cfg.get("use_rslora", True),
                          target_modules=target, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lcfg, adapter_name="solver")
        self.model.add_adapter("maker", lcfg)
        self.lr = {"solver": float(optim_cfg.get("lr_solver", 1e-5)), "maker": float(optim_cfg.get("lr_maker", 1e-5))}
        self.weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        self.params = {}
        for name in ("solver", "maker"):
            ps = [p for n, p in self.model.named_parameters() if "lora_" in n and f".{name}." in n]
            for p in ps:
                p.data = p.data.float()  # keep trainable LoRA weights in fp32
            self.params[name] = ps
        self.opt = {n: self._make_optimizer(n) for n in self.params}
        # fp16 (GPUs without bf16, e.g. RTX 20xx): scale the loss so small gradients don't round to zero
        self.scaler = ({n: torch.amp.GradScaler(self.device.type) for n in self.params}
                       if torch_dtype == torch.float16 else None)
        self.active = None
        self.use_adapter("solver")

    def _make_optimizer(self, name: str):
        ps = self.params[name]
        if self.optimizer_name == "paged_adamw_8bit" and self.device.type == "cuda":
            try:  # QLoRA's paged 8-bit Adam: ~4x smaller optimizer state, spills to RAM on memory spikes
                import bitsandbytes as bnb
                return bnb.optim.PagedAdamW8bit(ps, lr=self.lr[name], betas=(0.9, 0.99), weight_decay=self.weight_decay)
            except Exception as e:
                print(f"paged 8-bit AdamW unavailable ({e}); using torch AdamW")
        return torch.optim.AdamW(ps, lr=self.lr[name], betas=(0.9, 0.99), weight_decay=self.weight_decay)

    # ------------------------------------------------------------ GPU <-> RAM (rollout phase)
    def offload(self) -> bool:
        """Move the weights to system RAM so llama.cpp can use the whole GPU for rollouts.
        Returns False (and stays on the GPU) if this model/bitsandbytes combination can't be moved."""
        if self.device.type != "cuda":
            return False
        try:
            self.model.get_base_model().to("cpu")
        except Exception as e:
            print(f"WARNING: could not move the trainer to RAM ({e}); it stays on the GPU")
            self.model.get_base_model().to(self.home)
            return False
        self.device = torch.device("cpu")
        torch.cuda.empty_cache()
        return True

    def onload(self):
        if self.device != self.home:
            self.model.get_base_model().to(self.home)
            self.device = self.home

    @staticmethod
    def _load(model_dir, kwargs):
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText  # multimodal-only checkpoints
            return AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)

    def num_trainable(self) -> dict:
        return {n: sum(p.numel() for p in ps) for n, ps in self.params.items()}

    def use_adapter(self, name: str):
        if self.active != name:
            self.model.set_adapter(name)
            for n, ps in self.params.items():  # set_adapter only enables the active one
                for p in ps:
                    p.requires_grad_(n == name)
            self.active = name

    # ------------------------------------------------------------ log-probs
    def _body_and_head(self):
        base = self.model.get_base_model()
        body = base.model
        if hasattr(body, "language_model"):
            body = body.language_model
        return body, base.lm_head

    def token_logprobs(self, input_ids: list[int], start: int) -> torch.Tensor:
        """log p(token_i | tokens_<i) for i in [start, len). Chunked so the 248k-wide
        vocabulary never materialises for the whole sequence at once."""
        body, head = self._body_and_head()
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        if self.offload_activations and torch.is_grad_enabled():
            # keep the activations saved for backward in pinned system RAM instead of VRAM
            with torch.autograd.graph.save_on_cpu(pin_memory=torch.cuda.is_available()):
                h = body(input_ids=ids).last_hidden_state[0]
        else:
            h = body(input_ids=ids).last_hidden_state[0]
        hs = h[start - 1:len(input_ids) - 1]
        tgt = ids[0, start:]
        w = head.weight
        parts = []
        for c in range(0, hs.shape[0], self.logit_chunk):
            parts.append(checkpoint(_chunk_logprobs, hs[c:c + self.logit_chunk], tgt[c:c + self.logit_chunk], w,
                                    use_reentrant=False))
        return torch.cat(parts)

    # ------------------------------------------------------------ GRPO update
    def train_step(self, role: str, samples: list[Sample], grad_clip: float = 1.0, max_seq_len: int = 8192,
                   ppo_epochs: int = 1, clip_eps: float = 0.2, kl_beta: float = 0.0, lr_scale: float = 1.0) -> dict:
        self.onload()
        self.use_adapter(role)
        self.model.train()
        opt = self.opt[role]
        for g in opt.param_groups:
            g["lr"] = self.lr[role] * lr_scale
        live = [s for s in samples if s.advantage != 0 and any(s.loss_mask)]
        usable = [s for s in live if len(s.input_ids) <= max_seq_len]
        stats = {"samples": len(usable), "skipped_long": len(live) - len(usable), "loss": 0.0,
                 "grad_norm": 0.0, "tokens": 0, "clip_frac": 0.0, "kl": 0.0}
        if not usable:
            return stats
        total = sum(sum(s.loss_mask) for s in usable)
        stats["tokens"] = total
        starts = [s.loss_mask.index(1) for s in usable]
        old = [None] * len(usable)
        if ppo_epochs > 1:
            with torch.no_grad():
                old = [self.token_logprobs(s.input_ids, st).detach() for s, st in zip(usable, starts)]
        for _ in range(ppo_epochs):
            opt.zero_grad(set_to_none=True)
            loss_total, clip_total, kl_total = 0.0, 0.0, 0.0
            for i, (s, st) in enumerate(zip(usable, starts)):
                lp = self.token_logprobs(s.input_ids, st)
                mask = torch.tensor(s.loss_mask[st:], dtype=lp.dtype, device=lp.device)
                ref = None
                if kl_beta > 0:
                    with torch.no_grad(), self.model.disable_adapter():
                        ref = self.token_logprobs(s.input_ids, st)
                loss_sum, st_ = policy_loss(lp, lp.detach() if old[i] is None else old[i], mask,
                                            s.advantage, clip_eps, ref, kl_beta)
                scaled = loss_sum / total
                (self.scaler[role].scale(scaled) if self.scaler else scaled).backward()
                loss_total += float(loss_sum.detach()) / total
                clip_total += st_.get("clip_frac", 0.0) / len(usable)
                kl_total += st_.get("kl", 0.0) / len(usable)
            if self.scaler:
                self.scaler[role].unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(self.params[role], grad_clip)
            if self.scaler:
                self.scaler[role].step(opt)   # skips the step by itself if gradients overflowed
                self.scaler[role].update()
            elif torch.isfinite(gn):
                opt.step()
            stats["nonfinite_grads"] = stats.get("nonfinite_grads", 0) + int(not torch.isfinite(gn))
        stats.update(loss=loss_total, grad_norm=float(gn) if torch.isfinite(gn) else None, clip_frac=clip_total,
                     kl=kl_total)
        return stats

    # ------------------------------------------------------------ persistence
    def save(self, ckpt_dir: Path, with_optimizer: bool = True) -> dict[str, str]:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        out = {}
        for name in ("solver", "maker"):
            d = Path(ckpt_dir) / name
            d.mkdir(parents=True, exist_ok=True)
            sd = get_peft_model_state_dict(self.model, adapter_name=name)
            save_file({k: v.detach().to(torch.float16).cpu().contiguous() for k, v in sd.items()},
                      str(d / "adapter_model.safetensors"))
            cfg = self.model.peft_config[name].to_dict()
            cfg["base_model_name_or_path"] = str(self.model_dir)
            for k, v in list(cfg.items()):
                if isinstance(v, set):
                    cfg[k] = sorted(v)
            (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
            if with_optimizer:
                torch.save(self.opt[name].state_dict(), d / "optimizer.pt")
            out[name] = str(d)
        return out

    def load(self, ckpt_dir: Path, names=("solver", "maker"), with_optimizer: bool = True):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        for name in names:
            d = Path(ckpt_dir) / name
            sd = {k: v.float() for k, v in load_file(str(d / "adapter_model.safetensors")).items()}
            set_peft_model_state_dict(self.model, sd, adapter_name=name)
            if with_optimizer and (d / "optimizer.pt").exists():
                self.opt[name].load_state_dict(torch.load(d / "optimizer.pt", map_location=self.device))
            elif not with_optimizer:  # fresh optimizer moments after a reset
                self.opt[name] = self._make_optimizer(name)


def base_checkpoint_prefix(base_dir: str) -> str:
    """Return the layer prefix used by the base checkpoint, e.g. 'model.language_model.layers.'."""
    idx = Path(base_dir) / "model.safetensors.index.json"
    names = []
    if idx.exists():
        names = list(json.loads(idx.read_text())["weight_map"])
    else:
        from safetensors import safe_open
        for f in Path(base_dir).glob("*.safetensors"):
            with safe_open(str(f), "pt") as s:
                names += list(s.keys())
    for n in names:
        m = re.match(r"(.*?layers\.)\d+\.", n)
        if m and "mtp" not in n and "visual" not in n:
            return m.group(1)
    return "model.layers."


def export_for_llamacpp(peft_dir: str, base_dir: str, out_dir: str) -> str:
    """Copy an adapter, renaming tensors to the base checkpoint's naming (llama.cpp's converter
    maps names through the base model class, e.g. 'model.language_model.layers.N...')."""
    from safetensors.torch import load_file, save_file
    want = base_checkpoint_prefix(base_dir)
    sd = load_file(str(Path(peft_dir) / "adapter_model.safetensors"))
    renamed = {}
    for k, v in sd.items():
        m = re.match(r"(base_model\.model\.)(.*?layers\.)(\d+\..*)", k)
        renamed[(m.group(1) + want + m.group(3)) if m else k] = v
    os.makedirs(out_dir, exist_ok=True)
    save_file(renamed, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = json.loads((Path(peft_dir) / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = base_dir
    if cfg.get("use_rslora"):
        # llama.cpp always scales by alpha/r and its converter ignores use_rslora. rsLoRA trains with
        # alpha/sqrt(r), so export alpha*sqrt(r): then llama.cpp's alpha'/r equals alpha/sqrt(r).
        cfg["lora_alpha"] = float(cfg["lora_alpha"]) * float(cfg["r"]) ** 0.5
        cfg["use_rslora"] = False
    (Path(out_dir) / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return out_dir


class MockPolicy:
    """Stands in for Policy in --dry-run: no weights, no gradients, plausible numbers."""

    device = torch.device("cpu")

    def __init__(self, engine, seed: int = 0):
        import random
        self.engine, self.rng = engine, random.Random(seed)

    def num_trainable(self):
        return {"solver": 0, "maker": 0}

    def offload(self):
        return False

    def onload(self):
        pass

    def use_adapter(self, name):
        pass

    def train_step(self, role, samples, **kw):
        live = [s for s in samples if s.advantage != 0]
        if role == "solver":
            self.engine.bump(0.01 * len(live) / max(1, len(samples)), 0.0)
        else:
            self.engine.bump(0.0, 0.015 * len(live) / max(1, len(samples)))
        return {"samples": len(live), "skipped_long": 0, "tokens": sum(sum(s.loss_mask) for s in live),
                "loss": self.rng.gauss(0, 0.05), "grad_norm": abs(self.rng.gauss(0.5, 0.2)),
                "clip_frac": 0.0, "kl": 0.0}

    def save(self, ckpt_dir, with_optimizer=True):
        out = {}
        for name in ("solver", "maker"):
            d = Path(ckpt_dir) / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "mock.json").write_text(json.dumps({"skill": self.engine.skill, "maker": self.engine.maker_quality}))
            out[name] = str(d)
        return out

    def load(self, ckpt_dir, names=("solver", "maker"), with_optimizer=True):
        d = json.loads((Path(ckpt_dir) / "solver" / "mock.json").read_text())
        if "solver" in names:
            self.engine.skill = d["skill"]
        if "maker" in names:
            self.engine.maker_quality = d["maker"]
