"""Real-architecture plumbing on a miniature random model (no optimiser steps: nothing is trained).

Opt-in:  ADV_TINY_SRC=<dir with config.json + tokenizer files (+ .py code)> pytest tests/test_tiny_model.py
Works for Qwen3.5 (e.g. <models>/unsloth__Qwen3.5-2B) and K2-Horizon (<models>/IFM__K2-Horizon-7B).
"""
import os
import re

import pytest
import torch

SRC = os.environ.get("ADV_TINY_SRC")
pytestmark = pytest.mark.skipif(not SRC, reason="set ADV_TINY_SRC to a Qwen3.5 tokenizer/config folder")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tiny_model import make_tiny
    return make_tiny(SRC, str(tmp_path_factory.mktemp("tiny")))


@pytest.fixture(scope="module")
def policy(tiny):
    from adv.policy import Policy
    return Policy(tiny, {"r": 4, "alpha": 8}, {}, dtype="float32", device="cpu", trust_remote_code=True)


def test_lora_targets_and_text_only_load(policy):
    name = type(policy.model.get_base_model()).__name__
    assert name in ("Qwen3_5ForCausalLM", "K2HorizonForCausalLM")
    mods = {re.sub(r".*layers\.\d+\.", "", n.split(".lora_")[0]) for n, _ in policy.model.named_parameters()
            if "lora_A.solver" in n}
    want = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    if name.startswith("Qwen3_5"):  # hybrid model: Gated-DeltaNet layers too
        want |= {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"}
    assert mods == want


def test_chunked_logprobs_match_full_forward_and_grads_hit_only_active_adapter(policy, tiny):
    from adv.grpo import policy_loss
    seq = list(range(100, 160))
    policy.use_adapter("solver")
    policy.model.train()
    lp = policy.token_logprobs(seq, 20)
    with torch.no_grad():
        full = policy.model(input_ids=torch.tensor([seq])).logits[0].float().log_softmax(-1)
        ref = full[19:59].gather(-1, torch.tensor(seq[20:])[:, None]).squeeze(-1)
    assert (lp.detach() - ref).abs().max() < 1e-4
    loss, _ = policy_loss(lp, lp.detach(), torch.ones_like(lp), 1.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.params["solver"])
    assert all(p.grad is None for p in policy.params["maker"])
    assert all(p.grad is None for n, p in policy.model.named_parameters() if "lora_" not in n)
    policy.model.zero_grad(set_to_none=True)


def test_hf_engine_think_budget(policy, tiny):
    from adv.engines.base import GenParams
    from adv.engines.hf import HFEngine
    from adv.tokenizer import ChatTokenizer, thinking_open
    tok = ChatTokenizer(tiny, reasoning_effort="low", trust_remote_code=True)
    ids = tok.encode(tok.render_text([{"role": "user", "content": "hi"}], None, True))
    end = thinking_open(tok, ids)
    assert end is not None and end in tok.think_pairs.values()
    r = HFEngine(tok, policy).generate(ids, GenParams("maker", max_new_tokens=20, think_budget=5), end)
    assert r.token_ids[5] == end and r.loss_mask[5] == 0 and r.forced_think_end


def test_export_renames_and_rescales_rslora(policy, tiny, tmp_path):
    import json
    from safetensors import safe_open
    from adv.policy import export_for_llamacpp
    dirs = policy.save(tmp_path / "ck", with_optimizer=False)
    out = export_for_llamacpp(dirs["solver"], tiny, str(tmp_path / "ex"))
    with safe_open(out + "/adapter_model.safetensors", "pt") as s:
        from adv.policy import base_checkpoint_prefix
        prefix = base_checkpoint_prefix(tiny)
        assert all(("base_model.model." + prefix) in k for k in s.keys())
    cfg = json.loads(open(out + "/adapter_config.json").read())
    assert cfg["lora_alpha"] == 8 * 2 and cfg["use_rslora"] is False   # alpha * sqrt(r=4)
