"""MLX backend (Apple Silicon). Unit tests run whenever mlx is installed.

The real-model tests are opt-in and load ONE model per run (nothing is trained: backward passes
only, no optimizer step; MLX memory is capped as in training):
    ADV_MLX_MODELS=<models dir> ADV_MLX_ROLE=maker  pytest tests/test_mlx.py    # MiniCPM5-2B (1.4 GB)
    ADV_MLX_MODELS=<models dir> ADV_MLX_ROLE=solver pytest tests/test_mlx.py    # K2-Horizon-3.7B (4.1 GB)
"""
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from adv.engines.mlx_engine import ThinkingOnlyBiasMLX  # noqa: E402

MODELS = os.environ.get("ADV_MLX_MODELS")
ROLE = os.environ.get("ADV_MLX_ROLE", "")
REPO = {"maker": "openbmb__MiniCPM5-2B-MLX", "solver": "mlx-community__K2-Horizon-3.7B-4bit"}
FORMAT = {"maker": "qwen", "solver": "k2_horizon"}


def test_thinking_only_bias_mlx():
    vocab, think_end, marker = 10, 9, 3
    proc = ThinkingOnlyBiasMLX(prompt_len=2, think_end=think_end, bias={marker: -2.0}, budget=3)
    out = proc(mx.array([1, 2, 5]), mx.zeros((1, vocab)))           # inside thinking: biased
    assert out[0, marker].item() == -2.0 and out[0, 0].item() == 0.0
    out = proc(mx.array([1, 2, 5, 6, 7]), mx.zeros((1, vocab)))     # budget hit: force the closing tag
    assert out[0, think_end].item() == 0.0 and out[0, 0].item() == float("-inf") and proc.forced_at == 3
    out = proc(mx.array([1, 2, 5, 6, 7, 9]), mx.zeros((1, vocab)))  # after thinking: untouched
    assert out[0, marker].item() == 0.0


def test_thinking_ends_naturally_mlx():
    proc = ThinkingOnlyBiasMLX(prompt_len=1, think_end=9, bias={3: -2.0}, budget=100)
    proc(mx.array([1, 4]), mx.zeros((1, 10)))
    out = proc(mx.array([1, 4, 9]), mx.zeros((1, 10)))              # model closed its thinking itself
    assert out[0, 3].item() == 0.0 and proc.forced_at is None


def test_no_thinking_means_no_bias():
    proc = ThinkingOnlyBiasMLX(prompt_len=1, think_end=None, bias={3: -2.0}, budget=1)
    assert proc(mx.array([1, 4]), mx.zeros((1, 10)))[0, 3].item() == 0.0


# ------------------------------------------------------------------ real models (opt-in)
real = pytest.mark.skipif(not (MODELS and ROLE in REPO),
                          reason="set ADV_MLX_MODELS and ADV_MLX_ROLE=maker|solver to test one real MLX model")


@pytest.fixture(scope="module")
def policy():
    from adv.policy_mlx import MLXPolicy
    from adv.safety import Limits, MemoryGuard
    guard = MemoryGuard(Limits(), disk_path=MODELS)
    return MLXPolicy({ROLE: str(Path(MODELS) / REPO[ROLE])}, {"r": 16, "alpha": 32}, {}, trust_remote_code=True,
                     guard=guard)


@pytest.fixture(scope="module")
def toks():
    from adv.tokenizer import ChatTokenizer
    return {ROLE: ChatTokenizer(str(Path(MODELS) / REPO[ROLE]), True, "low", True)}


@real
def test_lora_on_every_projection_and_memory_capped(policy):
    assert policy.keys[ROLE] == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    assert policy.num_trainable()[ROLE] > 0
    assert policy.memory["mlx_limit_gb"] < 0.65 * policy.memory["total_gb"]


@real
def test_chat_format(toks):
    assert toks[ROLE].format == FORMAT[ROLE]


@real
def test_chunked_logprobs_match_full_forward_and_grads_stay_in_lora(policy):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    for role in (ROLE,):
        model = policy.models[role]
        ids = list(range(1000, 1300))
        lp = policy.token_logprobs(model, ids, 100)
        full = model(mx.array([ids]))[0].astype(mx.float32)
        ref = mx.take_along_axis(full[99:299], mx.array(ids[100:])[:, None], axis=-1)[:, 0] - mx.logsumexp(full[99:299], axis=-1)
        assert mx.abs(lp - ref).max().item() < 2e-2, role   # 4-bit weights, fp16 activations

        def loss_fn(m):
            return -policy.token_logprobs(m, ids, 100).sum()
        with policy._checkpointing(model):
            _, grads = nn.value_and_grad(model, loss_fn)(model)
        flat = tree_flatten(grads)
        assert flat and all("lora_" in k for k, _ in flat)               # only LoRA weights get gradients
        assert any(mx.abs(v).sum().item() > 0 for _, v in flat)


@real
def test_generation_forces_the_right_closing_tag(policy, toks):
    from adv.engines.base import GenParams
    from adv.engines.mlx_engine import MLXEngine
    from adv.tokenizer import thinking_open
    eng = MLXEngine(toks, policy)
    for role in (ROLE,):
        tok = toks[role]
        ids = tok.encode(tok.render_text([{"role": "user", "content": "Write a Python function that adds two numbers."}],
                                         None, True))
        end = thinking_open(tok, ids)
        assert end is not None, role
        r = eng.generate(ids, GenParams(role, max_new_tokens=24, think_budget=8, temperature=0.0), end)
        assert end in r.token_ids, role
        k = r.token_ids.index(end)
        if r.forced_think_end:   # thought for the whole budget: the closing tag was forced, and masked out
            assert k == 8 and r.loss_mask[8] == 0, role
        else:                    # closed its thinking itself (K2's short think_faster mode often does)
            assert k < 8 and all(r.loss_mask) and r.think_tokens == k, role


@real
def test_adapter_save_load_roundtrip(policy, tmp_path):
    from mlx.utils import tree_flatten
    before = {k: mx.array(v) for k, v in tree_flatten(policy.models[ROLE].trainable_parameters())}
    policy.save(tmp_path, with_optimizer=False)
    assert (tmp_path / ROLE / "adapters.safetensors").exists()
    policy.load(tmp_path, names=(ROLE,), with_optimizer=False)
    after = dict(tree_flatten(policy.models[ROLE].trainable_parameters()))
    assert all(mx.array_equal(before[k], after[k]).item() for k in before)


def test_memory_guard_rules():
    from adv.safety import Limits, MemoryDanger, MemoryGuard, check_numbers
    lim = Limits(min_free_ram_gb=1.5, min_free_percent=15, max_swap_growth_gb=2.0, min_free_disk_gb=10.0)
    assert check_numbers(6.0, 0.0, 40.0, lim) is None
    assert "RAM" in check_numbers(1.0, 0.0, 40.0, lim)
    # macOS: the kernel's pressure numbers replace psutil's free RAM (which counts file cache as used)
    assert check_numbers(1.0, 0.0, 40.0, lim, pressure=1, free_pct=77) is None
    assert check_numbers(6.0, 0.0, 40.0, lim, pressure=2, free_pct=30) is None
    assert "warning" in check_numbers(6.0, 0.0, 40.0, lim, pressure=2, free_pct=10)
    assert "critical" in check_numbers(6.0, 0.0, 40.0, lim, pressure=4, free_pct=40)
    assert "swap" in check_numbers(6.0, 3.0, 40.0, lim, pressure=1, free_pct=77)
    assert "swap" in check_numbers(6.0, 3.0, 40.0, lim)
    assert "disk" in check_numbers(6.0, 0.0, 8.0, lim)
    g = MemoryGuard(Limits(min_free_disk_gb=10 ** 9))       # impossible disk requirement -> trips
    with pytest.raises(MemoryDanger):
        g.check("test", force=True)
    with pytest.raises(MemoryDanger):                         # stays tripped, even without force
        g.check("later")
