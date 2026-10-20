"""The 1 tok/s guard on the 4090: bounded context checkpoints, VRAM checks, the start-time speed check,
stepping down to smaller llama-server settings, and the speed line in log.txt."""
import json
from types import SimpleNamespace

import pytest

import adv.config as C
import adv.engines.llamacpp as L
from adv import gpu
from adv.logbook import ETA, render_log
from adv.rollout_tune import fallback_ladder, rollout_settings, rung_launch
from adv.train import Trainer

SPEC = {"candidates": [{"type": "draft-mtp", "n_max": 3}], "args": ["--spec-draft-ngl", "all"]}
PROF = {"parallel_slots": 8, "kv_cache_types": ["q8_0", "q4_0", "f16"], "speculative": SPEC,
        "llama_gpu_layers": "all", "llama_extra_args": ["-fa", "on", "--fit", "off", "--cache-ram", "1024"]}
TUNED = {"rollout": {"ok": True, "kv_cache": "f16", "slots": 8, "speculative": {"type": "draft-mtp", "n_max": 3}}}


# ------------------------------------------------------------------ gpu.py
def test_parse_vram_and_no_gpu(monkeypatch):
    assert gpu.parse_vram("24564, 20100, 4464\n") == {"total": 24564, "used": 20100, "free": 4464}
    assert gpu.parse_vram("") is None and gpu.parse_vram("[N/A], 1, 2") is None
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)      # the Mac / CPU PCs
    assert gpu.vram() is None and gpu.fmt(None) == "n/a"


def test_only_our_own_leftover_server_is_stopped(tmp_path):
    exe = tmp_path / "tools" / "llama-server"
    exe.parent.mkdir()
    exe.write_text("")
    ours = [str(exe), "-m", "x.gguf", "--port", "8181"]
    assert gpu.is_our_server(str(exe), ours, str(exe), [8181])
    assert not gpu.is_our_server(str(exe), ours[:-1] + ["9000"], str(exe), [8181])     # someone else's port
    assert not gpu.is_our_server("/usr/bin/llama-server", ours, str(exe), [8181])       # someone else's binary
    assert not gpu.is_our_server(None, ours, str(exe), [8181])


# ------------------------------------------------------------------ engine
def test_context_checkpoints_are_bounded():
    cmd = L.LlamaCppEngine(None, "llama-server", "m.gguf")._cmd()
    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "4" and cmd[cmd.index("--checkpoint-min-step") + 1] == "512"
    prof = C.load()["profiles"]["cuda_large"]
    assert prof["ctx_checkpoints"] * prof["parallel_slots"] * 0.15 < 4    # GB of RAM at ~150 MB per checkpoint
    assert prof["min_decode_tps"] > 0 and prof["vram_headroom_gb"] >= 1


class _Resp:
    def __init__(self, d):
        self.d = d

    def raise_for_status(self):
        pass

    def json(self):
        return self.d


def _engine(monkeypatch, free_mb, tps, floor=12):
    eng = L.LlamaCppEngine(None, "llama-server", "m.gguf", gpu_layers="all", min_decode_tps=floor)
    monkeypatch.setattr(L.gpu, "vram", lambda: {"total": 24564, "used": 24564 - free_mb, "free": free_mb})
    eng.session = SimpleNamespace(post=lambda url, json, timeout: _Resp({"timings": {"predicted_per_second": tps}}))
    stopped = []
    eng.stop = lambda: stopped.append(True)
    return eng, stopped


def test_start_check_rejects_a_full_gpu_or_a_crawl(monkeypatch):
    eng, stopped = _engine(monkeypatch, free_mb=2000, tps=41.3)
    eng._check_health(None)                                    # healthy: no exception
    assert eng.last_check["tps"] == 41.3 and not stopped
    eng, stopped = _engine(monkeypatch, free_mb=2000, tps=1.2)
    with pytest.raises(L.EngineTooSlow, match="1.2 tok/s"):
        eng._check_health(None)
    assert stopped
    eng, stopped = _engine(monkeypatch, free_mb=120, tps=30.0)
    with pytest.raises(L.EngineTooSlow, match="GPU memory full"):
        eng._check_health(None)
    eng, _ = _engine(monkeypatch, free_mb=120, tps=1.0, floor=None)  # CPU-ish profiles: no speed floor ...
    eng.gpu_layers = 0                                                 # ... and no VRAM check without a GPU
    eng._check_health(None)


def test_speed_stats_from_llama_server_timings():
    eng = L.LlamaCppEngine(None, "llama-server", "m.gguf")
    assert eng.take_stats() == {}
    eng._timings = [{"predicted_per_second": 30.0, "predicted_n": 100, "prompt_n": 900, "prompt_ms": 450.0, "cache_n": 3000},
                    {"predicted_per_second": 20.0, "predicted_n": 50, "prompt_n": 100, "prompt_ms": 50.0, "cache_n": 0},
                    {"predicted_per_second": 99.0, "predicted_n": 2}]   # too short to count
    st = eng.take_stats()
    assert st == {"decode_tps_stream": 25.0, "prefill_tps": 2000.0, "prompt_cached_frac": 0.75, "prompt_tokens": 1000}
    assert eng.take_stats() == {}


# ------------------------------------------------------------------ the step-down ladder
def test_ladder_needs_less_vram_on_every_rung():
    rs = rollout_settings(PROF, TUNED, None)
    rungs = fallback_ladder(PROF, rs)
    assert [r["label"] for r in rungs] == ["as tuned", "speculative decoding off", "KV cache q4_0", "6 parallel slots",
                                           "4 parallel slots", rungs[-1]["label"]]
    assert rungs[-1]["gpu_layers"] == "auto" and rungs[-1]["fit_on"] and rungs[-1]["slots"] == 4
    first = rung_launch(PROF, rungs[0], None)
    assert first["spec_args"][:6] == ["-ctk", "f16", "-ctv", "f16", "--spec-type", "draft-mtp"]
    last = rung_launch(PROF, rungs[-1], None)
    assert last["extra_args"] == ["-fa", "on", "--cache-ram", "1024", "--fit", "on"] and last["parallel"] == 4
    assert "--spec-type" not in last["spec_args"]
    small = fallback_ladder({**PROF, "kv_cache_types": None, "llama_gpu_layers": "auto"},
                            rollout_settings({"parallel_slots": 4}, {}, None))
    assert [r["label"] for r in small] == ["as tuned"]          # nothing to step down to


class _FlakyEngine:
    """Too slow until it has been reconfigured `bad` times."""

    def __init__(self, bad):
        self.bad, self.configs, self.min_decode_tps, self.min_free_vram_mb = bad, [], 12, 300
        self.last_check = {"tps": 38.5}

    def reconfigure(self, **kw):
        self.configs.append(kw)

    def resume(self):
        if self.min_decode_tps and len(self.configs) < self.bad:
            raise L.EngineTooSlow("1.1 tok/s for one stream, expected at least 12", 1.1)


def _stub(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(C, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(C, "read_state", lambda name: json.loads((tmp_path / name).read_text())
                        if (tmp_path / name).exists() else {})
    stub = SimpleNamespace(prof=PROF, profile="cuda_large", engine=_FlakyEngine(bad), engine_note="")
    for name in ("_apply_rung", "_save_rung", "_draft_path", "_start_engine", "_init_ladder"):
        setattr(stub, name, (lambda n: lambda *a: getattr(Trainer, n)(stub, *a))(name))
    stub._init_ladder(rollout_settings(PROF, TUNED, None))
    return stub


def test_trainer_steps_down_until_fast_and_remembers_it(tmp_path, monkeypatch):
    stub = _stub(tmp_path, monkeypatch, bad=2)
    stub._start_engine()
    assert stub.rung == 2 and stub.rollout_label == "KV q4_0, 8 slots" and stub.spec_label == ""
    assert "now using 'KV cache q4_0'" in stub.engine_note and "38.5 tok/s" in stub.engine_note
    saved = json.loads((tmp_path / "engine.json").read_text())["rollout_fallback"]
    assert saved["rung"] == 2 and "1.1 tok/s" in saved["reason"]
    again = _stub(tmp_path, monkeypatch, bad=0)                     # next run starts on the rung that worked
    assert again.rung == 2 and again.engine.configs[0]["spec_args"] == ["-ctk", "q4_0", "-ctv", "q4_0"]


def test_trainer_keeps_going_when_nothing_smaller_is_left(tmp_path, monkeypatch):
    stub = _stub(tmp_path, monkeypatch, bad=99)
    stub._start_engine()
    assert stub.rung == len(stub.ladder) - 1 and stub.engine_note.startswith("WARNING: rollouts are slow")
    assert stub.engine.min_decode_tps is None


# ------------------------------------------------------------------ RAM + log
def test_releasing_gpu_caches_is_safe_without_cuda():
    from adv.policy import release_gpu_caches
    assert release_gpu_caches() in (True, False)


def test_log_shows_rollout_speed():
    cfg = {"run": {"name": "t", "total_steps": 10, "log_every": 1}, "decision": {"every": 5}}
    header = {"profile": "cuda_large", "model": "m", "engine": "llamacpp", "referee": "rules", "harness": "pi"}
    row = {"step": 0, "step_seconds": 60, "gen_tokens": 9000, "rollout_seconds": 60, "rollout_tps": 150.0,
           "decode_tps_stream": 24.5, "prefill_tps": 2100.0, "prompt_tokens": 50000, "prompt_cached_frac": 0.6}
    text = render_log(cfg=cfg, header=header, history=[row], eta=ETA(), started=0, verdicts=[], sample=None)
    assert "rollout speed 150.0 tok/s overall   24.5 tok/s per stream (median)" in text
    assert "prompt reading 2100 tok/s, 50000 tokens (0.60 reused from cache)" in text
