"""End-to-end: the whole loop with the fake model, including checkpoint + resume."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import adv.config as C
from adv.train import Trainer

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox_root(tmp_path, monkeypatch):
    shutil.copytree(PROJECT / "data", tmp_path / "data")
    shutil.copy(PROJECT / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    return tmp_path


def _args(**kw):
    base = dict(dry_run=True, resume=False, profile="cpu", no_pi=True, pi_node=None, pi_cli=None, max_hours=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _cfg(root, steps):
    return C.load(root / "config.yaml", {"run.total_steps": steps, "run.name": "t", "run.checkpoint_every": 2,
                                         "decision.every": 3})


def test_dry_run_then_resume(sandbox_root):
    Trainer(_cfg(sandbox_root, 4), _args()).run()
    assert (sandbox_root / "log.txt").read_text().startswith("FINISHED")
    assert (sandbox_root / "train.png").stat().st_size > 10_000
    run = sandbox_root / "runs" / "t"
    assert (run / "checkpoints" / "latest.txt").read_text().strip() == "step_00004"
    assert len(json.loads((run / "verdicts.json").read_text())) == 1

    Trainer(_cfg(sandbox_root, 6), _args(resume=True)).run()
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(6))
    assert (run / "checkpoints" / "latest.txt").read_text().strip() == "step_00006"


def test_phases_keep_trainer_and_llama_server_off_the_gpu_at_the_same_time():
    calls = []
    stub = SimpleNamespace(
        offload_trainer=True,
        policy=SimpleNamespace(offload=lambda: calls.append("trainer -> RAM") or True,
                               onload=lambda: calls.append("trainer -> GPU")),
        engine=SimpleNamespace(resume=lambda: calls.append("server up"), pause=lambda: calls.append("server down")))
    stub._start_engine = lambda: Trainer._start_engine(stub)
    Trainer._rollout_phase(stub)
    Trainer._train_phase(stub)
    assert calls == ["trainer -> RAM", "server up", "server down", "trainer -> GPU"]
    stub.policy.offload = lambda: False          # e.g. bitsandbytes can't move this model
    Trainer._rollout_phase(stub)
    assert stub.offload_trainer is False         # gives up offloading instead of crashing


def test_sync_adapters_can_leave_server_stopped(tmp_path, monkeypatch):
    import adv.engines.llamacpp as L
    monkeypatch.setattr(L, "convert_adapter_to_gguf", lambda src, base, out, *a: Path(out).write_text("gguf"))
    eng = L.LlamaCppEngine(None, "llama-server", "m.gguf", converter={"base_model_dir": ".", "script": "x"},
                           adapter_gguf_dir=str(tmp_path), swap_during_training=True)
    calls = []
    eng.stop, eng.start = (lambda: calls.append("stop")), (lambda: calls.append("start"))
    eng.sync_adapters({"maker": "a", "solver": "b"}, restart=False)
    assert calls == ["stop"] and set(eng.adapter_files) == {"maker", "solver"}
    eng.sync_adapters({"maker": "a", "solver": "b"})
    assert calls == ["stop", "stop", "start"]


def test_time_budget_stops_cleanly(sandbox_root):
    cfg = _cfg(sandbox_root, 50)
    cfg["run"]["time_budget_hours"] = 0.01   # 36 s: less than the reserve, so only the first step fits
    Trainer(cfg, _args()).run()
    log = (sandbox_root / "log.txt").read_text()
    assert log.startswith("STOPPED: 0.01 h time budget reached after 1 steps")
    run = sandbox_root / "runs" / "t"
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 1
    assert (run / "checkpoints" / "latest.txt").read_text().strip() == "step_00001"


def test_profile_overrides_only_affect_their_profile(sandbox_root):
    big = Trainer(_cfg(sandbox_root, 1), _args(profile="cuda_large"))
    small = Trainer(_cfg(sandbox_root, 1), _args(profile="cpu"))
    assert big.cfg["grpo"]["puzzles_per_step"] == 8 and small.cfg["grpo"]["puzzles_per_step"] == 4


def test_adapters_convert_concurrently_and_extra_args_reach_llama_server(tmp_path, monkeypatch):
    import threading
    import time as _t
    import adv.engines.llamacpp as L
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow_convert(src, base, out, *a):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _t.sleep(0.3)
        Path(out).write_text("gguf")
        with lock:
            active[0] -= 1
    monkeypatch.setattr(L, "convert_adapter_to_gguf", slow_convert)
    eng = L.LlamaCppEngine(None, "llama-server", "m.gguf", converter={"base_model_dir": ".", "script": "x"},
                           adapter_gguf_dir=str(tmp_path), extra_args=["-ctk", "q8_0", "--cache-ram", "2048"])
    eng.stop = lambda: None
    eng.sync_adapters({"maker": "a", "solver": "b"}, restart=False)
    assert peak[0] == 2
    cmd = eng._cmd()
    assert cmd[-4:] == ["-ctk", "q8_0", "--cache-ram", "2048"] and "--lora" in cmd


def test_memory_guard_stops_run_cleanly(sandbox_root):
    from adv.safety import Limits, MemoryGuard
    t = Trainer(_cfg(sandbox_root, 5), _args())
    t.guard = MemoryGuard(Limits())
    real_step = t.step

    def step_then_trip(step):
        row = real_step(step)
        if step == 1:
            t.guard.limits.min_free_disk_gb = 10 ** 9   # simulate running out of headroom
        return row
    t.step = step_then_trip
    t.run()
    log = (sandbox_root / "log.txt").read_text()
    assert log.startswith("STOPPED for safety: only") and "checkpoint saved" in log
    assert (sandbox_root / "runs" / "t" / "checkpoints" / "latest.txt").read_text().strip() == "step_00002"
