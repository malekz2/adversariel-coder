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
