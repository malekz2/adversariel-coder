"""Rollout tuning on llama.cpp profiles: KV-cache type, speculative decoding (MTP / DFlash2 / n-gram),
slot count. Every decision path of rollout_tune is exercised with a fake llama-server."""
from types import SimpleNamespace

import pytest

import adv.config as C
from adv.engines.llamacpp import LlamaCppEngine
from adv.rollout_tune import (agreement, agrees, needs_drafter, rollout_settings, speculative_args,
                              tune_rollouts)
from adv.train import Trainer

SPEC = {"draft_repo": "z-lab/Qwen3.8-27B-DFlash2-GGUF", "draft_file": "d.gguf",
        "candidates": [{"type": "draft-mtp,ngram-mod", "n_max": 3}, {"type": "draft-mtp", "n_max": 3},
                       {"type": "draft-dflash,ngram-mod", "n_max": 7}, {"type": "draft-dflash", "n_max": 7},
                       {"type": "ngram-mod"}],
        "slots": [8, 4], "min_gain": 1.05, "min_agreement": 0.95, "args": ["--spec-draft-ngl", "all"]}
PROF = {"parallel_slots": 8, "kv_cache_types": ["q8_0", "q4_0", "f16"], "speculative": SPEC}


# ------------------------------------------------------------------ flags
def test_speculative_flags():
    assert speculative_args(SPEC, {"type": "draft-mtp,ngram-mod", "n_max": 3}, None) == [
        "--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "3", "--spec-draft-ngl", "all"]   # MTP: no -md
    assert speculative_args(SPEC, {"type": "draft-dflash", "n_max": 7}, "/m/d.gguf")[:6] == [
        "--spec-type", "draft-dflash", "-md", "/m/d.gguf", "--spec-draft-n-max", "7"]
    assert "--spec-draft-n-max" not in speculative_args(SPEC, {"type": "ngram-mod"}, None)
    with pytest.raises(ValueError):
        speculative_args(SPEC, {"type": "draft-dflash,ngram-mod"}, None)
    assert needs_drafter("draft-dflash,ngram-mod") and not needs_drafter("draft-mtp,ngram-mod")


def test_engine_puts_tuned_flags_on_the_command_line():
    eng = LlamaCppEngine(None, "llama-server", "m.gguf", extra_args=["-fa", "on", "--fit", "off"],
                         spec_args=["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "draft-mtp"])
    assert eng._cmd()[-10:] == ["-fa", "on", "--fit", "off", "-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "draft-mtp"]


def test_cuda_large_profile_has_d_and_e():
    prof = C.load()["profiles"]["cuda_large"]
    assert prof["llama_gpu_layers"] == "all"
    assert prof["llama_extra_args"][prof["llama_extra_args"].index("--fit") + 1] == "off"
    assert "-ctk" not in prof["llama_extra_args"]                  # KV type now comes from the tuner
    assert prof["kv_cache_types"][0] == "q8_0"
    assert {c["type"] for c in prof["speculative"]["candidates"]} >= {"draft-mtp", "draft-dflash", "ngram-mod"}


def test_download_includes_the_drafter():
    from adv.download import plan
    items = plan(C.load(), "cuda_large")
    assert ("z-lab/Qwen3.8-27B-DFlash2-GGUF", "Qwen3.8-27B-DFlash2-Q4_K_M.gguf") in [(i["repo"], i["file"]) for i in items]


# ------------------------------------------------------------------ tolerant exactness
def test_agreement_tolerates_a_late_divergence_but_not_a_broken_output():
    ref = {(i, s): list(range(100)) for i in range(3) for s in (0.0, 1.0)}
    late = dict(ref)
    late[(2, 1.0)] = list(range(90)) + [-1] * 10                   # one near-tie flip at token 90
    ok, why = agrees(ref, late, 0.95)
    assert ok and "5/6 sequences identical" in why
    early = {k: [v[0]] + [-1] * 99 for k, v in ref.items()}         # diverges at token 1 everywhere
    assert not agrees(ref, early, 0.95)[0]
    assert agreement(ref, ref) == (1.0, 6)
    half = dict(ref)
    for k in list(half)[:4]:
        half[k] = list(range(99)) + [-1]                            # high fraction but most sequences differ
    assert not agrees(ref, half, 0.95)[0]


# ------------------------------------------------------------------ the tuner
class FakeProbe:
    """Fake llama-server. speed[(kv, type, slots)] -> tok/s; output_mode[type] in {"ok","changed","lora_ignored"};
    crash: set of (kv) or (type) that fail to start."""

    def __init__(self, speed, output_mode=None, crash=(), free_mb=None, single=None):
        self.speed, self.mode, self.crash = speed, output_mode or {}, set(crash)
        self.free_mb, self.single = free_mb or {}, single or {}   # (kv, type, slots) -> MiB free / tok/s alone
        self.cfg, self.starts = None, []

    def vram(self):
        return {"total": 24564, "used": 0, "free": self.free_mb.get(self.cfg, 20000)}

    def single_speed(self):
        return self.single.get(self.cfg)

    def start(self, args, log_name=None):
        kv = args[args.index("-ctk") + 1] if "-ctk" in args else None
        slots = int(args[args.index("-np") + 1])
        typ = args[args.index("--spec-type") + 1] if "--spec-type" in args else "none"
        self.starts.append((kv, typ, slots, list(args)))
        if kv in self.crash or typ in self.crash:
            raise RuntimeError("llama-server exited (1)")
        self.cfg = (kv, typ, slots)

    def stop(self):
        self.cfg = None

    def complete(self, prompt, scale, n, temperature=0.0, seed=0):
        base = [prompt[0] * 10 + (0 if scale == 0 else 5) + j for j in range(n)]
        mode = self.mode.get(self.cfg[1], "ok")
        if mode == "lora_ignored":
            base = [prompt[0] * 10 + j for j in range(n)]
        if mode == "changed":
            base = [base[0]] + [-7] * (n - 1)
        return base, {}

    def throughput(self, prompts, n, temperature=0.8):
        return {"tokens_per_s": self.speed.get(self.cfg, 1.0), "wall_s": 1.0, "draft_acceptance": 0.7}


def base_args(kv, slots):
    return ["-m", "m.gguf", "-np", str(slots)] + (["-ctk", kv, "-ctv", kv] if kv else [])


def _tune(probe, tmp_path, draft=True):
    d = tmp_path / "d.gguf"
    if draft:
        d.write_text("x")
    return tune_rollouts(probe, base_args, PROF, str(d), [[1], [2], [3]], [[4]] * 8, log=lambda *a: None)


def test_tuner_picks_fastest_kv_then_best_exact_speculation(tmp_path):
    speed = {("q8_0", "none", 8): 20.0,                      # simulates the reported silent CPU fallback
             ("q4_0", "none", 8): 60.0,
             ("q4_0", "draft-mtp,ngram-mod", 8): 62.0,       # < 5 % better than plain: not worth it
             ("q4_0", "draft-mtp,ngram-mod", 4): 70.0,       # best: fewer slots + speculation
             ("q4_0", "draft-mtp", 8): 66.0, ("q4_0", "draft-mtp", 4): 64.0}
    probe = FakeProbe(speed, {"draft-dflash,ngram-mod": "changed", "ngram-mod": "lora_ignored"},
                      crash={"f16", "draft-dflash"})
    res = _tune(probe, tmp_path)
    assert res["ok"] and res["kv_cache"] == "q4_0"
    assert res["speculative"] == {"type": "draft-mtp,ngram-mod", "n_max": 3} and res["slots"] == 4
    assert res["speedup"] == round(70 / 60, 2)
    assert res["kv_results"]["f16"].startswith("failed to start")
    assert res["spec_results"]["draft-dflash,ngram-mod @ 8 slots"].startswith("output changed")
    assert "draft-dflash,ngram-mod @ 4 slots" not in res["spec_results"]       # rejected types are not retried
    assert res["spec_results"]["draft-dflash @ 8 slots"].startswith("failed to start")
    assert res["spec_results"]["ngram-mod @ 8 slots"] == "LoRA had no effect with speculation on"
    # every speculative run used the KV type chosen in stage 1
    assert all(kv == "q4_0" for kv, typ, _, _ in probe.starts if typ != "none")


def test_tuner_keeps_plain_decoding_when_speculation_is_not_faster(tmp_path):
    speed = {("q8_0", "none", 8): 60.0, ("q4_0", "none", 8): 55.0, ("f16", "none", 8): 58.0}
    res = _tune(FakeProbe(speed), tmp_path)           # every speculative config defaults to 1 tok/s
    assert res["ok"] and res["kv_cache"] == "q8_0" and res["speculative"] is None and res["slots"] == 8
    assert res["speedup"] == 1.0


def test_tuner_skips_drafter_types_when_drafter_missing(tmp_path):
    res = _tune(FakeProbe({("q8_0", "none", 8): 50.0}), tmp_path, draft=False)
    assert res["spec_results"]["draft-dflash"] == "drafter not downloaded"
    assert res["spec_results"]["draft-dflash,ngram-mod"] == "drafter not downloaded"
    assert "draft-mtp @ 8 slots" in res["spec_results"]   # MTP needs no drafter file


def test_tuner_reports_failure_when_no_kv_type_starts(tmp_path):
    res = _tune(FakeProbe({}, crash={"q8_0", "q4_0", "f16"}), tmp_path)
    assert res["ok"] is False and "no KV-cache type" in res["reason"]


# ------------------------------------------------------------------ what training uses
def test_rollout_settings_defaults_before_tuning():
    rs = rollout_settings(PROF, {}, "/nope/d.gguf")
    assert rs["kv_cache"] == "q8_0" and rs["slots"] == 8 and rs["args"] == ["-ctk", "q8_0", "-ctv", "q8_0"]
    assert rs["spec_label"] == "" and "not tuned yet" in rs["note"]


def test_rollout_settings_use_the_tuned_choice(tmp_path):
    tuned = {"rollout": {"ok": True, "kv_cache": "q4_0", "slots": 4, "speedup": 1.17,
                         "speculative": {"type": "draft-mtp,ngram-mod", "n_max": 3}}}
    rs = rollout_settings(PROF, tuned, None)
    assert rs["slots"] == 4 and rs["spec_label"] == "draft-mtp,ngram-mod" and "-md" not in rs["args"]
    assert rs["args"][:4] == ["-ctk", "q4_0", "-ctv", "q4_0"] and rs["speedup"] == 1.17


def test_rollout_settings_turn_speculation_off_if_drafter_vanished(tmp_path):
    tuned = {"rollout": {"ok": True, "kv_cache": "q4_0", "slots": 8,
                         "speculative": {"type": "draft-dflash", "n_max": 7}}}
    rs = rollout_settings(PROF, tuned, str(tmp_path / "missing.gguf"))
    assert rs["spec_label"] == "" and "--spec-type" not in rs["args"] and "drafter" in rs["note"]
    assert rs["args"] == ["-ctk", "q4_0", "-ctv", "q4_0"]


def test_profiles_without_tuning_are_unchanged():
    rs = rollout_settings({"parallel_slots": 4}, {}, None)
    assert rs == {"kv_cache": None, "slots": 4, "args": [], "spec_label": "", "note": "", "speedup": None, "spec": None}


def test_trainer_reads_the_tuned_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "repo_dir", lambda repo: tmp_path)
    monkeypatch.setattr(C, "read_state", lambda name: {"rollout": {"ok": True, "kv_cache": "q4_0", "slots": 4,
                                                                   "speculative": {"type": "draft-mtp", "n_max": 3}}})
    stub = SimpleNamespace(prof={**PROF})
    stub._draft_path = lambda: Trainer._draft_path(stub)
    rs = Trainer._rollout_settings(stub)
    assert rs["slots"] == 4 and stub.spec_label == "draft-mtp" and stub.rollout_label == "KV q4_0, 4 slots"


def test_tuner_rejects_settings_that_only_fit_on_an_empty_gpu(tmp_path):
    speed = {("q8_0", "none", 8): 60.0, ("q4_0", "none", 8): 58.0, ("f16", "none", 8): 66.0,
             ("q8_0", "draft-dflash", 8): 90.0, ("q8_0", "draft-dflash", 4): 80.0}
    free = {("f16", "none", 8): 700, ("q8_0", "draft-dflash", 8): 400}       # < 1.5 GB headroom
    single = {("q4_0", "none", 8): 3.0}                                      # spilled into RAM
    probe = FakeProbe(speed, free_mb=free, single=single)
    d = tmp_path / "d.gguf"
    d.write_text("x")
    prof = {**PROF, "vram_headroom_gb": 1.5, "min_decode_tps": 12}
    res = tune_rollouts(probe, base_args, prof, str(d), [[1], [2], [3]], [[4]] * 8, log=lambda *a: None,
                        vram_fn=probe.vram)
    assert res["kv_results"]["f16"].startswith("too tight for training (0.7 GB")
    assert res["kv_results"]["q4_0"].startswith("too slow (3.0 tok/s")
    assert res["kv_cache"] == "q8_0"
    assert res["spec_results"]["draft-dflash @ 8 slots"].startswith("too tight")
    assert res["speculative"]["type"] == "draft-dflash" and res["slots"] == 4   # fewer slots still tried
