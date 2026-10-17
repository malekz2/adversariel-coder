import math

import torch

from adv.decision import ACTIONS, Controller, RuleReferee, build_state, parse_answers
from adv.grpo import group_advantages, policy_loss
from adv.logbook import ETA, atomic_write, fmt_dur, render_log
from adv.rewards import learnability, maker_reward, solver_reward


def test_hardware_profiles():
    from adv.hardware import pick_profile
    gpu = lambda name, gb: {"torch_cuda": True, "cuda_device": {"name": name, "vram_gb": gb}, "ram_gb": 16}  # noqa: E731
    assert pick_profile(gpu("NVIDIA GeForce RTX 2080 Ti", 11.8)) == "lawson_b131"   # Purdue LWSN B131
    assert pick_profile(gpu("NVIDIA GeForce RTX 3060", 12.9)) == "lawson_b131"
    assert pick_profile(gpu("NVIDIA GeForce RTX 3070", 8.6)) == "cuda_small"
    assert pick_profile(gpu("NVIDIA GeForce RTX 4060 Ti", 17.2)) == "cuda_small"
    assert pick_profile(gpu("NVIDIA GeForce RTX 4090", 25.4)) == "cuda_large"
    assert pick_profile({"ram_gb": 34.2}) == "cpu"            # OptiPlex 5000, 32 GB, no GPU
    assert pick_profile({"ram_gb": 17.1}) == "cpu_small"      # Lawson Linux labs (16 GB, no GPU)
    assert pick_profile({"ram_gb": 8.5}) == "cpu_tiny"


def test_learnability_shape():
    assert learnability(0.5) == 1.0 and learnability(0) == 0 and learnability(1) == 0
    assert math.isclose(learnability(0.25), 0.5)


def test_rewards():
    cfg = {"maker_invalid": -1.0, "similarity_penalty": 0.5, "similarity_free": 0.3, "full_pass_bonus": 0.5}
    assert maker_reward(False, 0.5, 0.0, cfg) == -1.0
    assert maker_reward(True, 0.5, 0.0, cfg) == 1.0
    assert maker_reward(True, 0.5, 1.0, cfg) == 0.5
    assert solver_reward(1.0, True, cfg) == 1.5 and solver_reward(0.5, True, cfg) == 0.5
    assert solver_reward(1.0, False, cfg) == 0.0


def test_group_advantages_and_dead_groups():
    adv, dead = group_advantages([1, 0, 1, 0, 1, 1, 1, 1], list("aaaabbbb"))
    assert dead == 0.5                       # group b all equal -> no signal
    assert adv[4:] == [0.0] * 4
    assert adv[0] > 0 > adv[1] and math.isclose(sum(adv[:4]), 0, abs_tol=1e-9)


def test_policy_loss_pushes_logprob_in_direction_of_advantage():
    for a in (1.0, -1.0):
        lp = torch.tensor([-1.0, -2.0], requires_grad=True)
        loss, _ = policy_loss(lp, lp.detach(), torch.ones(2), a)
        loss.backward()
        assert all((g < 0) == (a > 0) for g in lp.grad)  # descent raises log-prob iff advantage > 0


def test_ppo_clip_blocks_large_ratio():
    lp = torch.tensor([0.0], requires_grad=True)
    old = torch.tensor([-1.0])                 # ratio e > 1.2 with positive advantage -> clipped, no grad
    loss, st = policy_loss(lp, old, torch.ones(1), 1.0, clip_eps=0.2)
    loss.backward()
    assert lp.grad.item() == 0.0 and st["clip_frac"] == 1.0


def test_eta_and_format():
    e = ETA()
    for s in (100, 60, 60, 60):
        e.update(s)
    assert 55 < e.per_step() < 80
    assert e.remaining(10) == e.per_step() * 10
    assert fmt_dur(3725) == "1h02m" and fmt_dur(90061) == "1d01h01m"


def test_atomic_write(tmp_path):
    p = tmp_path / "log.txt"
    atomic_write(p, "a")
    atomic_write(p, "b")
    assert p.read_text() == "b" and not (tmp_path / "log.txt.tmp").exists()


def _history(n, pass_rate=0.5, valid=1.0):
    return [{"step": i, "solver_pass_rate": pass_rate, "maker_valid_rate": valid, "maker_learnability": 0.6,
             "maker_novelty_sim": 0.2, "dead_group_frac": 0.3, "think_forced_frac": 0.1, "step_seconds": 60}
            for i in range(n)]


def test_rule_referee_and_controller():
    cfg = {"stall_threshold": 0.6, "cooldown_steps": 15, "inject_steps": 5, "start_difficulty": 2}
    ref = RuleReferee()
    knobs = Controller(cfg).knobs
    v = parse_answers(ref.ask(build_state(_history(20, pass_rate=0.97), knobs)), 19, "rules", 0.0)
    assert v.diagnosis == "maker_too_easy" and v.action == "harder_puzzles" and v.stalled == 1.0
    c = Controller(cfg)
    assert c.apply(v, 19) == ["difficulty -> 3"]
    assert c.apply(v, 25) == ["skipped harder_puzzles (cooldown)"]
    for _ in range(5):
        c.last_action_step = -100
        c.apply(v, 200)
    assert c.knobs["difficulty"] == 5            # bounded
    healthy = parse_answers(ref.ask(build_state(_history(20), knobs)), 19, "rules", 0.0)
    assert healthy.diagnosis == "healthy" and Controller(cfg).apply(healthy, 19) == []


def test_parse_d3_style_answers():
    answers = {"stalled": {"type": "noul", "noul": 0.81},
               "diagnosis": {"type": "choice", "choice": "maker_too_hard",
                             "probabilities": {"maker_too_hard": 0.7, "healthy": 0.3}, "confidence": 0.5},
               "action": {"type": "choice", "choice": "easier_puzzles", "probabilities": {k: 0.1 for k in ACTIONS}},
               "progress": {"type": "score", "score": 1.4}}
    v = parse_answers(answers, 9, "d3-flash", 1.0)
    assert (v.stalled, v.diagnosis, v.action, v.progress) == (0.81, "maker_too_hard", "easier_puzzles", 1.4)


def test_render_log_handles_empty_and_full():
    cfg = {"run": {"name": "t", "total_steps": 10, "log_every": 1}, "decision": {"every": 5}}
    header = {"profile": "cpu", "model": "m", "engine": "mock", "referee": "rules", "harness": "pi"}
    assert "step 0 / 10" in render_log(cfg=cfg, header=header, history=[], eta=ETA(), started=0, verdicts=[], sample=None)
    e = ETA()
    e.update(60)
    text = render_log(cfg=cfg, header=header, history=_history(3), eta=e, started=0, verdicts=[], sample=None)
    assert "step 3 / 10" in text and "ETA" in text
