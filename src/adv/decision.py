"""The referee: every K steps, decide whether the game has stalled and what to change.

Adversarial training fails in recognisable ways (the GAN literature calls the classic one
"mode collapse"). Here the usual suspects are:
  maker_too_hard  - Solver almost never passes; no learning signal
  maker_too_easy  - Solver always passes; no learning signal
  maker_invalid   - most puzzles are broken
  mode_collapse   - the Maker keeps writing the same puzzle
  solver_plateau  - nothing moves at all
  overthinking    - thinking keeps hitting its budget

d3-flash is a *decision model*: you give it a JSON "state" plus typed questions (choice,
yes/no = "noul", score) and it returns a probability for every answer without generating
text. We run its GGUF through llama.cpp's /v1/systemone endpoint. If it cannot run (not enough
memory, old llama.cpp), a transparent rule-based referee answers the same questions.

The Controller turns a verdict into SMALL, BOUNDED changes and has a cooldown, so a wrong
call cannot wreck the run.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field

import requests

DIAGNOSES = {
    "healthy": "Training is progressing: rewards vary within groups and nothing is collapsing",
    "maker_too_hard": "The solver almost never passes the maker's puzzles",
    "maker_too_easy": "The solver almost always passes the maker's puzzles",
    "maker_invalid": "Most maker puzzles fail validation (broken reference solution or format)",
    "mode_collapse": "The maker keeps producing near-identical puzzles",
    "solver_plateau": "Solver pass rate and maker reward have been flat for a long time",
    "overthinking": "Thinking keeps running into its token budget",
}
ACTIONS = {
    "continue": "Change nothing",
    "easier_puzzles": "Ask the maker for lower difficulty",
    "harder_puzzles": "Ask the maker for higher difficulty",
    "raise_maker_temperature": "Sample the maker with more randomness to increase variety",
    "lower_learning_rate": "Halve both learning rates",
    "reset_maker": "Restore the maker adapter from its best earlier checkpoint",
    "inject_seed_puzzles": "Mix known-good seed puzzles into the solver's batch for a while",
}


@dataclass
class Verdict:
    step: int
    backend: str
    stalled: float
    diagnosis: str
    diagnosis_probs: dict
    action: str
    action_probs: dict
    progress: float | None = None
    applied: list = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def build_state(history: list[dict], knobs: dict, window: int = 10, examples: list[str] | None = None) -> dict:
    """Compact JSON summary of recent metrics for the referee (keep it small: CPU prompt eval)."""
    recent, prev = history[-window:], history[-2 * window:-window]

    def avg(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    keys = ["solver_pass_rate", "solver_test_frac", "maker_valid_rate", "maker_learnability",
            "maker_novelty_sim", "dead_group_frac", "think_tokens_solver", "think_forced_frac",
            "marker_rate_per_1k", "solver_loss", "maker_loss"]
    state = {
        "what_this_is": ("Two LoRA adapters on one LLM play a game. A MAKER writes Python puzzles, a SOLVER "
                         "tries them several times. The maker is rewarded most when the solver passes about "
                         "half the attempts (learnability 1.0); invalid puzzles score -1."),
        "metric_meanings": {
            "solver_pass_rate": "fraction of solver attempts passing all hidden tests (ideal mid-range 0.3-0.7)",
            "maker_valid_rate": "fraction of maker puzzles that passed validation",
            "maker_learnability": "mean 1-2|p-0.5|; 1.0 = puzzles exactly at the edge of solver ability",
            "maker_novelty_sim": "similarity of new puzzles to recent ones (high = repetitive)",
            "dead_group_frac": "fraction of puzzles where all attempts scored the same (no learning signal)",
            "think_forced_frac": "fraction of generations where thinking hit its budget",
        },
        "step": history[-1]["step"] if history else 0,
        "recent_window": {k: avg(recent, k) for k in keys},
        "previous_window": {k: avg(prev, k) for k in keys} if prev else "not enough history",
        "current_settings": knobs,
    }
    if examples:
        state["recent_puzzle_titles"] = examples[-6:]
    return state


QUESTIONS = {
    "stalled": {"type": "noul",
                "instructions": "Has this adversarial training run stalled or started collapsing, so that its settings should be changed?",
                "criteria": {"true": "Stalled, collapsing or wasting compute", "false": "Healthy enough to continue"}},
    "diagnosis": {"type": "choice", "instructions": "What best describes the current state of the run?",
                  "criteria": DIAGNOSES},
    "action": {"type": "choice", "instructions": "Which single adjustment would help the run most right now?",
               "criteria": ACTIONS},
    "progress": {"type": "score", "instructions": "How is the solver's ability trending?",
                 "criteria": ["getting worse", "flat", "slowly improving", "clearly improving"]},
}


class D3Referee:
    """d3-flash GGUF served by llama-server's /v1/systemone (started only when needed)."""

    backend = "d3-flash"

    def __init__(self, server_exe: str, gguf_path: str, port: int = 8383, gpu_layers: str = "0",
                 threads: int | None = None, keep_loaded: bool = False, log_file: str | None = None):
        self.server_exe, self.gguf_path, self.port = server_exe, gguf_path, port
        self.gpu_layers, self.threads, self.keep_loaded, self.log_file = gpu_layers, threads, keep_loaded, log_file
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None

    def _start(self):
        if self.proc and self.proc.poll() is None:
            return
        cmd = [self.server_exe, "-m", self.gguf_path, "--host", "127.0.0.1", "--port", str(self.port),
               "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "1", "-ngl", str(self.gpu_layers), "--no-webui"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        if self.log_file:
            cmd += ["--log-file", self.log_file]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"d3 llama-server exited ({self.proc.returncode})")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("d3 llama-server did not start")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def ask(self, state: dict) -> dict:
        self._start()
        try:
            r = requests.post(self.url + "/v1/systemone", json={"state": state, "questions": QUESTIONS}, timeout=1800)
            r.raise_for_status()
            return r.json()["answers"]
        finally:
            if not self.keep_loaded:
                self.stop()


class RuleReferee:
    """Plain thresholds. Also a good baseline to compare d3-flash against."""

    backend = "rules"

    def ask(self, state: dict) -> dict:
        w = state["recent_window"]
        prev = state["previous_window"] if isinstance(state["previous_window"], dict) else {}
        g = lambda k, d=0.0: w.get(k) if w.get(k) is not None else d  # noqa: E731
        diag = "healthy"
        if g("maker_valid_rate", 1) < 0.3:
            diag = "maker_invalid"
        elif g("solver_pass_rate", 0.5) < 0.08:
            diag = "maker_too_hard"
        elif g("solver_pass_rate", 0.5) > 0.9:
            diag = "maker_too_easy"
        elif g("maker_novelty_sim") > 0.6:
            diag = "mode_collapse"
        elif g("think_forced_frac") > 0.6:
            diag = "overthinking"
        elif prev and g("dead_group_frac") > 0.8 and abs(g("solver_pass_rate") - (prev.get("solver_pass_rate") or 0)) < 0.02:
            diag = "solver_plateau"
        action = {"healthy": "continue", "maker_invalid": "inject_seed_puzzles", "maker_too_hard": "easier_puzzles",
                  "maker_too_easy": "harder_puzzles", "mode_collapse": "raise_maker_temperature",
                  "overthinking": "continue", "solver_plateau": "lower_learning_rate"}[diag]
        one_hot = lambda keys, k: {x: (1.0 if x == k else 0.0) for x in keys}  # noqa: E731
        trend = None
        if prev and prev.get("solver_pass_rate") is not None and w.get("solver_pass_rate") is not None:
            d = w["solver_pass_rate"] - prev["solver_pass_rate"]
            trend = 0.0 if d < -0.03 else 1.0 if d < 0.02 else 2.0 if d < 0.08 else 3.0
        return {"stalled": {"noul": 0.0 if diag in ("healthy", "overthinking") else 1.0},
                "diagnosis": {"choice": diag, "probabilities": one_hot(DIAGNOSES, diag)},
                "action": {"choice": action, "probabilities": one_hot(ACTIONS, action)},
                "progress": {"score": trend}}


def parse_answers(answers: dict, step: int, backend: str, seconds: float) -> Verdict:
    def choice(key, options):
        a = answers.get(key) or {}
        probs = a.get("probabilities") or {}
        best = a.get("choice") or (max(probs, key=probs.get) if probs else "continue" if key == "action" else "healthy")
        return (best if best in options else list(options)[0]), {k: round(float(v), 3) for k, v in probs.items()}
    diag, dprobs = choice("diagnosis", DIAGNOSES)
    act, aprobs = choice("action", ACTIONS)
    stalled = float((answers.get("stalled") or {}).get("noul") or 0.0)
    prog = (answers.get("progress") or {}).get("score")
    return Verdict(step, backend, round(stalled, 3), diag, dprobs, act, aprobs,
                   None if prog is None else round(float(prog), 2), seconds=round(seconds, 1))


class Controller:
    """Owns the adjustable knobs and applies verdicts with bounds and a cooldown."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.knobs = {"difficulty": int(cfg.get("start_difficulty", 2)),
                      "maker_temperature": float(cfg.get("maker_temperature", 1.0)),
                      "lr_scale": 1.0, "inject_seeds_steps": 0}
        self.last_action_step = -10**9

    def apply(self, v: Verdict, step: int) -> list[str]:
        thr, cooldown = self.cfg.get("stall_threshold", 0.6), self.cfg.get("cooldown_steps", 15)
        if v.action == "continue" or v.stalled < thr:
            return []
        if step - self.last_action_step < cooldown:
            return [f"skipped {v.action} (cooldown)"]
        k, done = self.knobs, []
        if v.action == "easier_puzzles" and k["difficulty"] > 1:
            k["difficulty"] -= 1
            done.append(f"difficulty -> {k['difficulty']}")
        elif v.action == "harder_puzzles" and k["difficulty"] < 5:
            k["difficulty"] += 1
            done.append(f"difficulty -> {k['difficulty']}")
        elif v.action == "raise_maker_temperature" and k["maker_temperature"] < 1.3:
            k["maker_temperature"] = round(min(1.3, k["maker_temperature"] + 0.1), 2)
            done.append(f"maker temperature -> {k['maker_temperature']}")
        elif v.action == "lower_learning_rate" and k["lr_scale"] > 0.125:
            k["lr_scale"] /= 2
            done.append(f"lr scale -> {k['lr_scale']}")
        elif v.action == "inject_seed_puzzles":
            k["inject_seeds_steps"] = self.cfg.get("inject_steps", 5)
            done.append(f"inject seed puzzles for {k['inject_seeds_steps']} steps")
        elif v.action == "reset_maker":
            done.append("reset_maker")  # train loop performs the reload
        if done:
            self.last_action_step = step
        return done

    def state(self) -> dict:
        return {"knobs": self.knobs, "last_action_step": self.last_action_step}

    def load(self, s: dict):
        self.knobs.update(s.get("knobs", {}))
        self.last_action_step = s.get("last_action_step", self.last_action_step)


def consult(referee, history: list[dict], knobs: dict, step: int, examples: list[str]) -> Verdict:
    state = build_state(history, knobs, examples=examples)
    t0 = time.monotonic()
    try:
        answers = referee.ask(state)
        return parse_answers(answers, step, referee.backend, time.monotonic() - t0)
    except Exception as e:  # never let the referee crash training
        fallback = RuleReferee()
        v = parse_answers(fallback.ask(state), step, "rules (d3 failed)", time.monotonic() - t0)
        v.error = str(e)[:300]
        return v


def dumps_state_preview(history, knobs) -> str:
    return json.dumps(build_state(history, knobs), indent=1)[:4000]
