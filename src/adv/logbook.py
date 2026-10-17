"""Metrics history, ETA, and the human-readable log.txt (overwritten in place)."""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import time
from pathlib import Path


def atomic_write(path: Path, data: bytes | str):
    """Write to a temp file then rename, so a viewer never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode, **({} if isinstance(data, bytes) else {"encoding": "utf-8"})) as f:
        f.write(data)
    for attempt in range(5):  # Windows: another program may briefly hold the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.3 * (attempt + 1))
    with open(path, mode, **({} if isinstance(data, bytes) else {"encoding": "utf-8"})) as f:
        f.write(data)


def fmt_dur(s: float | None) -> str:
    if s is None or s != s:
        return "?"
    s = int(max(0, s))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class ETA:
    """Blend of an exponential moving average and the median of recent step times.
    The first steps include warm-up (loading, compilation), so the estimate settles after ~3 steps."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.ema: float | None = None
        self.recent: list[float] = []
        self.decision_cost: float = 0.0

    def update(self, step_seconds: float):
        self.ema = step_seconds if self.ema is None else self.alpha * step_seconds + (1 - self.alpha) * self.ema
        self.recent = (self.recent + [step_seconds])[-10:]

    def per_step(self) -> float | None:
        if self.ema is None:
            return None
        return 0.5 * self.ema + 0.5 * statistics.median(self.recent)

    def remaining(self, steps_left: int, decisions_left: int = 0) -> float | None:
        ps = self.per_step()
        return None if ps is None else ps * steps_left + self.decision_cost * decisions_left

    def state(self):
        return {"ema": self.ema, "recent": self.recent, "decision_cost": self.decision_cost}

    def load(self, s):
        self.ema, self.recent, self.decision_cost = s.get("ema"), s.get("recent", []), s.get("decision_cost", 0.0)


class History:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    def truncate_to(self, step: int):
        """On resume, drop rows logged after the checkpoint we restart from."""
        self.rows = [r for r in self.rows if r["step"] <= step]
        atomic_write(self.path, "".join(json.dumps(r) + "\n" for r in self.rows))

    def append(self, row: dict):
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _f(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def render_log(*, cfg: dict, header: dict, history: list[dict], eta: ETA, started: float,
               verdicts: list[dict], sample: dict | None) -> str:
    total = cfg["run"]["total_steps"]
    last = history[-1] if history else {}
    step = last.get("step", 0)
    done = step + 1 if history else 0
    w10, p10 = history[-10:], history[-20:-10]
    every = cfg["decision"]["every"]
    decisions_left = max(0, (total - done) // every)
    rem = eta.remaining(total - done, decisions_left)
    finish = (dt.datetime.now() + dt.timedelta(seconds=rem)).strftime("%a %d %b %H:%M") if rem is not None else "?"

    def trend(key):
        a, b = _avg(w10, key), _avg(p10, key)
        return "" if a is None or b is None else f" (Δ{a - b:+.2f} vs previous 10)"

    L = []
    L.append(f"ADVERSARIAL CODER  -  {cfg['run']['name']}        (this file is overwritten every {cfg['run']['log_every']} step(s))")
    L.append(f"updated {dt.datetime.now():%Y-%m-%d %H:%M:%S} | profile {header['profile']} ({header['model']}) | "
             f"engine {header['engine']} | referee {header['referee']} | solver harness {header['harness']}")
    L.append("")
    L.append(f"step {done} / {total}  {_bar(done / total)}  {100 * done / total:.1f}%")
    L.append(f"elapsed {fmt_dur(time.time() - started)} | last step {fmt_dur(last.get('step_seconds'))} | "
             f"avg {fmt_dur(eta.per_step())} | ETA {fmt_dur(rem)}  (≈ {finish})")
    if done < 3:
        L.append("  (ETA is rough for the first few steps: they include model loading and warm-up)")
    L.append("")
    L.append("--- SOLVER " + "-" * 60)
    L.append(f"pass rate (all hidden tests)  {_f(last.get('solver_pass_rate'))}   last-10 avg {_f(_avg(w10, 'solver_pass_rate'))}{trend('solver_pass_rate')}")
    L.append(f"mean fraction of tests passed {_f(last.get('solver_test_frac'))}   no-code attempts {_f(last.get('solver_no_code_frac'))}")
    if header["harness"] == "pi":
        L.append(f"Pi tool calls / attempt {_f(last.get('pi_tool_calls'), 1)}   tool errors {_f(last.get('pi_tool_errors'), 1)}   "
                 f"timeouts {last.get('pi_timeouts', 0)}   episodes cut by turn/context limit {last.get('turn_limit_hits', 0)}")
    L.append("--- MAKER " + "-" * 61)
    L.append(f"valid puzzles {_f(last.get('maker_valid_rate'))}   learnability {_f(last.get('maker_learnability'))}   "
             f"reward {_f(last.get('maker_reward'))}   last-10 learnability {_f(_avg(w10, 'maker_learnability'))}{trend('maker_learnability')}")
    L.append(f"solver success per puzzle (p̂): {last.get('p_hat_hist', '-')}   novelty sim {_f(last.get('maker_novelty_sim'))}")
    if last.get("invalid_reasons"):
        L.append(f"invalid reasons: {last['invalid_reasons']}")
    L.append("--- GENERATION " + "-" * 56)
    L.append(f"thinking tokens  solver {_f(last.get('think_tokens_solver'), 0)}  maker {_f(last.get('think_tokens_maker'), 0)}   "
             f"budget hit {_f(last.get('think_forced_frac'))}   overthinking markers {_f(last.get('marker_rate_per_1k'), 1)} per 1k thinking tokens")
    L.append(f"generated tokens this step {last.get('gen_tokens', 0)}   rollout time {fmt_dur(last.get('rollout_seconds'))}   "
             f"train time {fmt_dur(last.get('train_seconds'))}")
    L.append("--- TRAINING " + "-" * 58)
    L.append(f"solver: loss {_f(last.get('solver_loss'), 4)} grad-norm {_f(last.get('solver_grad_norm'), 3)} "
             f"samples {last.get('solver_samples', 0)} tokens {last.get('solver_tokens', 0)}   "
             f"dead groups {_f(last.get('dead_group_frac'))}")
    L.append(f"maker:  loss {_f(last.get('maker_loss'), 4)} grad-norm {_f(last.get('maker_grad_norm'), 3)} "
             f"samples {last.get('maker_samples', 0)} tokens {last.get('maker_tokens', 0)}   "
             f"skipped (too long) {last.get('skipped_long', 0)}   "
             f"overflow-skipped updates {(last.get('solver_nonfinite_grads') or 0) + (last.get('maker_nonfinite_grads') or 0)}")
    L.append(f"knobs: difficulty {last.get('knob_difficulty')}  maker temp {last.get('knob_maker_temperature')}  "
             f"lr scale {last.get('knob_lr_scale')}  seed-injection steps left {last.get('knob_inject_seeds_steps')}")
    L.append("--- REFEREE " + "-" * 59)
    if verdicts:
        v = verdicts[-1]
        dp = sorted(v["diagnosis_probs"].items(), key=lambda kv: -kv[1])[:3]
        L.append(f"step {v['step'] + 1} via {v['backend']} ({v['seconds']}s): P(stalled) = {v['stalled']:.2f} | "
                 f"diagnosis {v['diagnosis']} | suggested {v['action']} | applied: {', '.join(v['applied']) or 'nothing'}")
        L.append("   top diagnoses: " + ", ".join(f"{k} {p:.2f}" for k, p in dp))
        if v.get("error"):
            L.append(f"   d3 error (fell back to rules): {v['error']}")
        acted = [x for x in verdicts if x["applied"]]
        if acted:
            L.append("   interventions so far: " + "; ".join(f"step {x['step'] + 1}: {', '.join(x['applied'])}" for x in acted[-6:]))
    else:
        L.append(f"first check after step {every}")
    if sample:
        L.append("--- SAMPLE FROM THIS STEP " + "-" * 45)
        L.append(f"puzzle: {sample['title']}  (difficulty {sample['difficulty']}, p̂ = {sample['p_hat']:.2f})")
        L.append("  " + sample["statement"][:600].replace("\n", "\n  "))
        if sample.get("attempt"):
            L.append(f"best solver attempt ({sample['attempt_score']:.0%} of tests):")
            L.append("  " + sample["attempt"][:1200].replace("\n", "\n  "))
    L.append("")
    return "\n".join(L)
