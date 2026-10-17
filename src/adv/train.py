"""The adversarial training loop.

    python -m adv.train                 # train (what run.ps1 / run.sh call)
    python -m adv.train --resume        # continue from the latest checkpoint
    python -m adv.train --dry-run       # whole loop with a fake model: no weights, no GPU, minutes

One step:
  1. MAKER writes M puzzles (topic + difficulty hint + 2 seed examples as few-shot)
  2. sandbox validates them (reference solution runs, deterministic, sensible tests)
  3. SOLVER attempts every valid puzzle G times, inside Pi (or single-shot)
  4. hidden tests -> solver rewards; solver success rate p̂ -> maker learnability rewards
  5. GRPO update of the solver adapter, then of the maker adapter
  6. new adapters -> rollout engine; metrics -> log.txt + train.png; every K steps the referee
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config as C
from .bias import load_markers, resolve
from .decision import Controller, D3Referee, RuleReferee, consult
from .engines.base import GenParams
from .grpo import Sample, group_advantages
from .logbook import ETA, History, atomic_write, render_log
from .plots import render as render_png
from .proxy import RecordingProxy, Rollout
from .puzzles import (PI_SOLVER_PROMPT, NoveltyBuffer, Puzzle, extract_python_code, load_seed_puzzles,
                      maker_messages, parse_maker_output, pick_topic, solver_messages_single, validate,
                      write_workspace)
from .rewards import learnability, maker_reward, solver_reward
from .sandbox import run_function, score_against

P_BUCKETS = ["0", "0.25", "0.5", "0.75", "1"]


@dataclass
class MakerOut:
    rollout: Rollout
    puzzle: Puzzle | None
    reason: str
    novelty: float = 0.0


@dataclass
class Attempt:
    puzzle: Puzzle
    rollout: Rollout
    code: str | None
    test_frac: float
    reward: float
    pi: object = None


def _bucket(p: float) -> str:
    return P_BUCKETS[min(range(5), key=lambda i: abs(i * 0.25 - p))]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class Trainer:
    def __init__(self, cfg: dict, args):
        self.cfg, self.args = cfg, args
        self.rng = random.Random(cfg["run"]["seed"])
        self.profile = args.profile or C.resolve_profile(cfg)
        self.prof = cfg["profiles"][self.profile]
        self.run_dir = C.ROOT / "runs" / cfg["run"]["name"]
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        self.log_path, self.png_path = C.ROOT / "log.txt", C.ROOT / "train.png"
        self.history = History(self.run_dir / "metrics.jsonl")
        self.verdicts: list[dict] = []
        self.eta = ETA()
        self.novelty = NoveltyBuffer()
        self.titles: list[str] = []
        self.controller = Controller(cfg["decision"])
        self.best_maker = {"score": -1e9, "dir": None}
        self.started = time.time()
        self.start_step = 0
        self.tools = C.tools()
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self):
        cfg, prof, args = self.cfg, self.prof, self.args
        gen = cfg["generation"]
        if args.dry_run:
            from .engines.mock import MockEngine
            from .policy import MockPolicy
            from .tokenizer import MockTokenizer
            self.tok = MockTokenizer(gen["enable_thinking"])
            self.engine = MockEngine(self.tok, cfg["run"]["seed"])
            self.policy = MockPolicy(self.engine, cfg["run"]["seed"])
            self.engine_name, self.model_name = "mock", "fake model, no weights"
        else:
            import torch
            from .tokenizer import ChatTokenizer
            train_dir = C.repo_dir(prof["train_repo"])
            if not (train_dir / "config.json").exists():
                sys.exit(f"Model not found at {train_dir}. Run setup first (it downloads the models).")
            self.tok = ChatTokenizer(str(train_dir), gen["enable_thinking"], gen.get("reasoning_effort"))
            device = "cuda" if self.profile.startswith("cuda") and torch.cuda.is_available() else "cpu"
            if device == "cpu":
                import psutil
                torch.set_num_threads(psutil.cpu_count(logical=False) or os.cpu_count())
            from .policy import Policy
            self.engine_name = self._pick_engine()
            st = C.read_state("engine.json")
            if self.engine_name == "llamacpp" and st.get("lora_targets") == "mlp_only":
                cfg["lora"]["target_modules"] = "mlp_only"  # what the self-test found llama.cpp accepts
            self.policy = Policy(str(train_dir), cfg["lora"], cfg["optim"], prof["quant_4bit"], prof["dtype"], device)
            self.model_name = prof["train_repo"].split("/")[-1]
            if self.engine_name == "llamacpp":
                from .engines.llamacpp import LlamaCppEngine
                gguf = C.repo_dir(prof["gguf_repo"]) / prof["gguf_file"]
                self.engine = LlamaCppEngine(
                    self.tok, self.tools["llama_server"], str(gguf), port=cfg["ports"]["engine"],
                    ctx_per_slot=prof["ctx_per_slot"], parallel=prof["parallel_slots"],
                    gpu_layers=prof["llama_gpu_layers"], log_file=str(self.run_dir / "logs" / "llama-server.log"),
                    converter={"script": self.tools["convert_lora"], "gguf_py": self.tools["gguf_py"],
                               "python": sys.executable, "base_model_dir": str(train_dir)},
                    adapter_gguf_dir=str(self.run_dir / "adapters_gguf"),
                    swap_during_training=prof["swap_engine_during_training"])
            else:
                from .engines.hf import HFEngine
                self.engine = HFEngine(self.tok, self.policy)

        # overthinking bias (thinking-only), re-derived from the words with this tokenizer
        ob = cfg["overthinking_bias"]
        self.bias_report = None
        bias_ids = {}
        if ob["enabled"] and not args.dry_run:  # the mock tokenizer is character-level: nothing to bias
            self.bias_report = resolve(self.tok, load_markers(C.ROOT / ob["file"]))
            bias_ids = self.bias_report.ids
            print("overthinking bias:", self.bias_report.summary())
        self.params = {}
        for role in ("maker", "solver"):
            g = gen[role]
            self.params[role] = GenParams(adapter=role, max_new_tokens=g["max_new_tokens"], temperature=g["temperature"],
                                          top_p=g["top_p"], top_k=g["top_k"], think_budget=g["think_budget"],
                                          think_bias=bias_ids if role in ob["roles"] else {})
        self.proxy = RecordingProxy(self.tok, self.engine, self.params, max_ctx=prof["ctx_per_slot"],
                                    port=cfg["ports"]["proxy"])

        self.pi = None
        if cfg["solver"]["use_pi"] and not args.no_pi:
            node = args.pi_node or self.tools["node"]
            cli = args.pi_cli or self.tools["pi_cli"]
            if node and cli:
                from .pi_runner import PiRunner
                self.proxy.start_http()
                self.pi = PiRunner(node, cli, self.run_dir / "pi_agent", self.proxy.base_url, prof["ctx_per_slot"],
                                   gen["solver"]["max_new_tokens"], shell_path=self.tools["bash"],
                                   extra_path=[os.path.dirname(sys.executable)], tools=cfg["solver"]["pi_tools"],
                                   timeout_s=cfg["solver"]["timeout_s"])
            else:
                print("WARNING: Pi (node + pi-coding-agent) not found; falling back to single-shot solver")
        self.harness = "pi" if self.pi else "single-shot"
        self.referee = self._pick_referee()
        self.seeds = load_seed_puzzles(C.ROOT / "data" / "seed_puzzles.jsonl", cfg["sandbox"]["timeout_s"])

    def _pick_engine(self) -> str:
        choice = self.cfg.get("engine", "auto")
        if choice != "auto":
            return choice
        st = C.read_state("engine.json")
        if st.get("engine") == "llamacpp" and self.tools["llama_server"] and self.tools["convert_lora"]:
            return "llamacpp"
        return "hf"

    def _pick_referee(self):
        d = self.cfg["decision"]
        if self.args.dry_run or d["backend"] == "rules":
            return RuleReferee()
        gguf = C.repo_dir(d["d3_repo"]) / d["d3_file"]
        if not (gguf.exists() and self.tools["llama_server"]):
            print("referee: d3-flash GGUF or llama-server missing -> rule-based referee")
            return RuleReferee()
        if d["backend"] == "auto":
            import psutil
            free_gb = psutil.virtual_memory().available / 1e9
            if free_gb < gguf.stat().st_size / 1e9 + 2:
                print(f"referee: only {free_gb:.1f} GB RAM free -> rule-based referee")
                return RuleReferee()
        return D3Referee(self.tools["llama_server"], str(gguf), port=self.cfg["ports"]["d3"],
                         gpu_layers=d["gpu_layers"], keep_loaded=d["keep_loaded"],
                         log_file=str(self.run_dir / "logs" / "d3-server.log"))

    # ------------------------------------------------------------------ rollouts
    def _maker_one(self, msgs) -> MakerOut:
        rid = self.proxy.begin("maker", max_turns=1)
        try:
            res = self.proxy.chat(rid, msgs)
        finally:
            ro = self.proxy.end(rid)
        puzzle, reason = parse_maker_output(res["message"].get("content") or "")
        if puzzle is None:
            return MakerOut(ro, None, reason)
        v = validate(puzzle, self.cfg["sandbox"]["timeout_s"])
        if not v.valid:
            return MakerOut(ro, None, v.reason)
        return MakerOut(ro, puzzle, "ok")

    def _solve_one(self, p: Puzzle) -> Attempt:
        scfg = self.cfg["solver"]
        pi_res, code = None, None
        if self.pi:
            ws = Path(tempfile.mkdtemp(prefix="adv_ws_"))
            try:
                write_workspace(p, ws, scfg["visible_tests"])
                stub = (ws / "solution.py").read_text(encoding="utf-8")
                rid = self.proxy.begin("solver", max_turns=scfg["max_turns"])
                try:
                    pi_res = self.pi.run(ws, PI_SOLVER_PROMPT, rid)
                finally:
                    ro = self.proxy.end(rid)
                sol = ws / "solution.py"
                if sol.exists():
                    text = sol.read_text(encoding="utf-8", errors="replace")
                    code = text if text.strip() != stub.strip() else None
            finally:
                shutil.rmtree(ws, ignore_errors=True)
        else:
            rid = self.proxy.begin("solver", max_turns=1)
            try:
                res = self.proxy.chat(rid, solver_messages_single(p, scfg["visible_tests"]))
            finally:
                ro = self.proxy.end(rid)
            code = extract_python_code(res["message"].get("content") or "", p.function_name)
        frac = 0.0
        if code:
            rr = run_function(code, p.function_name, p.test_inputs, self.cfg["sandbox"]["timeout_s"],
                              max_memory_mb=self.cfg["sandbox"]["max_memory_mb"])
            frac = score_against(rr, p.expected)
        return Attempt(p, ro, code, frac, solver_reward(frac, code is not None, self.cfg["rewards"]), pi_res)

    def _pmap(self, fn, items):
        workers = max(1, min(len(items), getattr(self.engine, "max_parallel", 1)))
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, items))

    # ------------------------------------------------------------------ one step
    def step(self, step: int) -> dict:
        cfg, knobs = self.cfg, self.controller.knobs
        g = cfg["grpo"]
        t0 = time.monotonic()
        self.params["maker"].temperature = knobs["maker_temperature"]
        self.engine.resume()

        # 1-2. Maker writes puzzles; sandbox validates
        jobs = [maker_messages(pick_topic(self.rng), knobs["difficulty"], self.rng.sample(self.seeds, 2), self.titles)
                for _ in range(g["puzzles_per_step"])]
        makers = self._pmap(self._maker_one, jobs)
        for m in makers:
            if m.puzzle:
                m.novelty = self.novelty.max_similarity(m.puzzle.statement)
                self.novelty.add(m.puzzle.statement)
                self.titles.append(m.puzzle.title)

        # 3-4. Solver attempts each valid puzzle G times (+ seed puzzles if the referee asked)
        batch = [m.puzzle for m in makers if m.puzzle]
        if knobs["inject_seeds_steps"] > 0:
            batch += self.rng.sample(self.seeds, 2)
            knobs["inject_seeds_steps"] -= 1
        attempts = self._pmap(self._solve_one, [p for p in batch for _ in range(g["solver_attempts"])])
        t_roll = time.monotonic() - t0

        by_uid: dict[str, list[Attempt]] = {}
        for a in attempts:
            by_uid.setdefault(a.puzzle.uid, []).append(a)
        p_hat = {uid: sum(1.0 for a in atts if a.test_frac >= 1.0) / len(atts) for uid, atts in by_uid.items()}
        maker_rewards = [maker_reward(m.puzzle is not None, p_hat.get(m.puzzle.uid, 0.0) if m.puzzle else 0.0,
                                      m.novelty, cfg["rewards"]) for m in makers]

        # 5. GRPO samples. Solver: group = puzzle. Maker: baseline = this step's batch.
        s_adv, dead = group_advantages([a.reward for a in attempts], [a.puzzle.uid for a in attempts], g["scale_by_std"])
        m_adv, _ = group_advantages(maker_rewards, ["batch"] * len(makers), g["scale_by_std"])
        solver_samples = [Sample(ids, mask, adv, "solver", a.puzzle.uid, a.reward)
                          for a, adv in zip(attempts, s_adv) for ids, mask in a.rollout.sequences()]
        maker_samples = [Sample(ids, mask, adv, "maker", "batch", r)
                         for m, adv, r in zip(makers, m_adv, maker_rewards) for ids, mask in m.rollout.sequences()]

        t1 = time.monotonic()
        self.engine.pause()
        kw = dict(grad_clip=cfg["optim"]["grad_clip"], max_seq_len=self.prof["max_seq_len"], ppo_epochs=g["ppo_epochs"],
                  clip_eps=g["clip_eps"], kl_beta=g["kl_beta"], lr_scale=knobs["lr_scale"])
        s_st = self.policy.train_step("solver", solver_samples, **kw)
        m_st = self.policy.train_step("maker", maker_samples, **kw)
        self._sync_engine()
        t_train = time.monotonic() - t1

        # metrics
        turns = [t for r in [m.rollout for m in makers] + [a.rollout for a in attempts] for t in r.turns]
        think_s = [t.gen.think_tokens for a in attempts for t in a.rollout.turns]
        think_m = [t.gen.think_tokens for m in makers for t in m.rollout.turns]
        markers = sum(t.gen.marker_tokens for t in turns)
        think_total = sum(t.gen.think_tokens for t in turns)
        maker_valid = [m for m in makers if m.puzzle]
        counts = Counter(_bucket(p_hat[m.puzzle.uid]) for m in maker_valid if m.puzzle.uid in p_hat)
        row = {
            "step": step, "step_seconds": time.monotonic() - t0, "rollout_seconds": t_roll, "train_seconds": t_train,
            "solver_pass_rate": _mean([1.0 if a.test_frac >= 1 else 0.0 for a in attempts]),
            "solver_test_frac": _mean([a.test_frac for a in attempts]),
            "solver_no_code_frac": _mean([0.0 if a.code else 1.0 for a in attempts]),
            "maker_valid_rate": len(maker_valid) / max(1, len(makers)),
            "maker_learnability": _mean([learnability(p_hat[m.puzzle.uid]) for m in maker_valid if m.puzzle.uid in p_hat]),
            "maker_reward": _mean(maker_rewards),
            "maker_novelty_sim": _mean([m.novelty for m in maker_valid]),
            "p_hat_counts": dict(counts), "p_hat_hist": " ".join(f"{b}:{counts.get(b, 0)}" for b in P_BUCKETS),
            "invalid_reasons": dict(Counter(m.reason for m in makers if not m.puzzle)),
            "think_tokens_solver": _mean(think_s), "think_tokens_maker": _mean(think_m),
            "think_forced_frac": _mean([1.0 if t.gen.forced_think_end else 0.0 for t in turns]),
            "marker_rate_per_1k": 1000 * markers / think_total if think_total else None,
            "gen_tokens": sum(len(t.gen.token_ids) for t in turns),
            "dead_group_frac": dead, "skipped_long": s_st["skipped_long"] + m_st["skipped_long"],
            "turn_limit_hits": sum(1 for a in attempts if a.rollout.truncated),
        }
        for prefix, st in (("solver", s_st), ("maker", m_st)):
            for k in ("loss", "grad_norm", "samples", "tokens", "kl", "clip_frac"):
                row[f"{prefix}_{k}"] = st.get(k)
        if self.pi:
            pis = [a.pi for a in attempts if a.pi]
            row["pi_tool_calls"] = _mean([p.tool_calls for p in pis])
            row["pi_tool_errors"] = _mean([p.tool_errors for p in pis])
            row["pi_timeouts"] = sum(1 for p in pis if p.timed_out)
        for k, v in knobs.items():
            row[f"knob_{k}"] = v
        self.sample = self._pick_sample(maker_valid, by_uid, p_hat)
        return row

    def _pick_sample(self, maker_valid, by_uid, p_hat):
        if not maker_valid:
            return None
        m = min(maker_valid, key=lambda m: abs(p_hat.get(m.puzzle.uid, 0) - 0.5))
        atts = sorted(by_uid.get(m.puzzle.uid, []), key=lambda a: -a.test_frac)
        best = atts[0] if atts else None
        return {"title": m.puzzle.title, "difficulty": m.puzzle.difficulty, "p_hat": p_hat.get(m.puzzle.uid, 0.0),
                "statement": m.puzzle.statement, "attempt": best.code if best else None,
                "attempt_score": best.test_frac if best else 0.0}

    def _sync_engine(self):
        """Push the freshly trained adapters into the rollout engine."""
        if self.engine_name != "llamacpp":
            self.engine.resume()
            return
        from .policy import export_for_llamacpp
        export = self.run_dir / "adapters_export"
        raw = self.policy.save(export / "raw", with_optimizer=False)
        dirs = {n: export_for_llamacpp(d, str(C.repo_dir(self.prof["train_repo"])), str(export / n))
                for n, d in raw.items()}
        self.engine.sync_adapters(dirs)

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, step: int):
        ck = self.run_dir / "checkpoints" / f"step_{step + 1:05d}"
        self.policy.save(ck)
        recent = self.history.rows[-10:]
        score = _mean([r.get("maker_learnability") for r in recent]) or -1e9
        if score > self.best_maker["score"]:
            best = self.run_dir / "checkpoints" / "best_maker"
            shutil.rmtree(best, ignore_errors=True)
            shutil.copytree(ck / "maker", best / "maker")
            if (ck / "solver").exists():
                shutil.copytree(ck / "solver", best / "solver")
            self.best_maker = {"score": score, "dir": str(best), "step": step + 1}
        state = {"step": step, "controller": self.controller.state(), "eta": self.eta.state(),
                 "novelty": self.novelty.state(), "titles": self.titles[-200:], "verdicts": self.verdicts,
                 "best_maker": self.best_maker, "elapsed": time.time() - self.started,
                 "rng": [self.rng.getstate()[0], list(self.rng.getstate()[1]), self.rng.getstate()[2]]}
        (ck / "state.json").write_text(json.dumps(state), encoding="utf-8")
        atomic_write(self.run_dir / "checkpoints" / "latest.txt", ck.name)
        ckpts = sorted(p for p in (self.run_dir / "checkpoints").glob("step_*") if p.is_dir())
        for old in ckpts[:-self.cfg["run"]["keep_checkpoints"]]:
            shutil.rmtree(old, ignore_errors=True)

    def try_resume(self):
        latest = self.run_dir / "checkpoints" / "latest.txt"
        if not latest.exists():
            print("no checkpoint found; starting fresh")
            return
        ck = self.run_dir / "checkpoints" / latest.read_text().strip()
        st = json.loads((ck / "state.json").read_text(encoding="utf-8"))
        self.policy.load(ck)
        self.controller.load(st["controller"])
        self.eta.load(st["eta"])
        self.novelty.load(st["novelty"])
        self.titles, self.verdicts, self.best_maker = st["titles"], st["verdicts"], st["best_maker"]
        self.rng.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
        self.started = time.time() - st.get("elapsed", 0)
        self.start_step = st["step"] + 1
        self.history.truncate_to(st["step"])
        print(f"resumed from {ck.name} (next step {self.start_step + 1})")

    # ------------------------------------------------------------------ output
    def write_outputs(self, note: str = ""):
        header = {"profile": self.profile, "model": self.model_name, "engine": self.engine_name,
                  "referee": self.referee.backend, "harness": self.harness}
        text = render_log(cfg=self.cfg, header=header, history=self.history.rows, eta=self.eta, started=self.started,
                          verdicts=self.verdicts, sample=getattr(self, "sample", None))
        if self.bias_report:
            text += f"\noverthinking bias: {self.bias_report.summary()}\n"
        if note:
            text = note + "\n\n" + text
        atomic_write(self.log_path, text)
        total = self.cfg["run"]["total_steps"]
        done = self.history.rows[-1]["step"] + 1 if self.history.rows else 0
        rem = self.eta.remaining(total - done, max(0, (total - done) // self.cfg["decision"]["every"]))
        render_png(self.history.rows, self.verdicts,
                   f"{self.cfg['run']['name']}: step {done}/{total} ({self.model_name}, {self.harness})", rem, self.png_path)

    # ------------------------------------------------------------------ main loop
    def run(self):
        cfg = self.cfg
        total = cfg["run"]["total_steps"]
        max_h = self.args.max_hours if self.args.max_hours is not None else cfg["run"]["max_hours"]
        if self.args.resume:
            self.try_resume()
        if self.engine_name == "llamacpp":
            self._sync_engine()  # starts llama-server with the (initial or resumed) adapters
        else:
            self.engine.start()
        atomic_write(self.log_path, f"starting {cfg['run']['name']} on profile {self.profile} "
                                    f"({self.model_name}), engine {self.engine_name}, harness {self.harness}...\n")
        step = self.start_step - 1
        note = ""
        try:
            for step in range(self.start_step, total):
                row = self.step(step)
                self.eta.update(row["step_seconds"])
                self.history.append(row)
                if (step + 1) % cfg["decision"]["every"] == 0:
                    self._referee(step)
                if (step + 1) % cfg["run"]["checkpoint_every"] == 0:
                    self.save_checkpoint(step)
                if (step + 1) % cfg["run"]["log_every"] == 0 or step + 1 == total:
                    self.write_outputs()
                if max_h and (time.time() - self.started) / 3600 >= max_h:
                    note = f"STOPPED after {max_h} h (max_hours). Run resume to continue."
                    break
            else:
                note = "FINISHED all steps."
        except KeyboardInterrupt:
            note = "STOPPED by Ctrl+C. Run resume to continue."
        except Exception:
            note = "CRASHED:\n" + traceback.format_exc()
            raise
        finally:
            if step >= self.start_step:
                self.save_checkpoint(step)
            self.write_outputs(note)
            self.engine.stop()
            self.proxy.stop_http()
            if isinstance(self.referee, D3Referee):
                self.referee.stop()
            print(note)

    def _referee(self, step: int):
        v = consult(self.referee, self.history.rows, dict(self.controller.knobs), step, self.titles)
        v.applied = self.controller.apply(v, step)
        if "reset_maker" in v.applied:
            if self.best_maker.get("dir"):
                self.policy.load(Path(self.best_maker["dir"]), names=("maker",), with_optimizer=False)
                self._sync_engine()
                v.applied[v.applied.index("reset_maker")] = f"maker reset to step {self.best_maker.get('step')}"
            else:
                v.applied.remove("reset_maker")
        self.eta.decision_cost = v.seconds if not self.eta.decision_cost else 0.5 * (self.eta.decision_cost + v.seconds)
        self.verdicts.append(v.to_dict())
        atomic_write(self.run_dir / "verdicts.json", json.dumps(self.verdicts, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="fake model: tests the whole pipeline in minutes")
    ap.add_argument("--steps", type=int, help="override run.total_steps")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--profile", choices=["cuda_large", "cuda_small", "cpu", "cpu_tiny"])
    ap.add_argument("--name", help="override run.name")
    ap.add_argument("--no-pi", action="store_true", help="single-shot solver instead of Pi")
    ap.add_argument("--pi-node", help="path to node (default: tools/node)")
    ap.add_argument("--pi-cli", help="path to pi's cli.js (default: tools/pi)")
    args = ap.parse_args(argv)
    overrides = {}
    if args.steps:
        overrides["run.total_steps"] = args.steps
    if args.name:
        overrides["run.name"] = args.name
    if args.dry_run and not args.name:
        overrides["run.name"] = "dry_run"
    cfg = C.load(args.config, overrides)
    Trainer(cfg, args).run()


if __name__ == "__main__":
    main()
