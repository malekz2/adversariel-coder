"""Measure, on the real machine, which llama.cpp rollout settings are fastest without changing the output.

Run by setup's self-test (adv.selftest) on llama.cpp profiles. Two stages:

1. KV-cache type: time each `kv_cache_types` entry with plain decoding at full concurrency and keep the
   fastest that starts (one 4090 report says some KV types silently fall back to the CPU on this model).
2. Speculative decoding: for each candidate (e.g. MTP head + n-gram, DFlash2), at each slot count:
   - it must still respond to the LoRA adapter (scale 0 vs 1 give different tokens),
   - its greedy tokens must agree with plain decoding (min_agreement; batched verification can round
     differently, so a late divergence on one prompt is tolerated, an immediate one everywhere is not),
   - then its throughput is timed: total tokens for a batch of concurrent requests / wall time.
   The fastest config wins, but speculation is used only if it beats plain decoding by min_gain.

Why time at concurrency: speculation mostly helps a single stream; with 8 Pi attempts at once the GPU is
already busy, so its single-request speed-up (e.g. +90 % for MTP) does not carry over. Measuring "how long
does a batch of attempts take" is what matters for a training step.

The result is stored in state/engine.json under "rollout" and read back by rollout_settings() at training time.
"""
from __future__ import annotations

import concurrent.futures as cf
import tempfile
import time
from pathlib import Path

import requests

DRAFTER_TYPES = {"draft-simple", "draft-eagle3", "draft-dflash", "draft-dspark"}  # need a separate -md file


def needs_drafter(spec_type: str) -> bool:
    return any(t.strip() in DRAFTER_TYPES for t in spec_type.split(","))


def speculative_args(spec_cfg: dict, choice: dict, draft_path: str | None) -> list[str]:
    """llama-server flags for one speculative candidate, e.g. {"type": "draft-mtp,ngram-mod", "n_max": 3}.
    draft-mtp uses the MTP head inside the target GGUF; DFlash/DSpark/EAGLE/simple need a drafter file."""
    args = ["--spec-type", choice["type"]]
    if needs_drafter(choice["type"]):
        if not draft_path:
            raise ValueError(f"{choice['type']} needs a draft model")
        args += ["-md", draft_path]
    if choice.get("n_max"):
        args += ["--spec-draft-n-max", str(choice["n_max"])]
    return args + [str(a) for a in spec_cfg.get("args") or []]


def agreement(ref: dict, got: dict) -> tuple[float, int]:
    """(mean fraction of tokens before the first difference, number of identical sequences)."""
    fracs, identical = [], 0
    for key, r in ref.items():
        g = got.get(key, [])
        n = max(len(r), 1)
        first = next((i for i, (a, b) in enumerate(zip(r, g)) if a != b), min(len(r), len(g)))
        if g == r:
            identical += 1
            first = len(r)
        fracs.append(min(first, n) / n)
    return (sum(fracs) / len(fracs) if fracs else 0.0), identical


def agrees(ref: dict, got: dict, min_agreement: float) -> tuple[bool, str]:
    frac, identical = agreement(ref, got)
    ok = frac >= min_agreement and identical >= (len(ref) + 1) // 2
    return ok, f"agreement {frac:.2f}, {identical}/{len(ref)} sequences identical"


class LlamaServerProbe:
    """Starts/stops llama-server and sends test requests."""

    def __init__(self, server_exe: str, port: int, startup_timeout: float = 900):
        self.server_exe, self.port, self.startup_timeout = server_exe, port, startup_timeout
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"

    def start(self, args: list[str], log_name: str = "adv_tune_server.log"):
        import subprocess
        self.stop()
        log = Path(tempfile.gettempdir()) / log_name
        self.proc = subprocess.Popen([self.server_exe, *args, "--host", "127.0.0.1", "--port", str(self.port),
                                      "--log-file", str(log)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.startup_timeout:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.proc = None
                raise RuntimeError(f"llama-server exited ({code}); see {log}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError("llama-server did not start in time")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
            self.proc = None

    def complete(self, prompt: list[int], scale: float, n: int, temperature: float = 0.0, seed: int = 0):
        r = requests.post(self.url + "/completion", timeout=900, json={
            "prompt": prompt, "n_predict": n, "temperature": temperature, "seed": seed, "return_tokens": True,
            "cache_prompt": False, "lora": [{"id": 0, "scale": scale}]})
        r.raise_for_status()
        d = r.json()
        return list(d.get("tokens") or []), d.get("timings") or {}

    def single_speed(self, n: int = 48) -> float | None:
        """One request alone: generation tok/s from llama-server's timings (prompt excluded)."""
        r = requests.post(self.url + "/completion", timeout=900, json={
            "prompt": [1, 2, 3, 4, 5, 6, 7, 8], "n_predict": n, "temperature": 0.0, "ignore_eos": True,
            "cache_prompt": False, "lora": [{"id": 0, "scale": 0.0}]})
        r.raise_for_status()
        tps = (r.json().get("timings") or {}).get("predicted_per_second")
        return round(float(tps), 1) if tps else None

    def throughput(self, prompts: list[list[int]], n: int, temperature: float = 0.8) -> dict:
        """All prompts at once (like parallel Pi attempts): generated tokens per wall-clock second."""
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            outs = list(ex.map(lambda ip: self.complete(ip[1], 0.0, n, temperature, seed=ip[0]), enumerate(prompts)))
        wall = time.monotonic() - t0
        tokens = sum(len(t) for t, _ in outs)
        drafted = sum(int(tm.get("draft_n") or 0) for _, tm in outs)
        accepted = sum(int(tm.get("draft_n_accepted") or 0) for _, tm in outs)
        return {"tokens_per_s": round(tokens / wall, 1) if wall else 0.0, "wall_s": round(wall, 1),
                "draft_acceptance": round(accepted / drafted, 2) if drafted else None}


def _greedy(probe, prompts: list[list[int]], n: int) -> dict:
    return {(i, s): probe.complete(p, s, n)[0] for i, p in enumerate(prompts) for s in (0.0, 1.0)}


def _lora_matters(tokens: dict) -> bool:
    """The adapter must change the output for at least one prompt (scale 0 vs scale 1)."""
    prompts = {i for i, _ in tokens}
    return any(tokens[(i, 0.0)] != tokens[(i, 1.0)] for i in prompts)


def _fits(probe, prof: dict, vram_fn) -> str | None:
    """Why this running config can't be used in training, or None. Training keeps ~1 GB of the trainer's
    CUDA state on the GPU, which the tuner's empty GPU doesn't have: require that much free (plus margin),
    and a single-stream speed that rules out a silent spill into system RAM."""
    need_mb = float(prof.get("vram_headroom_gb") or 0) * 1024
    v = vram_fn() if vram_fn else None
    if v and need_mb and v["free"] < need_mb:
        return f"too tight for training ({v['free'] / 1024:.1f} GB VRAM free, needs {need_mb / 1024:.1f})"
    floor = prof.get("min_decode_tps")
    if floor:
        tps = probe.single_speed()
        if tps is not None and tps < float(floor):
            return f"too slow ({tps} tok/s for one stream, floor {floor})"
    return None


def tune_rollouts(probe, base_args, prof: dict, draft_path: str | None, exact_prompts: list[list[int]],
                  speed_prompts: list[list[int]], n_exact: int = 48, n_speed: int = 128, log=print,
                  vram_fn=None) -> dict:
    """base_args(kv_type, slots) -> llama-server args without speculation. Returns the chosen settings.
    vram_fn() -> {"free": MiB, ...} or None (see adv.gpu.vram)."""
    slots0 = int(prof["parallel_slots"])
    kv_types = prof.get("kv_cache_types") or [None]
    spec = prof.get("speculative") or {}
    report = {"kv_results": {}, "spec_results": {}}

    # ---- stage 1: KV-cache type, plain decoding, full concurrency
    best = None  # (tokens_per_s, kv, reference greedy tokens)
    for kv in kv_types:
        name = kv or "default"
        try:
            probe.start(base_args(kv, slots0))
        except RuntimeError as e:
            report["kv_results"][name] = f"failed to start: {str(e)[:160]}"
            log(f"[tune] KV {name}: failed to start")
            continue
        try:
            why_not = _fits(probe, prof, vram_fn)
            if why_not:
                report["kv_results"][name] = why_not
                log(f"[tune] KV {name}: rejected, {why_not}")
                continue
            ref = _greedy(probe, exact_prompts, n_exact)
            if not _lora_matters(ref):
                report["kv_results"][name] = "LoRA had no effect"
                log(f"[tune] KV {name}: LoRA had no effect")
                continue
            tp = probe.throughput(speed_prompts, n_speed)
            report["kv_results"][name] = tp
            log(f"[tune] KV {name}: {tp['tokens_per_s']} tok/s with {len(speed_prompts)} parallel requests")
            if best is None or tp["tokens_per_s"] > best[0]:
                best = (tp["tokens_per_s"], kv, ref)
        except Exception as e:
            report["kv_results"][name] = f"error: {str(e)[:160]}"
        finally:
            probe.stop()
    if best is None:
        return {**report, "ok": False, "reason": "no KV-cache type worked (see kv_results)"}
    base_tps, kv, ref = best
    chosen = {"kv_cache": kv, "slots": slots0, "speculative": None, "tokens_per_s": base_tps}
    report["baseline_tokens_per_s"] = base_tps

    # ---- stage 2: speculative candidates with the chosen KV type
    min_agree, min_gain = float(spec.get("min_agreement", 0.95)), float(spec.get("min_gain", 1.05))
    for cand in spec.get("candidates") or []:
        if needs_drafter(cand["type"]) and not (draft_path and Path(draft_path).exists()):
            report["spec_results"][cand["type"]] = "drafter not downloaded"
            continue
        for slots in spec.get("slots") or [slots0]:
            key = f"{cand['type']} @ {slots} slots"
            try:
                probe.start(base_args(kv, int(slots)) + speculative_args(spec, cand, draft_path))
            except RuntimeError as e:
                report["spec_results"][key] = f"failed to start: {str(e)[:160]}"
                log(f"[tune] {key}: failed to start")
                break  # same type at fewer slots would fail the same way
            try:
                why_not = _fits(probe, prof, vram_fn)
                if why_not:
                    report["spec_results"][key] = why_not
                    log(f"[tune] {key}: rejected, {why_not}")
                    continue  # fewer slots use less memory: try them
                got = _greedy(probe, exact_prompts, n_exact)
                if not _lora_matters(got):
                    report["spec_results"][key] = "LoRA had no effect with speculation on"
                    log(f"[tune] {key}: rejected (LoRA ignored)")
                    break
                ok, why = agrees(ref, got, min_agree)
                if not ok:
                    report["spec_results"][key] = f"output changed ({why})"
                    log(f"[tune] {key}: rejected, output changed ({why})")
                    break
                tp = probe.throughput(speed_prompts, n_speed)
                report["spec_results"][key] = {**tp, "check": why}
                acc = "nothing drafted" if tp["draft_acceptance"] is None else f"acceptance {tp['draft_acceptance']}"
                log(f"[tune] {key}: {tp['tokens_per_s']} tok/s ({acc}; {why})")
                if tp["tokens_per_s"] >= base_tps * min_gain and tp["tokens_per_s"] > chosen["tokens_per_s"]:
                    chosen = {"kv_cache": kv, "slots": int(slots), "speculative": dict(cand),
                              "tokens_per_s": tp["tokens_per_s"]}
            except Exception as e:
                report["spec_results"][key] = f"error: {str(e)[:160]}"
            finally:
                probe.stop()
    chosen["speedup"] = round(chosen["tokens_per_s"] / base_tps, 2) if base_tps else None
    return {**report, **chosen, "ok": True}


def rollout_settings(prof: dict, state: dict, draft_path: str | None) -> dict:
    """What training should launch llama-server with: the self-test's choice, or the profile defaults."""
    tuned = (state or {}).get("rollout") or {}
    kv_types = prof.get("kv_cache_types") or [None]
    kv = tuned.get("kv_cache") if tuned.get("ok") else kv_types[0]
    slots = int(tuned.get("slots") or prof["parallel_slots"]) if tuned.get("ok") else int(prof["parallel_slots"])
    args = (["-ctk", kv, "-ctv", kv] if kv else [])
    choice, label, note = tuned.get("speculative") if tuned.get("ok") else None, "", ""
    if choice:
        if needs_drafter(choice["type"]) and not (draft_path and Path(draft_path).exists()):
            note = f"speculative decoding off: drafter {Path(draft_path or '?').name} missing"
            choice = None
        else:
            args += speculative_args(prof.get("speculative") or {}, choice, draft_path)
            label = choice["type"]
    elif prof.get("speculative") and not tuned.get("ok"):
        note = "speculative decoding off: not tuned yet (run setup again to run the self-test)"
    return {"kv_cache": kv, "slots": slots, "args": args, "spec_label": label, "note": note,
            "speedup": tuned.get("speedup") if tuned.get("ok") else None, "spec": choice}


def fallback_ladder(prof: dict, rs: dict) -> list[dict]:
    """Launch settings to step down through when llama-server starts too slow (GPU memory overflowing):
    each rung needs less VRAM than the one before. Rung 0 is what rollout_settings chose."""
    gl = prof.get("llama_gpu_layers", "auto")
    cur = {"label": "as tuned", "kv": rs["kv_cache"], "slots": int(rs["slots"]), "spec": rs.get("spec"),
           "gpu_layers": gl, "fit_on": False}
    rungs = [dict(cur)]

    def add(label, **change):
        cur.update(change, label=label)
        if any(all(r[k] == cur[k] for k in ("kv", "slots", "spec", "gpu_layers", "fit_on")) for r in rungs):
            return
        rungs.append(dict(cur))

    if cur["spec"]:
        add("speculative decoding off", spec=None)
    if cur["kv"] and cur["kv"] != "q4_0" and "q4_0" in (prof.get("kv_cache_types") or []):
        add("KV cache q4_0", kv="q4_0")
    for n in (6, 4):
        if cur["slots"] > n:
            add(f"{n} parallel slots", slots=n)
    if str(gl) not in ("auto", "0"):
        add("-ngl auto, --fit on (llama.cpp keeps what does not fit on the CPU)", gpu_layers="auto", fit_on=True)
    return rungs


def rung_launch(prof: dict, rung: dict, draft_path: str | None) -> dict:
    """LlamaCppEngine.reconfigure() arguments for one rung of fallback_ladder."""
    args = ["-ctk", rung["kv"], "-ctv", rung["kv"]] if rung["kv"] else []
    if rung["spec"]:
        args += speculative_args(prof.get("speculative") or {}, rung["spec"], draft_path)
    extra = [str(a) for a in prof.get("llama_extra_args") or []]
    if rung["fit_on"]:
        if "--fit" in extra:
            i = extra.index("--fit")
            extra = extra[:i] + extra[i + 2:]
        extra += ["--fit", "on"]
    return {"parallel": rung["slots"], "spec_args": args, "gpu_layers": rung["gpu_layers"], "extra_args": extra}
