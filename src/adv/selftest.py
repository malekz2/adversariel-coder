"""Check, on THIS machine, that llama.cpp can serve our LoRA adapters (and d3-flash).

    python -m adv.selftest                 # uses the configured profile's models
    python -m adv.selftest --base DIR --gguf FILE   # explicit paths

It builds a small random LoRA straight from the checkpoint's tensor shapes (no need to load
27B into Python), converts it with llama.cpp's converter, starts llama-server with it and
checks that scale 0 and scale 1 give different tokens. The result goes to state/engine.json;
training uses llama.cpp only if this passed, otherwise the slower but always-correct HF engine.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests

from . import config as C
from .engines.llamacpp import convert_adapter_to_gguf
from .policy import base_checkpoint_prefix

TARGETS = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                   "gate_proj", "up_proj", "down_proj"],
           "mlp_only": ["gate_proj", "up_proj", "down_proj"]}


def _shapes(base_dir: Path) -> dict[str, tuple]:
    from safetensors import safe_open
    out = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.endswith(".weight") and "layers." in k and "mtp" not in k and "visual" not in k:
                    out[k] = tuple(s.get_slice(k).get_shape())
    return out


def make_random_adapter(base_dir: Path, out_dir: Path, targets: list[str], r: int = 4) -> Path:
    import torch
    from safetensors.torch import save_file
    prefix = base_checkpoint_prefix(str(base_dir))
    sd = {}
    torch.manual_seed(0)
    for name, shape in _shapes(base_dir).items():
        mod = name[:-len(".weight")]
        if not name.startswith(prefix) or mod.split(".")[-1] not in targets or len(shape) != 2:
            continue
        d_out, d_in = shape
        sd[f"base_model.model.{mod}.lora_A.weight"] = (torch.randn(r, d_in) * 0.02).half()
        sd[f"base_model.model.{mod}.lora_B.weight"] = (torch.randn(d_out, r) * 0.5).half()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": r, "lora_dropout": 0.0,
        "target_modules": targets, "bias": "none", "base_model_name_or_path": str(base_dir)}))
    return out_dir


def _serve_and_compare(server: str, gguf: str, lora_gguf: str, port: int, gpu_layers: str) -> tuple[bool, str]:
    import subprocess
    log = Path(tempfile.gettempdir()) / "adv_selftest_server.log"
    proc = subprocess.Popen([server, "-m", gguf, "--lora", lora_gguf, "--lora-init-without-apply", "--host",
                             "127.0.0.1", "--port", str(port), "-c", "2048", "-np", "1", "-ngl", str(gpu_layers),
                             "--no-webui", "--log-file", str(log)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 900:
            if proc.poll() is not None:
                return False, f"llama-server exited ({proc.returncode}); see {log}"
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            return False, "llama-server did not start"
        outs = []
        for scale in (0.0, 1.0):
            r = requests.post(f"http://127.0.0.1:{port}/completion", timeout=600, json={
                "prompt": [9707, 11, 1879, 0, 330], "n_predict": 6, "temperature": 0, "return_tokens": True,
                "lora": [{"id": 0, "scale": scale}]})
            outs.append(r.json().get("tokens"))
        if outs[0] == outs[1]:
            return False, f"LoRA had no effect (tokens {outs[0]})"
        return True, f"scale0 {outs[0]} vs scale1 {outs[1]}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def check_d3(server: str, gguf: str, port: int) -> tuple[bool, str]:
    import subprocess
    proc = subprocess.Popen([server, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-c", "4096",
                             "-b", "4096", "-ub", "4096", "-np", "1", "-ngl", "0", "--no-webui"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 900:
            if proc.poll() is not None:
                return False, f"llama-server exited ({proc.returncode})"
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        t1 = time.monotonic()
        r = requests.post(f"http://127.0.0.1:{port}/v1/systemone", timeout=900, json={
            "state": {"solver_pass_rate": [0.9, 0.95, 0.97, 1.0]},
            "questions": {"easy": {"type": "noul", "instructions": "Are the puzzles too easy for the solver?"}}})
        r.raise_for_status()
        return True, f"P(too easy) = {r.json()['answers']['easy']['noul']:.2f} in {time.monotonic() - t1:.1f}s"
    except Exception as e:
        return False, str(e)[:300]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--gguf")
    ap.add_argument("--profile")
    ap.add_argument("--skip-d3", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load()
    profile = a.profile or C.resolve_profile(cfg)
    prof = cfg["profiles"][profile]
    base = Path(a.base) if a.base else C.repo_dir(prof["train_repo"])
    gguf = a.gguf or str(C.repo_dir(prof["gguf_repo"]) / prof["gguf_file"])
    t = C.tools()
    result = {"engine": "hf", "profile": profile}
    if not (t["llama_server"] and t["convert_lora"]):
        result["reason"] = "llama-server or convert_lora_to_gguf.py missing"
    else:
        work = Path(tempfile.mkdtemp(prefix="adv_selftest_"))
        try:
            for name in ("all", "mlp_only"):
                print(f"[selftest] LoRA targets '{name}': building random adapter ...")
                ad = make_random_adapter(base, work / name, TARGETS[name])
                try:
                    out = convert_adapter_to_gguf(str(ad), str(base), str(work / f"{name}.gguf"), t["convert_lora"],
                                                  sys.executable, t["gguf_py"])
                except Exception as e:
                    print(f"[selftest]   conversion failed: {str(e)[-400:]}")
                    result[f"reason_{name}"] = "conversion failed"
                    continue
                ok, msg = _serve_and_compare(t["llama_server"], gguf, out, cfg["ports"]["engine"], prof["llama_gpu_layers"])
                print(f"[selftest]   {'OK' if ok else 'FAILED'}: {msg}")
                if ok:
                    result.update(engine="llamacpp", lora_targets=name, detail=msg)
                    break
                result[f"reason_{name}"] = msg
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if not a.skip_d3:
        d3 = C.repo_dir(cfg["decision"]["d3_repo"]) / cfg["decision"]["d3_file"]
        if t["llama_server"] and d3.exists():
            print("[selftest] d3-flash via /v1/systemone ...")
            ok, msg = check_d3(t["llama_server"], str(d3), cfg["ports"]["d3"])
            print(f"[selftest]   {'OK' if ok else 'FAILED'}: {msg}")
            result["d3"] = "ok" if ok else f"failed: {msg}"
        else:
            result["d3"] = "missing"
    (C.state_dir() / "engine.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["engine"] == "llamacpp" and result.get("lora_targets") == "mlp_only":
        print("NOTE: set lora.target_modules: mlp_only in config.yaml (llama.cpp rejected the other targets)")


if __name__ == "__main__":
    main()
