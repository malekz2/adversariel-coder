"""Load config.yaml, resolve the hardware profile and every tool/model path."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # the project folder (contains config.yaml)
IS_WINDOWS = os.name == "nt"


def load(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text(encoding="utf-8"))
    apply_overrides(cfg, overrides or {})
    return cfg


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Set dotted keys, e.g. {"grpo.puzzles_per_step": 8}, in place."""
    for dotted, value in overrides.items():
        node = cfg
        *parents, last = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return cfg


def state_dir() -> Path:
    d = ROOT / "state"
    d.mkdir(exist_ok=True)
    return d


def read_state(name: str) -> dict:
    p = state_dir() / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def resolve_profile(cfg: dict, hw: dict | None = None) -> str:
    if cfg.get("profile", "auto") != "auto":
        return cfg["profile"]
    hw = hw or read_state("hardware.json")
    if not hw:
        from .hardware import detect, pick_profile
        hw = detect()
        return pick_profile(hw)
    return hw.get("profile") or "cpu"


def models_dir() -> Path:
    p = read_state("paths.json").get("models_dir")
    return Path(p) if p else ROOT / "models"


def repo_dir(repo: str) -> Path:
    return models_dir() / repo.replace("/", "__")


def _find(base: Path, names: list[str]) -> str | None:
    if not base.exists():
        return None
    for n in names:
        hits = sorted(base.rglob(n))
        if hits:
            return str(hits[0])
    return None


def _pi_cli(pkg: Path) -> str | None:
    """Pi's Node entry point, from package.json "bin" (dist/ also holds a Bun build we must not use)."""
    try:
        rel = json.loads((pkg / "package.json").read_text(encoding="utf-8"))["bin"]["pi"]
    except (OSError, KeyError, ValueError):
        return None
    p = pkg / rel
    return str(p) if p.exists() else None


def tools() -> dict:
    t = Path(os.environ.get("ADV_TOOLS_DIR", ROOT / "tools"))
    exe = ".exe" if IS_WINDOWS else ""
    return {
        "llama_server": _find(t / "llama", [f"llama-server{exe}"]),
        "convert_lora": _find(t / "llama.cpp-src", ["convert_lora_to_gguf.py"]),
        "gguf_py": _find(t / "llama.cpp-src", ["gguf-py"]),
        "node": _find(t / "node", ["node.exe"] if IS_WINDOWS else ["node"]),
        "pi_cli": _pi_cli(t / "pi" / "node_modules" / "@earendil-works" / "pi-coding-agent"),
        "bash": _find(t / "git", ["bash.exe"]) if IS_WINDOWS else None,
    }
