"""Download the models a profile needs.   python -m adv.download [--profile cpu] [--models-dir D]"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import config as C

TRAIN_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py"]  # .py: custom model code


def plan(cfg: dict, profile: str, with_d3: bool = True) -> list[dict]:
    prof, d = cfg["profiles"][profile], cfg["decision"]
    items = [{"repo": prof["train_repo"], "file": None, "why": "training weights (safetensors)"},
             {"repo": prof["gguf_repo"], "file": prof["gguf_file"], "why": "rollout model (llama.cpp)"}]
    if with_d3:
        items.append({"repo": d["d3_repo"], "file": d["d3_file"], "why": "referee d3-flash"})
    return items


def sizes(items: list[dict]) -> list[float]:
    from huggingface_hub import HfApi
    api, out = HfApi(), []
    for it in items:
        info = api.model_info(it["repo"], files_metadata=True)
        if it["file"]:
            out.append(sum((s.size or 0) for s in info.siblings if s.rfilename == it["file"]) / 1e9)
        else:
            out.append(sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile")
    ap.add_argument("--models-dir")
    ap.add_argument("--no-d3", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load()
    profile = a.profile or C.resolve_profile(cfg)
    if a.models_dir:
        paths = C.read_state("paths.json")
        paths["models_dir"] = str(Path(a.models_dir).resolve())
        (C.state_dir() / "paths.json").write_text(json.dumps(paths, indent=2))
    mdir = C.models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    items = plan(cfg, profile, not a.no_d3)
    gb = sizes(items)
    free = shutil.disk_usage(mdir).free / 1e9
    print(f"profile {profile}: downloading to {mdir}")
    for it, s in zip(items, gb):
        print(f"  {s:6.1f} GB  {it['repo']}{'/' + it['file'] if it['file'] else ''}   ({it['why']})")
    need = sum(g for it, g in zip(items, gb) if not _present(it))
    print(f"  need {need:.1f} GB, free {free:.1f} GB")
    if need > free - 1:
        sys.exit("Not enough disk space. Choose another folder with --models-dir (e.g. on the PC's own drive).")
    from huggingface_hub import hf_hub_download, snapshot_download
    for it in items:
        dest = C.repo_dir(it["repo"])
        if _present(it):
            print(f"  already present: {dest}")
            continue
        if it["file"]:
            hf_hub_download(it["repo"], it["file"], local_dir=str(dest))
        else:
            snapshot_download(it["repo"], local_dir=str(dest), allow_patterns=TRAIN_PATTERNS)
        print(f"  done: {dest}")


def _present(it) -> bool:
    d = C.repo_dir(it["repo"])
    if it["file"]:
        return (d / it["file"]).exists()
    return (d / "config.json").exists() and any(d.glob("*.safetensors"))


if __name__ == "__main__":
    main()
