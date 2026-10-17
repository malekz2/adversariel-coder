"""Detect the machine and pick a hardware profile.  Run:  python -m adv.hardware"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys


def _nvidia_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append({"name": parts[0], "vram_gb": round(float(parts[1]) / 1024, 1)})
            except ValueError:
                pass
    return gpus


def detect() -> dict:
    import psutil
    info = {
        "os": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False) or os.cpu_count(),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpus": _nvidia_smi(), "torch_cuda": False,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_cuda"] = True
            props = torch.cuda.get_device_properties(0)
            info["cuda_device"] = {"name": props.name, "vram_gb": round(props.total_memory / 1e9, 1)}
    except Exception as e:  # torch missing or broken
        info["torch_error"] = str(e)
    return info


def pick_profile(info: dict) -> str:
    vram = (info.get("cuda_device") or {}).get("vram_gb", 0) if info.get("torch_cuda") else 0
    if vram >= 23.5:
        return "cuda_large"
    gpu = ((info.get("cuda_device") or {}).get("name") or "").lower()
    if 10 <= vram < 14 or "2080 ti" in gpu:   # e.g. Purdue LWSN B131's RTX 2080 Ti (11 GB)
        return "lawson_b131"
    if vram >= 7.5:
        return "cuda_small"
    ram = info.get("ram_gb", 0)
    if ram >= 28:            # K2-Horizon-7B in bf16: 18 GB weights + training + rollout engine
        return "cpu"
    return "cpu_small" if ram >= 14 else "cpu_tiny"


def main():
    info = detect()
    info["profile"] = pick_profile(info)
    print(json.dumps(info, indent=2))
    if len(sys.argv) > 2 and sys.argv[1] == "--write":
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
