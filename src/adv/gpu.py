"""NVIDIA GPU checks around llama-server starts (no-ops where there is no NVIDIA GPU, e.g. the Mac).

Why: on Windows the NVIDIA driver does not fail when VRAM runs out. It quietly moves the overflow to system
RAM ("Shared GPU memory" in Task Manager), and a 27B that should generate ~40 tok/s drops to ~1 tok/s. With
`--fit off` llama.cpp no longer protects us from that, so we check free VRAM ourselves.
"""
from __future__ import annotations

import os
import shutil
import subprocess


def vram() -> dict | None:
    """{"total": MiB, "used": MiB, "free": MiB} of GPU 0 via nvidia-smi, or None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.total,memory.used,memory.free",
                              "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL).stdout
        return parse_vram(out)
    except (OSError, subprocess.SubprocessError):
        return None


def parse_vram(out: str) -> dict | None:
    try:
        total, used, free = (int(float(x)) for x in out.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return {"total": total, "used": used, "free": free}


def fmt(v: dict | None) -> str:
    return "n/a" if not v else f"{v['free'] / 1024:.1f} of {v['total'] / 1024:.1f} GB free"


def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(p))


def is_our_server(exe: str | None, cmdline: list[str], server_exe: str, ports: list[int]) -> bool:
    """A llama-server started from OUR tools folder on one of OUR ports (never anything else)."""
    if not exe or _norm(exe) != _norm(server_exe):
        return False
    for flag in ("--port",):
        if flag in cmdline:
            i = cmdline.index(flag)
            if i + 1 < len(cmdline) and cmdline[i + 1] in {str(p) for p in ports}:
                return True
    return False


def kill_stale_servers(server_exe: str, ports: list[int], log=print) -> int:
    """Stop llama-servers left over from a crashed or closed run: they keep ~20 GB of VRAM."""
    try:
        import psutil
    except ImportError:
        return 0
    n = 0
    for p in psutil.process_iter(["pid", "exe", "cmdline"]):
        try:
            if p.pid == os.getpid() or not is_our_server(p.info.get("exe"), p.info.get("cmdline") or [],
                                                         server_exe, ports):
                continue
            log(f"stopping a leftover llama-server (pid {p.pid}) from an earlier run: it was holding GPU memory")
            p.terminate()
            try:
                p.wait(timeout=20)
            except psutil.TimeoutExpired:
                p.kill()
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return n
