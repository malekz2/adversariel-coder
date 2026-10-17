"""Stop cleanly before the computer runs out of memory or disk.

Why this exists: on a 16 GB Mac, MLX's default memory limit is 1.5x the GPU's recommended
working set (~19 GB, more than the machine has), and that limit is only a guideline anyway.
Once memory is gone macOS swaps to disk, and with little free disk the whole system freezes.
The guard watches memory, swap growth and free disk, and raises MemoryDanger early, so the
trainer saves a checkpoint and stops instead of taking the laptop down with it.

Memory on macOS: psutil's "available" counts cached file pages (e.g. model files just read
from disk) as used, so it can say 6 GB free while macOS itself reports 77 % free. On macOS
the guard therefore uses the kernel's own numbers: the memory-pressure level
(kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical) and the free
percentage (kern.memorystatus_level, the figure `memory_pressure` prints). Elsewhere it uses
psutil's available RAM.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import sys
import time
from dataclasses import dataclass

import psutil

GB = 1024 ** 3


class MemoryDanger(RuntimeError):
    """Raised when continuing would risk freezing the machine."""


PRESSURE_NAMES = {1: "normal", 2: "warning", 4: "critical"}


@dataclass
class Limits:
    min_free_ram_gb: float = 1.5      # non-macOS: stop if psutil's available RAM drops below this
    min_free_percent: float = 15.0    # macOS: stop at "warning" pressure once free memory is below this
    max_swap_growth_gb: float = 2.0   # stop if swap grew by more than this since the run started
    min_free_disk_gb: float = 10.0    # macOS needs free disk to swap into


_LIBC = None


def _sysctl_int(name: str) -> int | None:
    """Read an integer sysctl on macOS without spawning a process (no admin rights needed)."""
    global _LIBC
    if sys.platform != "darwin":
        return None
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
        val, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if _LIBC.sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0) != 0:
            return None
        return val.value
    except Exception:
        return None


def macos_pressure() -> tuple[int | None, int | None]:
    """(pressure level 1/2/4, free percentage) on macOS; (None, None) elsewhere."""
    return _sysctl_int("kern.memorystatus_vm_pressure_level"), _sysctl_int("kern.memorystatus_level")


def free_memory_gb() -> float:
    """Memory an app could get right now. On macOS that includes cache the kernel would drop."""
    _, free_pct = macos_pressure()
    if free_pct is not None:
        return free_pct / 100 * psutil.virtual_memory().total / GB
    return psutil.virtual_memory().available / GB


def check_numbers(free_ram_gb: float, swap_growth_gb: float, free_disk_gb: float, lim: Limits,
                  pressure: int | None = None, free_pct: int | None = None) -> str | None:
    """Pure decision rule (unit-tested): a reason to stop, or None. With macOS pressure numbers
    available they replace the free-RAM check (which counts reclaimable file cache as used)."""
    if pressure is not None:
        name = PRESSURE_NAMES.get(pressure, str(pressure))
        if pressure >= 4:
            return f"macOS memory pressure is {name} ({free_pct}% free)"
        if pressure >= 2 and free_pct is not None and free_pct < lim.min_free_percent:
            return (f"macOS memory pressure is {name} with only {free_pct}% free "
                    f"(limit {lim.min_free_percent:g}%)")
    elif free_ram_gb < lim.min_free_ram_gb:
        return f"only {free_ram_gb:.1f} GB of RAM left (limit {lim.min_free_ram_gb:g} GB)"
    if swap_growth_gb > lim.max_swap_growth_gb:
        return f"swap grew by {swap_growth_gb:.1f} GB (limit {lim.max_swap_growth_gb:g} GB)"
    if free_disk_gb < lim.min_free_disk_gb:
        return f"only {free_disk_gb:.1f} GB of free disk (limit {lim.min_free_disk_gb:g} GB)"
    return None


class MemoryGuard:
    def __init__(self, limits: Limits, disk_path: str = "/", min_interval_s: float = 0.5):
        self.limits, self.disk_path, self.min_interval_s = limits, disk_path, min_interval_s
        self.swap_start = psutil.swap_memory().used
        self._last = 0.0
        self.tripped: str | None = None   # once tripped, every later check fails (also from other threads)

    def snapshot(self) -> dict:
        pressure, free_pct = macos_pressure()
        return {"free_ram_gb": psutil.virtual_memory().available / GB,
                "swap_growth_gb": max(0, psutil.swap_memory().used - self.swap_start) / GB,
                "free_disk_gb": shutil.disk_usage(self.disk_path).free / GB,
                "pressure": pressure, "free_pct": free_pct}

    def check(self, where: str, force: bool = False):
        if self.tripped:
            raise MemoryDanger(self.tripped)
        now = time.monotonic()
        if not force and now - self._last < self.min_interval_s:  # cheap enough, but don't spam
            return
        self._last = now
        s = self.snapshot()
        reason = check_numbers(s["free_ram_gb"], s["swap_growth_gb"], s["free_disk_gb"], self.limits,
                               s["pressure"], s["free_pct"])
        if reason:
            self.tripped = f"{reason} during {where}"
            raise MemoryDanger(self.tripped)


def limit_mlx_memory(fraction: float = 0.6, cache_gb: float = 1.0) -> dict:
    """Cap MLX below physical RAM (its default is ~1.5x the GPU working set, i.e. more than the Mac has)."""
    import mlx.core as mx
    info = mx.device_info()
    total = int(info["memory_size"])
    limit = int(min(fraction * total, info.get("max_recommended_working_set_size", total)))
    mx.set_memory_limit(limit)
    mx.set_cache_limit(int(cache_gb * GB))
    return {"total_gb": total / GB, "mlx_limit_gb": limit / GB, "cache_limit_gb": cache_gb}
