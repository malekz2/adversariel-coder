"""Run model-written Python in a separate process with a time limit.

This is NOT a security sandbox. Code runs as a normal process owned by you, in a
throwaway temp directory, with a timeout and (on Linux/Mac) a memory cap. On a
no-admin Windows lab PC there is no stronger isolation available. The maker's
code is additionally checked against an import allow-list in puzzles.py.

The child process gets a JSON spec on stdin:
    {"code": "...", "function_name": "f", "inputs": [[arg1, arg2], ...]}
and prints one JSON line per input as it goes, so if it hangs on test 7 we still
know tests 0-6 finished.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

IS_WINDOWS = os.name == "nt"

RUNNER_SOURCE = r'''
import json, sys
sys.setrecursionlimit(20000)
spec = json.loads(sys.stdin.read())
MAX_OUT = spec.get("max_output_chars", 20000)
ns = {"__name__": "solution"}
try:
    exec(compile(spec["code"], "solution.py", "exec"), ns)
    fn = ns[spec["function_name"]]
except BaseException as e:
    print(json.dumps({"event": "load_error", "error": (type(e).__name__ + ": " + str(e))[:400]}), flush=True)
    sys.exit(0)
for i, args in enumerate(spec["inputs"]):
    try:
        args = json.loads(json.dumps(args))  # fresh copy so mutation cannot leak between tests
        out = fn(*args)
        try:
            text = json.dumps(out)
        except (TypeError, ValueError):
            if isinstance(out, (set, frozenset)):
                text = json.dumps(sorted(out, key=repr))
            else:
                raise TypeError("output is not JSON-serialisable: " + type(out).__name__)
        if len(text) > MAX_OUT:
            raise ValueError("output too large")
        print('{"event": "result", "i": %d, "ok": true, "output": %s}' % (i, text), flush=True)
    except BaseException as e:
        print(json.dumps({"event": "result", "i": i, "ok": False,
                          "error": (type(e).__name__ + ": " + str(e))[:300]}), flush=True)
print(json.dumps({"event": "done"}), flush=True)
'''


@dataclass
class TestOutcome:
    ok: bool
    output: object = None
    error: str | None = None


@dataclass
class RunResult:
    outcomes: list  # TestOutcome per input, or None if never reached (timeout)
    load_error: str | None = None
    timed_out: bool = False
    seconds: float = 0.0
    stderr: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        return (self.load_error is None and not self.timed_out
                and all(o is not None and o.ok for o in self.outcomes))


def _child_env(hash_seed: int) -> dict:
    keep = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONHASHSEED"] = str(hash_seed)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _limit_memory(max_mb: int):
    def apply():  # runs in the child before exec (POSIX only)
        try:
            import resource
            limit = max_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass  # macOS ignores RLIMIT_AS; fine
    return apply


def _kill_tree(proc: subprocess.Popen):
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, 9)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_function(code: str, function_name: str, inputs: list, timeout_s: float = 10.0,
                 python: str | None = None, hash_seed: int = 0, max_memory_mb: int = 2048,
                 max_output_chars: int = 20000) -> RunResult:
    """Execute `function_name(*args)` for every args-list in `inputs`."""
    python = python or sys.executable
    spec = json.dumps({"code": code, "function_name": function_name, "inputs": inputs,
                       "max_output_chars": max_output_chars})
    with tempfile.TemporaryDirectory(prefix="adv_sbx_") as tmp:
        runner = os.path.join(tmp, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(RUNNER_SOURCE)
        kwargs = dict(cwd=tmp, env=_child_env(hash_seed), stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _limit_memory(max_memory_mb)
        t0 = time.monotonic()
        proc = subprocess.Popen([python, "-I", runner], **kwargs)
        timed_out = False
        try:
            out, err = proc.communicate(spec.encode("utf-8"), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            out, err = proc.communicate()
        seconds = time.monotonic() - t0

    outcomes: list = [None] * len(inputs)
    load_error = None
    done = False
    for line in out.decode("utf-8", "replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # the solution printed something itself; ignore
        if not isinstance(ev, dict):
            continue
        kind = ev.get("event")
        if kind == "load_error":
            load_error = ev.get("error", "load error")
        elif kind == "result" and isinstance(ev.get("i"), int) and 0 <= ev["i"] < len(inputs):
            outcomes[ev["i"]] = TestOutcome(ok=bool(ev.get("ok")), output=ev.get("output"),
                                            error=ev.get("error"))
        elif kind == "done":
            done = True
    if not done and not timed_out and load_error is None:
        load_error = "process exited early (crash or memory limit)"
    return RunResult(outcomes=outcomes, load_error=load_error, timed_out=timed_out,
                     seconds=seconds, stderr=err.decode("utf-8", "replace")[-2000:])


def outputs_equal(a, b, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    """Structural equality that forgives tuple-vs-list and tiny float error."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(outputs_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(outputs_equal(a[k], b[k], rel_tol, abs_tol) for k in a)
    return a == b


def score_against(result: RunResult, expected: list) -> float:
    """Fraction of tests whose output matches `expected`."""
    if not expected:
        return 0.0
    hits = 0
    for outcome, want in zip(result.outcomes, expected):
        if outcome is not None and outcome.ok and outputs_equal(outcome.output, want):
            hits += 1
    return hits / len(expected)
