"""Puzzle format, Maker/Solver prompts, validation and novelty.

The Maker writes a puzzle as JSON:

    {"title": "...", "statement": "...", "function_name": "f",
     "signature": "def f(nums: list[int], k: int) -> int:",
     "reference_solution": "def f(nums, k):\n    ...",
     "test_inputs": [[[1, 2, 3], 2], [[5], 1], ...],
     "difficulty": 3}

The Maker only supplies test *inputs*. Expected outputs are computed by running
the Maker's own reference solution (the Absolute-Zero trick). That way the tests
can never contradict the reference, and the Maker cannot claim a puzzle is valid
without having solved it.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .sandbox import RunResult, outputs_equal, run_function

TOPICS = [
    "string manipulation", "arrays and prefix sums", "hash maps and counting", "two pointers",
    "sliding window", "sorting with custom keys", "intervals", "stacks", "queues and BFS on grids",
    "graphs given as edge lists", "dynamic programming on sequences", "dynamic programming on grids",
    "greedy choices", "number theory", "combinatorics with small inputs", "bit manipulation",
    "matrix transformations", "simulation of a simple process", "parsing a tiny expression language",
    "recursion and backtracking with small inputs", "binary search on the answer", "heaps",
]

DIFFICULTY_TEXT = {
    1: "very easy: a direct loop or a single built-in, one obvious edge case",
    2: "easy: one simple idea, a couple of edge cases",
    3: "medium: needs one non-obvious idea (e.g. a hash map, sorting first, or a small DP)",
    4: "hard: combines two ideas or needs careful edge-case handling",
    5: "very hard: a clever algorithm a strong programmer would need to think about",
}

ALLOWED_IMPORTS = {
    "math", "itertools", "collections", "functools", "heapq", "bisect", "re", "string",
    "typing", "dataclasses", "fractions", "decimal", "statistics", "operator", "copy",
}
BANNED_PATTERNS = [r"\bopen\s*\(", r"__import__", r"\beval\s*\(", r"\bexec\s*\(", r"\binput\s*\(",
                   r"\bcompile\s*\(", r"\bglobals\s*\(", r"\bbreakpoint\s*\(", r"\bexit\s*\(",
                   r"\bquit\s*\("]

MAKER_SYSTEM = """You are the PUZZLE MAKER in a training game. You invent self-contained Python \
coding puzzles for a SOLVER model. You score highest when the solver succeeds about half of the \
time: puzzles that are too easy or impossible score low, and broken puzzles score worst.

Rules:
- The puzzle is one pure Python function. Inputs and outputs are JSON values (int, float, str, \
bool, list, dict with string keys, None). No I/O, no randomness, no global state.
- The statement must be unambiguous and fully specify the expected output, including edge cases \
and tie-breaking. Do not reveal the algorithm or include the solution code in the statement.
- reference_solution is your correct solution. Only these imports are allowed: \
math, itertools, collections, functools, heapq, bisect, re, string, typing, fractions, decimal, \
statistics, operator, copy.
- test_inputs is a list of 8 to 20 argument lists (one list of positional arguments per test). \
Cover edge cases. Keep inputs small enough that a correct solution runs instantly.
- Expected outputs are computed by running your reference_solution, so do not write them.

Reply with your reasoning, then exactly one JSON object in a ```json code block with keys: \
title, statement, function_name, signature, reference_solution, test_inputs, difficulty (1-5)."""

SOLVER_SYSTEM_SINGLE = """You are the SOLVER. Solve the Python coding puzzle. Think it through, \
then reply with the complete function in one ```python code block. Do not print anything; just \
define the function."""

PI_SOLVER_PROMPT = """Solve the coding puzzle described in PROBLEM.md.

Write your solution in solution.py (keep the function name and signature). You can run \
`python run_tests.py` to check your code against the visible examples. The final grade uses \
additional hidden tests, so handle edge cases. Stop once you are confident the solution is \
correct."""


@dataclass
class Puzzle:
    title: str
    statement: str
    function_name: str
    signature: str
    reference_solution: str
    test_inputs: list
    difficulty: int = 3
    expected: list | None = None  # filled in by validate()
    source: str = "maker"         # "maker" or "seed"
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            h = hashlib.sha1((self.title + self.statement).encode("utf-8")).hexdigest()[:10]
            self.uid = h

    def visible(self, k: int) -> list[tuple[list, object]]:
        assert self.expected is not None
        return list(zip(self.test_inputs[:k], self.expected[:k]))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Puzzle":
        fields_ = {k: d[k] for k in Puzzle.__dataclass_fields__ if k in d}
        return Puzzle(**fields_)


@dataclass
class Validation:
    valid: bool
    reason: str = "ok"
    seconds: float = 0.0


# ---------------------------------------------------------------- parsing

def _extract_json_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, flags=re.S)
    for block in reversed(blocks):
        if block.strip().startswith("{"):
            return block
    # fallback: last balanced {...} in the text
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        end = text.rfind("}", 0, end)
    return None


def parse_maker_output(text: str) -> tuple[Puzzle | None, str]:
    raw = _extract_json_block(text)
    if raw is None:
        return None, "no_json"
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(d, dict):
        return None, "bad_json"
    need = {"title": str, "statement": str, "function_name": str, "signature": str,
            "reference_solution": str, "test_inputs": list}
    for key, typ in need.items():
        if not isinstance(d.get(key), typ):
            return None, f"missing_{key}"
    try:
        difficulty = int(d.get("difficulty", 3))
    except (TypeError, ValueError):
        difficulty = 3
    tests = d["test_inputs"]
    if not all(isinstance(t, list) for t in tests):
        # allow a single-argument shorthand: [1, 2, 3] -> [[1], [2], [3]]
        tests = [t if isinstance(t, list) else [t] for t in tests]
    return Puzzle(title=d["title"].strip(), statement=d["statement"].strip(),
                  function_name=d["function_name"].strip(), signature=d["signature"].strip(),
                  reference_solution=d["reference_solution"], test_inputs=tests,
                  difficulty=max(1, min(5, difficulty))), "ok"


def extract_python_code(text: str, function_name: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    for block in reversed(blocks):
        if re.search(rf"def\s+{re.escape(function_name)}\s*\(", block):
            return block
    return None


# ---------------------------------------------------------------- validation

def code_hygiene_error(code: str) -> str | None:
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", code, flags=re.M):
        mods = (m.group(1) or m.group(2) or "").replace(" ", "").split(",")
        for mod in mods:
            root = mod.split(".")[0]
            if root and root not in ALLOWED_IMPORTS:
                return f"disallowed_import:{root}"
    for pat in BANNED_PATTERNS:
        if re.search(pat, code):
            return "banned_call"
    return None


def validate(p: Puzzle, timeout_s: float = 10.0, python: str | None = None,
             min_tests: int = 5, max_tests: int = 40) -> Validation:
    if not re.fullmatch(r"[A-Za-z_]\w*", p.function_name):
        return Validation(False, "bad_function_name")
    if not (40 <= len(p.statement) <= 4000):
        return Validation(False, "statement_length")
    if len(p.reference_solution) > 6000:
        return Validation(False, "solution_too_long")
    if not re.search(rf"def\s+{re.escape(p.function_name)}\s*\(", p.reference_solution):
        return Validation(False, "function_not_defined")
    if not (min_tests <= len(p.test_inputs) <= max_tests):
        return Validation(False, "test_count")
    if len(json.dumps(p.test_inputs)) > 50_000:
        return Validation(False, "tests_too_large")
    err = code_hygiene_error(p.reference_solution)
    if err:
        return Validation(False, err)
    # leaking the solution into the statement makes the puzzle trivial
    for line in p.reference_solution.splitlines():
        line = line.strip()
        if len(line) >= 25 and not line.startswith("def ") and line in p.statement:
            return Validation(False, "solution_leaked")

    # run twice with different hash seeds: catches set-ordering and other nondeterminism
    r1 = run_function(p.reference_solution, p.function_name, p.test_inputs, timeout_s, python, 0)
    if not r1.all_ok:
        return Validation(False, _failure_reason(r1), r1.seconds)
    r2 = run_function(p.reference_solution, p.function_name, p.test_inputs, timeout_s, python, 1)
    if not r2.all_ok:
        return Validation(False, _failure_reason(r2), r1.seconds + r2.seconds)
    out1 = [o.output for o in r1.outcomes]
    out2 = [o.output for o in r2.outcomes]
    if not all(outputs_equal(a, b) for a, b in zip(out1, out2)):
        return Validation(False, "nondeterministic", r1.seconds + r2.seconds)
    if len({json.dumps(o, sort_keys=True) for o in out1}) < 2:
        return Validation(False, "constant_output", r1.seconds + r2.seconds)
    p.expected = out1
    return Validation(True, "ok", r1.seconds + r2.seconds)


def _failure_reason(r: RunResult) -> str:
    if r.timed_out:
        return "reference_timeout"
    if r.load_error:
        return "reference_load_error"
    return "reference_raised"


# ---------------------------------------------------------------- prompts

def _format_args(args: list) -> str:
    return ", ".join(repr(a) for a in args)


def problem_markdown(p: Puzzle, visible_k: int) -> str:
    lines = [f"# {p.title}", "", p.statement, "", "## Signature", "", "```python", p.signature,
             "```", "", "## Examples", ""]
    for args, out in p.visible(visible_k):
        lines.append(f"- `{p.function_name}({_format_args(args)})` returns `{out!r}`")
    return "\n".join(lines) + "\n"


def seed_example_text(p: Puzzle) -> str:
    d = {"title": p.title, "statement": p.statement, "function_name": p.function_name,
         "signature": p.signature, "reference_solution": p.reference_solution,
         "test_inputs": p.test_inputs[:6], "difficulty": p.difficulty}
    return "```json\n" + json.dumps(d, indent=1) + "\n```"


def maker_messages(topic: str, difficulty: int, examples: list[Puzzle], avoid_titles: list[str]) -> list[dict]:
    shots = "\n\n".join(seed_example_text(e) for e in examples)
    avoid = "; ".join(avoid_titles[-12:]) or "(none yet)"
    user = (f"Here are example puzzles in the required format (only the first few tests shown):\n\n"
            f"{shots}\n\nNow invent a NEW puzzle.\nTopic: {topic}\n"
            f"Target difficulty: {difficulty}/5 ({DIFFICULTY_TEXT[difficulty]})\n"
            f"Recent puzzle titles to avoid repeating: {avoid}")
    return [{"role": "system", "content": MAKER_SYSTEM}, {"role": "user", "content": user}]


def solver_messages_single(p: Puzzle, visible_k: int) -> list[dict]:
    return [{"role": "system", "content": SOLVER_SYSTEM_SINGLE},
            {"role": "user", "content": problem_markdown(p, visible_k)}]


RUN_TESTS_TEMPLATE = '''"""Visible example tests. Run: python run_tests.py"""
import json, math, traceback
from solution import {fn}

CASES = json.loads({cases!r})

def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

passed = 0
for args, want in CASES:
    try:
        got = {fn}(*args)
    except Exception:
        print("ERROR for input", args)
        traceback.print_exc()
        continue
    if same(got, want):
        passed += 1
    else:
        print("WRONG for input", args, "expected", repr(want), "got", repr(got))
print(f"{{passed}}/{{len(CASES)}} visible tests passed")
'''


def write_workspace(p: Puzzle, workdir: Path, visible_k: int):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "PROBLEM.md").write_text(problem_markdown(p, visible_k), encoding="utf-8")
    sig = p.signature.rstrip()
    if not sig.endswith(":"):
        sig += ":"
    (workdir / "solution.py").write_text(f"{sig}\n    raise NotImplementedError\n", encoding="utf-8")
    cases = json.dumps([[a, o] for a, o in p.visible(visible_k)])
    (workdir / "run_tests.py").write_text(RUN_TESTS_TEMPLATE.format(fn=p.function_name, cases=cases),
                                          encoding="utf-8")


# ---------------------------------------------------------------- seeds & novelty

def load_seed_puzzles(path: Path, timeout_s: float = 10.0, python: str | None = None) -> list[Puzzle]:
    puzzles = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = Puzzle.from_dict(json.loads(line))
            p.source = "seed"
            v = validate(p, timeout_s, python)
            if not v.valid:
                raise ValueError(f"seed puzzle {p.title!r} is invalid: {v.reason}")
            puzzles.append(p)
    return puzzles


def _shingles(text: str, n: int = 3) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


class NoveltyBuffer:
    """Remembers recent statements; similarity = max word-3-gram Jaccard overlap."""

    def __init__(self, maxlen: int = 200):
        self.items: deque = deque(maxlen=maxlen)

    def max_similarity(self, statement: str) -> float:
        s = _shingles(statement)
        best = 0.0
        for other in self.items:
            union = len(s | other)
            if union:
                best = max(best, len(s & other) / union)
        return best

    def add(self, statement: str):
        self.items.append(_shingles(statement))

    def state(self) -> list:
        return [sorted(x) for x in self.items]

    def load(self, items: list):
        self.items.clear()
        for x in items:
            self.items.append(set(x))


def pick_topic(rng: random.Random) -> str:
    return rng.choice(TOPICS)
