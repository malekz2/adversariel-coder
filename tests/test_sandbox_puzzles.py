import json
from pathlib import Path

from adv.puzzles import (NoveltyBuffer, Puzzle, code_hygiene_error, load_seed_puzzles, parse_maker_output,
                         validate, write_workspace)
from adv.sandbox import outputs_equal, run_function, score_against

ROOT = Path(__file__).resolve().parents[1]


def test_run_function_pass_fail_raise():
    code = "def f(x):\n    if x == 3:\n        raise ValueError('boom')\n    return x * 2\n"
    r = run_function(code, "f", [[1], [2], [3]], timeout_s=10)
    assert [o.ok for o in r.outcomes] == [True, True, False]
    assert r.outcomes[1].output == 4 and "ValueError" in r.outcomes[2].error
    assert score_against(r, [2, 5, 6]) == 1 / 3


def test_timeout_keeps_partial_results():
    code = "def f(x):\n    while x == 2:\n        pass\n    return x\n"
    r = run_function(code, "f", [[1], [2], [3]], timeout_s=3)
    assert r.timed_out
    assert r.outcomes[0].ok and r.outcomes[1] is None and r.outcomes[2] is None


def test_load_error_and_missing_function():
    assert run_function("def f(:\n", "f", [[1]]).load_error
    assert run_function("def g(x): return x\n", "f", [[1]]).load_error


def test_outputs_equal():
    assert outputs_equal([1, (2, 3)], [1, [2, 3]])
    assert outputs_equal(0.1 + 0.2, 0.3)
    assert not outputs_equal(True, 1)
    assert not outputs_equal({"a": 1}, {"a": 2})


def test_all_seeds_valid():
    seeds = load_seed_puzzles(ROOT / "data" / "seed_puzzles.jsonl")
    assert len(seeds) >= 25 and all(s.expected is not None for s in seeds)


def _maker_text(**over):
    d = dict(title="Double", statement="Return twice the integer x. The input is any integer, negative included.",
             function_name="double", signature="def double(x: int) -> int:",
             reference_solution="def double(x):\n    return 2 * x\n",
             test_inputs=[[1], [2], [-3], [0], [10], [7]], difficulty=1)
    d.update(over)
    return "thinking...\n```json\n" + json.dumps(d) + "\n```"


def test_parse_and_validate_good():
    p, why = parse_maker_output(_maker_text())
    assert why == "ok"
    v = validate(p)
    assert v.valid and p.expected == [2, 4, -6, 0, 20, 14]


def test_validation_rejects():
    cases = {
        "disallowed_import:os": dict(reference_solution="import os\ndef double(x):\n    return 2 * x\n"),
        "constant_output": dict(reference_solution="def double(x):\n    return 0\n"),
        "test_count": dict(test_inputs=[[1], [2]]),
        "reference_raised": dict(reference_solution="def double(x):\n    return 2 * y\n"),
        "nondeterministic": dict(reference_solution="def double(x):\n    return list({'a' + str(x), 'b', 'c', 'd', 'e'})\n"),
        "solution_leaked": dict(statement="Return twice x. Hint: return 2 * x + 0 * 12345678901234 does it.",
                                reference_solution="def double(x):\n    return 2 * x + 0 * 12345678901234\n"),
    }
    for reason, over in cases.items():
        p, _ = parse_maker_output(_maker_text(**over))
        assert validate(p).reason == reason, reason
    assert parse_maker_output("no json here")[1] == "no_json"


def test_hygiene():
    assert code_hygiene_error("from collections import Counter\nimport math, heapq\n") is None
    assert code_hygiene_error("import subprocess\n") == "disallowed_import:subprocess"
    assert code_hygiene_error("x = open('f')\n") == "banned_call"


def test_workspace_and_novelty(tmp_path):
    p, _ = parse_maker_output(_maker_text())
    validate(p)
    write_workspace(p, tmp_path, 3)
    assert "double(1)" in (tmp_path / "PROBLEM.md").read_text()
    (tmp_path / "solution.py").write_text("def double(x):\n    return x + x\n")
    import subprocess, sys
    out = subprocess.run([sys.executable, "run_tests.py"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "3/3 visible tests passed" in out
    nb = NoveltyBuffer()
    nb.add(p.statement)
    assert nb.max_similarity(p.statement) == 1.0
    assert nb.max_similarity("Something completely different about graphs and edges here") < 0.2
