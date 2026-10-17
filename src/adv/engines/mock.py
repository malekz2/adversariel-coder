"""A scripted fake model for `--dry-run`: exercises the whole loop without any weights.

It writes real (valid or deliberately broken) puzzles, real (correct or buggy) solutions,
and in Pi mode it actually calls Pi's tools (read PROBLEM.md, write solution.py, run the
tests). Its "skill" rises over time so log.txt and train.png look like a run in progress.
"""
from __future__ import annotations

import json
import random
import re
import threading

from .base import Engine, GenParams, GenResult

FAMILIES = [
    dict(stem="sum_multiples", sig="(n: int, k: int) -> int", args=lambda r: [r.randint(0, 60), r.randint(1, 9)],
         text="Return the sum of all non-negative integers strictly below n that are divisible by k.",
         good="def {f}(n, k):\n    return sum(i for i in range(n) if i % k == 0)\n",
         bad="def {f}(n, k):\n    return sum(i for i in range(n + 1) if i % k == 0)\n", difficulty=1),
    dict(stem="count_vowel_words", sig="(s: str) -> int", args=lambda r: [" ".join(r.choice(["apple", "tree", "sky", "egg", "orb", "nth"]) for _ in range(r.randint(0, 6)))],
         text="Count the words in s (split on single spaces, ignore empty pieces) that start with a vowel a, e, i, o or u.",
         good="def {f}(s):\n    return sum(1 for w in s.split(' ') if w and w[0] in 'aeiou')\n",
         bad="def {f}(s):\n    return sum(1 for w in s.split(' ') if w[:1] in 'aeiouy')\n", difficulty=2),
    dict(stem="max_window_sum", sig="(nums: list[int], k: int) -> int", args=lambda r: [[r.randint(-9, 9) for _ in range(r.randint(3, 9))], r.randint(1, 3)],
         text="Return the maximum sum of any k consecutive elements of nums. 1 <= k <= len(nums).",
         good="def {f}(nums, k):\n    return max(sum(nums[i:i + k]) for i in range(len(nums) - k + 1))\n",
         bad="def {f}(nums, k):\n    return max(sum(nums[i:i + k]) for i in range(len(nums) - k))\n", difficulty=3),
    dict(stem="balanced_digits", sig="(n: int) -> bool", args=lambda r: [r.randint(0, 99999)],
         text="A number is balanced if the sum of the digits in the first half equals the sum in the second half (the middle digit of an odd-length number is ignored). Return whether n is balanced.",
         good="def {f}(n):\n    s = str(n)\n    h = len(s) // 2\n    return sum(map(int, s[:h])) == sum(map(int, s[len(s) - h:]))\n",
         bad="def {f}(n):\n    s = str(n)\n    h = len(s) // 2\n    return sum(map(int, s[:h])) == sum(map(int, s[h:]))\n", difficulty=4),
]

MARKERS = [" wait", " Hmm", " Alternatively", " maybe", " but", " actually"]


class MockEngine(Engine):
    max_parallel = 4

    def __init__(self, tok, seed: int = 0):
        self.tok = tok
        self.rng = random.Random(seed)
        self.skill = 0.2          # trainer nudges this up after every "update"
        self.maker_quality = 0.5  # probability a puzzle is valid
        self.known: dict[str, dict] = {}
        self.lock = threading.Lock()

    def bump(self, solver_delta: float, maker_delta: float):
        self.skill = min(0.95, self.skill + solver_delta)
        self.maker_quality = min(0.95, self.maker_quality + maker_delta)

    def _think(self, budget: int) -> tuple[str, int, bool]:
        n_markers = max(0, int(self.rng.gauss(8 * (1 - self.skill), 2)))
        words = ["consider", "the", "edge", "case", "loop", "index", "sum"] + [m.strip() for m in MARKERS] * (n_markers // 3 + 1)
        length = self.rng.randint(budget // 4, int(budget * 1.2))
        forced = length > budget
        text = " ".join(self.rng.choice(words) for _ in range(min(length, budget) // 6))
        return text, n_markers, forced

    def _maker(self) -> str:
        fam = self.rng.choice(FAMILIES)
        f = f"{fam['stem']}_{self.rng.randint(100, 999)}"
        tests = [fam["args"](self.rng) for _ in range(self.rng.randint(8, 12))]
        ref = fam["good"].format(f=f)
        broken = self.rng.random() > self.maker_quality
        if broken:
            ref = ref.replace("return", "return undefined_name +", 1)
        with self.lock:
            self.known[f] = dict(good=fam["good"].format(f=f), bad=fam["bad"].format(f=f), difficulty=fam["difficulty"])
        d = dict(title=f"{fam['stem'].replace('_', ' ').title()} #{f[-3:]}", statement=fam["text"] + f" Implement {f}.",
                 function_name=f, signature=f"def {f}{fam['sig']}:", reference_solution=ref,
                 test_inputs=tests, difficulty=fam["difficulty"])
        return "I will write a puzzle about " + fam["stem"] + ".\n```json\n" + json.dumps(d) + "\n```"

    def _solution_for(self, text: str) -> str:
        m = re.search(r"def\s+(\w+)\s*\(", text)
        info = self.known.get(m.group(1)) if m else None
        if info is None:
            return "def unknown():\n    return None\n"
        p_correct = max(0.05, min(0.95, self.skill * 1.6 - 0.15 * info["difficulty"] + 0.2))
        return info["good"] if self.rng.random() < p_correct else info["bad"]

    def _pi_turn(self, prompt: str) -> str:
        responses = prompt.count("<tool_response>")
        if responses == 0:
            return "<tool_call>\n<function=read>\n<parameter=path>\nPROBLEM.md\n</parameter>\n</function>\n</tool_call>"
        if responses == 1:
            code = self._solution_for(prompt[prompt.rfind("<tool_response>"):])
            return ("I'll write the solution.\n<tool_call>\n<function=write>\n<parameter=path>\nsolution.py\n</parameter>\n"
                    f"<parameter=content>\n{code}\n</parameter>\n</function>\n</tool_call>")
        if responses == 2:
            return "<tool_call>\n<function=bash>\n<parameter=command>\npython run_tests.py\n</parameter>\n</function>\n</tool_call>"
        return "The solution is written in solution.py and the visible tests were run."

    def generate(self, prompt_ids, params: GenParams, in_thinking: bool) -> GenResult:
        tok = self.tok
        prompt = tok.decode(prompt_ids)
        think, markers, forced = self._think(params.think_budget) if in_thinking else ("", 0, False)
        if params.adapter == "maker":
            answer = self._maker()
        elif "# Tools" in prompt:
            answer = self._pi_turn(prompt)
        else:
            answer = "```python\n" + self._solution_for(prompt) + "```"
        think_ids = tok.encode(think) if in_thinking else []
        out = think_ids + ([tok.think_end] if in_thinking else []) + tok.encode("\n\n" + answer) + [tok.im_end]
        mask = [1] * len(out)
        if forced and in_thinking:
            mask[len(think_ids)] = 0
        return GenResult(out, mask, "stop", len(think_ids), forced, markers)
