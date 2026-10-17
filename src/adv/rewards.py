"""Reward functions: the heart of making the adversarial game stable.

A literal GAN-style zero-sum reward ("Maker wins when Solver fails") collapses: the
Maker learns to write impossible or ambiguous puzzles. Instead the Maker is paid for
puzzles at the EDGE of the Solver's ability (R-Zero / Absolute-Zero style):

    learnability(p) = 1 - 2 * |p - 0.5|        p = Solver's success rate on the puzzle

    p = 0.00 -> 0.0  (too hard: no learning signal for the Solver either)
    p = 0.50 -> 1.0  (perfect: the Solver's GRPO group has maximum reward variance)
    p = 1.00 -> 0.0  (too easy)

The coupling matters: GRPO only learns from groups whose rewards differ, and p = 0.5
is exactly where a group of attempts disagrees the most. The Maker is literally
rewarded for producing training signal for the Solver.
"""
from __future__ import annotations


def learnability(p_hat: float) -> float:
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def solver_reward(test_fraction: float, produced_code: bool, cfg: dict) -> float:
    """Fraction of hidden tests passed, plus a bonus for passing all of them."""
    if not produced_code:
        return 0.0
    r = test_fraction
    if test_fraction >= 1.0:
        r += cfg.get("full_pass_bonus", 0.5)
    return r


def maker_reward(valid: bool, p_hat: float, novelty_sim: float, cfg: dict) -> float:
    if not valid:
        return cfg.get("maker_invalid", -1.0)
    sim_pen = cfg.get("similarity_penalty", 0.5) * max(0.0, novelty_sim - cfg.get("similarity_free", 0.3)) / 0.7
    return learnability(p_hat) - sim_pen
