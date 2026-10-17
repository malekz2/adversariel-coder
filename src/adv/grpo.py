"""GRPO (Group Relative Policy Optimisation) written out by hand.

Policy-gradient RL in one paragraph: sample several answers, score them, then push up
the log-probability of tokens in answers that scored ABOVE their group's average and push
down the ones BELOW. GRPO's trick is that the "average" is just the mean reward of the
other attempts at the same prompt, so no value network (critic) is needed.

    advantage_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    loss        = - sum_t  A_i * ratio_t * mask_t   /  sum(mask)          (token-mean, DAPO style)
    ratio_t     = exp(logp_new_t - logp_old_t)   clipped to [1-eps, 1+eps] (PPO)

With one gradient step per batch (ppo_epochs = 1) the policy that sampled the data and
the policy being updated are the same, so ratio == 1 and this reduces to REINFORCE with a
group baseline. The clip only matters if you set ppo_epochs > 1.

Groups where every attempt got the same reward have advantage 0 and contribute nothing.
The fraction of such "dead" groups is one of the most important numbers in the log: if it
creeps toward 1.0, the game has stopped producing signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class Sample:
    """One training sequence: prompt + completion tokens, with a per-token loss mask."""
    input_ids: list[int]
    loss_mask: list[int]      # 1 = token the model generated and should learn from
    advantage: float
    role: str                 # "maker" or "solver"
    group: str = ""           # GRPO group id (solver: puzzle uid)
    reward: float = 0.0
    meta: dict = field(default_factory=dict)


def group_advantages(rewards: list[float], groups: list[str], scale_by_std: bool = True,
                     eps: float = 1e-4) -> tuple[list[float], float]:
    """Return (advantages, fraction_of_dead_groups)."""
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    adv = [0.0] * len(rewards)
    dead = 0
    for idx in by_group.values():
        rs = [rewards[i] for i in idx]
        mean = sum(rs) / len(rs)
        var = sum((r - mean) ** 2 for r in rs) / max(1, len(rs))
        std = math.sqrt(var)
        if std < 1e-8:
            dead += 1
            continue
        for i in idx:
            a = rewards[i] - mean
            adv[i] = a / (std + eps) if scale_by_std else a
    return adv, dead / max(1, len(by_group))


def policy_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, mask: torch.Tensor,
                advantage: float, clip_eps: float = 0.2,
                logp_ref: torch.Tensor | None = None, kl_beta: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Per-sequence clipped surrogate. Returns the SUM over masked tokens (caller normalises)."""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    per_tok = -torch.minimum(unclipped, clipped)
    stats = {}
    if logp_ref is not None and kl_beta > 0:
        # k3 estimator of KL(new || ref): unbiased, always >= 0
        d = logp_ref - logp_new
        kl = torch.exp(d) - d - 1
        per_tok = per_tok + kl_beta * kl
        stats["kl"] = float((kl * mask).sum() / mask.sum().clamp(min=1))
    loss_sum = (per_tok * mask).sum()
    with torch.no_grad():
        stats["clip_frac"] = float((((ratio - 1).abs() > clip_eps).float() * mask).sum() / mask.sum().clamp(min=1))
    return loss_sum, stats
