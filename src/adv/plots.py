"""train.png: six panels that tell you whether the game is healthy (overwritten in place)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .logbook import atomic_write, fmt_dur  # noqa: E402

C = {"solver": "#2a6fdb", "maker": "#e0662a", "good": "#2a9d5c", "bad": "#c8323b", "muted": "#8a8f98",
     "violet": "#7d4cdb"}


def _smooth(ys, k=5):
    out, acc = [], []
    for y in ys:
        acc = (acc + [y])[-k:]
        vals = [a for a in acc if a is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _series(rows, key):
    return [r.get(key) for r in rows]


def _line(ax, xs, ys, color, label, raw_alpha=0.25):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    px, py = zip(*pts)
    ax.plot(px, py, color=color, alpha=raw_alpha, lw=1)
    sm = [(x, y) for x, y in zip(xs, _smooth(ys)) if y is not None]
    ax.plot(*zip(*sm), color=color, lw=2, label=label)


def render(history: list[dict], verdicts: list[dict], title: str, eta_seconds: float | None, path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), dpi=100)
    fig.patch.set_facecolor("white")
    xs = [r["step"] + 1 for r in history]

    ax = axes[0][0]
    _line(ax, xs, _series(history, "solver_pass_rate"), C["solver"], "pass rate (all tests)")
    _line(ax, xs, _series(history, "solver_test_frac"), C["violet"], "mean test fraction")
    ax.axhspan(0.3, 0.7, color=C["good"], alpha=0.06, label="healthy band")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Solver: how often it cracks the Maker's puzzles")

    ax = axes[0][1]
    _line(ax, xs, _series(history, "maker_learnability"), C["maker"], "learnability 1-2|p̂-0.5|")
    _line(ax, xs, _series(history, "maker_valid_rate"), C["good"], "valid puzzle rate")
    _line(ax, xs, _series(history, "maker_novelty_sim"), C["muted"], "similarity to recent puzzles")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Maker: valid, novel, and at the edge of the Solver's ability")

    ax = axes[1][0]
    buckets = ["0", "0.25", "0.5", "0.75", "1"]
    cols = ["#c8323b", "#e8955a", "#2a9d5c", "#7fb0e8", "#2a6fdb"]
    stacks = [[(r.get("p_hat_counts") or {}).get(b, 0) for r in history] for b in buckets]
    totals = [max(1, sum(s[i] for s in stacks)) for i in range(len(history))]
    fracs = [[s[i] / totals[i] for i in range(len(history))] for s in stacks]
    if history:
        ax.stackplot(xs, *fracs, colors=cols, labels=[f"p̂={b}" for b in buckets], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Puzzle difficulty mix (green = learnable, red/blue = dead groups)")

    ax = axes[1][1]
    _line(ax, xs, _series(history, "think_tokens_solver"), C["solver"], "solver thinking tokens")
    _line(ax, xs, _series(history, "think_tokens_maker"), C["maker"], "maker thinking tokens")
    ax.set_ylabel("tokens")
    ax2 = ax.twinx()
    _line(ax2, xs, _series(history, "marker_rate_per_1k"), C["bad"], "overthinking markers / 1k")
    ax2.set_ylabel("markers per 1k thinking tokens", color=C["bad"])
    ax.set_title("Thinking length and overthinking-marker rate")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    ax = axes[2][0]
    _line(ax, xs, _series(history, "solver_grad_norm"), C["solver"], "solver grad norm")
    _line(ax, xs, _series(history, "maker_grad_norm"), C["maker"], "maker grad norm")
    _line(ax, xs, _series(history, "dead_group_frac"), C["muted"], "dead-group fraction")
    ax.set_title("Optimisation health")

    ax = axes[2][1]
    if verdicts:
        vx = [v["step"] + 1 for v in verdicts]
        ax.plot(vx, [v["stalled"] for v in verdicts], "o-", color=C["bad"], label="P(stalled)")
        for v in verdicts:
            if v["applied"]:
                ax.axvline(v["step"] + 1, color=C["violet"], alpha=0.5, ls="--")
                ax.text(v["step"] + 1, 0.98, v["action"].replace("_", " "), rotation=90, va="top", ha="right",
                        fontsize=7, color=C["violet"])
    ax.axhline(0.6, color=C["muted"], ls=":", lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Referee (d3-flash): stall probability and interventions")

    right = max(xs) + 0.5 if xs else 1
    for row in axes:
        for a in row:
            a.grid(alpha=0.25)
            a.set_xlabel("step")
            a.set_xlim(0.5, right)
            if a is not axes[1][1] and a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8, loc="best")
    fig.suptitle(f"{title}   |   ETA {fmt_dur(eta_seconds)}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    atomic_write(path, buf.getvalue())
