#!/usr/bin/env python
"""Plot greedy-design optimization trajectories and per-motif effects (Figure 6B/D).

Outputs (outputs/): sequence_optimization_scores.pdf, filtered_motif_effects_offtarget.pdf,
filtered_motif_effects_combined.pdf.

Run (env: featleak-dl):
    python run_plots.py --results outputs/all_results.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def default_results() -> str:
    """The design run in outputs/, overridable with FEATLEAK_FIG6_RESULTS."""
    return os.environ.get("FEATLEAK_FIG6_RESULTS", str(OUT / "all_results.pkl"))


def _traj(r, key):
    return np.array([i.item() for i in r[key]])


def panel_b(results):
    """On/off-target score trajectories."""
    max_len = max(max(len(_traj(r, "score_target_1")),
                      len(_traj(r, "score_target_2"))) for r in results)
    s1, s2 = [], []
    for r in results:
        a = _traj(r, "score_target_1"); b = _traj(r, "score_target_2")
        a = np.pad(a, (0, max_len - len(a)), constant_values=a[-1])
        b = np.pad(b, (0, max_len - len(b)), constant_values=b[-1])
        s1.append(a); s2.append(b)
    s1 = np.array(s1); s2 = np.array(s2)
    steps = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    for arr, color, label in [(s1, "#2166AC", "Melanocyte (on-target)"),
                              (s2, "#B2182B", "Keratinocyte (off-target)")]:
        for row in arr:
            ax.plot(steps, row, color=color, alpha=0.04, lw=0.6)
        ax.plot(steps, arr.mean(0), color=color, lw=3, label=label)
    ax.set_xlabel("Optimization Step"); ax.set_ylabel("Score")
    ax.set_title("Sequence Optimization Progress"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "sequence_optimization_scores.pdf")
    plt.close(fig)


def panel_d(results, metric="offtarget"):
    """Per-motif effect forest plot.

    metric="offtarget": effect on the off-target keratinocyte prediction only
        (matches the figure caption; a repressive motif lowers the off-target).
    metric="combined":  Delta(on-target) + Delta(off-target) (matches the figure's
        rendered x-axis label).
    """
    assert metric in ("offtarget", "combined")
    motif_effects = {}
    for r in results:
        s1 = _traj(r, "score_target_1"); s2 = _traj(r, "score_target_2")
        for m, motif in enumerate(r["motif_names_added"]):
            d_off = s2[m + 1] - s2[m]
            effect = d_off if metric == "offtarget" else (s1[m + 1] - s1[m]) + d_off
            motif_effects.setdefault(motif, []).append(effect)
    filtered = {k: v for k, v in motif_effects.items() if len(v) > 1}
    order = sorted(filtered, key=lambda k: np.mean(filtered[k]), reverse=True)
    n_rep = sum(np.mean(filtered[k]) < 0 for k in filtered)
    print(f"    [{metric}] {len(filtered)} motifs with >1 observation ({n_rep} repressive)")

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * 0.25)), dpi=200)
    for i, motif in enumerate(order):
        vals = filtered[motif]
        mean, std = np.mean(vals), np.std(vals)
        color = "darkred" if mean > 0 else "darkblue"
        ax.scatter(mean, i, s=55, color=color, alpha=0.85, zorder=3)
        ax.errorbar(mean, i, xerr=std, fmt="none", color=color, alpha=0.5, capsize=3)
    ax.axvline(0, color="0.5", lw=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
    xlabel = ("Average Motif Effect on off-target keratinocyte" if metric == "offtarget"
              else "Average Motif Effect (Combined Score Change)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Motif Names")
    ax.set_title(f"Motif Effects (n={len(order)} motifs with >1 observation, {metric})")
    import matplotlib.lines as mlines
    ax.legend(handles=[
        mlines.Line2D([], [], color="darkred", marker="o", ls="", label="Positive Effect"),
        mlines.Line2D([], [], color="darkblue", marker="o", ls="", label="Negative Effect"),
    ], loc="lower right")
    fig.tight_layout()
    fname = "filtered_motif_effects" + ("_offtarget" if metric == "offtarget" else "_combined")
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=default_results(),
                    help="path to a greedy-design results pickle "
                         "(default: outputs/all_results.pkl)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with open(args.results, "rb") as fh:
        results = pickle.load(fh)
    print(f"loaded {len(results)} designed sequences from {args.results}")
    panel_b(results)
    panel_d(results, metric="offtarget")   # matches the figure caption
    panel_d(results, metric="combined")    # matches the figure's rendered x-axis label
    print("done ->", OUT)


if __name__ == "__main__":
    main()
