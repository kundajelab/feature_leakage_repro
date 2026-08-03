#!/usr/bin/env python
"""Two-bar paper-comparable fidelity: AlphaGenome (pre-FT) vs AlphaGenome fine-tuned (C).

Same visual style as the 'Fidelity' panel of figure_objective_compare, but using the
PAPER-COMPARABLE metric (untransformed Pearson r across 1bp bins per interval, averaged
over 120 held-out chr8/chr9 16kb intervals) — ~0.65, not the generous region-total+log1p
0.93. De-leaking (objective C, ChromBPNet gate) leaves fidelity unchanged.

Numbers are the means printed by 21_paperstyle_fidelity.py (--num 120 --win 16384); this
script just re-plots them (matplotlib only, no GPU/OAK). CPU only.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                           "font.family": "sans-serif", "font.size": 12,
                           "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "outputs"

# paper-comparable fidelity (untransformed across-1bp-bin Pearson r, mean over 120 intervals)
LABELS = ["AlphaGenome", "AlphaGenome\nFine-tuned"]
VALUES = [0.6526, 0.6538]                 # pre-FT , objective C
COLORS = ["#c0392b", "#8e44ad"]           # match figure_objective_compare (pre=red, C=purple)


def main():
    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    ax.bar(range(2), VALUES, width=0.62, color=COLORS, alpha=0.9,
           edgecolor="0.2", linewidth=0.8)
    for i, v in enumerate(VALUES):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=12, color="0.15")
    ax.set_xticks(range(2)); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Pearson r vs experiment")
    ax.set_title("Fidelity\n(paper metric: untransformed, across 1-bp bins)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"figure_fidelity_bar_paper.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    print(f"pre-FT={VALUES[0]:.4f}  C(fine-tuned)={VALUES[1]:.4f}  delta={VALUES[1]-VALUES[0]:+.4f}")
    print("saved -> figure_fidelity_bar_paper.pdf/.svg/.png")


if __name__ == "__main__":
    main()
