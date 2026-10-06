#!/usr/bin/env python
"""Plot stacked AlphaGenome and ChromBPNet OCT-SOX insertion tracks at the chr17 example locus (Figure 2c/d).

Outputs (outputs/): tracks_screenshot_matched.{pdf,png}.

Run (env: featleak-dl):
    python run_tracks_screenshot.py --window 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

F2 = Path(__file__).parent / "outputs"
REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "chrombpnet_octsox_stats" / "outputs"
PEAK = ("chr17", 76671700, 76671820)      # the fibroblast enhancer peak (120 bp)
SIG = "#443152"      # track colour
INS = "#7f9dd1"      # OCT-SOX insertion dashed line
ROWS = ["Fibroblast DNase", "1 insertion", "2 insertions", "3 insertions", "4 insertions"]


def panel(fig, gs, row0, profs, bp, ins_bp, W, model, ylabel, locus_label):
    m = (bp >= -W) & (bp <= W)
    ymax = profs[:, m].max() * 1.1
    axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[row0 + i])
        ax.plot(bp[m], profs[i][m], color=SIG, lw=0.5)
        ax.set_ylim(0, ymax); ax.set_xlim(-W, W)
        for o in ins_bp[:i]:                       # cumulative insertion markers
            ax.axvline(o, ls="--", color=INS, lw=1.1)
        ax.set_yticks([0, round(ymax * 0.9)])
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
        if i == 0:
            ax.text(0.01, 0.82, "Fibroblast DNase", transform=ax.transAxes, fontsize=8)
            ax.text(0.99, 0.82, model, transform=ax.transAxes, fontsize=9, ha="right")
            ax.set_title(locus_label, fontsize=8)
        if i == 1:                                 # legend entry like the screenshot
            ax.plot([], [], ls="--", color=INS, lw=1.1, label="OCT-SOX Insertion")
            ax.legend(loc="upper right", fontsize=6.5, frameon=False)
        axes.append(ax)
    axes[-1].set_xlabel("Position", fontsize=9)
    # shared rotated y-label centred on the group
    mid = axes[2].get_position()
    fig.text(0.03, (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2,
             "Predicted Accessibility", rotation=90, va="center", fontsize=10)
    return axes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=500)
    args = ap.parse_args()
    W = args.window

    ag = np.load(F2 / "alphagenome_track_data_skin_biceps.npz", allow_pickle=True)
    cbp = np.load(EXP / "cbp_raw_chr17.npz", allow_pickle=True)

    # title shows the DISPLAYED genomic window (peak centre +/- W), not the 120 bp peak
    chrom, ps, pe = PEAK
    mid = (ps + pe) // 2
    locus_label = f"{chrom}:{mid - W:,}-{mid + W:,}  ({2 * W} bp window)"

    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(11, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1],
                  hspace=0.35)
    # panel c: AlphaGenome (rows 0-4)
    panel(fig, gs, 0, ag["profiles"], ag["bp"], ag["ins_bp"], W,
          "AlphaGenome", "Predicted Accessibility", locus_label)
    # panel d: ChromBPNet (rows 6-10)
    panel(fig, gs, 6, cbp["profiles_exp"], cbp["bp"], cbp["ins_bp"], W,
          "ChromBPNet", "Predicted Accessibility", locus_label)
    fig.text(0.02, 0.965, "c", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.485, "d", fontsize=15, fontweight="bold")

    fig.savefig(F2 / "tracks_screenshot_matched.pdf", bbox_inches="tight")
    fig.savefig(F2 / "tracks_screenshot_matched.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"window +/-{W} bp; AG peak={ag['profiles'][0].max():.1f} -> "
          f"{ag['profiles'][4].max():.1f}; CBP peak={cbp['profiles_exp'][0].max():.1f} -> "
          f"{cbp['profiles_exp'][4].max():.1f}")
    print("done ->", F2 / "tracks_screenshot_matched.pdf")


if __name__ == "__main__":
    main()
