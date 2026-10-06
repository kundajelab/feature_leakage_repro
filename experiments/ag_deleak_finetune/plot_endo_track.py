#!/usr/bin/env python
"""Endogenous OCT-SOX ablation at one locus, as predicted DNase tracks.

Reads an npz from `finetune_dermis_torch.py --stage endo_track`: predicted dermal
fibroblast DNase over the central 300 bp with the endogenous OCT-SOX intact and with its
14 bp dinucleotide-shuffled, for the pretrained and the fine-tuned head. Before
fine-tuning AlphaGenome predicts LESS accessibility with the motif present; afterwards
the two profiles coincide.

Run (any env with matplotlib):
    python plot_endo_track.py --npz endo_track_chr9_444726.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.6})
import matplotlib.pyplot as plt  # noqa: E402

D = Path(__file__).parent / "outputs" / "dermis_torch"
INTACT, ABL = "#C44E52", "#9AA0A8"
MOTIF_LEN = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--stem", default=None)
    a = ap.parse_args()
    d = np.load(D / a.npz)
    chrom, mid = str(d["chrom"]), int(d["mid"])
    n = len(d["pre_intact"])
    x = np.arange(n) - n // 2

    # separate y-axes: the fine-tuned head's absolute scale differs, and the point of the
    # panel is the gap between intact and ablated within each model
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for ax, pfx, title, eff in (
            (axes[0], "pre", "AlphaGenome", float(d["effect_pre"])),
            (axes[1], "ft", "AlphaGenome fine-tuned", float(d["effect_ft"]))):
        ax.plot(x, d[f"{pfx}_intact"], color=INTACT, lw=1.2, label="OCT-SOX intact")
        ax.plot(x, d[f"{pfx}_ablated"], color=ABL, lw=1.2, ls="--", label="OCT-SOX ablated")
        ax.axvspan(-MOTIF_LEN // 2, MOTIF_LEN // 2, color="0.85", zorder=0, lw=0)
        ax.set_title(f"{title}\nmotif effect {eff:+.2f} log2", fontsize=8)
        ax.set_xlabel("position relative to motif (bp)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("predicted DNase")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Endogenous OCT-SOX, {chrom}:{mid:,} (held out)", fontsize=8.5)
    fig.tight_layout()
    stem = a.stem or a.npz.replace(".npz", "")
    fig.savefig(D / f"{stem}.pdf"); fig.savefig(D / f"{stem}.png", dpi=250)
    print(f"{chrom}:{mid} effect {float(d['effect_pre']):+.3f} -> {float(d['effect_ft']):+.3f}")
    print("saved ->", D / f"{stem}.pdf")


if __name__ == "__main__":
    main()
