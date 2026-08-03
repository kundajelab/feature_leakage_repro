#!/usr/bin/env python
"""Figure 1 OCT-SOX ablation as log2 fold-change: whole-track vs local window.

log2FC = log2( pred(with motif) / pred(ablated) ), per OCT-SOX instance.
Two versions, because it matters a lot which signal you take the ratio of:
  * WHOLE-TRACK sum  -> log2FC collapses toward 0 (the local motif effect is a
    tiny fraction of the summed track), so it looks empty/uninformative.
  * LOCAL window (+/-~500 bp around the motif) -> the ratio is meaningful and
    negative = repressive (the leakage direction), comparable across models.

Reads before/after arrays saved by run_alphagenome.py / run_grelu.py.
Run: python tools/render_with_svg.py figures/figure1/run_logfc_boxplots.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"
MODELS = [("AlphaGenome", "alphagenome"), ("Borzoi", "borzoi"), ("Enformer", "enformer")]
EPS = 1e-3


def lfc(tag, suffix):
    b = np.load(OUT / f"fig1_{tag}_octsox_before{suffix}.npy")   # with motif
    a = np.load(OUT / f"fig1_{tag}_octsox_after{suffix}.npy")     # ablated
    return np.log2((b + EPS) / (a + EPS))


def panel(ax, suffix, title):
    data, labels, notes = [], [], []
    for name, tag in MODELS:
        try:
            v = lfc(tag, suffix)
        except FileNotFoundError:
            continue
        data.append(v); labels.append(name)
        p = stats.wilcoxon(v, alternative="less").pvalue
        notes.append((np.median(v), p))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.5,
                    patch_artist=True, medianprops=dict(color="black"))
    for pt in bp["boxes"]:
        pt.set(facecolor="lightgray", edgecolor="black")
    ax.axhline(0, ls="--", color="0.6")
    ax.set_ylabel("log2FC( with motif / ablated )")
    ax.set_title(title, fontsize=9)
    for i, (med, p) in enumerate(notes):
        star = "*" if p < 0.05 else ""
        ax.annotate(f"{med:+.2f}{star}", (i + 1, med), fontsize=7, ha="center",
                    va="bottom", color="#c1121f")
    return notes


def main() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
    panel(a1, "", "WHOLE-TRACK sum  →  log2FC diluted ~0")
    n = panel(a2, "_local", "LOCAL window (±~500 bp)  →  meaningful log2FC")
    fig.suptitle("Fig 1 OCT-SOX ablation as log2FC — negative = repressive (leakage)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_octsox_boxplot_logfc.pdf")
    fig.savefig(OUT / "fig1_octsox_boxplot_logfc.png", dpi=120)
    plt.close(fig)

    # standalone local-window boxplot (the one to actually use)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    panel(ax, "_local", "Fig 1 — OCT-SOX ablation, local-window log2FC")
    fig.tight_layout(); fig.savefig(OUT / "fig1_octsox_boxplot_logfc_local.pdf")
    plt.close(fig)
    print("done ->", OUT / "fig1_octsox_boxplot_logfc.pdf")


if __name__ == "__main__":
    main()
