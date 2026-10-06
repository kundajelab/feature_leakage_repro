#!/usr/bin/env python
"""Plot the shuffle-corrected OCT-SOX effect for AlphaGenome vs ChromBPNet (Figure 2 panel b).

Outputs (outputs/): panelb_octsox_specificity.{pdf,png,svg}.

Run (env: featleak-dl):
    python run_boxplot.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:   # Arial-metric font, so text stays editable and matches Arial/Helvetica
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                            "pdf.fonttype": 42, "ps.fonttype": 42})
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                           "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"
MULTI, CBP, GRID = "#C44E52", "#4C72B0", "#B8B8B8"


def _violin(ax, series, ylabel, title):
    data = [s[1] for s in series]
    parts = ax.violinplot(data, positions=range(len(data)), widths=0.78, showextrema=False)
    for pc, (_, _, c) in zip(parts["bodies"], series):
        pc.set_facecolor(c); pc.set_alpha(0.85); pc.set_edgecolor("white"); pc.set_linewidth(0.8)
    yt = max(np.percentile(d, 99) for _, d, _ in series)
    for i, (name, d, c) in enumerate(series):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color="0.15", lw=5, zorder=3)
        ax.plot(i, med, "o", color="white", ms=6, zorder=4)
        p = stats.wilcoxon(d, alternative="less").pvalue
        sig = "n.s." if p > 0.05 else ("$*{*}*$" if p < 1e-3 else "$*$")
        ax.text(i, yt + 0.15, sig, ha="center", va="bottom",
                fontsize=15 if sig != "n.s." else 11, color="0.35" if sig == "n.s." else c)
        ax.text(i, med, f"  {med:+.2f}", va="center", ha="left", fontsize=9.5)
    ax.axhline(0, ls=(0, (4, 3)), color=GRID, lw=1.1, zorder=1)
    ax.set_xticks(range(len(data))); ax.set_xticklabels([s[0] for s in series])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, pad=12)


def main() -> None:
    # both models read the SAME ENCODE experiment (ENCSR000EPO, fibroblast of dermis)
    ag = pd.read_csv(OUT / "ag_octsox_specificity_dermis_len524288.csv").specificity.values
    cbp = pd.read_csv(OUT / "cbp_octsox_specificity_dermis_matched.csv").specificity.values
    n = min(len(ag), len(cbp))
    series = [("AlphaGenome\n(multi-task)", ag, MULTI), ("ChromBPNet\n(single-task)", cbp, CBP)]

    fig, ax = plt.subplots(figsize=(5, 5.2))
    _violin(ax, series, "OCT-SOX effect  (shuffle-corrected log2FC)",
            f"OCT-SOX insertion across {n} dermal fibroblast enhancers\n(ENCODE ENCSR000EPO)")
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"panelb_octsox_specificity.{ext}", dpi=150 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    for name, d, _ in series:
        p = stats.wilcoxon(d, alternative="less").pvalue
        print(f"{name.splitlines()[0]}: n={len(d)} median={np.median(d):+.4f} Wilcoxon(<0) p={p:.3g}")
    print("done ->", OUT / "panelb_octsox_specificity.pdf")


if __name__ == "__main__":
    main()
