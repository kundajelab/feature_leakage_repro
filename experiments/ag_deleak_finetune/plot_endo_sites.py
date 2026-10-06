#!/usr/bin/env python
"""Endogenous OCT-SOX ablation screen (held-out chr8/chr9 dermis peaks), before vs after.

Reads outputs/dermis_torch/endo_sites_mm2.csv from
`finetune_dermis_torch.py --stage endo_screen`, where each site's motif effect is
log2(predicted signal with the motif / with the motif scrambled). A negative effect means
AlphaGenome reads the endogenous OCT-SOX as repressive, which is the leakage signature.

Run (any env with matplotlib):  python plot_endo_sites.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
PRE, FT = "#C44E52", "#8E6FB0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="endo_sites_mm2.csv")
    ap.add_argument("--stem", default="fig7_endo_ablation")
    a = ap.parse_args()
    d = pd.read_csv(D / a.csv)
    rep = d[d.effect_pre < 0]
    p = stats.wilcoxon(rep.effect_pre, rep.effect_ft).pvalue

    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    for r in d.itertuples():
        c = PRE if r.effect_pre < 0 else "0.75"
        ax.plot([0, 1], [r.effect_pre, r.effect_ft], color=c, lw=0.7,
                alpha=0.9 if r.effect_pre < 0 else 0.45, zorder=3 if r.effect_pre < 0 else 1)
    ax.scatter(np.zeros(len(rep)), rep.effect_pre, s=14, color=PRE, zorder=4, lw=0)
    ax.scatter(np.ones(len(rep)), rep.effect_ft, s=14, color=FT, zorder=4, lw=0)
    ax.axhline(0, ls=":", color="0.4", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AlphaGenome", "AlphaGenome\nfine-tuned"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("OCT-SOX motif effect (log2, ablation)")
    ax.set_title("Fine-tuning removes the repression at endogenous OCT-SOX sites\n"
                 f"{len(rep)} of {len(d)} held-out sites repressive before: median "
                 f"{rep.effect_pre.median():+.2f} to {rep.effect_ft.median():+.2f}, "
                 f"p = {p:.0e}", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(D / f"{a.stem}.pdf"); fig.savefig(D / f"{a.stem}.png", dpi=250)
    print(f"n={len(d)}; repressive before={len(rep)}; median effect all "
          f"{d.effect_pre.median():+.3f} -> {d.effect_ft.median():+.3f}")
    print("saved ->", D / f"{a.stem}.pdf")


if __name__ == "__main__":
    main()
