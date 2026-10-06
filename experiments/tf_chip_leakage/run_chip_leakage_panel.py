#!/usr/bin/env python
"""Plot the cross-TF repressive ChIP leakage panel (real vs shuffled log2FC and ISM logos) for AlphaGenome in K562.

Outputs (outputs/): chip_leakage_panel.{pdf,png}.

Run (env: featleak-dl):
    python tools/render_with_svg.py experiments/tf_chip_leakage/run_chip_leakage_panel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.size": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

OUT = Path(__file__).parent / "outputs"
REAL, SHUF = "#C44E52", "#9AA0A8"
ISM_PAIRS = [("NRF1", "REST"), ("YY1", "REST")]


def panel_bars(ax):
    d = pd.read_csv(OUT / "repressive_validated_K562.csv")
    d = d[d.p_vs_shuffle < 0.05].copy()
    # dedupe identical E-box motifs (MYC/MAX give the same result)
    d["pair"] = d.A + " → " + d.B
    d.loc[(d.A == "MAX") & (d.B == "REST"), "pair"] = "E-box (MYC/MAX) → REST"
    d = d[~((d.A == "MYC") & (d.B == "REST"))]
    d = d.sort_values("specificity")
    y = np.arange(len(d)); h = 0.38
    ax.barh(y + h / 2, d.dReal, h, color=REAL, label="real A motif")
    ax.barh(y - h / 2, d.dShuf, h, color=SHUF, label="shuffled A motif")
    ax.axvline(0, color="0.2", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(d.pair, fontsize=10)
    ax.set_xlabel("Δ predicted ChIP signal of TF B")
    ax.set_title("Motif specifically represses the WRONG TF", fontsize=11, pad=8)
    for yi, p in zip(y, d.p_vs_shuffle):
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*"
        ax.text(d.dReal.min() * 1.02, yi, star, va="center", ha="right",
                fontsize=10, color=REAL)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.text(0.98, 0.02, "← represses      activates →", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="0.4")


def logo(ax, arr, title, color_title):
    plot_logo(torch.tensor(arr, dtype=torch.float32), ax=ax)
    ax.set_title(title, fontsize=9.5, color=color_title, pad=3)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)


def main() -> None:
    fig = plt.figure(figsize=(13, 5.6))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.28, hspace=0.75)
    panel_bars(fig.add_subplot(gs[:, 0]))

    for row, (A, B) in enumerate(ISM_PAIRS):
        z = np.load(OUT / f"repressive_ism_{A}_{B}.npz")
        logo(fig.add_subplot(gs[row * 2, 1]), z["onA"],
             f"{A} motif  →  {A} ChIP   (on-target: ACTIVATES)", "#2E8B57")
        logo(fig.add_subplot(gs[row * 2 + 1, 1]), z["onB"],
             f"{A} motif  →  {B} ChIP   (leakage: REPRESSES)", "#C44E52")

    fig.text(0.02, 0.96, "a", fontsize=16, fontweight="bold")
    fig.text(0.55, 0.96, "b", fontsize=16, fontweight="bold")
    fig.suptitle("Feature leakage in TF ChIP-seq  (AlphaGenome, K562):  a TF motif represses a "
                 "different TF's binding", fontsize=12.5, y=1.02)
    fig.savefig(OUT / "chip_leakage_panel.pdf", bbox_inches="tight")
    fig.savefig(OUT / "chip_leakage_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done ->", OUT / "chip_leakage_panel.pdf")


if __name__ == "__main__":
    main()
