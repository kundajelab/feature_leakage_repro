#!/usr/bin/env python
"""REST-centric ChIP-leakage figure, panels a+b, from CACHED results (no AG).

Preview built while AlphaGenome is unavailable (OAK permission issue). Panel (a)
log2FC is approximated from the cached mean ChIP changes and the REST baseline
(~5139); panel (b) uses the cached ISM contribution arrays. The profiles (panel
c) and CTCF's ISM require re-running run_rest_leakage_figure.py once AG loads.

Run: python tools/render_with_svg.py experiments/tf_chip_leakage/run_rest_leakage_preview.py
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
REST_BASELINE = 5138.7            # from run_repressive.py baseline report

# panel a: leaker -> REST, cached mean deltas
PAIRS_A = [("NRF1", "NRF1"), ("YY1", "YY1"), ("E-box (MYC/MAX)", "MYC"), ("CTCF", "CTCF")]
# panel b: cached ISM (npz key, title, colour)
LOGOS_B = [("REST_CTCF", "onA", "REST motif → REST ChIP  (on-target: ACTIVATES)", "#2E8B57"),
           ("NRF1_REST", "onB", "NRF1 motif → REST ChIP  (leakage: REPRESSES)", REAL),
           ("YY1_REST", "onB", "YY1 motif → REST ChIP  (leakage: REPRESSES)", REAL),
           ("MYC_REST", "onB", "E-box (MYC/MAX) motif → REST ChIP  (leakage: REPRESSES)", REAL)]


def main() -> None:
    v = pd.read_csv(OUT / "repressive_validated_K562.csv")
    v = v[v.B == "REST"].set_index("A")

    def lfc(delta):
        return np.log2((REST_BASELINE + delta) / REST_BASELINE)

    rows = []
    for label, A in PAIRS_A:
        r = v.loc[A]
        rows.append({"leaker": label, "lfc_real": lfc(r.dReal),
                     "lfc_shuf": lfc(r.dShuf), "p": r.p_vs_shuffle})
    d = pd.DataFrame(rows).sort_values("lfc_real")

    fig = plt.figure(figsize=(11.5, 4.8))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.0, 1.25], wspace=0.32, hspace=0.6)

    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(d)); h = 0.38
    ax.barh(y + h / 2, d.lfc_real, h, color=REAL, label="real motif")
    ax.barh(y - h / 2, d.lfc_shuf, h, color=SHUF, label="shuffled motif")
    ax.axvline(0, color="0.2", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(d.leaker + " → REST", fontsize=10)
    ax.set_xlabel("REST ChIP  log2 fold-change")
    ax.set_title("Motif specifically represses REST", fontsize=11, pad=8)
    for yi, p in zip(y, d.p):
        ax.text(d.lfc_real.min() * 1.04, yi, "***" if p < 1e-3 else "**" if p < 1e-2 else "*",
                va="center", ha="right", fontsize=10, color=REAL)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    for r, (fkey, akey, ttl, col) in enumerate(LOGOS_B):
        a = fig.add_subplot(gs[r, 1])
        arr = np.load(OUT / f"repressive_ism_{fkey}.npz")[akey]
        plot_logo(torch.tensor(arr, dtype=torch.float32), ax=a)
        a.set_title(ttl, fontsize=9, color=col, pad=2)
        a.set_xticks([]); a.set_yticks([])
        for s in ("left", "bottom"):
            a.spines[s].set_visible(False)

    fig.text(0.02, 0.95, "a", fontsize=16, fontweight="bold")
    fig.text(0.45, 0.95, "b", fontsize=16, fontweight="bold")
    fig.suptitle("Feature leakage in TF ChIP-seq (AlphaGenome, K562): unrelated motifs "
                 "spuriously repress REST binding", fontsize=12, y=1.03)
    fig.savefig(OUT / "chip_leakage_panel_rest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "chip_leakage_panel_rest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(d.to_string(index=False))
    print("done ->", OUT / "chip_leakage_panel_rest.pdf")


if __name__ == "__main__":
    main()
