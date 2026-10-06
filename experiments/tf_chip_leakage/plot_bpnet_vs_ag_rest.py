#!/usr/bin/env python
"""REST ChIP leakage summary: AlphaGenome (K562 REST track) vs ENCODE REST BPNet models.

For the four motifs that repress AlphaGenome's REST track, plots the median log2FC
of predicted REST ChIP after inserting the motif (2 copies, +-120 bp from a central
REST motif): AlphaGenome as a bar, the four single-task REST BPNet models (one per
K562 REST ChIP-seq experiment) as dots (filled = the two experiments AlphaGenome's
REST tracks were trained on) shown individually.

Reads outputs/rest_leakage_logfc.csv and outputs/bpnet_rest/bpnet_rest_leakage.csv.
Run (any env with matplotlib): python plot_bpnet_vs_ag_rest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                            "font.size": 8, "axes.linewidth": 0.6})
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"
LEAK = [("NRF1", "NRF1"), ("YY1", "YY1"), ("E-box", "MYC"), ("CTCF", "CTCF")]
AG_LABEL = {"E-box": "E-box (MYC/MAX)"}
AG_COL, BP_COL, BP_BAR = "#C44E52", "#2B5C8A", "#C9D9EA"
# ENCODE experiments behind AlphaGenome's two K562 REST tracks (metadata: paired-end;
# one unmodified, one CRISPR eGFP-tagged REST). The other two are single-end.
MATCHED = ["ENCSR137ZMQ", "ENCSR054JMQ"]


def main():
    ag = pd.read_csv(OUT / "rest_leakage_logfc.csv").set_index("leaker")
    bp = pd.read_csv(OUT / "bpnet_rest" / "bpnet_rest_leakage.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    h = 0.34
    for i, (name, key) in enumerate(LEAK):
        y = len(LEAK) - 1 - i
        a = ag.loc[AG_LABEL.get(name, name), "lfc_real"]
        q = bp[bp["query"] == key]
        m = q[q.experiment.isin(MATCHED)].lfc_real.values
        u = q[~q.experiment.isin(MATCHED)].lfc_real.values
        ax.barh(y + h / 2, a, h, color=AG_COL)
        ax.scatter(m, np.full(len(m), y - h / 2), s=12, color=BP_COL, zorder=3, lw=0)
        ax.scatter(u, np.full(len(u), y - h / 2), s=12, facecolor="white", edgecolor=BP_COL,
                   zorder=3, lw=0.7)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_yticks(range(len(LEAK)))
    ax.set_yticklabels([f"{n} motif" for n, _ in LEAK][::-1])
    ax.set_xlabel("REST ChIP log2 fold-change\n(motif inserted twice beside a REST motif)")
    # direct labels instead of a legend
    y0 = len(LEAK) - 1
    ax.text(-0.008, y0 + h / 2, "AlphaGenome", ha="right", va="center", fontsize=7,
            color="white", fontweight="bold")
    ax.text(0.03, y0 - h / 2, "REST BPNet, same experiments\n(filled; open = other K562 REST)",
            ha="left", va="center", fontsize=6.5, color=BP_COL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "bpnet_rest" / "rest_leakage_ag_vs_bpnet.pdf")
    fig.savefig(OUT / "bpnet_rest" / "rest_leakage_ag_vs_bpnet.png", dpi=300)
    print("done")


if __name__ == "__main__":
    main()
