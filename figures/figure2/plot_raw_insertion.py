#!/usr/bin/env python
"""Figure 2b + Supp 2a (raw OCT-SOX insertion effect, matched ~1 kb readout) and supplements.

Reads the CSVs from run_raw_alphagenome.py, run_raw_grelu.py (enformer, borzoi) and
run_raw_chrombpnet.py, and writes to outputs/:
  panelb_raw_insertion.pdf          Fig 2b: AlphaGenome vs ChromBPNet (fold 0)
  supplement_raw_insertion.pdf      Supp 2a: Enformer, Borzoi vs ChromBPNet (fold 0)
  supplement_real_vs_shuffles.pdf   ChromBPNet: real OCT-SOX vs shuffled insertions
  raw_insertion_summary.csv         medians, two-sided Wilcoxon P, paired tests vs ChromBPNet
Tests: two-sided Wilcoxon signed-rank vs zero (n = 120); paired two-sided Wilcoxon vs ChromBPNet.

Run (env: featleak-dl, no GPU):  python plot_raw_insertion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8,
                            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(__file__).parent / "outputs"
MT, ST = "#C44E52", "#4C72B0"
YLAB = "OCT–SOX insertion effect\n(log2FC, inserted / original)"


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def violin(data, labels, colors, title, fname, width):
    fig, ax = plt.subplots(figsize=(width, 3.2))
    pos = np.arange(len(data))
    parts = ax.violinplot(data, positions=pos, showextrema=False, widths=0.8)
    for b, c in zip(parts["bodies"], colors):
        b.set_facecolor(c); b.set_edgecolor("none"); b.set_alpha(0.85)
    top = max(np.max(d) for d in data)
    for i, d in enumerate(data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color="k", lw=4)
        ax.scatter(i, med, color="white", s=14, zorder=3, edgecolor="k", lw=0.4)
        ax.text(i + 0.12, med, f"{med:+.2f}", va="center", fontsize=6)
        ax.text(i, top + 0.25, stars(wilcoxon(d).pvalue), ha="center", color=colors[i], fontsize=8)
    ax.axhline(0, color="0.5", ls="--", lw=0.6)
    ax.set_ylim(top=top + 0.7)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    ax.set_ylabel(YLAB); ax.set_title(title, fontsize=7, pad=4)
    despine(ax); fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


def main() -> None:
    ag = pd.read_csv(OUT / "raw_insertion_alphagenome.csv").set_index("locus")
    enf = pd.read_csv(OUT / "raw_insertion_enformer.csv").set_index("locus")
    bor = pd.read_csv(OUT / "raw_insertion_borzoi.csv").set_index("locus")
    cbp_all = pd.read_csv(OUT / "raw_insertion_chrombpnet.csv")
    cbp = cbp_all[cbp_all.motif == "real"]
    loci = ag.index
    cbp0 = cbp.set_index("locus").lfc_total.loc[loci]

    violin([ag.lfc_1kb.values, cbp0.values],
           ["AlphaGenome\n(multi-task)", "ChromBPNet\n(single-task)"], [MT, ST],
           "OCT–SOX insertion lowers AlphaGenome's\npredicted accessibility; ChromBPNet's is unchanged",
           "panelb_raw_insertion.pdf", 3.2)
    violin([enf.lfc_1kb.loc[loci].values, bor.lfc_1kb.loc[loci].values, cbp0.values],
           ["Enformer", "Borzoi", "ChromBPNet"], [MT, MT, ST],
           "Enformer and Borzoi also lower predicted accessibility\nafter OCT–SOX insertion; ChromBPNet does not",
           "supplement_raw_insertion.pdf", 3.8)
    # real OCT-SOX vs shuffled insertions (ChromBPNet, old central readout)
    if cbp_all.motif.nunique() > 1:
        m = cbp_all.groupby("motif").lfc_central.median().sort_values()
        n_less = int((m.drop("real") > m["real"]).sum())
        fig, ax = plt.subplots(figsize=(6, 3.2))
        cols = [MT if k == "real" else ("#E1A75B" if k.startswith("orig") else "#9AA5B1") for k in m.index]
        ax.bar(range(len(m)), m.values, color=cols, width=0.85)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks([])
        ax.set_xlabel("inserted 14-mer (4 copies), sorted")
        ax.set_ylabel("median log2FC, central accessibility\n(120 enhancers)", fontsize=7)
        ax.set_title(f"ChromBPNet: real OCT–SOX is less disruptive than {len(m) - 1 - n_less} "
                     f"of {len(m) - 1} shuffled insertions", fontsize=8)
        ax.legend(handles=[Patch(color=MT, label="OCT–SOX"),
                           Patch(color="#E1A75B", label="3 shuffles of the original Fig 2b control"),
                           Patch(color="#9AA5B1", label="new shuffles")],
                  fontsize=6, frameon=False, loc="lower right")
        despine(ax); fig.tight_layout(); fig.savefig(OUT / "supplement_real_vs_shuffles.pdf"); plt.close(fig)

    rows = []
    for name, s in [("AlphaGenome, central 1,024 bp", ag.lfc_1kb),
                    ("AlphaGenome, central 300 bp (old readout)", ag.lfc_300bp),
                    ("Enformer, central 1,024 bp", enf.lfc_1kb),
                    ("Enformer, +/-300 bp (old readout)", enf.lfc_300bp),
                    ("Borzoi, central 1,024 bp", bor.lfc_1kb),
                    ("Borzoi, +/-300 bp (old readout)", bor.lfc_300bp),
                    ("ChromBPNet, total counts", cbp0)]:
        v = s.loc[loci].values
        rows.append(dict(comparison=name, n=len(v), median=np.median(v), p_two_sided=wilcoxon(v).pvalue))
    for name, s in [("AlphaGenome", ag.lfc_1kb), ("Enformer", enf.lfc_1kb), ("Borzoi", bor.lfc_1kb)]:
        v = s.loc[loci].values
        rows.append(dict(comparison=f"paired: {name} 1 kb vs ChromBPNet", n=len(v),
                         median=np.median(v - cbp0.values), p_two_sided=wilcoxon(v, cbp0.values).pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "raw_insertion_summary.csv", index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
