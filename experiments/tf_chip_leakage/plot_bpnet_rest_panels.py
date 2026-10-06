#!/usr/bin/env python
"""REST-BPNet version of the AlphaGenome REST ChIP leakage figure (same layout).

  a  REST ChIP log2FC after inserting each motif (real vs shuffled), same x-range
     as the AlphaGenome panel so bars are directly comparable
  b  ISM -> REST ChIP: REST's own motif (on-target) and the four motifs that leak
     into AlphaGenome's REST track; leaking-motif logos share one y-scale
  c  REST ChIP profile with NRF1 / YY1 inserted (real vs shuffled)

Writes the combined figure plus one PDF per panel (Illustrator-editable text).
Run (env: featleak-dl):  python plot_bpnet_rest_panels.py --experiment ENCSR000BMW
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tangermeme.plot import plot_logo

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
    "font.size": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "bpnet_rest"
AG_CSV = Path(__file__).parent / "outputs" / "rest_leakage_logfc.csv"
REAL, SHUF, ON = "#C44E52", "#9AA0A8", "#2E8B57"
LEAKERS = [("NRF1", "NRF1"), ("YY1", "YY1"), ("E-box (MYC/MAX)", "MYC"), ("CTCF", "CTCF")]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def panel_a(ax, df, xlim):
    d = df.set_index("query")
    labels = [n for n, _ in LEAKERS]
    real = np.array([d.loc[k, "lfc_real"] for _, k in LEAKERS])
    shuf = np.array([d.loc[k, "lfc_shuf"] for _, k in LEAKERS])
    pv = np.array([d.loc[k, "p_vs_shuffle"] for _, k in LEAKERS])
    order = np.argsort(real)
    y = np.arange(len(order)); h = 0.38
    ax.barh(y + h / 2, real[order], h, color=REAL, label="real motif")
    ax.barh(y - h / 2, shuf[order], h, color=SHUF, label="shuffled motif")
    ax.axvline(0, color="0.2", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([labels[i] + " → REST" for i in order], fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_xlabel("REST ChIP  log2 fold-change")
    ax.set_title("Effect of each motif on REST", fontsize=11, pad=8)
    ax.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.08))


def logo(ax, arr, title, color, ylim=None):
    plot_logo(torch.tensor(arr, dtype=torch.float32), ax=ax)
    ax.set_title(title, fontsize=9, color=color, pad=2)
    ax.set_xticks([])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["bottom"].set_visible(False)


def panel_b(axes, z):
    leak = [z[f"ism_{k}"] for _, k in LEAKERS]
    m = max(np.abs(a.sum(0)).max() for a in leak) * 1.15
    lims = (-m, m)
    logo(axes[0], z["ism_REST"], "REST motif → REST ChIP  (on-target)", ON)
    for ax, (name, _), arr in zip(axes[1:], LEAKERS, leak):
        logo(ax, arr, f"{name} motif → REST ChIP", REAL, lims)


def panel_c(ax, z, key, xb):
    ax.plot(xb, z["prof_base"], color="#222", lw=1.2, label="REST motif only")
    ax.plot(xb, z[f"prof_{key}_real"], color=REAL, lw=1.2, label=f"+ {key} (real)")
    ax.plot(xb, z[f"prof_{key}_shuf"], color=SHUF, lw=1.0, ls="--", label=f"+ {key} (shuffled)")
    ax.set_title(f"REST ChIP profile — {key} insertion", fontsize=9.5, pad=3)
    ax.set_ylabel("pred. REST ChIP (counts/bp)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    e = args.experiment
    df = pd.read_csv(OUT / "bpnet_rest_leakage.csv")
    df = df[df.experiment == e]
    z = np.load(OUT / f"bpnet_rest_panels_{e}.npz")
    ag = pd.read_csv(AG_CSV)
    lo = min(ag.lfc_real.min(), ag.lfc_shuf.min(), -0.03) * 1.1
    lk = df[df["query"].isin([k for _, k in LEAKERS])]
    hi = max(ag.lfc_shuf.max(), ag.lfc_real.max(), lk.lfc_real.max(), lk.lfc_shuf.max(),
             0.03) * 1.15
    xlim = (lo, hi)
    xb = np.arange(len(z["prof_base"])) - len(z["prof_base"]) // 2

    # ---- combined figure, same layout as the AlphaGenome panel ----
    fig = plt.figure(figsize=(15, 6.6))
    gs = GridSpec(5, 3, figure=fig, width_ratios=[1.05, 1.15, 1.0], wspace=0.32, hspace=0.75)
    panel_a(fig.add_subplot(gs[:, 0]), df, xlim)
    panel_b([fig.add_subplot(gs[r, 1]) for r in range(5)], z)
    a1 = fig.add_subplot(gs[0:2, 2]); panel_c(a1, z, "NRF1", xb)
    a2 = fig.add_subplot(gs[3:5, 2]); panel_c(a2, z, "YY1", xb)
    a2.set_xlabel("position relative to centre (bp)", fontsize=8)
    for x, l in ((0.02, "a"), (0.365, "b"), (0.68, "c")):
        fig.text(x, 0.965, l, fontsize=16, fontweight="bold")
    fig.savefig(OUT / f"bpnet_rest_panel_{e}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"bpnet_rest_panel_{e}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- separate editable panels ----
    f, ax = plt.subplots(figsize=(5, 6.6)); panel_a(ax, df, xlim); f.tight_layout()
    f.savefig(OUT / f"bpnet_rest_{e}_a_logfc.pdf"); plt.close(f)
    f, axs = plt.subplots(5, 1, figsize=(5.5, 6.6)); panel_b(axs, z); f.tight_layout()
    f.savefig(OUT / f"bpnet_rest_{e}_b_ism.pdf"); plt.close(f)
    f, axs = plt.subplots(2, 1, figsize=(4.8, 6.6))
    panel_c(axs[0], z, "NRF1", xb); panel_c(axs[1], z, "YY1", xb)
    axs[1].set_xlabel("position relative to centre (bp)", fontsize=8); f.tight_layout()
    f.savefig(OUT / f"bpnet_rest_{e}_c_profiles.pdf"); plt.close(f)
    print("done", e)


if __name__ == "__main__":
    main()
