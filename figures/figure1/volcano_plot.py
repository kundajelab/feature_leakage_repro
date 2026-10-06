"""Shared volcano plot of OCT-SOX marginalization slopes across DNase tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

GREEN, ORANGE, NS = "#2E8B57", "#E8853A", "#B7C0CC"
STEM, FIB, LINE = "#7B3FA0", "#111111", "#3B6FB0"
STEM_RE = r"(stem|ipsc|ips[\s\-]|ips df|embryonic|pluripot|hesc|hues|nt2|\bh[179]\b)"


def volcano(df: pd.DataFrame, title: str, outpath, alpha_fdr: float = 0.05,
            xlabel: str = "OCT-SOX effect per insertion  (log2FC)"):
    y = -np.log10(df.pval.clip(lower=1e-300).values)
    slope = df.slope.values
    sig = df.qval.values < alpha_fdr
    samp = df["sample"].astype(str).str.lower()
    stem = samp.str.contains(STEM_RE, regex=True).values
    fib = samp.str.contains("fibroblast").values
    rep, pos, ns = sig & (slope < 0), sig & (slope > 0), ~sig
    other = ~stem & ~fib                      # plain points (no special marker)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(slope[ns & other], y[ns & other], s=11, c=NS, alpha=0.45,
               edgecolor="none", zorder=1, rasterized=True)
    ax.scatter(slope[rep & other], y[rep & other], s=13, c=GREEN, alpha=0.55,
               edgecolor="none", zorder=2, rasterized=True)
    ax.scatter(slope[pos & other], y[pos & other], s=13, c=ORANGE, alpha=0.6,
               edgecolor="none", zorder=2, rasterized=True)
    # fibroblast: black-ringed circles coloured by their significance
    fc = np.where(rep[fib], GREEN, np.where(pos[fib], ORANGE, NS))
    ax.scatter(slope[fib], y[fib], s=42, c=fc, edgecolor=FIB, linewidth=1.0, zorder=5)
    # stem/ESC/iPSC: purple stars
    ax.scatter(slope[stem], y[stem], s=90, marker="*", c=STEM, edgecolor="white",
               linewidth=0.4, zorder=6)

    ax.axvline(0, ls="--", color=LINE, lw=1)
    thr = -np.log10(df.pval[sig].max()) if sig.any() else -np.log10(alpha_fdr)
    ax.axhline(thr, ls=":", color=LINE, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"significance  $-\log_{10}(p)$")
    ax.set_title(title, fontsize=11, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # direction cue under the x-axis
    xl = ax.get_xlim()
    ax.text(xl[0] + 0.02 * (xl[1] - xl[0]), -0.19, "← represses",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color=GREEN,
            ha="left", va="top")
    ax.text(xl[1] - 0.02 * (xl[1] - xl[0]), -0.19, "activates →",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color=ORANGE,
            ha="right", va="top")

    handles = [
        Line2D([], [], marker="o", ls="", mfc=GREEN, mec="none", ms=7,
               label=f"repressive  ({int(rep.sum())})"),
        Line2D([], [], marker="o", ls="", mfc=ORANGE, mec="none", ms=7,
               label=f"positive  ({int(pos.sum())})"),
        Line2D([], [], marker="o", ls="", mfc=NS, mec="none", ms=7, label="n.s."),
        Line2D([], [], marker="*", ls="", mfc=STEM, mec="white", ms=12,
               label=f"stem/ESC/iPSC  ({int(stem.sum())})"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=FIB, ms=8, mew=1.2,
               label=f"fibroblast  ({int(fib.sum())})"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9, handletextpad=0.4, labelspacing=0.7)

    fig.tight_layout()
    fig.savefig(Path(outpath), bbox_inches="tight", dpi=200)
    fig.savefig(Path(outpath).with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
