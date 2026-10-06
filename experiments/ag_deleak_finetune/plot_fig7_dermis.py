#!/usr/bin/env python
"""Figure 7 panels b-d from finetune_dermis_torch.py outputs (dermis-matched, full context).

  b  fidelity: Pearson r vs observed ENCFF302JEV, held-out chr8/9 16 kb windows
  c  ISM at the held-out OCT-SOX example, before vs after fine-tuning
  d  log2FC after a single OCT-SOX insertion (held-out chr8/9 enhancers):
     AlphaGenome before, after, and the ChromBPNet referee

Run (any env with matplotlib + logomaker):  python plot_fig7_dermis.py
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
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.6})
import matplotlib.pyplot as plt  # noqa: E402
import logomaker  # noqa: E402

D = Path(__file__).parent / "outputs" / "dermis_torch"
PRE, FT, CBP = "#C44E52", "#8E6FB0", "#4C72B0"


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_b():
    f = pd.read_csv(D / "panel_b_fidelity.csv")
    m = [f.pearson_pre.mean(), f.pearson_ft.mean()]
    fig, ax = plt.subplots(figsize=(2.0, 2.4))
    bars = ax.bar(["AlphaGenome", "AlphaGenome\nfine-tuned"], m, color=[PRE, FT],
                  width=0.6, edgecolor="0.2", linewidth=0.5)
    for b, v in zip(bars, m):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0, max(m) * 1.2)
    ax.set_ylabel("Pearson r vs observed DNase")
    ax.set_title(f"Held-out chr8/9 ({len(f)} x 16 kb)\nobserved ENCFF302JEV", fontsize=7)
    despine(ax); fig.tight_layout()
    fig.savefig(D / "fig7_b_fidelity.pdf"); fig.savefig(D / "fig7_b_fidelity.png", dpi=250)
    plt.close(fig)
    return m


def logo_matrix(vals, ref):
    oh = np.zeros_like(vals)
    oh[np.arange(len(ref)), ["ACGT".index(c) for c in ref]] = 1
    return (vals - vals.mean(1, keepdims=True)) * oh


def panel_c(npz="panel_c_ism.npz", stem="fig7_c_ism"):
    d = np.load(D / npz)
    ref, chrom, start = str(d["ref"]), str(d["chrom"]), int(d["start"])
    mats = [logo_matrix(d["pre"], ref), logo_matrix(d["ft"], ref)]
    lo = min(m.min() for m in mats) * 1.1; hi = max(m.max() for m in mats) * 1.1
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 1.9), sharex=True)
    for ax, m, lab in zip(axes, mats, ["AlphaGenome", "AlphaGenome\nfine-tuned"]):
        logomaker.Logo(pd.DataFrame(m, columns=list("ACGT")), ax=ax, color_scheme="classic")
        ax.axhline(0, color="0.5", lw=0.5); ax.set_ylim(lo, hi)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center")
        ax.set_xticks([]); despine(ax); ax.spines["bottom"].set_visible(False)
    ins = bool(d["inserted"]) if "inserted" in d.files else False
    endo = bool(d["endogenous"]) if "endogenous" in d.files else False
    if ins or endo:
        ms = int(d["motif_start"])
        lo_ = axes[-1].get_ylim()[0]
        axes[-1].plot([ms - 0.5, ms + 13.5], [lo_, lo_], color="k", lw=1.5,
                      clip_on=False, solid_capstyle="butt")
        axes[-1].text(ms + 6.5, lo_ - 0.04 * (hi - lo),
                      "endogenous OCT-SOX" if endo else "inserted OCT-SOX", ha="center",
                      va="top", fontsize=6)
    what = "endogenous OCT-SOX, " if endo else "OCT-SOX inserted, " if ins else ""
    axes[0].set_title(f"ISM, {what}{chrom}:{start + 1:,}-{start + len(ref):,} (held out)",
                      fontsize=7)
    fig.tight_layout()
    fig.savefig(D / f"{stem}.pdf"); fig.savefig(D / f"{stem}.png", dpi=250)
    plt.close(fig)


def panel_d():
    t = pd.read_csv(D / "panel_d_heldout.csv")
    data = [t.ag_pre.values, t.ag_ft.values, t.cbp_lfc.values]
    fig, ax = plt.subplots(figsize=(2.6, 2.8))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.2))
    for patch, c in zip(bp["boxes"], [PRE, FT, CBP]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.axhline(0, ls=":", color="0.4", lw=0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["AlphaGenome", "AlphaGenome\nfine-tuned", "ChromBPNet\n(referee)"],
                       fontsize=7)
    ax.set_ylabel("log2FC after OCT-SOX insertion")
    ax.set_title(f"{len(t)} held-out chr8/9 dermis enhancers", fontsize=7)
    despine(ax); fig.tight_layout()
    fig.savefig(D / "fig7_d_octsox_box.pdf"); fig.savefig(D / "fig7_d_octsox_box.png", dpi=250)
    plt.close(fig)
    return {k: float(np.median(v)) for k, v in zip(["pre", "ft", "cbp"], data)}, \
        stats.wilcoxon(t.ag_pre, t.ag_ft).pvalue


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ism-npz", default="panel_c_ism.npz", help="panel c ISM npz in outputs/")
    ap.add_argument("--ism-stem", default="fig7_c_ism", help="output basename for panel c")
    a = ap.parse_args()
    m = panel_b(); panel_c(a.ism_npz, a.ism_stem); med, p = panel_d()
    print(f"b: Pearson r {m[0]:.3f} -> {m[1]:.3f}")
    print(f"d: medians pre {med['pre']:+.3f}, fine-tuned {med['ft']:+.3f}, "
          f"ChromBPNet {med['cbp']:+.3f}; paired Wilcoxon p={p:.2g}")
    print("done ->", D)


if __name__ == "__main__":
    main()
