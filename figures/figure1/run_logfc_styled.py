#!/usr/bin/env python
"""Styled single-box OCT-SOX log2FC boxplots (Fig-1 look) for AlphaGenome & ChromBPNet.

Matches the compact reference style: one narrow gray box, black median, a "*" when
the one-sided Wilcoxon (repressive, median<0) is significant. Both use the same
per-locus log2 fold-change of predicted accessibility on a LOCAL window:
  * AlphaGenome: log2( pred_local(with motif) / pred_local(ablated) ), +/-~500 bp,
    from the arrays saved by run_alphagenome.py.
  * ChromBPNet:  (logcounts_with - logcounts_ablated)/ln2 over its 1000 bp output,
    from run_chrombpnet_logfc.py.
Negative = motif represses predicted accessibility (leakage direction).

Run: python tools/render_with_svg.py figures/figure1/run_logfc_styled.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"
EPS = 1e-3


def styled_box(logfc, label, outstem, ylim=None):
    p_less = stats.wilcoxon(logfc, alternative="less").pvalue
    p_two = stats.wilcoxon(logfc).pvalue
    fig, ax = plt.subplots(figsize=(2, 4))
    bp = ax.boxplot([logfc], tick_labels=[label], showfliers=False, widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor="lightgray", color="black"),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"), capprops=dict(color="black"))
    ax.axhline(0, ls="--", color="0.6")
    ax.set_ylabel("OCT-SOX effect  (log2FC, with motif / ablated)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if p_less < 0.05:
        ax.set_title("*", fontsize=16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{outstem}.pdf")
    plt.close(fig)
    print(f"{label}: n={len(logfc)} median log2FC={np.median(logfc):+.4f} "
          f"mean={logfc.mean():+.4f} | Wilcoxon repressive(one-sided) p={p_less:.3g} | "
          f"two-sided p={p_two:.3g} -> {outstem}.pdf")
    return p_less


def main() -> None:
    # Both models read ENCSR000EPO (fibroblast of dermis) and score the SAME OCT-SOX
    # sites: run_chrombpnet_logfc.py (ChromBPNet dermis) and run_alphagenome_torch.py
    # (AlphaGenome CL:0002551, 2,048 bp input: run with --seq-len 2048 --out-suffix _2kb).
    ag = np.load(OUT / "fig1_alphagenome_octsox_logfc_dermis_2kb.npy")
    cbp_path = OUT / "fig1_chrombpnet_octsox_logfc_dermis.npy"
    cbp = np.load(cbp_path) if cbp_path.exists() else None
    if cbp is not None:
        cbp = cbp[:len(ag)]          # same sites, same order (first n of the saved list)

    # shared y-limits from the WHISKER extent (fliers are hidden), so the boxes
    # are readable and not squashed by a few extreme-ratio outliers.
    def whisker_abs(d):
        q1, q3 = np.percentile(d, [25, 75]); iqr = q3 - q1
        lo = d[d >= q1 - 1.5 * iqr].min(); hi = d[d <= q3 + 1.5 * iqr].max()
        return max(abs(lo), abs(hi))
    m = whisker_abs(ag) if cbp is None else max(whisker_abs(ag), whisker_abs(cbp))
    m *= 1.15
    ylim = (-m, m)

    styled_box(ag, "AlphaGenome", "fig1_alphagenome_octsox_logfc_styled", ylim)
    if cbp is not None:
        styled_box(cbp, "ChromBPNet", "fig1_chrombpnet_octsox_logfc_styled", ylim)
    else:
        print("ChromBPNet log2FC not found; run run_chrombpnet_logfc.py first.")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
