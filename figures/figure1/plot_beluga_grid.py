#!/usr/bin/env python
"""Plot Figure 1 E-H from run_beluga_grid.py outputs.

One PDF per motif column (four stacked method rows) plus a full 4x4 grid PDF.
Each point is the mean effect over the hits of one affinity quartile for one
Beluga DNase task; tasks are sorted by the strongest-quartile (Q4) mean.

Options:
    --hit-set genome|ccre     which hit sample to plot
    --band none|sd|ci95       shading around the Q4 means (+/-1 SD across hits,
                              or the 95% CI of the mean)
    --mark-sig                rug under the axis for tasks whose Q4 effect is
                              significantly < 0 (Wilcoxon signed-rank, BH FDR<0.05)

Run (any env with numpy/scipy/matplotlib):
    python plot_beluga_grid.py --hit-set genome
"""

from __future__ import annotations

import argparse
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
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "outputs" / "beluga_grid"

MOTIFS = [("OCT-SOX", "OCTSOX", "Blues"), ("AP-1", "AP1", "Reds"),
          ("HNF4", "HNF4", "Greens"), ("CTCF", "CTCF", "Purples")]
METHODS = [("deeplift", "DeepLIFT/SHAP"), ("ism", "Saturation\nmutagenesis"),
           ("ablation", "Ablation"), ("marginalization", "Marginalization")]
QLABELS = ["1st quartile (weakest)", "2nd quartile", "3rd quartile",
           "4th quartile (strongest)"]


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1)
    return out


def nice_ticks(lo, hi):
    """Three y ticks: rounded minimum (if clearly negative), 0 and rounded max."""
    def rnd(v):
        if v == 0:
            return 0.0
        mag = 10 ** np.floor(np.log10(abs(v)))
        return np.sign(v) * np.floor(abs(v) / mag * 2) / 2 * mag
    ticks = [0.0, rnd(hi)]
    if lo < -0.2 * hi:          # only label the minimum when it will not collide
        ticks.insert(0, rnd(lo))
    return sorted(set(ticks))


def masked_mean(x, m):
    return (x * m).sum(0) / np.maximum(m.sum(0), 1)


def draw(ax, eff, quart, cmap, band, mark_sig, show_n=True, mask=None):
    """eff: (n_hits, n_tasks); quart: (n_hits,) in 0..3.

    mask: optional (n_hits, n_tasks) bool, hits to use for each task.
    """
    if mask is None:
        mask = np.ones_like(eff, dtype=bool)
    means = [masked_mean(eff[quart == k], mask[quart == k]) for k in range(4)]
    q4 = np.where(mask[quart == 3], eff[quart == 3], np.nan)
    order = np.argsort(means[3])[::-1]
    x = np.arange(len(order))
    colors = [plt.get_cmap(cmap)(0.35 + 0.2 * k) for k in range(4)]

    if band != "none":
        sd = np.nanstd(q4, 0, ddof=1)
        n_eff = np.sum(~np.isnan(q4), 0)
        half = sd if band == "sd" else 1.96 * sd / np.sqrt(n_eff)
        m = means[3][order]
        ax.fill_between(x, m - half[order], m + half[order], color=colors[3],
                        alpha=0.18, lw=0, zorder=1)
    ax.axhline(0, color="0.35", lw=0.6, zorder=2)
    for k in range(4):
        ax.scatter(x, means[k][order], s=1.6, color=colors[k], lw=0, zorder=3 + k,
                   label=QLABELS[k], rasterized=False)

    # y range from the means (the SD band may exceed it and is clipped)
    lo = min(m.min() for m in means); hi = max(m.max() for m in means)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad * (2.5 if mark_sig else 1), hi + pad)
    ax.set_yticks(nice_ticks(lo, hi))
    ax.set_xlim(-8, len(order) + 8)
    ax.set_xticks([0, 100, 200, 300])

    p = np.array([stats.wilcoxon(c[~np.isnan(c)]).pvalue for c in q4.T])
    q = bh(p)
    neg = (q < 0.05) & (means[3] < 0)
    if mark_sig:
        y0 = lo - pad * 1.9
        xs = np.where(neg[order])[0]
        ax.plot(xs, np.full(len(xs), y0), "|", ms=2.2, mew=0.5,
                color=colors[3], zorder=6)
    if show_n:
        ax.text(0.98, 0.93, f"{neg.sum()}/{len(neg)} cell types\nQ4 < 0 (FDR<0.05)",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                color="0.25")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return neg.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-set", default="genome", choices=["genome", "ccre"])
    ap.add_argument("--band", default="none", choices=["none", "sd", "ci95"])
    ap.add_argument("--mark-sig", action="store_true")
    ap.add_argument("--no-counts", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--file-tag", default="", help="data file suffix, e.g. _big")
    ap.add_argument("--open-frac", type=float, default=None,
                    help="per task, keep only the hits in the top fraction of predicted "
                         "accessibility WITHOUT the motif (baseline - ablation effect)")
    args = ap.parse_args()

    tag = args.tag or f"{args.hit_set}_{args.band}{'_sig' if args.mark_sig else ''}"
    out = DATA / "plots"
    out.mkdir(parents=True, exist_ok=True)
    data = {m[1]: np.load(DATA / f"beluga_{m[1]}_{args.hit_set}{args.file_tag}.npz")
            for m in MOTIFS}
    masks = {}
    for key, d in data.items():
        if args.open_frac is None:
            masks[key] = None
            continue
        free = d["baseline"] - d["ablation"]          # logit with the motif shuffled away
        m = np.zeros_like(free, dtype=bool)
        for k in range(4):
            idx = d["quartile"] == k
            thr = np.quantile(free[idx], 1 - args.open_frac, axis=0)
            m[idx] = free[idx] >= thr
        masks[key] = m
    set_label = {"genome": "genome-wide FIMO hits",
                 "ccre": "FIMO hits in ENCODE cCREs"}[args.hit_set]
    if args.open_frac is not None:
        set_label += (f"; per cell type, top {args.open_frac:.0%} of hits by predicted "
                      "accessibility without the motif")

    ylab = "\u0394 logit"
    # --- full 4x4 grid -----------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 5.6), sharex=True)
    summary = []
    for j, (name, key, cmap) in enumerate(MOTIFS):
        d = data[key]
        for i, (meth, mlabel) in enumerate(METHODS):
            ax = axes[i, j]
            n = draw(ax, d[meth], d["quartile"], cmap, args.band, args.mark_sig,
                     show_n=not args.no_counts, mask=masks[key])
            summary.append((name, meth, n))
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{mlabel}\n({ylab})", fontsize=6.5)
            if i == 3:
                ax.set_xlabel("Cell type (sorted by Q4 effect)", fontsize=6.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6,
               markerscale=3, bbox_to_anchor=(0.5, -0.005))
    band_txt = {"none": "", "sd": "; shading: Q4 mean \u00b1 1 SD across hits",
                "ci95": "; shading: Q4 95% CI"}[args.band]
    fig.suptitle(f"Beluga motif effects across 334 DNase tasks ({set_label}, "
                 f"{int((data[MOTIFS[0][1]]['quartile'] == 3).sum())} hits/quartile{band_txt})", fontsize=7.5, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=0.6, w_pad=0.8)
    fig.savefig(out / f"fig1_EH_grid_{tag}.pdf")
    fig.savefig(out / f"fig1_EH_grid_{tag}.png", dpi=220)
    plt.close(fig)

    # --- one PDF per motif column (panels e-h) -------------------------------
    for j, (name, key, cmap) in enumerate(MOTIFS):
        d = data[key]
        fig, axes = plt.subplots(4, 1, figsize=(1.9, 5.0), sharex=True)
        for i, (meth, mlabel) in enumerate(METHODS):
            draw(axes[i], d[meth], d["quartile"], cmap, args.band, args.mark_sig,
                 show_n=not args.no_counts, mask=masks[key])
            axes[i].set_ylabel(f"{mlabel}\n({ylab})", fontsize=6)
        axes[0].set_title(name, fontsize=8)
        axes[-1].set_xlabel("Cell type", fontsize=6.5)
        fig.tight_layout(h_pad=0.5)
        fig.savefig(out / f"fig1_{'efgh'[j]}_{key}_{tag}.pdf")
        plt.close(fig)

    with open(out / f"summary_{tag}.tsv", "w") as fh:
        fh.write("motif\tmethod\tn_celltypes_Q4_sig_negative\n")
        for r in summary:
            fh.write("\t".join(map(str, r)) + "\n")
    print("wrote", out, tag)


if __name__ == "__main__":
    main()
