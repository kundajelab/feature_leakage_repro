#!/usr/bin/env python
"""Plot ISM logos at an endogenous OCT-SOX site for the original and fine-tuned heads (Figure 7c).

Outputs (outputs/dermis_torch/): fig7_c_ism_endo_<chrom>_<mid>.{pdf,png}.

Run (env: featleak-dl):
    python plot_fig7c_endo.py --site chr9:92379377
"""

from __future__ import annotations

import argparse
from pathlib import Path

import logomaker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = Path(__file__).resolve().parent / "outputs" / "dermis_torch" / "endo_screen"
ML = 14
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                                         "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 7})


def contributions(v, ref, strand):
    c = v - v.mean(1, keepdims=True)
    oh = np.zeros_like(c)
    oh[np.arange(len(ref)), ["ACGT".index(b) for b in ref.upper()]] = 1
    c = c * oh
    return c[::-1, ::-1] if strand == "-" else c       # reverse positions, ACGT -> TGCA


def main(site):
    chrom, mid = site.split(":"); mid = int(mid)
    d = np.load(D / f"ism_{chrom}_{mid}.npz")
    s = pd.read_csv(D / "screen.csv").set_index(["chrom", "mid"]).loc[(chrom, mid)]
    ref, fl, start = str(d["ref"]), int(d["flank"]), int(d["start"])
    mats = [contributions(d[k], ref, s.strand) for k in ("pre", "ft")]
    lo = min(m.min() for m in mats) * 1.1; hi = max(max(m.max() for m in mats) * 1.1, 0.1)

    fig, axes = plt.subplots(2, 1, figsize=(2.6, 2.0), sharex=True)
    for ax, m, lab in zip(axes, mats, ["AlphaGenome", "AlphaGenome fine-tuned"]):
        logomaker.Logo(pd.DataFrame(m, columns=list("ACGT")), ax=ax, color_scheme="classic")
        ax.axhline(0, color="0.5", lw=0.5); ax.set_ylim(lo, hi)
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, ha="left", va="bottom")
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
    y = axes[-1].get_ylim()[0]
    axes[-1].plot([fl - 0.5, fl + ML - 0.5], [y, y], color="k", lw=1.5, clip_on=False,
                  solid_capstyle="butt")
    axes[-1].text(fl + ML / 2 - 0.5, y - 0.06 * (hi - lo), "OCT–SOX", ha="center", va="top")
    fig.suptitle(f"{chrom}:{start + 1:,}–{start + len(ref):,}", fontsize=7)
    fig.supylabel("ISM contribution (log2)", fontsize=7)
    fig.tight_layout()
    stem = D.parent / f"fig7_c_ism_endo_{chrom}_{mid}"
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=300)
    print(f"{stem}.pdf  ({s.strand}, {s.site}, {s.distance} bp from summit; ablation "
          f"{s.eff1k_pre:+.2f} -> {s.eff1k_ft:+.2f} log2)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="chr9:92379377")
    main(ap.parse_args().site)
