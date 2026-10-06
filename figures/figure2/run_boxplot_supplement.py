#!/usr/bin/env python
"""Figure 2 supplement: OCT-SOX effect for Enformer / Borzoi vs matched ChromBPNet.

Same shuffle-corrected SPECIFICITY (log2FC(real) - log2FC(shuffled)) as panel b, for
the two other multi-task models (Enformer, Borzoi; canonical fibroblast-of-dermis DNase
task ENCFF302JEV) against the EXPERIMENT-MATCHED ChromBPNet dermis model (ENCSR000EPO).
Enformer/Borzoi show OCT-SOX-specific repression (leakage); matched ChromBPNet does not.

Reads outputs/grelu_{enformer,borzoi}_canonical.npz (keys real, shuf) and
outputs/cbp_octsox_specificity_dermis.csv. Run (env: featleak-dl, no GPU):
    python run_boxplot_supplement.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                           "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt  # noqa: E402

from run_boxplot import _violin, MULTI, CBP  # shared violin helper

OUT = Path(__file__).parent / "outputs"


def main() -> None:
    enf = np.load(OUT / "grelu_enformer_canonical.npz")
    bor = np.load(OUT / "grelu_borzoi_canonical.npz")
    cbp = pd.read_csv(OUT / "cbp_octsox_specificity_dermis_matched.csv").specificity.values
    series = [("Enformer\n(dermis)", enf["real"] - enf["shuf"], MULTI),
              ("Borzoi\n(dermis)", bor["real"] - bor["shuf"], MULTI),
              ("ChromBPNet\n(dermis)", cbp, CBP)]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _violin(ax, series, "OCT-SOX effect  (shuffle-corrected log2FC)",
            "OCT-SOX insertion: Enformer / Borzoi vs matched ChromBPNet")
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"supplement_octsox_specificity.{ext}", dpi=150 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    for name, d, _ in series:
        from scipy import stats
        p = stats.wilcoxon(d, alternative="less").pvalue
        print(f"{name.splitlines()[0]}: n={len(d)} median={np.median(d):+.4f} Wilcoxon(<0) p={p:.3g}")
    print("done ->", OUT / "supplement_octsox_specificity.pdf")


if __name__ == "__main__":
    main()
