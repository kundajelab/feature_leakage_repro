#!/usr/bin/env python
"""Figure 4C + supplement (d,e): single-task vs multi-task spurious sensitivity.

Trains CNNs to predict AP1 (FOS) ChIP-seq peaks in fibroblast under increasing
proportions of random background regions, and measures how strongly an OCT4
(OCT-SOX) motif -- which should be irrelevant to AP1 in fibroblast -- affects the
AP1 prediction. The multi-task model (jointly predicting OCT4 peaks) is more
sensitive to OCT-SOX than the single-task model: this is the feature-leakage
signature of union-of-peaks training.

  * Panel C: split-violin of OCT-SOX effect on AP1 prediction, single vs
    multi-task, across background proportions {0.0, 0.2, 0.4, 0.6}.
  * Supp (d,e): AUC bar charts vs background proportion.

Outputs (outputs/): single_vs_multi_split_violin.pdf,
    single_task_auc_barplot.pdf, multi_task_auc_barplot.pdf, fig4_effects.pkl.

Run (env: featleak-dl):
    python run_experiment.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pybedtools  # noqa: E402
import seaborn as sns  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from helpers_3 import merge_bed_files_and_label  # noqa: E402
from helpers_4 import (  # noqa: E402
    train_multitask_model, train_singletask_model,
    test_motif_effect_multitask, test_motif_effect_singletask,
)

OUT = Path(__file__).parent / "outputs"
PROPORTIONS = [0.0, 0.2, 0.4, 0.6]


def build_data():
    hg38 = str(config.resolve("hg38_fasta", must_exist=True))
    fos = str(config.resolve("chip_fos_fibroblast", must_exist=True))
    oct4 = str(config.resolve("chip_oct4_hesc", must_exist=True))
    peaks, labels = merge_bed_files_and_label(fos, oct4, peak_size=300)
    peaks.to_csv(OUT / "labeled_peaks.bed", sep="\t", header=False, index=False)

    active = pybedtools.BedTool(str(OUT / "labeled_peaks.bed"))
    bg_parts = []
    for key in ("gc_neg_fos", "gc_neg_oct4"):
        bt = pybedtools.BedTool(str(config.resolve(key, must_exist=True)))
        bg_parts.append(bt.window(active, w=1000, v=True).to_dataframe())
    background = pd.concat(bg_parts)[["chrom", "start", "end"]]
    return hg38, peaks, labels, background


def main() -> None:
    OUT.mkdir(exist_ok=True)
    hg38, peaks, labels, background = build_data()
    print(f"peaks={peaks.shape[0]}  background={background.shape[0]}")

    bg_loci, _ = extract_loci(background.sample(n=1000), hg38, in_window=300,
                              return_filtered=True, verbose=True)

    effects_multi, effects_single, auc_multi, auc_single = {}, {}, {}, {}
    for p in PROPORTIONS:
        print(f"[multi]  proportion={p}")
        m, a = train_multitask_model(peaks, labels, background, hg38,
                                     relative_proportion=p)
        auc_multi[p] = a
        effects_multi[p] = test_motif_effect_multitask(m, bg_loci, OCT_SOX, task_idx=0)
        print(f"[single] proportion={p}")
        s, a = train_singletask_model(peaks, labels, background, hg38, task_idx=0,
                                      relative_proportion=p)
        auc_single[p] = a
        effects_single[p] = test_motif_effect_singletask(s, bg_loci, OCT_SOX)

    with open(OUT / "fig4_effects.pkl", "wb") as fh:
        pickle.dump(dict(effects_multi=effects_multi, effects_single=effects_single,
                         auc_multi=auc_multi, auc_single=auc_single), fh)

    _split_violin(effects_single, effects_multi)
    _auc_bars(auc_single, "Single-task", "single_task_auc_barplot.pdf")
    _auc_bars(auc_multi, "Multi-task", "multi_task_auc_barplot.pdf")
    print("done ->", OUT)


def _split_violin(single, multi):
    keys = sorted(single)
    rows = []
    for k in keys:
        for v in np.asarray(single[k]).ravel():
            rows.append((k, float(v), "Single-task"))
        for v in np.asarray(multi[k]).ravel():
            rows.append((k, float(v), "Multi-task"))
    df = pd.DataFrame(rows, columns=["proportion", "effect", "model"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df, x="proportion", y="effect", hue="model", split=True,
                   inner="quartile", ax=ax, palette=["#4C9BD4", "#7DBE7E"])
    ax.axhline(0, ls="--", color="red", lw=1)
    ax.set_xlabel("Proportion of training data composed of background regions")
    ax.set_ylabel("Effect distribution")
    ax.set_title("Single-task vs Multi-task Effects")
    fig.tight_layout(); fig.savefig(OUT / "single_vs_multi_split_violin.pdf")
    plt.close(fig)


def _auc_bars(auc, label, fname):
    keys = sorted(auc)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar([str(k) for k in keys], [auc[k] for k in keys], color="#6E97C9")
    for b, k in zip(bars, keys):
        ax.text(b.get_x() + b.get_width() / 2, auc[k] + 0.01, f"{auc[k]:.4f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AUC Score")
    ax.set_xlabel("Relative Background Size")
    ax.set_title(f"{label} Model Performance\np(AP1 Peak)")
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


if __name__ == "__main__":
    main()
