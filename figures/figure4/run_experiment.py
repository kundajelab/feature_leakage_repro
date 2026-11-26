#!/usr/bin/env python
"""Train single-task and multi-task AP1 CNNs across background proportions and measure the OCT-SOX effect (Figure 4c, supplement d, e).

Outputs (outputs/): single_vs_multi_split_violin.pdf, single_task_auc_barplot.pdf,
multi_task_auc_barplot.pdf, fig4_effects.pkl and labeled_peaks.bed.

Run (env: featleak-dl):
    python run_experiment.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
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
MIN_AUC = 0.55   # fits at or below chance are reported and left out of the summary


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one fit per seed per proportion; effects are pooled "
                         "across seeds and AUCs averaged.")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    hg38, peaks, labels, background = build_data()
    print(f"peaks={peaks.shape[0]}  background={background.shape[0]}")

    # Fixed measurement background, shared by every fit so that effects are comparable.
    bg_loci, _ = extract_loci(background.sample(n=1000, random_state=0), hg38,
                              in_window=300, return_filtered=True, verbose=True)

    effects_multi = {p: [] for p in PROPORTIONS}
    effects_single = {p: [] for p in PROPORTIONS}
    auc_multi = {p: [] for p in PROPORTIONS}
    auc_single = {p: [] for p in PROPORTIONS}
    dropped = []

    for seed in args.seeds:
        for p in PROPORTIONS:
            print(f"[multi]  proportion={p} seed={seed}")
            m, a = train_multitask_model(peaks, labels, background, hg38,
                                         relative_proportion=p, random_state=seed)
            if a <= MIN_AUC:
                dropped.append(("multi", p, seed, a))
            else:
                auc_multi[p].append(a)
                effects_multi[p].append(
                    test_motif_effect_multitask(m, bg_loci, OCT_SOX, task_idx=0))

            print(f"[single] proportion={p} seed={seed}")
            s, a = train_singletask_model(peaks, labels, background, hg38, task_idx=0,
                                          relative_proportion=p, random_state=seed)
            if a <= MIN_AUC:
                dropped.append(("single", p, seed, a))
            else:
                auc_single[p].append(a)
                effects_single[p].append(
                    test_motif_effect_singletask(s, bg_loci, OCT_SOX))

    for kind, p, seed, a in dropped:
        print(f"  {kind} proportion={p} seed={seed}: AUC={a:.3f} "
              f"<= {MIN_AUC}, left out of the summary")

    for p in PROPORTIONS:
        for name, eff, auc in (("multi", effects_multi, auc_multi),
                               ("single", effects_single, auc_single)):
            if not eff[p]:
                raise RuntimeError(
                    f"every {name} fit at proportion {p} was at or below chance; "
                    f"rerun with more seeds")
            print(f"  {name} proportion={p}: n_fits={len(eff[p])} "
                  f"mean AUC={np.mean(auc[p]):.4f} "
                  f"median effect={np.median(np.concatenate([np.asarray(e).ravel() for e in eff[p]])):+.4f}")

    with open(OUT / "fig4_effects.pkl", "wb") as fh:
        pickle.dump(dict(effects_multi=effects_multi, effects_single=effects_single,
                         auc_multi=auc_multi, auc_single=auc_single,
                         seeds=list(args.seeds), dropped=dropped), fh)

    _split_violin(effects_single, effects_multi)
    _auc_bars({k: float(np.mean(v)) for k, v in auc_single.items()},
              "Single-task", "single_task_auc_barplot.pdf")
    _auc_bars({k: float(np.mean(v)) for k, v in auc_multi.items()},
              "Multi-task", "multi_task_auc_barplot.pdf")
    print("done ->", OUT)


def _split_violin(single, multi):
    keys = sorted(single)
    rows = []
    for k in keys:
        for fit in single[k]:
            for v in np.asarray(fit).ravel():
                rows.append((k, float(v), "Single-task"))
        for fit in multi[k]:
            for v in np.asarray(fit).ravel():
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
