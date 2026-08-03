#!/usr/bin/env python
"""Figure 1C (Borzoi / Enformer via gReLU): OCT-SOX ablation effect boxplot.

Reproduces the original borzoi_v2 / enformer_v2 ablation exactly:
  * OCT-SOX FIMO hits within 200-500 bp of a fibroblast DNase peak summit.
  * Extract each model's input window centred on the motif; dinucleotide-shuffle
    the central 14 bp (the OCT-SOX motif).
  * effect = sum over ALL bins of (signal_with_motif - signal_ablated) for the
    fibroblast DNase task (ENCODE track ENCFF302JEV).
  * A predominantly NEGATIVE effect => the motif is predicted to REPRESS
    accessibility (the implausible result). Wilcoxon tests effect < 0.

Run (env: featleak-dl):
    python run_grelu.py --model borzoi --n-loci 200     # or --model enformer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import grelu.data.dataset  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ersatz import dinucleotide_shuffle  # noqa: E402
from tangermeme.utils import characters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.models import load_grelu_model  # noqa: E402
from fig1_utils import annotate_with_peak_distances  # noqa: E402

OUT = Path(__file__).parent / "outputs"
FIBROBLAST_TASK = "ENCFF302JEV"   # fibroblast DNase track used by both models
OCTSOX_WINDOW = 14


def seq_len_of(model, which):
    dp = model.data_params
    return dp["train_seq_len"] if which == "borzoi" else dp["train"]["seq_len"]


def ablate_effect(model, seq_len, task_idx, hits, n_loci, batch_size):
    """(with_motif - ablated) summed over all bins for the fibroblast task."""
    h = (hits.filter(["sequence_name", "start", "stop"])
         .rename(columns={"sequence_name": "chromosome", "stop": "end"})
         .astype({"start": "int", "end": "int"}).reset_index(drop=True))
    loci, idx = extract_loci(h.sample(min(n_loci, len(h))),
                             str(config.resolve("hg38_fasta")), in_window=seq_len,
                             return_filtered=True, verbose=True)
    mask = loci.sum(axis=1).sum(axis=1) == seq_len
    loci = loci[mask]
    a0 = seq_len // 2 - OCTSOX_WINDOW // 2
    ablated = dinucleotide_shuffle(loci, start=a0, end=a0 + OCTSOX_WINDOW, n=1).squeeze(1)

    ds = grelu.data.dataset.SeqDataset([characters(x) for x in loci], seq_len=seq_len)
    ds_abl = grelu.data.dataset.SeqDataset([characters(x) for x in ablated], seq_len=seq_len)
    with_motif = model.predict_on_dataset(ds, devices=0, batch_size=batch_size)
    ablated_pred = model.predict_on_dataset(ds_abl, devices=0, batch_size=batch_size)
    with_sum = with_motif[:, task_idx, :].sum(axis=1)     # per-locus (with motif, whole track)
    abl_sum = ablated_pred[:, task_idx, :].sum(axis=1)     # per-locus (ablated, whole track)
    # local central window (+/-500 bp) so a meaningful log2FC can be computed
    nbins = with_motif.shape[-1]
    binsize = seq_len / nbins
    c = nbins // 2; w = max(1, int(500 / binsize))
    with_local = with_motif[:, task_idx, c - w:c + w].sum(axis=1)
    abl_local = ablated_pred[:, task_idx, c - w:c + w].sum(axis=1)
    return with_sum - abl_sum, with_sum, abl_sum, with_local, abl_local


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["borzoi", "enformer"], default="borzoi")
    ap.add_argument("--n-loci", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    project = args.model
    name = "human_fold0" if args.model == "borzoi" else "human"
    model = load_grelu_model(project, name)
    seq_len = seq_len_of(model, args.model)
    tasks = pd.DataFrame(model.data_params["tasks"])
    task_idx = int(tasks.loc[tasks.name == FIBROBLAST_TASK].index.values[0])

    hits = annotate_with_peak_distances(
        config.resolve("fimo_octsox", must_exist=True),
        config.resolve("peaks_enformer_fibroblast", must_exist=True))
    hits = hits[hits.distance.between(200, 500)]

    scores, with_sum, abl_sum, with_local, abl_local = ablate_effect(
        model, seq_len, task_idx, hits, args.n_loci, args.batch_size)
    scores = np.asarray(scores)
    np.save(OUT / f"fig1_{args.model}_octsox_before.npy", np.asarray(with_sum))
    np.save(OUT / f"fig1_{args.model}_octsox_after.npy", np.asarray(abl_sum))
    np.save(OUT / f"fig1_{args.model}_octsox_before_local.npy", np.asarray(with_local))
    np.save(OUT / f"fig1_{args.model}_octsox_after_local.npy", np.asarray(abl_local))
    w = stats.wilcoxon(scores, alternative="less")
    print(f"{args.model}: mean OCT-SOX ablation effect={scores.mean():.4f} "
          f"(Wilcoxon repressive p={w.pvalue:.3g})")

    fig, ax = plt.subplots(figsize=(2, 4))
    ax.boxplot([scores], labels=[args.model], showfliers=False, widths=0.5,
               patch_artist=True, boxprops=dict(facecolor="lightgray", color="black"),
               medianprops=dict(color="black"))
    ax.axhline(0, ls="--", color="0.6")
    ax.set_ylabel("OCT-SOX Motif Effect Size (Ablation)")
    if w.pvalue < 0.05:
        ax.set_title("*", fontsize=16)
    fig.tight_layout(); fig.savefig(OUT / f"fig1_{args.model}_octsox_boxplot.pdf")
    plt.close(fig)
    np.save(OUT / f"fig1_{args.model}_octsox_effect.npy", scores)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
