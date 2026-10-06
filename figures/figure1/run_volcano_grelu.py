#!/usr/bin/env python
"""Compute and plot the OCT-SOX marginalization volcano across Enformer or Borzoi DNase tasks (Supplementary Figure 2).

Outputs (outputs/): volcano_<model>_raw.npz and volcano_<model>[_specificity].csv/.pdf.

Run (env: featleak-dl):
    python run_volcano_grelu.py --model enformer --n-bg 40
    python run_volcano_grelu.py --model borzoi --n-bg 40
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
from statsmodels.stats.multitest import multipletests  # noqa: E402
import grelu.data.dataset  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.utils import characters  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

from common import config  # noqa: E402
from common.models import load_grelu_model  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from volcano_plot import volcano  # noqa: E402

OUT = Path(__file__).parent / "outputs"
# flank insertions (outside the enhancer core) so we don't overwrite native motifs
OFFSETS = [-150, 150, -200, 200, -250, 250, -300, 300]   # up to 8 insertions
BINSIZE = {"enformer": 128, "borzoi": 32}
READOUT_HALF = 500


def shuffled_motif(motif, seed=100):
    rng = np.random.default_rng(seed)
    return "".join(rng.permutation(list(motif)))


def compute_levels(model, X, motif, seq_len, mid, idx, w, bs):
    """Return (K+1, N, n_tasks) central DNase after inserting 0..K `motif` copies."""
    levels, cur = [], X
    for k in range(len(OFFSETS) + 1):
        if k > 0:
            o = OFFSETS[k - 1]
            cur = substitute(cur, motif, start=mid + o - len(motif) // 2)
        ds = grelu.data.dataset.SeqDataset([characters(x) for x in cur], seq_len=seq_len)
        pred = model.predict_on_dataset(ds, devices=0, batch_size=bs)
        B = pred.shape[-1]; c = B // 2
        levels.append(np.asarray(pred[:, idx, c - w:c + w + 1].sum(axis=-1)))
        print(f"  level {k}/{len(OFFSETS)} predicted", flush=True)
    return np.stack(levels)


def real_backgrounds(n, seq_len):
    """Top-n real fibroblast DNase enhancers (baseline accessibility present)."""
    peaks = pd.read_csv(config.resolve("peaks_fibroblast"), sep="\t", header=None)
    peaks = peaks.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    peaks = peaks[peaks.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    top = peaks.sort_values("signal", ascending=False).head(n * 2).reset_index(drop=True)
    loci = extract_loci(top[["chrom", "start", "end"]], str(config.resolve("hg38_fasta")),
                        in_window=seq_len, verbose=False)
    keep = [i for i in range(len(loci)) if int(loci[i].sum()) == seq_len][:n]
    return loci[keep]


def seq_len_of(model, which):
    dp = model.data_params
    return dp["train_seq_len"] if which == "borzoi" else dp["train"]["seq_len"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["borzoi", "enformer"], required=True)
    ap.add_argument("--n-bg", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    bs = args.batch_size or (2 if args.model == "borzoi" else 4)

    name = "human_fold0" if args.model == "borzoi" else "human"
    model = load_grelu_model(args.model, name)
    seq_len = seq_len_of(model, args.model)
    tasks = pd.DataFrame(model.data_params["tasks"])
    dnase = tasks[tasks.assay == "DNASE"]
    idx = dnase.index.to_numpy()
    mid = seq_len // 2
    w = max(1, round(READOUT_HALF / BINSIZE[args.model]))
    print(f"{args.model}: seq_len={seq_len}, {len(idx)} DNase tasks", flush=True)

    X = real_backgrounds(args.n_bg, seq_len)
    print(f"  {X.shape[0]} real fibroblast enhancer backgrounds", flush=True)
    print("[real OCT-SOX]", flush=True)
    Yr = compute_levels(model, X, OCT_SOX, seq_len, mid, idx, w, bs)
    print("[shuffled OCT-SOX control]", flush=True)
    Ys = compute_levels(model, X, shuffled_motif(OCT_SOX), seq_len, mid, idx, w, bs)

    K1, N, T = Yr.shape
    x = np.repeat(np.arange(K1), N)
    F = 1e-2 * np.median(Yr)                      # floor for log (near-zero tasks)
    logr = np.log2(Yr + F); logs = np.log2(Ys + F)
    np.savez(OUT / f"volcano_{args.model}_raw.npz", Yr=Yr, Ys=Ys, x=x,
             task=dnase.name.values, sample=dnase["sample"].values)
    sl_r, p_r = np.zeros(T), np.ones(T)          # log2FC slope per insertion
    sl_s, p_s = np.zeros(T), np.ones(T)          # specificity (real - shuffle), log2FC
    dfree = 2 * len(x) - 4
    for t in range(T):
        lr = stats.linregress(x, logr[:, :, t].reshape(-1))
        ls = stats.linregress(x, logs[:, :, t].reshape(-1))
        sl_r[t] = lr.slope; p_r[t] = lr.pvalue
        d = lr.slope - ls.slope
        se = np.hypot(lr.stderr, ls.stderr)
        sl_s[t] = d
        p_s[t] = 2 * stats.t.sf(abs(d / se), dfree) if se > 0 else 1.0

    meta = dict(task=dnase.name.values, sample=dnase["sample"].values)
    M = args.model.capitalize()
    for tag, slope, pval, ttl, xl in [
            ("", sl_r, p_r, f"{M}: OCT-SOX effect across all DNase tasks",
             "OCT-SOX effect per insertion  (log2FC)"),
            ("_specificity", sl_s, p_s, f"{M}: OCT-SOX motif-specific effect (real − shuffled)",
             "OCT-SOX-specific effect per insertion  (log2FC)")]:
        df = pd.DataFrame({**meta, "slope": slope, "pval": pval,
                           "qval": multipletests(pval, method="fdr_bh")[1]})
        df.to_csv(OUT / f"volcano_{args.model}{tag}.csv", index=False)
        volcano(df, ttl, OUT / f"volcano_{args.model}{tag}.pdf", xlabel=xl)
        nrep = int(((df.qval < 0.05) & (df.slope < 0)).sum())
        npos = int(((df.qval < 0.05) & (df.slope > 0)).sum())
        print(f"{args.model}{tag}: {nrep} repressive, {npos} positive, "
              f"{len(df)-nrep-npos} n.s.", flush=True)


if __name__ == "__main__":
    main()
