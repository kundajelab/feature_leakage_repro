#!/usr/bin/env python
"""Compute the Enformer / Borzoi OCT-SOX insertion effect (real vs shuffled motif) with gReLU (Figure 2 supplement).

Outputs (outputs/): grelu_<model>_canonical.npz (keys real, shuf).

Run (env: featleak-dl):
    python run_marg_grelu.py --model enformer --n-loci 40
    python run_marg_grelu.py --model borzoi   --n-loci 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import grelu.data.dataset  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.utils import characters  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import config  # noqa: E402
from common.models import load_grelu_model  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from common.motif_controls import shuffled_motifs  # noqa: E402

OUT = Path(__file__).parent / "outputs"
CANONICAL = "ENCFF302JEV"                  # fibroblast of dermis (paper task)
OFFSETS = [-150, 150, -200, 200]
READOUT_HALF = 300
BIN_SIZE = {"enformer": 128, "borzoi": 32}
K_SHUF = 3


def seq_len_of(model, which):
    dp = model.data_params
    return dp["train_seq_len"] if which == "borzoi" else dp["train"]["seq_len"]


def predict_central(model, seqs, seq_len, task_idx, w_bins, batch_size):
    ds = grelu.data.dataset.SeqDataset(seqs, seq_len=seq_len)
    pred = model.predict_on_dataset(ds, devices=0, batch_size=batch_size)
    B = pred.shape[-1]; c = B // 2
    return pred[:, task_idx, c - w_bins:c + w_bins + 1].sum(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["borzoi", "enformer"], required=True)
    ap.add_argument("--n-loci", type=int, default=120)
    ap.add_argument("--peaks-key", default="peaks_dermis",
                    help="enhancer substrate: dermis fibroblast peaks (ENCSR000EPO)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bs = args.batch_size or (2 if args.model == "borzoi" else 4)

    name = "human_fold0" if args.model == "borzoi" else "human"
    model = load_grelu_model(args.model, name)
    seq_len = seq_len_of(model, args.model)
    tasks = pd.DataFrame(model.data_params["tasks"])
    fib = tasks[(tasks.assay == "DNASE") &
                tasks["sample"].astype(str).str.contains("fibroblast", case=False)]
    ci = list(fib.name.values).index(CANONICAL)
    fib_idx = fib.index.to_numpy()
    w_bins = max(1, round(READOUT_HALF / BIN_SIZE[args.model]))

    peaks = pd.read_csv(config.resolve(args.peaks_key), sep="\t", header=None)
    peaks = peaks.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    peaks = peaks[peaks.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    top = peaks.sort_values("signal", ascending=False).head(args.n_loci * 2).reset_index(drop=True)
    loci = extract_loci(top[["chrom", "start", "end"]], str(config.resolve("hg38_fasta")),
                        in_window=seq_len, verbose=False)
    keep = [i for i in range(len(loci)) if int(loci[i].sum()) == seq_len][:args.n_loci]
    base = loci[keep]; mid = seq_len // 2

    def build(motif):
        cur = base.clone()
        for o in OFFSETS:
            cur = substitute(cur, motif, start=mid + o)
        return [characters(x) for x in cur]

    s0 = predict_central(model, [characters(x) for x in base], seq_len, fib_idx, w_bins, bs)
    real = np.log2((predict_central(model, build(OCT_SOX), seq_len, fib_idx, w_bins, bs) + 1e-3) / (s0 + 1e-3))
    shuf = np.mean([np.log2((predict_central(model, build(m), seq_len, fib_idx, w_bins, bs) + 1e-3) / (s0 + 1e-3))
                    for m in shuffled_motifs(OCT_SOX, K_SHUF)], axis=0)

    np.savez(OUT / f"grelu_{args.model}_canonical.npz",
             real=real[:, ci], shuf=shuf[:, ci], name=CANONICAL)
    spec = real[:, ci] - shuf[:, ci]
    w = wilcoxon(spec, alternative="less")
    print(f"{args.model} {CANONICAL}: raw median={np.median(real[:, ci]):+.4f}  "
          f"SPECIFICITY median={np.median(spec):+.4f} (Wilcoxon<0 p={w.pvalue:.3g})")
    print("done ->", OUT / f"grelu_{args.model}_canonical.npz")


if __name__ == "__main__":
    main()
