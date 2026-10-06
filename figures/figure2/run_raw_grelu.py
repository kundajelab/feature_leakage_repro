#!/usr/bin/env python
"""Compute the raw OCT-SOX insertion effect in Enformer / Borzoi on the Figure 2 enhancers (Supp 2a).

Outputs (outputs/): raw_insertion_<model>.csv.

Run (env: featleak-dl):
    CUDA_VISIBLE_DEVICES=0 python run_raw_grelu.py --model enformer
    CUDA_VISIBLE_DEVICES=0 python run_raw_grelu.py --model borzoi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx

from enhancers import config, insert, top_dermis_enhancers, window
from common.models import load_grelu_model
from common.motifs import OCT_SOX

OUT = Path(__file__).parent / "outputs"
TASK, EPS = "ENCFF302JEV", 1e-3
BIN = {"enformer": 128, "borzoi": 32}


def main() -> None:
    import grelu.data.dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["enformer", "borzoi"], required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    model = load_grelu_model(args.model, "human_fold0" if args.model == "borzoi" else "human")
    dp = model.data_params
    L = dp["train_seq_len"] if args.model == "borzoi" else dp["train"]["seq_len"]
    tasks = pd.DataFrame(dp["tasks"])
    ti = int(tasks.index[tasks.name == TASK][0])
    b = BIN[args.model]

    enh = top_dermis_enhancers()
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    base = [window(fa, r.chrom, r.mid, L) for r in enh.itertuples()]

    def predict(seqs):
        ds = grelu.data.dataset.SeqDataset(seqs, seq_len=L)
        p = model.predict_on_dataset(ds, devices=0, batch_size=2 if args.model == "borzoi" else 4)
        p = p[:, ti, :]
        c, k, w = p.shape[-1] // 2, 512 // b, max(1, round(300 / b))   # bin c starts at the centre
        return p[:, c - k:c + k].sum(-1), p[:, c - w:c + w + 1].sum(-1)

    (w0, n0), (w1, n1) = predict(base), predict([insert(q, OCT_SOX) for q in base])
    df = pd.DataFrame({"locus": enh.locus,
                       "lfc_1kb": np.log2((w1 + EPS) / (w0 + EPS)),
                       "lfc_300bp": np.log2((n1 + EPS) / (n0 + EPS))})
    df.to_csv(OUT / f"raw_insertion_{args.model}.csv", index=False)
    print(args.model, "median log2FC (1 kb):", round(float(df.lfc_1kb.median()), 3))


if __name__ == "__main__":
    main()
