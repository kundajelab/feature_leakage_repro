#!/usr/bin/env python
"""Compute the raw OCT-SOX insertion effect in AlphaGenome (PyTorch port) on the Figure 2 enhancers (Figure 2b).

Outputs (outputs/): raw_insertion_alphagenome.csv.

Run (env: featleak-agtorch):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python run_raw_alphagenome.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import torch

from enhancers import config, insert, top_dermis_enhancers, window
from common.motifs import OCT_SOX

OUT = Path(__file__).parent / "outputs"
L, CHANNEL, EPS = 524288, 60, 1e-3


def main() -> None:
    from alphagenome_pytorch import AlphaGenome
    OUT.mkdir(exist_ok=True)
    model = AlphaGenome.from_pretrained(
        str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
    c0 = L // 2

    @torch.no_grad()
    def score(q):
        idx = lut[np.frombuffer(q.encode(), dtype=np.uint8)]
        x = torch.zeros((1, L, 4), dtype=torch.float32)
        ok = idx >= 0                                  # N bases stay all-zero
        x[0, np.where(ok)[0], idx[ok]] = 1
        t = model.predict(x.cuda(), organism_index=0)["dnase"][1][0, :, CHANNEL]
        return float(t[c0 - 512:c0 + 512].sum()), float(t[c0 - 150:c0 + 150].sum())

    rows = []
    for r in top_dermis_enhancers().itertuples():
        q = window(fa, r.chrom, r.mid, L)
        (w0, n0), (w1, n1) = score(q), score(insert(q, OCT_SOX))
        rows.append(dict(locus=r.locus, lfc_1kb=np.log2((w1 + EPS) / (w0 + EPS)),
                         lfc_300bp=np.log2((n1 + EPS) / (n0 + EPS))))
        print(len(rows), r.locus, f"{rows[-1]['lfc_1kb']:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "raw_insertion_alphagenome.csv", index=False)


if __name__ == "__main__":
    main()
