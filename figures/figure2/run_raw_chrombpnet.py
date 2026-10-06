#!/usr/bin/env python
"""Figure 2b / Supp 2a reference: raw OCT-SOX insertion effect in ChromBPNet.

Same 120 enhancers and insertions as run_raw_alphagenome.py, scored with the ChromBPNet model
trained on ENCSR000EPO (bias-corrected, fold 0 = `chrombpnet_dermis`, the model used in Fig 2b
and Fig 7). Effect = (count_head(inserted) - count_head(original)) / ln 2
= log2[(counts_inserted + 1) / (counts_original + 1)] over the 1,000 bp output.

With --n-shuffles K (default 50) it also scores K mononucleotide shuffles of OCT-SOX (seed 2026)
plus the 3 shuffles of the original Fig 2b control, for the shuffle-comparison supplement.

Run (env: featleak-dl):
    CUDA_VISIBLE_DEVICES=0 python run_raw_chrombpnet.py
Writes outputs/raw_insertion_chrombpnet.csv (long format: motif, locus, lfc_total, lfc_central).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bpnetlite.bpnet import BPNet
from bpnetlite.io import extract_loci
from tangermeme.ersatz import substitute

from enhancers import OFFSETS, config, top_dermis_enhancers
from common.motifs import OCT_SOX
from common.motif_controls import shuffled_motifs

OUT = Path(__file__).parent / "outputs"
IN_WIN = 2114


def readouts(model, X):
    """Total predicted counts, and counts within the old Fig 2b central profile window."""
    cen, tot = [], []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            yp, yc = model(X[i:i + 32].cuda())
            counts = (torch.exp(yc) - 1.0).squeeze(-1)
            logits = yp.reshape(yp.shape[0], -1)
            probs = torch.softmax(logits - logits.mean(-1, keepdim=True), -1)
            c, w = probs.shape[-1] // 2, int(150 / (IN_WIN / probs.shape[-1]))
            cen.append((counts * probs[:, c - w:c + w].sum(-1)).cpu().numpy())
            tot.append(counts.cpu().numpy())
    return np.concatenate(cen), np.concatenate(tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    enh = top_dermis_enhancers()
    loci = extract_loci(enh[["chrom", "start", "end"]], str(config.resolve("hg38_fasta")),
                        in_window=IN_WIN, verbose=False).float()
    mid = IN_WIN // 2

    rng = np.random.default_rng(2026)
    motifs = {"real": OCT_SOX,
              **{f"orig{j}": m for j, m in enumerate(shuffled_motifs(OCT_SOX, 3))},
              **{f"new{j}": "".join(rng.permutation(list(OCT_SOX))) for j in range(args.n_shuffles)}}

    def insert_all(motif):
        cur = loci.clone()
        for o in OFFSETS:
            cur = substitute(cur, motif, start=mid + o)
        return cur

    model = BPNet.from_chrombpnet(
        str(config.resolve("chrombpnet_dermis", must_exist=True))).cuda().eval()
    c0, t0 = readouts(model, loci)
    rows = []
    for name, m in motifs.items():
        c1, t1 = readouts(model, insert_all(m))
        rows += [dict(motif=name, seq=m, locus=enh.locus[i],
                      lfc_total=np.log2((t1[i] + 1) / (t0[i] + 1)),
                      lfc_central=np.log2((c1[i] + 1e-3) / (c0[i] + 1e-3)))
                 for i in range(len(enh))]
    pd.DataFrame(rows).to_csv(OUT / "raw_insertion_chrombpnet.csv", index=False)


if __name__ == "__main__":
    main()
