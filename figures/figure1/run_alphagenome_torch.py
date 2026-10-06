#!/usr/bin/env python
"""Figure 1C (AlphaGenome, full-length input): OCT-SOX ablation log2 fold-change.

Scores exactly the OCT-SOX sites written by run_chrombpnet_logfc.py (FIMO hits
200-500 bp from a dermal-fibroblast DNase summit), so AlphaGenome and ChromBPNet
are compared on the same loci and the same ENCODE experiment (ENCSR000EPO).

For each site the 524,288 bp window centred on the motif is scored with the motif
intact and with the central 14 bp dinucleotide-shuffled. Effect is the log2
fold-change of predicted dermal-fibroblast DNase (channel 60, CL:0002551) summed
over +/-500 bp around the motif:

    log2FC = log2( signal(with motif) / signal(ablated) )     (<0: motif represses)

Uses the PyTorch port of AlphaGenome (all-folds weights, matches the JAX model).

Run (env: featleak-agtorch):
    python run_alphagenome_torch.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfaidx  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from tangermeme.ersatz import dinucleotide_shuffle  # noqa: E402
from tangermeme.utils import characters, one_hot_encode  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs"
MOTIF_LEN = 14
LOCAL_HALF = 500
EPS = 1e-3


def main() -> None:
    from alphagenome_pytorch import AlphaGenome

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_dermis")
    ap.add_argument("--channel", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=524288)
    ap.add_argument("--max-loci", type=int, default=None)
    ap.add_argument("--out-suffix", default="", help="extra suffix for the saved array")
    args = ap.parse_args()
    L, c0 = args.seq_len, args.seq_len // 2

    sites = pd.read_csv(OUT / f"fig1_octsox_sites{args.tag}.csv")
    if args.max_loci:
        sites = sites.head(args.max_loci)
    model = AlphaGenome.from_pretrained(
        str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i

    @torch.no_grad()
    def local_signal(q):
        idx = lut[np.frombuffer(q.encode(), dtype=np.uint8)]
        x = torch.zeros((1, L, 4), dtype=torch.float32)
        ok = idx >= 0
        x[0, np.where(ok)[0], idx[ok]] = 1
        out = model.predict(x.cuda(), organism_index=0)
        return float(out["dnase"][1][0, c0 - LOCAL_HALF:c0 + LOCAL_HALF, args.channel].sum())

    a0, a1 = c0 - MOTIF_LEN // 2, c0 + MOTIF_LEN // 2
    before, after = [], []
    for i, r in enumerate(sites.itertuples()):
        mid = (int(r.start) + int(r.end)) // 2
        w0 = mid - L // 2
        q = str(fa[r.chromosome][max(w0, 0):w0 + L]).upper()
        q = "N" * (L - len(q)) + q if w0 < 0 else q + "N" * (L - len(q))
        core = one_hot_encode(q[a0:a1]).unsqueeze(0).float()
        shuf = characters(dinucleotide_shuffle(core, n=1, random_state=i)[0, 0])
        before.append(local_signal(q))
        after.append(local_signal(q[:a0] + shuf + q[a1:]))
        if i % 100 == 0:
            print(f"  {i}/{len(sites)}", flush=True)
    before, after = np.array(before), np.array(after)
    logfc = np.log2((before + EPS) / (after + EPS))
    np.save(OUT / f"fig1_alphagenome_octsox_logfc{args.tag}{args.out_suffix}.npy", logfc)
    w = stats.wilcoxon(logfc, alternative="less")
    print(f"AlphaGenome-torch (ch{args.channel}, {L:,} bp) OCT-SOX log2FC over n={len(logfc)}: "
          f"median={np.median(logfc):+.4f} mean={logfc.mean():+.4f} "
          f"Wilcoxon repressive p={w.pvalue:.3g}")
    print("saved ->", OUT / f"fig1_alphagenome_octsox_logfc{args.tag}{args.out_suffix}.npy")


if __name__ == "__main__":
    main()
