#!/usr/bin/env python
"""Compute the AlphaGenome (PyTorch port) OCT-SOX insertion effect at full input length (Figure 2b).

Outputs (outputs/): ag_octsox_specificity<tag>.csv.

Run (env: featleak-agtorch):
    python run_marg_alphagenome_torch.py --n-loci 120 --seq-len 524288
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfaidx  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OCT_SOX = "ATTGTTATGCAAAT"
OFFSETS = [-150, 150, -200, 200]
CENTRAL = 150
EPS = 1e-3
K_SHUF = 3


def shuffled(motif, k, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        m = list(motif); rng.shuffle(m); out.append("".join(m))
    return out


def insert(seq, motif, mid):
    s = list(seq)
    for o in OFFSETS:
        s[mid + o:mid + o + len(motif)] = list(motif)
    return "".join(s)


def main() -> None:
    from alphagenome_pytorch import AlphaGenome

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loci", type=int, default=120)
    ap.add_argument("--peaks-key", default="peaks_dermis")
    ap.add_argument("--channel", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=524288)
    ap.add_argument("--tag", default="_dermis_len524288")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L, c0 = args.seq_len, args.seq_len // 2

    model = AlphaGenome.from_pretrained(
        str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    pk = pd.read_csv(config.resolve(args.peaks_key), sep="\t", header=None).rename(
        columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    pk = pk[pk.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    pk = pk.sort_values("signal", ascending=False).head(args.n_loci + 10)

    seqs, loci = [], []
    for _, r in pk.iterrows():
        s = (int(r.start) + int(r.end)) // 2 - L // 2
        q = str(fa[r.chrom][s:s + L]).upper() if s >= 0 else ""
        if len(q) == L and not (set(q[c0 - 1000:c0 + 1000]) - set("ACGT")):
            seqs.append(q); loci.append(f"{r.chrom}:{int(r.start)}")
        if len(seqs) == args.n_loci:
            break

    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i

    @torch.no_grad()
    def score(q):
        idx = lut[np.frombuffer(q.encode(), dtype=np.uint8)]
        x = torch.zeros((1, L, 4), dtype=torch.float32)
        ok = idx >= 0                      # N bases stay all-zero
        x[0, np.where(ok)[0], idx[ok]] = 1
        out = model.predict(x.cuda(), organism_index=0)
        return float(out["dnase"][1][0, c0 - CENTRAL:c0 + CENTRAL, args.channel].sum())

    shuf = shuffled(OCT_SOX, K_SHUF)
    s0 = np.array([score(q) for q in seqs])
    sR = np.array([score(insert(q, OCT_SOX, c0)) for q in seqs])
    sS = np.mean([[score(insert(q, m, c0)) for q in seqs] for m in shuf], axis=0)
    lfc_real = np.log2((sR + EPS) / (s0 + EPS))
    lfc_shuf = np.log2((sS + EPS) / (s0 + EPS))
    spec = lfc_real - lfc_shuf
    out = OUT / f"ag_octsox_specificity{args.tag}.csv"
    pd.DataFrame({"locus": loci, "log2fc_real": lfc_real, "log2fc_shuf": lfc_shuf,
                  "specificity": spec}).to_csv(out, index=False)
    print(f"AlphaGenome-torch (DNase ch{args.channel}, {L:,} bp) OCT-SOX over {len(spec)} "
          f"enhancers: raw median={np.median(lfc_real):+.4f}  shuffle median="
          f"{np.median(lfc_shuf):+.4f}  SPECIFICITY median={np.median(spec):+.4f} "
          f"(Wilcoxon<0 p={wilcoxon(spec, alternative='less').pvalue:.3g})")
    print("done ->", out)


if __name__ == "__main__":
    main()
