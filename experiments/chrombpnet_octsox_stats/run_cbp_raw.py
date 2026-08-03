#!/usr/bin/env python
"""ChromBPNet raw per-bp profiles at chr17 enhancer: exp vs 2^ count inversion.

bpnet-lite trains the count head on ln(counts+1) (natural log), so the correct
total-count inversion is exp(logcounts)-1, not 2**logcounts-1. This regenerates
the chr17 profiles at TRUE base-pair resolution (no smoothing) under both
inversions so we can (a) show the y-axis magnitude difference and (b) feed the
matched-track plots.

Run (env: featleak-dl): python run_cbp_raw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from bpnetlite.bpnet import BPNet  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
LOCUS = ("chr17", 76671700, 76671820)
OFFSETS = [-150, 150, -200, 200]


def profiles(model, X):
    with torch.no_grad():
        logits, logcounts = model(X.float().to(DEVICE))
    logits = logits.reshape(logits.shape[0], -1)
    shape = torch.softmax(logits - logits.mean(-1, keepdim=True), -1)   # (B, L)
    c_exp = torch.exp(logcounts) - 1.0                                  # natural log inverse
    c_pow2 = torch.pow(2.0, logcounts) - 1.0                            # base-2 (old)
    p_exp = (shape * c_exp).cpu().numpy()
    p_pow2 = (shape * c_pow2).cpu().numpy()
    return p_exp, p_pow2, c_exp.squeeze().cpu().numpy(), c_pow2.squeeze().cpu().numpy(), \
        logcounts.squeeze().cpu().numpy()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--locus", default=None, help="chrom:start:end override")
    ap.add_argument("--tag", default="chr17")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = BPNet.from_chrombpnet(
        str(config.resolve("chrombpnet_fibroblast", must_exist=True))).to(DEVICE).eval()
    if args.locus:
        c_, s_, e_ = args.locus.split(":"); chrom, start, end = c_, int(s_), int(e_)
    else:
        chrom, start, end = LOCUS
    peak = pd.DataFrame([[chrom, start, end]], columns=["chrom", "start", "end"])
    loci = extract_loci(peak, str(config.resolve("hg38_fasta")), verbose=False).float()
    seq_len = loci.shape[2]; mid = seq_len // 2
    ins = [mid + o for o in OFFSETS]

    levels = [loci]; cur = loci
    for o in ins:
        cur = substitute(cur, OCT_SOX, start=o)
        levels.append(cur)

    p_exp, p_pow2 = [], []
    for lv in levels:
        pe, pp, ce, cp, lc = profiles(model, lv)
        p_exp.append(pe[0]); p_pow2.append(pp[0])
    p_exp = np.array(p_exp); p_pow2 = np.array(p_pow2)      # (5, 1000)

    prof_len = p_exp.shape[1]; trim = (seq_len - prof_len) // 2
    bp = np.arange(prof_len) - prof_len // 2
    ins_bp = np.array([o - trim - prof_len // 2 for o in ins])
    np.savez(OUT / f"cbp_raw_{args.tag}.npz", profiles_exp=p_exp, profiles_pow2=p_pow2,
             bp=bp, ins_bp=ins_bp)

    # native-level report
    pe, pp, ce, cp, lc = profiles(model, loci)
    print(f"native logcounts (ln(counts+1)) = {float(lc):.3f}")
    print(f"  exp(logcounts)-1  = {float(ce):8.1f} total counts  -> peak {p_exp[0].max():.3f} /bp")
    print(f"  2**logcounts-1    = {float(cp):8.1f} total counts  -> peak {p_pow2[0].max():.3f} /bp")
    print(f"  ratio exp/pow2    = {float(ce)/float(cp):.2f}x")
    print("saved ->", OUT / "cbp_raw_chr17.npz")


if __name__ == "__main__":
    main()
