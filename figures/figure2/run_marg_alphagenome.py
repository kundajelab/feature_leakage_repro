#!/usr/bin/env python
"""Compute the AlphaGenome OCT-SOX insertion effect (real vs shuffled motif) on fibroblast peaks (Figure 2).

Outputs (outputs/): ag_octsox_specificity<tag>.csv.

Run (env: alphagenome-ft):
    python run_marg_alphagenome.py --n-loci 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "alphagenome_finetune"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from common.motif_controls import shuffled_motifs  # noqa: E402
import ft_core  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 2048; C0 = SEQ_LEN // 2; CENTRAL = 150; CH = 207     # channel 207 = skin-biceps NTR:0000521
OFFSETS = [-150, 150, -200, 200]; EPS = 1e-3; BATCH = 8
K_SHUF = 3


def insert(seq, motif, mid):
    s = list(seq)
    for o in OFFSETS:
        s[mid + o:mid + o + len(motif)] = list(motif)
    return "".join(s)


def csum(fwd, params, state, seqs):
    out = []
    for i in range(0, len(seqs), BATCH):
        c = seqs[i:i + BATCH]
        ch, _, _ = fwd(params, state, jnp.asarray(ft_core.encode_batch(c)), jnp.zeros((len(c),), jnp.int32))
        out.append(np.asarray(ch, np.float32)[:, C0 - CENTRAL:C0 + CENTRAL].sum(1))
    return np.concatenate(out)


def main() -> None:
    global SEQ_LEN, C0
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loci", type=int, default=120)
    ap.add_argument("--peaks-key", default="peaks_fibroblast")
    ap.add_argument("--channel", type=int, default=CH,
                    help="AlphaGenome DNase channel (207 skin biceps, 60 dermis CL:0002551)")
    ap.add_argument("--tag", default="", help="suffix for the output csv")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN,
                    help="AlphaGenome input length (bp); the readout stays the central 300 bp")
    args = ap.parse_args()
    SEQ_LEN, C0 = args.seq_len, args.seq_len // 2
    OUT.mkdir(parents=True, exist_ok=True)
    import pyfaidx
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    peaks = pd.read_csv(config.resolve(args.peaks_key), sep="\t", header=None).rename(
        columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    peaks = peaks[peaks.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    peaks = peaks.sort_values("signal", ascending=False).head(args.n_loci)

    model = ft_core.load_model()
    fwd = ft_core.make_dnase_forward(ft_core.get_apply_fn(model), channel=args.channel)
    params, state = model._params, model._base_model._state
    mid = SEQ_LEN // 2
    shuf = shuffled_motifs(OCT_SOX, K_SHUF)
    base, real, shufs = [], [], {j: [] for j in range(K_SHUF)}
    for _, r in peaks.iterrows():
        s = (int(r.start) + int(r.end)) // 2 - SEQ_LEN // 2
        seq = str(fa[r.chrom][s:s + SEQ_LEN]).upper()
        if len(seq) != SEQ_LEN or (set(seq) - set("ACGT")):
            continue
        base.append(seq); real.append(insert(seq, OCT_SOX, mid))
        for j, m in enumerate(shuf):
            shufs[j].append(insert(seq, m, mid))

    s0 = csum(fwd, params, state, base); sR = csum(fwd, params, state, real)
    sS = np.mean([csum(fwd, params, state, shufs[j]) for j in range(K_SHUF)], 0)
    lfc_real = np.log2((sR + EPS) / (s0 + EPS)); lfc_shuf = np.log2((sS + EPS) / (s0 + EPS))
    spec = lfc_real - lfc_shuf
    pd.DataFrame({"log2fc_real": lfc_real, "log2fc_shuf": lfc_shuf, "specificity": spec}).to_csv(
        OUT / f"ag_octsox_specificity{args.tag}.csv", index=False)
    w = wilcoxon(spec, alternative="less")
    print(f"AlphaGenome (DNase ch{args.channel}, 2kb) OCT-SOX over {len(spec)} enhancers: "
          f"raw median={np.median(lfc_real):+.4f}  SPECIFICITY median={np.median(spec):+.4f} "
          f"(Wilcoxon<0 p={w.pvalue:.3g})")
    print("done ->", OUT / f"ag_octsox_specificity{args.tag}.csv")


if __name__ == "__main__":
    main()
