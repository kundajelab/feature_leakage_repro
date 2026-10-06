#!/usr/bin/env python
"""Compute ISM around an endogenous OCT-SOX site for the original and fine-tuned heads (Figure 7c).

Outputs (outputs/dermis_torch/endo_screen/): ism_<chrom>_<mid>.npz.

Run (env: featleak-agtorch, GPU):
    python endo_ism.py chr9:92379377
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finetune_dermis_torch as ft  # noqa: E402
from endo_screen import HALF, OUT  # noqa: E402

FLANK = 3


def main(sites):
    import pyfaidx
    import torch
    ag = ft.AG()
    w = torch.load(ft.ft_head_path())
    wf, bf, sf = w["w"].cuda(), w["b"].cuda(), w["s"].cuda()
    fa = pyfaidx.Fasta(str(ft.config.resolve("hg38_fasta")))
    ml = len(ft.OCT_SOX)

    def score(seq):
        with torch.no_grad():
            e = ag.embed(seq, half=HALF).cuda()
            return (ag.head_pred(e, ag.w0, ag.b0, ag.s0).sum().item(),
                    ag.head_pred(e, wf, bf, sf).sum().item())

    OUT.mkdir(parents=True, exist_ok=True)
    for site in sites:
        chrom, mid = site.split(":"); mid = int(mid)
        q = ft.fetch(fa, chrom, mid)
        p0 = ft.C0 - ml // 2 - FLANK
        n = ml + 2 * FLANK
        base = score(q)
        pre = np.zeros((n, 4)); post = np.zeros((n, 4))
        for i in range(n):
            p = p0 + i
            for k, b in enumerate("ACGT"):
                pre[i, k], post[i, k] = base if b == q[p] else score(q[:p] + b + q[p + 1:])
        np.savez(OUT / f"ism_{chrom}_{mid}.npz", pre=np.log2(pre), ft=np.log2(post),
                 ref=q[p0:p0 + n], chrom=chrom, start=mid - ft.C0 + p0, flank=FLANK)
        print("saved", site, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
