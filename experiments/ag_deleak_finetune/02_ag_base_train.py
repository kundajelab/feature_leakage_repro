#!/usr/bin/env python
"""Predict pre-fine-tuning AlphaGenome DNase profiles on the training-chromosome enhancers.

Outputs (outputs/): ag_base_train.npz.

Run (env: alphagenome-ft):
    python 02_ag_base_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "alphagenome_finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from common import config  # noqa: E402
import ft_core  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 2048
BATCH = 8
C0 = SEQ_LEN // 2


def main() -> None:
    import pyfaidx
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    oracle = pd.read_csv(OUT / "cbp_oracle_train.csv")

    model = ft_core.load_model()
    fwd = ft_core.make_dnase_forward(ft_core.get_apply_fn(model))
    params, state = model._params, model._base_model._state

    seqs, keep = [], []
    for _, r in oracle.iterrows():
        center = (int(r.start) + int(r.end)) // 2
        s = center - SEQ_LEN // 2
        seq = str(fa[r.chrom][s:s + SEQ_LEN]).upper()
        ok = len(seq) == SEQ_LEN and not (set(seq) - set("ACGT"))
        keep.append(ok)
        if ok:
            seqs.append(seq)
    oracle = oracle[np.array(keep)].reset_index(drop=True)

    profs = []
    for i in range(0, len(seqs), BATCH):
        chunk = seqs[i:i + BATCH]
        X = jnp.asarray(ft_core.encode_batch(chunk))
        org = jnp.zeros((len(chunk),), jnp.int32)
        ch, _, _ = fwd(params, state, X, org)
        profs.append(np.asarray(ch, np.float32))
    profs = np.concatenate(profs, 0)

    np.savez(OUT / "ag_base_train.npz",
             chrom=oracle.chrom.values, start=oracle.start.values, end=oracle.end.values,
             cbp_lfc_real=oracle.cbp_lfc_real.values.astype(np.float32),
             ag_base_profiles=profs.astype(np.float32))
    print(f"AG_base on {len(seqs)} train enhancers ({sorted(oracle.chrom.unique())}) "
          f"-> ag_base_train.npz")


if __name__ == "__main__":
    main()
