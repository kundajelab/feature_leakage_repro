"""Composition-matched shuffled negative-control motifs.

A mononucleotide shuffle keeps the exact base composition of a motif (e.g. OCT-SOX)
but destroys the motif itself, so any residual prediction change after inserting a
shuffled motif is a GENERIC "insert 14 bp of this composition" artifact rather than a
motif-specific effect. We use a few fixed seeds and average, so the control never
hinges on one lucky permutation.

Used to turn a raw insertion log2FC into a shuffle-corrected SPECIFICITY:

    specificity = log2FC(real motif inserted) - log2FC(shuffled motif inserted)

which is still in log2 fold-change units, but with the generic-disruption baseline
subtracted out. This matters for base-resolution models like ChromBPNet, whose raw
log2FC is dominated by generic insertion disruption.
"""

from __future__ import annotations

import numpy as np


def shuffled_motifs(motif: str, k: int = 3, seed0: int = 100) -> list[str]:
    """Return `k` mononucleotide-shuffled variants of `motif` (fixed seeds)."""
    outs = []
    for j in range(k):
        rng = np.random.default_rng(seed0 + j)
        outs.append("".join(rng.permutation(list(motif))))
    return outs
