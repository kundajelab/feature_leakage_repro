"""Composition-matched shuffled control motifs for insertion experiments."""

from __future__ import annotations

import numpy as np


def shuffled_motifs(motif: str, k: int = 3, seed0: int = 100) -> list[str]:
    """Return `k` mononucleotide-shuffled variants of `motif` (fixed seeds)."""
    outs = []
    for j in range(k):
        rng = np.random.default_rng(seed0 + j)
        outs.append("".join(rng.permutation(list(motif))))
    return outs
