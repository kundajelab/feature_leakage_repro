"""The shared Figure 2 enhancer set: the top-N dermal-fibroblast DNase peaks (ENCSR000EPO).

Every model in the raw-insertion analysis (Fig 2b, Supp 2a) is scored on exactly these
sequences, centred on the peak midpoint, with OCT-SOX written at the same offsets.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from common import config  # noqa: E402

OFFSETS = [-150, 150, -200, 200]     # OCT-SOX copies, bp from the enhancer centre


def top_dermis_enhancers(n: int = 120) -> pd.DataFrame:
    """Top-n ENCFF184KAM peaks on chr1-22 by signalValue; columns chrom/start/end/mid/locus."""
    pk = pd.read_csv(config.resolve("peaks_dermis"), sep="\t", header=None)
    pk = pk.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    pk = pk[pk.chrom.isin([f"chr{i}" for i in range(1, 23)])]
    top = pk.sort_values("signal", ascending=False).head(n).reset_index(drop=True)
    top["mid"] = (top.start + top.end) // 2
    top["locus"] = top.chrom + ":" + top.start.astype(str)
    return top[["chrom", "start", "end", "mid", "signal", "locus"]]


def window(fasta, chrom: str, mid: int, length: int) -> str:
    """Uppercase sequence of `length` bp centred on `mid`, N-padded past chromosome ends."""
    s = mid - length // 2
    q = str(fasta[chrom][max(s, 0):s + length]).upper()
    return "N" * (length - len(q)) + q if s < 0 else q + "N" * (length - len(q))


def insert(seq: str, motif: str) -> str:
    """Write `motif` (forward strand, replacing bases) at every OFFSET from the centre."""
    c, s = len(seq) // 2, list(seq)
    for o in OFFSETS:
        s[c + o:c + o + len(motif)] = list(motif)
    return "".join(s)
