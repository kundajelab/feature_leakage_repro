"""GC-matched random negative regions sampled from a genome FASTA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pybedtools
import pyfaidx

MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def _gc(seq: str) -> float:
    s = seq.upper()
    if not s:
        return np.nan
    return (s.count("G") + s.count("C")) / len(s)


def gc_matched_negatives(fasta_path: str, peaks: pd.DataFrame, *, width: int = 300,
                         n_target: int = 40000, n_candidates: int = 300000,
                         exclude_window: int = 1000, gc_bins: int = 20,
                         seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame of GC-matched negative windows (chrom,start,end).

    Windows are drawn at random from the main chromosomes, excluded if they lie
    within ``exclude_window`` bp of any peak or contain an N, then resampled so
    their GC-content histogram matches that of ``peaks``.
    """
    rng = np.random.default_rng(seed)
    fa = pyfaidx.Fasta(fasta_path)
    chroms = [c for c in MAIN_CHROMS if c in fa]
    sizes = np.array([len(fa[c]) for c in chroms], dtype=np.int64)

    peaks = peaks[["chrom", "start", "end"]].copy()
    bins = np.linspace(0.0, 1.0, gc_bins + 1)

    # GC histogram of the peaks (the distribution we want to match).
    pgc = []
    for c, s, e in peaks.itertuples(index=False):
        if c in fa:
            g = _gc(str(fa[c][int(s):int(e)]))
            if not np.isnan(g):
                pgc.append(g)
    peak_hist, _ = np.histogram(pgc, bins=bins)
    peak_frac = peak_hist / max(peak_hist.sum(), 1)

    # Random candidate windows on the main chromosomes.
    ci = rng.integers(0, len(chroms), size=n_candidates)
    starts = (rng.random(n_candidates) * (sizes[ci] - width)).astype(np.int64)
    cand = pd.DataFrame({"chrom": [chroms[i] for i in ci],
                         "start": starts, "end": starts + width})

    # Drop candidates near any peak.
    cbt = pybedtools.BedTool.from_dataframe(cand)
    pbt = pybedtools.BedTool.from_dataframe(peaks)
    cand = cbt.window(pbt, w=exclude_window, v=True).to_dataframe(
        names=["chrom", "start", "end"])

    # GC content of surviving candidates; drop windows with Ns.
    gcs, keep = [], []
    for c, s, e in cand[["chrom", "start", "end"]].itertuples(index=False):
        seq = str(fa[c][int(s):int(e)]).upper()
        if "N" in seq:
            gcs.append(np.nan); keep.append(False)
        else:
            gcs.append(_gc(seq)); keep.append(True)
    cand = cand.assign(gc=gcs)[keep].reset_index(drop=True)

    # Resample per GC bin to match the peak distribution.
    cand_bin = np.clip(np.digitize(cand["gc"].values, bins) - 1, 0, gc_bins - 1)
    target = (peak_frac * n_target).astype(int)
    picks = []
    for b in range(gc_bins):
        idx = np.where(cand_bin == b)[0]
        if len(idx) == 0 or target[b] == 0:
            continue
        picks.append(rng.choice(idx, size=min(target[b], len(idx)), replace=False))
    sel = np.concatenate(picks) if picks else np.array([], dtype=int)
    return cand.iloc[sel][["chrom", "start", "end"]].reset_index(drop=True)
