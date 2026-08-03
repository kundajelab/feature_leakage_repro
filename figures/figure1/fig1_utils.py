"""Shared helpers for Figure 1 (affinity-quartile sorted-scatter machinery).

Adapted from enformer_insertions/utils.py -- the parts that split FIMO motif
hits into affinity quartiles and render the sorted per-cell-type scatter used in
panels E-H.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # editable text in Illustrator
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

CHROMS = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]


def annotate_with_peak_distances(motif_hits_path, peaks_path):
    """Annotate FIMO hits with distance to the nearest narrowPeak summit."""
    from tqdm import tqdm

    hits = pd.read_csv(motif_hits_path, sep="\t")
    peaks = pd.read_csv(peaks_path, sep="\t", header=None)
    hits = hits[hits.sequence_name.astype(str).apply(lambda x: len(x) <= 5).values]
    hits = hits.dropna().copy()
    hits["start"] = hits["start"].astype(int)
    hits["stop"] = hits["stop"].astype(int)
    hits.reset_index(drop=True, inplace=True)
    peak_groups = {c: p[[1, 2, 9]].values for c, p in peaks.groupby(0)}
    dists = []
    for row in tqdm(hits.itertuples(), total=len(hits)):
        if row.sequence_name not in peak_groups:
            dists.append(None); continue
        pk = peak_groups[row.sequence_name]
        summit = pk[:, 0] + pk[:, 2]
        d = np.minimum(np.abs(summit - row.start), np.abs(summit - row.stop))
        dists.append(d.min())
    hits["distance"] = dists
    return hits


def sample_loci_by_affinity(fimo_tsv, sample_size=500, num_groups=4, seed=0):
    """Split FIMO motif hits into ``num_groups`` affinity (score) quantile bins.

    Returns a list of DataFrames (weakest -> strongest) each with columns
    chromosome/start/end/score, subsampled to ``sample_size``.
    """
    df = pd.read_csv(fimo_tsv, sep="\t")
    df = df[df.sequence_name.isin(CHROMS)]
    df = (df.filter(["sequence_name", "start", "stop", "score"])
          .rename(columns={"sequence_name": "chromosome", "stop": "end"})
          .astype({"start": "int", "end": "int"})
          .reset_index(drop=True))
    qs = df["score"].quantile([(i + 1) / num_groups for i in range(num_groups)]).values
    out = []
    for i in range(num_groups):
        if i == 0:
            grp = df[df["score"] <= qs[i]]
        else:
            grp = df[(df["score"] > qs[i - 1]) & (df["score"] <= qs[i])]
        n = min(sample_size, len(grp))
        out.append(grp.sample(n, random_state=seed).reset_index(drop=True))
    return out


def plot_sorted_scatters(results, title, path, base_color="Reds"):
    """Sorted per-cell-type scatter across affinity groups (panels E-H).

    ``results`` is a list of per-cell-type effect arrays, weakest-affinity first.
    Cell types are sorted by the strongest-affinity group's effect.
    """
    sort_idx = results[-1].argsort()[::-1]
    x = np.arange(len(sort_idx))
    n = len(results)
    labels = ["1stQuartile(Weakest)", "2ndQuartile", "3rdQuartile",
              "4thQuartile(Strongest)"]
    fig, ax = plt.subplots(figsize=(4, 3.2))
    for i, result in enumerate(results):
        color = plt.get_cmap(base_color)((n + i) / (2 * n))
        ax.scatter(x, result[sort_idx], color=color, s=6,
                   label=labels[i] if n == 4 else f"group {i}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Cell Type"); ax.set_ylabel("Attribution")
    ax.set_title(title)
    ax.legend(fontsize=5, markerscale=1.5, loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
