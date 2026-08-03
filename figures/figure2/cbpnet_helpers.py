"""Helpers for Figure 2 ChromBPNet ablation / marginalization.

Vendored and trimmed from chrombpnet_2/utils.py -- the heavy alphagenome/grelu
top-level imports of the original are dropped so this module only needs
tangermeme + pandas + numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tangermeme.io import extract_loci
from tangermeme.ersatz import dinucleotide_shuffle
from tqdm import tqdm


def annotate_with_peak_distances(motif_hits_path, peaks_path) -> pd.DataFrame:
    """Annotate each FIMO motif hit with distance to the nearest peak summit."""
    hits = pd.read_csv(motif_hits_path, sep="\t")
    peaks = pd.read_csv(peaks_path, sep="\t", header=None)

    hits = hits[hits.sequence_name.astype(str).apply(lambda x: len(x) <= 5).values]
    hits = hits.dropna().copy()
    hits["start"] = hits["start"].astype(int)
    hits["stop"] = hits["stop"].astype(int)
    hits.reset_index(drop=True, inplace=True)

    # peak columns: 0=chrom 1=start 2=end 9=summit offset (narrowPeak)
    peak_groups = {chrom: p[[1, 2, 9]].values for chrom, p in peaks.groupby(0)}

    starts, ends, dists = [], [], []
    for row in tqdm(hits.itertuples(), total=len(hits)):
        if row.sequence_name not in peak_groups:
            starts.append(None); ends.append(None); dists.append(None)
            continue
        pk = peak_groups[row.sequence_name]
        summit = pk[:, 0] + pk[:, 2]
        distances = np.minimum(np.abs(summit - row.start), np.abs(summit - row.stop))
        j = int(np.argmin(distances))
        starts.append(pk[j, 0]); ends.append(pk[j, 1]); dists.append(distances[j])
    hits["closest_enhancer_start"] = starts
    hits["closest_enhancer_end"] = ends
    hits["distance"] = dists
    return hits


def ablate_motifs(motif_hits, motif_length, fasta, seq_len=2114, num_loci=1000,
                  random_state=None):
    """Extract loci centred on motif hits and dinuc-shuffle the central window.

    Returns (loci, loci_ablated) one-hot tensors of shape (N, 4, seq_len).
    Pass ``random_state`` to make the locus sample reproducible.
    """
    hits = (motif_hits.filter(["sequence_name", "start", "stop"])
            .rename(columns={"sequence_name": "chromosome", "stop": "end"})
            .astype({"start": "int", "end": "int"})
            .reset_index(drop=True))
    hits_sampled = hits.sample(min(num_loci, len(hits)), random_state=random_state)
    loci, idx = extract_loci(hits_sampled, str(fasta), in_window=seq_len,
                             verbose=True, return_filtered=True)
    hits_sampled = hits_sampled.iloc[idx.numpy()]
    mask = loci.sum(axis=1).sum(axis=1) == seq_len
    loci = loci[mask]

    a0 = seq_len // 2 - motif_length // 2
    a1 = seq_len // 2 + motif_length // 2
    loci_ablated = dinucleotide_shuffle(loci, start=a0, end=a1, n=1).squeeze()
    return loci, loci_ablated


def find_quantile(reference_array, query_value):
    """Fraction of reference values <= query_value (empirical CDF)."""
    return np.sum(reference_array <= query_value) / len(reference_array)


# OCT-SOX insertion offsets relative to the sequence midpoint, in the order the
# original notebooks applied them (used to build 0..8 insertion levels).
INSERTION_OFFSETS = [-50, +50, -100, +100, -150, +150, -200, +200]

# Example fibroblast enhancer for the accessibility-track panels (C/D). A strong,
# clean fibroblast DNase peak (signalValue ~50) selected by the screen in
# experiments/chrombpnet_profile_check: AlphaGenome's peak drops to ~60% under
# OCT-SOX insertion (leakage) while ChromBPNet holds ~87% (no leakage).
EXAMPLE_LOCUS = ("chr5", 140401460, 140401760)
# OCT-SOX insertion offsets from the sequence midpoint. IMPORTANT: these are in
# the peak FLANK (outside the accessible core) so the insertion does not
# overwrite the enhancer's own motifs -- otherwise the peak would drop in BOTH
# models just from deleting native sequence, not from OCT-SOX leakage.
TRACK_INSERT_OFFSETS = [-150, +150, -200, +200]


def scaled_profile_wrapper():
    """Return the ScaledProfileWrapper class (imports torch lazily)."""
    import torch

    class ScaledProfileWrapper(torch.nn.Module):
        """Per-position expected counts = softmax(logits) * (exp(logcounts) - 1).

        The count head is trained on ln(counts+1) (natural log; see bpnet-lite
        losses.log1pMSELoss and chrombpnet._log/_exp), so the correct total-count
        inversion is exp(logcounts)-1, NOT 2**logcounts-1.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model
            self.flatten = torch.nn.Flatten()
            self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self, X, X_ctl=None, **kwargs):
            logits, logcounts = self.model(X, X_ctl, **kwargs)
            B, C, L = logits.shape
            logits_flat = self.flatten(logits)
            logits_flat = logits_flat - logits_flat.mean(dim=-1, keepdims=True)
            probs_flat = self.softmax(logits_flat)
            counts = torch.exp(logcounts) - 1.0
            return (probs_flat * counts).view(B, C, L)

    return ScaledProfileWrapper
