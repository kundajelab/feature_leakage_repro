"""Scalar DNase scoring of sequences with the local AlphaGenome v1 (JAX) model."""

from __future__ import annotations

import numpy as np

# cardiac fibroblast ontology term used throughout Figs 1-2
FIBROBLAST_ONTOLOGY = "CL:0002548"


def _output_types():
    from alphagenome.models import dna_output
    return dna_output.OutputType


def dnase_score(model, sequence: str, ontology: str = FIBROBLAST_ONTOLOGY,
                n_bins: int | None = None) -> float:
    """Scalar fibroblast DNase score for one sequence.

    ``n_bins=None`` sums the ENTIRE DNase track over all positions and matching
    tracks -- this matches the original ablation/marginalization notebooks
    (``r.dnase.values.sum()``). Pass an integer ``n_bins`` to instead sum only
    the central bins (used for the example-track panels).

    ``sequence`` must be a DNA string whose length is a supported AlphaGenome
    input size (e.g. 2048, 16384, ...).
    """
    OutputType = _output_types()
    out = model.predict_sequence(
        sequence=sequence,
        requested_outputs=[OutputType.DNASE],
        ontology_terms=[ontology],
    )
    vals = np.asarray(out.dnase.values)          # (positions, tracks)
    if n_bins is None:
        return float(vals.sum())
    if vals.ndim == 1:
        vals = vals[:, None]
    track = vals.mean(axis=1)
    c = len(track) // 2
    return float(track[c - n_bins // 2: c + n_bins // 2].sum())


def score_sequences(model, seqs, ontology: str = FIBROBLAST_ONTOLOGY,
                    n_bins: int | None = None, progress: bool = True):
    """Score an iterable of DNA strings -> np.ndarray of DNase scores."""
    it = seqs
    if progress:
        from tqdm import tqdm
        it = tqdm(seqs)
    return np.array([dnase_score(model, s, ontology, n_bins) for s in it])
