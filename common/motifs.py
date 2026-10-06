"""Motif consensus strings and Vierstra archetype lookups shared across figures."""

from __future__ import annotations

import functools

import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Canonical consensus strings used for insertion / ablation experiments.
# Consensus sequences used for motif insertion and ablation.
# --------------------------------------------------------------------------- #
AP1 = "ATGAGTCAC"          # AP-1 / FOS::JUN (counting + entanglement)
AP1_SHORT = "TGAGTCA"      # 7 bp core used by some ablation windows
OCT_SOX = "ATTGTTATGCAAAT"  # OCT4::SOX2 heterodimer consensus (insertions)
CTCF = "GCCGCCAGGGGGCGCCG"  # CTCFL_HUMAN.H11MO.0.A consensus
HNF4 = "CAAAGTCCA"          # HNF4A core

# Motif window widths (bp) used by the affinity-quartile ablation machinery,
# keyed by the lower-case association-file motif name.
ABLATE_WINDOWS = {"ap-1": 6, "ctcf": 10, "hnf4": 7, "oct-sox": 8}


@functools.lru_cache(maxsize=1)
def load_vierstra_motifs() -> pd.DataFrame:
    """Load the Vierstra archetype table filtered to annotated seed motifs.

    Returns a DataFrame with columns ``Cluster_ID``, ``Consensus``, ``Motif``.
    """
    annots = pd.read_excel(
        config.resolve("vierstra_motif_annotations_xlsx", must_exist=True),
        engine="openpyxl",
    )
    archetypes = pd.read_csv(
        config.resolve("vierstra_motifs_tsv", must_exist=True), sep="\t"
    )
    return archetypes[archetypes["Motif"].isin(annots["Seed_motif"])][
        ["Cluster_ID", "Consensus", "Motif"]
    ].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def motif_to_seq() -> dict[str, str]:
    """Map archetype motif name (e.g. ``CTCFL_HUMAN.H11MO.0.A``) -> consensus."""
    df = load_vierstra_motifs()
    consensus_to_cluster = df.set_index("Consensus")["Motif"].to_dict()
    return {v: k for k, v in consensus_to_cluster.items()}


def candidate_motif_seqs() -> list[str]:
    """List of consensus strings used as candidate insertions (Fig 6 design)."""
    return load_vierstra_motifs()["Consensus"].tolist()


def filtered_motifs_for_ap1(threshold: int = 50) -> dict[str, str]:
    """Archetype motifs whose consensus is *dissimilar* to AP-1.

    Used by Fig 3 to build the leakage-scan set: motifs with a
    ``fuzz.partial_ratio`` to AP-1 >= ``threshold`` are dropped so the counting
    task cannot be trivially confounded by AP-1-like sequences. Returns a dict
    mapping archetype motif name -> consensus (~37 motifs at threshold 50).
    """
    from fuzzywuzzy import fuzz

    out = {}
    for name, seq in motif_to_seq().items():
        if fuzz.partial_ratio(seq.upper(), AP1.upper()) < threshold:
            out[name] = seq
    return out
