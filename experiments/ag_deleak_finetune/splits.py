"""Chromosome holdout for the de-leaking experiment.

Fine-tuning (enhancers + fidelity anchors) uses TRAIN chromosomes only; ALL
evaluation (leakage on held-out enhancers, fidelity Pearson r, insertion effect,
attributions) uses TEST chromosomes only. No chromosome is shared, so there is no
train/test sequence contamination.
"""

from __future__ import annotations

TEST_CHROMS = {"chr8", "chr9"}
AUTOSOMES = ["chr" + str(i) for i in range(1, 23)]


def chrom_filter(df, which: str):
    """Filter a peaks dataframe (needs a 'chrom' column) by split."""
    df = df[df.chrom.isin(AUTOSOMES)]
    if which == "test":
        return df[df.chrom.isin(TEST_CHROMS)]
    if which == "train":
        return df[~df.chrom.isin(TEST_CHROMS)]
    if which == "all":
        return df
    raise ValueError(which)
