"""Train/test chromosome split (chr8/chr9 held out) for the fine-tuning experiment."""

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
