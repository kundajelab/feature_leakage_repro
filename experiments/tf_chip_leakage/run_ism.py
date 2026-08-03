#!/usr/bin/env python
"""ISM proof that TF-motif ChIP leakage in AlphaGenome is REAL (motif-specific).

For a (inserted motif TF, measured ChIP TF) pair, insert the inserted TF's motif
into many random synthetic sequences, then compute in-silico saturation
mutagenesis of the MEASURED TF's ChIP-seq output over the motif window, and
average the per-position contribution across sequences.

If the measured TF's ChIP prediction attributes to the *distinct* inserted
motif (a sequence it should not care about), the leakage is genuine and
motif-specific -- not a generic response to "some insertion". Each pair is
chosen so the inserted motif is a DIFFERENT motif family from the measured TF.

An on-target control (measured TF's own motif -> measured TF) validates that the
ISM recovers the correct motif.

Run (env: featleak-alphagenome):
    python run_ism.py --n-seqs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tangermeme.utils import random_one_hot, characters  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from run import TF_MOTIFS, TF_FAMILY  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384
CENTER_BP = 2001
BASES = "ACGT"

# (inserted motif TF, measured ChIP TF, label). First is an on-target control;
# the rest insert a motif from a DIFFERENT family than the measured TF.
PAIRS = [
    ("MAX",  "MAX",  "control: MAX motif -> MAX ChIP"),
    ("JUN",  "MAX",  "AP-1 motif -> MAX ChIP"),
    ("CTCF", "JUND", "CTCF motif -> JUND(AP-1) ChIP"),
    ("REST", "SPI1", "REST motif -> SPI1(ETS) ChIP"),
    ("NRF1", "YY1",  "NRF1 motif -> YY1 ChIP"),
]


def tf_chip_scalar(model, seq, ontology, tf, OutputType):
    """Central-window sum of the measured TF's ChIP output (avg over its tracks).

    Reads the TF columns from the RETURNED output metadata so the indexing is
    always aligned with the prediction (not a pre-filtered dataframe).
    """
    out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.CHIP_TF],
                                 ontology_terms=[ontology])
    vals = np.asarray(out.chip_tf.values)          # (positions, tracks)
    cols = np.where(np.asarray(out.chip_tf.metadata.transcription_factor) == tf)[0]
    c = vals.shape[0] // 2
    half = max(int(CENTER_BP / 2 * vals.shape[0] / SEQ_LEN), 1)
    central = vals[c - half:c + half].sum(axis=0)  # (tracks,)
    return float(central[cols].mean())


def averaged_ism(model, ontology, inserted_tf, measured_tf, n_seqs, seed,
                 OutputType, meta):
    motif = TF_MOTIFS[inserted_tf]
    win = len(motif) + 4
    mid = SEQ_LEN // 2
    m0 = mid - len(motif) // 2                      # motif start
    w0, w1 = m0 - 2, m0 + len(motif) + 2            # ISM window

    contribs = []
    X = random_one_hot((n_seqs, 4, SEQ_LEN), random_state=seed)
    for k in range(n_seqs):
        # insert the motif
        s = list(characters(X[k]))
        for i, ch in enumerate(motif):
            s[m0 + i] = ch
        ref = tf_chip_scalar(model, "".join(s), ontology, measured_tf, OutputType)
        imp = np.zeros(w1 - w0)
        ref_oh = np.zeros((4, w1 - w0))
        for j, pos in enumerate(range(w0, w1)):
            rb = s[pos]
            ref_oh[BASES.index(rb), j] = 1.0
            alts = []
            for b in BASES:
                if b == rb:
                    continue
                s2 = s.copy(); s2[pos] = b
                alts.append(tf_chip_scalar(model, "".join(s2), ontology, measured_tf,
                                           OutputType))
            imp[j] = ref - np.mean(alts)            # >0: ref base drives measured TF
        contribs.append(ref_oh * imp[None, :])
        print(f"    [{inserted_tf}->{measured_tf}] seq {k+1}/{n_seqs} ref={ref:.1f}",
              flush=True)
    return np.mean(contribs, axis=0), motif


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=10)
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_alphagenome()
    OutputType = _output_types()
    meta = model.output_metadata().chip_tf
    cell = meta[meta.biosample_name == args.cell_type].reset_index(drop=True)
    ontology = cell.ontology_curie.iloc[0]
    # column indices are into the FULL requested output; request returns tracks for
    # this ontology only, in the same order as `cell`.
    print(f"{args.cell_type} ({ontology}): {len(cell)} CHIP_TF tracks")

    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(9, 1.5 * len(PAIRS)))
    for ax, (ins, meas, label) in zip(np.atleast_1d(axes), PAIRS):
        print(f"\n=== {label} ===", flush=True)
        contrib, motif = averaged_ism(model, ontology, ins, meas, args.n_seqs,
                                      args.seed, OutputType, cell)
        plot_logo(torch.tensor(contrib, dtype=torch.float32), ax=ax)
        strong = "REAL LEAKAGE" if ins != meas and contrib.sum() > 0 else ""
        ax.set_title(f"{label}   (inserted {ins} motif '{motif}', "
                     f"Σattr={contrib.sum():.1f}) {strong}", fontsize=8)
        np.save(OUT / f"ism_{ins}_to_{meas}.npy", contrib)
    fig.tight_layout()
    fig.savefig(OUT / f"ism_leakage_examples_{args.cell_type}.pdf")
    plt.close(fig)
    print("\ndone ->", OUT / f"ism_leakage_examples_{args.cell_type}.pdf")


if __name__ == "__main__":
    main()
