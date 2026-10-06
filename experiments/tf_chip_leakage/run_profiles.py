#!/usr/bin/env python
"""Plot AlphaGenome ChIP-seq profiles of a measured TF averaged over backgrounds, before vs after inserting another TF's motif.

Outputs (outputs/): marginalization_profiles_<cell>.pdf.

Run (env: featleak-alphagenome):
    python run_profiles.py --n-backgrounds 16 --cell-type K562
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from tangermeme.utils import random_one_hot, characters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from run import TF_MOTIFS  # noqa: E402
from run_ism import PAIRS  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384
ZOOM_BP = 4000            # window (bp) to show around the insertion


def chip_tf_values(model, seq, ontology, OutputType):
    out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.CHIP_TF],
                                 ontology_terms=[ontology])
    return np.asarray(out.chip_tf.values), np.asarray(out.chip_tf.metadata.transcription_factor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-backgrounds", type=int, default=16)
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_alphagenome()
    OutputType = _output_types()
    meta = model.output_metadata().chip_tf
    ontology = meta[meta.biosample_name == args.cell_type].ontology_curie.iloc[0]
    print(f"{args.cell_type} ({ontology})")

    X = random_one_hot((args.n_backgrounds, 4, SEQ_LEN), random_state=args.seed)
    mid = SEQ_LEN // 2
    inserted_motifs = {ins: TF_MOTIFS[ins] for ins, _, _ in PAIRS}

    tf_cols = None
    # before: predict each background once (gives all TF profiles)
    print("predicting backgrounds ...", flush=True)
    before_vals = []
    for k in range(args.n_backgrounds):
        v, tf_cols = chip_tf_values(model, characters(X[k]), ontology, OutputType)
        before_vals.append(v)
    before_vals = np.array(before_vals)                  # (N, P, tracks)

    # after: for each distinct inserted motif, predict each background+motif
    after_vals = {}
    for ins, motif in inserted_motifs.items():
        print(f"inserting {ins} motif ({motif}) ...", flush=True)
        m0 = mid - len(motif) // 2
        vs = []
        for k in range(args.n_backgrounds):
            s = list(characters(X[k]))
            for i, ch in enumerate(motif):
                s[m0 + i] = ch
            v, _ = chip_tf_values(model, "".join(s), ontology, OutputType)
            vs.append(v)
        after_vals[ins] = np.array(vs)

    # profile length -> bp-per-bin; zoom window in bins
    P = before_vals.shape[1]
    bp_per_bin = SEQ_LEN / P
    center = P // 2
    half = int(ZOOM_BP / 2 / bp_per_bin)
    x_bp = (np.arange(P) - center) * bp_per_bin

    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(8, 1.7 * len(PAIRS)), sharex=True)
    for ax, (ins, meas, label) in zip(np.atleast_1d(axes), PAIRS):
        cols = np.where(tf_cols == meas)[0]
        before = before_vals[:, :, cols].mean(axis=2).mean(axis=0)   # (P,)
        after = after_vals[ins][:, :, cols].mean(axis=2).mean(axis=0)
        sl = slice(center - half, center + half)
        ax.plot(x_bp[sl], before[sl], color="0.55", lw=1.4, label="before (background)")
        ax.plot(x_bp[sl], after[sl], color="#B2182B", lw=1.6, label="after (motif inserted)")
        ax.axvline(0, ls="--", color="#1764ab", lw=0.8)
        ax.set_ylabel(f"{meas}\nChIP", fontsize=8)
        ax.set_title(f"{label}", fontsize=8, loc="left")
        ax.legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel(f"position relative to insertion (bp)  —  {args.cell_type}")
    fig.tight_layout()
    fig.savefig(OUT / f"marginalization_profiles_{args.cell_type}.pdf")
    plt.close(fig)
    print("done ->", OUT / f"marginalization_profiles_{args.cell_type}.pdf")


if __name__ == "__main__":
    main()
