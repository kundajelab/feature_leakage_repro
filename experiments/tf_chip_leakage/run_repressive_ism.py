#!/usr/bin/env python
"""ISM of the repressive-leakage motifs, averaged over marginalization backgrounds.

For each repressive pair (A represses B): build synthetic sequences = random
background + B's own motif at centre + A's motif in the flank (mid+120), then
saturation-mutate the A-motif window and read out, from the SAME prediction,
BOTH the inserted TF A's ChIP signal and the target TF B's ChIP signal. Averaged
over many backgrounds this gives two contribution logos over the A motif:

  * A -> A  (positive control): mutating A's bases lowers A's own ChIP -> a
    coherent POSITIVE A-motif logo (the model reads the motif correctly).
  * A -> B  (leakage): if the repression is motif-specific, mutating the SAME A
    bases RAISES B's ChIP -> a coherent NEGATIVE A-motif logo.

Same bases, opposite sign across the two tasks = representation superposition.

Run (env: featleak-alphagenome):
    python run_repressive_ism.py --n-bg 10
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
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from run import TF_MOTIFS  # noqa: E402
from run_repressive import chip, central_for_tf, SEQ_LEN  # noqa: E402

OUT = Path(__file__).parent / "outputs"
A_OFFSET = 120           # A motif inserted at mid + A_OFFSET (flank of B)
PAD = 3
BASES = "ACGT"
PAIRS = [("NRF1", "REST"), ("YY1", "REST"), ("MYC", "REST"),
         ("REST", "CTCF"), ("MAX", "NRF1")]


def ism_pair(model, ontology, OT, A, B, n_bg, seed):
    X = random_one_hot((n_bg, 4, SEQ_LEN), random_state=seed)
    mid = SEQ_LEN // 2
    X = substitute(X, TF_MOTIFS[B], start=mid - len(TF_MOTIFS[B]) // 2)   # B at centre
    # match the marginalization context: A at BOTH flanks (+/-A_OFFSET); we ISM
    # the +A_OFFSET copy while the -A_OFFSET copy stays intact.
    X = substitute(X, TF_MOTIFS[A], start=mid - A_OFFSET - len(TF_MOTIFS[A]) // 2)
    a0 = mid + A_OFFSET - len(TF_MOTIFS[A]) // 2
    X = substitute(X, TF_MOTIFS[A], start=a0)
    w0, w1 = a0 - PAD, a0 + len(TF_MOTIFS[A]) + PAD
    win = w1 - w0
    contribA, contribB = [], []
    for n in range(n_bg):
        seq = characters(X[n])
        cA = np.zeros((4, win)); cB = np.zeros((4, win))
        for i in range(win):
            for b in range(4):
                mut = seq[:w0 + i] + BASES[b] + seq[w0 + i + 1:]
                track = chip(model, mut, ontology, OT)
                cA[b, i] = central_for_tf(track, A)
                cB[b, i] = central_for_tf(track, B)
        ref = [BASES.index(seq[w0 + i]) for i in range(win)]
        oh = np.zeros((4, win)); oh[ref, range(win)] = 1.0
        contribA.append((cA - cA.mean(axis=0, keepdims=True)) * oh)   # ref-base contribution
        contribB.append((cB - cB.mean(axis=0, keepdims=True)) * oh)
    return np.mean(contribA, axis=0), np.mean(contribB, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=10)
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_alphagenome(); OT = _output_types()
    meta = model.output_metadata().chip_tf
    ontology = meta[meta.biosample_name == args.cell_type].ontology_curie.iloc[0]

    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(13, 2.0 * len(PAIRS)), squeeze=False)
    for row, (A, B) in enumerate(PAIRS):
        cA, cB = ism_pair(model, ontology, OT, A, B, args.n_bg, args.seed)
        np.savez(OUT / f"repressive_ism_{A}_{B}.npz", onA=cA, onB=cB)
        plot_logo(torch.tensor(cA, dtype=torch.float32), ax=axes[row, 0])
        axes[row, 0].set_title(f"{A} motif ISM → {A} ChIP (positive control)   "
                               f"net={cA.sum():+.0f}", fontsize=9)
        plot_logo(torch.tensor(cB, dtype=torch.float32), ax=axes[row, 1])
        axes[row, 1].set_title(f"{A} motif ISM → {B} ChIP (leakage)   "
                               f"net={cB.sum():+.0f}", fontsize=9)
        print(f"{A}->{B}: onA net={cA.sum():+.1f}  onB net={cB.sum():+.1f}", flush=True)
    fig.suptitle(f"ISM over the inserted motif — repressive cross-TF leakage "
                 f"(AlphaGenome {args.cell_type}, avg over {args.n_bg} backgrounds)\n"
                 "same bases ACTIVATE the motif's own TF (left) but REPRESS the "
                 "off-target TF (right)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"repressive_ism_{args.cell_type}.pdf")
    plt.close(fig)
    print("done ->", OUT / f"repressive_ism_{args.cell_type}.pdf")


if __name__ == "__main__":
    main()
