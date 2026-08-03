#!/usr/bin/env python
"""On-target control: do the SAME ChIP tasks correctly predict their target TF?

For each measured TF used in the leakage plots (MAX, JUND, SPI1, YY1), this:
  1. prints/saves EXACTLY which AlphaGenome CHIP_TF tracks are averaged for it
     (name + assay + TF + biosample) so the task identity is verifiable, and
  2. marginalizes its predicted ChIP profile over N random backgrounds for three
     conditions -- background (before), its OWN motif inserted (on-target), and
     the leaking motif inserted (off-target) -- and plots all three per TF.

If the on-target (own-motif) curve peaks strongly, the task genuinely predicts
its target TF; the off-target curve peaking too is the leakage.

Run (env: featleak-alphagenome):
    python run_ontarget.py --n-backgrounds 16 --cell-type K562
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tangermeme.utils import random_one_hot, characters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from run import TF_MOTIFS  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384
ZOOM_BP = 4000

# measured TF -> (its OWN-motif TF, the LEAKING-motif TF)
TASKS = {
    "MAX":  ("MAX",  "JUN"),    # own = E-box; leak = AP-1
    "JUND": ("JUND", "CTCF"),   # own = AP-1; leak = CTCF
    "SPI1": ("SPI1", "REST"),   # own = ETS/PU.1; leak = REST NRSE
    "YY1":  ("YY1",  "NRF1"),   # own = YY1; leak = NRF1
}


def chip_values(model, seq, ontology, OutputType):
    out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.CHIP_TF],
                                 ontology_terms=[ontology])
    return np.asarray(out.chip_tf.values), out.chip_tf.metadata


def insert(seq_chars, motif):
    s = list(seq_chars)
    m0 = SEQ_LEN // 2 - len(motif) // 2
    for i, ch in enumerate(motif):
        s[m0 + i] = ch
    return "".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-backgrounds", type=int, default=16)
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_alphagenome()
    OutputType = _output_types()
    meta_all = model.output_metadata().chip_tf
    ontology = meta_all[meta_all.biosample_name == args.cell_type].ontology_curie.iloc[0]

    # ---- 1. verify the tracks used per measured TF ----
    lines = [f"Tracks averaged per measured TF in {args.cell_type} ({ontology}):\n"]
    for tf in TASKS:
        sub = meta_all[(meta_all.biosample_name == args.cell_type) &
                       (meta_all.transcription_factor == tf)]
        lines.append(f"== {tf}: {len(sub)} track(s) ==")
        for _, r in sub.iterrows():
            lines.append(f"   name='{r['name']}' | assay='{r['Assay title']}' | "
                         f"TF={r.transcription_factor} | {r.biosample_name}")
        lines.append("")
    (OUT / f"tracks_used_{args.cell_type}.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- 2. marginalization profiles: before / on-target / leakage ----
    X = random_one_hot((args.n_backgrounds, 4, SEQ_LEN), random_state=args.seed)
    bg = [characters(X[k]) for k in range(args.n_backgrounds)]

    unique_motifs = {t for own, leak in TASKS.values() for t in (own, leak)}
    motif_str = {t: TF_MOTIFS[t] for t in unique_motifs}

    print("predicting backgrounds ...", flush=True)
    before = []
    tf_col = None
    for s in bg:
        v, m = chip_values(model, s, ontology, OutputType)
        before.append(v); tf_col = np.asarray(m.transcription_factor)
    before = np.array(before)

    after = {}
    for t, motif in motif_str.items():
        print(f"inserting {t} motif ({motif}) ...", flush=True)
        vs = [chip_values(model, insert(s, motif), ontology, OutputType)[0] for s in bg]
        after[t] = np.array(vs)

    P = before.shape[1]
    bp = SEQ_LEN / P
    c = P // 2
    half = int(ZOOM_BP / 2 / bp)
    x = (np.arange(P) - c) * bp
    sl = slice(c - half, c + half)

    fig, axes = plt.subplots(len(TASKS), 1, figsize=(8, 1.9 * len(TASKS)), sharex=True)
    for ax, (meas, (own, leak)) in zip(np.atleast_1d(axes), TASKS.items()):
        cols = np.where(tf_col == meas)[0]
        bprof = before[:, :, cols].mean(2).mean(0)
        onprof = after[own][:, :, cols].mean(2).mean(0)
        leakprof = after[leak][:, :, cols].mean(2).mean(0)
        ax.plot(x[sl], bprof[sl], color="0.6", lw=1.3, label="background")
        ax.plot(x[sl], onprof[sl], color="#1B7837", lw=1.8,
                label=f"on-target: {own} motif")
        ax.plot(x[sl], leakprof[sl], color="#B2182B", lw=1.8,
                label=f"leakage: {leak} motif")
        ax.axvline(0, ls="--", color="0.4", lw=0.7)
        ax.set_ylabel(f"{meas}\nChIP", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel(f"position relative to insertion (bp)  —  {args.cell_type} "
                        f"TF ChIP-seq")
    axes[0].set_title("Same task predicts its target TF (green) AND leaks a "
                      "distinct motif (red)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"ontarget_vs_leakage_profiles_{args.cell_type}.pdf")
    plt.close(fig)
    print("done ->", OUT / f"ontarget_vs_leakage_profiles_{args.cell_type}.pdf")


if __name__ == "__main__":
    main()
