#!/usr/bin/env python
"""Gradient x input attribution of the FOS output and TF-MoDISco motif discovery (Figure 4b).

Computes contribution scores for the union-trained two-task Keras model with respect to
the FOS output logit, then runs TF-MoDISco over them to recover the motifs that drive the
FOS prediction.

Outputs (outputs/): modisco_fos_logit.h5, fos_logit_gradxinput.npz, fos_logit_seqs.npz
and the TF-MoDISco report in outputs/modisco_report_fos/.

Run (env: featleak-tf):
    python run_modisco_keras.py
    python run_modisco_keras.py --stage report      # report only, from an existing h5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs"

# The Keras label matrix orders its columns FOS second; the final dense layer carries the
# logits before the sigmoid activation.
FOS_COL = 1
LOGIT_LAYER = "dense_2"
WINDOW = 300

BASES = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}


def one_hot(seq: str) -> np.ndarray:
    """One-hot encode, mapping any base outside ACGT to an all-zero column."""
    return np.array([BASES.get(n, [0, 0, 0, 0]) for n in seq], dtype=np.float32)


def build_sequences() -> np.ndarray:
    """The labelled peak windows, in the model's own train/val/test order."""
    peaks_dir = Path(config.resolve("keras_fibr_ipsc_dir", must_exist=True))
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta", must_exist=True)))
    bed = pd.read_csv(peaks_dir / "labeled_peaks.bed", sep="\t", header=0)
    labels = np.load(peaks_dir / "labeled_peaks.npy")

    seqs = np.array([
        one_hot(str(fa[r["chrom"]][(r["start"] + r["end"]) // 2 - WINDOW // 2:
                                   (r["start"] + r["end"]) // 2 + WINDOW // 2]))
        for _, r in bed.iterrows()
    ])
    x_tr, x_te, y_tr, _ = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    x_tr, x_va, _, _ = train_test_split(x_tr, y_tr, test_size=0.2, random_state=42)
    return np.concatenate([x_tr, x_va, x_te])


def grad_x_input(model, seqs: np.ndarray, out_index: int, batch: int = 512) -> np.ndarray:
    """Gradient of output ``out_index`` with respect to the input, times the input."""
    import tensorflow as tf

    parts = []
    for i in range(0, len(seqs), batch):
        xb = tf.convert_to_tensor(seqs[i:i + batch], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            y = model(xb)[:, out_index]
        parts.append((tape.gradient(y, xb) * xb).numpy())
        print(f"    {min(i + batch, len(seqs))}/{len(seqs)}", flush=True)
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "attribute", "modisco", "report"],
                    default="all")
    ap.add_argument("--max-seqlets", type=int, default=50000)
    ap.add_argument("--window", type=int, default=400,
                    help="TF-MoDISco sliding-window width (bp).")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    attr_path = OUT / "fos_logit_gradxinput.npz"
    seqs_path = OUT / "fos_logit_seqs.npz"
    h5_path = OUT / "modisco_fos_logit.h5"

    if args.stage in ("all", "attribute"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import tensorflow as tf

        print("[1/3] building input sequences ...", flush=True)
        seqs = build_sequences()
        print(f"    {seqs.shape[0]} sequences of {seqs.shape[1]} bp", flush=True)

        model = tf.keras.models.load_model(
            str(config.resolve("keras_multitask_model", must_exist=True)))
        logit_model = tf.keras.Model(model.input, model.get_layer(LOGIT_LAYER).output)

        print("[2/3] gradient x input on the FOS logit ...", flush=True)
        attr = grad_x_input(logit_model, seqs, out_index=FOS_COL)
        np.savez_compressed(attr_path, attr.transpose(0, 2, 1))
        np.savez_compressed(seqs_path, seqs.transpose(0, 2, 1))
        print(f"    -> {attr_path.name}, {seqs_path.name}", flush=True)

    if args.stage in ("all", "modisco"):
        import modiscolite

        print("[3/3] TF-MoDISco ...", flush=True)
        seqs = np.load(seqs_path)["arr_0"]
        attr = np.load(attr_path)["arr_0"]
        pos, neg = modiscolite.tfmodisco.TFMoDISco(
            hypothetical_contribs=attr.transpose(0, 2, 1).astype(np.float32),
            one_hot=seqs.transpose(0, 2, 1).astype(np.float32),
            max_seqlets_per_metacluster=args.max_seqlets,
            sliding_window_size=20, flank_size=5,
            target_seqlet_fdr=0.05, n_leiden_runs=2,
        )
        modiscolite.io.save_hdf5(str(h5_path), pos, neg, window_size=args.window)
        print(f"    -> {h5_path.name}", flush=True)

    if args.stage in ("all", "report"):
        import modiscolite

        report_dir = OUT / "modisco_report_fos"
        report_dir.mkdir(exist_ok=True)
        modiscolite.report.report_motifs(
            str(h5_path), str(report_dir) + "/",
            img_path_suffix="./", top_n_matches=3,
            meme_motif_db=str(config.resolve("jaspar_meme", must_exist=True)),
        )
        print(f"    report -> {report_dir}", flush=True)

    print("done ->", OUT)


if __name__ == "__main__":
    main()
