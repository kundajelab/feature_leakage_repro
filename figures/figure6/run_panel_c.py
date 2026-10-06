#!/usr/bin/env python
"""Plot DeepSHAP attributions of a designed sequence for the melanocyte and keratinocyte tasks (Figure 6C).

Outputs (outputs/): designed_sequence_deepshap.{pdf,png,svg}.

Run (env: featleak-dl):
    python run_panel_c.py --results outputs/all_results.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tangermeme.deep_lift_shap import deep_lift_shap  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_beluga, beluga_task_index  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEGACY = os.environ.get("FEATLEAK_FIG6_RESULTS",
                        "/users/kobbad/feat_leakage/sequence_design/all_results_new.pkl")
BASES = "ACGT"


def default_results() -> str:
    local = OUT / "all_results.pkl"
    return str(local) if local.exists() else LEGACY


def one_hot(seq: str) -> np.ndarray:
    idx = {b: i for i, b in enumerate(BASES)}
    x = np.zeros((4, len(seq)), np.float32)
    for i, ch in enumerate(seq):
        j = idx.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x


def reconstruct(rec) -> str:
    seq = list(rec["start_sequence"])
    for motif, pos in zip(rec["sequence_added"], rec["idx_inserted"]):
        i = int(pos)
        seq[i:i + len(motif)] = list(motif)
    return "".join(seq)


def pick_record(results):
    """Prefer a designed sequence containing both HNF4 and CTCF motifs."""
    for rec in results:
        names = " ".join(str(x) for x in rec["motif_names_added"]).upper()
        if re.search("HNF4", names) and re.search("CTCF", names):
            return rec, "HNF4+CTCF"
    # fallback: the sequence with the most inserted motifs
    rec = max(results, key=lambda r: len(r["motif_names_added"]))
    return rec, "most-motifs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=default_results())
    ap.add_argument("--pad", type=int, default=15, help="bp padding around inserted motifs")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    with open(args.results, "rb") as fh:
        results = pickle.load(fh)
    rec, how = pick_record(results)
    print(f"picked a designed sequence ({how}); motifs = {list(rec['motif_names_added'])}")

    seq = reconstruct(rec)
    oh = one_hot(seq)
    X = torch.tensor(oh[None], dtype=torch.float32)

    mel = beluga_task_index("Melanocyte")
    ker = beluga_task_index("Keratinocyte")
    model = load_beluga(device=DEVICE)

    attr_mel = deep_lift_shap(model, X, target=mel, device=DEVICE)
    attr_ker = deep_lift_shap(model, X, target=ker, device=DEVICE)
    c_mel = (np.asarray(attr_mel[0]) * oh)
    c_ker = (np.asarray(attr_ker[0]) * oh)

    # window spanning the inserted motifs (capped for a readable logo)
    starts = [int(p) for p in rec["idx_inserted"]]
    ends = [int(p) + len(m) for p, m in zip(rec["idx_inserted"], rec["sequence_added"])]
    lo = max(0, min(starts) - args.pad)
    hi = min(len(seq), max(ends) + args.pad)
    if hi - lo > 260:                      # too wide -> center on the CTCF (or first) motif
        names = [str(n).upper() for n in rec["motif_names_added"]]
        ci = next((k for k, n in enumerate(names) if "CTCF" in n), 0)
        centre = starts[ci] + (ends[ci] - starts[ci]) // 2
        lo, hi = max(0, centre - 90), min(len(seq), centre + 90)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    plot_logo(c_mel[:, lo:hi], ax=axes[0]); axes[0].set_title("Melanocyte (on-target)", fontsize=10)
    plot_logo(c_ker[:, lo:hi], ax=axes[1]); axes[1].set_title("Keratinocyte (off-target)", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.6); ax.set_ylabel("DeepSHAP")
    axes[1].set_xlabel(f"designed sequence position {lo}-{hi} (inserted motifs: "
                       f"{', '.join(str(n) for n in rec['motif_names_added'])})", fontsize=7)
    fig.suptitle("Designed sequence: inserted motifs are repressive for the off-target task",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"designed_sequence_deepshap.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)
    print(f"on-target melanocyte total contrib over window = {c_mel[:, lo:hi].sum():+.3f}")
    print(f"off-target keratinocyte total contrib over window = {c_ker[:, lo:hi].sum():+.3f}")
    print("saved -> designed_sequence_deepshap.pdf/.png/.svg")


if __name__ == "__main__":
    main()
