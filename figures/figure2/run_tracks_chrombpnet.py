#!/usr/bin/env python
"""Predict ChromBPNet accessibility profiles at an example peak with 0-4 OCT-SOX insertions (Figure 2d).

Outputs (outputs/): chrombpnet_insertion_tracks_<tag>.pdf and chrombpnet_track_data_<tag>.npz.

Run (env: featleak-dl):
    python run_tracks_chrombpnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bpnetlite.bpnet import BPNet  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.predict import predict  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from cbpnet_helpers import scaled_profile_wrapper, EXAMPLE_LOCUS, TRACK_INSERT_OFFSETS  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
PEAK_WIDTH = 180
SMOOTH = 20      # display smoothing (bp) for the base-resolution profile


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--locus", default=None, help="chrom:start:end override")
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.locus:
        c, s, e = args.locus.split(":"); chrom, start, end = c, int(s), int(e)
    else:
        chrom, start, end = EXAMPLE_LOCUS
    peak = pd.DataFrame([[chrom, start, end]], columns=["chrom", "start", "end"])

    model = scaled_profile_wrapper()(
        BPNet.from_chrombpnet(str(config.resolve("chrombpnet_fibroblast",
                                                 must_exist=True)))).to(DEVICE).eval()

    loci = extract_loci(peak, str(config.resolve("hg38_fasta")), verbose=False)
    seq_len = loci.shape[2]
    mid = seq_len // 2
    # Insert OCT-SOX in the peak FLANK (outside the accessible core) so we don't
    # overwrite the enhancer's own motifs.
    ins = [mid + o for o in TRACK_INSERT_OFFSETS]

    levels = [loci]
    cur = loci
    for o in ins:
        cur = substitute(cur, OCT_SOX, start=o)
        levels.append(cur)
    # ChromBPNet is base-resolution; lightly smooth so the enhancer envelope is
    # visible (the raw profile is a spiky Tn5/DNase cut-site track).
    profiles = np.array([
        gaussian_filter1d(predict(model, lv, device=DEVICE).squeeze().numpy(), SMOOTH)
        for lv in levels])                           # (5, prof_len)

    prof_len = profiles.shape[1]
    trim = (seq_len - prof_len) // 2                 # profile is central crop
    ins_p = [o - trim for o in ins]
    c = prof_len // 2; half = 300                    # zoom to the enhancer
    lo, hi = max(c - half, 0), min(c + half, prof_len)
    ymax = profiles[:, lo:hi].max() * 1.2            # SAME axis for all panels
    labels = ["Fibroblast DNase", "1 insertion", "2 insertions",
              "3 insertions", "4 insertions"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 4), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(lo, hi), profiles[i, lo:hi], color="#441853", lw=1.2)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(labels[i], fontsize=7)
        for o in ins_p[:i]:                          # markers for inserted motifs
            ax.axvline(o, ls="--", color="#1764ab", lw=0.9)
    axes[0].set_title(f"ChromBPNet  {chrom}:{start}-{end}")
    axes[-1].set_xlabel("Position")
    fig.tight_layout(); fig.savefig(OUT / f"chrombpnet_insertion_tracks_{args.tag}.pdf")
    plt.close(fig)

    # save profile data in bp coords (1 bp/position; profile = central crop)
    bp = np.arange(prof_len) - prof_len // 2
    np.savez(OUT / f"chrombpnet_track_data_{args.tag}.npz", profiles=profiles, bp=bp,
             ins_bp=np.array([o - trim - prof_len // 2 for o in ins]),
             ylabel="Predicted DNase (expected Tn5 counts / bp)", model="ChromBPNet")
    print(f"profile length={prof_len}, global ymax={ymax:.3f}")
    print("done ->", OUT / f"chrombpnet_insertion_tracks_{args.tag}.pdf")


if __name__ == "__main__":
    main()
