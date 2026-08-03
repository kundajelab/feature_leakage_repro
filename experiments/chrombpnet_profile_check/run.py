#!/usr/bin/env python
"""Diagnose ChromBPNet profile computation + find clean strong-enhancer examples.

1. Introspects the raw ChromBPNet outputs (profile logits + log-counts) so we can
   confirm the profile = softmax(logits) * (exp(logcounts) - 1) is computed right.
2. Screens the STRONGEST fibroblast DNase peaks (by narrowPeak signalValue),
   centres each on its midpoint, predicts, and plots a grid of profiles so we can
   pick loci that look like real enhancer peaks (a clean central enrichment).

Run (env: featleak-dl):
    python run.py --top 36
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from bpnetlite.bpnet import BPNet  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
IN_WIN = 2114


def scaled_profile(logits, logcounts):
    """softmax over positions * total counts (per-position expected coverage)."""
    logits = logits.reshape(logits.shape[0], -1)
    logits = logits - logits.mean(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    counts = torch.exp(logcounts) - 1.0                  # natural-log inverse
    return (probs * counts).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=36)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = BPNet.from_chrombpnet(str(config.resolve("chrombpnet_fibroblast",
                                                     must_exist=True))).to(DEVICE).eval()

    # narrowPeak: 0 chrom 1 start 2 end ... 6 signalValue 9 summit-offset
    peaks = pd.read_csv(config.resolve("peaks_fibroblast"), sep="\t", header=None)
    peaks = peaks.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    chroms = ["chr" + str(i) for i in range(1, 23)]
    peaks = peaks[peaks.chrom.isin(chroms)]
    top = peaks.sort_values("signal", ascending=False).head(args.top).reset_index(drop=True)
    top["width"] = top.end - top.start
    print(f"strongest {args.top} peaks: signal {top.signal.min():.1f}-{top.signal.max():.1f}, "
          f"median width {int(top.width.median())}")

    df = top[["chrom", "start", "end"]].copy()
    loci = extract_loci(df, str(config.resolve("hg38_fasta")), in_window=IN_WIN,
                        verbose=False)

    # introspect one raw prediction
    with torch.no_grad():
        y = model(loci[:1].float().to(DEVICE))
    print("raw model(X) returns:", type(y),
          [tuple(t.shape) for t in y] if isinstance(y, (tuple, list)) else y.shape)

    with torch.no_grad():
        profs = []
        for i in range(0, len(loci), 8):
            yp, yc = model(loci[i:i+8].float().to(DEVICE))
            profs.append(scaled_profile(yp, yc))
        profs = np.concatenate(profs)          # (N, out_len)
    print("profile length:", profs.shape[1])

    # grid of profiles
    n = len(profs); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.4 * nrow))
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off"); continue
        ax.plot(profs[i], color="#441853", lw=0.8)
        ax.set_title(f"{top.chrom[i]}:{top.start[i]} s={top.signal[i]:.0f} "
                     f"tot={profs[i].sum():.0f}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("ChromBPNet scaled profiles at strongest fibroblast DNase peaks",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "strong_peak_profiles.pdf"); plt.close(fig)

    # rank by "peakiness": fraction of signal in central 200 bp
    L = profs.shape[1]; c = L // 2; w = int(200 / (IN_WIN / L))
    central_frac = profs[:, c-w:c+w].sum(1) / profs.sum(1)
    top["central_frac"] = central_frac
    top["total"] = profs.sum(1)
    ranked = top.sort_values("central_frac", ascending=False)
    ranked[["chrom", "start", "end", "signal", "central_frac", "total"]].to_csv(
        OUT / "peak_ranking.csv", index=False)
    print("\nmost peak-like (highest central 200bp fraction):")
    print(ranked[["chrom", "start", "end", "signal", "central_frac"]].head(8).to_string())
    print("done ->", OUT)


if __name__ == "__main__":
    main()
