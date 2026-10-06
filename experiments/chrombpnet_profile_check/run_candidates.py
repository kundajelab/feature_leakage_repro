#!/usr/bin/env python
"""Plot ChromBPNet fibroblast DNase profiles at the top-ranked enhancers with 0-4 OCT-SOX insertions.

Outputs (outputs/): candidate_insertions.pdf.

Run (env: featleak-dl):
    python run_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from bpnetlite.bpnet import BPNet  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
IN_WIN = 2114
SIGMA = 20            # display smoothing (bp)
N_TOP = 6


def scaled_profile(model, X):
    yp, yc = model(X.float().to(DEVICE))
    logits = yp.reshape(yp.shape[0], -1)
    logits = logits - logits.mean(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    counts = torch.exp(yc) - 1.0                          # natural-log inverse
    return (probs * counts).detach().cpu().numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(OUT / "peak_ranking.csv")
    # EXPERIMENT-MATCHED ChromBPNet (skin-biceps ENCSR562ACY), consistent with the
    # AlphaGenome skin-biceps task; not the old cardiac model.
    model = BPNet.from_chrombpnet(str(config.resolve("chrombpnet_skinbiceps",
                                                     must_exist=True))).to(DEVICE).eval()
    fasta = str(config.resolve("hg38_fasta"))

    # Extract PER ROW: extract_loci silently drops off-contig loci, which previously
    # misaligned labels and produced empty slices (-> "X must be one-hot" in substitute).
    panels = []
    for _, row in ranked.iterrows():
        df1 = pd.DataFrame([{"chrom": row.chrom, "start": int(row.start), "end": int(row.end)}])
        l = extract_loci(df1, fasta, in_window=IN_WIN, verbose=False).float()
        if l.shape[0] == 1 and int(l.sum()) == IN_WIN:       # extracted + fully one-hot (no N)
            panels.append((row, l))
        if len(panels) >= N_TOP:
            break
    n = len(panels)
    mid = IN_WIN // 2
    offs = [mid - 30, mid + 30, mid - 60, mid + 60]     # insert within the peak

    fig, axes = plt.subplots(n, 5, figsize=(14, 2.0 * n), sharex="row", squeeze=False)
    with torch.no_grad():
        for r, (row, base) in enumerate(panels):
            levels = [base]
            cur = base
            for o in offs:
                cur = substitute(cur, OCT_SOX, start=o)
                levels.append(cur)
            profs = np.array([gaussian_filter1d(scaled_profile(model, lv).squeeze(), SIGMA)
                              for lv in levels])
            prof_len = profs.shape[1]; trim = (IN_WIN - prof_len) // 2
            c = prof_len // 2; w = 300
            ymax = profs[:, c-w:c+w].max() * 1.15
            for col in range(5):
                ax = axes[r][col]
                ax.plot(np.arange(c-w, c+w), profs[col, c-w:c+w], color="#441853", lw=1)
                ax.set_ylim(0, ymax); ax.set_yticks([])
                for o in offs[:col]:
                    ax.axvline(o - trim, ls="--", color="#1764ab", lw=0.7)
                if col == 0:
                    ax.set_ylabel(f"{row.chrom}:{row.start}\n"
                                  f"s={row.signal:.0f} cf={row.central_frac:.2f}", fontsize=7)
                if r == 0:
                    ax.set_title(["no insert", "1", "2", "3", "4"][col], fontsize=8)
    fig.suptitle("ChromBPNet (skin-biceps ENCSR562ACY) OCT-SOX insertion (smoothed) at clean "
                 "enhancers — peak stays stable (no leakage)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "candidate_insertions.pdf"); plt.close(fig)
    print(f"used {n} clean enhancers")
    print("done ->", OUT / "candidate_insertions.pdf")


if __name__ == "__main__":
    main()
