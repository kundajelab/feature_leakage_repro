#!/usr/bin/env python
"""Quantify ChromBPNet central fibroblast DNase signal at the top-ranked enhancers with 0-4 OCT-SOX insertions.

Outputs (outputs/): chrombpnet_quant.csv.

Run (env: featleak-dl):
    python run_quantify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from bpnetlite.bpnet import BPNet  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
IN_WIN = 2114
N_TOP = 6


def central_total(model, X):
    yp, yc = model(X.float().to(DEVICE))
    counts = (torch.exp(yc) - 1.0).squeeze()               # total counts (natural-log inverse)
    logits = yp.reshape(yp.shape[0], -1)
    probs = torch.softmax(logits - logits.mean(-1, keepdim=True), -1).squeeze()
    L = probs.shape[-1]; c = L // 2; w = int(150 / (IN_WIN / L))
    frac_central = probs[c-w:c+w].sum()
    return float(counts * frac_central)                    # central expected counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(OUT / "peak_ranking.csv").head(N_TOP).reset_index(drop=True)
    model = BPNet.from_chrombpnet(str(config.resolve("chrombpnet_fibroblast",
                                                     must_exist=True))).to(DEVICE).eval()
    mid = IN_WIN // 2
    offs = [mid - 150, mid + 150, mid - 200, mid + 200]

    rows = []
    with torch.no_grad():
        for r in range(N_TOP):
            df = pd.DataFrame({"chrom": [ranked.chrom[r]], "start": [int(ranked.start[r])],
                               "end": [int(ranked.end[r])]})
            base = extract_loci(df, str(config.resolve("hg38_fasta")), in_window=IN_WIN,
                                verbose=False).float()
            vals = [central_total(model, base)]
            cur = base
            for o in offs:
                cur = substitute(cur, OCT_SOX, start=o)
                vals.append(central_total(model, cur))
            rows.append({"locus": f"{ranked.chrom[r]}:{ranked.start[r]}",
                         "cbp_0": vals[0], "cbp_4": vals[-1],
                         "cbp_retained": vals[-1] / vals[0]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "chrombpnet_quant.csv", index=False)
    print("ChromBPNet central peak, OCT-SOX insertion (retained = signal_4 / signal_0):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
