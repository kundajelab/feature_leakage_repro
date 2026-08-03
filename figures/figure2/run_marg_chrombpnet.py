#!/usr/bin/env python
"""Figure 2 (panel b + supplement): ChromBPNet OCT-SOX marginalization on real peaks.

Insert the OCT-SOX consensus into the flanks of the top-N fibroblast DNase peaks and
record the log2 fold-change of predicted central accessibility (native vs +OCT-SOX),
AND the same for composition-matched SHUFFLED OCT-SOX. The reported effect is the
shuffle-corrected SPECIFICITY = log2FC(real) - log2FC(shuffled) -- still in log2FC
units, but with the generic-insertion-disruption baseline subtracted (essential for
base-resolution ChromBPNet, whose raw log2FC is dominated by generic disruption).

Use the EXPERIMENT-MATCHED ChromBPNet for the model being compared:
  --chrombpnet-key chrombpnet_skinbiceps   (ENCSR562ACY)  -> matches AlphaGenome (panel b)
  --chrombpnet-key chrombpnet_dermis       (ENCSR000EPO)  -> matches Enformer/Borzoi (supplement)

Writes outputs/cbp_octsox_specificity_<tag>.csv. Run (env: featleak-dl):
    python run_marg_chrombpnet.py --chrombpnet-key chrombpnet_skinbiceps --tag skinbiceps --n-loci 120
    python run_marg_chrombpnet.py --chrombpnet-key chrombpnet_dermis     --tag dermis     --n-loci 120
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
from tangermeme.ersatz import substitute  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from common.motif_controls import shuffled_motifs  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_WIN = 2114
OFFSETS = [-150, 150, -200, 200]      # 4 flank insertions (marginalization)
K_SHUF = 3                            # composition-matched shuffle controls (averaged)


def central_total(model, X):
    yp, yc = model(X.float().to(DEVICE))
    counts = (torch.exp(yc) - 1.0).squeeze(-1)                    # natural-log inverse
    logits = yp.reshape(yp.shape[0], -1)
    probs = torch.softmax(logits - logits.mean(-1, keepdim=True), -1)
    L = probs.shape[-1]; c = L // 2; w = int(150 / (IN_WIN / L))
    return (counts * probs[:, c-w:c+w].sum(-1)).detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loci", type=int, default=120)
    ap.add_argument("--chrombpnet-key", default="chrombpnet_skinbiceps")
    ap.add_argument("--tag", default="skinbiceps")
    ap.add_argument("--peaks-key", default="peaks_fibroblast",
                    help="common enhancer substrate (top-N by signalValue)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = BPNet.from_chrombpnet(str(config.resolve(args.chrombpnet_key,
                                                     must_exist=True))).to(DEVICE).eval()
    peaks = pd.read_csv(config.resolve(args.peaks_key), sep="\t", header=None)
    peaks = peaks.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    peaks = peaks[peaks.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    top = peaks.sort_values("signal", ascending=False).head(args.n_loci).reset_index(drop=True)

    loci = extract_loci(top[["chrom", "start", "end"]], str(config.resolve("hg38_fasta")),
                        in_window=IN_WIN, verbose=False).float()
    mid = IN_WIN // 2

    def batched(X):
        with torch.no_grad():
            return np.concatenate([central_total(model, X[i:i+16]) for i in range(0, len(X), 16)])

    def insert_all(motif):
        cur = loci.clone()
        for o in OFFSETS:
            cur = substitute(cur, motif, start=mid + o)
        return cur

    s0 = batched(loci)
    lfc_real = np.log2((batched(insert_all(OCT_SOX)) + 1e-3) / (s0 + 1e-3))
    # composition-matched shuffle control: log2FC after inserting shuffled OCT-SOX
    lfc_shuf = np.mean([np.log2((batched(insert_all(m)) + 1e-3) / (s0 + 1e-3))
                        for m in shuffled_motifs(OCT_SOX, K_SHUF)], axis=0)
    specificity = lfc_real - lfc_shuf     # shuffle-corrected log2FC (OCT-SOX-specific)

    df = pd.DataFrame({"locus": [f"{top.chrom[i]}:{top.start[i]}" for i in range(len(top))],
                       "signal": top.signal.values,
                       "log2fc_real": lfc_real, "log2fc_shuf": lfc_shuf, "specificity": specificity})
    out = OUT / f"cbp_octsox_specificity_{args.tag}.csv"
    df.to_csv(out, index=False)
    w = wilcoxon(specificity, alternative="less")
    print(f"ChromBPNet[{args.chrombpnet_key}] OCT-SOX over {len(df)} enhancers: "
          f"raw log2FC median={np.median(lfc_real):+.4f}  shuffle median={np.median(lfc_shuf):+.4f}  "
          f"SPECIFICITY median={np.median(specificity):+.4f} (Wilcoxon<0 p={w.pvalue:.3g})")
    print("done ->", out)


if __name__ == "__main__":
    main()
