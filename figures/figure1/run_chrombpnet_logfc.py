#!/usr/bin/env python
"""ChromBPNet OCT-SOX ablation effect as per-locus log2 fold-change.

Fig-1-style ablation, matched to run_alphagenome.py: take OCT-SOX FIMO hits
200-500 bp from a fibroblast DNase peak summit, extract the ChromBPNet input
window centred on the motif, dinucleotide-shuffle the central 14 bp, and compare
predicted accessibility with vs. without the motif. ChromBPNet's count head is
trained on ln(counts+1), so a clean per-locus log2 fold-change is

    log2FC = ( logcounts(with motif) - logcounts(ablated) ) / ln(2)

over the model's 1000 bp output (which is itself a local window, centred on the
motif). Negative = motif represses predicted accessibility (leakage direction);
single-task ChromBPNet should be ~0. Saves the per-locus array and prints a
one-sided Wilcoxon test.

Run (env: featleak-dl):
    python run_chrombpnet_logfc.py --num-loci 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figures" / "figure2"))   # for cbpnet_helpers

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from bpnetlite.bpnet import BPNet, CountWrapper  # noqa: E402
from tangermeme.predict import predict  # noqa: E402

from common import config  # noqa: E402
from cbpnet_helpers import annotate_with_peak_distances, ablate_motifs  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
SEQ_LEN = 2114
LN2 = np.log(2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-loci", type=int, default=1000)
    ap.add_argument("--peaks-key", default="peaks_dermis",
                    help="fibroblast DNase peaks (default: dermis, ENCSR000EPO)")
    ap.add_argument("--chrombpnet-key", default="chrombpnet_dermis",
                    help="ChromBPNet trained on the same ENCODE experiment as the peaks")
    ap.add_argument("--tag", default="_dermis")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    hits = annotate_with_peak_distances(
        config.resolve("fimo_octsox", must_exist=True),
        config.resolve(args.peaks_key, must_exist=True))
    hits = hits[hits.distance.between(200, 500)]   # match run_alphagenome.py exactly

    # random_state=42 -> same OCT-SOX loci as the AlphaGenome panel
    loci, ablated, sites = ablate_motifs(hits, motif_length=14,
                                         fasta=config.resolve("hg38_fasta"),
                                         seq_len=SEQ_LEN, num_loci=args.num_loci,
                                         random_state=42, return_sites=True)
    # the exact sites, so run_alphagenome_torch.py scores the same loci
    sites.to_csv(OUT / f"fig1_octsox_sites{args.tag}.csv", index=False)

    # CountWrapper returns the count-head output = ln(counts+1)  (natural log)
    model = CountWrapper(BPNet.from_chrombpnet(
        str(config.resolve(args.chrombpnet_key, must_exist=True)))).to(DEVICE).eval()
    lc_with = predict(model, loci, device=DEVICE, batch_size=64).squeeze().numpy()
    lc_abl = predict(model, ablated, device=DEVICE, batch_size=64).squeeze().numpy()
    logfc = (lc_with - lc_abl) / LN2                      # log2( (counts_with+1)/(counts_abl+1) )

    np.save(OUT / f"fig1_chrombpnet_octsox_logfc{args.tag}.npy", logfc)
    w = stats.wilcoxon(logfc, alternative="less")         # H1: repressive (median < 0)
    w2 = stats.wilcoxon(logfc)                             # two-sided (any effect)
    print(f"ChromBPNet OCT-SOX log2FC over n={len(logfc)} loci:")
    print(f"  median={np.median(logfc):+.4f}  mean={logfc.mean():+.4f}")
    print(f"  Wilcoxon one-sided (repressive, median<0): p={w.pvalue:.3g}")
    print(f"  Wilcoxon two-sided (any effect):          p={w2.pvalue:.3g}")
    print("saved ->", OUT / f"fig1_chrombpnet_octsox_logfc{args.tag}.npy")


if __name__ == "__main__":
    main()
