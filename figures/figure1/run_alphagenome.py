#!/usr/bin/env python
"""Compute the Figure 1C OCT-SOX ablation effect on fibroblast DNase with the local AlphaGenome model.

Outputs (outputs/): fig1_alphagenome_octsox_boxplot.pdf and
fig1_alphagenome_octsox_{effect,before,after,before_local,after_local}.npy.

Run (env: featleak-alphagenome):
    python run_alphagenome.py --n-loci 100 --seq-len 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.ersatz import dinucleotide_shuffle  # noqa: E402
from tangermeme.utils import characters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.models import load_alphagenome  # noqa: E402
from common.ag_score import score_sequences, FIBROBLAST_ONTOLOGY, _output_types  # noqa: E402
from fig1_utils import annotate_with_peak_distances  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OCTSOX_WINDOW = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loci", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    model = load_alphagenome()
    # OCT-SOX hits 200-500 bp from a fibroblast peak summit (as in the original).
    hits = annotate_with_peak_distances(
        config.resolve("fimo_octsox", must_exist=True),
        config.resolve("peaks_fibroblast", must_exist=True))
    hits = hits[hits.distance.between(200, 500)]
    hits = (hits.filter(["sequence_name", "start", "stop"])
            .rename(columns={"sequence_name": "chromosome", "stop": "end"})
            .astype({"start": "int", "end": "int"}).reset_index(drop=True))
    hits = hits.sample(min(args.n_loci, len(hits)), random_state=42)

    # tangermeme.io.extract_loci drops N-containing loci (ignore=['N']).
    loci = extract_loci(hits, str(config.resolve("hg38_fasta")),
                        in_window=args.seq_len, verbose=True).float()
    a0 = args.seq_len // 2 - OCTSOX_WINDOW // 2
    ablated = dinucleotide_shuffle(loci, start=a0, end=a0 + OCTSOX_WINDOW, n=1).squeeze(1)

    # Capture BOTH the full-track DNase sum (the published effect) and a local
    # central-window sum (+/-500 bp), so a meaningful log2FC can be computed on
    # the local signal (log2FC of the whole-track sum dilutes to ~0).
    OutputType = _output_types()
    LOCAL_HALF_BP = 500

    def _score2(seq):
        out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.DNASE],
                                     ontology_terms=[FIBROBLAST_ONTOLOGY])
        v = np.asarray(out.dnase.values); v = v.sum(1) if v.ndim == 2 else v
        L = len(v); c = L // 2; w = max(1, int(LOCAL_HALF_BP * L / args.seq_len))
        return float(v.sum()), float(v[c - w:c + w].sum())

    bw, bl = zip(*[_score2(characters(x)) for x in loci])
    aw, al = zip(*[_score2(characters(x)) for x in ablated])
    before, before_local = np.array(bw), np.array(bl)
    after, after_local = np.array(aw), np.array(al)
    effect = before - after   # >0: activating; <0: repressive (implausible)
    w = stats.wilcoxon(effect, alternative="less")
    print(f"AlphaGenome OCT-SOX ablation mean effect={effect.mean():.4f} "
          f"(wilcoxon 'repressive' p={w.pvalue:.3g})")

    fig, ax = plt.subplots(figsize=(2, 4))
    ax.boxplot([effect], labels=["AlphaGenome"], showfliers=False, widths=0.5,
               patch_artist=True, boxprops=dict(facecolor="lightgray", color="black"),
               medianprops=dict(color="black"))
    ax.axhline(0, ls="--", color="0.6")
    ax.set_ylabel("OCT-SOX Motif Effect Size (Ablation)")
    if w.pvalue < 0.05:
        ax.set_title("*", fontsize=16)
    fig.tight_layout(); fig.savefig(OUT / "fig1_alphagenome_octsox_boxplot.pdf")
    plt.close(fig)
    np.save(OUT / "fig1_alphagenome_octsox_effect.npy", effect)
    np.save(OUT / "fig1_alphagenome_octsox_before.npy", before)   # with motif (whole track)
    np.save(OUT / "fig1_alphagenome_octsox_after.npy", after)     # ablated (whole track)
    np.save(OUT / "fig1_alphagenome_octsox_before_local.npy", before_local)  # +/-500bp
    np.save(OUT / "fig1_alphagenome_octsox_after_local.npy", after_local)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
