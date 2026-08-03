#!/usr/bin/env python
"""Figure 2C (AlphaGenome, local model): example accessibility tracks.

Predicts AlphaGenome's fibroblast DNase profile at the example peak
(chr2:72,788,900-72,789,080) before and after inserting 1-4 OCT-SOX motifs.
AlphaGenome's predicted peak SHRINKS as OCT-SOX motifs are added -- the spurious
repression (contrast with ChromBPNet, run_tracks_chrombpnet.py).

Run (env: featleak-alphagenome):
    python run_tracks_alphagenome.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.utils import characters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

import argparse  # noqa: E402

from common import config  # noqa: E402
from common.models import load_alphagenome  # noqa: E402
from common.ag_score import FIBROBLAST_ONTOLOGY, _output_types  # noqa: E402
from cbpnet_helpers import EXAMPLE_LOCUS, TRACK_INSERT_OFFSETS  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384    # local AlphaGenome min window (matches the locus-selection quant)
PEAK_WIDTH = 180
ONTOLOGY = FIBROBLAST_ONTOLOGY   # overridden by --ontology


def dnase_profile(model, seq):
    OutputType = _output_types()
    out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.DNASE],
                                 ontology_terms=[ONTOLOGY])
    vals = np.asarray(out.dnase.values)      # (positions, tracks)
    return vals.sum(axis=1) if vals.ndim == 2 else vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locus", default=None, help="chrom:start:end override")
    ap.add_argument("--ontology", default=FIBROBLAST_ONTOLOGY)
    ap.add_argument("--tag", default="cardiac")
    args = ap.parse_args()
    global ONTOLOGY
    ONTOLOGY = args.ontology
    OUT.mkdir(exist_ok=True)
    if args.locus:
        c, s, e = args.locus.split(":"); chrom, start, end = c, int(s), int(e)
    else:
        chrom, start, end = EXAMPLE_LOCUS
    peak = pd.DataFrame([[chrom, start, end]], columns=["chrom", "start", "end"])
    loci = extract_loci(peak, str(config.resolve("hg38_fasta")), in_window=SEQ_LEN,
                        verbose=False)
    mid = SEQ_LEN // 2
    offs = [mid + o for o in TRACK_INSERT_OFFSETS]  # flank insertions (see cbpnet_helpers)

    levels = [loci]
    cur = loci
    for o in offs:
        cur = substitute(cur, "ATTGTTATGCAAAT", start=o)
        levels.append(cur)

    model = load_alphagenome()
    profiles = np.array([dnase_profile(model, characters(lv[0])) for lv in levels])

    L = profiles.shape[1]
    ratio = L / SEQ_LEN
    offs_p = [int(o * ratio) for o in offs]
    c = L // 2; half = int(2000 * ratio)             # zoom to +/-2 kb around peak
    lo, hi = max(c - half, 0), min(c + half, L)
    ymax = profiles[:, lo:hi].max() * 1.2
    labels = ["Fibroblast DNase", "1 insertion", "2 insertions",
              "3 insertions", "4 insertions"]
    fig, axes = plt.subplots(5, 1, figsize=(8, 5), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(lo, hi), profiles[i, lo:hi], color="#441853", lw=1)
        ax.set_ylim(0, ymax); ax.set_ylabel(labels[i], fontsize=7)
        for o in offs_p[:i]:
            ax.axvline(o, ls="--", color="#1764ab", lw=0.8)
    axes[0].set_title(f"AlphaGenome ({args.tag} {ONTOLOGY})  {chrom}:{start}-{end}")
    axes[-1].set_xlabel("Position (DNase bins)")
    fig.tight_layout(); fig.savefig(OUT / f"alphagenome_insertion_tracks_{args.tag}.pdf")
    plt.close(fig)

    # save profile data in bp coords (profile spans SEQ_LEN bp)
    bp_per = SEQ_LEN / L
    bp = (np.arange(L) - L // 2) * bp_per
    np.savez(OUT / f"alphagenome_track_data_{args.tag}.npz", profiles=profiles, bp=bp,
             ins_bp=np.array([o - SEQ_LEN // 2 for o in offs]),
             ylabel=f"Predicted DNase ({args.tag}, a.u.)", model="AlphaGenome")
    print("done ->", OUT / f"alphagenome_insertion_tracks_{args.tag}.pdf")


if __name__ == "__main__":
    main()
