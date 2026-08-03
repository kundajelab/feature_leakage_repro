#!/usr/bin/env python
"""Check AlphaGenome shows OCT-SOX repression at the clean candidate enhancers.

For the top ChromBPNet-clean enhancers, run the same 0-4 OCT-SOX insertion and
plot AlphaGenome's fibroblast DNase profile. We want a locus where AlphaGenome's
peak SHRINKS with insertions (leakage) while ChromBPNet's stays stable -- the
ideal replacement example for Fig 2 C/D.

Run (env: featleak-alphagenome):
    python run_ag_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.utils import characters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types, FIBROBLAST_ONTOLOGY  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384
N_TOP = 4


def dnase_profile(model, seq, OutputType):
    out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.DNASE],
                                 ontology_terms=[FIBROBLAST_ONTOLOGY])
    v = np.asarray(out.dnase.values)
    return v.sum(axis=1) if v.ndim == 2 else v


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(OUT / "peak_ranking.csv").head(N_TOP).reset_index(drop=True)
    model = load_alphagenome()
    OutputType = _output_types()

    mid = SEQ_LEN // 2
    offs = [mid - 30, mid + 30, mid - 60, mid + 60]

    fig, axes = plt.subplots(N_TOP, 5, figsize=(14, 2.0 * N_TOP), sharex="row")
    for r in range(N_TOP):
        df = pd.DataFrame({"chrom": [ranked.chrom[r]], "start": [int(ranked.start[r])],
                           "end": [int(ranked.end[r])]})
        base = extract_loci(df, str(config.resolve("hg38_fasta")), in_window=SEQ_LEN,
                            verbose=False)
        levels = [base]; cur = base
        for o in offs:
            cur = substitute(cur, OCT_SOX, start=o)
            levels.append(cur)
        profs = [gaussian_filter1d(dnase_profile(model, characters(lv[0]), OutputType), 2)
                 for lv in levels]
        profs = np.array(profs)
        P = profs.shape[1]; ratio = P / SEQ_LEN; c = P // 2; w = int(600 * ratio)
        ymax = profs[:, c-w:c+w].max() * 1.15
        for col in range(5):
            ax = axes[r, col]
            ax.plot(np.arange(c-w, c+w), profs[col, c-w:c+w], color="#441853", lw=1)
            ax.set_ylim(0, ymax); ax.set_yticks([])
            for o in offs[:col]:
                ax.axvline(int(o * ratio), ls="--", color="#1764ab", lw=0.7)
            if col == 0:
                ax.set_ylabel(f"{ranked.chrom[r]}:{ranked.start[r]}", fontsize=7)
            if r == 0:
                ax.set_title(["no insert", "1", "2", "3", "4"][col], fontsize=8)
    fig.suptitle("AlphaGenome fibroblast DNase, OCT-SOX insertion — peak should "
                 "SHRINK (leakage)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "ag_candidate_insertions.pdf"); plt.close(fig)
    print("done ->", OUT / "ag_candidate_insertions.pdf")


if __name__ == "__main__":
    main()
