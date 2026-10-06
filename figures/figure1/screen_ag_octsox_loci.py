#!/usr/bin/env python
"""Screen OCT-SOX loci by AlphaGenome ablation effect to pick a Figure 1A example.

Outputs (outputs/panel_a/): ag_octsox_screen_ctx<context>.tsv.

Run (env: featleak-alphagenome):
    python screen_ag_octsox_loci.py --n-loci 150 --context 524288
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfaidx  # noqa: E402

from common import config  # noqa: E402
from common.models import load_alphagenome  # noqa: E402
from fig1_utils import annotate_with_peak_distances  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "panel_a"
ONTOLOGY = "CL:0002548"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loci", type=int, default=150)
    ap.add_argument("--context", type=int, default=524288)
    ap.add_argument("--n-shuffles", type=int, default=2)
    args = ap.parse_args()
    from alphagenome.models import dna_output

    hits = annotate_with_peak_distances(config.resolve("fimo_octsox", must_exist=True),
                                        config.resolve("peaks_fibroblast", must_exist=True))
    hits = hits[hits.distance.between(200, 500)]
    hits = hits[hits.score >= hits.score.quantile(0.5)]
    hits = hits.sample(min(args.n_loci, len(hits)), random_state=7).reset_index(drop=True)

    model = load_alphagenome()
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    rng = np.random.default_rng(0)
    L, k = args.context, 1024

    def central(seq):
        out = model.predict_sequence(sequence=seq,
                                     requested_outputs=[dna_output.OutputType.DNASE],
                                     ontology_terms=[ONTOLOGY])
        v = np.asarray(out.dnase.values)
        v = v.mean(1) if v.ndim == 2 else v
        return float(v[L // 2 - k:L // 2 + k].sum())

    rows = []
    for i, r in hits.iterrows():
        s0, e0 = int(r.start) - 1, int(r.stop)
        c = s0 + (e0 - s0) // 2
        w0 = c - L // 2
        seq = str(fa[r.sequence_name][w0:w0 + L]).upper()
        if len(seq) != L or "N" in seq[L // 2 - 5000:L // 2 + 5000]:
            continue
        a, b = s0 - w0, e0 - w0
        before = central(seq)
        after = []
        for _ in range(args.n_shuffles):
            mot = list(seq[a:b]); rng.shuffle(mot)
            after.append(central(seq[:a] + "".join(mot) + seq[b:]))
        lfc = float(np.log2((np.mean(after) + 1e-6) / (before + 1e-6)))
        rows.append(dict(chrom=r.sequence_name, start=int(r.start), stop=int(r.stop),
                         strand=r.strand, score=float(r.score),
                         matched=r.matched_sequence, before=before, lfc_ablated=lfc))
        print(f"{i}: {r.sequence_name}:{r.start} before={before:.1f} "
              f"log2(ablated/orig)={lfc:+.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values("lfc_ablated", ascending=False)
    df.to_csv(OUT / f"ag_octsox_screen_ctx{L}.tsv", sep="\t", index=False)
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
