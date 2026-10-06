#!/usr/bin/env python
"""Figure 7c: screen endogenous OCT-SOX sites on held-out chr8/9, before vs after fine-tuning.

Sites are the Figure 1 FIMO OCT-SOX hits (AC0636) within 500 bp of a dermal-fibroblast
DNase summit (ENCFF184KAM) on chr8/chr9, the chromosomes held out from fine-tuning. Each
site is centred in the 524,288 bp window and its 14 bp are dinucleotide-shuffled
(Fig 1's ablation) NSHUF times. Readout A = predicted dermis DNase (CL:0002551) summed
over +-500 bp around the motif (as in Fig 1c), and over the central 300 bp; both heads
(original, fine-tuned) are read from the same trunk embedding.

    effect = log2(A_intact / mean_k A_shuffled_k)    (negative = motif lowers prediction)

Run (env featleak-agtorch, one GPU per shard), then merge:
    python endo_screen.py --shard 0 --nshard 3     # x3, in parallel
    python endo_screen.py --merge                   # -> endo_screen/screen.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "figures" / "figure2"))
import finetune_dermis_torch as ft  # noqa: E402

OUT = ft.OUT / "endo_screen"
NSHUF = 5
MAX_DIST = 500
HALF = 500                      # readout +-500 bp around the motif


def candidates():
    from cbpnet_helpers import annotate_with_peak_distances
    h = annotate_with_peak_distances(ft.config.resolve("fimo_octsox", must_exist=True),
                                     ft.config.resolve("peaks_dermis", must_exist=True))
    h = h[h.sequence_name.isin(sorted(ft.HELDOUT)) & (h.distance <= MAX_DIST)]
    return pd.DataFrame({
        "chrom": h.sequence_name.values,
        "mid": ((h.start.values + h.stop.values) // 2).astype(int),
        "strand": h.strand.values, "site": h.matched_sequence.values,
        "fimo_score": h.score.values, "distance": h.distance.values,
    }).reset_index(drop=True)


def screen(shard, nshard):
    import pyfaidx
    import torch
    ag = ft.AG()
    w = torch.load(ft.ft_head_path())
    wf, bf, sf = w["w"].cuda(), w["b"].cuda(), w["s"].cuda()
    fa = pyfaidx.Fasta(str(ft.config.resolve("hg38_fasta")))
    ml = len(ft.OCT_SOX)
    c0, c1 = HALF - ft.CENTRAL, HALF + ft.CENTRAL

    def score(seq):
        """(pre_1k, ft_1k, pre_300, ft_300)."""
        with torch.no_grad():
            e = ag.embed(seq, half=HALF).cuda()
            a = ag.head_pred(e, ag.w0, ag.b0, ag.s0); b = ag.head_pred(e, wf, bf, sf)
        return (a.sum().item(), b.sum().item(), a[c0:c1].sum().item(), b[c0:c1].sum().item())

    cand = candidates()
    rows = []
    for j, r in cand.iterrows():
        if j % nshard != shard:
            continue
        q = ft.fetch(fa, r.chrom, int(r.mid))
        p = ft.C0 - ml // 2
        it = np.array(score(q))
        sh = np.array([score(ft.dinuc_shuffle_window(q, p, p + ml, s)) for s in range(NSHUF)])
        eff = np.log2((it + ft.EPS) / (sh.mean(0) + ft.EPS))
        each = np.log2((it[0] + ft.EPS) / (sh[:, 0] + ft.EPS))
        rows.append(dict(r, A1k_pre=it[0], A1k_ft=it[1], A300_pre=it[2], A300_ft=it[3],
                         eff1k_pre=eff[0], eff1k_ft=eff[1], eff300_pre=eff[2],
                         eff300_ft=eff[3], eff1k_pre_min=each.min(), eff1k_pre_max=each.max()))
        print(f"{j} {r.chrom}:{r.mid} d={r.distance} pre={eff[0]:+.3f} "
              f"[{each.min():+.2f},{each.max():+.2f}] ft={eff[1]:+.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"screen_{shard}.csv", index=False)


def merge():
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("screen_[0-9]*.csv"))])
    df = df.sort_values("eff1k_pre").reset_index(drop=True)
    df.to_csv(OUT / "screen.csv", index=False)
    robust = df[(df.eff1k_pre < -0.3) & (df.eff1k_pre_max < 0)]
    print(f"{len(df)} sites; median effect {df.eff1k_pre.median():+.3f} -> "
          f"{df.eff1k_ft.median():+.3f}; {len(robust)} robustly repressive "
          f"(< -0.3, all {NSHUF} shuffles < 0), of which "
          f"{(robust.eff1k_ft.abs() < 0.2).sum()} have |effect| < 0.2 after fine-tuning")
    print(robust[["chrom", "mid", "strand", "site", "distance", "eff1k_pre", "eff1k_pre_max",
                  "eff1k_ft"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else screen(a.shard, a.nshard)
