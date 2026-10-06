#!/usr/bin/env python
"""Compute and plot Beluga per-cell-type motif effects by affinity quartile (Figure 1 E-H).

Outputs (outputs/): fig1_beluga_<motif>_<method>.pdf and fig1_beluga_<method>_effects.npz.

Run (env: featleak-dl):
    python run_beluga.py --method ablation --sample-size 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ablate import ablate  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.predict import predict  # noqa: E402
from tangermeme.utils import random_one_hot  # noqa: E402

from common import config  # noqa: E402
from common.models import load_beluga, load_beluga_task_names  # noqa: E402
from common.motifs import OCT_SOX, AP1, HNF4, CTCF  # noqa: E402
from fig1_utils import sample_loci_by_affinity, plot_sorted_scatters  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
WIN = 2000

# motif -> (config FIMO key, consensus string for insertion, window bp, cmap)
MOTIFS = {
    "OCT-SOX": ("fimo_octsox", OCT_SOX, 14, "Blues"),
    "AP-1": ("fimo_ap1", AP1, 7, "Reds"),
    "HNF4": ("fimo_hnf4", HNF4, 9, "Greens"),
    "CTCF": ("fimo_ctcf", CTCF, 17, "Purples"),
}


def dnase_task_indices():
    names = load_beluga_task_names()
    return np.array([i for i, n in enumerate(names) if "DNase" in n])


def ablation_effect(model, fimo_key, motif_len, dnase_idx, sample_size):
    """Per-DNase-task effect for each affinity quartile via motif ablation.

    Uses tangermeme.ablate.ablate (n=20 mononucleotide shuffles of the central
    motif window), exactly as the paper's Methods describe. This matters: a
    dinucleotide shuffle of a short (<=~10 bp) window is often a no-op and
    silently produces a zero effect (the AP-1 bug).
    """
    quartiles = sample_loci_by_affinity(config.resolve(fimo_key, must_exist=True),
                                        sample_size=sample_size, num_groups=4)
    # Fixed central window covering the motif core (>= motif_len, matches the
    # original 995-1005 = 10 bp used for Beluga's 2000 bp input).
    half = max(motif_len, 10) // 2
    a0, a1 = WIN // 2 - half, WIN // 2 + half
    results = []
    for grp in quartiles:
        loci, keep = extract_loci(grp, str(config.resolve("hg38_fasta")),
                                  in_window=WIN, return_filtered=True, verbose=False)
        mask = loci.sum(axis=1).sum(axis=1) == WIN
        loci = loci[mask]
        y_before, y_after = ablate(model, loci, a0, a1, device=DEVICE, batch_size=64)
        before = y_before[:, dnase_idx]                       # (N, 334)
        after = y_after[:, :, dnase_idx].mean(axis=1)         # avg over 20 shuffles
        # effect = signal(with motif) - signal(without motif); >0 activating,
        # <0 repressive. The implausible repression is the negative tail.
        results.append((before - after).mean(axis=0).numpy())
    return results


def marginalization_effect(model, motif, dnase_idx, sample_size):
    """Per-DNase-task effect for one motif via insertion into random backgrounds.

    (No affinity quartiles for the synthetic-background method; returns a single
    array wrapped in a length-1 list for plotting compatibility.)"""
    X = random_one_hot((sample_size, 4, WIN), random_state=0).type(torch.float32)
    before = predict(model, X, device=DEVICE, batch_size=64)[:, dnase_idx]
    Xin = substitute(X, motif)
    after = predict(model, Xin, device=DEVICE, batch_size=64)[:, dnase_idx]
    return [(after - before).mean(axis=0).numpy()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ablation", "marginalization"],
                    default="ablation")
    ap.add_argument("--sample-size", type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    model = load_beluga(device=DEVICE)
    dnase_idx = dnase_task_indices()
    print(f"{len(dnase_idx)} Beluga DNase tasks; method={args.method}")

    saved = {}
    for name, (fimo_key, seq, mlen, cmap) in MOTIFS.items():
        print(f"[{name}] ...")
        if args.method == "ablation":
            results = ablation_effect(model, fimo_key, mlen, dnase_idx, args.sample_size)
        else:
            results = marginalization_effect(model, seq, dnase_idx, args.sample_size)
        n_repress = int((results[-1] < 0).sum())
        print(f"    strongest group: {n_repress}/{len(dnase_idx)} cell types repressed")
        plot_sorted_scatters(results, f"{name} ({args.method})",
                             OUT / f"fig1_beluga_{name}_{args.method}.pdf", cmap)
        for i, r in enumerate(results):
            saved[f"{name}_q{i}"] = r
    np.savez(OUT / f"fig1_beluga_{args.method}_effects.npz", **saved)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
