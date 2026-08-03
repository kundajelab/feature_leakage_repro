#!/usr/bin/env python
"""Figure 5 (extra): does multi-task feature leakage depend on training-set size?

Holds the motif library FIXED (same 100 motifs, effects and task assignments)
and varies the number of training sequences. For each size it trains the
additive multi-task CNN and measures:
  * r_leak  = corr(true effect size, leakage effect on INACTIVE tasks)
  * r_pred  = corr(true effect size, predicted effect on ACTIVE tasks)
  * mean_leak = mean leakage effect

Hypothesis: with fewer training sequences the model cannot cleanly separate the
motifs (superposition), so leakage is stronger / more correlated with effect
size; with more data it learns cleanly and the leakage correlation weakens.

Outputs (outputs/): data_sweep.csv, data_sweep_leakage_vs_size.pdf.

Run (env: featleak-dl):
    python run_data_sweep.py --sizes 500 1000 2500 5000 10000 25000 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # editable text in Illustrator
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

import sim_helpers as H  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def train_and_measure(motif_dict, n_samples, num_tasks, epochs, lr, device):
    seqs, labels = H.generate_dataset(motif_dict, n_samples, num_tasks=num_tasks)
    tr, va, _ = H.create_dataloaders(seqs, labels)
    model, *_ = H.train_simple_model(tr, va, num_tasks=num_tasks,
                                     num_epochs=epochs, lr=lr, device=device)
    model = model.to(device).eval()
    rows = [H.measure_feature_leakage(m, model, v["sparsity"], v["effect"])
            for m, v in motif_dict.items()]
    df = pd.DataFrame(rows)
    return {
        "n_samples": n_samples,
        "r_leak": pearsonr(df.actual_effect_size, df.leakage_effect)[0],
        "r_pred": pearsonr(df.actual_effect_size, df.predicted_effect)[0],
        "mean_leak": float(df.leakage_effect.mean()),
        "r_prop_leak": pearsonr(df.proportion_tasks_active, df.leakage_effect)[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2500, 5000, 10000, 25000, 50000])
    ap.add_argument("--num-motifs", type=int, default=100)
    ap.add_argument("--num-tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # SAME motif library for every training-set size (comparable effect sizes).
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    motif_dict = H.make_motif_dict(args.num_motifs, 10, num_tasks=args.num_tasks)

    results = []
    for n in args.sizes:
        print(f"\n=== training on {n} sequences ===", flush=True)
        # Re-seed per size so only the dataset SIZE differs, not the RNG stream.
        np.random.seed(args.seed + 1); torch.manual_seed(args.seed + 1)
        r = train_and_measure(motif_dict, n, args.num_tasks, args.epochs,
                              args.lr, args.device)
        print(f"    n={n}: r_leak={r['r_leak']:.3f} r_pred={r['r_pred']:.3f} "
              f"mean_leak={r['mean_leak']:.4f}", flush=True)
        results.append(r)
        pd.DataFrame(results).to_csv(OUT / "data_sweep.csv", index=False)

    df = pd.DataFrame(results)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(df.n_samples, df.r_leak, "o-", color="#B23A6E")
    a1.axhline(0, ls="--", color="0.6")
    a1.set_xscale("log"); a1.set_xlabel("# training sequences")
    a1.set_ylabel("corr(effect size, leakage)  [r_leak]")
    a1.set_title("Leakage correlation vs training-set size")
    a2.plot(df.n_samples, df.r_pred, "o-", color="#4C9BD4", label="r_pred (active)")
    a2.plot(df.n_samples, df.mean_leak * 50, "s-", color="#B23A6E",
            label="mean leakage ×50")
    a2.axhline(0, ls="--", color="0.6")
    a2.set_xscale("log"); a2.set_xlabel("# training sequences")
    a2.set_title("Fit quality & mean leakage"); a2.legend()
    fig.tight_layout(); fig.savefig(OUT / "data_sweep_leakage_vs_size.pdf")
    plt.close(fig)
    print("\ndone ->", OUT)


if __name__ == "__main__":
    main()
