#!/usr/bin/env python
"""Train a multi-task CNN on simulated motif sequences and measure per-motif leakage onto inactive tasks (Figure 5A/B, Supplementary a).

The defaults put the model in the data-limited regime that genomic training sets occupy:
5,000 sequences for 100 motifs across 50 tasks. Training runs to convergence with early
stopping on validation loss and the best checkpoint is restored before measurement.

Outputs (outputs/): leakage_effect_size_correlation.pdf, effect_size_correlation.pdf,
simulation_metrics.csv, model_trained.pth, motif_dict_trained.pkl.

Run (env: featleak-dl):
    python run_simulation.py --seed 1
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # editable text in Illustrator
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

import sim_helpers as H  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-motifs", type=int, default=100)
    ap.add_argument("--motif-length", type=int, default=10)
    ap.add_argument("--num-tasks", type=int, default=50) 
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=600,
                    help="maximum epochs; training stops early on --patience.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=46,
                    help="epochs without a validation improvement before stopping.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/4] building motif library + dataset ...")
    motif_dict = H.make_motif_dict(
        args.num_motifs, args.motif_length, num_tasks=args.num_tasks
    )
    sequences, labels = H.generate_dataset(
        motif_dict, args.n_samples, num_tasks=args.num_tasks
    )
    train_loader, val_loader, _ = H.create_dataloaders(sequences, labels,
                                                       batch_size=args.batch_size)

    print("[2/4] training multi-task CNN ...")
    model, train_losses, val_losses = H.train_simple_model(
        train_loader, val_loader, num_tasks=args.num_tasks,
        num_epochs=args.epochs, lr=args.lr, device=args.device,
        patience=args.patience,
    )
    print(f"    min val loss = {min(val_losses):.3f}")
    torch.save(model.state_dict(), OUT / "model_trained.pth")
    with open(OUT / "motif_dict_trained.pkl", "wb") as fh:
        pickle.dump(motif_dict, fh)

    print("[3/4] measuring per-motif leakage ...")
    model = model.to(args.device).eval()
    rows = []
    for motif, meta in motif_dict.items():
        rows.append(
            H.measure_feature_leakage(
                motif, model.cpu(), meta["sparsity"], meta["effect"]
            )
        )
        model = model.to(args.device)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation_metrics.csv", index=False)

    r_leak, p_leak = pearsonr(df.actual_effect_size, df.leakage_effect)
    r_pred, p_pred = pearsonr(df.actual_effect_size, df.predicted_effect)
    print(f"    leakage vs true effect:   r={r_leak:.3f} (p={p_leak:.1e})")
    print(f"    predicted vs true effect: r={r_pred:.3f} (p={p_pred:.1e})")

    print("[4/4] plotting ...")
    _scatter(df.actual_effect_size, df.leakage_effect,
             "Actual Effect Size", "Leakage Effect",
             f"r = {r_leak:.3f}\np = {p_leak:.1e}",
             "Correlation Between Actual and Leakage Effect Sizes",
             OUT / "leakage_effect_size_correlation.pdf")
    _scatter(df.actual_effect_size, df.predicted_effect,
             "Actual Effect Size", "Predicted Effect",
             f"r = {r_pred:.3f}\np = {p_pred:.1e}",
             "Correlation Between Actual and Predicted Effect Sizes",
             OUT / "effect_size_correlation.pdf")
    print("done ->", OUT)


def _scatter(x, y, xlabel, ylabel, annot, title, path) -> None:
    x = np.asarray(x); y = np.asarray(y)
    reg = LinearRegression().fit(x[:, None], y)
    xs = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(x, y, s=18, alpha=0.6, color="#4C9BD4", edgecolor="none")
    ax.plot(xs, reg.predict(xs[:, None]), color="#B23A6E", lw=2)
    ax.text(0.03, 0.97, annot, transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
