#!/usr/bin/env python
"""Train an AP-1 counting head on frozen Beluga and measure motif leakage into it (Figure 3).

Outputs (outputs/): scatter_plot_with_fit.pdf, logo_<motif>.pdf, top_4_motifs_violin_plot.pdf,
motif_significance_analysis.pdf and motif_effect_results.npz.

Run (env: featleak-dl):
    python run.py --epochs 5 --train-size 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from tqdm import tqdm  # noqa: E402
from tangermeme.utils import random_one_hot  # noqa: E402
from tangermeme.ersatz import substitute, multisubstitute  # noqa: E402
from tangermeme.deep_lift_shap import deep_lift_shap  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_beluga  # noqa: E402
from common.motifs import AP1, filtered_motifs_for_ap1  # noqa: E402
from common.beluga_model import (  # noqa: E402
    train_ap1_motif_counting,
    measure_motif_insertion_effect,
    SyntheticMotifCountDataset,
)

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Representative motifs for the DeepLIFT logo panels C-G.
LOGO_MOTIFS = {
    "AP1": None,  # special-cased (two AP1 motifs implanted)
    "CTCFL_HUMAN.H11MO.0.A": "CTCF",
    "EWSR1-FLI1_MA0149.1": "EWSR1-FLI1",
    "NFKB1_MA0105.4": "NFKB1",
    "ZIC1_C2H2_1": "ZIC1",
}


def scatter_performance(trained, base_mdl):
    """Supp (a): predicted vs actual count."""
    ds = SyntheticMotifCountDataset(N=1000, L=2000, motif=AP1, max_count=5, seed=42)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds, labels = [], []
    trained.eval()
    with torch.no_grad():
        for seqs, y in loader:
            out = trained(seqs.to(DEVICE))[:, 2002].cpu().numpy()
            preds.extend(out); labels.extend(y.numpy())
    preds, labels = np.array(preds), np.array(labels)
    r = stats.pearsonr(labels, preds)[0]
    slope, intercept, *_ = stats.linregress(labels, preds)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.scatter(labels, preds, alpha=0.3, s=18, color="steelblue", edgecolors="none")
    ax.plot(labels, slope * labels + intercept, "r-", lw=2)
    ax.text(0.05, 0.95, f"R = {r:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Actual Number of AP1 Motifs")
    ax.set_ylabel("Predicted Number\nof AP1 Motifs")
    fig.tight_layout(); fig.savefig(OUT / "scatter_plot_with_fit.pdf"); plt.close(fig)
    print(f"    counting performance R = {r:.3f} (R^2={r2_score(labels, preds):.3f})")


def deeplift_logo(base_mdl, name, seq, label):
    """Panels C-G: DeepLIFT contribution logo for the counting output."""
    if name == "AP1":
        X = random_one_hot((1, 4, 2000)).type(torch.float32)
        X = multisubstitute(X, [AP1, AP1], 30)
        X_attr = deep_lift_shap(base_mdl, X, target=2002, random_state=0)
        attr = X_attr[0, :, 950:1050]
    else:
        X = random_one_hot((100, 4, 2000)).type(torch.float32)
        X = substitute(X, seq)
        X_attr = deep_lift_shap(base_mdl, X, target=2002, random_state=0)
        attr = X_attr.mean(axis=0)[:, 950:1050]
    fig = plt.figure(figsize=(10, 2), dpi=200)
    ax = plt.subplot(111)
    plot_logo(attr, ax=ax)
    ax.set_title(label)
    fig.tight_layout(); fig.savefig(OUT / f"logo_{label}.pdf"); plt.close(fig)


def violin_top4(results):
    """Panel B: violin plot of top-4 leakage effects by |mean|."""
    order = sorted(results, key=lambda k: abs(results[k].mean()), reverse=True)
    top4 = order[:4]
    print("    top-4 leakage motifs:", top4)
    data = [results[m] for m in top4]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, positions=range(1, 5), showmeans=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.axhline(0, ls="--", color="0.4")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([m.split("_")[0] for m in top4], rotation=30, ha="right")
    ax.set_ylabel("Motif Effect"); ax.set_xlabel("Motifs")
    ax.set_title("Top 4 Motifs by Absolute Mean Effect")
    fig.tight_layout(); fig.savefig(OUT / "top_4_motifs_violin_plot.pdf"); plt.close(fig)


def significance(results):
    """Supp (b/c): pie of significant motifs + FDR p-value histogram."""
    names = list(results)
    pvals = []
    for m in names:
        e = results[m][results[m] != 0]
        pvals.append(wilcoxon(e, alternative="two-sided")[1] if len(e) else 1.0)
    pvals = np.array(pvals)
    rejected, p_adj, *_ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    n_sig = int(rejected.sum()); n_tot = len(names)
    print(f"    {n_sig}/{n_tot} motifs significant (FDR<0.05)")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.pie([n_sig, n_tot - n_sig],
           labels=[f"Significant\n({n_sig})", f"Non-significant\n({n_tot - n_sig})"],
           autopct="%1.1f%%", colors=["#E8503A", "#8C9A9A"])
    a1.set_title("Proportion of Motifs Significantly\nDifferent from Zero")
    a2.hist(p_adj, bins=20, color="#7EA8D6", edgecolor="white")
    a2.axvline(0.05, ls="--", color="red", label="FDR threshold (0.05)")
    a2.set_xlabel("FDR-corrected p-values"); a2.set_ylabel("Count"); a2.legend()
    a2.set_title("Distribution of FDR-corrected p-values")
    fig.tight_layout(); fig.savefig(OUT / "motif_significance_analysis.pdf"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--num-sequences", type=int, default=100,
                    help="sequences per motif for the leakage marginalization")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print("[1/5] loading Beluga + adding counting head ...")
    base_mdl = load_beluga(device=DEVICE, with_counting=True)

    print("[2/5] training AP1 counting head (frozen backbone) ...")
    trained, _ = train_ap1_motif_counting(
        model=base_mdl, num_epochs=args.epochs, batch_size=32,
        learning_rate=1e-4, motif=AP1, max_count=5,
        train_size=args.train_size, val_size=1000,
        device=DEVICE, freeze_backbone=True,
    )
    scatter_performance(trained, base_mdl)

    print("[3/5] DeepLIFT logos (Panels C-G) ...")
    filtered = filtered_motifs_for_ap1(threshold=50)
    for name, label in LOGO_MOTIFS.items():
        seq = None if name == "AP1" else filtered.get(name)
        deeplift_logo(base_mdl, name, seq, label or "AP1")

    print(f"[4/5] leakage marginalization over {len(filtered)} motifs ...")
    results = {}
    for name, seq in tqdm(filtered.items(), total=len(filtered)):
        results[name] = measure_motif_insertion_effect(
            trained, seq, num_sequences=args.num_sequences,
            sequence_length=2000, device=DEVICE, seed=42)
    np.savez(OUT / "motif_effect_results.npz",
             **{k: v for k, v in results.items()})

    print("[5/5] violin + significance plots ...")
    violin_top4(results)
    significance(results)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
