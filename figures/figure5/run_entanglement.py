#!/usr/bin/env python
"""Figure 5C/D/E + supplement (b): entanglement of Beluga hidden activations.

Reproduces the evidence that AP-1 and CTCF motifs share overlapping internal
representations in Beluga's final conv layer (model.0.14):
  * Panel D: scatter of mean AP-1 vs CTCF activation deltas (r ~ 0.27).
  * Panel E: an AP-1 probe trained on those activations also detects CTCF
             above chance (AP1 AUC ~ 0.88, CTCF AUC ~ 0.65).
  * Supplement (b): probe validation AUC per epoch.

Outputs (outputs/): ctcf_ap1_correlation.pdf, roc_curve_comparison.pdf,
    validation_auc.pdf, entanglement_metrics.txt.

Run (env: featleak-dl):
    python run_entanglement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve, auc  # noqa: E402
from tangermeme.utils import random_one_hot  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # editable text in Illustrator
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_beluga  # noqa: E402
from common.motifs import AP1, motif_to_seq  # noqa: E402
from common.beluga_model import (  # noqa: E402
    extract_activations_batch,
    BelugaProbeWithClampedActivations,
    SyntheticMotifClassificationDataset,
)

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def activation_entanglement(mdl, ctcf_motif):
    """Panel D: correlation of AP1 vs CTCF activation deltas at model.0.14."""
    X = random_one_hot((100, 4, 2000)).type(torch.float32)
    X_ctcf = substitute(X, ctcf_motif)
    X_ap1 = substitute(X, AP1)

    act = extract_activations_batch(mdl, "model.0.14", X, batch_size=32)
    act_ctcf = extract_activations_batch(mdl, "model.0.14", X_ctcf, batch_size=32)
    act_ap1 = extract_activations_batch(mdl, "model.0.14", X_ap1, batch_size=32)

    ctcf_delta = [(act_ctcf[i] - act[i]).flatten() for i in range(len(act))]
    ap1_delta = [(act_ap1[i] - act[i]).flatten() for i in range(len(act))]
    ctcf_activations = np.mean(ctcf_delta, axis=0)
    ap1_activations = np.mean(ap1_delta, axis=0)
    r, p = pearsonr(ctcf_activations, ap1_activations)

    reg = LinearRegression().fit(ctcf_activations[:, None], ap1_activations)
    xs = np.linspace(ctcf_activations.min(), ctcf_activations.max(), 100)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(ctcf_activations, ap1_activations, s=16, alpha=0.5,
               color="#2E86AB", edgecolor="none")
    ax.plot(xs, reg.predict(xs[:, None]), color="#A23B72", lw=2)
    ax.text(0.04, 0.96, f"r = {r:.3f}\np = {p:.1e}", transform=ax.transAxes,
            va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xlabel("CTCF Activations"); ax.set_ylabel("AP1 Activations")
    fig.tight_layout(); fig.savefig(OUT / "ctcf_ap1_correlation.pdf"); plt.close(fig)
    return r, p


def train_probe(mdl, ctcf_motif, num_epochs=20, patience=5):
    """Panels E + supp (b): AP1 probe on Beluga activations, test on CTCF."""
    probe = BelugaProbeWithClampedActivations(mdl, None, 1).to(DEVICE)
    probe.beluga_model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    full = SyntheticMotifClassificationDataset(N=14_000, L=2000, motif=AP1, seed=42)
    train_set, val_set, test_set = random_split(
        full, [10_000, 2_000, 2_000],
        generator=torch.Generator().manual_seed(123))
    mk = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh,
                                       num_workers=4, pin_memory=True)
    train_loader = mk(train_set, 32, True)
    val_loader = mk(val_set, 64, False)
    test_loader = mk(test_set, 64, False)

    @torch.no_grad()
    def collect(loader):
        probe.eval()
        lg, tg = [], []
        for X, y in loader:
            lg.append(probe(X.to(DEVICE)).float().cpu())
            tg.append(y.float().unsqueeze(-1))
        return (torch.cat(lg).squeeze(1).numpy(),
                torch.cat(tg).squeeze(1).numpy())

    def val_auc_now():
        lg, tg = collect(val_loader)
        return roc_auc_score(tg, lg)

    history = {"val_auc": []}
    best_auc, best_state, no_improve = -np.inf, None, 0
    for epoch in range(1, num_epochs + 1):
        probe.train()
        for X, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(X.to(DEVICE)), y.to(DEVICE).unsqueeze(-1))
            loss.backward(); opt.step()
        va = val_auc_now()
        history["val_auc"].append(va)
        print(f"  epoch {epoch:02d} val_auc={va:.4f}")
        if va > best_auc:
            best_auc, no_improve = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop at epoch {epoch}")
                break
    if best_state:
        probe.load_state_dict(best_state)

    ap1_lg, ap1_tg = collect(test_loader)
    ctcf_test = SyntheticMotifClassificationDataset(N=2_000, L=2000,
                                                    motif=ctcf_motif, seed=42)
    ctcf_lg, ctcf_tg = collect(mk(ctcf_test, 64, False))
    ap1_fpr, ap1_tpr, _ = roc_curve(ap1_tg, ap1_lg); ap1_auc = auc(ap1_fpr, ap1_tpr)
    ctcf_fpr, ctcf_tpr, _ = roc_curve(ctcf_tg, ctcf_lg); ctcf_auc = auc(ctcf_fpr, ctcf_tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(ap1_fpr, ap1_tpr, color="#2E86C1", lw=2.5,
            label=f"AP1 (AUC = {ap1_auc:.3f})")
    ax.plot(ctcf_fpr, ctcf_tpr, color="#E74C3C", lw=2.5,
            label=f"CTCF (AUC = {ctcf_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#7F8C8D", lw=2, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "roc_curve_comparison.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ys = [0.5] + history["val_auc"]
    ax.plot(np.arange(len(ys)), ys, color="#27AE60", lw=2.5, label="Validation AUC")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Validation AUC"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "validation_auc.pdf"); plt.close(fig)
    return ap1_auc, ctcf_auc


def main() -> None:
    OUT.mkdir(exist_ok=True)
    mdl = load_beluga(device=DEVICE)
    ctcf_motif = motif_to_seq()["CTCFL_HUMAN.H11MO.0.A"]

    print("[1/2] activation entanglement (Panel D) ...")
    r, p = activation_entanglement(mdl, ctcf_motif)
    print(f"    AP1 vs CTCF activation r = {r:.3f} (p={p:.1e})")

    print("[2/2] training AP1 probe (Panel E + supp b) ...")
    ap1_auc, ctcf_auc = train_probe(mdl, ctcf_motif)
    print(f"    AP1 probe AUC = {ap1_auc:.3f} ; CTCF (cross) AUC = {ctcf_auc:.3f}")

    with open(OUT / "entanglement_metrics.txt", "w") as fh:
        fh.write(f"activation_correlation_r\t{r:.4f}\n")
        fh.write(f"activation_correlation_p\t{p:.3e}\n")
        fh.write(f"ap1_probe_auc\t{ap1_auc:.4f}\n")
        fh.write(f"ctcf_cross_auc\t{ctcf_auc:.4f}\n")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
