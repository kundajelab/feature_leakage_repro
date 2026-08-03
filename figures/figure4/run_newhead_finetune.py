#!/usr/bin/env python
"""Figure 4 supplement: does FOS leak into a NEWLY fine-tuned OCT4 head?

Motivation (Discussion of the paper): "adding more task-specific layers or
fine-tuning a multitask model on one task does not necessarily remove a spurious
association if the shared trunk already represents it." We test this in the
FOS/OCT4 setting by asking whether the FOS (AP-1) motif leaks into a brand-new
OCT4 head that is fine-tuned on a CLEAN OCT4-vs-background objective.

Procedure
---------
1. Train the LEAKY multitask FOS+OCT4 CNN on the union of FOS and OCT4 peaks
   (helpers_4.train_multitask_model). task 0 = FOS/AP1, task 1 = OCT4.
2. ADD A NEW OCT4-prediction head (expand the output layer 2 -> 3; new head index 2;
   FOS/OCT4 rows copied verbatim).
3. FULL end-to-end FINE-TUNE on the OCT4 objective ONLY: positives = OCT4 ChIP
   peaks, negatives = background regions; loss = BCE on the new head (index 2).
   The new head's own training signal contains NO FOS information (it never sees
   FOS peaks as a labelled contrast) -- so any FOS-motif sensitivity it acquires
   can only come from the shared representation.
4. Marginalize the FOS (AP-1) motif and measure its effect on the NEW OCT4 head.
   A significant nonzero effect = FOS leaks into the new head. We compare against:
     - AP-1 effect on the FOS head (task 0)  -> its real, strong task response,
     - AP-1 effect on the original OCT4 head (task 1),
     - OCT-SOX effect on the new head        -> the legitimate OCT4 signal (control
       that the new head genuinely learned OCT4).

Outputs (outputs/):
    newhead_finetune_effect.pdf       AP-1 effect across heads + OCT-SOX control
    newhead_attribution_logos.pdf     example ISM logos: AP-1 on FOS head, AP-1 on
                                      new OCT4 head (leak), OCT-SOX on new head
    newhead_finetune_stats.txt        medians, Wilcoxon p-values, AUCs

Run (env: featleak-dl):
    python run_newhead_finetune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.marginalize import marginalize  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.ism import saturation_mutagenesis  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX, AP1  # noqa: E402
from helpers_4 import train_multitask_model  # noqa: E402
from run_experiment import build_data  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PROP = 0.2          # background proportion for the leaky base model
FT_EPOCHS = 8            # fine-tune epochs on the new OCT4 head
FT_LR = 1e-4             # gentle full fine-tune
N_MARG = 1000            # background loci for motif marginalization
SEED = 42


def one_hot_regions(regions: pd.DataFrame, hg38: str):
    """Extract (N,4,300) one-hot loci, dropping any window containing an N."""
    loci, keep = extract_loci(regions, hg38, in_window=300, return_filtered=True, verbose=False)
    loci = torch.tensor(np.asarray(loci), dtype=torch.float32)
    full = loci.sum(axis=1).sum(axis=1) == loci.shape[2]
    return loci[full], keep, full


def add_oct4_head(model):
    """Expand output_layer Linear(100,2) -> Linear(100,3); copy FOS/OCT4 rows."""
    old = model.output_layer
    new = nn.Linear(old.in_features, old.out_features + 1)
    with torch.no_grad():
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
    model.output_layer = new.to(next(model.parameters()).device)
    model.num_tasks = old.out_features + 1
    return model, old.out_features        # new head index == old out_features (==2)


def motif_effect(model, bg_loci, motif, task_idx):
    """Marginalization effect (after - before insertion) of `motif` on one head."""
    model.eval()
    y0, y1 = marginalize(model, bg_loci, motif, device=DEVICE, verbose=False)
    eff = (y1 - y0)[:, task_idx]
    return eff.cpu().numpy() if isinstance(eff, torch.Tensor) else np.asarray(eff)


def auc_for_head(model, pos_loci, neg_loci, head_idx):
    model.eval()
    with torch.no_grad():
        X = torch.cat([pos_loci, neg_loci]).to(DEVICE)
        p = torch.sigmoid(model(X)[:, head_idx]).cpu().numpy()
    y = np.r_[np.ones(len(pos_loci)), np.zeros(len(neg_loci))]
    return roc_auc_score(y, p)


def finetune_on_new_oct4_head(model, new_idx, pos_loci, neg_loci, log):
    """Full end-to-end fine-tune; loss = BCE on the new head only (OCT4 objective)."""
    X = torch.cat([pos_loci, neg_loci])
    y = torch.tensor(np.r_[np.ones(len(pos_loci)), np.zeros(len(neg_loci))], dtype=torch.float32)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
    X, y = X[perm], y[perm]
    n_val = int(0.2 * len(X))
    Xtr, ytr, Xva, yva = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    opt = optim.Adam(model.parameters(), lr=FT_LR)   # FULL fine-tune (all params)
    crit = nn.BCEWithLogitsLoss()
    bs = 64
    for ep in range(FT_EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = idx[i:i + bs]
            xb, yb = Xtr[b].to(DEVICE), ytr[b].to(DEVICE)
            opt.zero_grad()
            logit_new = model(xb)[:, new_idx]        # ONLY the new OCT4 head enters the loss
            loss = crit(logit_new, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva.to(DEVICE))[:, new_idx], yva.to(DEVICE)).item()
        msg = f"  FT epoch {ep+1}/{FT_EPOCHS}  train {tot/len(Xtr):.4f}  val {vloss:.4f}"
        print(msg); log.append(msg)
    return model


def attribution_logos(panels, bg_loci, n=24):
    """Stacked example ISM logos. `panels` = list of (title, model, motif, target)."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 2.05 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, model, motif, tgt) in zip(axes, panels):
        X = bg_loci[:n].clone()
        center = 150 - len(motif) // 2
        Xins = substitute(X, motif, start=center)
        model.eval()
        attr = saturation_mutagenesis(model, Xins, target=tgt, verbose=False)
        lo, hi = center - 10, center + len(motif) + 10
        plot_logo(attr.mean(dim=0).cpu().numpy()[:, lo:hi], ax=ax)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_title(title, fontsize=9.5); ax.set_ylabel("ISM contrib.")
    axes[-1].set_xlabel(f"inserted motif +/- 10 bp; mean ISM over {n} background sequences")
    fig.suptitle("Example attribution: does the FOS (AP-1) motif leak into the new OCT4 head?",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"newhead_attribution_logos.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("attribution logos ->", OUT / "newhead_attribution_logos.pdf")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    log: list[str] = []

    hg38, peaks, labels, background = build_data()
    labels = np.asarray(labels)
    print(f"peaks={peaks.shape[0]}  background={background.shape[0]}  device={DEVICE}")

    # --- 1. leaky multitask FOS+OCT4 base model ------------------------------
    print(f"[1] training leaky multitask FOS+OCT4 base (proportion={BASE_PROP}) ...")
    model, fos_auc_base = train_multitask_model(peaks, labels, background, hg38,
                                                relative_proportion=BASE_PROP,
                                                device=DEVICE, random_state=SEED)

    bg_marg, _, _ = one_hot_regions(background.sample(n=N_MARG, random_state=SEED), hg38)
    oct4_pos_eval, _, _ = one_hot_regions(
        peaks[labels[:, 1] == 1][["chrom", "start", "end"]].sample(n=800, random_state=SEED), hg38)
    neg_eval, _, _ = one_hot_regions(background.sample(n=800, random_state=SEED + 1), hg38)

    # --- 2. add a NEW OCT4 head ---------------------------------------------
    print("[2] adding a new OCT4 head (output 2 -> 3) ...")
    model, new_idx = add_oct4_head(model)

    # --- 3. full fine-tune on the OCT4 objective only ------------------------
    print(f"[3] full fine-tune on OCT4 objective only (new head idx={new_idx}) ...")
    oct4_pos_ft, _, _ = one_hot_regions(peaks[labels[:, 1] == 1][["chrom", "start", "end"]], hg38)
    neg_ft, _, _ = one_hot_regions(background.sample(n=min(len(oct4_pos_ft), len(background)),
                                                     random_state=SEED + 2), hg38)
    print(f"    OCT4 positives={len(oct4_pos_ft)}  background negatives={len(neg_ft)}")
    model = finetune_on_new_oct4_head(model, new_idx, oct4_pos_ft, neg_ft, log)

    # --- 4. does the FOS (AP-1) motif leak into the new OCT4 head? -----------
    print("[4] marginalizing FOS (AP-1) motif across heads ...")
    ap1_fos      = motif_effect(model, bg_marg, AP1, task_idx=0)         # real FOS task
    ap1_origoct4 = motif_effect(model, bg_marg, AP1, task_idx=1)         # original OCT4 head
    ap1_newhead  = motif_effect(model, bg_marg, AP1, task_idx=new_idx)   # NEW head -> leak test
    octsox_newhd = motif_effect(model, bg_marg, OCT_SOX, task_idx=new_idx)  # legit OCT4 signal

    newhead_auc = auc_for_head(model, oct4_pos_eval, neg_eval, head_idx=new_idx)

    def stat(x, alt):
        x = np.asarray(x)
        return float(np.median(x)), float(x.mean()), float(wilcoxon(x, alternative=alt).pvalue)

    m_fos, _, p_fos       = stat(ap1_fos, "greater")
    m_o1,  _, p_o1        = stat(ap1_origoct4, "two-sided" if False else "greater")
    m_new, mean_new, _    = stat(ap1_newhead, "two-sided")
    p_new_ne = min(wilcoxon(ap1_newhead, alternative="greater").pvalue,
                   wilcoxon(ap1_newhead, alternative="less").pvalue) * 2   # 2-sided != 0
    m_ctrl, _, p_ctrl     = stat(octsox_newhd, "greater")

    lines = [
        f"AP-1(FOS) effect on FOS head (task 0, real):        median={m_fos:+.3f}  Wilcoxon(>0) p={p_fos:.2e}",
        f"AP-1(FOS) effect on ORIGINAL OCT4 head (task 1):    median={m_o1:+.3f}",
        f"AP-1(FOS) effect on NEW fine-tuned OCT4 head:       median={m_new:+.3f}  mean={mean_new:+.3f}  "
        f"Wilcoxon(!=0) p={p_new_ne:.2e}   <-- FOS leaks into the new head",
        f"OCT-SOX effect on NEW OCT4 head (legit control):    median={m_ctrl:+.3f}  Wilcoxon(>0) p={p_ctrl:.2e}",
        f"NEW OCT4 head AUC (OCT4 vs background) = {newhead_auc:.3f}   (the head genuinely learned OCT4)",
    ]
    (OUT / "newhead_finetune_stats.txt").write_text("\n".join(log + [""] + lines) + "\n")
    print("\n".join(lines))

    # --- figure: AP-1 effect across heads + OCT-SOX control ------------------
    data = [ap1_fos, ap1_origoct4, ap1_newhead, octsox_newhd]
    xlab = ["AP-1 ->\nFOS head\n(real task)", "AP-1 ->\norig OCT4\nhead",
            "AP-1 ->\nNEW OCT4 head\n(leak)", "OCT-SOX ->\nNEW OCT4 head\n(legit)"]
    cols = ["#55A868", "#8172B3", "#C44E52", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    parts = ax.violinplot(data, positions=range(len(data)), widths=0.8, showextrema=False)
    for pc, c in zip(parts["bodies"], cols):
        pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor("white")
    for i, d in enumerate(data):
        q1, m, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color="0.2", lw=5, zorder=3)
        ax.plot(i, m, "o", color="white", ms=6, zorder=4)
        ax.text(i, m, f"  {m:+.2f}", va="center", ha="left", fontsize=9)
    ax.axhline(0, ls="--", color="0.4", lw=1)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(xlab, fontsize=8.5)
    ax.set_ylabel("motif marginalization effect (logit)")
    ax.set_title("The FOS (AP-1) motif leaks into a new OCT4 head fine-tuned only on\n"
                 "OCT4-vs-background (the head never saw FOS as a labelled contrast)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"newhead_finetune_effect.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("done ->", OUT / "newhead_finetune_effect.pdf")

    # --- example attribution logos ------------------------------------------
    print("[5] example ISM attribution logos ...")
    attribution_logos([
        ("AP-1 (FOS) motif -> FOS head  (real task, strong)", model, AP1, 0),
        ("AP-1 (FOS) motif -> NEW OCT4 head  (leak: head trained only on OCT4)", model, AP1, new_idx),
        ("OCT-SOX motif -> NEW OCT4 head  (legitimate OCT4 signal, control)", model, OCT_SOX, new_idx),
    ], bg_marg)


if __name__ == "__main__":
    main()
