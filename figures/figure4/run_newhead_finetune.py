#!/usr/bin/env python
"""Fine-tune a new head on a multi-task FOS/OCT4 CNN and test whether the other factor's motif leaks into it (Figure 4 supplement).

Outputs (outputs/): newhead[_fos]_finetune_effect.{pdf,png,svg},
newhead[_fos]_attribution_logos.{pdf,png,svg} and newhead[_fos]_finetune_stats.txt.

Run (env: featleak-dl):
    python run_newhead_finetune.py --new-head oct4
    python run_newhead_finetune.py --new-head fos
"""

from __future__ import annotations

import argparse
import os
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

import pybedtools  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX, AP1  # noqa: E402
from common.gc_negatives import gc_matched_negatives  # noqa: E402
from helpers_3 import merge_bed_files_and_label  # noqa: E402
from helpers_4 import train_multitask_model, train_singletask_model  # noqa: E402


def build_data():
    """FOS+OCT4 merged peaks/labels + a background of non-peak regions.

    Background prefers the curated GC-matched negatives (config keys
    gc_neg_fos / gc_neg_oct4); if those BEDs are not readable on this machine it
    falls back to GC-matched negatives sampled from the genome, so the experiment
    stays self-contained.
    """
    hg38 = str(config.resolve("hg38_fasta", must_exist=True))
    fos = str(config.resolve("chip_fos_fibroblast", must_exist=True))
    oct4 = str(config.resolve("chip_oct4_hesc", must_exist=True))
    peaks, labels = merge_bed_files_and_label(fos, oct4, peak_size=300)

    active = pybedtools.BedTool.from_dataframe(peaks[["chrom", "start", "end"]])
    bg_parts, curated = [], True
    for key in ("gc_neg_fos", "gc_neg_oct4"):
        p = config.resolve(key)
        try:
            readable = p.is_file() and os.access(p, os.R_OK)
        except OSError:
            readable = False
        if not readable:
            curated = False
            break
        bt = pybedtools.BedTool(str(p))
        bg_parts.append(bt.window(active, w=1000, v=True).to_dataframe())
    if curated:
        background = pd.concat(bg_parts)[["chrom", "start", "end"]]
    else:
        print("  curated gc_neg_* BEDs not readable -> generating GC-matched "
              "negatives from the genome")
        background = gc_matched_negatives(hg38, peaks[["chrom", "start", "end"]],
                                          width=300, n_target=40000, seed=SEED)
    return hg38, peaks, labels, background

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PROP = 0.2          # background proportion for the leaky base model
FT_EPOCHS = 8            # fine-tune epochs on the new head
FT_LR = 1e-4             # gentle full fine-tune
N_MARG = 1000            # background loci for motif marginalization
SEED = 42

# Task/motif roles for each new-head choice. task 0 = FOS/AP1, task 1 = OCT4.
#   pos_label_col : which peak label column supplies the fine-tune positives
#   leak_*        : the OTHER factor's motif, whose leak into the new head we test
#   legit_*       : the new head's OWN motif (legitimate-signal control)
#   own_task_idx  : head on which the leak motif is a real, strong response
#   orig_idx      : the original head of the new head's task
NEWHEAD = {
    # single_task_control: only meaningful for the FOS direction. In fibroblast,
    # OCT-SOX is truly irrelevant, so a from-scratch single-task FOS model gives a
    # clean ~0 null. In the OCT4 direction a from-scratch single-task OCT4 model
    # assigns AP-1 a real repressive (anti-correlation) effect in hESC, which is
    # not spurious leakage, so we omit that confounded control there.
    # *_disp are the ChIP-target display names used in the violin labels
    # (the AP-1 motif is the FOS target; the OCT-SOX motif is the OCT4 target).
    "oct4": dict(
        pos_label_col=1, task_name="OCT4",
        leak_motif=AP1, leak_name="AP-1 (FOS)", leak_short="AP-1", leak_disp="FOS",
        legit_motif=OCT_SOX, legit_name="OCT-SOX", legit_disp="OCT4",
        own_task_idx=0, own_task_name="FOS", orig_idx=1, orig_name="OCT4",
        single_task_control=False, suffix="",
    ),
    "fos": dict(
        pos_label_col=0, task_name="FOS",
        leak_motif=OCT_SOX, leak_name="OCT-SOX", leak_short="OCT-SOX", leak_disp="OCT4",
        legit_motif=AP1, legit_name="AP-1 (FOS)", legit_disp="FOS",
        own_task_idx=1, own_task_name="OCT4", orig_idx=0, orig_name="FOS",
        single_task_control=True, suffix="_fos",
    ),
}


def one_hot_regions(regions: pd.DataFrame, hg38: str):
    """Extract (N,4,300) one-hot loci, dropping any window containing an N."""
    loci, keep = extract_loci(regions, hg38, in_window=300, return_filtered=True, verbose=False)
    loci = torch.tensor(np.asarray(loci), dtype=torch.float32)
    full = loci.sum(axis=1).sum(axis=1) == loci.shape[2]
    return loci[full], keep, full


def add_head(model):
    """Expand output_layer Linear(100,2) -> Linear(100,3); copy existing rows."""
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


def finetune_on_new_head(model, new_idx, pos_loci, neg_loci, log):
    """Full end-to-end fine-tune; loss = BCE on the new head only (one task)."""
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
            logit_new = model(xb)[:, new_idx]        # ONLY the new head enters the loss
            loss = crit(logit_new, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva.to(DEVICE))[:, new_idx], yva.to(DEVICE)).item()
        msg = f"  FT epoch {ep+1}/{FT_EPOCHS}  train {tot/len(Xtr):.4f}  val {vloss:.4f}"
        print(msg); log.append(msg)
    return model


def attribution_logos(panels, bg_loci, cfg, n=24):
    """Stacked example ISM logos.

    `panels` = list of (title, model, motif, target[, group]). Panels sharing the
    same non-None `group` are drawn on a common y-axis (the larger of the group),
    so e.g. the leak row and the single-task control row are directly comparable
    and a flat track reads as flat instead of being autoscaled to look like a
    motif. Panels with no group autoscale to their own magnitude.
    """
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 2.05 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    # First pass: compute all attribution slices + their per-panel magnitudes.
    slices, groups, mags = [], [], []
    for p in panels:
        title, model, motif, tgt = p[0], p[1], p[2], p[3]
        group = p[4] if len(p) > 4 else None
        X = bg_loci[:n].clone()
        center = 150 - len(motif) // 2
        Xins = substitute(X, motif, start=center)
        model.eval()
        attr = saturation_mutagenesis(model, Xins, target=tgt, verbose=False)
        lo, hi = center - 10, center + len(motif) + 10
        sl = attr.mean(dim=0).cpu().numpy()[:, lo:hi]
        slices.append(sl); groups.append(group)
        mags.append(1.05 * np.abs(sl).sum(axis=0).max())   # tallest stacked column
    # A grouped panel uses the max magnitude across its group.
    group_pad = {}
    for g, m in zip(groups, mags):
        if g is not None:
            group_pad[g] = max(group_pad.get(g, 0.0), m)
    for ax, p, sl, g, m in zip(axes, panels, slices, groups, mags):
        plot_logo(sl, ax=ax)
        ax.axhline(0, color="0.6", lw=0.7)
        pad = group_pad[g] if g is not None else m
        ax.set_ylim(-pad, pad)
        ax.set_title(p[0], fontsize=9.5); ax.set_ylabel("contributions")
        ax.set_xticks([])          # no x ticks / tick labels / axis title
    fig.suptitle(f"Example attribution: does the {cfg['leak_name']} motif leak into "
                 f"the new {cfg['task_name']} head?", fontsize=11)
    fig.tight_layout()
    stem = f"newhead{cfg['suffix']}_attribution_logos"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("attribution logos ->", OUT / f"{stem}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-head", choices=list(NEWHEAD), default="oct4",
                    help="which new task head to add and fine-tune")
    args = ap.parse_args()
    cfg = NEWHEAD[args.new_head]

    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    log: list[str] = []

    hg38, peaks, labels, background = build_data()
    labels = np.asarray(labels)
    col = cfg["pos_label_col"]
    print(f"new head = {cfg['task_name']}   peaks={peaks.shape[0]}  "
          f"background={background.shape[0]}  device={DEVICE}")

    # --- 1. leaky multitask FOS+OCT4 base model ------------------------------
    print(f"[1] training leaky multitask FOS+OCT4 base (proportion={BASE_PROP}) ...")
    model, _ = train_multitask_model(peaks, labels, background, hg38,
                                     relative_proportion=BASE_PROP,
                                     device=DEVICE, random_state=SEED)

    bg_marg, _, _ = one_hot_regions(background.sample(n=N_MARG, random_state=SEED), hg38)
    pos_eval, _, _ = one_hot_regions(
        peaks[labels[:, col] == 1][["chrom", "start", "end"]].sample(n=800, random_state=SEED), hg38)
    neg_eval, _, _ = one_hot_regions(background.sample(n=800, random_state=SEED + 1), hg38)

    # --- 2. add a NEW head for the chosen task ------------------------------
    print(f"[2] adding a new {cfg['task_name']} head (output 2 -> 3) ...")
    model, new_idx = add_head(model)

    # --- 3. full fine-tune on the chosen task's objective only --------------
    print(f"[3] full fine-tune on {cfg['task_name']} objective only (new head idx={new_idx}) ...")
    pos_ft, _, _ = one_hot_regions(peaks[labels[:, col] == 1][["chrom", "start", "end"]], hg38)
    neg_ft, _, _ = one_hot_regions(background.sample(n=min(len(pos_ft), len(background)),
                                                     random_state=SEED + 2), hg38)
    print(f"    {cfg['task_name']} positives={len(pos_ft)}  background negatives={len(neg_ft)}")
    model = finetune_on_new_head(model, new_idx, pos_ft, neg_ft, log)

    # --- 3b. single-task control model trained FROM SCRATCH on the same task -
    # (no shared multitask trunk -> the reference for how much the trunk leaks).
    # Only for the FOS direction; see NEWHEAD note on single_task_control.
    st_model = st_auc = leak_st = None
    if cfg["single_task_control"]:
        print(f"[3b] training single-task {cfg['task_name']} model from scratch ...")
        st_model, st_auc = train_singletask_model(peaks, labels, background, hg38,
                                                  task_idx=col, relative_proportion=BASE_PROP,
                                                  device=DEVICE, random_state=SEED)

    # --- 4. does the OTHER factor's motif leak into the new head? -----------
    print(f"[4] marginalizing {cfg['leak_name']} motif across heads ...")
    leak_own  = motif_effect(model, bg_marg, cfg["leak_motif"], task_idx=cfg["own_task_idx"])  # real task
    leak_orig = motif_effect(model, bg_marg, cfg["leak_motif"], task_idx=cfg["orig_idx"])      # orig head
    leak_new  = motif_effect(model, bg_marg, cfg["leak_motif"], task_idx=new_idx)              # NEW -> leak
    if st_model is not None:
        leak_st = motif_effect(st_model, bg_marg, cfg["leak_motif"], task_idx=0)               # single-task ctrl
    legit_new = motif_effect(model, bg_marg, cfg["legit_motif"], task_idx=new_idx)             # legit signal

    newhead_auc = auc_for_head(model, pos_eval, neg_eval, head_idx=new_idx)

    def stat(x, alt):
        x = np.asarray(x)
        return float(np.median(x)), float(x.mean()), float(wilcoxon(x, alternative=alt).pvalue)

    m_own, _, p_own    = stat(leak_own, "greater")
    m_orig, _, _       = stat(leak_orig, "greater")
    m_new, mean_new, _ = stat(leak_new, "two-sided")
    p_new_ne = min(wilcoxon(leak_new, alternative="greater").pvalue,
                   wilcoxon(leak_new, alternative="less").pvalue) * 2   # 2-sided != 0
    m_ctrl, _, p_ctrl  = stat(legit_new, "greater")

    ln, tn = cfg["leak_name"], cfg["task_name"]
    lines = [
        f"{ln} effect on {cfg['own_task_name']} head (real):        median={m_own:+.3f}  Wilcoxon(>0) p={p_own:.2e}",
        f"{ln} effect on ORIGINAL {cfg['orig_name']} head:          median={m_orig:+.3f}",
        f"{ln} effect on NEW fine-tuned {tn} head:                  median={m_new:+.3f}  mean={mean_new:+.3f}  "
        f"Wilcoxon(!=0) p={p_new_ne:.2e}   <-- {ln} leaks into the new head",
    ]
    if leak_st is not None:
        m_st, mean_st, _ = stat(leak_st, "two-sided")
        p_st_ne = min(wilcoxon(leak_st, alternative="greater").pvalue,
                      wilcoxon(leak_st, alternative="less").pvalue) * 2   # 2-sided != 0
        lines.append(
            f"{ln} effect on SINGLE-TASK {tn} (from scratch):           median={m_st:+.3f}  mean={mean_st:+.3f}  "
            f"Wilcoxon(!=0) p={p_st_ne:.2e}   <-- from-scratch control")
    lines += [
        f"{cfg['legit_name']} effect on NEW {tn} head (legit ctrl): median={m_ctrl:+.3f}  Wilcoxon(>0) p={p_ctrl:.2e}",
        f"NEW {tn} head AUC ({tn} vs background) = {newhead_auc:.3f}   (the head genuinely learned {tn})",
    ]
    if st_auc is not None:
        lines.append(f"SINGLE-TASK {tn} AUC ({tn} vs background) = {st_auc:.3f}")
    (OUT / f"newhead{cfg['suffix']}_finetune_stats.txt").write_text("\n".join(log + [""] + lines) + "\n")
    print("\n".join(lines))

    # --- figure: leak-motif effect across heads (+ single-task) + legit control
    # Labels read "<motif> motif effect on <task> ChIP-seq task (<context>)".
    ld, gd = cfg["leak_disp"], cfg["legit_disp"]
    own, orig = cfg["own_task_name"], cfg["orig_name"]
    data = [leak_own, leak_orig, leak_new]
    xlab = [f"{ld} motif effect on\n{own} ChIP-seq task",
            f"{ld} motif effect on\n{orig} ChIP-seq task\n(original)",
            f"{ld} motif effect on\n{tn} ChIP-seq task\n(after finetuning)"]
    cols = ["#55A868", "#8172B3", "#C44E52"]
    if leak_st is not None:
        data.append(leak_st)
        xlab.append(f"{ld} motif effect on\n{tn} ChIP-seq task\n(single-task, from scratch)")
        cols.append("#DD8452")
    data.append(legit_new)
    xlab.append(f"{gd} motif effect on\n{tn} ChIP-seq task\n(after finetuning)")
    cols.append("#4C72B0")

    fig, ax = plt.subplots(figsize=(1.95 * len(data) + 0.5, 5.4))
    parts = ax.violinplot(data, positions=range(len(data)), widths=0.72, showextrema=False)
    for pc, c in zip(parts["bodies"], cols):
        pc.set_facecolor(c); pc.set_alpha(0.80)
        pc.set_edgecolor("0.25"); pc.set_linewidth(0.8)
    for i, d in enumerate(data):
        p2, q1, m, q3, p98 = np.percentile(d, [2.5, 25, 50, 75, 97.5])
        ax.vlines(i, p2, p98, color="0.2", lw=1.2, zorder=3)          # whisker
        ax.vlines(i, q1, q3, color="0.2", lw=6, zorder=4)             # IQR box
        ax.plot(i, m, "o", color="white", ms=6, mec="0.2", mew=0.9, zorder=5)  # median
        ytxt, va = (p98, "bottom") if m >= 0 else (p2, "top")         # value by the tail
        ax.text(i + 0.20, ytxt, f"{m:+.2f}", ha="left", va=va, fontsize=11)
    ax.axhline(0, ls="--", color="0.4", lw=1, zorder=1)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(xlab, fontsize=10)
    ax.set_xlim(-0.65, len(data) - 1 + 0.85)
    ax.set_ylabel("Motif marginalization effect (logits)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = f"newhead{cfg['suffix']}_finetune_effect"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("done ->", OUT / f"{stem}.pdf")

    # --- example attribution logos ------------------------------------------
    print("[5] example ISM attribution logos ...")
    # The leak row and the single-task control row share one y-axis (group
    # "leak_ctrl") so they are directly comparable at the scale of the larger.
    panels = [
        (f"{cfg['leak_name']} motif -> {cfg['own_task_name']} head  (real task, strong)",
         model, cfg["leak_motif"], cfg["own_task_idx"]),
        (f"{cfg['leak_name']} motif -> NEW {tn} head  (leak: head trained only on {tn})",
         model, cfg["leak_motif"], new_idx, "leak_ctrl"),
    ]
    if st_model is not None:
        panels.append(
            (f"{cfg['leak_name']} motif -> single-task {tn} from scratch  (control)",
             st_model, cfg["leak_motif"], 0, "leak_ctrl"))
    panels.append(
        (f"{cfg['legit_name']} motif -> NEW {tn} head  (legitimate {tn} signal, control)",
         model, cfg["legit_motif"], new_idx))
    attribution_logos(panels, bg_marg, cfg)


if __name__ == "__main__":
    main()
