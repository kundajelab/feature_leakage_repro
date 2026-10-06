#!/usr/bin/env python
"""Compute and plot AlphaGenome REST ChIP log2FC, ISM and profiles after motif insertion in K562 (Supplementary Fig. 2e-g).

Outputs (outputs/): rest_leakage_logfc.csv, rest_leakage_ism.npz, chip_leakage_panel_rest.{pdf,png}.

Run (env: featleak-alphagenome):
    python run_rest_leakage_figure.py --n-bg 12 --n-ism 8
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
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from tangermeme.utils import random_one_hot, characters  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.size": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from run import TF_MOTIFS  # noqa: E402
from run_repressive import chip, central_for_tf, profile_for_tf, SEQ_LEN  # noqa: E402


def shuffled_motifs(motif, k, seed0=100):
    return ["".join(np.random.default_rng(seed0 + j).permutation(list(motif)))
            for j in range(k)]

OUT = Path(__file__).parent / "outputs"
TARGET = "REST"
FL = [-120, 120]                 # two flank insertions
PAD, BASES = 3, "ACGT"
REAL, SHUF = "#C44E52", "#9AA0A8"
LEAKERS = [("NRF1", TF_MOTIFS["NRF1"]), ("YY1", TF_MOTIFS["YY1"]),
           ("E-box (MYC/MAX)", TF_MOTIFS["MYC"]), ("CTCF", TF_MOTIFS["CTCF"])]


def insert(X, motif):
    mid = SEQ_LEN // 2
    cur = X
    for o in FL:
        cur = substitute(cur, motif, start=mid + o - len(motif) // 2)
    return cur


def ism_to_rest(model, base_X, motif, ontology, OT, at_center):
    """ISM the `motif` window (centre for REST control, +120 flank otherwise),
    reading out REST ChIP. Returns mean (4, win) ref-base contribution."""
    mid = SEQ_LEN // 2
    if at_center:
        X = base_X                                   # REST already at centre
        m0 = mid - len(motif) // 2
    else:
        X = insert(base_X, motif)                    # motif at both flanks
        m0 = mid + 120 - len(motif) // 2             # ISM the +120 copy
    w0, w1 = m0 - PAD, m0 + len(motif) + PAD
    contribs = []
    for n in range(X.shape[0]):
        seq = characters(X[n]); win = w1 - w0
        attr = np.zeros((4, win))
        for i in range(win):
            for b in range(4):
                mut = seq[:w0 + i] + BASES[b] + seq[w0 + i + 1:]
                attr[b, i] = central_for_tf(chip(model, mut, ontology, OT), TARGET)
        ref = [BASES.index(seq[w0 + i]) for i in range(win)]
        oh = np.zeros((4, win)); oh[ref, range(win)] = 1.0
        contribs.append((attr - attr.mean(0, keepdims=True)) * oh)
    return np.mean(contribs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=12)
    ap.add_argument("--n-ism", type=int, default=8)
    ap.add_argument("--cell-type", default="K562")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    model = load_alphagenome(); OT = _output_types()
    meta = model.output_metadata().chip_tf
    ontology = meta[meta.biosample_name == args.cell_type].ontology_curie.iloc[0]

    mid = SEQ_LEN // 2
    X = random_one_hot((args.n_bg, 4, SEQ_LEN), random_state=0)
    base = substitute(X, TF_MOTIFS[TARGET], start=mid - len(TF_MOTIFS[TARGET]) // 2)
    s_base = np.array([central_for_tf(chip(model, characters(b), ontology, OT), TARGET)
                       for b in base])
    print(f"REST baseline signal mean={s_base.mean():.0f}", flush=True)

    # ---- panel a: log2FC (real vs shuffled) + panel c profiles ----
    rows, profiles = [], {}
    for name, motif in LEAKERS:
        s_real = np.array([central_for_tf(chip(model, characters(x), ontology, OT), TARGET)
                           for x in insert(base, motif)])
        s_shuf = []
        for sm in shuffled_motifs(motif, 3):
            s_shuf.append([central_for_tf(chip(model, characters(x), ontology, OT), TARGET)
                           for x in insert(base, sm)])
        s_shuf = np.array(s_shuf).mean(0)
        lfc_r = np.log2((s_real + 1e-3) / (s_base + 1e-3))
        lfc_s = np.log2((s_shuf + 1e-3) / (s_base + 1e-3))
        p = wilcoxon(lfc_r, lfc_s).pvalue
        rows.append({"leaker": name, "lfc_real": np.median(lfc_r),
                     "lfc_shuf": np.median(lfc_s), "p": p})
        print(f"  {name}->REST: log2FC real={np.median(lfc_r):+.2f} "
              f"shuf={np.median(lfc_s):+.2f} p={p:.2g}", flush=True)
        if name in ("NRF1", "YY1"):                  # profiles for these two
            xb, pb = profile_for_tf(model, [characters(b) for b in base], ontology, OT, TARGET)
            xr, pr = profile_for_tf(model, [characters(x) for x in insert(base, motif)],
                                    ontology, OT, TARGET)
            sm = shuffled_motifs(motif, 1)[0]
            xs, ps = profile_for_tf(model, [characters(x) for x in insert(base, sm)],
                                    ontology, OT, TARGET)
            profiles[name] = (xb, pb, pr, ps)
    dfa = pd.DataFrame(rows)
    dfa.to_csv(OUT / "rest_leakage_logfc.csv", index=False)

    # ---- panel b: ISM -> REST ----
    Xi = base[:args.n_ism]
    logos = [("REST", ism_to_rest(model, Xi, TF_MOTIFS["REST"], ontology, OT, True),
              "REST motif → REST ChIP  (on-target: ACTIVATES)", "#2E8B57")]
    for name, motif in LEAKERS:
        c = ism_to_rest(model, Xi, motif, ontology, OT, False)
        logos.append((name, c, f"{name} motif → REST ChIP  (leakage: REPRESSES)", REAL))
        print(f"  ISM {name}->REST net={c.sum():+.1f}", flush=True)
    np.savez(OUT / "rest_leakage_ism.npz", **{n: c for n, c, _, _ in logos})

    # ---------- assemble figure ----------
    fig = plt.figure(figsize=(15, 6.6))
    gs = GridSpec(5, 3, figure=fig, width_ratios=[1.05, 1.15, 1.0],
                  wspace=0.32, hspace=0.6)

    ax = fig.add_subplot(gs[:, 0])           # panel a
    d = dfa.sort_values("lfc_real"); y = np.arange(len(d)); h = 0.38
    ax.barh(y + h / 2, d.lfc_real, h, color=REAL, label="real motif")
    ax.barh(y - h / 2, d.lfc_shuf, h, color=SHUF, label="shuffled motif")
    ax.axvline(0, color="0.2", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(d.leaker + " → REST", fontsize=10)
    ax.set_xlabel("REST ChIP  log2 fold-change")
    ax.set_title("Motif specifically represses REST", fontsize=11, pad=8)
    for yi, p in zip(y, d.p):
        ax.text(d.lfc_real.min() * 1.03, yi, "***" if p < 1e-3 else "**" if p < 1e-2 else "*",
                va="center", ha="right", fontsize=10, color=REAL)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    for r, (_, arr, ttl, col) in enumerate(logos):        # panel b
        a = fig.add_subplot(gs[r, 1])
        plot_logo(torch.tensor(arr, dtype=torch.float32), ax=a)
        a.set_title(ttl, fontsize=9, color=col, pad=2)
        a.set_xticks([]); a.set_yticks([])
        for s in ("left", "bottom"):
            a.spines[s].set_visible(False)

    for r, name in enumerate(["NRF1", "YY1"]):            # panel c
        a = fig.add_subplot(gs[r * 2:r * 2 + 2, 2] if r == 0 else gs[3:5, 2])
        xb, pb, pr, ps = profiles[name]
        a.plot(xb, pb, color="#222", lw=1.4, label="REST motif only")
        a.plot(xb, pr, color=REAL, lw=1.4, label=f"+ {name} (real)")
        a.plot(xb, ps, color=SHUF, lw=1.2, ls="--", label=f"+ {name} (shuffled)")
        a.set_title(f"REST ChIP profile — {name} insertion", fontsize=9.5, pad=3)
        a.set_ylabel("pred. REST ChIP", fontsize=8)
        a.legend(fontsize=6.5, loc="upper right")
        if r == 1:
            a.set_xlabel("position relative to centre (bp)", fontsize=8)

    fig.text(0.02, 0.965, "a", fontsize=16, fontweight="bold")
    fig.text(0.365, 0.965, "b", fontsize=16, fontweight="bold")
    fig.text(0.68, 0.965, "c", fontsize=16, fontweight="bold")
    fig.suptitle("Feature leakage in TF ChIP-seq (AlphaGenome, K562): unrelated motifs "
                 "spuriously repress REST binding", fontsize=12.5, y=1.02)
    fig.savefig(OUT / "chip_leakage_panel_rest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "chip_leakage_panel_rest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done ->", OUT / "chip_leakage_panel_rest.pdf")


if __name__ == "__main__":
    main()
