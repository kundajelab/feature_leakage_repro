#!/usr/bin/env python
"""Compute and plot the OCT-SOX insertion log2FC boxplot and ISM logos for the JAX fine-tuned DNase head.

Outputs (outputs/): fig6_panel_e_data.npz, fig6_panel_e_octsox_box.{pdf,svg,png},
fig6_panel_g_ism.{pdf,svg,png}.

Run (env: alphagenome-ft):
    python 30_fig6_panels_eg.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "alphagenome_finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
import matplotlib.pyplot as plt
from alphagenome.data import genome
from alphagenome.visualization import plot_components
from common import config
from common.motifs import OCT_SOX
import ft_core

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 2048; C0 = SEQ_LEN // 2; CENTRAL = 150; SINGLE_OFF = -100
ISM_HALF = 16; READ_HALF = 1000; CH = 207; BASES = "ACGT"; BATCH = 32; EPS = 1e-3
LOCUS = ("chr8", 40414037)
C_HEAD = OUT / "ft_dnase_C_lam1.0_lr0.002_s300_cmp.pkl"
PRE_COL, FT_COL = "#C44E52", "#8E6FB0"


def load_ft_head(params, leaves):
    def repl(path, leaf):
        ps = "/".join(str(getattr(k, "key", k)) for k in path)
        return jnp.asarray(leaves[ps]) if ps in leaves else leaf
    return jax.tree_util.tree_map_with_path(repl, params)


def prof207(fwd, p, state, seqs):
    out = []
    for i in range(0, len(seqs), BATCH):
        c = seqs[i:i + BATCH]
        _, full, _ = fwd(p, state, jnp.asarray(ft_core.encode_batch(c)), jnp.zeros((len(c),), jnp.int32))
        out.append(np.asarray(full, np.float32)[..., CH])
    return np.concatenate(out, 0)


def ism_logo(fwd, p, state, seq):
    w0, w1 = C0 - ISM_HALF, C0 + ISM_HALF; win = w1 - w0
    muts = [seq[:w0 + i] + b + seq[w0 + i + 1:] for i in range(win) for b in BASES]
    vals = prof207(fwd, p, state, muts)[:, C0 - READ_HALF:C0 + READ_HALF].mean(1).reshape(win, 4).T
    ref = [BASES.index(seq[w0 + i]) for i in range(win)]
    oh = np.zeros((4, win)); oh[ref, range(win)] = 1.0
    return (vals - vals.mean(0, keepdims=True)) * oh          # (4,win)


def main():
    import pyfaidx
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    model = ft_core.load_model(); fwd = ft_core.make_dnase_forward(ft_core.get_apply_fn(model))
    base_params, state = model._params, model._base_model._state
    with open(C_HEAD, "rb") as fh:
        c_params = load_ft_head(base_params, pickle.load(fh)["dnase_leaves"])
    variants = {"pre": base_params, "ft": c_params}          # keys; display labels below
    LABELS = {"pre": "AlphaGenome", "ft": "AlphaGenome\nFine-tune"}

    # ---- Panel e: single-OCT-SOX log2FC over held-out enhancers -------------
    orc = pd.read_csv(OUT / "cbp_oracle_test.csv")
    bseq, iseq = [], []
    for _, r in orc.iterrows():
        s = (int(r.start) + int(r.end)) // 2 - SEQ_LEN // 2
        seq = str(fa[r.chrom][s:s + SEQ_LEN]).upper()
        if len(seq) != SEQ_LEN or (set(seq) - set("ACGT")):
            continue
        lst = list(seq); lst[C0 + SINGLE_OFF:C0 + SINGLE_OFF + len(OCT_SOX)] = list(OCT_SOX)
        bseq.append(seq); iseq.append("".join(lst))
    lfc = {}
    for name, p in variants.items():
        s0 = prof207(fwd, p, state, bseq)[:, C0 - CENTRAL:C0 + CENTRAL].sum(1)
        s1 = prof207(fwd, p, state, iseq)[:, C0 - CENTRAL:C0 + CENTRAL].sum(1)
        lfc[name] = np.log2((s1 + EPS) / (s0 + EPS))
    np.savez(OUT / "fig6_panel_e_data.npz", pre=lfc["pre"], ft=lfc["ft"])
    pre_med, ft_med = np.median(lfc["pre"]), np.median(lfc["ft"])
    print(f"panel e: n={len(bseq)} enhancers  pre median={pre_med:+.3f}  FT median={ft_med:+.3f}")

    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    bp = ax.boxplot([lfc["pre"], lfc["ft"]], widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch, col in zip(bp["boxes"], [PRE_COL, FT_COL]):
        patch.set_facecolor(col); patch.set_alpha(0.85)
    ax.axhline(0, ls=":", color="0.4", lw=1)
    ax.set_xticklabels([LABELS["pre"], LABELS["ft"]], fontsize=9)
    ax.set_ylabel("LogFC after OCT-SOX insertion")
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig6_panel_e_octsox_box.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)

    # ---- Panel g: ISM logo at the endogenous OCT-SOX locus ------------------
    chrom, mid = LOCUS
    s = mid - SEQ_LEN // 2
    seq = str(fa[chrom][s:s + SEQ_LEN]).upper()
    interval = genome.Interval(chrom, mid - ISM_HALF, mid + ISM_HALF)
    comps = []
    for name, p in variants.items():
        c = ism_logo(fwd, p, state, seq)
        print(f"panel g: {LABELS[name].replace(chr(10), ' ')} net ISM = {c.sum():+.3f}", flush=True)
        comps.append(plot_components.SeqLogo(scores=c.T.astype(np.float32),
                                             scores_interval=interval,
                                             ylabel=LABELS[name].replace("\n", " ")))
    fig = plot_components.plot(comps, interval=interval, fig_width=9,
                              title=f"OCT-SOX ISM at {chrom}:{mid:,} (held-out): AlphaGenome vs fine-tuned")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig6_panel_g_ism.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    print("saved -> fig6_panel_e_octsox_box.* and fig6_panel_g_ism.*")


if __name__ == "__main__":
    main()
