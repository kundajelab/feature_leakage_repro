#!/usr/bin/env python
"""Plot Figure 1A: observed and predicted fibroblast DNase tracks with the OCT-SOX ISM logo per model.

Outputs (outputs/panel_a/plots/): fig1_a_<model>_<window>bp.pdf/.png.

Run (env: featleak-dl):
    python plot_panel_a_tracks.py --models alphagenome_dermis,borzoi,enformer \
        --ag-ism panel_a_alphagenome_dermis_ctx2048.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyBigWig  # noqa: E402
import logomaker  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:   # Arial-metric font, so text stays editable and matches Arial/Helvetica
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

from common import config  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "outputs" / "panel_a"
# key, display name, observed bigWig key, side label, observed label, predicted label
MODELS = [
    # every model reads the same ENCODE experiment, ENCSR000EPO (fibroblast of dermis),
    # and the observed track is ENCFF302JEV, the file Borzoi/Enformer were trained on
    ("alphagenome_dermis", "AlphaGenome", "bigwig_dermis",
     "Dermal fibroblast DNase\n(ENCSR000EPO)", "Observed\nENCFF302JEV",
     "Predicted\nCL:0002551 DNase"),
    ("borzoi", "Borzoi", "bigwig_dermis",
     "Dermal fibroblast DNase\n(ENCSR000EPO)", "Observed\nENCFF302JEV",
     "Predicted\ntarget ENCFF302JEV"),
    ("enformer", "Enformer", "bigwig_dermis",
     "Dermal fibroblast DNase\n(ENCSR000EPO)", "Observed\nENCFF302JEV",
     "Predicted\ntarget ENCFF302JEV"),
]
TRACK_COLOR = "#3b1f5c"
HL_COLOR = "#f4d35e"


def bin_mean(x, w):
    n = len(x) // w
    return x[:n * w].reshape(n, w).mean(1)


def plot_one(key, name, bw_key, track_label, obs_label, pred_label, window,
             disp_bin, npz_file, out, ism_file=None):
    trk = np.load(npz_file)                      # tracks (model's full input context)
    d = np.load(ism_file) if ism_file else trk   # ISM (optionally a shorter context)
    ism_ctx = int(d["context"])
    chrom = str(trk["chrom"]); center = int(trk["center"])
    t0 = int(trk["track_start"])
    w0 = center - window // 2
    sl = slice(w0 - t0, w0 - t0 + window)
    pred = trk["pred"][sl]
    bw = pyBigWig.open(str(config.resolve(bw_key, must_exist=True)))
    obs = np.nan_to_num(np.array(bw.values(chrom, w0, w0 + window), dtype=np.float32))
    obs_b, pred_b = bin_mean(obs, disp_bin), bin_mean(np.nan_to_num(pred), disp_bin)
    x = w0 + np.arange(len(obs_b)) * disp_bin

    ism = d["ism"]; seq = str(d["ref_seq"]); L = len(seq)
    ism_start = int(d["ism_start"])
    contrib = -ism.sum(1) / 3.0                 # ref-base contribution (-mean log2FC)
    M = np.zeros((L, 4)); M[np.arange(L), ["ACGT".index(c) for c in seq]] = contrib
    ms, me = int(d["motif_start"]) - ism_start, int(d["motif_end"]) - ism_start

    fig = plt.figure(figsize=(6.8, 2.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.55, 1.35], hspace=0.3,
                          left=0.25, right=0.98, top=0.86, bottom=0.12)
    axo, axp, axl = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
    for ax, y, lab in ((axo, obs_b, obs_label), (axp, pred_b, pred_label)):
        ax.fill_between(x, 0, y, step="post", color=TRACK_COLOR, lw=0)
        ax.axvspan(ism_start, ism_start + L, color=HL_COLOR, alpha=0.9, lw=0, zorder=0)
        ymax = float(np.nanmax(y)) * 1.05
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, float(f"{ymax / 1.05:.2g}")])
        ax.set_xlim(w0, w0 + window)
        ax.set_xticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=5.5)
    fig.text(0.025, 0.55, name, rotation=90, ha="center", va="center", fontsize=8,
             fontweight="bold")
    fig.text(0.06, 0.55, track_label, rotation=90, ha="center", va="center",
             fontsize=6)

    # identical scale bar in every model's panel
    sb = 2000 if window >= 8000 else 500
    xs = w0 + window * 0.02
    axo.plot([xs, xs + sb], [axo.get_ylim()[1]] * 2, color="k", lw=1, clip_on=False)
    axo.text(xs + sb / 2, axo.get_ylim()[1] * 1.08, f"{sb:,} bp", ha="center",
             va="bottom", fontsize=6)
    axo.set_title(f"{chrom}:{w0 + 1:,}-{w0 + window:,}  ({window:,} bp window)",
                  fontsize=6.5, pad=8)

    logomaker.Logo(pd.DataFrame(M, columns=list("ACGT")), ax=axl,
                   color_scheme="classic")
    lo = axl.get_ylim()[0]
    axl.plot([ms - 0.5, me - 0.5], [lo, lo], color="k", lw=1.5, clip_on=False,
             solid_capstyle="butt")
    axl.axhline(0, color="0.5", lw=0.5)
    axl.set_xlim(-0.5, L - 0.5)
    axl.set_xticks([])
    for s in ("top", "right", "bottom"):
        axl.spines[s].set_visible(False)
    ism_lab = "ISM\ncontribution"
    if ism_ctx != int(trk["context"]):
        ism_lab += f"\n({ism_ctx:,} bp input)"
    axl.set_ylabel(ism_lab, rotation=0, ha="right", va="center", fontsize=6)
    axl.tick_params(labelsize=5.5)
    axl.text(1.0, -0.06, f"{chrom}:{ism_start + 1:,}-{ism_start + L:,} ({L} bp)",
             transform=axl.transAxes, ha="right", va="top", fontsize=5.5, color="0.3")
    axl.text((ms + me - 1) / 2 / (L - 1), -0.06, "OCT-SOX motif",
             transform=axl.transAxes, ha="center", va="top", fontsize=6)

    # zoom lines: highlight edges on the predicted track -> logo corners
    for gx, lx in ((ism_start, -0.5), (ism_start + L, L - 0.5)):
        con = ConnectionPatch(xyA=(gx, 0), coordsA=axp.transData,
                              xyB=(lx, axl.get_ylim()[1]), coordsB=axl.transData,
                              color="0.45", lw=0.5, ls="--")
        fig.add_artist(con)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--disp-bin", type=int, default=16,
                    help="bp per displayed bar (display only)")
    ap.add_argument("--ag-file", default="panel_a_alphagenome_dermis.npz",
                    help="run_panel_a_tracks.py output to use for AlphaGenome")
    ap.add_argument("--ag-ism", default=None,
                    help="optional separate run_panel_a_tracks.py output for the AG ISM")
    ap.add_argument("--models", default="alphagenome_dermis,borzoi,enformer")
    args = ap.parse_args()
    out_dir = DATA / "plots"
    out_dir.mkdir(exist_ok=True)
    for key, name, bw_key, lab, obs_lab, pred_lab in MODELS:
        if key not in args.models.split(","):
            continue
        f = DATA / (args.ag_file if key == "alphagenome_dermis" else f"panel_a_{key}.npz")
        tag = (Path(args.ag_file).stem.replace("panel_a_alphagenome_dermis", "")
               if key == "alphagenome_dermis" else "")
        ism = DATA / args.ag_ism if (key == "alphagenome_dermis" and args.ag_ism) else None
        if ism is not None:
            tag += "_ism" + Path(args.ag_ism).stem.replace("panel_a_alphagenome", "")
        plot_one(key, name, bw_key, lab, obs_lab, pred_lab, args.window, args.disp_bin, f,
                 out_dir / f"fig1_a_{key}{tag}_{args.window}bp", ism_file=ism)
        print("wrote", key)


if __name__ == "__main__":
    main()
