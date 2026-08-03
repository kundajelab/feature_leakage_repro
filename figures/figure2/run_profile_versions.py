#!/usr/bin/env python
"""Matched-window profile figures at chr17, two rendering versions.

Uses the CORRECTED ChromBPNet count inversion exp(logcounts)-1 (natural log;
see run_cbp_raw.py) and the skin-biceps AlphaGenome track.

  version "both_smoothed": AlphaGenome AND ChromBPNet both Gaussian-smoothed
                           (clean accessibility envelopes).
  version "cbp_bp":        ChromBPNet at true base-pair resolution (raw Tn5
                           counts/bp), AlphaGenome smoothed for readability.

Run (any env): python run_profile_versions.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

F2 = Path(__file__).parent / "outputs"
REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "chrombpnet_octsox_stats" / "outputs"
AG_NPZ = F2 / "alphagenome_track_data_skin_biceps.npz"
CBP_NPZ = EXP / "cbp_raw_chr17.npz"
W = 500
LABELS = ["Fibroblast DNase", "1 insertion", "2 insertions", "3 insertions", "4 insertions"]
CBP_YL = "Predicted DNase\n(expected Tn5 counts / bp)"
AG_YL = "Predicted DNase\n(skin-biceps, a.u.)"


def smooth(prof, bp, sigma_bp):
    if sigma_bp <= 0:
        return prof
    d = abs(bp[1] - bp[0])
    return gaussian_filter1d(prof, sigma_bp / d)


def make(version, cbp_sigma, ag_sigma):
    cbp = np.load(CBP_NPZ, allow_pickle=True)
    ag = np.load(AG_NPZ, allow_pickle=True)
    cbp_prof = cbp["profiles_exp"]; cbp_bp = cbp["bp"]; cbp_ins = cbp["ins_bp"]
    ag_prof = ag["profiles"]; ag_bp = ag["bp"]; ag_ins = ag["ins_bp"]

    cols = [("ChromBPNet\n" + CBP_YL, cbp_prof, cbp_bp, cbp_ins, cbp_sigma),
            ("AlphaGenome\n" + AG_YL, ag_prof, ag_bp, ag_ins, ag_sigma)]
    fig, axes = plt.subplots(5, 2, figsize=(11, 6), sharex=True)
    for col, (title, prof, bp, ins, sig) in enumerate(cols):
        m = (bp >= -W) & (bp <= W)
        sm = np.array([smooth(prof[r], bp, sig) for r in range(5)])
        ymax = sm[:, m].max() * 1.2
        for r in range(5):
            axx = axes[r, col]
            axx.plot(bp[m], sm[r][m], color="#441853", lw=1.1)
            axx.set_ylim(0, ymax)
            for o in ins[:r]:
                axx.axvline(o, ls="--", color="#1764ab", lw=0.8)
            if col == 0:
                axx.set_ylabel(LABELS[r], fontsize=8)
        axes[0, col].set_title(title, fontsize=9)
        axes[-1, col].set_xlabel("position relative to enhancer centre (bp)")
        axes[-1, col].set_xlim(-W, W)
    sub = ("both models Gaussian-smoothed" if version == "both_smoothed"
           else "ChromBPNet at base-pair resolution, AlphaGenome smoothed")
    fig.suptitle(f"chr17:76,671,700  OCT-SOX insertion (+/-{W} bp) — {sub}\n"
                 "ChromBPNet counts un-logged as exp(logcounts)-1. Note different y UNITS.",
                 fontsize=10)
    fig.tight_layout()
    out = F2 / f"tracks_matched_{version}.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=110)
    plt.close(fig)
    print("done ->", out)


def main() -> None:
    make("both_smoothed", cbp_sigma=20, ag_sigma=25)
    make("cbp_bp", cbp_sigma=0, ag_sigma=25)


if __name__ == "__main__":
    main()
