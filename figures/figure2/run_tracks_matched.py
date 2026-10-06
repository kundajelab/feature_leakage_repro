#!/usr/bin/env python
"""Fig 2 C/D as a MATCHED figure: same bp window for ChromBPNet and AlphaGenome.

Loads the profile data saved by run_tracks_chrombpnet.py and
run_tracks_alphagenome.py and plots both on the SAME genomic window (bp), with
each model's y-axis units labelled (they are not comparable in absolute value:
ChromBPNet = expected Tn5 counts/bp; AlphaGenome = its own predicted DNase
signal). ChromBPNet's peak stays; AlphaGenome's shrinks with OCT-SOX.

Run (any env, no GPU): python run_tracks_matched.py --window 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:   # Arial-metric font, so text stays editable and matches Arial/Helvetica
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                            "pdf.fonttype": 42, "ps.fonttype": 42})
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"
LABELS = ["No insertion", "1 OCT-SOX", "2 OCT-SOX", "3 OCT-SOX", "4 OCT-SOX"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=500, help="+/- bp shown")
    ap.add_argument("--cbp-tag", default="default")
    ap.add_argument("--ag-tag", default="skin_biceps")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    args = ap.parse_args()

    cbp = np.load(OUT / f"chrombpnet_track_data_{args.cbp_tag}.npz", allow_pickle=True)
    ag = np.load(OUT / f"alphagenome_track_data_{args.ag_tag}.npz", allow_pickle=True)
    W = args.window

    fig, axes = plt.subplots(5, 2, figsize=(11, 6), sharex=True)
    for col, d in enumerate([cbp, ag]):
        bp = d["bp"]; profs = d["profiles"]; ins_bp = d["ins_bp"]
        m = (bp >= -W) & (bp <= W)
        ymax = profs[:, m].max() * 1.2
        for r in range(5):
            ax = axes[r, col]
            ax.plot(bp[m], profs[r][m], color="#441853", lw=1.2)
            ax.set_ylim(0, ymax)
            for o in ins_bp[:r]:
                ax.axvline(o, ls="--", color="#1764ab", lw=0.8)
            if col == 0:
                ax.set_ylabel(LABELS[r], fontsize=8)
        axes[0, col].set_title(f"{d['model']}\n{str(d['ylabel'])}", fontsize=9)
        axes[-1, col].set_xlabel("position relative to enhancer centre (bp)")
        axes[-1, col].set_xlim(-W, W)
    fig.tight_layout()
    fig.savefig(OUT / f"tracks_matched_window{args.suffix}.pdf")
    fig.savefig(OUT / f"tracks_matched_window{args.suffix}.png", dpi=110)
    plt.close(fig)
    print(f"ChromBPNet bp range: [{cbp['bp'].min():.0f}, {cbp['bp'].max():.0f}], "
          f"AlphaGenome: [{ag['bp'].min():.0f}, {ag['bp'].max():.0f}]")
    print("done ->", OUT / "tracks_matched_window.pdf")


if __name__ == "__main__":
    main()
