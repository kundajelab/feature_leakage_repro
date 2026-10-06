#!/usr/bin/env python
"""Plot held-out AP1 AUC vs background proportion for single-task and multi-task models (Figure 4 supplement d, e).

Outputs (outputs/): supp_d_single_task_auc[_rerun].{pdf,png} and supp_e_multi_task_auc[_rerun].{pdf,png}.

Run (env: featleak-dl):
    python plot_auc_bars.py
    python plot_auc_bars.py --from-pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:   # Arial-metric font, so text stays editable and matches Arial/Helvetica
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6})
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"

# Manuscript run (fibr_ipsc_peaks_v2/analysis_3.ipynb, cells 7-8).
MANUSCRIPT_AUC = {
    "single": {0.0: 0.9804, 0.2: 0.9714, 0.4: 0.9726, 0.6: 0.9686},
    "multi": {0.0: 0.9805, 0.2: 0.9754, 0.4: 0.9756, 0.6: 0.9757},
}


def bars(auc, title, path):
    keys = sorted(auc)
    fig, ax = plt.subplots(figsize=(2.2, 2.0))
    b = ax.bar([str(k) for k in keys], [auc[k] for k in keys], width=0.6,
               color="#6E97C9", edgecolor="black", linewidth=0.6)
    for bar, k in zip(b, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, auc[k] + 0.015, f"{auc[k]:.4f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Relative Background Size")
    ax.set_ylabel("AUC Score")
    ax.set_title(f"{title} Model Performance\np(AP1 Peak)", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-pkl", action="store_true")
    args = ap.parse_args()
    if args.from_pkl:
        d = pickle.load(open(OUT / "fig4_effects.pkl", "rb"))
        auc = {"single": d["auc_single"], "multi": d["auc_multi"]}
        tag = "_rerun"
    else:
        auc, tag = MANUSCRIPT_AUC, ""
    bars(auc["single"], "Single-task", OUT / f"supp_d_single_task_auc{tag}")
    bars(auc["multi"], "Multi-task", OUT / f"supp_e_multi_task_auc{tag}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
