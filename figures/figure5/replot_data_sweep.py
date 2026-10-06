#!/usr/bin/env python
"""Re-plot the training-set-size sweep figure from the cached data_sweep.csv.

Outputs (outputs/): data_sweep_leakage_vs_size.pdf.

Run (env: featleak-dl):
    python tools/render_with_svg.py figures/figure5/replot_data_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def main() -> None:
    df = pd.read_csv(OUT / "data_sweep.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(df.n_samples, df.r_leak, "o-", color="#B23A6E")
    a1.axhline(0, ls="--", color="0.6")
    a1.set_xscale("log"); a1.set_xlabel("# training sequences")
    a1.set_ylabel("corr(effect size, leakage)  [r_leak]")
    a1.set_title("Leakage correlation vs training-set size")
    a2.plot(df.n_samples, df.r_pred, "o-", color="#4C9BD4", label="r_pred (active)")
    a2.plot(df.n_samples, df.mean_leak * 50, "s-", color="#B23A6E",
            label="mean leakage ×50")
    a2.axhline(0, ls="--", color="0.6")
    a2.set_xscale("log"); a2.set_xlabel("# training sequences")
    a2.set_title("Fit quality & mean leakage"); a2.legend()
    fig.tight_layout(); fig.savefig(OUT / "data_sweep_leakage_vs_size.pdf")
    plt.close(fig)
    print("done ->", OUT / "data_sweep_leakage_vs_size.pdf")


if __name__ == "__main__":
    main()
