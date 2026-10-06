#!/usr/bin/env python
"""Re-plot the OCT-SOX volcanoes from the cached volcano CSVs.

Outputs (outputs/): volcano_<model>.pdf and volcano_<model>_specificity.pdf.

Run (env: featleak-dl):
    python run_volcano_replot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd  # noqa: E402
from volcano_plot import volcano  # noqa: E402

OUT = Path(__file__).parent / "outputs"
XL_REAL = "Δ predicted accessibility per OCT-SOX insertion\n(fraction of baseline)"
XL_SPEC = "OCT-SOX-specific effect per insertion\n(real − shuffled, fraction of baseline)"

for model in ["alphagenome", "enformer", "borzoi"]:
    name = model.capitalize()
    real = OUT / f"volcano_{model}.csv"
    if real.exists():
        volcano(pd.read_csv(real),
                f"{name}: OCT-SOX effect across all DNase tasks",
                OUT / f"volcano_{model}.pdf", xlabel=XL_REAL)
    spec = OUT / f"volcano_{model}_specificity.csv"
    if spec.exists():
        volcano(pd.read_csv(spec),
                f"{name}: OCT-SOX motif-specific effect (real − shuffled)",
                OUT / f"volcano_{model}_specificity.pdf", xlabel=XL_SPEC)
    print("replotted", model)
