#!/usr/bin/env python
"""Figure 6 (design step): greedy motif insertion with Beluga.

Designs sequences that maximise the DIFFERENTIAL accessibility between an
on-target cell type (melanocyte) and an off-target cell type (keratinocyte),
using greedy insertion of Vierstra archetype motifs. The point of the figure is
that the design is driven largely by motifs that *repress* the off-target cell
type -- a symptom of feature leakage.

Direction note: the original notebooks were internally inconsistent about which
cell type was on/off target. This script fixes the convention to match the
Figure 6 caption: on-target = melanocyte (score should go UP), off-target =
keratinocyte (score should go DOWN). ``score_target_1`` is the on-target
(melanocyte) trajectory and ``score_target_2`` the off-target (keratinocyte)
trajectory, matching the plotting script.

Outputs (outputs/): all_results.pkl (list of per-sequence greedy trajectories,
consumed by run_plots.py) and edited_seqs.pkl (final designed sequences).

Run (env: featleak-dl, ~minutes per sequence):
    python run_design.py --n-sequences 50 --y-target 6
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from tangermeme.predict import predict  # noqa: E402
from tangermeme.utils import random_one_hot  # noqa: E402

from common.models import load_beluga, beluga_task_index  # noqa: E402
from common.motifs import motif_to_seq  # noqa: E402
from custom_greedy import greedy_substitution  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On-target (melanocyte, pushed up) and off-target (keratinocyte, pushed down).
ON_TARGET = "Penis_Foreskin_Melanocyte_Primary_Cells_skin01_DNase.all.peaks_None"
OFF_TARGET = "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02_DNase.all.peaks_None"


class BelugaDifferential(torch.nn.Module):
    """Return (on-target logit, off-target logit) so greedy can push them apart."""

    def __init__(self, beluga, task_up, task_down):
        super().__init__()
        self.model = beluga
        self.task_up = task_up
        self.task_down = task_down

    def forward(self, X):
        y = self.model(X)
        return y[:, self.task_up][:, None], y[:, self.task_down][:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=50)
    ap.add_argument("--y-target", type=float, default=6.0)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    mdl = load_beluga(device=DEVICE)
    task_up = beluga_task_index("Melanocyte")
    task_down = beluga_task_index("Keratinocyte")
    assert task_up == 1383 and task_down == 1367, (task_up, task_down)
    print(f"on-target (melanocyte)={task_up}  off-target (keratinocyte)={task_down}")
    wrapped = BelugaDifferential(mdl, task_up, task_down)

    m2s = motif_to_seq()
    motif_names = list(m2s.keys())
    motifs = [m2s[n] for n in motif_names]
    print(f"{len(motifs)} candidate motifs")

    rand = random_one_hot((args.n_sequences, 4, 2000), random_state=args.seed)
    y_target = torch.full((1, 1), args.y_target, device=DEVICE)

    all_results, edited_seqs = [], []
    for seq in tqdm(rand, total=len(rand)):
        seq = seq.unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, dtype=bool, device=DEVICE)
        edited, added, results, names_added = greedy_substitution(
            wrapped, seq, motifs, y_target, device=DEVICE,
            mask=mask, window=args.window, motif_names=motif_names,
        )
        results["motif_names_added"] = names_added
        all_results.append(results)
        edited_seqs.append(edited.detach().cpu())
        with open(OUT / "all_results.pkl", "wb") as fh:
            pickle.dump(all_results, fh)
    with open(OUT / "edited_seqs.pkl", "wb") as fh:
        pickle.dump(edited_seqs, fh)
    print(f"done: {len(all_results)} designs -> {OUT}")


if __name__ == "__main__":
    main()
