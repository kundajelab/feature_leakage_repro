# Figure 6 — Sequence design exploits off-target motif responses

Env: `featleak-dl`.

Greedy motif insertion with Beluga to design sequences predicted to be accessible in
melanocytes (on-target, task 1383) but not in keratinocytes (off-target, task 1367).
Starting from random 2 kb sequences, each step tries every Vierstra archetype consensus
motif at every position in the central 200 bp and keeps the substitution that brings the
on-target minus off-target logit closest to the target (6). `custom_greedy.py` implements
the greedy substitution.

```bash
CUDA_VISIBLE_DEVICES=0 python run_design.py --n-sequences 500 --y-target 6   # designs (slow; GPU)
python run_plots.py --results outputs/all_results.pkl                        # panels b, d; Supp 6
CUDA_VISIBLE_DEVICES=0 python run_panel_c.py --results outputs/all_results.pkl   # panel c (DeepSHAP)
```

| Panel | Output |
|---|---|
| a | schematic |
| b | `sequence_optimization_scores.pdf`: on- and off-target logits across steps |
| c | `designed_sequence_deepshap.pdf`: DeepSHAP for a designed sequence, both tasks |
| d | `filtered_motif_effects_{offtarget,combined}.pdf`: per-motif effect (mean ± s.d.) |

Panel d is written with two metrics: the change in the off-target prediction, and the
combined score change (Δ on-target + Δ off-target).

`notebooks/analysis_2.ipynb` is the analysis notebook for this figure.
