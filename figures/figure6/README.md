# Figure 6 — Sequence design exploits off-target motif responses

The AlphaGenome fine-tuning panels that used to be part of this figure are now
Figure 7 (`experiments/ag_deleak_finetune/`).

## Panel → script map

| Panel | What | Script → output |
|-------|------|-----------------|
| a | greedy motif-insertion schematic | hand-drawn diagram (not code-generated) |
| b | optimization progress (melanocyte↑ / keratinocyte↓) | `figure6/run_plots.py` → `sequence_optimization_scores.pdf` |
| c | DeepSHAP of a designed seq (HNF4A+CTCF); on-target + vs off-target − | `figure6/run_panel_c.py` → `designed_sequence_deepshap.pdf` |
| d | per-motif effect forest (mostly repressive) | `figure6/run_plots.py` → `filtered_motif_effects_{offtarget,combined}.pdf` |

## Design panels (a–d) — `figure6/`, env `featleak-dl`

**Warning:** without `outputs/all_results.pkl`, `run_plots.py` falls back to the legacy
`all_results_new.pkl`. That file came from `deepsea_sequence_design_v2.py`, which pushed
**keratinocyte up and melanocyte down** (the reverse of the figure), so do not use it for
the paper. Regenerate the designs with `run_design.py` (melanocyte 1383 up, keratinocyte
1367 down; asserted in the script). The published panels came from an earlier run of
`analysis_2.ipynb` with melanocyte task 1385 (the `fdr0.01` peak track) as the on-target;
that run's results were overwritten.

```bash
python run_plots.py        # panels b + d (fast, no GPU; uses outputs/all_results.pkl if present, else the legacy run)
python run_panel_c.py      # panel c (GPU: Beluga DeepSHAP of a designed sequence)
python run_design.py --n-sequences 50 --y-target 6   # regenerate the designs (SLOW, ~18 min/sequence)
```
- Panel b: on-target (melanocyte) rises while off-target (keratinocyte) drops — matches the manuscript shape.
- Panel c: inserted HNF4A/CTCF motifs get **positive** contribution for the on-target melanocyte task and **negative** for the off-target keratinocyte task.
- Panel d: emitted with **two metrics** — `_offtarget` (effect on keratinocyte only; matches the caption) and `_combined` (Δon+Δoff; matches the rendered x-axis). Most inserted motifs are repressive.
- `custom_greedy.py` is Jacob Schreiber's greedy-substitution engine, vendored unchanged. Direction: on-target melanocyte = Beluga task 1383 (up), off-target keratinocyte = 1367 (down).



## Source notebook

`notebooks/analysis_2.ipynb` produced the published Figure 6 b–d and Supplementary Fig 6
panels, with saved outputs. It loads design results written by
`notebooks/deepsea_sequence_design_v2.py` (`all_results_new.pkl`) and by
`notebooks/sequence_design.py` (`7.14.designs/`), and imports `custom_greedy.py`
(identical to the copy here) and `notebooks/model.py`. The result pickles are in
`/users/kobbad/feat_leakage/sequence_design/`. Check the up/down task indices in each
design script against the panel labels before reusing these results.
