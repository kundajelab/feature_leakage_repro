# Figure 5 — Entanglement contributes to feature leakage

## Scripts (env: `featleak-dl`)

### `run_simulation.py` — Panels A, B + supplement (a)
A balanced synthetic setting: 100 random 10 bp motifs, each with a random effect
active in a random subset of 50 tasks; labels are the summed effect of implanted
motifs; an additive multi-task CNN is trained; each motif's effect on its
**inactive** tasks (leakage) is measured by marginalization.

```bash
python run_simulation.py     # defaults: --n-samples 10000 --seed 1 --epochs 150
```
Outputs: `leakage_effect_size_correlation.pdf` (B), `effect_size_correlation.pdf`
(supp a), `simulation_metrics.csv`.


### `run_data_sweep.py` — leakage DOES depend on training-set size (non-monotonic)
Similarily to our figure 3, we find that if you increase the amount of data, feature leakage can dissapear. 
```bash
python run_data_sweep.py --sizes 500 1000 2500 5000 10000 25000 50000
```
Outputs: `data_sweep.csv`, `data_sweep_leakage_vs_size.pdf`.


### `run_entanglement.py` — Panels C, D, E + supplement (b)
Beluga's final conv layer (`model.0.14`) activations for AP-1 vs CTCF insertions
are correlated; an AP-1 probe trained on those activations also detects CTCF.

```bash
python run_entanglement.py
```
Outputs: `ctcf_ap1_correlation.pdf` (D), `roc_curve_comparison.pdf` (E),
`validation_auc.pdf` (supp b).

