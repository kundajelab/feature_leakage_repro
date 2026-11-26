# Figure 5 — Synthetic tasks and representation probes reveal off-target motif responses

Env: `featleak-dl`.

## Notebooks

| notebook | panels |
|---|---|
| `notebooks/simulation_analysis.ipynb` | 5b, Supp 5a |
| `notebooks/feature_entanglement.ipynb` | 5d, 5e, Supp 5b |

## Synthetic multi-task simulation (Fig. 5b, Supp. Fig. 5a)

```bash
python run_simulation.py --seed 1          # 5,000 sequences, the published panel
```

100 random 10 bp motifs, each with an effect size (0.2–2.0) on a random subset of 50
tasks; motifs are inserted into random 2 kb sequences and each task's label is the sum of
the effects of its active motifs. An additive multi-task CNN is trained, and each motif's
effect on its active and inactive tasks is measured by inserting it into 100 random
sequences.

The defaults put the model in the data-limited regime that genomic training sets occupy:
5,000 sequences for 100 motifs across 50 tasks, batch size 32, up to 600 epochs with early
stopping after 46 epochs without a validation improvement, and the best checkpoint restored
before measurement. Three seeds (1, 2, 3) were run; the figure shows seed 1.

Outputs: `leakage_effect_size_correlation.pdf` (5b), `effect_size_correlation.pdf`
(Supp 5a), `simulation_metrics.csv`.

## Beluga activations and AP-1 probe (Fig. 5d, e, Supp. Fig. 5b)

```bash
python run_entanglement.py
```

Changes in the final convolutional layer (`model.0.14`, before ReLU) caused by inserting
AP-1 or CTCF into 100 random sequences are correlated across channel × position values. A
probe trained on these activations to detect AP-1 is evaluated on AP-1 and on CTCF.

Outputs: `ctcf_ap1_correlation.pdf` (5d), `roc_curve_comparison.pdf` (5e),
`validation_auc.pdf` (Supp 5b).

Random sequences are not seeded, so reruns give slightly different numbers.
