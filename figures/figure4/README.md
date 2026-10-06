# Figure 4 — Training-locus selection and multi-task learning induce off-target motif responses

Env: `featleak-dl` (Fig. 4b: `featleak-tf`).

Small CNNs are trained to predict FOS ChIP-seq peaks in IMR-90 fibroblasts (ENCSR124AIG),
alone or together with POU5F1/OCT4 peaks in H1 cells (ENCSR000BMU), and the effect of
inserting an OCT–SOX motif on the FOS prediction is measured. The peak sets and GC-matched
background regions are shipped in `data/figure4/`.

## Notebooks

| notebook | panels |
|---|---|
| `notebooks/analysis_3.ipynb` | 4a (ISM), 4c, Supp 4a–e |
| `notebooks/panel_b_modisco_keras.ipynb` | 4b: TensorFlow/Keras models, gradient × input → TF-MoDISco (env `featleak-tf`) |

Fig. 4c asterisks: two-sided Mann–Whitney U test, single- vs multi-task model at each
background fraction, Benjamini–Hochberg across the four fractions.

## Scripts

```bash
python run_experiment.py          # Fig 4c and Supp 4d, e
python run_modisco.py             # ISM logos (Fig 4a, Supp 4a–c)
python run_modisco.py --modisco   # also runs TF-MoDISco
python plot_auc_bars.py           # Supp 4d, e AUC bars
```

`run_experiment.py` trains the multi-task and union-trained single-task models with
background regions added at 0%, 20%, 40% and 60% of the number of peaks and writes
`single_vs_multi_split_violin.pdf` (4c), `single_task_auc_barplot.pdf` and
`multi_task_auc_barplot.pdf` (Supp 4d, e).

## Supplementary Fig. 4f–h — fine-tuning a new output head

```bash
python run_newhead_finetune.py --new-head fos
```

1. Train the multi-task FOS + OCT4 model on the union of peaks (20% background).
2. Add a new FOS output (final layer 2 → 3 outputs; existing rows copied).
3. Fine-tune end to end on the new output only, with FOS peaks as positives and GC-matched
   background regions as negatives.
4. Insert OCT–SOX or AP-1 into 1,000 background sequences and measure the change in each
   output's logit; a single-task FOS model trained from scratch is the reference.

Outputs: `newhead_fos_finetune_effect.pdf` (4h), `newhead_fos_attribution_logos.pdf` (4g),
`newhead_fos_finetune_stats.txt`.

Background regions are sampled without a fixed seed, so reruns give slightly different
effect sizes.
