# Figure 3 — AP1-counting task on frozen Beluga reveals feature leakage

## Script (env: `featleak-dl`)

```bash
python run.py --epochs 5 --train-size 5000
```

Pipeline: load Beluga, add a 2003rd (counting) output, freeze everything except
that output's weights, train it to count AP-1 motifs in random DNA, then measure
how inserting each of ~37 AP-1-dissimilar motifs changes the predicted count
(marginalization leakage).

## Outputs
- `scatter_plot_with_fit.pdf` — supp (a), predicted vs actual count.
- `top_4_motifs_violin_plot.pdf` — Panel B, largest leakage effects.
- `logo_{AP1,CTCF,EWSR1-FLI1,NFKB1,ZIC1}.pdf` — Panels C–G, DeepLIFT logos.
- `motif_significance_analysis.pdf` — supp (b/c), significance pie + FDR hist.
- `motif_effect_results.npz` — per-motif leakage effect arrays.



## Source notebook (published numbers)

`notebooks/train_ap_1_counting.ipynb` is the notebook that produced Figure 3 and
Supplementary Fig 3, with its saved outputs (held-out Pearson r = 0.776; 23 of 37 motifs
significant after Benjamini–Hochberg; top four CTCF(L), EWSR1–FLI1, NFKB1, ZIC1). It imports
`notebooks/model.py`, whose Beluga, dataset and training code matches `common/beluga_model.py`.
Relative paths in the notebook (`../deepSEA_negatives/`, `../ap1_counting/data/`) refer to
the original analysis directory, `/users/kobbad/feat_leakage/` (Beluga weights and the
Vierstra motif tables; see `common/config.py`).

`run.py` reimplements the notebook. Training and background sequences are generated
without a fixed seed, so a rerun gives similar but not identical numbers (e.g. r ≈ 0.76,
19–20 of 37 motifs significant).
