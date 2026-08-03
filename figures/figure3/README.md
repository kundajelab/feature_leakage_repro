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


