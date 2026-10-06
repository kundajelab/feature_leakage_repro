# Figure 3 — An AP-1 counting head on frozen Beluga responds to other motifs

Env: `featleak-dl`.

`notebooks/train_ap_1_counting.ipynb` produced Figure 3 and Supplementary Fig. 3 (saved
outputs included). `run.py` runs the same analysis from the command line:

```bash
python run.py --epochs 5 --train-size 5000
```

A counting output is added to Beluga's final linear layer and trained, with the rest of
the model frozen, to count AP-1 motifs (0–5 copies) inserted into random 2 kb sequences.
Each of 37 AP-1-dissimilar Vierstra consensus motifs is then inserted into 100 random
backgrounds carrying two AP-1 copies, and its effect on the predicted count is tested with
a two-sided Wilcoxon signed-rank test (Benjamini–Hochberg across motifs).

## Outputs (`outputs/`)

| file | panel |
|---|---|
| `scatter_plot_with_fit.pdf` | Supp 3a: predicted vs inserted count |
| `top_4_motifs_violin_plot.pdf` | 3b: the four motifs with the largest mean effect |
| `logo_{AP1,CTCF,EWSR1-FLI1,NFKB1,ZIC1}.pdf` | 3c–g: DeepLIFT/SHAP logos |
| `motif_significance_analysis.pdf` | Supp 3b, c |
| `motif_effect_results.npz` | per-motif effects |

Random training and background sequences are not seeded, so reruns give slightly
different numbers.
