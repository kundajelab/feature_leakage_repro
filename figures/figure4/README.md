# Figure 4 — Multi-task > single-task sensitivity from biased locus selection


## `run_experiment.py` — Panel C + supplement (d,e) (env: `featleak-dl`)
Trains CNNs to predict AP1 (FOS) fibroblast ChIP-seq peaks under increasing
proportions of random background regions, and measures the OCT-SOX motif's
effect on the AP1 prediction (which should be ~0 in fibroblast).

```bash
python run_experiment.py
```
Outputs: `single_vs_multi_split_violin.pdf` (C), `single_task_auc_barplot.pdf`
and `multi_task_auc_barplot.pdf` (supp d,e), `fig4_effects.pkl`.



`plot_auc_bars.py` redraws supplement (d, e) from the manuscript run's AUCs
(`--from-pkl` plots a fresh `run_experiment.py` result instead).

## `run_modisco.py` — Panel A/B + supplement ISM logos (env: `featleak-dl`)
Trains the three canonical models (leaky single-task, single-task, multi-task),
computes ISM contribution logos at a joint AP1+OCT4 locus (Panel A / supplement
a–c), and (optionally) runs modiscolite to check which motifs each model learned
(Panel B table: does the AP1 model spuriously learn OCT-SOX?).

```bash
python run_modisco.py            # ISM logos (supp a–c)
python run_modisco.py --modisco  # also runs the slow TF-MoDISco Panel-B step
```
The MoDISco step (`--modisco`) is the slowest; it reports per-model whether AP-1
and OCT-SOX motifs are discovered, reproducing the Panel B checkmark/X table.

## `run_newhead_finetune.py` — supplement: a factor still leaks into a newly fine-tuned head (env: `featleak-dl`)

1. Train the **leaky multitask FOS+OCT4** CNN on the union of peaks.
2. **Add a new task head** for the chosen factor (output layer 2→3; existing
   FOS/OCT4 rows copied).
3. **Full end-to-end fine-tune on that factor's objective only** — positives =
   that factor's peaks, negatives = background. The new head's own training signal
   contains **no information about the other transcription factor** (it never sees the other
   factor's peaks as a labelled contrast), so any sensitivity it acquires to the
   other factor's motif must come from the shared representation.
4. Marginalize the **other factor's motif** and measure its effect on the **new
   head** — a nonzero effect means that motif leaked in. Compared against the leak
   motif on its own real task and the new head's own motif (legitimate signal /
   control).

Both directions are run with `--new-head`:

```bash
python run_newhead_finetune.py --new-head oct4   # does FOS (AP-1) leak into a new OCT4 head?
python run_newhead_finetune.py --new-head fos    # does OCT-SOX leak into a new FOS head?
```

The **`fos`** run also trains a **single-task FOS model from scratch** as a control
and adds it as an extra violin / logo row:
`newhead[_fos]_finetune_effect.pdf` (leak-motif effect across heads + control),
`newhead[_fos]_attribution_logos.pdf` (ISM logos: leak motif on its real task,
leak motif on the new head = leak, the new head's own motif = control),
`newhead[_fos]_finetune_stats.txt`.

**Background note:** the fine-tune negatives come from the curated GC-matched
negatives (`gc_neg_fos` / `gc_neg_oct4`).

## Source notebooks (published numbers)

| notebook | panels | notes |
|---|---|---|
| `notebooks/analysis_3.ipynb` | 4a (ISM), 4c, Supp 4a–e | imports `helpers_3` / `helpers_4` (identical to the copies here). Fig 4c asterisks: two-sided Mann–Whitney U, single- vs multi-task at each background fraction, Benjamini–Hochberg across the four fractions. Supp 4a,b use the multi-task model trained with 0% background; Supp 4a–c contributions are ISM. |
| `notebooks/panel_b_modisco_keras.ipynb` | 4b | the earlier TensorFlow/Keras version of the three models (gradient × input → TF-MoDISco), env `featleak-tf`; imports `notebooks/panel_b_keras_helpers.py`. |

`run_experiment.py`, `run_modisco.py` and `run_newhead_finetune.py` reimplement these with
the same helpers. Background sampling is unseeded, so the OCT–SOX effect of a given
model varies between runs (e.g. the union-trained single-task model at 20% background:
mean −3.9 in the notebook, −0.5 in a script rerun); the multi-task model is more negative
in every run. The "single-task FOS model trained from scratch" in `run_newhead_finetune.py`
(Supp 4h) uses the union-trained recipe (FOS labels on the union of peaks + 20% background).
