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
