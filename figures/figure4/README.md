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

## `run_newhead_finetune.py` — supplement: FOS leaks into a new fine-tuned OCT4 head (env: `featleak-dl`)

1. Train the **leaky multitask FOS+OCT4** CNN on the union of peaks.
2. **Add a new OCT4-prediction head** (output layer 2→3; FOS/OCT4 rows copied).
3. **Full end-to-end fine-tune on the OCT4 objective only** — positives = OCT4
   peaks, negatives = background. The new head's own training signal contains **no
   FOS information** (it never sees FOS peaks as a labelled contrast), so any FOS
   sensitivity it acquires must come from the shared representation.
4. Marginalize the **FOS (AP-1) motif** and measure its effect on the **new OCT4
   head** — a nonzero effect means FOS leaked in. Compared against AP-1 on the FOS
   head (real task) and OCT-SOX on the new head (legitimate OCT4 signal / control).

```bash
python run_newhead_finetune.py
```
Outputs: `newhead_finetune_effect.pdf` (AP-1 effect across heads + OCT-SOX
control), `newhead_attribution_logos.pdf` (ISM logos: AP-1 on FOS head, AP-1 on
new OCT4 head = leak, OCT-SOX on new head = control), `newhead_finetune_stats.txt`.

