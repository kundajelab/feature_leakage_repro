# Figure 6 — Feature leakage affects sequence design, and is mitigated by fine-tuning

The updated figure spans two directories: the **design** panels (a–d) live here in
`figures/figure6/`; the **AlphaGenome fine-tuning** panels (e–g) are produced by
`experiments/ag_deleak_finetune/` (the de-leaking work).

## Panel → script map

| Panel | What | Script → output |
|-------|------|-----------------|
| a | greedy motif-insertion schematic | hand-drawn diagram (not code-generated) |
| b | optimization progress (melanocyte↑ / keratinocyte↓) | `figure6/run_plots.py` → `sequence_optimization_scores.pdf` |
| c | DeepSHAP of a designed seq (HNF4A+CTCF); on-target + vs off-target − | `figure6/run_panel_c.py` → `designed_sequence_deepshap.pdf` |
| d | per-motif effect forest (mostly repressive) | `figure6/run_plots.py` → `filtered_motif_effects_{offtarget,combined}.pdf` |
| e | OCT-SOX insertion LogFC, AlphaGenome vs Fine-tune | `ag_deleak_finetune/30_fig6_panels_eg.py` → `fig6_panel_e_octsox_box.pdf` |
| f | PearsonR fidelity 0.653 vs 0.654 | `ag_deleak_finetune/27_fidelity_bar_prec.py` → `figure_fidelity_bar_paper.pdf` |
| g | ISM at chr8:40,414,037, before vs after fine-tune | `ag_deleak_finetune/30_fig6_panels_eg.py` → `fig6_panel_g_ism.pdf` |

## Design panels (a–d) — `figure6/`, env `featleak-dl`

```bash
python run_plots.py        # panels b + d (fast, no GPU; uses outputs/all_results.pkl if present, else the legacy run)
python run_panel_c.py      # panel c (GPU: Beluga DeepSHAP of a designed sequence)
python run_design.py --n-sequences 50 --y-target 6   # regenerate the designs (SLOW, ~18 min/sequence)
```
- Panel b: on-target (melanocyte) rises while off-target (keratinocyte) drops — matches the manuscript shape.
- Panel c: inserted HNF4A/CTCF motifs get **positive** contribution for the on-target melanocyte task and **negative** for the off-target keratinocyte task.
- Panel d: emitted with **two metrics** — `_offtarget` (effect on keratinocyte only; matches the caption) and `_combined` (Δon+Δoff; matches the rendered x-axis). Most inserted motifs are repressive.
- `custom_greedy.py` is Jacob Schreiber's greedy-substitution engine, vendored unchanged. Direction: on-target melanocyte = Beluga task 1383 (up), off-target keratinocyte = 1367 (down).


## Fine-tuning panels (e–g) — code in `experiments/ag_deleak_finetune/`

Panels e–g are the **AlphaGenome de-leaking** experiment: fine-tune AlphaGenome's
native fibroblast (skin-biceps, NTR:0000521 / channel 207) DNase head so it stops
spuriously reacting to OCT-SOX, **using ChromBPNet as a gate**, without losing
predictive fidelity. All of this code lives in `experiments/ag_deleak_finetune/`
(env `alphagenome-ft`; needs the AlphaGenome checkpoint); see that folder's
`DESIGN.md` for the method. Only the DNase head leaves are trainable
(~1,538 params); the backbone is frozen. Chromosomes 8/9 are held out.

The AlphaGenome fine-tuning uses the `alphagenome-ft` library from
[genomicsxai.github.io](https://genomicsxai.github.io/) (env setup + the three
required package patches are documented in `alphagenome_finetune/`).

Reproduce the fine-tuning + panels (env `alphagenome-ft`, from `experiments/ag_deleak_finetune/`):
```bash
# 1. build the ChromBPNet oracle gate + AlphaGenome baseline targets (held-out chr8/9):
python 01_cbp_oracle.py --split train      # ChromBPNet OCT-SOX gate signal (train enhancers)
python 01_cbp_oracle.py --split test       # + the held-out chr8/9 eval enhancers (panel e/g)
python 02_ag_base_train.py                 # AlphaGenome base predictions (self-distill fidelity anchors)
# 2. fine-tune the native DNase head with ChromBPNet as a GATE (objective C):
python 03_finetune.py --objective C --lam 1.0 --lr 0.002 --steps 300 --tag cmp
#    -> outputs/ft_dnase_C_lam1.0_lr0.002_s300_cmp.pkl   (the fine-tuned head leaves)
# 3. render the figure-6 panels:
python 30_fig6_panels_eg.py                # panel e (OCT-SOX LogFC box) + panel g (ISM logo)
python 27_fidelity_bar_prec.py             # panel f (PearsonR fidelity barplot)
```

| Panel | code (`experiments/ag_deleak_finetune/`) → output |
|-------|---------------------------------------------------|
| e | `30_fig6_panels_eg.py` → `fig6_panel_e_octsox_box.pdf` |
| f | `27_fidelity_bar_prec.py` → `figure_fidelity_bar_paper.pdf` |
| g | `30_fig6_panels_eg.py` → `fig6_panel_g_ism.pdf` |
