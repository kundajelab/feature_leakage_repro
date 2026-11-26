# Figure 7 — ChromBPNet-gated fine-tuning of AlphaGenome

The code for this figure lives in
[`../../experiments/ag_deleak_finetune/`](../../experiments/ag_deleak_finetune/), which
holds the fine-tuning pipeline and the panel scripts. See that directory's README for the
full method description.

Env: `featleak-agtorch` unless noted; outputs in
`experiments/ag_deleak_finetune/outputs/dermis_torch/`.

## Method

Only AlphaGenome's linear DNase output head is trained — 1,538 parameters — on top of a
frozen trunk (`alphagenome-pytorch` port, all-folds checkpoint, 524,288 bp input). A
matched single-task ChromBPNet (ENCSR000EPO) acts as the reference: for each enhancer and
OCT–SOX insertion, the pair is gated when ChromBPNet's |Δ| is below 0.2. The loss combines
a fidelity term on unedited sequence with a gated term driving AlphaGenome's insertion
effect toward zero. 300 training and 120 test peaks (chromosomes 8 and 9 held out from
fine-tuning), Adam at 2 × 10⁻³, 300 steps of 8 enhancers, seed 0.

## Panels

| Panel | Script → output |
|---|---|
| a | schematic |
| b | `finetune_dermis_torch.py --stage eval` → `plot_fig7_dermis.py` → `fig7_b_fidelity.pdf` |
| c | `endo_screen.py` → `endo_ism.py` → `plot_fig7c_endo.py` → `fig7_c_ism_endo_chr9_92379377.pdf` |
| d | `finetune_dermis_torch.py --stage eval` → `plot_fig7_dermis.py` → `fig7_d_octsox_box.pdf` |

## Run

```bash
cd ../../experiments/ag_deleak_finetune

python finetune_dermis_torch.py --stage cache    # trunk embeddings (GPU)
python finetune_dermis_torch.py --stage gate     # ChromBPNet gate (env featleak-dl)
python finetune_dermis_torch.py --stage train    # fine-tune the 1,538 parameters
python finetune_dermis_torch.py --stage obs      # observed bigWig windows (env featleak-dl)
python finetune_dermis_torch.py --stage eval     # panels b, d
python plot_fig7_dermis.py                       # env featleak-dl

# panel c
python endo_screen.py --shard 0 --nshard 3       # shards 0-2, one GPU each
python endo_screen.py --merge                    # endo_screen/screen.csv
python endo_ism.py chr9:92379377                 # ISM, original and fine-tuned head (GPU)
python plot_fig7c_endo.py --site chr9:92379377   # env featleak-dl
```
