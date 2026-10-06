# Figure 7 — ChromBPNet-gated fine-tuning of AlphaGenome

Fine-tunes AlphaGenome's dermal-fibroblast DNase track so that it stops responding to
OCT–SOX insertions where the matched ChromBPNet model predicts no change, with the trunk
and all other outputs frozen.

All components use one ENCODE experiment, ENCSR000EPO (fibroblast of dermis): AlphaGenome
track CL:0002551 (PyTorch port, all-folds v1 weights, 524,288 bp input), the ChromBPNet
model trained on ENCSR000EPO (`models/chrombpnet/`), enhancers from its peaks
(ENCFF184KAM) and observed signal ENCFF302JEV.

## Method (`finetune_dermis_torch.py`)

- **Trainable parameters:** this track's slice of the 1 bp DNase head (1,536 weights,
  bias and residual scale; 1,538 parameters). Trunk embeddings are computed once per
  sequence and cached, so training is heads-only at full context.
- **Data:** the 300 highest-signal peaks outside chr8/9 (training) and the 120
  highest-signal peaks on chr8 and chr9 (evaluation).
- **Edit:** four OCT–SOX copies at −150, +150, −200 and +200 bp.
- **Insertion effect (both models):** Δ = log2[A(edited)/A(original)]. AlphaGenome: A =
  predicted signal summed over the central 300 bp. ChromBPNet: A = predicted total counts
  over its 1,000 bp output (count head; Δ = difference in log counts / ln 2).
- **Gate:** a pair is penalised only if |Δ_ChromBPNet| < 0.2.
- **Loss:** L = L_fid + L_gate. L_fid = mean squared difference of log1p predictions
  versus the original head over the central 300 bp of unedited enhancers; L_gate =
  Σ_gated Δ_AG² / (n_gated + 1).
- **Optimisation:** Adam, learning rate 2e-3, 300 steps of 8 enhancers, seed 0.
- **Evaluation:** one OCT–SOX copy at −100 bp (not used in training); Pearson r against
  ENCFF302JEV over 16,384 bp around each held-out enhancer.

The fine-tuned head is shipped as `data/figure7/ft_head_dermis.pt` and is used whenever
`outputs/dermis_torch/ft_head_dermis.pt` has not been generated.

## Panel c — endogenous OCT–SOX site

`endo_screen.py` scores the FIMO OCT–SOX matches (AC0636) on chr8/9 within 500 bp of a
dermal-fibroblast DNase summit: the central 14 bp of each match are dinucleotide-shuffled
5 times, and the effect is log2(A_original / mean A_shuffled) with A summed over ±500 bp,
for the original and fine-tuned heads. `endo_ism.py` computes ISM over the motif ±3 bp at
a selected site (log2 A at the reference base minus the mean over the four bases).

## Run

```bash
# env featleak-agtorch unless noted; outputs in outputs/dermis_torch/
python finetune_dermis_torch.py --stage cache    # trunk embeddings (GPU)
python finetune_dermis_torch.py --stage gate     # ChromBPNet gate (env featleak-dl)
python finetune_dermis_torch.py --stage train    # fine-tune the 1,538 parameters
python finetune_dermis_torch.py --stage obs      # observed bigWig windows (env featleak-dl)
python finetune_dermis_torch.py --stage eval     # panels b, d
python plot_fig7_dermis.py                       # fig7_b_fidelity.pdf, fig7_d_octsox_box.pdf (env featleak-dl)

# panel c
python endo_screen.py --shard 0 --nshard 3       # run shards 0-2 in parallel, one GPU each
python endo_screen.py --merge                    # endo_screen/screen.csv
python endo_ism.py chr9:92379377                 # ISM, original and fine-tuned head (GPU)
python plot_fig7c_endo.py --site chr9:92379377   # fig7_c_ism_endo_chr9_92379377.pdf (env featleak-dl)
```

| Panel | Output |
|---|---|
| a | schematic |
| b | `fig7_b_fidelity.pdf` (`panel_b_fidelity.csv`) |
| c | `fig7_c_ism_endo_chr9_92379377.pdf` (`endo_screen/ism_chr9_92379377.npz`) |
| d | `fig7_d_octsox_box.pdf` (`panel_d_heldout.csv`) |
