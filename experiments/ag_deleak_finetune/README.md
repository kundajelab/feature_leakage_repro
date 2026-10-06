# Figure 7 — ChromBPNet-gated fine-tuning of AlphaGenome

Fine-tunes AlphaGenome's dermal-fibroblast DNase track so that it stops responding to
OCT–SOX insertions where the matched ChromBPNet predicts no change, with the backbone and
all other outputs frozen.

Everything reads one ENCODE experiment, **ENCSR000EPO** (fibroblast of dermis):
AlphaGenome track CL:0002551 (DNase channel 60; PyTorch port, all-folds v1 weights, full
524,288 bp input), ChromBPNet `chrombpnet_dermis` (fold 0, trained on ENCSR000EPO),
enhancers from its peaks (ENCFF184KAM), observed signal ENCFF302JEV.

## Method (`finetune_dermis_torch.py`)

- **Trainable**: this track's slice of the 1 bp DNase head — 1,536 weights + bias +
  residual scale = 1,538 parameters. Trunk embeddings are cached once per sequence
  (float16), so training is heads-only at full context.
- **Data**: 300 highest-signal peaks outside chr8/9 (train); 120 highest-signal peaks on
  chr8 (68) and chr9 (52) (evaluation). Peaks with non-ACGT bases within ±1 kb skipped.
- **Edit**: four OCT–SOX copies at −150, +150, −200, +200 bp (forward strand).
- **Gate**: a pair is penalised only if |ChromBPNet log2FC| < 0.2 (count head; 111/300 = 37%).
- **Loss**: L = L_fid + L_gate. L_fid = mean squared difference of log1p predictions vs the
  frozen model over the central 300 bp of the unedited enhancer; L_gate = Σ_gated Δ_AG² /
  (n_gated + 1), with Δ_AG = log2 ratio of the central-300 bp sum (pseudocount 1e-3).
- **Optimisation**: Adam, lr 2e-3, 300 steps × 8 enhancers (sampled with replacement),
  seed 0, final weights.
- **Evaluation**: one OCT–SOX copy at −100 bp (never used in training); one-sided Wilcoxon
  vs zero for the original head, paired two-sided Wilcoxon before vs after. Fidelity =
  Pearson r vs ENCFF302JEV over 16,384 bp per enhancer, averaged over 120 windows (chr8/9
  are held out from fine-tuning only, not from the all-folds checkpoint).
- **Endogenous sites (panel c, `endo_screen.py`, `endo_ism.py`)**: all 493 Figure 1 FIMO
  OCT–SOX hits within 500 bp of a dermis summit on chr8/9. Each 14 bp site is
  dinucleotide-shuffled 5 times (Fig 1's ablation); effect = log2(A_intact / mean
  A_shuffled), A = predicted signal summed over ±500 bp around the motif, for the original
  and fine-tuned head. The panel shows ISM (motif ±3 bp; log2 A at the reference base minus
  the mean over the four bases; minus-strand sites drawn in motif orientation) at
  chr9:92,379,377, one of the most repressive sites in the screen (ablation −0.99 log2,
  all 5 shuffles ≤ −0.70; after fine-tuning +0.32). Across all sites the median ablation
  effect is ≈0, so this is a selected example, not the typical site.

Results (`outputs/dermis_torch/eval.log`): insertion effect median −0.29 → −0.03
(paired P = 3.7e-21; ChromBPNet −0.01); fidelity r 0.619 → 0.609.

## Reproduce

```bash
# env featleak-agtorch unless noted; outputs in outputs/dermis_torch/
python finetune_dermis_torch.py --stage cache    # trunk embeddings (GPU)
python finetune_dermis_torch.py --stage gate     # ChromBPNet gate (env featleak-dl)
python finetune_dermis_torch.py --stage train    # fine-tune 1,538 parameters
python finetune_dermis_torch.py --stage obs      # observed bigWig windows (env featleak-dl)
python finetune_dermis_torch.py --stage eval     # panels b, d
python plot_fig7_dermis.py                       # fig7_b_fidelity / fig7_d_octsox_box

# panel c: endogenous OCT-SOX screen + ISM (outputs in outputs/dermis_torch/endo_screen/)
python endo_screen.py --shard 0 --nshard 3       # x3 shards, one GPU each (~40 min)
python endo_screen.py --merge                    # screen.csv, ranked by original-head effect
python endo_ism.py chr9:92379377                 # narrow ISM, both heads (GPU, ~10 min)
python plot_fig7c_endo.py --site chr9:92379377   # fig7_c_ism_endo_chr9_92379377.pdf (env featleak-dl)

# alternative panel c: ISM at an OCT-SOX INSERTED into one held-out enhancer
python finetune_dermis_torch.py --stage ism --ism-locus chr9:127021880
python plot_fig7_dermis.py                       # also writes fig7_c_ism.pdf
```

| Panel | Output |
|---|---|
| a | schematic |
| b | `fig7_b_fidelity.pdf` (`panel_b_fidelity.csv`) |
| c | `fig7_c_ism_endo_chr9_92379377.pdf` (`endo_screen/ism_chr9_92379377.npz`, `endo_screen/screen.csv`; endogenous OCT–SOX) |
| c (alt.) | `fig7_c_ism.pdf` (`panel_c_ism.npz`; OCT–SOX inserted at −100 bp) |
| d | `fig7_d_octsox_box.pdf` (`panel_d_heldout.csv`) |

`plot_endo_sites.py`, `plot_endo_track.py` and the `--stage endo*` options of
`finetune_dermis_torch.py` are an earlier single-shuffle, 300 bp-readout version of the
endogenous screen (exploratory; superseded by `endo_screen.py`).

## Older files

`01_cbp_oracle.py`, `02_ag_base_train.py`, `03_finetune.py`, `27_fidelity_bar_prec.py`,
`30_fig6_panels_eg.py` and `DESIGN.md` are an earlier JAX version of this experiment on
the skin-biceps track (NTR:0000521, channel 207). They are kept for reference and are not
used for Figure 7.
