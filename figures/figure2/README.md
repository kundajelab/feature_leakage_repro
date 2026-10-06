# Figure 2 — OCT–SOX insertion: multi-task models vs a matched single-task model

Every model reads the **same ENCODE DNase-seq experiment, ENCSR000EPO** (fibroblast of
dermis, CL:0002551):

| model | output used | input |
|---|---|---|
| AlphaGenome (PyTorch port, all-folds v1) | DNase track CL:0002551 (channel 60) | 524,288 bp |
| Enformer (gReLU `human`) | DNase task ENCFF302JEV | 196,608 bp |
| Borzoi (gReLU `human_fold0`) | DNase task ENCFF302JEV | 524,288 bp |
| ChromBPNet (bias-corrected, fold 0) | count head (total counts over 1,000 bp) | 2,114 bp |

## Method (raw insertion effect, Fig 2b / Supp 2a)

1. **Enhancers**: the 120 highest-signal ENCSR000EPO DNase peaks (ENCFF184KAM) on
   chr1–22, centred on the peak midpoint (`enhancers.py`). All models use the same 120.
2. **Insertion**: four copies of the OCT–SOX consensus `ATTGTTATGCAAAT` written on the
   forward strand at −150, +150, −200 and +200 bp from the centre, replacing the bases.
3. **Effect** = log2(inserted / original) over a matched ~1 kb window:
   AlphaGenome, Enformer, Borzoi — predicted signal summed over the central 1,024 bp
   (pseudocount 1e-3); ChromBPNet — (count head inserted − original) / ln 2, i.e.
   log2[(counts + 1) ratio] over its 1,000 bp output (the same readout as the Fig 7 gate).
   No shuffled-motif subtraction.
4. **Tests**: two-sided Wilcoxon signed-rank vs zero (n = 120); paired two-sided Wilcoxon
   of each multi-task model vs ChromBPNet on the same enhancers.

Results (`outputs/raw_insertion_summary.csv`): AlphaGenome median −0.56 (P = 6e-21),
Enformer −0.76 (P = 2e-21), Borzoi −0.53 (P = 2e-20), ChromBPNet −0.08 (P = 0.07).

## Panel → script map

| Panel | What | Script → output |
|---|---|---|
| a | insertion schematic | hand-drawn |
| **b** | AlphaGenome vs ChromBPNet, raw insertion effect | `run_raw_alphagenome.py`, `run_raw_chrombpnet.py` → `plot_raw_insertion.py` → `panelb_raw_insertion.pdf` |
| c, d | example tracks before/after 1–4 insertions (ENCSR000EPO models) | `run_tracks_dermis.py --model {alphagenome,chrombpnet}` → `run_tracks_matched.py` |
| supp a | Enformer, Borzoi vs ChromBPNet | `run_raw_grelu.py --model {enformer,borzoi}` → `plot_raw_insertion.py` → `supplement_raw_insertion.pdf` |
| supp (new) | real OCT–SOX vs 53 shuffled insertions (ChromBPNet) | `run_raw_chrombpnet.py --n-shuffles 50` → `supplement_real_vs_shuffles.pdf` |
| supp b–d | per-task volcano plots | `../figure1/run_volcano_*.py` (see Figure 1 README) |

## Reproduce

```bash
# AlphaGenome (env featleak-agtorch, ~25 min on one L40S)
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_raw_alphagenome.py
# Enformer, Borzoi (env featleak-dl, a few minutes each)
CUDA_VISIBLE_DEVICES=0 python run_raw_grelu.py --model enformer
CUDA_VISIBLE_DEVICES=0 python run_raw_grelu.py --model borzoi
# ChromBPNet, real + 53 shuffled insertions (env featleak-dl, ~1 min)
CUDA_VISIBLE_DEVICES=0 python run_raw_chrombpnet.py
# plots + summary table (no GPU)
python plot_raw_insertion.py
```

## Earlier version (shuffle-corrected)

The previous Fig 2b / Supp 2a reported `log2FC(real) − log2FC(shuffled)` with 3 fixed
shuffles, a central 300 bp (AlphaGenome), ±300 bp (Enformer/Borzoi) or ~140 bp profile
window (ChromBPNet): `run_marg_alphagenome_torch.py`, `run_marg_grelu.py`,
`run_marg_chrombpnet.py` → `run_boxplot.py`, `run_boxplot_supplement.py`. Two notes on
that version: AlphaGenome's three shuffles differ from those of the other models
(`random.Random(0)` vs `common.motif_controls`, seeds 100–102), and two of the three
shuffles are among the most disruptive of 53 tested in ChromBPNet, which inflated
ChromBPNet's corrected effect (+0.26 with 3 shuffles, +0.14 with 50). The older chr17
track scripts (`run_tracks_alphagenome.py`, `run_tracks_chrombpnet.py`,
`run_tracks_screenshot.py`) use the skin-biceps AlphaGenome track (NTR:0000521) at 16 kb.
