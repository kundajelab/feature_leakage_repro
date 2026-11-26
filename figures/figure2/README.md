# Figure 2 — OCT–SOX insertion: multi-task models vs a matched single-task model

All models read the same ENCODE DNase-seq experiment, ENCSR000EPO (fibroblast of dermis):

| model | output | input |
|---|---|---|
| AlphaGenome (PyTorch port, all-folds v1) | DNase track CL:0002551 | 524,288 bp (16,384 bp for panels c, d) |
| Enformer (gReLU `human`) | DNase task ENCFF302JEV | 196,608 bp |
| Borzoi (gReLU `human_fold0`) | DNase task ENCFF302JEV | 524,288 bp |
| ChromBPNet (bias-corrected, fold 0) | count head (total counts over 1,000 bp) | 2,114 bp |

## Method (Fig. 2b, Supplementary Fig. 2a)

1. **Enhancers:** the 120 highest-signal ENCSR000EPO DNase peaks (ENCFF184KAM) on
   chr1–22, centred on the peak midpoint (`enhancers.py`). All models use the same 120.
2. **Insertion:** four copies of the OCT–SOX consensus `ATTGTTATGCAAAT` written on the
   forward strand at −150, +150, −200 and +200 bp from the centre.
3. **Effect:** log2(inserted / original) of predicted accessibility. AlphaGenome, Enformer
   and Borzoi: signal summed over the central 1,024 bp (pseudocount 1e-3). ChromBPNet:
   (count head inserted − original) / ln 2, over its 1,000 bp output.
4. **Tests:** two-sided Wilcoxon signed-rank test against zero (n = 120), and a paired
   two-sided Wilcoxon test of each multi-task model against ChromBPNet.

## Panels

| Panel | Script → output |
|---|---|
| a | schematic |
| b | `run_raw_alphagenome.py`, `run_raw_chrombpnet.py` → `plot_raw_insertion.py` → `panelb_raw_insertion.pdf` |
| c, d | `run_tracks_dermis.py --model {alphagenome,chrombpnet}` → `run_tracks_matched.py` |
| Supp a | `run_raw_grelu.py --model {enformer,borzoi}` → `plot_raw_insertion.py` → `supplement_raw_insertion.pdf` |
| Supp b–d | `../figure1/run_volcano_*.py` (see the Figure 1 README) |

## Run

```bash
# Fig 2b / Supp 2a
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_raw_alphagenome.py   # featleak-agtorch
CUDA_VISIBLE_DEVICES=0 python run_raw_grelu.py --model enformer      # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_raw_grelu.py --model borzoi        # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_raw_chrombpnet.py                  # featleak-dl
python plot_raw_insertion.py                                         # featleak-dl; also writes raw_insertion_summary.csv

# Fig 2c, d: one enhancer with 0-4 OCT-SOX insertions
LOCUS=chr8:104659580:104659820
TAG=chr8_104659580_16k
python run_tracks_dermis.py --model alphagenome --locus $LOCUS --seq-len 16384 --tag $TAG  # featleak-agtorch
python run_tracks_dermis.py --model chrombpnet  --locus $LOCUS --tag $TAG                 # featleak-dl
python run_tracks_matched.py --cbp-tag $TAG --ag-tag $TAG --suffix _dermis16k_chr8
```

In panels c and d, OCT–SOX is inserted cumulatively at −150, +150, −200 and +200 bp;
AlphaGenome shows the CL:0002551 profile and ChromBPNet the base-resolution expected counts
(profile probabilities × predicted total counts), over ±500 bp around the enhancer centre.
AlphaGenome reads a 16,384 bp window here, so that both models receive a comparable amount
of sequence context around the enhancer.
