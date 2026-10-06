# Figure 1 — Multi-task models assign unexpected negative effects to canonical motifs

## Panel a — tracks and ISM logos

```bash
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model borzoi                            # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model enformer                          # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model alphagenome_dermis                # featleak-agtorch (tracks, 524 kb)
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model alphagenome_dermis --context 2048 # featleak-agtorch (ISM, 2 kb)
python plot_panel_a_tracks.py --models alphagenome_dermis,borzoi,enformer \
    --ag-ism panel_a_alphagenome_dermis_ctx2048.npz                                           # featleak-dl
```

Observed (ENCFF302JEV) and predicted dermal-fibroblast DNase at an OCT–SOX locus for each
model, on a common 16,384 bp window. All three models use the ENCSR000EPO output
(AlphaGenome CL:0002551; Borzoi/Enformer ENCFF302JEV). ISM covers the 100 bp around the
motif: each substitution is scored as the log2 fold change of predicted DNase summed over
the central 2 kb, and letter height is minus the mean over the three substitutions. The
AlphaGenome ISM uses a 2,048 bp input.

The same panels can also be produced with the notebooks in [`notebooks/`](notebooks/).

## Panel c — endogenous OCT–SOX ablation, AlphaGenome vs ChromBPNet

```bash
python run_chrombpnet_logfc.py                                                     # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_alphagenome_torch.py --seq-len 2048 --max-loci 400 --out-suffix _2kb   # featleak-agtorch
python run_logfc_styled.py                                                         # featleak-dl
```

OCT–SOX matches (Vierstra archetype AC0636) 200–500 bp from a dermal-fibroblast DNase
summit (ENCFF184KAM); 1,000 are sampled (seed 42) and the first 400 without ambiguous
bases are scored by both models. The central 14 bp of each match are replaced with a
dinucleotide shuffle.
- AlphaGenome: CL:0002551, 2,048 bp input, signal summed over ±500 bp;
  effect = log2[(S_original + 0.001)/(S_ablated + 0.001)].
- ChromBPNet (ENCSR000EPO): count head; effect = (Δ log counts)/ln 2.

Boxplots and one-sided Wilcoxon signed-rank tests are produced by `run_logfc_styled.py`.

## Panels e–h — Beluga, four estimators × four motif-score quartiles

```bash
for m in OCT-SOX AP-1 HNF4 CTCF; do
  python run_beluga_grid.py --motif $m --hit-set genome --n-per-quartile 3000 --tag _big
done
python plot_beluga_grid.py --hit-set genome --file-tag _big --band ci95 --mark-sig
```

JASPAR FIMO matches (MA0142.1, MA0476.1, MA0114.4, MA0139.1) are split into genome-wide
score quartiles and 3,000 matches per quartile are sampled (seed 0). For each of Beluga's
334 DNase tasks, the effect on the pre-sigmoid logit is estimated by:
- DeepLIFT/SHAP (10 dinucleotide-shuffled references, summed over the motif span),
- ISM (reference minus mean substitution, summed over the span),
- ablation (10 shuffles of the span),
- marginalization (motif inserted into 5 dinucleotide shuffles of its own locus).

Example logos are made with `run_example_logos.py` (20 shuffled references):

```bash
python run_example_logos.py --stage final_paired --pick OCT-SOX=chr15:70407064 \
    AP-1=chr1:145974199 HNF4=chr5:169108419 CTCF=chr1:174983404
```

## Supplementary Fig. 1a — ablation across tasks

The AlphaGenome column is produced by `notebooks/supp1a_alphagenome/motif_ablations.ipynb`,
which uses the AlphaGenome API (set `ALPHAGENOME_API_KEY`). Vierstra-archetype FIMO
matches are sampled at 200 per score quartile, the central 14 bp are dinucleotide-shuffled,
and the effect for each DNase track is the intact minus the ablated prediction.

## Supplementary Fig. 2b–d — per-track volcano plots

```bash
python run_volcano_alphagenome.py                 # featleak-alphagenome
python run_volcano_grelu.py --model enformer      # featleak-dl
python run_volcano_grelu.py --model borzoi        # featleak-dl
```

Every DNase track is scored on the 40 strongest cardiac-fibroblast DNase peaks
(ENCFF942AKV) carrying 0–8 cumulative OCT–SOX insertions (±150, ±200, ±250, ±300 bp), or
the same insertions of a shuffled motif. Per track, log2 signal (±500 bp) is regressed on
the number of insertions and the real-minus-shuffled slope is tested with a two-sided
t-test. `volcano_plot.py` colours points by Benjamini–Hochberg q < 0.05; the y-axis shows
unadjusted −log10 P.
