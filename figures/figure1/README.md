# Figure 1 — Multi-task models learn implausible motif associations


## Panel A — tracks + ISM logos on a common window (`run_panel_a_tracks.py`)
```bash
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model borzoi        # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model enformer      # featleak-dl
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model alphagenome_dermis                  # featleak-agtorch (tracks, 524 kb)
CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model alphagenome_dermis --context 2048   # featleak-agtorch (ISM, 2 kb)
python plot_panel_a_tracks.py --models alphagenome_dermis,borzoi,enformer \
    --ag-ism panel_a_alphagenome_dermis_ctx2048.npz                                             # featleak-dl
```
Observed (ENCODE bigWig) and predicted fibroblast DNase for each model's example
OCT-SOX locus, drawn on the same 16,384 bp window with an identical scale bar. ISM
covers the 100 bp centred on the motif (log2 fold-change of the central-2 kb
predicted DNase for every substitution; logo height = -mean log2FC of the three
substitutions). The highlighted band is drawn in genomic coordinates and is exactly
the 100 bp shown in the logo. The AlphaGenome logo uses a 2,048 bp input (as in the
original notebook); at this locus the OCT-SOX effect weakens with longer inputs
(motif sum -3.8 at 2 kb, -0.6 at 16 kb, -0.1 at 524 kb). All three models use the
dermal-fibroblast DNase output of ENCSR000EPO (AlphaGenome CL:0002551; Borzoi/Enformer
ENCFF302JEV), and the observed track is ENCFF302JEV. (`--model alphagenome` is an earlier
cardiac-fibroblast CL:0002548 version.) `--context` overrides the model input length;
`screen_ag_octsox_loci.py` screens AlphaGenome loci at full context.

## Panel A — original attribution notebooks
Panel A is reproduced by the **notebooks**, under
[`notebooks/`](notebooks/) (one per model). See `notebooks/README.md`. In short:
```bash
conda activate featleak-dl
export ALPHAGENOME_API_KEY=...   # AlphaGenome notebook only
cd notebooks/borzoi && jupyter nbconvert --to notebook --execute oct_sox_attribs.ipynb --inplace
```
- `notebooks/alphagenome/oct_sox_attribs.ipynb` — `score_ism_variants` +
  `plot_components.SeqLogo` (OCT-SOX motif points down = repressive) + observed
  vs predicted DNase tracks.
- `notebooks/borzoi/oct_sox_attribs.ipynb`, `notebooks/enformer/enformer_attribs.ipynb`
  — `ISM_predict` + `plot_ISM(method="logo")` + profile tracks.

## Panel C — endogenous OCT–SOX ablation, AlphaGenome vs ChromBPNet (ENCSR000EPO)
```bash
python run_chrombpnet_logfc.py                     # featleak-dl: 400 sites + ChromBPNet effects
CUDA_VISIBLE_DEVICES=0 python run_alphagenome_torch.py --seq-len 2048 --max-loci 400 --out-suffix _2kb   # featleak-agtorch
python run_logfc_styled.py                         # featleak-dl: boxplots + one-sided Wilcoxon
```
OCT–SOX matches (Vierstra AC0636) 200–500 bp from an ENCFF184KAM summit; 1,000 sampled
(seed 42), the first 400 N-free sites used for both models; the central 14 bp are
dinucleotide-shuffled once. AlphaGenome: CL:0002551, 2,048 bp input, signal summed over
±500 bp, log2 ratio (pseudocount 1e-3). ChromBPNet: count head, (Δ log counts) / ln 2.
At the full 524,288 bp input (`--seq-len 524288`) the AlphaGenome effect is null
(median +0.000, P = 0.62; `outputs/ag_torch_fig1c_dermis.log`).

## Panel C — earlier ablation boxplots (`run_grelu.py`, `run_alphagenome.py`)
```bash
python run_grelu.py --model borzoi --n-loci 200        # Borzoi / Enformer (featleak-dl)
python run_grelu.py --model enformer --n-loci 200
python run_alphagenome.py --n-loci 200                 # AlphaGenome (featleak-alphagenome)
```
For OCT-SOX hits 200–500 bp from a fibroblast peak summit, the effect is
`signal_with_motif − signal_ablated` (summed over all bins of the fibroblast
DNase track ENCFF302JEV). 

## Panels E–H — Beluga, four methods x four affinity quartiles (`run_beluga_grid.py`)
```bash
for m in OCT-SOX AP-1 HNF4 CTCF; do
  python run_beluga_grid.py --motif $m --hit-set genome --n-per-quartile 3000 --tag _big
done
python plot_beluga_grid.py --hit-set genome --file-tag _big --band ci95 --mark-sig   # final
python plot_beluga_grid.py --hit-set genome --file-tag _big --band none --no-counts  # plain
```
JASPAR FIMO hits (MA0142.1, MA0476.1, MA0114.4, MA0139.1) are split into genome-wide
score quartiles; 3,000 hits per quartile (seed 0). For each of Beluga's 334 DNase tasks the
effect on the pre-sigmoid logit is estimated by DeepLIFT/SHAP (10 dinucleotide-shuffled
references, summed over the motif span), ISM (reference minus mean substitution,
summed over the span), ablation (10 span shuffles) and marginalization (motif inserted
into 5 dinucleotide shuffles of its own locus). DeepLIFT uses tangermeme's rescale /
max-pool rules, evaluated for all tasks at once through the linearised network
(matches `tangermeme.deep_lift_shap` to <1e-4). `--hit-set ccre` restricts hits to
ENCODE cCREs (V3). Per-hit x per-task effects are saved for re-plotting.

Example logos (`run_example_logos.py`): strongest-quartile hits screened for loci whose
motif dominates the local DeepLIFT signal in two cell types (`--stage paired`), then
rendered with locus and cell type (`--stage final_paired --pick OCT-SOX=chr15:70407064
AP-1=chr1:145974199 HNF4=chr5:169108419 CTCF=chr1:174983404`).

## Panels E–H (original ablation-only script: `run_beluga.py`, env `featleak-dl`)
```bash
python run_beluga.py --method ablation --sample-size 400    # or --method marginalization
```
Per-cell-type effect across Beluga's 334 DNase tasks, split into four
FIMO-affinity quartiles, via `tangermeme.ablate.ablate` (the paper's method).

## Supplementary Fig 1a — ablation across tasks

The AlphaGenome column was produced by `notebooks/supp1a_alphagenome/motif_ablations.ipynb`
(with its `utils.py`, `ablate_motifs`): Vierstra-archetype FIMO hits, 200 per score
quartile, one dinucleotide shuffle of the central 14 bp, effect = intact − ablated per
DNase track. The API key in the notebook was replaced with
`os.environ["ALPHAGENOME_API_KEY"]`.

## Supplementary Fig 2b–d — per-task volcano plots

`run_volcano_alphagenome.py` (env `featleak-alphagenome`) and `run_volcano_grelu.py`
(env `featleak-dl`) score all DNase tracks: 30 fibroblast peaks, up to 8 cumulative
OCT–SOX insertions (±150/200/250/300 bp) and shuffled controls; per track, log2 signal is
regressed on insertion number and the real-minus-shuffled slope is tested (two-sided t).
`volcano_plot.py` colours points by Benjamini–Hochberg q < 0.05; the y-axis is raw −log10 P.
