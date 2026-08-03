# Figure 1 — Multi-task models learn implausible motif associations


## Panel A — OCT-SOX contribution logos + tracks (original notebooks)
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

## Panel C — ablation boxplots (`run_grelu.py`, `run_alphagenome.py`)
```bash
python run_grelu.py --model borzoi --n-loci 200        # Borzoi / Enformer (featleak-dl)
python run_grelu.py --model enformer --n-loci 200
python run_alphagenome.py --n-loci 200                 # AlphaGenome (featleak-alphagenome)
```
For OCT-SOX hits 200–500 bp from a fibroblast peak summit, the effect is
`signal_with_motif − signal_ablated` (summed over all bins of the fibroblast
DNase track ENCFF302JEV). A **negative** median (with `*` = Wilcoxon repressive)
is the implausible repression.

## Panels E–H — Beluga, 4 affinity quartiles (`run_beluga.py`, env `featleak-dl`)
```bash
python run_beluga.py --method ablation --sample-size 400    # or --method marginalization
```
Per-cell-type effect across Beluga's 334 DNase tasks, split into four
FIMO-affinity quartiles, via `tangermeme.ablate.ablate` (the paper's method).
