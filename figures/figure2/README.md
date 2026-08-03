# Figure 2 — ChromBPNet (single-task) resists OCT-SOX leakage


## Method (marginalization on real peaks)

1. **Substrate = real fibroblast DNase peaks** (`peaks_fibroblast`, top-N by signal).
2. Insert the **OCT-SOX consensus** (`ATTGTTATGCAAAT`) into the flanks (±150/±200 bp)
   and predict accessibility **before vs after** (panel a schematic).
3. Effect = **shuffle-corrected log2 fold-change** ("specificity"):
   `specificity = log2FC(real OCT-SOX inserted) − log2FC(shuffled OCT-SOX inserted)`.
4. Panels c/d: the predicted **track** at an example locus (chr17:76,671,260–76,672,260),
   before insertion and after 1–4 OCT-SOX insertions.

Each multi-task model is compared to the ChromBPNet trained on the **same ENCODE
experiment** as its fibroblast-DNase readout:

| multi-task model | fibroblast track | matched ChromBPNet |
|---|---|---|
| AlphaGenome | skin-biceps DNase, NTR:0000521 | **ENCSR562ACY** (`chrombpnet_skinbiceps`) |
| Enformer / Borzoi | dermis DNase, ENCFF302JEV | **ENCSR000EPO** (`chrombpnet_dermis`) |

## Panel → script map

| Panel | What | Script → output |
|---|---|---|
| a | marginalization schematic | hand-drawn diagram |
| **b** | OCT-SOX specificity, AlphaGenome vs matched ChromBPNet | `run_boxplot.py` → `panelb_octsox_specificity.pdf` |
| c | AlphaGenome example tracks (chr17) before/after | `run_tracks_alphagenome.py` → `alphagenome_insertion_tracks_*.pdf` |
| d | ChromBPNet example tracks (chr17) before/after | `run_tracks_chrombpnet.py` (+ `run_tracks_screenshot.py`) |
| supp | Enformer/Borzoi vs matched ChromBPNet dermis | `run_boxplot_supplement.py` → `supplement_octsox_specificity.pdf` |

## Reproduce

Panel b (`alphagenome-ft` + `featleak-dl`):
```bash
# 1. per-enhancer OCT-SOX specificity (real vs shuffled), matched models:
conda run -n alphagenome-ft python run_marg_alphagenome.py --n-loci 120              # AG skin-biceps ch207 @2kb -> ag_octsox_specificity.csv
conda run -n featleak-dl    python run_marg_chrombpnet.py --chrombpnet-key chrombpnet_skinbiceps --tag skinbiceps --n-loci 120
# 2. plot
conda run -n featleak-dl python run_boxplot.py            # -> panelb_octsox_specificity.pdf
```

Supplement (`featleak-dl`):
```bash
python run_marg_grelu.py --model enformer --n-loci 120
python run_marg_grelu.py --model borzoi   --n-loci 120
python run_marg_chrombpnet.py --chrombpnet-key chrombpnet_dermis --tag dermis --n-loci 120
python run_boxplot_supplement.py          # -> supplement_octsox_specificity.pdf
```
Panels c/d (example tracks at chr17:76,671,260–76,672,260):
```bash
conda run -n featleak-alphagenome python run_tracks_alphagenome.py --locus chr17:76671700-76671820 --tag chr17
conda run -n featleak-dl          python run_tracks_chrombpnet.py  --locus chr17:76671700-76671820 --tag chr17
conda run -n featleak-dl          python run_tracks_screenshot.py  # assembles the stacked c/d composite
```
