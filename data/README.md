# Data provenance

None of the large data or model-weight files are copied into this repo. Instead
`common/config.py` resolves each resource by key, trying (1) a
`FEATLEAK_<KEY>` environment variable, (2) `common/paths.yaml`, then (3) the
built-in default location on the Kundaje-lab cluster. Run
`python -m common.config` to see what resolves on your machine.

To re-home the data elsewhere, either edit `common/paths.yaml` or set env vars,
e.g. `export FEATLEAK_HG38_FASTA=/my/hg38.fa`.

## Files shipped in this repo

Small inputs that are not available elsewhere in the form used here are committed under
`data/` (and one model under `models/`); `common/config.py` looks for them first.
`data_availability.xlsx` lists every external dataset and model with its accession and link.

| path | what | used in |
|------|------|---------|
| `figure4/ENCSR124AIG_peaks_inliers.bed.gz` | FOS ChIP-seq peaks, IMR-90 (lab-processed from ENCODE ENCSR124AIG) | Fig 4, Supp 4 |
| `figure4/ENCSR000BMU_peaks_inliers.bed.gz` | POU5F1/OCT4 ChIP-seq peaks, H1 (lab-processed from ENCODE ENCSR000BMU) | Fig 4, Supp 4 |
| `figure4/ENCSR124AIG_gc_neg_only.bed.gz`, `figure4/ENCSR000BMU_gc_neg_only.bed.gz` | GC-matched background regions for each experiment | Fig 4, Supp 4 |
| `motifs/motifs.tsv`, `motifs/motif_annotations.xlsx` | Vierstra motif clustering v1.0 consensus table and annotations | Figs 3, 5, 6 |
| `beluga/deepsea_beluga_names.txt` | the 2,002 DeepSEA/Beluga task names | Figs 1e–h, 6 |
| `figure7/ft_head_dermis.pt` | fine-tuned AlphaGenome dermis DNase head (1,538 parameters) | Fig 7 |
| `../models/chrombpnet/ENCSR000EPO_chrombpnet_wo_bias.h5` | ChromBPNet, fibroblast of dermis (ENCSR000EPO), fold 0, bias-corrected | Figs 1c, 2, 7 |

## Reference genome
| key | default | source |
|-----|---------|--------|
| `hg38_fasta` | `$AKUNDAJE/jmschr/common/hg38/hg38.fa` | GENCODE GRCh38 |

## Model weights
| key | what | source / accession |
|-----|------|--------------------|
| `beluga_weights` | PyTorch Beluga/DeepSEA, 2002 tasks (598 MB) | Kipoi `DeepSEA/beluga` |
| `alphagenome_weights_dir` | AlphaGenome v1 local weights | `$AKUNDAJE/kobbad/alphagenome/alphagenome-v1` |
| `sei_weights` | SEI model | Zenodo (SEI authors) |
| `chrombpnet_fibroblast` | ChromBPNet DNase, cardiac fibroblast | ENCODE **ENCSR000ENH** (chromatin-atlas-2022) |
| `chrombpnet_encsr000eor` | ChromBPNet DNase (Borzoi/Enformer-matched fibroblast) | ENCODE **ENCSR000EOR** |
| `chrombpnet_encsr128gbn` | ChromBPNet DNase (second tissue / spleen) | ENCODE **ENCSR128GBN** |

Borzoi (`human_fold0`) and Enformer (`human`) are fetched on demand by gReLU
(`grelu.resources.load_model`) and are not registered here.

## Motif resources
| key | what | source |
|-----|------|--------|
| `vierstra_motifs_tsv` | Vierstra archetype consensus table | Vierstra motif compendium |
| `vierstra_motif_annotations_xlsx` | archetype annotations | Vierstra |
| `beluga_task_names` | 2002 Beluga task (cell-type × assay) names | DeepSEA/Beluga metadata |
| `jaspar_meme` | JASPAR2022 vertebrate core (for MoDISco report) | JASPAR |
| `fimo_octsox`,`fimo_ap1`,`fimo_hnf4`,`fimo_ctcf` | genome-wide FIMO hits per motif | FIMO on GRCh38 |
| `assoc_octsox`,`assoc_ap1`,`assoc_hnf4`,`assoc_ctcf` | genome-wide JASPAR motif hits (Fig 1 E-H) | FIMO on GRCh38, see below |
| `encode_ccre_bed` | ENCODE cCREs V3, GRCh38 (optional Fig 1 E-H hit filter) | <https://downloads.wenglab.org/V3/GRCh38-cCREs.bed> |

The `assoc_*` hit lists are FIMO (MEME suite) scans of hg38 with JASPAR MA0142.1
(Pou5f1::Sox2), MA0476.1 (FOS), MA0114.4 (HNF4A) and MA0139.1 (CTCF), default
p < 1e-4, e.g.

```bash
fimo --text --thresh 1e-4 MA0142.1.meme hg38.fa > oct-sox.txt   # likewise ap-1, hnf4, ctcf
```

Canonical consensus strings used for insertion/ablation live in
`common/motifs.py`: AP-1 `ATGAGTCAC`, OCT-SOX `ATTGTTATGCAAAT`,
CTCF `GCCGCCAGGGGGCGCCG`, HNF4 `CAAAGTCCA`.

## Peak / signal data
| key | what | accession |
|-----|------|-----------|
| `peaks_fibroblast` | cardiac fibroblast DNase peaks | ENCODE **ENCFF942AKV** |
| `bigwig_fibroblast_ag` | cardiac fibroblast DNase signal | ENCODE **ENCFF424KBD** |
| `bigwig_cardiac_fibroblast_eni` | adult cardiac fibroblast DNase signal (experiment behind AlphaGenome's CL:0002548 DNase track; Fig 1A observed) | ENCODE **ENCFF258OVM** (ENCSR000ENI) |
| `peaks_enformer_fibroblast` | fibroblast peaks (Borzoi/Enformer) | ENCODE **ENCFF184KAM** |
| `bigwig_enformer_fibroblast` | fibroblast signal (Borzoi/Enformer) | ENCODE **ENCFF361BTT** |
| `peaks_spleen` | second-tissue peaks | ENCODE **ENCFF052RRA** |

## Figure 4 ChIP-seq
| key | what | accession |
|-----|------|-----------|
| `chip_fos_fibroblast` | FOS/AP-1 ChIP-seq peaks, fibroblast | ENCODE **ENCSR124AIG** |
| `chip_oct4_hesc` | POU5F1/OCT4 ChIP-seq peaks, hESC | ENCODE **ENCSR000BMU** |
| `gc_neg_fos`, `gc_neg_oct4` | GC-matched background regions | tfatlas processed_data |

All ENCODE accessions are downloadable from <https://www.encodeportal.org>.
