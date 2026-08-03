# Data provenance

None of the large data or model-weight files are copied into this repo. Instead
`common/config.py` resolves each resource by key, trying (1) a
`FEATLEAK_<KEY>` environment variable, (2) `common/paths.yaml`, then (3) the
built-in default location on the Kundaje-lab cluster. Run
`python -m common.config` to see what resolves on your machine.

To re-home the data elsewhere, either edit `common/paths.yaml` or set env vars,
e.g. `export FEATLEAK_HG38_FASTA=/my/hg38.fa`.

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
| `assoc_octsox`,`assoc_ap1`,`assoc_hnf4`,`assoc_ctcf` | affinity-scored hit lists | precomputed (jmschr) |

Canonical consensus strings used for insertion/ablation live in
`common/motifs.py`: AP-1 `ATGAGTCAC`, OCT-SOX `ATTGTTATGCAAAT`,
CTCF `GCCGCCAGGGGGCGCCG`, HNF4 `CAAAGTCCA`.

## Peak / signal data
| key | what | accession |
|-----|------|-----------|
| `peaks_fibroblast` | cardiac fibroblast DNase peaks | ENCODE **ENCFF942AKV** |
| `bigwig_fibroblast_ag` | cardiac fibroblast DNase signal | ENCODE **ENCFF424KBD** |
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
