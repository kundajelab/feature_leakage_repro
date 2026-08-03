# Model weights

This directory holds the **pretrained model weights** used by the figures. It is
**git-ignored** (nothing here is committed) because the files are large and/or
license-restricted. Download the weights you need and place them here with the
exact filenames below; `common/config.py` looks in this directory **first** for
every model, so no code changes are required once the files are in place.

You only need the models for the figures you intend to run (see the table at the
end). Override the location with `FEATLEAK_MODELS_ROOT=/somewhere/else` if you
prefer to keep weights outside the repo.

```
models/
  deepsea.beluga.pth                 # Beluga / DeepSEA  (Figs 1E-H, 3, 5C-E)
  alphagenome-v1/                    # AlphaGenome v1 local checkpoint dir (Figs 1-2)
  sei/
    sei.pth                          # SEI weights (optional; not in main Figs 1-4)
    sei.py                           # SEI architecture module (must sit beside sei.pth)
  chrombpnet/
    ENCSR000ENH_chrombpnet_wo_bias.h5   # cardiac fibroblast DNase   (Fig 2 B,D,F)
    ENCSR562ACY_chrombpnet_wo_bias.h5   # skin-biceps (AlphaGenome-matched)
    ENCSR000EPO_chrombpnet_wo_bias.h5   # dermis (Enformer/Borzoi-matched)
    ENCSR000EOR_chrombpnet_wo_bias.h5   # fibroblast (Fig 2 supplement)
    ENCSR128GBN_chrombpnet_wo_bias.h5   # spleen (Fig 2 supplement)
```

## How to obtain each model

| File | Model | Size | Source / how to get it |
|------|-------|------|------------------------|
| `deepsea.beluga.pth` | Beluga (DeepSEA, 2002 tasks) | ~570 MB | Kipoi `DeepSEA/beluga` — `kipoi get-example DeepSEA/beluga` or download `deepsea.beluga.pth` from the Kipoi model zoo (`http://kipoi.org/models/DeepSEA/beluga/`). |
| `alphagenome-v1/` | AlphaGenome v1 (JAX) | ~730 MB (tarball) | repo(`https://github.com/google-deepmind/alphagenome`).
| `sei/sei.pth` (+ `sei.py`) | SEI | ~3.4 GB | Authors' Zenodo release for SEI (`sei-framework`). Keep the `sei.py` architecture file next to the `.pth` (the loader imports it from the same dir). Not required for Figures 1–4. |
| `chrombpnet/<ENCSR>_chrombpnet_wo_bias.h5` | ChromBPNet (bias-factorized, `wo_bias` = bias-corrected) | ~25 MB each | Pretrained ENCODE4 ChromBPNet models — see the dedicated section below. |

### ChromBPNet models (ENCODE4 model zoo)

The **ENCODE portal hosts the DNase *data*** (peaks/signal) for these accessions.

- **Hugging Face** — the kundajelab ENCODE ChromBPNet zoo, one repo per experiment, named
  `kundajelab/encode-chrombpnet-DNASE-<DATA_ENCSR>-<BIAS_ENCSR>`
  (e.g. skin-biceps `ENCSR562ACY`; browse https://huggingface.co/kundajelab).

Each repo has 5 folds; use the **fold_0** bias-corrected model
(`fold_0/model.chrombpnet_nobias.fold_0.*.h5`, i.e. the `wo_bias`/`nobias` file) and save it
here under the filename in the table below. 

| save as | accession | biosample |
|---|---|---|
| `chrombpnet/ENCSR000ENH_chrombpnet_wo_bias.h5` | ENCSR000ENH | cardiac fibroblast |
| `chrombpnet/ENCSR562ACY_chrombpnet_wo_bias.h5` | ENCSR562ACY | fibroblast of skin of left biceps (AlphaGenome-matched) |
| `chrombpnet/ENCSR000EPO_chrombpnet_wo_bias.h5` | ENCSR000EPO | fibroblast of dermis (Enformer/Borzoi-matched) |
| `chrombpnet/ENCSR000EOR_chrombpnet_wo_bias.h5` | ENCSR000EOR | fibroblast |
| `chrombpnet/ENCSR128GBN_chrombpnet_wo_bias.h5` | ENCSR128GBN | spleen |

Borzoi (`human_fold0`) and Enformer (`human`) are **not** stored here — they are
downloaded automatically at runtime by [gReLU](https://github.com/Genentech/gReLU)
the first time `run_grelu.py` / `run_volcano_grelu.py` run.

## Which figures need which models

| Figure | Models required |
|--------|-----------------|
| **1** (A, C, E–H) | Beluga; AlphaGenome (Panel C/A); Borzoi + Enformer via gReLU (auto); ChromBPNet cardiac (log2FC comparison) |
| **2** | ChromBPNet cardiac (`ENCSR000ENH`); AlphaGenome; (supplement: `ENCSR000EOR`, `ENCSR128GBN`, matched skin-biceps/dermis) |
| **3** | Beluga only |
| **4** | none — the FOS/OCT4 CNNs are trained from scratch from ENCODE peak files (see `data/README.md`) |

After placing files, verify everything resolves:

```bash
conda activate featleak-dl
python -m common.config      # every KEY you need should say OK
```
