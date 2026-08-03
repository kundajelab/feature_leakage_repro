# Feature Leakage — Figure Reproduction

Reproducible scripts and environments for the main figures (1–6) of
*"Multitask Deep Learning Models for Regulatory Genomics are Sensitive to
Spurious Sequence Features." (Obbad et al, 2026)

The paper shows that multi-task sequence models (DeepSEA/Beluga, Enformer,
Borzoi, AlphaGenome) learn **biologically implausible motif associations**
("feature leakage") — e.g. predicting that OCT-SOX, AP-1, and HNF4 motifs
*repress* chromatin accessibility in cell types where those factors are not
expressed — while a single-task models do not seem to learn these spurious associations.

In this repo we demonstrate examples of feature leakage in contemporary models. We also perform simulations that suggest that "feature leakage" is caused by design decisions (multi-tasking) & training data selection. We also show a proof of concept that suggests that feature leakage can be corrected in some settings. 

Please reach out with questions.

---

## Layout

```
feature_leakage_repro/
  common/              shared, framework-agnostic code
    config.py            central path registry (edit here to re-home data)
    paths.yaml           optional per-machine path overrides
    beluga_model.py      vendored PyTorch Beluga/DeepSEA + datasets/probe
    models.py            model loaders (Beluga, ChromBPNet, AlphaGenome, gReLU, SEI)
    motifs.py            Vierstra motif tables + canonical consensus strings
  env/
    setup_envs.sh        create the conda envs (clone or from-yml)
    *.environment.yml    exported specs for each env
  data/
    README.md            data provenance + ENCODE accessions
  models/
    README.md            how to download each model weight (git-ignored dir)
  figures/
    figure1/ ... figure6/   one directory per figure: run_*.py + README + outputs/
  tools/
    render_with_svg.py   run a figure script, emit an editable .svg beside each .pdf
    pdf_to_png.py        rasterize output PDFs to PNG (preview/thumbnails)
  Makefile               convenience targets (make fig3, make fig5, ...)
```

Every figure script writes its outputs (PDFs + a metrics file) to
`figures/figureN/outputs/`.

---

## Environments

Three **new** conda environments (the setup script never touches existing envs):

| env | stack | used by |
|-----|-------|---------|
| `featleak-dl` | PyTorch 2.7 (cu118), tangermeme 0.5, bpnet-lite, gReLU, modiscolite | Figs 2–6, Beluga/Borzoi/Enformer parts of Fig 1 |
| `featleak-alphagenome` | JAX, local AlphaGenome v1 | AlphaGenome panels of Figs 1–2 |
| `alphagenome-ft` | JAX + [`alphagenome-ft`](https://genomicsxai.github.io/) | AlphaGenome **fine-tuning** (Fig 2b, Fig 6 e–g) |
| `featleak-tf` | TensorFlow/Keras 2.15, tf-modisco | optional TF-Keras variants (Methods text) |

The AlphaGenome **fine-tuning** (Fig 6 e–g, and the 2 kb-context AlphaGenome read in
Fig 2b) uses the `alphagenome-ft` library from
[genomicsxai.github.io](https://genomicsxai.github.io/); env setup and the three
package patches it needs are documented in `alphagenome_finetune/`.

Create them:

```bash
bash env/setup_envs.sh              # clone from the working lab envs (fast, exact)
# or, on a fresh machine:
bash env/setup_envs.sh --from-yml   # solve from environment.yml (portable)
```

Then, for any figure: `conda activate featleak-dl` (or `featleak-alphagenome`).

GPUs: pass `CUDA_VISIBLE_DEVICES=<id>` before any script. The paths in
`common/config.py` assume the Kundaje-lab layout under
`/oak/stanford/groups/akundaje` and `/users/kobbad/feat_leakage`; see
`data/README.md` and override via env vars / `common/paths.yaml` elsewhere.

Sanity-check that all data resolves:
```bash
conda activate featleak-dl
python -m common.config          # prints OK/MISS for every resource
```

---

## Model weights

Download the ones you need into the  `models/` directory; `common/config.py`
looks in `models/` first, so once the files are in place no code changes are needed.
Full filenames/sizes and a per-accession table are in [`models/README.md`](models/README.md).

| model | where to get it | put in |
|-------|-----------------|--------|
| **Beluga / DeepSEA** (~570 MB) | Kipoi model zoo — [`DeepSEA/beluga`](http://kipoi.org/models/DeepSEA/beluga/) (`deepsea.beluga.pth`) | `models/deepsea.beluga.pth` |
| **ChromBPNet** (~25 MB each) | ENCODE4 ChromBPNet model zoo — **Hugging Face** [`kundajelab`](https://huggingface.co/kundajelab) (`encode-chrombpnet-DNASE-<ENCSR>-<bias>`) or **Synapse** [`syn59449898`](https://www.synapse.org/Synapse:syn59449898/files/); use the fold_0 bias-corrected (`nobias`/`wo_bias`) `.h5`. **Not on the ENCODE portal** (that hosts the DNase data, not the trained models). | `models/chrombpnet/<ENCSR>_chrombpnet_wo_bias.h5` |
| **Borzoi / Enformer** | auto-downloaded at runtime by [gReLU](https://github.com/Genentech/gReLU) — nothing to place | (cached by gReLU) |
| **AlphaGenome v1** (~730 MB) | access-restricted (Google DeepMind); obtain the v1 checkpoint dir via the AlphaGenome release | `models/alphagenome-v1/` |
| **SEI** (~3.4 GB, optional; not Figs 1–4) | authors' Zenodo release (`sei-framework`) | `models/sei/sei.pth` (+ `sei.py`) |

Figure 3 needs only Beluga; Figure 4 needs no weights (it trains its CNNs from ENCODE
peaks). After placing files, run `python -m common.config` — the keys you need should say OK.

---

## Reproducing each figure

Run from the figure's directory with the listed env. Times are on one L40S.

| Fig | what it shows | script(s) | env | ~time |
|-----|---------------|-----------|-----|-------|
| **1 A** | OCT-SOX contribution logos + tracks for AlphaGenome/Borzoi/Enformer | `figure1/notebooks/*` (original notebooks, verbatim) | dl | heavy |
| **1 C** | OCT-SOX ablation boxplots | `figure1/run_grelu.py`, `figure1/run_alphagenome.py` | dl / ag | heavy |
| **1 E–H** | Beluga: OCT-SOX/AP-1/HNF4 repress in various cell types | `figure1/run_beluga.py` | dl | ~5 min |
| **2 b** | OCT-SOX specificity: AlphaGenome represses, matched ChromBPNet (skin-biceps) does not | `figure2/run_marg_{alphagenome,chrombpnet}.py` + `run_boxplot.py` | ag / dl | ~min |
| **2 c,d** | example tracks at chr17: AlphaGenome peak shrinks, ChromBPNet stable | `figure2/run_tracks_{alphagenome,chrombpnet}.py` | ag / dl | ~min |
| **2 supp** | Enformer/Borzoi vs matched ChromBPNet (dermis) | `figure2/run_marg_grelu.py` + `run_boxplot_supplement.py` | dl | heavy/min |
| **3** | AP1-counting head on frozen Beluga leaks to many motifs | `figure3/run.py` | dl | ~8 min |
| **4** | Multi-task model more OCT-SOX-sensitive than single-task | `figure4/run_experiment.py` (+ `run_modisco.py`) | dl | ~6 min |
| **4 supp** | adding a new OCT4 head + fine-tuning does *not* remove the leak | `figure4/run_newhead_finetune.py` | dl | ~6 min |
| **5 A,B** | Balanced synthetic simulation: leakage correlates with effect size | `figure5/run_simulation.py` | dl | ~10 min |
| **5 C–E** | Beluga activations entangle AP-1 and CTCF; AP1 probe detects CTCF | `figure5/run_entanglement.py` | dl | ~5 min |
| **6** | Sequence design driven by off-target-repressive motifs | `figure6/run_plots.py` (fast) / `run_design.py` (slow) | dl | s / hrs |

Example:
```bash
conda activate featleak-dl
cd figures/figure3 && CUDA_VISIBLE_DEVICES=0 python run.py
```

See each `figures/figureN/README.md` for panel-by-panel detail and the expected
result.
