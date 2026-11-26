# Feature leakage in multi-task sequence-to-omics models

Code, environments and notebooks for the figures of
*"Multi-task sequence-to-omics models learn spurious off-target predictive motif
associations across cell contexts"* (Obbad, Nair, Min, Noble, Schreiber, Kundaje; 2026).

Multi-task sequence models (DeepSEA/Beluga, Enformer, Borzoi, AlphaGenome) assign
biologically implausible effects to well-understood motifs in off-target cell contexts,
for example predicting that inserting an OCT–SOX motif lowers fibroblast accessibility.
This repository reproduces the analyses showing that biased training-locus selection and
multi-task objectives produce these associations, that model-guided sequence design
exploits them, and that a matched single-task model (ChromBPNet) can guide fine-tuning to
attenuate one of them.

---

## Layout

```
feature_leakage_repro/
  common/              shared code
    config.py            path registry for all data and models
    beluga_model.py      PyTorch Beluga/DeepSEA, synthetic datasets, probe
    models.py            model loaders (Beluga, ChromBPNet, AlphaGenome, gReLU)
    motifs.py            motif tables and consensus sequences
    motif_controls.py    shuffled-motif controls
  env/                 conda environment specifications
  data/                small input files shipped with the repo + data provenance
  models/              model weights (download instructions in models/README.md)
  figures/figureN/     Figures 1–7: scripts, notebooks, README
  experiments/
    ag_deleak_finetune/  Figure 7: ChromBPNet-gated fine-tuning of AlphaGenome
                         (figures/figure7/README.md points here)
    tf_chip_leakage/     Supplementary Fig. 2e–g: REST ChIP-seq, AlphaGenome vs BPNet
  tools/               rendering helpers
```

Scripts write their outputs to `figures/figureN/outputs/` or `experiments/<name>/outputs/`.

Figures 3–6 and the AlphaGenome column of Supplementary Fig. 1a were produced with the
notebooks in each figure's `notebooks/` folder, which are included with their saved
outputs. The `run_*.py` scripts in the same folders run the same analyses from the command
line. These analyses draw random sequences, so reruns give slightly different numbers.

---

## Environments

| env | stack | used for |
|-----|-------|----------|
| `featleak-dl` | PyTorch 2.7, tangermeme, bpnet-lite, gReLU, modiscolite | Beluga, ChromBPNet, Enformer, Borzoi; Figs 1–6 |
| `featleak-agtorch` | PyTorch 2.5 + [`alphagenome-pytorch`](https://github.com/rastogiruchir/alphagenome-pytorch) | full-length AlphaGenome (524,288 bp): Figs 1a, 1c, 2, 7 |
| `featleak-alphagenome` | JAX + AlphaGenome v1 | Supplementary Fig. 2b and 2e–g |
| `featleak-tf` | TensorFlow 2.15, tf-modisco | Fig. 4b; REST BPNet models (Supplementary Fig. 2e–g) |

```bash
bash env/setup_envs.sh --from-yml
```

Pass `CUDA_VISIBLE_DEVICES=<id>` to select a GPU. Full-length AlphaGenome runs need
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on a 46 GB GPU.

Every data file and model is resolved through `common/config.py`. Point it at your local
copies with `common/paths.yaml` or `FEATLEAK_<KEY>` environment variables, then check that
everything resolves:

```bash
python -m common.config
```

---

## Data and models

- [`data/README.md`](data/README.md): input data, the files shipped in `data/`, and the
  full list of public datasets with accessions (`data/data_availability.xlsx`).
- [`models/README.md`](models/README.md): where to download each model and where to put it.

| model | source |
|-------|--------|
| DeepSEA/Beluga | Kipoi [`DeepSEA/beluga`](https://kipoi.org/models/DeepSEA/beluga/) |
| Enformer, Borzoi | downloaded automatically by [gReLU](https://github.com/Genentech/gReLU) |
| AlphaGenome v1 (JAX) | Google DeepMind, [`google/alphagenome-all-folds`](https://huggingface.co/google/alphagenome-all-folds) (access-restricted) |
| AlphaGenome (PyTorch port) | [`gtca/alphagenome_pytorch`](https://huggingface.co/gtca/alphagenome_pytorch) |
| ChromBPNet, fibroblast of dermis (ENCSR000EPO) | shipped in `models/chrombpnet/` |
| BPNet, K562 REST ChIP-seq | ENCODE (links in [`experiments/tf_chip_leakage/README.md`](experiments/tf_chip_leakage/README.md)) |

---

## Figure → code

The fibroblast analyses in Figs 1c, 2 and 7 use one ENCODE DNase-seq experiment,
ENCSR000EPO (fibroblast of dermis): AlphaGenome track CL:0002551, Enformer/Borzoi task
ENCFF302JEV, the ChromBPNet model trained on ENCSR000EPO, and enhancers from its peaks
(ENCFF184KAM).

### Main figures

| Fig | panel | code | env |
|-----|-------|------|-----|
| 1 | a | `figure1/run_panel_a_tracks.py` → `plot_panel_a_tracks.py`; notebooks in `figure1/notebooks/` | agtorch / dl |
| 1 | c | `figure1/run_chrombpnet_logfc.py` → `run_alphagenome_torch.py` → `run_logfc_styled.py` | dl / agtorch |
| 1 | e–h | `figure1/run_beluga_grid.py` → `plot_beluga_grid.py`; logos `run_example_logos.py` | dl |
| 2 | b | `figure2/run_raw_alphagenome.py`, `run_raw_chrombpnet.py` → `plot_raw_insertion.py` | agtorch / dl |
| 2 | c, d | `figure2/run_tracks_dermis.py` → `run_tracks_matched.py` | agtorch / dl |
| 3 | b–g | `figure3/notebooks/train_ap_1_counting.ipynb`; script `figure3/run.py` | dl |
| 4 | a, c | `figure4/notebooks/analysis_3.ipynb`; scripts `figure4/run_experiment.py`, `run_modisco.py` | dl |
| 4 | b | `figure4/run_modisco_keras.py`; notebook `figure4/notebooks/panel_b_modisco_keras.ipynb` | tf |
| 5 | b | `figure5/notebooks/simulation_analysis.ipynb`; script `figure5/run_simulation.py` | dl |
| 5 | d, e | `figure5/notebooks/feature_entanglement.ipynb`; script `figure5/run_entanglement.py` | dl |
| 6 | b–d | `figure6/run_design.py` → `run_plots.py`, `run_panel_c.py`; notebook `figure6/notebooks/analysis_2.ipynb` | dl |
| 7 | b, d | `experiments/ag_deleak_finetune/finetune_dermis_torch.py` → `plot_fig7_dermis.py` | agtorch / dl |
| 7 | c | `experiments/ag_deleak_finetune/endo_screen.py` → `endo_ism.py` → `plot_fig7c_endo.py` | agtorch / dl |

Panels 1b, 1d, 2a, 3a, 5a, 5c, 6a and 7a are schematics.

### Supplementary figures

| Supp | panel | code |
|------|-------|------|
| 1 | a | AlphaGenome: `figure1/notebooks/supp1a_alphagenome/motif_ablations.ipynb` |
| 2 | a | `figure2/run_raw_grelu.py --model {enformer,borzoi}` → `plot_raw_insertion.py` |
| 2 | b–d | `figure1/run_volcano_alphagenome.py`, `run_volcano_grelu.py` → `volcano_plot.py` |
| 2 | e–g | `experiments/tf_chip_leakage/` (see its README) |
| 3 | a–c | `figure3/notebooks/train_ap_1_counting.ipynb` (script `figure3/run.py`) |
| 4 | a–e | `figure4/notebooks/analysis_3.ipynb`; AUC bars `figure4/plot_auc_bars.py` |
| 4 | f–h | `figure4/run_newhead_finetune.py --new-head fos` |
| 5 | a, b | `figure5/notebooks/simulation_analysis.ipynb`, `feature_entanglement.ipynb` |
| 6 | a–c | `figure6/run_plots.py`; notebook `figure6/notebooks/analysis_2.ipynb` |

Each figure folder has a README with commands and parameters.

```bash
conda activate featleak-dl
cd figures/figure2 && CUDA_VISIBLE_DEVICES=0 python run_raw_chrombpnet.py && python plot_raw_insertion.py
```
