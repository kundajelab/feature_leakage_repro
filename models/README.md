# Model weights

`common/config.py` looks in this directory first for every model. Download the models you
need and place them here with the filenames below. Only the dermal-fibroblast ChromBPNet
model is included in the repository.

```
models/
  deepsea.beluga.pth                          # DeepSEA/Beluga
  alphagenome-v1/                             # AlphaGenome v1 (JAX), all folds
  alphagenome-pytorch/model_all_folds.safetensors   # AlphaGenome PyTorch port
  chrombpnet/ENCSR000EPO_chrombpnet_wo_bias.h5      # ChromBPNet, fibroblast of dermis (included)
  bpnet_rest_k562/<experiment>/fold_{0..4}/   # ENCODE BPNet models, K562 REST ChIP-seq
```

| file | model | source |
|------|-------|--------|
| `deepsea.beluga.pth` | DeepSEA/Beluga (2,002 outputs) | Kipoi [`DeepSEA/beluga`](https://kipoi.org/models/DeepSEA/beluga/); direct download https://zenodo.org/record/3402406/files/deepsea.beluga.pth |
| `alphagenome-v1/` | AlphaGenome v1, all folds (JAX) | [`google/alphagenome-all-folds`](https://huggingface.co/google/alphagenome-all-folds) (access-restricted); code at https://github.com/google-deepmind/alphagenome |
| `alphagenome-pytorch/model_all_folds.safetensors` | AlphaGenome v1, all folds (PyTorch port) | [`gtca/alphagenome_pytorch`](https://huggingface.co/gtca/alphagenome_pytorch) |
| `chrombpnet/ENCSR000EPO_chrombpnet_wo_bias.h5` | ChromBPNet, fibroblast of dermis (ENCSR000EPO), fold 0, bias-corrected | included in this repository |
| `bpnet_rest_k562/` | ENCODE BPNet models for K562 REST ChIP-seq | download links in [`experiments/tf_chip_leakage/README.md`](../experiments/tf_chip_leakage/README.md) |

Enformer (`human`) and Borzoi (`human_fold0`) are downloaded automatically by
[gReLU](https://github.com/Genentech/gReLU) the first time they are used.

## Models needed per figure

| Figure | Models |
|--------|--------|
| 1 | Beluga (e–h); AlphaGenome PyTorch port (a, c); Enformer, Borzoi (a); ChromBPNet (c) |
| 2 | AlphaGenome PyTorch port; Enformer, Borzoi; ChromBPNet |
| 3, 5, 6 | Beluga (the Fig. 5 simulation trains its own model) |
| 4 | none (models are trained from the peak files in `data/figure4/`) |
| 7 | AlphaGenome PyTorch port; ChromBPNet |
| Supp 2b–d | AlphaGenome v1 (JAX); Enformer, Borzoi |
| Supp 2e–g | AlphaGenome v1 (JAX); ENCODE REST BPNet models |

Check what resolves with `python -m common.config`.
