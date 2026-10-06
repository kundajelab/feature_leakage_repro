# Supplementary Fig. 2e–g — REST ChIP-seq: AlphaGenome vs ENCODE BPNet

Do unrelated TF motifs change predicted REST binding in K562?

**Design (identical for both models):** random background sequences with the REST motif at
the centre; NRF1, YY1, MYC/MAX E-box or CTCF written twice, at −120 and +120 bp; three
shuffled versions of each query motif as controls. Effect = log2 fold change of predicted
REST ChIP signal over the central 600 bp, relative to the background with only the REST
motif. Real vs shuffled motifs are compared with a two-sided paired Wilcoxon signed-rank
test.

- **AlphaGenome:** K562 REST tracks (ENCSR137ZMQ, ENCSR054JMQ) averaged; 16,384 bp input;
  12 backgrounds; local JAX model (`models/alphagenome-v1/`).
- **BPNet:** ENCODE models trained on the same experiments, five folds averaged; 256
  backgrounds. Models for ENCSR000BMW and ENCSR000ATM are included as additional controls.

| Panel | Script (env) |
|---|---|
| e, AlphaGenome | `run_rest_leakage_figure.py` (`featleak-alphagenome`) |
| e, BPNet | `run_bpnet_rest_leakage.py` (`featleak-tf`) → `plot_bpnet_vs_ag_rest.py` (`featleak-dl`) |
| f, g | AlphaGenome: `run_rest_leakage_figure.py`; BPNet: `run_bpnet_rest_panels.py` (`featleak-tf`) → `plot_bpnet_rest_panels.py` |

```bash
python run_rest_leakage_figure.py --n-bg 12 --n-ism 8        # featleak-alphagenome
python run_bpnet_rest_leakage.py --n-bg 256                  # featleak-tf
python run_bpnet_rest_panels.py --experiment ENCSR137ZMQ     # featleak-tf
python plot_bpnet_vs_ag_rest.py                                # featleak-dl
python plot_bpnet_rest_panels.py --experiment ENCSR137ZMQ    # featleak-dl
```

## BPNet models

Download each ENCODE "models" tarball and extract it into
`models/bpnet_rest_k562/<experiment>/`, so that `fold_0/` … `fold_4/` sit inside it:

| experiment | ENCODE model annotation | models file |
|---|---|---|
| ENCSR137ZMQ | ENCSR831SVZ | https://www.encodeproject.org/files/ENCFF980HTW/@@download/ENCFF980HTW.tar.gz |
| ENCSR054JMQ | ENCSR759WXZ | https://www.encodeproject.org/files/ENCFF005ZXD/@@download/ENCFF005ZXD.tar.gz |
| ENCSR000BMW | ENCSR364RHG | https://www.encodeproject.org/files/ENCFF912NJL/@@download/ENCFF912NJL.tar.gz |
| ENCSR000ATM | ENCSR867KCV | https://www.encodeproject.org/files/ENCFF716GEH/@@download/ENCFF716GEH.tar.gz |

The directory can be changed with the `FEATLEAK_BPNET_REST_K562_DIR` environment variable.
