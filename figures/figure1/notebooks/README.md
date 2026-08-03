# Figure 1A — original attribution notebooks (verbatim)


1. the hard-coded AlphaGenome API key was replaced with
   `os.environ["ALPHAGENOME_API_KEY"]` (so no secret is committed), and
2. hard-coded GPU UUIDs were replaced with `os.environ.get('CUDA_VISIBLE_DEVICES','0')`.

Run them to reproduce the manuscript's Panel A logos/tracks identically.

| notebook | model | env | notes |
|----------|-------|-----|-------|
| `alphagenome/oct_sox_attribs.ipynb` | AlphaGenome | `featleak-dl` (has the `alphagenome` API client) | needs network + `export ALPHAGENOME_API_KEY=...`. The ISM logo cells build `seqlogo_plot.pdf` (`score_ism_variants` → `ism.ism_matrix` → `plot_components.SeqLogo`); the track cells build `trackplot.pdf` (observed bigWig vs predicted DNase). |
| `borzoi/oct_sox_attribs.ipynb` | Borzoi (gReLU) | `featleak-dl` | GPU. ISM logo via `grelu.interpret.score.ISM_predict` + `plot_ISM(method="logo")` → `ism_plot.pdf`; profile track → `profile_plot.pdf`. Imports its local `utils.py` (copied alongside). |
| `enformer/enformer_attribs.ipynb` | Enformer (gReLU) | `featleak-dl` | GPU. Same structure as Borzoi. Imports its local `utils.py`. |

```bash
conda activate featleak-dl
export ALPHAGENOME_API_KEY=...             # only for the AlphaGenome notebook
cd notebooks/borzoi
jupyter nbconvert --to notebook --execute oct_sox_attribs.ipynb --inplace
# or open in Jupyter and run all cells
```

The data paths inside the notebooks point at the lab store (genome FASTA, FIMO
OCT-SOX hits, ENCODE peaks/bigWigs) documented in `../../../data/README.md`.

**Offline AlphaGenome option:** the notebook uses the hosted API. To run the
AlphaGenome ISM fully offline with the local v1 weights instead, swap
`dna_client.create(API_KEY)` for
`from alphagenome_research.model import dna_model; model = dna_model.create(WEIGHTS_DIR)`
(see `common/models.load_alphagenome`) — the local `AlphaGenomeModel` exposes the
same `score_ism_variants` method, so the rest of the notebook is unchanged. Run
that variant in the `featleak-alphagenome` env.
