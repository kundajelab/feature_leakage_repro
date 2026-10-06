# Figure 1a notebooks

| notebook | model | env | notes |
|----------|-------|-----|-------|
| `alphagenome/oct_sox_attribs.ipynb` | AlphaGenome (API) | `featleak-dl` | needs network access and `export ALPHAGENOME_API_KEY=...` |
| `borzoi/oct_sox_attribs.ipynb` | Borzoi (gReLU) | `featleak-dl` | GPU; imports the local `utils.py` |
| `enformer/enformer_attribs.ipynb` | Enformer (gReLU) | `featleak-dl` | GPU; imports the local `utils.py` |
| `supp1a_alphagenome/motif_ablations.ipynb` | AlphaGenome (API) | `featleak-dl` | Supplementary Fig. 1a, AlphaGenome column |

```bash
conda activate featleak-dl
export ALPHAGENOME_API_KEY=...        # AlphaGenome notebooks only
cd borzoi && jupyter nbconvert --to notebook --execute oct_sox_attribs.ipynb --inplace
```

Edit the data paths in the first cells (genome FASTA, FIMO matches, ENCODE peaks and
bigWigs) to your local copies; see [`../../../data/README.md`](../../../data/README.md).
