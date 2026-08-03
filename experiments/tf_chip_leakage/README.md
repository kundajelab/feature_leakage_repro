# Experiment: TF-motif → wrong-TF ChIP leakage in AlphaGenome

**Question:** does inserting one TF's motif change AlphaGenome's predicted
**ChIP-seq** signal for a *different* TF that doesn't bind that motif?

**Method (marginalization):** in a single cell type (default K562), take N random
background sequences, and for each well-known TF motif:
1. insert the motif at the centre,
2. predict AlphaGenome `CHIP_TF` before vs after,
3. record the change in the central-window signal for **every** TF track in that
   cell type (averaged over that TF's tracks).

This yields a `(inserted motif) × (measured ChIP TF)` effect matrix. The
diagonal (a motif's own TF) is the expected on-target response; **off-diagonal,
cross-family** responses are feature leakage. TFs are grouped into motif
families (`TF_FAMILY`) so that legitimate motif-sharing — e.g. JUN/JUND/FOSL1
all binding AP-1, GATA1/GATA2, MAX/MYC binding the E-box — is not miscounted as
leakage.

```bash
conda activate featleak-alphagenome
python run.py --n-backgrounds 24 --cell-type K562     # or --cell-type HepG2
```

## Outputs (`outputs/`)
- `leakage_matrix_<cell>.pdf` — raw Δ ChIP heatmap (rows = inserted motif,
  columns = measured TF). Family blocks appear on the diagonal.
- `leakage_matrix_<cell>_normalized.pdf` — each row divided by its on-target Δ,
  so off-diagonal leakage is on a common scale.
- `leakage_matrix_<cell>.csv` — the raw effect matrix.
- `leakage_report_<cell>.txt` — per-motif on-target Δ and top cross-family
  leakage.

Uses the **local** AlphaGenome v1 model (offline). HepG2 (539 TF tracks) and
K562 (306) have the most ChIP-seq TFs; `HEK293`, `GM12878`, `MCF-7` also work.

## `run_ism.py` — is the leakage motif-specific and REAL? (ISM proof)
A marginalization effect alone doesn't prove *specific* leakage — the measured
TF might just respond to "any insertion". `run_ism.py` settles it: for a
(inserted motif TF, measured TF) pair where the inserted motif is a **different
family** than the measured TF, it inserts the motif into many synthetic
sequences and computes ISM of the **measured (wrong) TF's** ChIP output over the
motif window, averaged across sequences.
```bash
python run_ism.py --n-seqs 10 --cell-type K562
```
Output: `ism_leakage_examples_K562.pdf` (one seqlogo per pair) + `ism_*_to_*.npy`.

**Result (averaged over 10 sequences):** the on-target **control** (MAX motif →
MAX ChIP) recovers the E-box, validating the method. Every leakage pair tested
then shows the measured (wrong) TF's ChIP output specifically attributing to the
*distinct* inserted motif — the seqlogo reconstructs the inserted motif:
- **CTCF motif → JUND (AP-1) ChIP** (Σattr≈1657) — JUND reads the CTCF motif.
- **REST motif → SPI1 (ETS) ChIP** (Σattr≈2735) — SPI1 reads the REST NRSE motif.
- **NRF1 motif → YY1 ChIP** (Σattr≈858) — YY1 reads the NRF1 motif.
- **AP-1 motif → MAX ChIP** (Σattr≈234) — MAX reads the AP-1 motif.

All are genuine, motif-specific cross-TF leakage. Note: AP-1→MAX was flat/noisy
at n=2 (Σattr≈0) and only emerged once averaged over ~10 sequences — a concrete
reason to average over MANY synthetic sequences before concluding a marginalized
effect is or isn't real.

## `run_profiles.py` — before/after ChIP profiles
The profile view of the same marginalization: for each pair it averages the
MEASURED TF's predicted ChIP-seq **profile** over N random backgrounds, before
vs after inserting the motif at the centre.
```bash
python run_profiles.py --n-backgrounds 16 --cell-type K562
```
Output: `marginalization_profiles_K562.pdf` (before = gray, after = red, dashed
line = insertion site). The control (MAX->MAX) shows a peak at the insertion;
the leakage pairs (CTCF->JUND, REST->SPI1, NRF1->YY1, AP-1->MAX) show the *wrong*
TF's ChIP signal rising where a motif it does not bind was inserted.
