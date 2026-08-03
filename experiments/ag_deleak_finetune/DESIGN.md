# De-leaking AlphaGenome's fibroblast DNase head with a ChromBPNet oracle

## Goal
Fine-tune AlphaGenome's **native** skin-fibroblast DNase head (track `NTR:0000521`,
channel 207) so it stops over-reacting to the OCT-SOX motif, WITHOUT degrading its
agreement with experimental data.

## The phenomenon (confirmed)
On strong skin-biceps fibroblast enhancers, inserting OCT-SOX in the flanks:
- **ChromBPNet (oracle)**: log2FC(real) ≈ **-0.03** (no effect); log2FC(shuffle) ≈ **-0.42**
  (generic disruption). Real is *less* repressive than random in 97.8% of enhancers →
  **no OCT-SOX-specific repression.**
- **AlphaGenome (leaking)**: real OCT-SOX drives an OCT-SOX-*specific* extra drop
  (track_screen specificity ≈ +0.12 for NTR:0000521; recomputed on these enhancers in
  step 2). This gap is the leakage.

Units differ between models, so everything is done in **log2 fold-change** (unitless,
relative), per the plan.

## Data streams for fine-tuning (only the 6 `alphagenome/head/dnase/*` leaves train)
1. **Fidelity anchor (don't degrade).** Real skin-biceps peaks -> real experimental
   DNase signal (bigWig). Keeps Pearson r intact. Loss = per-bp count loss
   (Poisson / log1p-MSE / native-multinomial; picked in step 3) on `predictions_1bp[...,207]`.
2. **De-leak (remove OCT-SOX response).** For enhancer E and its OCT-SOX-inserted E'
   (4 flank offsets), teach AG to reproduce ChromBPNet's *relative* effect. Two
   formulations, both compared:

   **(A) Corrected-target invariance.** Precompute frozen `AG_base(E)` (pre-FT). Target
   for E' = `AG_base(E) * 2^(cbp_lfc_real(E))`. Since cbp_lfc_real ≈ 0, the head learns
   E' ≈ E (no OCT-SOX drop). Units never cross models — ChromBPNet only supplies the
   scalar relative factor. Runs as ordinary supervised targets (cheap: one forward/step).

   **(B) Online paired-logFC penalty.** Loss = `(AG_log2fc(E',E) - cbp_lfc_real(E))^2`,
   AG_log2fc computed live from the current model (two forwards/step). More direct.

   Optional shuffle-insertion examples (target `2^(cbp_lfc_shuf)`) guard against AG
   learning "any insertion is a no-op" instead of "OCT-SOX specifically is benign."

Total loss = fidelity + lambda * de-leak. lambda swept.

## Splits
400 oracle enhancers -> de-leak train / held-out leakage-eval. Pearson-r fidelity
uses a separate held-out peak set (disjoint from de-leak enhancers).

## Success criteria (step 6)
- OCT-SOX specificity (median real-vs-shuffle log2FC) on held-out enhancers -> ~0
  (matches ChromBPNet), down from the pre-FT AG value.
- Pearson r vs experimental bigWig on held-out peaks unchanged (within noise).
- Report A vs B.
