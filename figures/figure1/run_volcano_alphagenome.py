#!/usr/bin/env python
"""Compute and plot the OCT-SOX marginalization volcano across all AlphaGenome DNase tracks (Supplementary Figure 2).

Outputs (outputs/): volcano_alphagenome_raw.npz and volcano_alphagenome[_specificity].csv/.pdf.

Run (env: featleak-alphagenome):
    python run_volcano_alphagenome.py --n-bg 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.utils import characters  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

from common import config  # noqa: E402
from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from volcano_plot import volcano  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384
OFFSETS = [-150, 150, -200, 200, -250, 250, -300, 300]     # flank insertions
READOUT_HALF = 500


def shuffled_motif(motif, seed=100):
    rng = np.random.default_rng(seed)
    return "".join(rng.permutation(list(motif)))


def compute_levels(model, X, motif, OT, n_bg):
    """Return (K+1, N, n_tracks) central DNase and the track metadata."""
    mid = SEQ_LEN // 2
    levels, meta, cur = [], None, X
    for k in range(len(OFFSETS) + 1):
        if k > 0:
            o = OFFSETS[k - 1]
            cur = substitute(cur, motif, start=mid + o - len(motif) // 2)
        rows = []
        for n in range(n_bg):
            central, md = central_all(model, characters(cur[n]), OT)
            rows.append(central); meta = md
        levels.append(np.array(rows))
        print(f"  level {k}/{len(OFFSETS)} predicted", flush=True)
    return np.stack(levels), meta


def real_backgrounds(n):
    peaks = pd.read_csv(config.resolve("peaks_fibroblast"), sep="\t", header=None)
    peaks = peaks.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    peaks = peaks[peaks.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    top = peaks.sort_values("signal", ascending=False).head(n * 2).reset_index(drop=True)
    loci = extract_loci(top[["chrom", "start", "end"]], str(config.resolve("hg38_fasta")),
                        in_window=SEQ_LEN, verbose=False)
    keep = [i for i in range(len(loci)) if int(loci[i].sum()) == SEQ_LEN][:n]
    return loci[keep]


def central_all(model, seq, OT):
    out = model.predict_sequence(sequence=seq, requested_outputs=[OT.DNASE],
                                 ontology_terms=None)          # all DNase tracks
    vals = np.asarray(out.dnase.values)                        # (positions, n_tracks)
    L = vals.shape[0]; c = L // 2; w = max(1, int(READOUT_HALF * L / SEQ_LEN))
    return vals[c - w:c + w].sum(axis=0), out.dnase.metadata   # (n_tracks,), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    model = load_alphagenome(); OT = _output_types()
    X = real_backgrounds(args.n_bg)
    print(f"  {X.shape[0]} real fibroblast enhancer backgrounds", flush=True)
    print("[real OCT-SOX]", flush=True)
    Yr, meta = compute_levels(model, X, OCT_SOX, OT, args.n_bg)
    print("[shuffled OCT-SOX control]", flush=True)
    Ys, _ = compute_levels(model, X, shuffled_motif(OCT_SOX), OT, args.n_bg)

    K1, N, T = Yr.shape
    x = np.repeat(np.arange(K1), N)
    F = 1e-2 * np.median(Yr)
    logr = np.log2(Yr + F); logs = np.log2(Ys + F)
    sl_r, p_r = np.zeros(T), np.ones(T)
    sl_s, p_s = np.zeros(T), np.ones(T)
    dfree = 2 * len(x) - 4
    for t in range(T):
        lr = stats.linregress(x, logr[:, :, t].reshape(-1))
        ls = stats.linregress(x, logs[:, :, t].reshape(-1))
        sl_r[t] = lr.slope
        p_r[t] = lr.pvalue if np.isfinite(lr.pvalue) else 1.0
        d = lr.slope - ls.slope; se = np.hypot(lr.stderr, ls.stderr)
        sl_s[t] = d
        p_s[t] = 2 * stats.t.sf(abs(d / se), dfree) if se > 0 else 1.0

    sample = (meta["biosample_name"].values if "biosample_name" in meta.columns
              else meta.get("ontology_curie", pd.Series([""] * T)).values)
    name = meta["name"].values if "name" in meta.columns else np.arange(T)
    np.savez(OUT / "volcano_alphagenome_raw.npz", Yr=Yr, Ys=Ys, x=x, task=name, sample=sample)
    m = dict(task=name, sample=sample)
    for tag, slope, pval, ttl, xl in [
            ("", sl_r, p_r, "AlphaGenome: OCT-SOX effect across all DNase tasks",
             "OCT-SOX effect per insertion  (log2FC)"),
            ("_specificity", sl_s, p_s, "AlphaGenome: OCT-SOX motif-specific effect (real − shuffled)",
             "OCT-SOX-specific effect per insertion  (log2FC)")]:
        df = pd.DataFrame({**m, "slope": slope, "pval": pval,
                           "qval": multipletests(pval, method="fdr_bh")[1]})
        df.to_csv(OUT / f"volcano_alphagenome{tag}.csv", index=False)
        volcano(df, ttl, OUT / f"volcano_alphagenome{tag}.pdf", xlabel=xl)
        nrep = int(((df.qval < 0.05) & (df.slope < 0)).sum())
        npos = int(((df.qval < 0.05) & (df.slope > 0)).sum())
        print(f"AlphaGenome{tag}: {nrep} repressive, {npos} positive, "
              f"{len(df)-nrep-npos} n.s.", flush=True)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
