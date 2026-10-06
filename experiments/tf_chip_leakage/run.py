#!/usr/bin/env python
"""Compute AlphaGenome's TF ChIP-seq response to each inserted TF motif across all TF tracks in one cell type.

Outputs (outputs/): leakage_matrix_<cell>.csv, leakage_matrix_<cell>.pdf,
leakage_matrix_<cell>_normalized.pdf, leakage_report_<cell>.txt.

Run (env: featleak-alphagenome):
    python run.py --n-backgrounds 24 --cell-type K562
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tangermeme.utils import random_one_hot, characters  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384          # local AlphaGenome min window
CENTER_BP = 2001         # central window (bp) to aggregate the local effect

# Curated set of well-known TFs with distinctive, mostly non-overlapping motifs.
# Keys must match AlphaGenome's `transcription_factor` field. Consensus strings
# are canonical strong instances.
TF_MOTIFS = {
    "CTCF":  "GCCGCCAGGGGGCGCCG",
    "GATA1": "AGATAAGATAA",
    "GATA2": "AGATAAGATAA",
    "JUN":   "TGAGTCATGAGTCA",     # AP-1
    "JUND":  "TGAGTCATGAGTCA",
    "FOSL1": "TGAGTCATGAGTCA",
    "MAX":   "CACGTGCACGTG",       # E-box
    "MYC":   "CACGTGCACGTG",
    "SPI1":  "AAAGAGGAAGTG",       # PU.1 / ETS
    "NRF1":  "TGCGCATGCGCA",
    "REST":  "TTCAGCACCACGGACAGCGCC",
    "SP1":   "GGGGCGGGGCGGGG",
    "EGR1":  "TGCGTGGGCGT",
    "E2F1":  "TTTGGCGCGAAA",
    "YY1":   "AAAATGGCGGCC",
    "TBP":   "TATAAAAGGC",
}

# Motif families: TFs in the same family bind the SAME motif, so a response there
# is EXPECTED, not leakage. Cross-family responses are the leakage of interest.
TF_FAMILY = {
    "CTCF": "CTCF",
    "GATA1": "GATA", "GATA2": "GATA",
    "JUN": "AP-1", "JUND": "AP-1", "FOSL1": "AP-1",
    "MAX": "E-box", "MYC": "E-box",
    "SPI1": "ETS", "NRF1": "NRF1", "REST": "REST", "SP1": "SP",
    "EGR1": "EGR", "E2F1": "E2F", "YY1": "YY1", "TBP": "TBP",
}


def chip_tf_output(model, seq, ontology):
    OutputType = _output_types()
    out = model.predict_sequence(
        sequence=seq, requested_outputs=[OutputType.CHIP_TF],
        ontology_terms=[ontology])
    return out.chip_tf


def central_sum_by_tf(track_data, tf_order):
    """Sum the central window per column, then average columns per TF."""
    vals = np.asarray(track_data.values)          # (positions, n_tracks)
    c = vals.shape[0] // 2
    half = int(CENTER_BP / 2 * vals.shape[0] / SEQ_LEN)  # bins for CENTER_BP
    half = max(half, 1)
    central = vals[c - half:c + half].sum(axis=0)        # (n_tracks,)
    tfs = np.asarray(track_data.metadata["transcription_factor"])
    s = pd.Series(central, index=tfs)
    return s.groupby(level=0).mean().reindex(tf_order).values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-backgrounds", type=int, default=24)
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_alphagenome()
    OutputType = _output_types()
    meta = model.output_metadata().chip_tf
    cell = meta[meta.biosample_name == args.cell_type]
    if len(cell) == 0:
        raise SystemExit(f"no CHIP_TF tracks for {args.cell_type}")
    ontology = cell.ontology_curie.iloc[0]
    avail_tfs = sorted(cell.transcription_factor.unique())
    print(f"{args.cell_type} ({ontology}): {len(cell)} tracks, {len(avail_tfs)} TFs")

    inserted = [tf for tf in TF_MOTIFS if tf in avail_tfs]
    print(f"inserting motifs for {len(inserted)} TFs present in {args.cell_type}: {inserted}")
    if len(inserted) < 2:
        raise SystemExit("need >=2 TFs with both a motif and a track")

    # Background sequences (shared reference).
    X = random_one_hot((args.n_backgrounds, 4, SEQ_LEN), random_state=args.seed)
    bg_strs = [characters(x) for x in X]

    # measured-TF axis = all TFs we inserted (so it's a square-ish leakage matrix)
    measured = inserted
    # effect[i, j] = change in ChIP for measured[j] when motif for inserted[i] added
    effect = np.zeros((len(inserted), len(measured)))
    mid = SEQ_LEN // 2

    print("predicting backgrounds ...")
    before = np.array([central_sum_by_tf(chip_tf_output(model, s, ontology), measured)
                       for s in bg_strs])                # (N, n_measured)

    for i, tf in enumerate(inserted):
        motif = TF_MOTIFS[tf]
        print(f"[{i+1}/{len(inserted)}] inserting {tf} motif ({motif}) ...", flush=True)
        Xin = substitute(X, motif, start=mid - len(motif) // 2)
        after = np.array([central_sum_by_tf(chip_tf_output(model, characters(x), ontology),
                                            measured) for x in Xin])
        effect[i] = (after - before).mean(axis=0)

    df = pd.DataFrame(effect, index=[f"{t} motif" for t in inserted], columns=measured)
    df.to_csv(OUT / f"leakage_matrix_{args.cell_type}.csv")

    _heatmaps(df, inserted, measured, args.cell_type)
    _report(df, inserted, measured, args.cell_type)
    print("done ->", OUT)


def _heatmaps(df, inserted, measured, cell):
    eff = df.values
    # raw effect
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(measured), 1 + 0.6 * len(inserted)))
    vmax = np.nanpercentile(np.abs(eff), 98) or 1
    im = ax.imshow(eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(measured))); ax.set_xticklabels(measured, rotation=90, fontsize=7)
    ax.set_yticks(range(len(inserted))); ax.set_yticklabels([f"{t} motif" for t in inserted], fontsize=7)
    ax.set_xlabel("measured ChIP-seq TF"); ax.set_ylabel("inserted motif")
    ax.set_title(f"AlphaGenome TF-motif marginalization ({cell})")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Δ ChIP signal")
    fig.tight_layout(); fig.savefig(OUT / f"leakage_matrix_{cell}.pdf"); plt.close(fig)

    # row-normalised (each row / its own diagonal) to expose off-target leakage
    norm = eff.copy().astype(float)
    for i in range(len(inserted)):
        d = eff[i, measured.index(inserted[i])] if inserted[i] in measured else np.nan
        norm[i] = eff[i] / d if d and np.isfinite(d) and d != 0 else np.nan
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(measured), 1 + 0.6 * len(inserted)))
    im = ax.imshow(norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(measured))); ax.set_xticklabels(measured, rotation=90, fontsize=7)
    ax.set_yticks(range(len(inserted))); ax.set_yticklabels([f"{t} motif" for t in inserted], fontsize=7)
    ax.set_xlabel("measured ChIP-seq TF"); ax.set_ylabel("inserted motif")
    ax.set_title(f"Effect normalised to on-target ({cell})")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Δ / on-target Δ")
    fig.tight_layout(); fig.savefig(OUT / f"leakage_matrix_{cell}_normalized.pdf"); plt.close(fig)


def _report(df, inserted, measured, cell):
    lines = [f"# AlphaGenome TF-ChIP leakage ({cell})",
             "# 'leakage' = effect on a TF from a DIFFERENT motif family "
             "(same-family responses are expected motif sharing).\n"]
    for i, tf in enumerate(inserted):
        row = df.iloc[i]
        fam = TF_FAMILY.get(tf)
        on = row.get(tf, np.nan)
        # off-target = different family only
        cross = row[[m for m in measured if TF_FAMILY.get(m) != fam]]
        top = cross.sort_values(ascending=False).head(3)
        lines.append(f"{tf} motif [{fam}]: on-target Δ={on:.2f}; "
                     f"top cross-family LEAKAGE: " +
                     ", ".join(f"{k}({TF_FAMILY.get(k)})={v:.2f}" for k, v in top.items()))
    txt = "\n".join(lines)
    print("\n" + txt)
    (Path(OUT) / f"leakage_report_{cell}.txt").write_text(txt)


if __name__ == "__main__":
    main()
