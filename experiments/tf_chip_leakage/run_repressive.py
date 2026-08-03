#!/usr/bin/env python
"""Find REPRESSIVE cross-TF feature leakage in AlphaGenome (K562 TF ChIP-seq).

Positive cross-TF effects (motif A raises TF B's predicted ChIP) can be genuine
cooperativity / co-binding, so they're weak evidence of leakage. A motif A that
REPRESSES a *different-family* TF B's binding has no co-binding explanation, so
it's a cleaner superposition/leakage signature.

Repression needs a baseline to repress (on random DNA B's signal is ~0), so:
  1. base = random background + B's OWN motif at centre  -> creates B ChIP signal
  2. insert cross-family motif A in the FLANK (mid +/-120)
  3. measure the change in B's central ChIP signal.
A composition-matched SHUFFLE of A is the control: real repression that the
shuffle does NOT reproduce = motif-specific leakage, not a generic-insertion
artifact. We also require A to ACTIVATE its own TF (on-target sanity check).

Stage 1 scans all cross-family (A,B) pairs (real only) to rank repressors;
stage 2 adds the shuffle control + Wilcoxon for the top pairs.

Run (env: featleak-alphagenome):
    python run_repressive.py --n-bg 12 --top 8
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
from scipy.stats import wilcoxon  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common.models import load_alphagenome  # noqa: E402
from common.ag_score import _output_types  # noqa: E402
from run import TF_MOTIFS, TF_FAMILY  # noqa: E402  (reuse the curated motif set)

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 16384
CENTER_BP = 600          # window around centre to read B's signal
FLANKS = [-120, 120]     # where A is inserted (flank of B)


def shuffled(motif, k, seed0=100):
    out = []
    for j in range(k):
        rng = np.random.default_rng(seed0 + j)
        out.append("".join(rng.permutation(list(motif))))
    return out


def chip(model, seq, ontology, OT):
    return model.predict_sequence(sequence=seq, requested_outputs=[OT.CHIP_TF],
                                  ontology_terms=[ontology]).chip_tf


def central_for_tf(track, tf):
    vals = np.asarray(track.values)
    c = vals.shape[0] // 2
    half = max(1, int(CENTER_BP / 2 * vals.shape[0] / SEQ_LEN))
    central = vals[c - half:c + half].sum(axis=0)
    tfs = np.asarray(track.metadata["transcription_factor"])
    return float(pd.Series(central, index=tfs).groupby(level=0).mean().get(tf, np.nan))


def profile_for_tf(model, seqs, ontology, OT, tf, half_bp=1200):
    """Mean CHIP profile (positions) for TF `tf` over a list of sequences."""
    acc, coords = None, None
    for s in seqs:
        track = chip(model, s, ontology, OT)
        vals = np.asarray(track.values)
        tfs = np.asarray(track.metadata["transcription_factor"])
        cols = np.where(tfs == tf)[0]
        prof = vals[:, cols].mean(axis=1)
        acc = prof if acc is None else acc + prof
    prof = acc / len(seqs)
    L = len(prof); c = L // 2; half = max(1, int(half_bp * L / SEQ_LEN))
    x = (np.arange(L) - c) * (SEQ_LEN / L)
    return x[c - half:c + half], prof[c - half:c + half]


def insert_flanks(X, motif):
    mid = SEQ_LEN // 2
    cur = X
    for o in FLANKS:
        cur = substitute(cur, motif, start=mid + o - len(motif) // 2)
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=12)
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--k-shuf", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_alphagenome(); OT = _output_types()
    meta = model.output_metadata().chip_tf
    cell = meta[meta.biosample_name == args.cell_type]
    ontology = cell.ontology_curie.iloc[0]
    avail = sorted(cell.transcription_factor.unique())
    tfs = [t for t in TF_MOTIFS if t in avail]
    print(f"{args.cell_type} ({ontology}): {len(tfs)} usable TFs: {tfs}", flush=True)

    X = random_one_hot((args.n_bg, 4, SEQ_LEN), random_state=args.seed)
    mid = SEQ_LEN // 2

    # base sequences per TF B (random bg + B motif at centre) and B's baseline signal
    base_seq, base_sig, bg_sig = {}, {}, {}
    for B in tfs:
        Xb = substitute(X, TF_MOTIFS[B], start=mid - len(TF_MOTIFS[B]) // 2)
        base_seq[B] = Xb
        base_sig[B] = np.array([central_for_tf(chip(model, characters(x), ontology, OT), B)
                                for x in Xb])
        bg_sig[B] = np.array([central_for_tf(chip(model, characters(x), ontology, OT), B)
                              for x in X])
        print(f"  B={B}: baseline(+own motif)={base_sig[B].mean():.1f}  "
              f"bg={bg_sig[B].mean():.1f}", flush=True)

    # on-target check: B must actually respond to its own motif
    responders = [B for B in tfs if base_sig[B].mean() - bg_sig[B].mean() > 1.0]
    print(f"\nTFs that respond to their own motif (usable as B): {responders}\n", flush=True)

    # STAGE 1: scan cross-family (A,B), real A in flank, Delta = B(base+A) - B(base)
    rows, draws = [], {}
    for B in responders:
        for A in tfs:
            if TF_FAMILY.get(A) == TF_FAMILY.get(B):
                continue
            Xa = insert_flanks(base_seq[B], TF_MOTIFS[A])
            sig = np.array([central_for_tf(chip(model, characters(x), ontology, OT), B)
                            for x in Xa])
            d = sig - base_sig[B]
            draws[(A, B)] = d
            rows.append({"A": A, "B": B, "dReal": float(d.mean()),
                         "frac_repressed": float((d < 0).mean())})
        print(f"  scanned repressors of {B}", flush=True)
    scan = pd.DataFrame(rows).sort_values("dReal").reset_index(drop=True)
    scan.to_csv(OUT / f"repressive_scan_{args.cell_type}.csv", index=False)
    np.savez(OUT / f"repressive_draws_{args.cell_type}.npz",
             **{f"{A}|{B}": d for (A, B), d in draws.items()})
    print("\n=== top repressive cross-family (A represses B) ===")
    print(scan.head(12).to_string(index=False), flush=True)

    # STAGE 2: composition-matched shuffle control + Wilcoxon for the top repressors
    top = scan.head(args.top)
    val = []
    for r in top.itertuples(index=False):
        A, B = r.A, r.B
        d_shuf = []
        for sm in shuffled(TF_MOTIFS[A], args.k_shuf):
            Xs = insert_flanks(base_seq[B], sm)
            s = np.array([central_for_tf(chip(model, characters(x), ontology, OT), B)
                          for x in Xs])
            d_shuf.append(s - base_sig[B])
        d_shuf = np.mean(d_shuf, axis=0)
        d_real = draws[(A, B)]
        p_rep = wilcoxon(d_real, alternative="less").pvalue
        p_spec = wilcoxon(d_real, d_shuf).pvalue          # real vs shuffle (specificity)
        val.append({"A": A, "B": B, "dReal": float(d_real.mean()),
                    "dShuf": float(d_shuf.mean()),
                    "specificity": float(d_shuf.mean() - d_real.mean()),
                    "p_repress": p_rep, "p_vs_shuffle": p_spec})
        print(f"  {A}->{B}: real={d_real.mean():.1f} shuf={d_shuf.mean():.1f} "
              f"p_rep={p_rep:.2g} p_spec={p_spec:.2g}", flush=True)
    vdf = pd.DataFrame(val)
    vdf.to_csv(OUT / f"repressive_validated_{args.cell_type}.csv", index=False)
    print("\n=== validated repressive leakage (real vs shuffle) ===")
    print(vdf.to_string(index=False), flush=True)

    # STAGE 3: before/after ChIP profile plots for the specificity-passing top pairs
    keep = vdf[(vdf.p_vs_shuffle < 0.05) & (vdf.specificity > 0)].head(6)
    if len(keep) == 0:
        keep = vdf.head(4)
    n = len(keep)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.1 * n), squeeze=False)
    for ax, r in zip(axes[:, 0], keep.itertuples(index=False)):
        A, B = r.A, r.B
        xb, pb = profile_for_tf(model, [characters(x) for x in base_seq[B]],
                                ontology, OT, B)
        Xa = insert_flanks(base_seq[B], TF_MOTIFS[A])
        xa, pa = profile_for_tf(model, [characters(x) for x in Xa], ontology, OT, B)
        Xs = insert_flanks(base_seq[B], shuffled(TF_MOTIFS[A], 1)[0])
        xs, ps = profile_for_tf(model, [characters(x) for x in Xs], ontology, OT, B)
        ax.plot(xb, pb, color="#222", lw=1.6, label=f"{B} ChIP (own motif only)")
        ax.plot(xa, pa, color="#c1121f", lw=1.6, label=f"+ {A} motif in flank (real)")
        ax.plot(xs, ps, color="#888", lw=1.2, ls="--", label=f"+ shuffled {A} (control)")
        for o in FLANKS:
            ax.axvline(o, color="#1764ab", lw=0.7, ls=":")
        ax.set_title(f"{A} motif REPRESSES {B} ChIP  "
                     f"(Δreal={r.dReal:.0f}, Δshuf={r.dShuf:.0f}, "
                     f"spec p={r.p_vs_shuffle:.1e})", fontsize=9)
        ax.set_ylabel("pred. ChIP")
        ax.legend(fontsize=6, loc="upper right")
    axes[-1, 0].set_xlabel("position relative to centre (bp)")
    fig.suptitle(f"Repressive cross-TF feature leakage — AlphaGenome {args.cell_type}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"repressive_profiles_{args.cell_type}.pdf")
    plt.close(fig)
    print("done ->", OUT / f"repressive_profiles_{args.cell_type}.pdf")


if __name__ == "__main__":
    main()
