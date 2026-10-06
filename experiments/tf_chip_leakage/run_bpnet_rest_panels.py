#!/usr/bin/env python
"""Data for the REST-BPNet version of the AlphaGenome REST leakage panel.

For one ENCODE REST BPNet model (K562; 5-fold ensemble), computes the same three
readouts as run_rest_leakage_figure.py does for AlphaGenome:
  b  ISM -> REST ChIP (central 600 bp counts): REST motif at the centre (on-target)
     and each leaking motif at the +120 bp flank copy; ref-base contribution
     = prediction minus mean over the 4 bases, averaged over --n-ism backgrounds
  c  mean predicted REST ChIP profile (both strands) over --n-bg backgrounds:
     REST motif only, + NRF1 / YY1 (real), + NRF1 / YY1 (shuffled)
Panel a (log2FC real vs shuffled) comes from run_bpnet_rest_leakage.py.

Run (env: chrombpnet, TF 2.8):
    python run_bpnet_rest_panels.py --experiment ENCSR000BMW
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import run_bpnet_rest_leakage as R

PAD = 3
LEAKERS = [("NRF1", "NRF1"), ("YY1", "YY1"), ("E-box (MYC/MAX)", "MYC"), ("CTCF", "CTCF")]


def profiles(model, seqs, bs=128):
    """Mean predicted profile (counts/bp, strands summed) over seqs, mean over folds."""
    tf = model.tf
    acc = np.zeros(R.OUT_LEN)
    for i in range(0, len(seqs), bs):
        x = R.one_hot(seqs[i:i + bs]); n = len(x)
        feed = dict(sequence=x, counts_bias_input_0=np.zeros((n, 2), np.float32),
                    profile_bias_input_0=np.zeros((n, R.OUT_LEN, 2), np.float32))
        for s in model.sigs:
            p = s(**{k: tf.constant(v) for k, v in feed.items()})
            lg = p["profile_predictions"].numpy().reshape(n, -1)
            pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            cnt = np.exp(p["logcounts_predictions"].numpy().reshape(n, 1))
            acc += (cnt * pr).reshape(n, R.OUT_LEN, 2).sum((0, 2))
    return acc / (len(seqs) * len(model.sigs))


def ism(model, seqs, w0, w1):
    """Mean ref-base contribution (4, w1-w0) to central REST counts."""
    win = w1 - w0
    muts = [s[:w0 + i] + b + s[w0 + i + 1:] for s in seqs for i in range(win) for b in R.BASES]
    vals = model.central_counts(muts).reshape(len(seqs), win, 4)
    out = []
    for s, v in zip(seqs, vals):
        v = v.T                                                  # (4, win)
        oh = np.zeros((4, win)); oh[[R.BASES.index(s[w0 + i]) for i in range(win)], range(win)] = 1
        out.append((v - v.mean(0, keepdims=True)) * oh)
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--n-bg", type=int, default=256)
    ap.add_argument("--n-ism", type=int, default=32)
    args = ap.parse_args()
    model = R.FoldEnsemble(args.experiment)
    rng = np.random.default_rng(0)                    # same backgrounds as the log2FC run
    mid = R.IN_LEN // 2
    bg = ["".join(rng.choice(list(R.BASES), R.IN_LEN)) for _ in range(args.n_bg)]
    rest = R.TF_MOTIFS["REST"]
    base = [R.put(s, rest, mid - len(rest) // 2) for s in bg]

    def insert(seqs, motif):
        out = []
        for s in seqs:
            for o in R.FL:
                s = R.put(s, motif, mid + o - len(motif) // 2)
            out.append(s)
        return out

    res = {}
    r0 = mid - len(rest) // 2
    res["ism_REST"] = ism(model, base[:args.n_ism], r0 - PAD, r0 + len(rest) + PAD)
    for name, key in LEAKERS:
        m = R.TF_MOTIFS[key]
        m0 = mid + 120 - len(m) // 2
        res[f"ism_{key}"] = ism(model, insert(base[:args.n_ism], m), m0 - PAD, m0 + len(m) + PAD)
        print(f"ISM {name}->REST net={res[f'ism_{key}'].sum():+.2f}", flush=True)
    res["prof_base"] = profiles(model, base)
    for key in ("NRF1", "YY1"):
        m = R.TF_MOTIFS[key]
        res[f"prof_{key}_real"] = profiles(model, insert(base, m))
        res[f"prof_{key}_shuf"] = profiles(model, insert(base, R.shuffled(m, 1)[0]))
    out = R.OUT / f"bpnet_rest_panels_{args.experiment}.npz"
    np.savez(out, **res)
    print("saved", out)


if __name__ == "__main__":
    main()
