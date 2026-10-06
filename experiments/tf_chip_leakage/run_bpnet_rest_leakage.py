#!/usr/bin/env python
"""Do single-task REST BPNet models show the REST ChIP-seq leakage seen in AlphaGenome?

Same design as run_rest_leakage_figure.py (AlphaGenome, K562 REST ChIP):
  * random backgrounds with REST's own motif at the centre (creates REST signal)
  * a query motif written twice in the flanks (+/-120 bp)
  * readout = predicted REST ChIP counts in the central 600 bp
  * effect = log2FC(real motif) vs log2FC(3 shuffled copies of the motif)

Models: the ENCODE BPNet models trained on each K562 REST ChIP-seq experiment
(annotations ENCSR364RHG / ENCSR759WXZ / ENCSR831SVZ / ENCSR867KCV), TensorFlow
SavedModels, predictions averaged over the 5 folds; control-track inputs set to 0
(as allowed by the ENCODE README). Input 2,114 bp, output 1,000 bp x 2 strands.

Run (env: chrombpnet, TF 2.8):
    python run_bpnet_rest_leakage.py --n-bg 256
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon



def _load_tf_motifs():
    """TF_MOTIFS from run.py (the AlphaGenome screen), read without importing it."""
    import ast
    tree = ast.parse((Path(__file__).parent / "run.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TF_MOTIFS":
            return ast.literal_eval(node.value)
    raise KeyError("TF_MOTIFS")


TF_MOTIFS = _load_tf_motifs()

OUT = Path(__file__).parent / "outputs" / "bpnet_rest"
MODELS = Path("/users/kobbad/feat_leakage/data/bpnet_rest_k562")
EXPERIMENTS = ["ENCSR000BMW", "ENCSR137ZMQ", "ENCSR000ATM", "ENCSR054JMQ"]
IN_LEN, OUT_LEN = 2114, 1000
FL = [-120, 120]
CENTER_BP = 600
BASES = "ACGT"


def one_hot(seqs):
    lut = np.full(256, -1, dtype=np.int64)
    for i, c in enumerate(BASES):
        lut[ord(c)] = i
    idx = np.stack([lut[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])
    x = np.zeros(idx.shape + (4,), dtype=np.float32)
    n, p = np.where(idx >= 0)
    x[n, p, idx[n, p]] = 1
    return x


def put(seq, motif, start):
    return seq[:start] + motif + seq[start + len(motif):]


def shuffled(motif, k, seed0=100):            # same shuffles as the AlphaGenome figure
    return ["".join(np.random.default_rng(seed0 + j).permutation(list(motif)))
            for j in range(k)]


class FoldEnsemble:
    def __init__(self, exp):
        import tensorflow as tf
        self.tf = tf
        paths = sorted(glob.glob(str(MODELS / exp / "fold_*" / "savedmodel" / "*" / "saved_model.pb")))
        assert len(paths) == 5, paths
        self.sigs = [tf.saved_model.load(str(Path(p).parent)).signatures["serving_default"]
                     for p in paths]

    def central_counts(self, seqs, bs=128):
        """Predicted counts in the central CENTER_BP (both strands), mean over folds."""
        out = []
        for i in range(0, len(seqs), bs):
            x = one_hot(seqs[i:i + bs]); n = len(x)
            feed = dict(sequence=x, counts_bias_input_0=np.zeros((n, 2), np.float32),
                        profile_bias_input_0=np.zeros((n, OUT_LEN, 2), np.float32))
            acc = np.zeros(n)
            for s in self.sigs:
                p = s(**{k: self.tf.constant(v) for k, v in feed.items()})
                logits = p["profile_predictions"].numpy().reshape(n, -1)       # (n, 2000)
                prob = np.exp(logits - logits.max(1, keepdims=True))
                prob = (prob / prob.sum(1, keepdims=True)).reshape(n, OUT_LEN, 2)
                counts = np.exp(p["logcounts_predictions"].numpy().reshape(n))
                c = OUT_LEN // 2
                acc += counts * prob[:, c - CENTER_BP // 2:c + CENTER_BP // 2].sum((1, 2))
            out.append(acc / len(self.sigs))
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=256)
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    mid = IN_LEN // 2
    bg = ["".join(rng.choice(list(BASES), IN_LEN)) for _ in range(args.n_bg)]
    rest = TF_MOTIFS["REST"]
    base = [put(s, rest, mid - len(rest) // 2) for s in bg]

    def insert(seqs, motif):
        out = []
        for s in seqs:
            for o in FL:
                s = put(s, motif, mid + o - len(motif) // 2)
            out.append(s)
        return out

    queries = [m for m in TF_MOTIFS if m not in ("REST", "GATA2", "JUND", "FOSL1", "MAX")]
    rows = []
    for exp in args.experiments.split(","):
        model = FoldEnsemble(exp)
        s_bg = model.central_counts(bg)
        s_base = model.central_counts(base)
        on = np.log2((s_base + 1e-3) / (s_bg + 1e-3))
        print(f"[{exp}] on-target: REST motif vs none log2FC median={np.median(on):+.2f}",
              flush=True)
        rows.append(dict(experiment=exp, query="REST (on-target, vs no motif)",
                         lfc_real=np.median(on), lfc_shuf=np.nan, specificity=np.nan,
                         p_vs_shuffle=np.nan, n=len(on)))
        for q in queries:
            motif = TF_MOTIFS[q]
            s_real = model.central_counts(insert(base, motif))
            s_shuf = np.mean([model.central_counts(insert(base, m)) for m in shuffled(motif, 3)], 0)
            lr = np.log2((s_real + 1e-3) / (s_base + 1e-3))
            ls = np.log2((s_shuf + 1e-3) / (s_base + 1e-3))
            p = wilcoxon(lr, ls).pvalue
            rows.append(dict(experiment=exp, query=q, lfc_real=np.median(lr),
                             lfc_shuf=np.median(ls), specificity=np.median(lr - ls),
                             p_vs_shuffle=p, n=len(lr)))
            print(f"[{exp}] {q:6s}->REST real={np.median(lr):+.3f} shuf={np.median(ls):+.3f} "
                  f"spec={np.median(lr - ls):+.3f} p={p:.2g}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bpnet_rest_leakage.csv", index=False)
    print("saved", OUT / "bpnet_rest_leakage.csv")


if __name__ == "__main__":
    main()
