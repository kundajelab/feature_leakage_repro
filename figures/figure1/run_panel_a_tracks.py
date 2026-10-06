#!/usr/bin/env python
"""Figure 1A: observed / predicted fibroblast DNase + OCT-SOX ISM logo, per model.

For each model the example OCT-SOX locus is fixed (the FIMO hit shown in the
manuscript). The script saves, on a common bp coordinate system:

    pred   predicted fibroblast DNase, bins expanded to 1 bp
    ism    (ISM_BP, 4) log2 fold-change of the central-2 kb predicted DNase for
           every single-base substitution in the ISM_BP window centred on the motif

The observed ENCODE bigWig and the plotting (common window width, to-scale
highlight) are handled by plot_panel_a_tracks.py.

Run:
    CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model borzoi      # featleak-dl
    CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model enformer    # featleak-dl
    CUDA_VISIBLE_DEVICES=0 python run_panel_a_tracks.py --model alphagenome # featleak-alphagenome
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pyfaidx  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "panel_a"
ISM_BP = 100            # ISM window (bp), centred on the motif
SCORE_BP = 2048         # predicted DNase summed over this central window for ISM
TRACK_BP = 65536        # stored track span (plots crop to a common window)

# model -> (motif chrom, FIMO start (1-based), FIMO stop, observed bigWig key, context bp)
LOCI = {
    "alphagenome": ("chr8", 40414031, 40414043, "bigwig_fibroblast_ag", 524288),
    "borzoi": ("chr20", 20464157, 20464169, "bigwig_enformer_fibroblast", 524288),
    "enformer": ("chr5", 115472525, 115472537, "bigwig_enformer_fibroblast", 196608),
    # dermis track via the PyTorch port; locus chosen by screen (see README)
    "alphagenome_dermis": ("chr17", 80859790, 80859802, "bigwig_dermis", 524288),
}
# fibroblast DNase task per model (same ENCODE experiments as the original notebooks)
AG_ONTOLOGY = "CL:0002548"          # cardiac fibroblast (ENCFF424KBD observed)
GRELU_TASK = "ENCFF302JEV"          # fibroblast of dermis (ENCFF361BTT observed)

BASES = "ACGT"


def one_hot(seq):
    lut = {c: i for i, c in enumerate(BASES)}
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in lut:
            oh[lut[c], i] = 1
    return oh


def ism_variants(seq, center):
    """All single-base substitutions in the ISM window -> list of (pos, base, seq)."""
    a = center - ISM_BP // 2
    out = []
    for p in range(a, a + ISM_BP):
        for b in BASES:
            if b != seq[p].upper():
                out.append((p - a, BASES.index(b), seq[:p] + b + seq[p + 1:]))
    return out


# --------------------------------------------------------------------------- #
# gReLU (Borzoi / Enformer)
# --------------------------------------------------------------------------- #
def run_grelu(name, seq):
    import grelu.resources
    import pandas as pd
    import torch

    proj, mname, bin_bp = {"borzoi": ("borzoi", "human_fold0", 32),
                           "enformer": ("enformer", "human", 128)}[name]
    model = grelu.resources.load_model(project=proj, model_name=mname)
    model = model.to("cuda").eval()
    tasks = pd.DataFrame(model.data_params["tasks"])
    hit = tasks.index[tasks.apply(lambda r: r.astype(str).str.contains(GRELU_TASK).any(),
                                  axis=1)].tolist()
    assert len(hit) == 1, hit
    t = hit[0]
    print(f"{name}: task {t} = {tasks.iloc[t].to_dict()}")

    @torch.no_grad()
    def predict(seqs):
        X = torch.from_numpy(np.stack([one_hot(s) for s in seqs])).cuda()
        y = model(X)[:, t, :].float().cpu().numpy()     # (B, bins)
        return y

    L = len(seq)
    base = predict([seq])[0]
    n_bins = len(base)
    off = (L - n_bins * bin_bp) // 2              # first bp covered by bin 0
    pred_bp = np.repeat(base, bin_bp)             # track in bp over [off, off+n_bins*bin)
    c_bin = (L // 2 - off) // bin_bp
    k = SCORE_BP // bin_bp // 2

    def score(y):
        return y[:, c_bin - k:c_bin + k].sum(1)

    ref_score = score(base[None])[0]
    variants = ism_variants(seq, L // 2)
    scores = []
    bs = 4
    for i in range(0, len(variants), bs):
        scores.append(score(predict([v[2] for v in variants[i:i + bs]])))
        print(f"  ISM {i + bs}/{len(variants)}", end="\r", flush=True)
    scores = np.concatenate(scores)
    return pred_bp, off, ref_score, variants, scores, str(tasks.iloc[t].get("description", GRELU_TASK))


# --------------------------------------------------------------------------- #
# AlphaGenome, PyTorch port (all-folds weights; matches JAX), full-length input
# --------------------------------------------------------------------------- #
AG_DERMIS_CHANNEL = 60   # CL:0002551 fibroblast of dermis = ENCODE ENCSR000EPO


def run_alphagenome_torch(seq):
    import torch
    from alphagenome_pytorch import AlphaGenome

    model = AlphaGenome.from_pretrained(
        str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(BASES):
        lut[ord(ch)] = i

    @torch.no_grad()
    def predict(s):
        idx = lut[np.frombuffer(s.encode(), dtype=np.uint8)]
        x = torch.zeros((1, len(s), 4), dtype=torch.float32)
        ok = idx >= 0
        x[0, np.where(ok)[0], idx[ok]] = 1
        out = model.predict(x.cuda(), organism_index=0, heads=("dnase",))
        return out["dnase"][1][0, :, AG_DERMIS_CHANNEL].float().cpu().numpy()

    L = len(seq)
    base = predict(seq)
    c, k = L // 2, SCORE_BP // 2
    ref_score = base[c - k:c + k].sum()
    variants = ism_variants(seq, c)
    scores = []
    for i, v in enumerate(variants):
        scores.append(predict(v[2])[c - k:c + k].sum())
        print(f"  ISM {i + 1}/{len(variants)}", end="\r", flush=True)
    return base, 0, ref_score, variants, np.array(scores), "fibroblast of dermis DNase (CL:0002551)"


# --------------------------------------------------------------------------- #
# AlphaGenome (local weights)
# --------------------------------------------------------------------------- #
def run_alphagenome(seq):
    from alphagenome.models import dna_output
    from common.models import load_alphagenome

    model = load_alphagenome()

    def predict(s):
        out = model.predict_sequence(sequence=s,
                                     requested_outputs=[dna_output.OutputType.DNASE],
                                     ontology_terms=[AG_ONTOLOGY])
        v = np.asarray(out.dnase.values)
        return v.mean(1) if v.ndim == 2 else v

    L = len(seq)
    base = predict(seq)
    assert len(base) == L, (len(base), L)
    c, k = L // 2, SCORE_BP // 2
    ref_score = base[c - k:c + k].sum()
    variants = ism_variants(seq, c)
    scores = []
    for i, v in enumerate(variants):
        scores.append(predict(v[2])[c - k:c + k].sum())
        print(f"  ISM {i + 1}/{len(variants)}", end="\r", flush=True)
    return base, 0, ref_score, variants, np.array(scores), "cardiac fibroblast DNase (CL:0002548)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(LOCI), required=True)
    ap.add_argument("--locus", default=None,
                    help="override the example motif as chrom:start-stop (FIMO, 1-based)")
    ap.add_argument("--context", type=int, default=None,
                    help="override the model input length (bp)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    chrom, s1, e1, bw_key, ctx = LOCI[args.model]
    ctx = args.context or ctx
    suffix = "" if args.context is None else f"_ctx{ctx}"
    if args.locus:
        chrom, rng_ = args.locus.split(":")
        s1, e1 = (int(v) for v in rng_.replace(",", "").split("-"))
        suffix += f"_{chrom}_{s1}"
    s0, e0 = s1 - 1, e1                                 # 0-based half-open motif
    center = s0 + (e0 - s0) // 2
    w0 = center - ctx // 2
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    seq = str(fa[chrom][w0:w0 + ctx]).upper()
    assert len(seq) == ctx

    if args.model == "alphagenome":
        pred_ctx, off, ref_score, variants, scores, task = run_alphagenome(seq)
    elif args.model == "alphagenome_dermis":
        pred_ctx, off, ref_score, variants, scores, task = run_alphagenome_torch(seq)
    else:
        pred_ctx, off, ref_score, variants, scores, task = run_grelu(args.model, seq)

    # ISM matrix: log2FC of central-window DNase, 0 at the reference base
    ism = np.zeros((ISM_BP, 4), dtype=np.float32)
    eps = 1e-6
    for (p, b, _), sc in zip(variants, scores):
        ism[p, b] = np.log2((sc + eps) / (ref_score + eps))
    ism_start = center - ISM_BP // 2                   # genomic 0-based start

    # common bp coordinate window for tracks
    t0 = center - TRACK_BP // 2
    pred = np.full(TRACK_BP, np.nan, dtype=np.float32)
    g0 = w0 + off                                        # genomic start of pred_ctx
    a = max(t0, g0); b = min(t0 + TRACK_BP, g0 + len(pred_ctx))
    pred[a - t0:b - t0] = pred_ctx[a - g0:b - g0]

    np.savez_compressed(
        OUT / f"panel_a_{args.model}{suffix}.npz", chrom=chrom, motif_start=s0, motif_end=e0,
        center=center, track_start=t0, pred=pred, bigwig_key=bw_key, ism=ism,
        ism_start=ism_start, ref_seq=seq[ism_start - w0:ism_start - w0 + ISM_BP],
        ref_score=ref_score, task=task, context=ctx)
    print(f"\nsaved {OUT / f'panel_a_{args.model}{suffix}.npz'}  ref central-2kb DNase={ref_score:.3f}")


if __name__ == "__main__":
    main()
