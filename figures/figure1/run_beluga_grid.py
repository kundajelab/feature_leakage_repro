#!/usr/bin/env python
"""Compute Beluga per-hit, per-task motif effects with four estimators by affinity quartile (Figure 1 E-H).

Outputs (outputs/beluga_grid/): beluga_<motif>_<hit-set><tag>.npz.

Run (env: featleak-dl):
    python run_beluga_grid.py --motif OCT-SOX --hit-set genome --n-per-quartile 3000 --tag _big
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfaidx  # noqa: E402
import torch  # noqa: E402
from tangermeme.deep_lift_shap import (  # noqa: E402
    _clear_hooks, _maxpool, _nonlinear, _register_hooks)
from tangermeme.ersatz import dinucleotide_shuffle  # noqa: E402

from common import config  # noqa: E402
from common.models import load_beluga, load_beluga_task_names  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "beluga_grid"
WIN = 2000
CHROMS = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]

# motif -> association-hit key (JASPAR FIMO scans: MA0142.1, MA0476.1, MA0114.4, MA0139.1)
MOTIFS = {"OCT-SOX": "assoc_octsox", "AP-1": "assoc_ap1",
          "HNF4": "assoc_hnf4", "CTCF": "assoc_ctcf"}


class BelugaLogits(torch.nn.Module):
    """Beluga without the final sigmoid, restricted to the DNase tasks.

    The (1, k) Conv2d/MaxPool2d body is rewritten as the numerically identical
    Conv1d/MaxPool1d stack, which is several times faster for the batched
    DeepLIFT backward passes.
    """

    def __init__(self, beluga, task_idx):
        super().__init__()
        layers = []
        for m in beluga.model[0]:
            if isinstance(m, torch.nn.Conv2d):
                c = torch.nn.Conv1d(m.in_channels, m.out_channels, m.kernel_size[1])
                c.weight.data = m.weight.data[:, :, 0, :].clone()
                c.bias.data = m.bias.data.clone()
                layers.append(c)
            elif isinstance(m, torch.nn.MaxPool2d):
                layers.append(torch.nn.MaxPool1d(4, 4))
            elif isinstance(m, torch.nn.ReLU):
                layers.append(torch.nn.ReLU())
        self.body = torch.nn.Sequential(*layers)
        lin1, lin2 = beluga.model[1][2][1], beluga.model[1][4][1]
        self.head = torch.nn.Sequential(torch.nn.Flatten(), lin1, torch.nn.ReLU(), lin2)
        self.register_buffer("task_idx", torch.as_tensor(task_idx))

    def forward(self, x):
        y = self.head(self.body(x[:, [0, 2, 1, 3]]))
        return y[:, self.task_idx]


# rescale-rule handlers for the only nonlinearities Beluga uses
_NON_LINEAR_OPS = {torch.nn.ReLU: _nonlinear, torch.nn.MaxPool1d: _maxpool}


def dnase_tasks():
    names = load_beluga_task_names()
    idx = np.array([i for i, n in enumerate(names) if "DNase" in n])
    return idx, [names[i] for i in idx]


# --------------------------------------------------------------------------- #
# hit sampling
# --------------------------------------------------------------------------- #
def overlaps_bed(hits, bed_path):
    bed = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2])
    flag = np.zeros(len(hits), dtype=bool)
    for chrom, sub in hits.groupby("chrom"):
        b = bed[bed[0] == chrom].sort_values(1)
        if b.empty:
            continue
        starts, ends = b[1].values, b[2].values
        s0 = sub.start.values - 1
        e0 = sub.stop.values
        # closest element starting at or before the hit end
        j = np.searchsorted(starts, e0, side="left") - 1
        ok = j >= 0
        jj = np.clip(j, 0, None)
        flag[sub.index.values] = ok & (ends[jj] > s0)
    return flag


def sample_hits(motif, hit_set, n_per_q, seed=0):
    df = pd.read_csv(config.resolve(MOTIFS[motif], must_exist=True), sep="\t",
                     usecols=["sequence_name", "start", "stop", "strand", "score",
                              "matched_sequence"])
    df = df[df.sequence_name.isin(CHROMS)].rename(columns={"sequence_name": "chrom"})
    df = df.reset_index(drop=True)
    # quartile boundaries always come from the genome-wide score distribution
    q = df.score.quantile([0.25, 0.5, 0.75]).values
    df["quartile"] = np.searchsorted(q, df.score.values, side="left")
    if hit_set == "ccre":
        df = df[overlaps_bed(df, config.resolve("encode_ccre_bed", must_exist=True))]
        df = df.reset_index(drop=True)
    out = []
    for k in range(4):
        grp = df[df.quartile == k]
        out.append(grp.sample(min(n_per_q * 2, len(grp)), random_state=seed))
    return pd.concat(out).reset_index(drop=True), q


def one_hot_windows(hits, fasta_path):
    """(N, 4, WIN) one-hot windows centred on each hit + motif span in window."""
    fa = pyfaidx.Fasta(str(fasta_path))
    lut = np.full(256, 4, dtype=np.int64)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i
        lut[ord(c.lower())] = i
    X, spans, keep = [], [], []
    for r in hits.itertuples():
        s0, e0 = r.start - 1, r.stop
        w0 = s0 + (e0 - s0) // 2 - WIN // 2
        seq = str(fa[r.chrom][w0:w0 + WIN])
        if len(seq) != WIN:
            keep.append(False); continue
        codes = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
        if (codes == 4).any():
            keep.append(False); continue
        oh = np.zeros((4, WIN), dtype=np.float32)
        oh[codes, np.arange(WIN)] = 1
        X.append(oh); spans.append((s0 - w0, e0 - w0)); keep.append(True)
    return torch.from_numpy(np.stack(X)), np.array(spans), np.array(keep)


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, X, device, bs=128):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).float().cpu())
    return torch.cat(out)


def ism_effect(model, X, spans, device):
    """sum over span of logit(ref) - mean(logit(alt)) -> (N, T)."""
    res = []
    for i in range(len(X)):
        a, b = spans[i]
        x = X[i]
        muts = []
        for p in range(a, b):
            ref = int(x[:, p].argmax())
            for alt in range(4):
                if alt == ref:
                    continue
                m = x.clone(); m[:, p] = 0; m[alt, p] = 1
                muts.append(m)
        y = predict(model, torch.cat([x[None], torch.stack(muts)]), device)
        alts = y[1:].view(b - a, 3, -1).mean(1)          # (L, T)
        res.append((y[0][None] - alts).sum(0))
    return torch.stack(res).numpy()


def ablation_effect(model, X, spans, device, n=10, seed=0):
    """logit(original) - mean logit(span mononucleotide-shuffled) -> (N, T)."""
    rng = np.random.default_rng(seed)
    before = predict(model, X, device)
    after = torch.zeros_like(before)
    for k in range(n):
        Xs = X.clone()
        for i, (a, b) in enumerate(spans):
            Xs[i, :, a:b] = X[i, :, a:b][:, rng.permutation(b - a)]
        after += predict(model, Xs, device)
    return (before - after / n).numpy()


def marginalization_effect(model, X, spans, device, n=5, seed=0):
    """mean over n shuffled-locus backgrounds of logit(bg+motif) - logit(bg)."""
    bg = dinucleotide_shuffle(X, n=n, random_state=seed)          # (N, n, 4, L)
    eff = 0
    for k in range(n):
        B = bg[:, k].clone()
        Bm = B.clone()
        for i, (a, b) in enumerate(spans):
            Bm[i, :, a:b] = X[i, :, a:b]
        eff = eff + (predict(model, Bm, device) - predict(model, B, device))
    return (eff / n).numpy()


def deeplift_effect(model, X, spans, device, n_refs=10, seed=0, pair_bs=384):
    """DeepLIFT/SHAP attribution summed over the motif span, for every task.

    Uses tangermeme's rescale-rule hooks, but batches (hit, reference, task)
    triples so all 334 tasks are explained without 334 separate calls.
    Returns (N, T) effects and the max convergence delta.
    """
    T = len(model.task_idx)
    refs = dinucleotide_shuffle(X, n=n_refs, random_state=seed)   # (N, R, 4, L)
    for m in model.modules():
        m._NON_LINEAR_OPS = _NON_LINEAR_OPS
    model.apply(_register_hooks)
    out = np.zeros((len(X), T), dtype=np.float32)
    max_delta = 0.0
    r_idx, t_idx = np.meshgrid(np.arange(n_refs), np.arange(T), indexing="ij")
    r_idx, t_idx = r_idx.ravel(), t_idx.ravel()
    try:
        for i in range(len(X)):
            a, b = spans[i]
            xi = X[i].to(device)
            ri = refs[i].to(device)
            acc = torch.zeros(T, device=device)
            for s in range(0, len(r_idx), pair_bs):
                rr = torch.as_tensor(r_idx[s:s + pair_bs], device=device)
                tt = torch.as_tensor(t_idx[s:s + pair_bs], device=device)
                P = len(rr)
                _X = xi[None].expand(P, -1, -1).clone().requires_grad_()
                _R = ri[rr].clone().requires_grad_()
                with torch.autograd.set_grad_enabled(True):
                    y = model(torch.cat([_X, _R]))
                    y = y[torch.arange(2 * P, device=device), torch.cat([tt, tt])]
                    mult = torch.autograd.grad(y.sum(), _X)[0]
                diff = (_X - _R).detach()
                delta = (y[:P] - y[P:] - (diff * mult).sum((1, 2))).abs().max().item()
                max_delta = max(max_delta, delta)
                span_attr = (diff[:, :, a:b] * mult[:, :, a:b]).sum((1, 2))
                acc.index_add_(0, tt, span_attr.detach())
            out[i] = (acc / n_refs).cpu().numpy()
    finally:
        model.apply(_clear_hooks)
    return out, max_delta


def _pool_idx(z):
    _, idx = torch.nn.functional.max_pool1d(z, 4, 4, return_indices=True)
    return idx


@torch.no_grad()
def deeplift_fast(model, X, R, U):
    """Exact DeepLIFT (tangermeme rescale + maxpool rules) for ALL tasks at once.

    DeepLIFT's modified backward is linear in the output gradient, so the
    attribution <U, multipliers_t> for every task t equals one forward pass of U
    through the transposed (linearised) network. X, R, U: (P, 4, L) input,
    reference and weighting (e.g. (X - R) masked to the motif span).
    Returns (P, T).
    """
    perm = [0, 2, 1, 3]
    x, r, u = X[:, perm], R[:, perm], U[:, perm]
    for m in model.body:
        if isinstance(m, torch.nn.Conv1d):
            x, r = m(x), m(r)
            u = torch.nn.functional.conv1d(u, m.weight)
        elif isinstance(m, torch.nn.ReLU):
            d_in = x - r
            d_out = torch.relu(x) - torch.relu(r)
            mult = torch.where(d_in.abs() < 1e-6, (x > 0).to(x.dtype), d_out / d_in)
            x, r, u = torch.relu(x), torch.relu(r), u * mult
        elif isinstance(m, torch.nn.MaxPool1d):
            d_in = x - r
            ia, ib = _pool_idx(x), _pool_idx(r)
            ox, orf = m(x), m(r)
            M = torch.maximum(ox, orf)
            small = d_in.abs() < 1e-7
            safe = torch.where(small, torch.ones_like(d_in), d_in)
            ca = torch.where(small, torch.ones_like(d_in), 1.0 / safe)   # a-route
            cb = torch.where(small, torch.zeros_like(d_in), 1.0 / safe)  # b-route
            ua = torch.gather(u * ca, 2, ia) * (M - orf)
            # small-delta positions route the plain gradient (coefficient 1)
            sa = torch.gather(small.to(u.dtype), 2, ia)
            ua = ua * (1 - sa) + torch.gather(u, 2, ia) * sa
            ub = torch.gather(u * cb, 2, ib) * (ox - M)
            u = ua + ub
            x, r = ox, orf
    lin1, lin2 = model.head[1], model.head[3]
    x, r, u = x.flatten(1), r.flatten(1), u.flatten(1)
    x, r, u = lin1(x), lin1(r), u @ lin1.weight.T
    d_in = x - r
    mult = torch.where(d_in.abs() < 1e-6, (x > 0).to(x.dtype),
                       (torch.relu(x) - torch.relu(r)) / d_in)
    u = (u * mult) @ lin2.weight.T
    return u[:, model.task_idx]


def deeplift_effect_fast(model, X, spans, device, n_refs=10, seed=0, bs=64):
    """Span-summed DeepLIFT/SHAP for every task via deeplift_fast -> (N, T).

    Also returns the max summation-to-delta error over all tasks (full-input
    attribution vs logit(x) - logit(ref)).
    """
    refs = dinucleotide_shuffle(X, n=n_refs, random_state=seed)   # (N, R, 4, L)
    N = len(X)
    out = np.zeros((N, len(model.task_idx)), dtype=np.float32)
    max_delta = 0.0
    pair_h = np.repeat(np.arange(N), n_refs)
    pair_r = np.tile(np.arange(n_refs), N)
    acc = torch.zeros(N, len(model.task_idx))
    for s in range(0, len(pair_h), bs):
        h, rr = pair_h[s:s + bs], pair_r[s:s + bs]
        Xb = X[h].to(device)
        Rb = refs[h, rr].to(device)
        D = Xb - Rb
        mask = torch.zeros_like(D)
        for k, i in enumerate(h):
            a, b = spans[i]
            mask[k, :, a:b] = 1
        span_attr = deeplift_fast(model, Xb, Rb, D * mask)
        if s == 0 or s % (bs * 20) == 0:   # summation-to-delta check
            full = deeplift_fast(model, Xb, Rb, D)
            dy = model(Xb) - model(Rb)
            max_delta = max(max_delta, (full - dy).abs().max().item())
        acc.index_add_(0, torch.as_tensor(h), span_attr.float().cpu())
    out[:] = (acc / n_refs).numpy()
    return out, max_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", choices=list(MOTIFS), required=True)
    ap.add_argument("--hit-set", choices=["genome", "ccre"], default="genome")
    ap.add_argument("--n-per-quartile", type=int, default=400)
    ap.add_argument("--n-refs", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="ablation,marginalization,ism,deeplift")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(0)

    task_idx, task_names = dnase_tasks()
    model = BelugaLogits(load_beluga(device=dev), task_idx).to(dev).eval()

    hits, qbounds = sample_hits(args.motif, args.hit_set, args.n_per_quartile)
    X, spans, keep = one_hot_windows(hits, config.resolve("hg38_fasta"))
    hits = hits[keep].reset_index(drop=True)
    # keep exactly n per quartile after N-filtering
    sel = np.concatenate([np.where(hits.quartile.values == k)[0][:args.n_per_quartile]
                          for k in range(4)])
    hits, X, spans = hits.iloc[sel].reset_index(drop=True), X[sel], spans[sel]
    print(f"[{args.motif}/{args.hit_set}] {len(hits)} hits "
          f"({np.bincount(hits.quartile, minlength=4)}), {len(task_idx)} DNase tasks",
          flush=True)

    # baseline logits: lets plots condition on sites the model calls accessible
    res = {"baseline": predict(model, X, dev).numpy()}
    for meth in args.methods.split(","):
        t0 = time.time()
        if meth == "ablation":
            res[meth] = ablation_effect(model, X, spans, dev)
        elif meth == "marginalization":
            res[meth] = marginalization_effect(model, X, spans, dev)
        elif meth == "ism":
            res[meth] = ism_effect(model, X, spans, dev)
        elif meth == "deeplift":
            res[meth], d = deeplift_effect_fast(model, X, spans, dev, n_refs=args.n_refs)
            res["deeplift_max_delta"] = np.array(d)
            print(f"    deeplift max convergence delta {d:.2e}")
        print(f"    {meth}: {time.time() - t0:.0f}s", flush=True)

    tag = args.motif.replace("-", "")
    fname = f"beluga_{tag}_{args.hit_set}{args.tag}.npz"
    np.savez_compressed(
        OUT / fname,
        **res, quartile=hits.quartile.values, score=hits.score.values,
        chrom=hits.chrom.values.astype(str), start=hits.start.values,
        stop=hits.stop.values, strand=hits.strand.values.astype(str),
        matched=hits.matched_sequence.values.astype(str), spans=spans,
        quartile_bounds=qbounds, task_names=np.array(task_names))
    print("saved", OUT / fname)


if __name__ == "__main__":
    main()
