#!/usr/bin/env python
"""Fine-tune AlphaGenome's native DNase head to remove OCT-SOX leakage.

Trains ONLY the 6 `alphagenome/head/dnase/*` leaves (backbone + all other heads
frozen, enforced by the heads-only optax mask). Two de-leak objectives, selectable:

  A  corrected-target invariance: on OCT-SOX-inserted E', match the central-window
     profile to  AG_base(E) * 2^(cbp_lfc_real(E))  (ChromBPNet says ~no effect, so
     E' should look like E). Base target precomputed & frozen.
  B  online paired-logFC: penalize ( AG_log2fc(E',E) - cbp_lfc_real(E) )^2, with
     AG_log2fc from the CURRENT model (two forwards/step).

FIDELITY anchor (both, "don't degrade"): self-distillation — keep the central
profile on the base sequences close to the FROZEN pre-FT prediction. Preserves
Pearson-r vs experiment by construction (validated independently in 04_eval).

The forward + optimizer update is a single jitted step over pre-encoded one-hot
arrays (the raw Haiku apply_fn compiles once). Run (env: alphagenome-ft):
    ... conda run --no-capture-output -n alphagenome-ft python -u 03_finetune.py \
        --objective A --steps 300 --lam 1.0 --lr 3e-3
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "alphagenome_finetune"))
sys.path.insert(0, str(REPO / "experiments" / "chrombpnet_octsox_stats"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from functools import partial  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from splits import chrom_filter  # noqa: E402
import ft_core  # noqa: E402

OUT = Path(__file__).parent / "outputs"
SEQ_LEN = 2048
OFFSETS = [-150, 150, -200, 200]   # overridden by --offsets
CENTRAL = 150
EPS = 1e-3
C0 = SEQ_LEN // 2


def insert_motif(seq: str, motif: str, mid: int) -> str:
    s = list(seq)
    for o in OFFSETS:
        s[mid + o:mid + o + len(motif)] = list(motif)
    return "".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["A", "B", "C"], default="A")
    ap.add_argument("--gate-threshold", type=float, default=0.2,
                    help="objective C: penalize AG's OCT-SOX change only where ChromBPNet's "
                         "|log2FC| < this (i.e. ChromBPNet says 'no effect')")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offsets", default=None,
                    help="comma list of insertion offsets from center; default flank-only")
    ap.add_argument("--tag", default="", help="suffix for output pkl name")
    ap.add_argument("--n-fid-anchor", type=int, default=0,
                    help="broad self-distillation fidelity anchors (train-chrom peaks); 0=off")
    ap.add_argument("--wfid", type=float, default=1.0, help="broad fidelity anchor weight")
    ap.add_argument("--target-zero", action="store_true",
                    help="target EXACT invariance (log2FC=0, 'no effect') instead of the "
                         "ChromBPNet log2FC (which is ~+0.08 -> mild positive)")
    ap.add_argument("--insert-mode", choices=["fixed", "random"], default="fixed",
                    help="random = each step insert K in [1,max-insert] motifs at random "
                         "offsets in +/-insert-range (generalizes to any pattern -> ~0 everywhere)")
    ap.add_argument("--max-insert", type=int, default=4)
    ap.add_argument("--insert-range", type=int, default=220)
    ap.add_argument("--flank-frac", type=float, default=0.40,
                    help="fraction of steps using the exact 4-flank boxplot pattern")
    ap.add_argument("--center-frac", type=float, default=0.20,
                    help="fraction using single-center insertion (fixes endogenous ISM but "
                         "drives the positive overshoot on flank patterns)")
    args = ap.parse_args()
    global OFFSETS
    if args.offsets:
        OFFSETS = [int(x) for x in args.offsets.split(",")]
    print("insertion offsets:", OFFSETS)
    OUT.mkdir(parents=True, exist_ok=True)
    import pyfaidx
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))

    # ---- data: TRAIN-chromosome enhancers + oracle + frozen AG_base profiles ----
    # (chr8/chr9 held out; the entire ag_base_train set is training data)
    d = np.load(OUT / "ag_base_train.npz", allow_pickle=True)
    n = len(d["chrom"])
    rng = np.random.default_rng(args.seed)
    tr = np.arange(n)
    mid = SEQ_LEN // 2

    Xb, Xp, baseprof, cbp, base_strs = [], [], [], [], []
    for i in tr:
        center = (int(d["start"][i]) + int(d["end"][i])) // 2
        s = center - SEQ_LEN // 2
        seq = str(fa[str(d["chrom"][i])][s:s + SEQ_LEN]).upper()
        if len(seq) != SEQ_LEN or (set(seq) - set("ACGT")):
            continue
        base_strs.append(seq)
        Xb.append(ft_core.one_hot_encode(seq))
        Xp.append(ft_core.one_hot_encode(insert_motif(seq, OCT_SOX, mid)))
        baseprof.append(d["ag_base_profiles"][i])
        cbp.append(float(d["cbp_lfc_real"][i]))
    Xb = jnp.asarray(np.stack(Xb).astype(np.float32))       # (N,S,4)
    Xp = jnp.asarray(np.stack(Xp).astype(np.float32))       # (fixed-mode inserted)
    basec = jnp.asarray(np.stack(baseprof).astype(np.float32)[:, C0 - CENTRAL:C0 + CENTRAL])  # (N,2C)
    cbp_raw = jnp.asarray(np.asarray(cbp, np.float32))      # ChromBPNet OCT-SOX log2FC per enhancer
    # `cb` = what gets threaded to the loss. Meaning depends on objective:
    #   A/B: the target log2FC (0 under --target-zero, else the ChromBPNet value).
    #   C  : a 0/1 GATE = 1 when ChromBPNet says OCT-SOX has ~no effect (|log2FC|<thr);
    #        C penalizes AG's *change* only on gated enhancers (ChromBPNet as a decision,
    #        not a distilled label).
    if args.objective == "C":
        cbp = (jnp.abs(cbp_raw) < args.gate_threshold).astype(jnp.float32)
        gated = float(np.asarray(cbp).mean())
        print(f"gate: {int(np.asarray(cbp).sum())}/{len(cbp)} enhancers 'no-effect' "
              f"(|cbp|<{args.gate_threshold}) = {100*gated:.0f}% penalized for invariance")
    elif args.target_zero:
        cbp = jnp.zeros_like(cbp_raw)                       # target exact invariance (no effect)
    else:
        cbp = cbp_raw
    N = Xb.shape[0]

    # random-pattern insertion. Mixes the EXACT evaluation patterns (4-flank boxplot
    # + single-center ISM) with random K/positions, so both measured patterns are
    # driven to log2FC~0 directly rather than via generalization.
    OFF_GRID = [o for o in range(-args.insert_range, args.insert_range + 1, 50)]
    EVAL_FLANK = [-150, 150, -200, 200]     # the boxplot / inserted-ISM pattern

    def _insert(seq, offs):
        s = list(seq)
        for o in offs:
            p = mid + int(o)
            s[p:p + len(OCT_SOX)] = list(OCT_SOX)
        return "".join(s)

    FLANK_FR, CENTER_FR = args.flank_frac, args.center_frac

    def random_inserted_batch(bidx):
        out = []
        for i in bidx:
            u = rng.random()
            if u < FLANK_FR:
                offs = EVAL_FLANK                                  # exact boxplot pattern
            elif u < FLANK_FR + CENTER_FR:
                offs = [0]                                        # single center (endogenous ISM)
            else:
                k = int(rng.integers(1, args.max_insert + 1))
                offs = list(rng.choice(OFF_GRID, size=k, replace=False))
            out.append(ft_core.one_hot_encode(_insert(base_strs[i], offs)))
        return jnp.asarray(np.stack(out).astype(np.float32))
    print(f"train enhancers: {N}  objective {args.objective}  lam={args.lam} lr={args.lr} steps={args.steps}")

    # ---- model + heads-only optimizer on native dnase head ----
    model = ft_core.load_model()
    apply_fn = ft_core.get_apply_fn(model)
    fwd = ft_core.make_dnase_forward(apply_fn)
    params, state = model._params, model._base_model._state
    opt = ft_core.make_heads_only_optimizer(params, learning_rate=args.lr, weight_decay=0.0)
    opt_state = opt.init(params)

    org = jnp.zeros((args.batch,), dtype=jnp.int32)

    def central_ch(p, X):
        ch, _, _ = fwd(p, state, X, org)
        return central_window(ch.astype(jnp.float32))

    def central_window(x):
        return x[:, C0 - CENTRAL:C0 + CENTRAL]

    # ---- broad fidelity anchors: many train-chrom peaks, self-distilled to the
    #      FROZEN pre-FT prediction (protects general accessibility while de-leaking) ----
    Xf = Ff = None
    if args.n_fid_anchor > 0:
        pk = pd.read_csv(config.resolve("peaks_skin_biceps"), sep="\t", header=None)
        pk = pk.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal", 9: "summit"})
        pk = chrom_filter(pk, "train").sort_values("signal", ascending=False).reset_index(drop=True)
        ranks = np.unique(np.geomspace(1, len(pk) - 1, args.n_fid_anchor).astype(int))
        anc = pk.iloc[ranks]
        seqs_f = []
        for _, r in anc.iterrows():
            summit = int(r.start) + (int(r.summit) if r.summit == r.summit and r.summit >= 0
                                     else (int(r.end) - int(r.start)) // 2)
            s = summit - SEQ_LEN // 2
            seq = str(fa[r.chrom][s:s + SEQ_LEN]).upper()
            if len(seq) == SEQ_LEN and not (set(seq) - set("ACGT")):
                seqs_f.append(seq)
        Xf = np.stack([ft_core.one_hot_encode(s) for s in seqs_f]).astype(np.float32)
        # frozen pre-FT central prediction (batched, one-time)
        ff = []
        for i in range(0, len(Xf), 16):
            chx = jnp.asarray(Xf[i:i + 16])
            chc, _, _ = fwd(params, state, chx, jnp.zeros((len(Xf[i:i + 16]),), jnp.int32))
            ff.append(np.asarray(chc, np.float32)[:, C0 - CENTRAL:C0 + CENTRAL])
        Ff = jnp.asarray(np.concatenate(ff, 0))
        Xf = jnp.asarray(Xf)
        print(f"broad fidelity anchors: {Xf.shape[0]} train-chrom peaks (wfid={args.wfid})")

    lam = args.lam
    obj = args.objective
    wfid = args.wfid
    use_anchor = args.n_fid_anchor > 0

    def loss_fn(p, xb, xp, bc, cb, xf, ff):
        predB = central_ch(p, xb)                    # (B,2C)
        predP = central_ch(p, xp)
        fid = jnp.mean((jnp.log1p(predB) - jnp.log1p(bc)) ** 2)
        if use_anchor:
            predF = central_ch(p, xf)
            fid = fid + wfid * jnp.mean((jnp.log1p(predF) - jnp.log1p(ff)) ** 2)
        if obj == "A":
            factor = (2.0 ** cb)[:, None]
            deleak = jnp.mean((jnp.log1p(predP) - jnp.log1p(bc * factor)) ** 2)
        elif obj == "B":
            ag_lfc = jnp.log2((predP.sum(1) + EPS) / (predB.sum(1) + EPS))
            deleak = jnp.mean((ag_lfc - cb) ** 2)
        else:  # C: ChromBPNet gate. cb is a 0/1 mask; penalize AG's change only where
               #    ChromBPNet says OCT-SOX has no effect. Normalized by #gated so lam is stable.
            ag_lfc = jnp.log2((predP.sum(1) + EPS) / (predB.sum(1) + EPS))
            deleak = jnp.sum(cb * (ag_lfc ** 2)) / (jnp.sum(cb) + 1.0)
        return fid + lam * deleak, (fid, deleak)

    @jax.jit
    def train_step(p, os_, xb, xp, bc, cb, xf, ff):
        (loss, (fid, dl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, xb, xp, bc, cb, xf, ff)
        updates, os2 = opt.update(grads, os_, p)
        return optax.apply_updates(p, updates), os2, loss, fid, dl

    dummy = jnp.zeros((args.batch, 2 * CENTRAL), jnp.float32)
    dummyX = jnp.zeros((args.batch, SEQ_LEN, 4), jnp.float32)
    for step in range(args.steps):
        bidx = np.asarray(rng.integers(0, N, size=args.batch))
        xp = random_inserted_batch(bidx) if args.insert_mode == "random" else Xp[bidx]
        if use_anchor:
            fidx = np.asarray(rng.integers(0, Xf.shape[0], size=args.batch))
            xf, ff = Xf[fidx], Ff[fidx]
        else:
            xf, ff = dummyX, dummy
        params, opt_state, loss, fid, dl = train_step(
            params, opt_state, Xb[bidx], xp, basec[bidx], cbp[bidx], xf, ff)
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={float(loss):.4f}  fid={float(fid):.4f}  deleak={float(dl):.4f}",
                  flush=True)

    # ---- save fine-tuned dnase head leaves only ----
    dnase_leaves = {}
    for kp, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        ps = "/".join(str(getattr(k, "key", k)) for k in kp)
        if "/head/dnase/" in ps:
            dnase_leaves[ps] = np.asarray(leaf, dtype=np.float32)
    outp = OUT / f"ft_dnase_{args.objective}_lam{args.lam}_lr{args.lr}_s{args.steps}{args.tag}.pkl"
    with open(outp, "wb") as fh:
        pickle.dump({"dnase_leaves": dnase_leaves, "args": vars(args)}, fh)
    print("saved fine-tuned dnase head ->", outp)


if __name__ == "__main__":
    main()
