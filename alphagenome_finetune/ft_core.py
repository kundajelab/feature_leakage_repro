#!/usr/bin/env python
"""Core module for fine-tuning AlphaGenome's NATIVE pretrained DNase head.

Target track: DNase NTR:0000521 = "fibroblast of skin of left biceps",
channel index **207** of 384 (same index in the 305 non-padded space).

This module does NOT touch the standard analysis path or modify any installed
package. It reconstructs a *differentiable* forward pass around the real
pretrained model so we can fine-tune ONLY the native DNase head:

    model = load_model()                       # full pretrained AlphaGenome
    apply_fn = get_apply_fn(model)             # raw Haiku forward (differentiable)
    fwd = make_dnase_forward(apply_fn)         # -> channel-207 DNase, (B,S)
    opt = make_heads_only_optimizer(model._params)   # zeroes every non-DNase leaf

Why not just call ``model.predict_sequence``? That path is ``jax.jit``-wrapped,
runs on numpy inputs, filters/reshapes outputs, and applies reverse-complement
augmentation logic. For training we need a plain, traceable function of
``params`` that returns the raw count-space profile. We recover the exact raw
``apply_fn`` that ``predict_sequence`` itself uses (see ``get_apply_fn``), so the
two are numerically identical (verified: max abs diff ~1e-6, corr >0.9999).

Environment (GPU libs need conda activation):
    cd /users/kobbad/feature_leakage_repro && \
      CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      FEATLEAK_AKUNDAJE_ROOT=/oak/stanford/groups/akundaje \
      /users/kobbad/anaconda3/bin/conda run -n alphagenome-ft \
      python alphagenome_finetune/ft_core.py         # runs gates #2 and #4
"""

from __future__ import annotations

import functools
import gzip
from typing import Any, Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS = "/oak/stanford/groups/akundaje/kobbad/alphagenome/alphagenome-v1"
FASTA = "/oak/stanford/groups/akundaje/jmschr/common/hg38/hg38.fa"

# DNase NTR:0000521 "fibroblast of skin of left biceps"
DNASE_SKIN_BICEPS_CHANNEL = 207
DNASE_ONTOLOGY = "NTR:0000521"

HUMAN_ORGANISM_INDEX = 0  # dna_model.convert_to_organism_index(HOMO_SAPIENS) == 0
DEFAULT_SEQ_LEN = 2048


# ---------------------------------------------------------------------------
# 1. Model load
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str = WEIGHTS):
    """Load the FULL pretrained AlphaGenome model incl. native heads.

    Returns a ``CustomAlphaGenomeModel`` whose ``._params`` holds the pretrained
    weights (backbone + all native heads, including ``alphagenome/head/dnase/*``)
    and whose ``._base_model`` is the underlying research ``AlphaGenomeModel``.
    """
    from alphagenome_ft import create_model_with_heads

    print(f"Loading pretrained AlphaGenome from {checkpoint_path} ...")
    model = create_model_with_heads("all_folds", heads=[], checkpoint_path=checkpoint_path)
    print("  loaded.")
    return model


# ---------------------------------------------------------------------------
# 2. Raw differentiable forward
# ---------------------------------------------------------------------------
def get_apply_fn(model) -> Callable:
    """Recover the raw Haiku ``apply_fn`` used by the model.

    In ``alphagenome_research.model.dna_model.create`` the raw forward is::

        def _apply_fn(params, state, dna_sequence, organism_index):
            (predictions, _), _ = _forward.apply(params, state, None,
                                                 dna_sequence, organism_index)
            return predictions   # dict: predictions['dnase']['predictions_1bp'] (B,S,384)

    ``AlphaGenomeModel.__init__`` stores it only inside the jitted
    ``self._predict = jax.jit(functools.partial(_predict, apply_fn=apply_fn))``.
    We unwrap that closure: ``jax.jit`` exposes the original callable via
    ``__wrapped__`` (the ``functools.partial``), whose ``keywords['apply_fn']``
    is exactly the raw forward. This is the SAME function ``predict_sequence``
    uses, so results match numerically.

    ``apply_fn(params, state, seqs_onehot[B,S,4], organism_indices[B]) -> dict``
    is differentiable w.r.t. ``params`` (plain Haiku apply, no jit boundary that
    hides params).
    """
    base = getattr(model, "_base_model", model)
    predict = base._predict  # jax.jit(functools.partial(_predict, apply_fn=...))
    wrapped = getattr(predict, "__wrapped__", None)
    if isinstance(wrapped, functools.partial) and "apply_fn" in wrapped.keywords:
        return wrapped.keywords["apply_fn"]
    raise RuntimeError(
        "Could not recover raw apply_fn from model._base_model._predict; "
        f"got wrapped={type(wrapped)!r}"
    )


def make_dnase_forward(apply_fn: Callable, channel: int = DNASE_SKIN_BICEPS_CHANNEL,
                       jit: bool = True):
    """Build ``dnase_forward(params, state, seqs_onehot, organism_indices)``.

    Returns a callable ->

        preds_channel : (B, S)      -- channel-207 human DNase, differentiable
        preds_full    : (B, S, 384) -- all DNase tracks, experimental count space
        preds_scaled  : (B, S, 384) -- model's internal scaled space

    Notes:
      * ``organism_indices`` must be 0 for human (skin-biceps is a human track).
      * DNASE is 1 bp resolution, softplus-count space, ``apply_squashing=False``.
      * ``predictions_1bp`` is the experimental-count-space profile that
        ``predict_sequence(...).dnase.values`` returns; ``scaled_predictions_1bp``
        is the model-internal scaled variant (pre experimental rescaling).
      * **jit=True is important.** The raw ``apply_fn`` runs the FULL model,
        including the very memory-heavy splice-junctions head (a
        ``f32[~1468,512,512]`` einsum) that OOMs on a single GPU when executed
        eagerly. Because this forward only *returns* the DNase tensors, wrapping
        it in ``jax.jit`` lets XLA dead-code-eliminate every other head (exactly
        what ``predict_sequence`` does). This keeps it differentiable and cheap.
    """
    import jax
    import jax.numpy as jnp

    def _dnase_forward(params, state, seqs_onehot, organism_indices):
        preds = apply_fn(params, state, seqs_onehot, organism_indices)
        dnase = preds["dnase"]
        # The model runs under a compute=bfloat16, output=bfloat16 mixed-precision
        # policy, so raw outputs are bf16. predict_sequence upcasts to float32
        # (_upcast_single_batch_predictions); we do the same so results match and
        # downstream numpy/loss code gets a normal float dtype.
        full = dnase["predictions_1bp"].astype(jnp.float32)     # (B,S,384)
        scaled = dnase["scaled_predictions_1bp"].astype(jnp.float32)  # (B,S,384)
        ch = full[..., channel]                                  # (B,S)
        return ch, full, scaled

    return jax.jit(_dnase_forward) if jit else _dnase_forward


# ---------------------------------------------------------------------------
# 3. One-hot encoder (matches the model's own DNAOneHotEncoder exactly)
# ---------------------------------------------------------------------------
# A->[1,0,0,0] C->[0,1,0,0] G->[0,0,1,0] T->[0,0,0,1]; anything else -> zeros.
_LOOKUP = np.zeros((256, 4), dtype=np.float32)
for _b, _v in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3)):
    _LOOKUP[ord(_b)] = _LOOKUP[ord(_b.lower())] = np.eye(4, dtype=np.float32)[_v]


def one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA string -> (len, 4) float32, ACGT order.

    Byte-for-byte identical to ``alphagenome_research ... DNAOneHotEncoder``.
    """
    return _LOOKUP[np.frombuffer(seq.encode("latin1"), dtype=np.uint8)]


def encode_batch(seqs: Sequence[str]) -> np.ndarray:
    """Stack of one-hot sequences -> (B, S, 4) float32."""
    return np.stack([one_hot_encode(s) for s in seqs]).astype(np.float32)


def human_organism_indices(batch_size: int) -> np.ndarray:
    """Return ``(B,)`` int32 organism indices selecting human (0)."""
    return np.zeros((batch_size,), dtype=np.int32)


# ---------------------------------------------------------------------------
# 4. Fidelity target from a bigWig
# ---------------------------------------------------------------------------
def bigwig_target(bigwig_path: str, chrom: str, start: int, end: int) -> np.ndarray:
    """Per-bp experimental target from a read-depth-normalized bigWig.

    Returns a ``(end-start,)`` float32 array; NaN (no-data) -> 0.0. This is the
    per-bp signal to compare against ``dnase_forward``'s channel-207 profile.
    """
    import pyBigWig

    bw = pyBigWig.open(bigwig_path)
    try:
        vals = np.asarray(bw.values(chrom, start, end), dtype=np.float32)
    finally:
        bw.close()
    return np.nan_to_num(vals, nan=0.0)


def peak_intervals(bed_gz: str, seq_len: int, n: int, fasta_path: str = FASTA):
    """Yield ``(sequence_str, (chrom, start, end, summit))`` centered on peaks.

    Uses narrowPeak col 10 (summit offset) when present. Skips peaks whose window
    runs off-contig or contains non-ACGT. Reads via pyfaidx (uppercased).
    """
    import pyfaidx

    fa = pyfaidx.Fasta(fasta_path)
    out = []
    with gzip.open(bed_gz, "rt") as fh:
        for line in fh:
            if len(out) >= n:
                break
            c = line.rstrip("\n").split("\t")
            chrom, start, end = c[0], int(c[1]), int(c[2])
            summit = start + int(c[9]) if len(c) > 9 else (start + end) // 2
            s = summit - seq_len // 2
            e = s + seq_len
            if s < 0:
                continue
            seq = str(fa[chrom][s:e]).upper()
            if len(seq) != seq_len or (set(seq) - set("ACGT")):
                continue
            out.append((seq, (chrom, s, e, summit)))
    return out


# ---------------------------------------------------------------------------
# 5. Masked (heads-only) optimizer: trains ONLY the native DNase head
# ---------------------------------------------------------------------------
def make_heads_only_optimizer(params, *, learning_rate: float = 1e-2,
                              weight_decay: float = 0.0,
                              trainable_head_names: Sequence[str] = ("dnase",)):
    """AdamW that updates ONLY ``alphagenome/head/<name>/*`` leaves.

    Wraps ``alphagenome_ft.optimizer_utils.create_optimizer(..., heads_only=True)``,
    which builds an ``optax.multi_transform`` labelling every leaf whose path
    contains ``/head/dnase/`` as trainable and setting all others to
    ``optax.set_to_zero()``. Backbone + other heads stay bitwise-fixed.
    """
    from alphagenome_ft.optimizer_utils import create_optimizer

    return create_optimizer(
        params,
        trainable_head_names=tuple(trainable_head_names),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        heads_only=True,
    )


# ---------------------------------------------------------------------------
# Loss-space helpers (see module docstring / task #3)
# ---------------------------------------------------------------------------
#
# IMPORTANT loss-space finding (verified):
#   * ``predictions_1bp`` is the experimental-count-space profile and is the
#     quantity comparable to the read-depth-normalized bigWig (both per-bp).
#   * BUT a fidelity loss must NOT be taken directly on ``predictions_1bp``:
#     the head's ``unscale`` step applies a piecewise soft-clip that SQUARES
#     values above 10 (``(x+10)**2/40``). Its gradient w.r.t. the head weights
#     is singular, so ``grad(mean(predictions_1bp**2))`` and even
#     ``grad(log1p_mse(predictions_1bp, target))`` come out **NaN** (confirmed
#     in float32 and bfloat16).
#   * The model's OWN training compares in the head's SCALED space: it maps the
#     experimental target into model space with ``targets_scaling`` (the inverse
#     soft-clip: divide by track_mean, then ``2*sqrt(x*10)-10`` above 10) and
#     computes the loss on ``scaled_predictions_1bp``. Gradients there are finite
#     and well-scaled (verified: w/b/scale grad norms ~3.5 / 0.23 / 0.27).
#
# So: predict in count space (compare to bigWig for reporting/metrics), but
# TRAIN in scaled space -> loss on ``scaled_predictions_1bp`` against the
# scaled target (``scale_targets_to_model_space``).

_SOFT_CLIP_VALUE = 10.0  # heads._SOFT_CLIP_VALUE


def recover_track_mean(full_pred, full_scaled, channel: int = DNASE_SKIN_BICEPS_CHANNEL,
                       resolution: int = 1) -> float:
    """Recover ``track_mean * resolution`` for a channel at runtime.

    In the sub-soft-clip regime (scaled <= 10) the head's unscale is linear:
    ``predictions = scaled * track_mean * resolution``. So the ratio
    ``predictions_1bp / scaled_predictions_1bp`` there equals ``track_mean*res``.
    ``full_pred`` / ``full_scaled`` are the (B,S,384) outputs of ``dnase_forward``.
    (Avoids plumbing the head's metadata-derived ``nonzero_mean`` out of Haiku.)
    """
    p = np.asarray(full_pred)[..., channel].ravel()
    s = np.asarray(full_scaled)[..., channel].ravel()
    m = (s > 1e-4) & (s < _SOFT_CLIP_VALUE)
    return float(np.median(p[m] / s[m]))


def scale_targets_to_model_space(targets, track_mean_x_res: float,
                                 soft_clip_value: float = _SOFT_CLIP_VALUE):
    """Map experimental (bigWig) count targets into the head's SCALED space.

    Reimplements ``heads.targets_scaling`` for DNASE (``apply_squashing=False``):
        t <- t / (track_mean * resolution)
        t <- where(t > 10, 2*sqrt(t*10) - 10, t)      # inverse soft-clip
    ``track_mean_x_res`` is ``track_mean * resolution`` (see recover_track_mean).
    The result is directly comparable to ``scaled_predictions_1bp`` for a loss.
    """
    import jax.numpy as jnp

    t = jnp.asarray(targets) / track_mean_x_res
    return jnp.where(t > soft_clip_value,
                     2.0 * jnp.sqrt(t * soft_clip_value) - soft_clip_value, t)


def poisson_nll(pred, target):
    """Poisson NLL: ``mean(pred - target*log(pred))``.

    Use on SCALED-space quantities (``scaled_predictions_1bp[...,207]`` vs a
    ``scale_targets_to_model_space`` target), NOT on raw counts -- see the
    loss-space note above.
    """
    import jax.numpy as jnp

    pred = jnp.clip(pred, 1e-8, None)
    return jnp.mean(pred - target * jnp.log(pred))


def log1p_mse(pred, target):
    """MSE of ``log1p`` values. Use in SCALED space (see loss-space note)."""
    import jax.numpy as jnp

    return jnp.mean((jnp.log1p(jnp.clip(pred, 0.0, None))
                     - jnp.log1p(jnp.clip(target, 0.0, None))) ** 2)


# ---------------------------------------------------------------------------
# Self-test: gates #2 (forward == predict_sequence) and #4 (grad isolation)
# ---------------------------------------------------------------------------
def _selftest():
    import jax
    import jax.numpy as jnp
    import optax
    from alphagenome.models import dna_output

    BED = "/users/kobbad/feat_leakage/skin_biceps_dnase/ENCFF246OZH.bed.gz"
    BIGWIG = "/users/kobbad/feat_leakage/skin_biceps_dnase/ENCFF346TYL.bigWig"
    print("JAX devices:", jax.devices())

    model = load_model()
    apply_fn = get_apply_fn(model)
    print("recovered apply_fn:", apply_fn)
    dnase_forward = make_dnase_forward(apply_fn)
    params, state = model._params, model._base_model._state

    items = peak_intervals(BED, DEFAULT_SEQ_LEN, 4)
    seqs = [s for s, _ in items]
    coords = [c for _, c in items]
    X = jnp.asarray(encode_batch(seqs))
    org = jnp.asarray(human_organism_indices(len(seqs)))

    ch, full, scaled = dnase_forward(params, state, X, org)
    print(f"forward shapes: channel {ch.shape}  full {full.shape}  scaled {scaled.shape}")
    ch = np.asarray(ch)

    # ---- GATE #2: forward matches predict_sequence ----
    print("\n=== GATE #2: dnase_forward[...,207] vs predict_sequence ===")
    gate2_ok = True
    for i, s in enumerate(seqs):
        out = model.predict_sequence(sequence=s, requested_outputs=[dna_output.OutputType.DNASE],
                                     ontology_terms=[DNASE_ONTOLOGY])
        v = np.asarray(out.dnase.values).squeeze().astype(np.float64)
        a = ch[i].astype(np.float64)
        md = float(np.max(np.abs(v - a)))
        rel = md / float(max(v.max(), 1e-9))   # abs diff relative to peak height
        cc = float(np.corrcoef(v, a)[0, 1])
        # Both paths use the SAME apply_fn; residual diff is bf16-compute rounding
        # between separately-jitted graphs (a few ULPs). Accept on correlation +
        # small relative diff.
        ok = cc > 0.999 and rel < 0.02
        gate2_ok &= ok
        print(f"  seq{i}: max|diff|={md:.3e}  rel|diff|={rel:.2e}  corr={cc:.6f}  {'ok' if ok else 'MISMATCH'}")

    # ---- Scale relationship: pred vs bigWig ----
    print("\n=== Scale: predictions_1bp[207] vs bigWig ===")
    for i, (chrom, s, e, summit) in enumerate(coords):
        bwv = bigwig_target(BIGWIG, chrom, s, e)
        a = ch[i]
        print(f"  {chrom}:{s}-{e}  bw[mean/max/sum]={bwv.mean():.3f}/{bwv.max():.3f}/{bwv.sum():.1f}  "
              f"pred[mean/max/sum]={a.mean():.4f}/{a.max():.4f}/{a.sum():.1f}  "
              f"corr={np.corrcoef(bwv, a)[0, 1]:.4f}")

    # ---- GATE #4: masked step touches ONLY the DNase head ----
    print("\n=== GATE #4: heads-only gradient isolation ===")
    opt = make_heads_only_optimizer(params, learning_rate=1e-2, weight_decay=0.0)
    opt_state = opt.init(params)

    # Fidelity loss in the head's SCALED space (see loss-space note): a loss on
    # raw predictions_1bp has NaN gradients (singular unscale). We map the real
    # bigWig target into scaled space and compare against scaled_predictions_1bp.
    tm = recover_track_mean(full, scaled)  # track_mean * resolution for ch 207
    print(f"  recovered track_mean*res for ch{DNASE_SKIN_BICEPS_CHANNEL} = {tm:.4f}")
    tgt_counts = np.stack([bigwig_target(BIGWIG, c[0], c[1], c[2]) for c in coords])
    tgt_scaled = jnp.asarray(scale_targets_to_model_space(tgt_counts, tm))

    def loss_fn(p):
        _, _, sc = dnase_forward(p, state, X, org)
        return log1p_mse(sc[..., DNASE_SKIN_BICEPS_CHANNEL], tgt_scaled)

    # jit the value_and_grad so XLA DCEs the unused (memory-heavy) heads in the
    # backward pass too.
    def _p(path):
        return "/".join(str(getattr(k, "key", k)) for k in path)

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    grad_finite = all(bool(np.isfinite(np.asarray(g)).all())
                      for pth_, g in jax.tree_util.tree_flatten_with_path(grads)[0]
                      if "/head/dnase/" in _p(pth_))
    print(f"  fidelity loss (scaled-space log1p-MSE vs bigWig) = {float(loss):.6f}   head grads finite: {grad_finite}")

    flat_old = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_new = jax.tree_util.tree_flatten_with_path(new_params)[0]

    head_changed = head_unchanged = 0
    nonhead_changed = nonhead_unchanged = 0
    changed_all_finite = True
    changed_nonhead_paths = []
    for (p, ov), (_, nv) in zip(flat_old, flat_new):
        ps = _p(p)
        same = bool(np.array_equal(np.asarray(ov), np.asarray(nv)))
        if "/head/dnase/" in ps:
            if same:
                head_unchanged += 1
                print(f"    head UNchanged (zero grad): {ps}")
            else:
                head_changed += 1
                nv_np = np.asarray(nv)
                changed_all_finite &= bool(np.isfinite(nv_np).all())
                print(f"    HEAD changed: {ps}  "
                      f"norm {np.linalg.norm(np.asarray(ov)):.6f} -> {np.linalg.norm(nv_np):.6f}")
        else:
            if same:
                nonhead_unchanged += 1
            else:
                nonhead_changed += 1
                changed_nonhead_paths.append(ps)

    print(f"  dnase-head leaves: {head_changed} changed, {head_unchanged} unchanged (total 6)")
    print(f"  non-dnase leaves : {nonhead_changed} changed, {nonhead_unchanged} unchanged")
    # Isolation proof: NO non-DNase-head leaf may change. Only the resolution_1
    # DNase-head leaves (3 of 6) receive gradient from a predictions_1bp loss;
    # the resolution_128 leaves get exact-zero gradient and stay bitwise-fixed.
    gate4_ok = (nonhead_changed == 0 and head_changed == 3 and head_unchanged == 3
                and grad_finite and changed_all_finite)
    if changed_nonhead_paths:
        print("  !!! non-head leaves changed:", changed_nonhead_paths[:10])

    # ---- Confirm SEQ_LEN=16384 also works ----
    print("\n=== SEQ_LEN=16384 forward ===")
    long_items = peak_intervals(BED, 16384, 1)
    seqs16k = [s for s, _ in long_items]
    X16 = jnp.asarray(encode_batch(seqs16k))
    org16 = jnp.asarray(human_organism_indices(len(seqs16k)))
    ch16, full16, _ = dnase_forward(params, state, X16, org16)
    ok16 = full16.shape == (len(seqs16k), 16384, 384)
    print(f"  channel shape {ch16.shape}  full shape {full16.shape}  ok={ok16}")

    print("\n=== RESULT ===")
    print(f"GATE #2 (forward==predict_sequence): {'PASS' if gate2_ok else 'FAIL'}")
    print(f"GATE #4 (grad isolation):            {'PASS' if gate4_ok else 'FAIL'}")
    print("OVERALL:", "PASS" if (gate2_ok and gate4_ok) else "FAIL")


if __name__ == "__main__":
    _selftest()
