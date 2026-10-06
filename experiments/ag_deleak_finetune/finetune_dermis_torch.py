#!/usr/bin/env python
"""Gated fine-tuning of AlphaGenome's dermal-fibroblast DNase track and its evaluation (Figure 7).

Outputs (outputs/dermis_torch/): cached embeddings, ft_head_dermis.pt, panel_b_fidelity.csv,
panel_d_heldout.csv and ISM npz files.

Run (env: featleak-agtorch unless noted):
    python finetune_dermis_torch.py --stage cache    # trunk embeddings (GPU)
    python finetune_dermis_torch.py --stage gate     # ChromBPNet gate (env featleak-dl)
    python finetune_dermis_torch.py --stage train
    python finetune_dermis_torch.py --stage obs      # observed bigWig windows (env featleak-dl)
    python finetune_dermis_torch.py --stage eval     # panels b, d (GPU)
    python finetune_dermis_torch.py --stage ism --ism-locus chr9:127021880   # GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "dermis_torch"
L = 524288
C0 = L // 2
CENTRAL = 150
CH = 60                                   # CL:0002551 fibroblast of dermis
OCT_SOX = "ATTGTTATGCAAAT"
TRAIN_OFFSETS = [-150, 150, -200, 200]
EVAL_OFFSET = -100
HELDOUT = {"chr8", "chr9"}
EPS = 1e-3


def ft_head_path():
    """Fine-tuned head from --stage train, else the copy shipped in data/figure7/."""
    p = OUT / "ft_head_dermis.pt"
    return p if p.exists() else REPO / "data" / "figure7" / "ft_head_dermis.pt"
EXAMPLE = ("chr8", 40414037)              # held-out ISM example (panel c), same as before


def insert(seq, offsets, mid, motif=OCT_SOX):
    s = list(seq)
    for o in offsets:
        s[mid + o:mid + o + len(motif)] = list(motif)
    return "".join(s)


def enhancers(n_train, n_test):
    pk = pd.read_csv(config.resolve("peaks_dermis"), sep="\t", header=None).rename(
        columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    pk = pk[pk.chrom.isin(["chr" + str(i) for i in range(1, 23)])]
    pk = pk.sort_values("signal", ascending=False)
    pk["mid"] = (pk.start + pk.end) // 2
    train = pk[~pk.chrom.isin(HELDOUT)].head(n_train * 2)
    test = pk[pk.chrom.isin(HELDOUT)].head(n_test * 2)
    return train, test


def fetch(fa, chrom, mid, length=L):
    w0 = mid - length // 2
    q = str(fa[chrom][max(w0, 0):w0 + length]).upper()
    return "N" * (length - len(q)) + q if w0 < 0 else q + "N" * (length - len(q))


class AG:
    """PyTorch AlphaGenome: 1 bp trunk embeddings and the pretrained dermis head slice."""

    def __init__(self):
        import torch
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.heads import predictions_scaling
        self.torch = torch
        self.scaling = predictions_scaling
        self.model = AlphaGenome.from_pretrained(
            str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
        self.model.eval()
        head = self.model.heads["dnase"]
        self.head = head
        self.w0 = head.convs["1"].weight[0, CH].detach().float().clone()     # (1536,)
        self.b0 = head.convs["1"].bias[0, CH].detach().float().clone()
        self.s0 = head.residual_scales["1"][0, CH].detach().float().clone()
        self.tm = head.track_means[0, CH:CH + 1].detach().float().clone()   # (1,)
        self.clip = head.prediction_soft_clip["1"]
        self.lut = np.full(256, -1, dtype=np.int64)
        for i, c in enumerate("ACGT"):
            self.lut[ord(c)] = i

    def onehot(self, q):
        idx = self.lut[np.frombuffer(q.encode(), dtype=np.uint8)]
        x = self.torch.zeros((1, len(q), 4), dtype=self.torch.float32)
        ok = idx >= 0
        x[0, np.where(ok)[0], idx[ok]] = 1
        return x.cuda()

    def embed(self, q, half=CENTRAL):
        """Central (2*half) bp of 1 bp trunk embeddings, float16, (2*half, 1536)."""
        with self.torch.no_grad():
            out = self.model.predict(self.onehot(q), organism_index=0, return_embeddings=True)
        e = out["embeddings_1bp"][0, C0 - half:C0 + half]
        return e.half().cpu()

    def head_pred(self, emb, w, b, s):
        """Experimental-space prediction of the dermis track from embeddings (..., 1536)."""
        F = self.torch.nn.functional
        x = F.softplus(emb.float() @ w + b) * F.softplus(s)
        x = self.clip(x)
        return x * self.tm[0]  # resolution 1, no squashing for DNase


def stage_cache(args):
    import pyfaidx
    torch = __import__("torch")
    OUT.mkdir(parents=True, exist_ok=True)
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    ag = AG()
    train, test = enhancers(args.n_train, args.n_test)

    def build(df, n, offsets):
        E, Ep, rows = [], [], []
        for r in df.itertuples():
            q = fetch(fa, r.chrom, int(r.mid))
            if set(q[C0 - 1000:C0 + 1000]) - set("ACGT"):
                continue
            E.append(ag.embed(q)); Ep.append(ag.embed(insert(q, offsets, C0)))
            rows.append((r.chrom, int(r.start), int(r.end), int(r.mid), float(r.signal)))
            if len(rows) % 25 == 0:
                print(f"  {len(rows)}/{n}", flush=True)
            if len(rows) == n:
                break
        return torch.stack(E), torch.stack(Ep), pd.DataFrame(
            rows, columns=["chrom", "start", "end", "mid", "signal"])

    print("caching train embeddings (4 flank insertions)...", flush=True)
    Etr, Eptr, tr = build(train, args.n_train, TRAIN_OFFSETS)
    torch.save({"E": Etr, "Ep": Eptr}, OUT / "train_emb.pt"); tr.to_csv(OUT / "train.csv", index=False)
    print("caching held-out chr8/9 embeddings (single insertion at -100)...", flush=True)
    Ete, Epte, te = build(test, args.n_test, [EVAL_OFFSET])
    torch.save({"E": Ete, "Ep": Epte}, OUT / "test_emb.pt"); te.to_csv(OUT / "test.csv", index=False)
    print("done", len(tr), len(te))


def stage_gate(args):
    """ChromBPNet dermis log2FC for the same insertions (env: featleak-dl)."""
    import pyfaidx
    import torch
    from bpnetlite.bpnet import BPNet, CountWrapper
    from tangermeme.predict import predict
    from tangermeme.utils import one_hot_encode
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    model = CountWrapper(BPNet.from_chrombpnet(
        str(config.resolve("chrombpnet_dermis", must_exist=True)))).cuda().eval()
    W = 2114

    def lfc(df, offsets):
        base, ins = [], []
        for r in df.itertuples():
            q = str(fa[r.chrom][r.mid - W // 2:r.mid + W // 2]).upper()
            base.append(one_hot_encode(q)); ins.append(one_hot_encode(insert(q, offsets, W // 2)))
        b = predict(model, torch.stack(base).float(), batch_size=64).squeeze().numpy()
        i = predict(model, torch.stack(ins).float(), batch_size=64).squeeze().numpy()
        return (i - b) / np.log(2)                       # log2((counts+1) ratio)

    tr = pd.read_csv(OUT / "train.csv"); te = pd.read_csv(OUT / "test.csv")
    tr["cbp_lfc"] = lfc(tr, TRAIN_OFFSETS); te["cbp_lfc"] = lfc(te, [EVAL_OFFSET])
    tr.to_csv(OUT / "train.csv", index=False); te.to_csv(OUT / "test.csv", index=False)
    gate = (tr.cbp_lfc.abs() < args.gate_threshold).mean()
    print(f"ChromBPNet dermis: train median log2FC={tr.cbp_lfc.median():+.3f}, "
          f"gated (|lfc|<{args.gate_threshold}) = {100 * gate:.0f}%; "
          f"held-out median={te.cbp_lfc.median():+.3f}")


def stage_train(args):
    import torch
    ag = AG()
    d = torch.load(OUT / "train_emb.pt")
    E, Ep = d["E"].cuda(), d["Ep"].cuda()
    tr = pd.read_csv(OUT / "train.csv")
    gate = torch.tensor((tr.cbp_lfc.abs() < args.gate_threshold).values, dtype=torch.float32).cuda()
    with torch.no_grad():
        base = ag.head_pred(E, ag.w0, ag.b0, ag.s0)                   # frozen targets (N, 300)
    w = ag.w0.clone().requires_grad_(); b = ag.b0.clone().requires_grad_()
    s = ag.s0.clone().requires_grad_()
    opt = torch.optim.Adam([w, b, s], lr=args.lr)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(args.steps):
        idx = torch.randint(0, len(E), (args.batch,), generator=g).cuda()
        pB = ag.head_pred(E[idx], w, b, s); pP = ag.head_pred(Ep[idx], w, b, s)
        fid = ((torch.log1p(pB) - torch.log1p(base[idx])) ** 2).mean()
        lfc = torch.log2((pP.sum(1) + EPS) / (pB.sum(1) + EPS))
        gi = gate[idx]
        deleak = (gi * lfc ** 2).sum() / (gi.sum() + 1.0)
        loss = fid + args.lam * deleak
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d} loss={loss.item():.4f} fid={fid.item():.4f} "
                  f"deleak={deleak.item():.4f}", flush=True)
    torch.save({"w": w.detach().cpu(), "b": b.detach().cpu(), "s": s.detach().cpu(),
                "args": vars(args)}, OUT / "ft_head_dermis.pt")
    print("saved", OUT / "ft_head_dermis.pt")


def stage_obs(args):
    """Observed ENCFF302JEV around held-out enhancers (env: featleak-dl, has pyBigWig)."""
    import pyBigWig
    bw = pyBigWig.open(str(config.resolve("bigwig_dermis", must_exist=True)))
    te = pd.read_csv(OUT / "test.csv").head(args.n_fid)
    obs = np.stack([np.nan_to_num(np.array(bw.values(r.chrom, r.mid - FID_HALF, r.mid + FID_HALF)))
                    for r in te.itertuples()])
    np.save(OUT / "obs_test_windows.npy", obs.astype(np.float32))
    print("saved observed windows", obs.shape)


FID_HALF = 8192


def stage_eval(args):
    import pyfaidx
    import torch
    from scipy import stats
    ag = AG()
    ft = torch.load(ft_head_path())
    wf, bf, sf = ft["w"].cuda(), ft["b"].cuda(), ft["s"].cuda()

    # panel d: single OCT-SOX at -100 on held-out chr8/9 enhancers
    d = torch.load(OUT / "test_emb.pt")
    E, Ep = d["E"].cuda(), d["Ep"].cuda()
    with torch.no_grad():
        pre = torch.log2((ag.head_pred(Ep, ag.w0, ag.b0, ag.s0).sum(1) + EPS)
                         / (ag.head_pred(E, ag.w0, ag.b0, ag.s0).sum(1) + EPS)).cpu().numpy()
        post = torch.log2((ag.head_pred(Ep, wf, bf, sf).sum(1) + EPS)
                          / (ag.head_pred(E, wf, bf, sf).sum(1) + EPS)).cpu().numpy()
    te = pd.read_csv(OUT / "test.csv")
    te["ag_pre"], te["ag_ft"] = pre, post
    te.to_csv(OUT / "panel_d_heldout.csv", index=False)
    print(f"panel d (n={len(te)}): AlphaGenome median={np.median(pre):+.3f} "
          f"(p<0 {stats.wilcoxon(pre, alternative='less').pvalue:.2g}); fine-tuned median="
          f"{np.median(post):+.3f}; ChromBPNet median={te.cbp_lfc.median():+.3f}; "
          f"paired p={stats.wilcoxon(pre, post).pvalue:.2g}")

    # panel b: fidelity vs observed ENCFF302JEV, held-out chr8/9 16 kb windows
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    obs_all = np.load(OUT / "obs_test_windows.npy")          # from --stage obs
    half = FID_HALF
    r_pre, r_ft = [], []
    for j, r in enumerate(te.head(args.n_fid).itertuples()):
        q = fetch(fa, r.chrom, int(r.mid))
        e = ag.embed(q, half=half).cuda()
        obs = obs_all[j]
        with torch.no_grad():
            a = ag.head_pred(e, ag.w0, ag.b0, ag.s0).cpu().numpy()
            f = ag.head_pred(e, wf, bf, sf).cpu().numpy()
        r_pre.append(np.corrcoef(a, obs)[0, 1]); r_ft.append(np.corrcoef(f, obs)[0, 1])
    fid = pd.DataFrame({"pearson_pre": r_pre, "pearson_ft": r_ft})
    fid.to_csv(OUT / "panel_b_fidelity.csv", index=False)
    print(f"panel b: mean Pearson r pre={np.nanmean(r_pre):.3f} ft={np.nanmean(r_ft):.3f} "
          f"(n={len(fid)} held-out 16 kb windows)")

    # legacy ISM at an endogenous chr8 site; NOT the figure's panel c (that is --stage ism),
    # so it is written to its own file and never overwrites panel_c_ism.npz
    chrom, pos = EXAMPLE
    q = fetch(fa, chrom, pos)
    halfI = 16
    muts, where = [], []
    for i in range(C0 - halfI, C0 + halfI):
        for k, bse in enumerate("ACGT"):
            muts.append(q[:i] + bse + q[i + 1:]); where.append((i - (C0 - halfI), k))
    vals_pre = np.zeros((2 * halfI, 4)); vals_ft = np.zeros((2 * halfI, 4))
    for m, (i, k) in zip(muts, where):
        e = ag.embed(m, half=1000).cuda()
        with torch.no_grad():
            vals_pre[i, k] = ag.head_pred(e, ag.w0, ag.b0, ag.s0).mean().item()
            vals_ft[i, k] = ag.head_pred(e, wf, bf, sf).mean().item()
    np.savez(OUT / "eval_ism_chr8_legacy.npz", pre=vals_pre, ft=vals_ft,
             ref=q[C0 - halfI:C0 + halfI], start=pos - halfI, chrom=chrom)
    print("legacy chr8 ISM saved (panel c comes from --stage ism)")


def stage_ism(args):
    """Panel c ISM, before vs after, around an OCT-SOX INSERTED at -100 bp of a held-out
    enhancer (the same single insertion as panel d)."""
    import pyfaidx
    import torch
    ag = AG()
    ft = torch.load(ft_head_path())
    wf, bf, sf = ft["w"].cuda(), ft["b"].cuda(), ft["s"].cuda()
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    chrom, mid = args.ism_locus.split(":"); mid = int(mid)
    q = insert(fetch(fa, chrom, mid), [EVAL_OFFSET], C0)
    centre = C0 + EVAL_OFFSET + len(OCT_SOX) // 2          # centre of the inserted motif
    halfI = 16
    vals_pre = np.zeros((2 * halfI, 4)); vals_ft = np.zeros((2 * halfI, 4))
    for i in range(2 * halfI):
        p = centre - halfI + i
        for k, bse in enumerate("ACGT"):
            e = ag.embed(q[:p] + bse + q[p + 1:], half=1000).cuda()
            with torch.no_grad():
                vals_pre[i, k] = ag.head_pred(e, ag.w0, ag.b0, ag.s0).mean().item()
                vals_ft[i, k] = ag.head_pred(e, wf, bf, sf).mean().item()
    g0 = mid - C0 + centre - halfI                           # genomic start of the window
    np.savez(OUT / (args.ism_out or "panel_c_ism.npz"), pre=vals_pre, ft=vals_ft,
             ref=q[centre - halfI:centre + halfI], start=g0, chrom=chrom,
             inserted=True, motif_start=halfI - len(OCT_SOX) // 2)
    print(f"panel c ISM saved ({chrom}:{mid}, OCT-SOX inserted at {EVAL_OFFSET} bp)")


def octsox_sites(seq, max_mm):
    """OCT-SOX consensus matches in `seq` with <= max_mm mismatches, either strand.

    Returns (start position in `seq`, mismatches, strand, matched 14-mer).
    """
    rc = str.maketrans("ACGT", "TGCA")
    queries = [(OCT_SOX, "+"), (OCT_SOX.translate(rc)[::-1], "-")]
    n = len(OCT_SOX)
    hits = []
    for p in range(len(seq) - n + 1):
        sub = seq[p:p + n]
        if set(sub) - set("ACGT"):
            continue
        for q, strand in queries:
            mm = sum(a != b for a, b in zip(sub, q))
            if mm <= max_mm:
                hits.append((p, mm, strand, sub))
                break
    return hits


def endogenous_candidates(fa, max_mm):
    """Near-consensus endogenous OCT-SOX sites inside held-out (chr8/chr9) dermis peaks.

    Returns a DataFrame sorted by mismatches then peak signal, one row per site, with
    `mid` = the centre of the motif (used as the centre of the 524 kb model window).
    """
    pk = pd.read_csv(config.resolve("peaks_dermis"), sep="\t", header=None).rename(
        columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    pk = pk[pk.chrom.isin(sorted(HELDOUT))].sort_values("signal", ascending=False)
    rows = []
    for r in pk.itertuples():
        s = str(fa[r.chrom][int(r.start):int(r.end)]).upper()
        for p, mm, strand, sub in octsox_sites(s, max_mm):
            rows.append(dict(chrom=r.chrom, mstart=int(r.start) + p, mismatches=mm,
                             strand=strand, site=sub, signal=float(r.signal)))
    df = pd.DataFrame(rows)
    df["mid"] = df.mstart + len(OCT_SOX) // 2
    return df.sort_values(["mismatches", "signal"], ascending=[True, False]).reset_index(drop=True)


def fimo_candidates(chroms):
    """Figure 1's endogenous OCT-SOX sites, restricted to `chroms`.

    Same selection as figures/figure1/run_chrombpnet_logfc.py: FIMO OCT-SOX hits
    200-500 bp from a dermal-fibroblast DNase peak summit (ENCSR000EPO), i.e. in the
    peak flank, so Fig 1 and Fig 7 score the same kind of site on the same task.
    """
    sys.path.insert(0, str(REPO / "figures" / "figure2"))
    from cbpnet_helpers import annotate_with_peak_distances

    h = annotate_with_peak_distances(config.resolve("fimo_octsox", must_exist=True),
                                     config.resolve("peaks_dermis", must_exist=True))
    h = h[h.distance.between(200, 500) & h.sequence_name.isin(chroms)]
    df = pd.DataFrame({
        "chrom": h.sequence_name.values,
        "mid": ((h.start.values + h.stop.values) // 2).astype(int),
        "strand": h.strand.values, "site": h.matched_sequence.values,
        "fimo_score": h.score.values, "distance": h.distance.values,
    })
    df["mstart"] = df.mid - len(OCT_SOX) // 2
    df["mismatches"] = -1                      # not a consensus-mismatch selection
    df["signal"] = np.nan
    return df.sort_values("fimo_score", ascending=False).reset_index(drop=True)


def dinuc_shuffle_window(seq, start, end, seed):
    """Dinucleotide-shuffle seq[start:end] (Fig 1's ablation), returning the new string."""
    import torch
    from tangermeme.ersatz import dinucleotide_shuffle
    from tangermeme.utils import characters, one_hot_encode

    x = one_hot_encode(seq[start:end]).unsqueeze(0).float()
    y = dinucleotide_shuffle(x, n=1, random_state=seed)[0, 0]
    return seq[:start] + characters(y) + seq[end:]


def stage_endo_screen(args):
    """Rank held-out enhancers carrying an endogenous OCT-SOX by the leakage signature.

    Each site is placed at the centre of the 524 kb window and ablated by scrambling its
    14 bp (composition preserved); the original and ablated sequence are scored with both
    heads from the same trunk embedding. The motif effect is
      effect = log2(signal(original) / signal(ablated)) over the central 300 bp,
    so a NEGATIVE effect means AlphaGenome reads the OCT-SOX as repressive, which is the
    leakage signature; after fine-tuning the effect should be ~0. Two forward passes per
    site, ~60x cheaper than ISM, so this is what picks the panel c locus.
    """
    import pyfaidx
    import torch
    ag = AG()
    ft = torch.load(ft_head_path())
    wf, bf, sf = ft["w"].cuda(), ft["b"].cuda(), ft["s"].cuda()
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    rng = np.random.default_rng(args.seed)

    if args.endo_source == "fimo":
        cand = fimo_candidates(sorted(HELDOUT))
        what = "Fig 1 FIMO OCT-SOX hits 200-500 bp from a dermis peak summit"
    else:
        cand = endogenous_candidates(fa, args.endo_max_mm)
        what = f"consensus matches (<= {args.endo_max_mm} mismatches) inside held-out peaks"
    print(f"{len(cand)} endogenous OCT-SOX sites: {what}; scoring "
          f"{min(args.endo_n_cand, len(cand) - args.endo_skip)}", flush=True)

    rows = []
    for r in cand.iloc[args.endo_skip:args.endo_skip + args.endo_n_cand].itertuples():
        q = fetch(fa, r.chrom, int(r.mid))
        p = C0 - len(OCT_SOX) // 2                       # motif sits at the window centre
        if set(q[p:p + len(OCT_SOX)]) - set("ACGT"):
            print(f"  skip {r.chrom}:{r.mid} (non-ACGT)", flush=True)
            continue
        if args.endo_source == "fimo":                   # Fig 1 ablation: dinuc shuffle
            qs = dinuc_shuffle_window(q, p, p + len(OCT_SOX), args.seed)
        else:
            qs = q[:p] + "".join(rng.permutation(list(q[p:p + len(OCT_SOX)]))) + \
                q[p + len(OCT_SOX):]
        with torch.no_grad():
            e0 = ag.embed(q).cuda(); e1 = ag.embed(qs).cuda()
            pre0 = ag.head_pred(e0, ag.w0, ag.b0, ag.s0).sum().item()
            pre1 = ag.head_pred(e1, ag.w0, ag.b0, ag.s0).sum().item()
            ft0 = ag.head_pred(e0, wf, bf, sf).sum().item()
            ft1 = ag.head_pred(e1, wf, bf, sf).sum().item()
        rows.append(dict(chrom=r.chrom, mid=int(r.mid), mstart=int(r.mstart),
                         signal=float(r.signal), mismatches=int(r.mismatches),
                         strand=r.strand, site=r.site, seq=q[p:p + len(OCT_SOX)],
                         effect_pre=np.log2((pre0 + EPS) / (pre1 + EPS)),
                         effect_ft=np.log2((ft0 + EPS) / (ft1 + EPS))))
        print(f"  {len(rows):3d} {r.chrom}:{r.mid} mm={r.mismatches} {r.strand} {r.site} "
              f"effect_pre={rows[-1]['effect_pre']:+.3f} "
              f"effect_ft={rows[-1]['effect_ft']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df["recovered"] = df.effect_ft - df.effect_pre    # how much repression fine-tuning removes
    df = df.sort_values("effect_pre")                 # most repressive first
    out = OUT / (args.endo_out or "endo_sites.csv")
    df.to_csv(out, index=False)
    print(f"\nscreened {len(df)} endogenous OCT-SOX sites -> {out}")
    print(df.head(args.endo_top).to_string(index=False))


def stage_endo_ism(args):
    """Panel c ISM at an ENDOGENOUS OCT-SOX in a held-out enhancer, before vs after.

    --endo-site chrom:mid:offset (offset = motif start relative to the window centre, as
    reported by --stage endo_screen); with no --endo-site, the top row of endo_sites.csv.
    """
    import pyfaidx
    import torch
    ag = AG()
    ft = torch.load(ft_head_path())
    wf, bf, sf = ft["w"].cuda(), ft["b"].cuda(), ft["s"].cuda()
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    if args.endo_site:
        chrom, mid, off = args.endo_site.split(":")
        mid, off = int(mid), int(off)
    else:                       # most repressive site from the ablation screen
        top = pd.read_csv(OUT / (args.endo_out or "endo_sites.csv")).iloc[0]
        chrom, mid, off = top.chrom, int(top.mid), -(len(OCT_SOX) // 2)
    q = fetch(fa, chrom, mid)
    centre = C0 + off + len(OCT_SOX) // 2                  # centre of the endogenous motif
    halfI = 16
    vals_pre = np.zeros((2 * halfI, 4)); vals_ft = np.zeros((2 * halfI, 4))
    # readout = the central 300 bp the head was trained on (same window as the ablation
    # screen and panel d); averaging over a wider window dilutes a local footprint
    for i in range(2 * halfI):
        p = centre - halfI + i
        for k, bse in enumerate("ACGT"):
            e = ag.embed(q[:p] + bse + q[p + 1:], half=CENTRAL).cuda()
            with torch.no_grad():
                vals_pre[i, k] = ag.head_pred(e, ag.w0, ag.b0, ag.s0).sum().item()
                vals_ft[i, k] = ag.head_pred(e, wf, bf, sf).sum().item()
    g0 = mid - C0 + centre - halfI                          # genomic start of the window
    np.savez(OUT / (args.ism_out or "panel_c_ism_endogenous.npz"), pre=vals_pre, ft=vals_ft,
             ref=q[centre - halfI:centre + halfI], start=g0, chrom=chrom,
             endogenous=True, motif_start=halfI - len(OCT_SOX) // 2)
    print(f"endogenous panel c ISM saved ({chrom}:{mid}, OCT-SOX at {off:+d} bp)")


def stage_endo_track(args):
    """Predicted DNase profile at an endogenous OCT-SOX with the motif intact vs ablated.

    The same ablation as the screen and as Fig 1 (dinucleotide shuffle of the 14 bp),
    shown as tracks: before fine-tuning AlphaGenome predicts LESS accessibility with the
    motif present (leakage); after fine-tuning the two profiles coincide.
    --endo-site chrom:mid:offset, else the most repressive row of the screen CSV.
    """
    import pyfaidx
    import torch
    ag = AG()
    ft = torch.load(ft_head_path())
    wf, bf, sf = ft["w"].cuda(), ft["b"].cuda(), ft["s"].cuda()
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    if args.endo_site:
        chrom, mid, off = args.endo_site.split(":")
        mid, off = int(mid), int(off)
    else:
        top = pd.read_csv(OUT / (args.endo_out or "endo_sites_fimo.csv")).iloc[0]
        chrom, mid, off = top.chrom, int(top.mid), -(len(OCT_SOX) // 2)
    q = fetch(fa, chrom, mid)
    p = C0 + off
    qs = dinuc_shuffle_window(q, p, p + len(OCT_SOX), args.seed)
    prof = {}
    with torch.no_grad():
        for tag, seq in (("intact", q), ("ablated", qs)):
            e = ag.embed(seq, half=CENTRAL).cuda()
            prof[f"pre_{tag}"] = ag.head_pred(e, ag.w0, ag.b0, ag.s0).cpu().numpy()
            prof[f"ft_{tag}"] = ag.head_pred(e, wf, bf, sf).cpu().numpy()
    eff_pre = np.log2((prof["pre_intact"].sum() + EPS) / (prof["pre_ablated"].sum() + EPS))
    eff_ft = np.log2((prof["ft_intact"].sum() + EPS) / (prof["ft_ablated"].sum() + EPS))
    np.savez(OUT / (args.ism_out or f"endo_track_{chrom}_{mid}.npz"), chrom=chrom, mid=mid,
             offset=off, motif=q[p:p + len(OCT_SOX)], effect_pre=eff_pre, effect_ft=eff_ft,
             **prof)
    print(f"{chrom}:{mid} motif {q[p:p + len(OCT_SOX)]}  effect {eff_pre:+.3f} -> {eff_ft:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "gate", "train", "obs", "eval", "ism",
                                        "endo_screen", "endo_ism", "endo_track"],
                    required=True)
    ap.add_argument("--ism-locus", default="chr9:127021880",
                    help="held-out enhancer centre for panel c (chrom:mid)")
    ap.add_argument("--ism-out", default=None, help="output npz name (default panel_c_ism.npz)")
    ap.add_argument("--endo-site", default=None,
                    help="endo_ism locus as chrom:mid:offset (default: top endo_sites.csv row)")
    ap.add_argument("--endo-source", choices=["consensus", "fimo"], default="consensus",
                    help="candidate sites: consensus matches in peaks, or Fig 1's FIMO hits")
    ap.add_argument("--endo-max-mm", type=int, default=3,
                    help="mismatches allowed against the OCT-SOX consensus (endo_screen)")
    ap.add_argument("--endo-n-cand", type=int, default=120,
                    help="number of endogenous sites to score (endo_screen)")
    ap.add_argument("--endo-skip", type=int, default=0,
                    help="skip this many matched sites before scoring (resume a screen)")
    ap.add_argument("--endo-out", default=None, help="screen output CSV name")
    ap.add_argument("--endo-top", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--n-fid", type=int, default=120)
    ap.add_argument("--gate-threshold", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"cache": stage_cache, "gate": stage_gate, "train": stage_train,
     "obs": stage_obs, "eval": stage_eval, "ism": stage_ism,
     "endo_screen": stage_endo_screen, "endo_ism": stage_endo_ism,
     "endo_track": stage_endo_track}[args.stage](args)


if __name__ == "__main__":
    main()
