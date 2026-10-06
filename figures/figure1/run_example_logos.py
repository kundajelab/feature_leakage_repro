#!/usr/bin/env python
"""Figure 1 E-H logos: labelled DeepLIFT examples of positive / negative motif hits.

For each (motif, Beluga DNase task, expected sign) pair, strongest-quartile FIMO
hits are screened with span-summed DeepLIFT (fast exact implementation from
run_beluga_grid.py). The top candidates are re-attributed at base resolution
(tangermeme DeepLIFT/SHAP, 20 dinucleotide-shuffled references, single task) and
ranked by how cleanly the motif dominates the local attribution:

    cleanliness = |attribution in motif span| / sum |attribution| in +/-40 bp

Stage 1 (--stage screen) writes candidate attributions and a contact sheet PNG
per pair for visual inspection. Stage 2 (--stage final --pick PAIR=RANK ...)
renders the chosen examples as editable PDFs labelled with locus + cell type.

Run (env: featleak-dl):
    python run_example_logos.py --stage screen --device cuda:0
    python run_example_logos.py --stage paired --n-cand 60 --device cuda:0
    python run_example_logos.py --stage final_paired --pick OCT-SOX=chr15:70407064 \
        AP-1=chr1:145974199 HNF4=chr5:169108419 CTCF=chr1:174983404
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import logomaker  # noqa: E402
from tangermeme.deep_lift_shap import deep_lift_shap  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as _fm  # noqa: E402
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    try:   # Arial-metric font, so text stays editable and matches Arial/Helvetica
        _fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{_f}")
    except (FileNotFoundError, RuntimeError):
        pass
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
    "axes.linewidth": 0.6})
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from common.models import load_beluga, load_beluga_task_names  # noqa: E402
import run_beluga_grid as grid  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "example_logos"
FLANK = 40            # bp each side of the motif used for the cleanliness score
SHOW = 10             # bp each side of the motif shown in the final logo

# pair -> (motif, absolute Beluga task index, display label, expected sign)
PAIRS = {
    "OCTSOX_pos": ("OCT-SOX", 54, "H1-hESC", +1),
    "OCTSOX_neg": ("OCT-SOX", 1048, "IMR-90 fibroblast", -1),
    "AP1_pos": ("AP-1", 1048, "IMR-90 fibroblast", +1),
    "AP1_neg": ("AP-1", 1184, "CD19+ B cells", -1),
    "HNF4_pos": ("HNF4", 1778, "Small intestine", +1),
    "HNF4_neg": ("HNF4", 1527, "Fetal brain", -1),
    "CTCF_pos": ("CTCF", 54, "H1-hESC", +1),
    "CTCF_fib": ("CTCF", 1048, "IMR-90 fibroblast", +1),
}
COLORS = {"OCT-SOX": "#2166ac", "AP-1": "#b2182b", "HNF4": "#1b7837", "CTCF": "#6a3d9a"}


def q4_hits(motif, n, seed=1):
    df = pd.read_csv(config.resolve(grid.MOTIFS[motif], must_exist=True), sep="\t",
                     usecols=["sequence_name", "start", "stop", "strand", "score",
                              "matched_sequence"])
    df = df[df.sequence_name.isin(grid.CHROMS)].rename(columns={"sequence_name": "chrom"})
    q75 = df.score.quantile(0.75)
    df = df[df.score > q75].sample(n, random_state=seed).reset_index(drop=True)
    df["quartile"] = 3
    return df


def screen(args):
    dev = args.device
    beluga = load_beluga(device=dev)
    names = load_beluga_task_names()
    task_idx, _ = grid.dnase_tasks()
    model = grid.BelugaLogits(beluga, task_idx).to(dev).eval()
    pos_in_dnase = {int(t): i for i, t in enumerate(task_idx)}

    cache = {}
    for pair, (motif, tabs, label, sign) in PAIRS.items():
        if motif not in cache:
            hits = q4_hits(motif, args.n_pool)
            X, spans, keep = grid.one_hot_windows(hits, config.resolve("hg38_fasta"))
            hits = hits[keep].reset_index(drop=True)
            span_eff, _ = grid.deeplift_effect_fast(model, X, spans, dev, n_refs=10)
            cache[motif] = (hits, X, spans, span_eff)
        hits, X, spans, span_eff = cache[motif]
        t = pos_in_dnase[tabs]
        order = np.argsort(sign * span_eff[:, t])[::-1][:args.n_cand].copy()

        # base-resolution DeepLIFT for this single task
        single = grid.BelugaLogits(beluga, [tabs]).to(dev).eval()
        attr = deep_lift_shap(single, X[order], target=0, n_shuffles=20,
                              random_state=0, device=dev, batch_size=64,
                              warning_threshold=1e9).numpy()          # (C, 4, L)
        rows = []
        for k, i in enumerate(order):
            a, b = spans[i]
            per_pos = attr[k].sum(0)
            loc = per_pos[a - FLANK:b + FLANK]
            span_sum = per_pos[a:b].sum()
            clean = abs(span_sum) / (np.abs(loc).sum() + 1e-9)
            consist = (np.sign(per_pos[a:b]) == sign).mean()
            rows.append(dict(rank_screen=k, hit=int(i), chrom=hits.chrom[i],
                             start=int(hits.start[i]), stop=int(hits.stop[i]),
                             strand=hits.strand[i], score=float(hits.score[i]),
                             matched=hits.matched_sequence[i], span_attr=float(span_sum),
                             clean=float(clean), sign_consistency=float(consist)))
        cand = pd.DataFrame(rows)
        cand["good"] = cand.clean * cand.sign_consistency * np.abs(cand.span_attr)
        cand = cand.sort_values("good", ascending=False).reset_index(drop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        cand.to_csv(OUT / f"candidates_{pair}.tsv", sep="\t", index=False)
        amap = {int(i): attr[k] for k, i in enumerate(order)}
        np.savez_compressed(OUT / f"attr_{pair}.npz",
                            attr=np.stack([amap[h] for h in cand.hit]),
                            spans=spans[cand.hit.values],
                            onehot=X[cand.hit.values].numpy())
        contact_sheet(pair, cand, np.stack([amap[h] for h in cand.hit]),
                      spans[cand.hit.values], names[tabs], sign)
        print(f"{pair}: task {names[tabs]}; best clean={cand.clean[0]:.2f} "
              f"span={cand.span_attr[0]:.3f}", flush=True)


def to_df(attr, a, b, strand):
    """(4, L) attribution -> logomaker frame over motif +/- SHOW, motif orientation."""
    m = attr[:, a - SHOW:b + SHOW]
    if strand == "-":
        m = m[::-1, ::-1]          # reverse complement (ACGT -> TGCA)
    return pd.DataFrame(m.T, columns=list("ACGT"))


def contact_sheet(pair, cand, attr, spans, task_name, sign, n=12):
    n = min(n, len(cand))
    fig, axes = plt.subplots(n, 1, figsize=(4.5, 0.9 * n))
    for k in range(n):
        a, b = spans[k]
        df = to_df(attr[k], a - 20, b + 20, cand.strand[k])
        logomaker.Logo(df, ax=axes[k], color_scheme="classic")
        axes[k].axhline(0, color="0.6", lw=0.4)
        axes[k].axvspan(SHOW + 20 - 0.5, SHOW + 20 + (b - a) - 0.5, color="gold", alpha=0.15)
        axes[k].set_xticks([])
        axes[k].set_title(f"#{k} {cand.chrom[k]}:{cand.start[k]}-{cand.stop[k]}({cand.strand[k]}) "
                          f"clean={cand.clean[k]:.2f} span={cand.span_attr[k]:.3f}", fontsize=6)
    fig.suptitle(f"{pair}: {task_name}", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"contact_{pair}.png", dpi=150)
    plt.close(fig)


def final(args):
    picks = dict(p.split("=") for p in args.pick)
    names = load_beluga_task_names()
    for pair, rank in picks.items():
        motif, tabs, label, sign = PAIRS[pair]
        cand = pd.read_csv(OUT / f"candidates_{pair}.tsv", sep="\t")
        d = np.load(OUT / f"attr_{pair}.npz")
        k = int(rank)
        a, b = d["spans"][k]
        r = cand.iloc[k]
        df = to_df(d["attr"][k], a, b, r.strand)
        # coordinates of the displayed window (1-based, inclusive)
        g0, g1 = int(r.start) - SHOW, int(r.stop) + SHOW
        fig, ax = plt.subplots(figsize=(2.0, 0.85))
        logomaker.Logo(df, ax=ax, color_scheme="classic")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        lo, hi = ax.get_ylim()
        m = max(abs(lo), abs(hi))
        ax.set_ylim(-m if lo < 0 else lo, hi if hi > 0 else m * 0.1)
        ax.set_ylabel("DeepLIFT\n(Δ logit)", fontsize=6)
        ax.set_title(f"{motif} in {label}\n{r.chrom}:{g0:,}-{g1:,} ({r.strand})",
                     fontsize=6, color=COLORS[motif], pad=2)
        fig.tight_layout(pad=0.2)
        fig.savefig(OUT / f"fig1_logo_{pair}.pdf")
        fig.savefig(OUT / f"fig1_logo_{pair}.png", dpi=300)
        plt.close(fig)
        print(f"{pair}: {names[tabs]} {r.chrom}:{g0}-{g1} span={r.span_attr:.3f} clean={r.clean:.2f}")

# motif -> (pair key for row 1, pair key for row 2): same locus shown in both cell types
PAIRED = {"OCT-SOX": ("OCTSOX_pos", "OCTSOX_neg"), "AP-1": ("AP1_pos", "AP1_neg"),
          "HNF4": ("HNF4_pos", "HNF4_neg"), "CTCF": ("CTCF_pos", "CTCF_fib")}


def paired(args):
    """Screen for single loci that are clean in BOTH cell types of a motif pair."""
    dev = args.device
    beluga = load_beluga(device=dev)
    task_idx, _ = grid.dnase_tasks()
    model = grid.BelugaLogits(beluga, task_idx).to(dev).eval()
    col = {int(t): i for i, t in enumerate(task_idx)}
    for motif, (k1, k2) in PAIRED.items():
        n_pool = args.n_pool * (4 if motif == "OCT-SOX" else 1)
        hits = q4_hits(motif, n_pool, seed=2)
        X, spans, keep = grid.one_hot_windows(hits, config.resolve("hg38_fasta"))
        hits = hits[keep].reset_index(drop=True)
        eff, _ = grid.deeplift_effect_fast(model, X, spans, dev, n_refs=10)
        t1, t2 = PAIRS[k1][1], PAIRS[k2][1]
        s1, s2 = PAIRS[k1][3], PAIRS[k2][3]
        e1, e2 = s1 * eff[:, col[t1]], s2 * eff[:, col[t2]]
        # rank by the weaker of the two (scaled) sign-correct effects
        score = np.minimum(e1 / np.abs(e1).max(), e2 / np.abs(e2).max())
        order = np.argsort(score)[::-1][:args.n_cand].copy()
        attrs = {}
        for key, t in ((k1, t1), (k2, t2)):
            single = grid.BelugaLogits(beluga, [t]).to(dev).eval()
            attrs[key] = deep_lift_shap(single, X[order], target=0, n_shuffles=20,
                                        random_state=0, device=dev, batch_size=64,
                                        warning_threshold=1e9).numpy()
        rows = []
        for j, i in enumerate(order):
            a, b = spans[i]
            r = dict(hit=int(i), chrom=hits.chrom[i], start=int(hits.start[i]),
                     stop=int(hits.stop[i]), strand=hits.strand[i],
                     score=float(hits.score[i]), matched=hits.matched_sequence[i])
            for key, sgn in ((k1, s1), (k2, s2)):
                pp = attrs[key][j].sum(0)
                span = pp[a:b].sum()
                r[f"span_{key}"] = float(span)
                r[f"clean_{key}"] = float(abs(span) / (np.abs(pp[a - FLANK:b + FLANK]).sum() + 1e-9))
                r[f"ok_{key}"] = bool(np.sign(span) == sgn)
            rows.append(r)
        cand = pd.DataFrame(rows)
        cand["good"] = (np.minimum(cand[f"clean_{k1}"], cand[f"clean_{k2}"])
                        * cand[f"ok_{k1}"] * cand[f"ok_{k2}"])
        sort = cand.sort_values("good", ascending=False).index.values
        cand = cand.loc[sort].reset_index(drop=True)
        tag = motif.replace("-", "")
        cand.to_csv(OUT / f"paired_{tag}.tsv", sep="\t", index=False)
        np.savez_compressed(OUT / f"paired_{tag}.npz",
                            **{k: attrs[k][sort] for k in (k1, k2)},
                            spans=spans[order][sort], onehot=X[order][sort].numpy())
        # contact sheet: two columns (cell type 1 | cell type 2), shared y per row
        n = min(10, len(cand))
        fig, axes = plt.subplots(n, 2, figsize=(8, 0.9 * n))
        for r in range(n):
            a, b = spans[order][sort][r]
            for c, key in enumerate((k1, k2)):
                df = to_df(attrs[key][sort][r], a - 15, b + 15, cand.strand[r])
                ax = axes[r, c]
                logomaker.Logo(df, ax=ax, color_scheme="classic")
                ax.axhline(0, color="0.6", lw=0.4); ax.set_xticks([])
                ax.set_title(f"#{r} {cand.chrom[r]}:{cand.start[r]} {PAIRS[key][2]} "
                             f"clean={cand[f'clean_{key}'][r]:.2f} "
                             f"span={cand[f'span_{key}'][r]:.2f}", fontsize=5.5)
        fig.tight_layout()
        fig.savefig(OUT / f"contact_paired_{tag}.png", dpi=140)
        plt.close(fig)
        print(motif, cand.head(3)[["chrom", "start", f"clean_{k1}", f"clean_{k2}",
                                    f"span_{k1}", f"span_{k2}"]].to_string(), flush=True)

def final_paired(args):
    """Render the chosen paired loci: one editable PDF per (motif, cell type)."""
    picks = dict(p.split("=") for p in args.pick)
    for motif, rank in picks.items():
        k1, k2 = PAIRED[motif]
        tag = motif.replace("-", "")
        cand = pd.read_csv(OUT / f"paired_{tag}.tsv", sep="\t")
        d = np.load(OUT / f"paired_{tag}.npz")
        if ":" in rank:          # chrom:start of the FIMO hit (robust to re-ranking)
            chrom, start = rank.split(":")
            hit = np.where((cand.chrom == chrom) & (cand.start == int(start)))[0]
            assert len(hit) == 1, f"{rank} not among the {motif} candidates"
            k = int(hit[0])
        else:
            k = int(rank)
        r = cand.iloc[k]
        a, b = d["spans"][k]
        g0, g1 = int(r.start) - SHOW, int(r.stop) + SHOW
        for key in (k1, k2):
            _, _, label, sign = PAIRS[key]
            df = to_df(d[key][k], a, b, r.strand)
            fig, ax = plt.subplots(figsize=(2.2, 0.95))
            logomaker.Logo(df, ax=ax, color_scheme="classic")
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_xticks([])
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)
            lo, hi = ax.get_ylim()
            if sign > 0:
                ax.set_ylim(min(lo, -0.08 * hi), hi)
            else:
                ax.set_ylim(lo, max(hi, -0.08 * lo))
            ax.set_ylabel("DeepLIFT\n(\u0394 logit)", fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{motif} | {label}\n{r.chrom}:{g0:,}-{g1:,} ({r.strand})",
                         fontsize=6, color=COLORS[motif], pad=2)
            fig.tight_layout(pad=0.2)
            fig.savefig(OUT / f"fig1_logo_{key}.pdf")
            fig.savefig(OUT / f"fig1_logo_{key}.png", dpi=300)
            plt.close(fig)
            print(f"{key}: {label} {r.chrom}:{g0}-{g1}({r.strand}) "
                  f"span={r[f'span_{key}']:.3f} clean={r[f'clean_{key}']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "paired", "final", "final_paired"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-pool", type=int, default=3000)
    ap.add_argument("--n-cand", type=int, default=40)
    ap.add_argument("--pick", nargs="*", default=[])
    args = ap.parse_args()
    {"screen": screen, "paired": paired, "final": final,
     "final_paired": final_paired}[args.stage](args)


if __name__ == "__main__":
    main()
