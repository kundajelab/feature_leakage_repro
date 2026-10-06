#!/usr/bin/env python
"""Screen the most repressive dermis OCT-SOX sites with AlphaGenome ISM to pick the Figure 1A example.

Outputs (outputs/panel_a/): ag_dermis_ism_screen.tsv/.npz and ag_dermis_ism_contact.png.

Run (env: featleak-agtorch):
    python screen_ag_dermis_ism.py --top 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfaidx  # noqa: E402
import torch  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "panel_a"
L, C0, ISM_BP, SCORE_BP, CH, EPS = 2048, 1024, 100, 2048, 60, 1e-6


def main():
    from alphagenome_pytorch import AlphaGenome

    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    sites = pd.read_csv(Path(__file__).parent / "outputs" / "fig1_octsox_sites_dermis.csv").head(400)
    lfc = np.load(Path(__file__).parent / "outputs" / "fig1_alphagenome_octsox_logfc_dermis_2kb.npy")
    sites["lfc2kb"] = lfc
    cand = sites.sort_values("lfc2kb").head(args.top)

    model = AlphaGenome.from_pretrained(
        str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))
    lut = np.full(256, -1, dtype=np.int64)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i

    def batch_score(seqs):
        out = []
        for i in range(0, len(seqs), 32):
            idx = np.stack([lut[np.frombuffer(q.encode(), dtype=np.uint8)] for q in seqs[i:i + 32]])
            x = torch.zeros((len(idx), L, 4)); b, p = np.where(idx >= 0)
            x[b, p, idx[b, p]] = 1
            with torch.no_grad():
                o = model.predict(x.cuda(), organism_index=0, heads=("dnase",))
            out.append(o["dnase"][1][:, :, CH].float().sum(1).cpu().numpy())
        return np.concatenate(out)

    rows, mats = [], {}
    for r in cand.itertuples():
        mid = (int(r.start) + int(r.end)) // 2
        q = str(fa[r.chromosome][mid - L // 2:mid + L // 2]).upper()
        a = C0 - ISM_BP // 2
        muts, where = [], []
        for pos in range(a, a + ISM_BP):
            for k, b in enumerate("ACGT"):
                if b != q[pos]:
                    muts.append(q[:pos] + b + q[pos + 1:]); where.append((pos - a, k))
        ref = batch_score([q])[0]
        sc = batch_score(muts)
        ism = np.zeros((ISM_BP, 4))
        for (i, k), s in zip(where, sc):
            ism[i, k] = np.log2((s + EPS) / (ref + EPS))
        c = -ism.sum(1) / 3
        M = np.zeros((ISM_BP, 4)); M[np.arange(ISM_BP), ["ACGT".index(x) for x in q[a:a + ISM_BP]]] = c
        ms, me = int(r.start) - 1 - (mid - L // 2) - a, int(r.end) - (mid - L // 2) - a
        key = f"{r.chromosome}:{int(r.start)}-{int(r.end)}"
        rows.append(dict(locus=key, lfc2kb=r.lfc2kb, motif_sum=c[ms:me].sum(),
                         clean=-c[ms:me].sum() / np.abs(c).sum()))
        mats[key] = (M, ms, me)
    t = pd.DataFrame(rows).sort_values("clean", ascending=False).reset_index(drop=True)
    t.to_csv(OUT / "ag_dermis_ism_screen.tsv", sep="\t", index=False)
    print(t.round(3).to_string())
    np.savez(OUT / "ag_dermis_ism_screen.npz", loci=t.locus.values,
             **{f"m{i}": mats[k][0] for i, k in enumerate(t.locus)},
             spans=np.array([mats[k][1:] for k in t.locus]))
    try:
        plot_contact(t, mats)
    except ImportError:
        print("logomaker not in this env; run plot_contact from featleak-dl")


def plot_contact(t=None, mats=None):
    import logomaker
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if t is None:
        d = np.load(OUT / "ag_dermis_ism_screen.npz", allow_pickle=True)
        t = pd.read_csv(OUT / "ag_dermis_ism_screen.tsv", sep="\t")
        mats = {k: (d[f"m{i}"], *d["spans"][i]) for i, k in enumerate(d["loci"])}
    n = min(12, len(t))
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.1 * n))
    for i in range(n):
        M, ms, me = mats[t.locus[i]]
        logomaker.Logo(pd.DataFrame(M, columns=list("ACGT")), ax=axes[i], color_scheme="classic")
        axes[i].axvspan(ms - .5, me - .5, color="gold", alpha=.2); axes[i].set_xticks([])
        axes[i].set_title(f"#{i} {t.locus[i]} clean={t.clean[i]:.2f} motif={t.motif_sum[i]:.2f}",
                          fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "ag_dermis_ism_contact.png", dpi=120)


if __name__ == "__main__":
    main()
