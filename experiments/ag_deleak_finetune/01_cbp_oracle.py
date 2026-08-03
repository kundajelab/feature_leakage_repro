#!/usr/bin/env python
"""Oracle step: ChromBPNet OCT-SOX effect per enhancer (the de-leak label source).

For the top-N skin-biceps fibroblast DNase peaks (the substrate whose AlphaGenome
NTR:0000521 head we will de-leak), score with the fibroblast ChromBPNet:
  s0        base central signal
  s_real    OCT-SOX inserted in the flanks (4 offsets)  -> real effect
  s_shuf    mono-shuffled OCT-SOX inserted (k seeds, averaged) -> generic control
and record log2FC(real) and log2FC(shuffle). ChromBPNet is our oracle for "what
OCT-SOX SHOULD do to fibroblast accessibility" (no OCT-SOX-specific drop); the
gap log2FC(real) - log2FC(shuffle) is ~0 for ChromBPNet but large (negative) for
the leaking AlphaGenome head. These per-enhancer log2FC(real) values become the
target relative-effect AlphaGenome is fine-tuned to reproduce.

Run (env: featleak-dl):
    conda run -n featleak-dl python 01_cbp_oracle.py --n-loci 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from bpnetlite.bpnet import BPNet  # noqa: E402
from bpnetlite.io import extract_loci  # noqa: E402
from tangermeme.ersatz import substitute  # noqa: E402

from common import config  # noqa: E402
from common.motifs import OCT_SOX  # noqa: E402
from common.motif_controls import shuffled_motifs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import chrom_filter  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda"
IN_WIN = 2114
OFFSETS = [-150, 150, -200, 200]
EPS = 1e-3


def central_total(model, X):
    """counts * central-profile-prob over +/-150 bp."""
    yp, yc = model(X.float().to(DEVICE))
    counts = (torch.exp(yc) - 1.0).squeeze(-1)                 # natural-log inverse
    logits = yp.reshape(yp.shape[0], -1)
    probs = torch.softmax(logits - logits.mean(-1, keepdim=True), -1)
    L = probs.shape[-1]; c = L // 2; w = int(150 / (IN_WIN / L))
    return (counts * probs[:, c - w:c + w].sum(-1)).detach().cpu().numpy()


def insert_all(loci, motif, mid):
    out = loci.clone()
    for o in OFFSETS:
        out = substitute(out, motif, start=mid + o)
    return out


def batched(model, X, bs=16):
    return np.concatenate([central_total(model, X[i:i + bs]) for i in range(0, len(X), bs)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loci", type=int, default=400)
    ap.add_argument("--k-shuf", type=int, default=3)
    ap.add_argument("--peaks-key", default="peaks_skin_biceps")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="chromosome split (chr8/chr9 held out as test)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = BPNet.from_chrombpnet(str(config.resolve("chrombpnet_fibroblast",
                                                     must_exist=True))).to(DEVICE).eval()
    peaks = pd.read_csv(config.resolve(args.peaks_key), sep="\t", header=None)
    peaks = peaks.rename(columns={0: "chrom", 1: "start", 2: "end", 6: "signal"})
    peaks = chrom_filter(peaks, args.split)
    top = peaks.sort_values("signal", ascending=False).head(args.n_loci).reset_index(drop=True)
    print(f"split={args.split}: {len(top)} enhancers on {sorted(top.chrom.unique())}")

    loci = extract_loci(top[["chrom", "start", "end"]], str(config.resolve("hg38_fasta")),
                        in_window=IN_WIN, verbose=False).float()
    mid = IN_WIN // 2
    shuf_motifs = shuffled_motifs(OCT_SOX, k=args.k_shuf)

    with torch.no_grad():
        s0 = batched(model, loci)
        s_real = batched(model, insert_all(loci, OCT_SOX, mid))
        s_shuf_k = np.stack([batched(model, insert_all(loci, m, mid)) for m in shuf_motifs])
    s_shuf = s_shuf_k.mean(0)

    lfc_real = np.log2((s_real + EPS) / (s0 + EPS))
    lfc_shuf = np.log2((s_shuf + EPS) / (s0 + EPS))
    spec = lfc_real - lfc_shuf

    df = pd.DataFrame({
        "chrom": top.chrom, "start": top.start.astype(int), "end": top.end.astype(int),
        "signal": top.signal.values,
        "cbp_s0": s0, "cbp_s_real": s_real, "cbp_s_shuf": s_shuf,
        "cbp_lfc_real": lfc_real, "cbp_lfc_shuf": lfc_shuf, "cbp_specificity": spec,
    })
    df.to_csv(OUT / f"cbp_oracle_{args.split}.csv", index=False)
    print(f"ChromBPNet oracle over {len(df)} skin-biceps enhancers (split={args.split}):")
    print(f"  log2FC(real)  median={np.median(lfc_real):+.4f}")
    print(f"  log2FC(shuf)  median={np.median(lfc_shuf):+.4f}")
    print(f"  specificity   median={np.median(spec):+.4f}  (oracle: ~0 = no OCT-SOX-specific effect)")
    print("done ->", OUT / f"cbp_oracle_{args.split}.csv")


if __name__ == "__main__":
    main()
