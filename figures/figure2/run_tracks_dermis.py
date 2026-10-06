#!/usr/bin/env python
"""Figure 2 C/D (dermis-matched): predicted DNase at one enhancer with 0-4 OCT-SOX insertions.

Both models read the same ENCODE experiment, ENCSR000EPO (fibroblast of dermis):
  --model chrombpnet   `chrombpnet_dermis`, base-resolution expected counts (env featleak-dl)
  --model alphagenome  CL:0002551 DNase track, full 524,288 bp input, PyTorch port (env featleak-agtorch)

OCT-SOX is written at the flank offsets in cbpnet_helpers.TRACK_INSERT_OFFSETS
(1, 2, 3, then 4 insertions). Writes outputs/<model>_track_data_<tag>.npz in the
format read by run_tracks_matched.py:

    python run_tracks_dermis.py --model chrombpnet  --locus chr6:162964880:162965180
    python run_tracks_dermis.py --model alphagenome --locus chr6:162964880:162965180
    python run_tracks_matched.py --cbp-tag dermis --ag-tag dermis --suffix _dermis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pyfaidx  # noqa: E402

from common import config  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OCT_SOX = "ATTGTTATGCAAAT"
OFFSETS = [-150, +150, -200, +200]      # = cbpnet_helpers.TRACK_INSERT_OFFSETS
HALF = 500                              # bp shown each side of the enhancer centre
AG_CH = 60                              # CL:0002551 fibroblast of dermis


def insert(seq, offsets, mid):
    s = list(seq)
    for o in offsets:
        s[mid + o:mid + o + len(OCT_SOX)] = list(OCT_SOX)
    return "".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["chrombpnet", "alphagenome"], required=True)
    ap.add_argument("--locus", required=True, help="chrom:start:end of the enhancer")
    ap.add_argument("--tag", default="dermis")
    args = ap.parse_args()
    chrom, s, e = args.locus.split(":")
    mid_g = (int(s) + int(e)) // 2
    fa = pyfaidx.Fasta(str(config.resolve("hg38_fasta")))

    import torch
    if args.model == "chrombpnet":
        from bpnetlite.bpnet import BPNet
        from tangermeme.utils import one_hot_encode
        from cbpnet_helpers import scaled_profile_wrapper
        W, OUTW = 2114, 1000
        model = scaled_profile_wrapper()(BPNet.from_chrombpnet(
            str(config.resolve("chrombpnet_dermis", must_exist=True)))).cuda().eval()
        q = str(fa[chrom][mid_g - W // 2:mid_g + W // 2]).upper()

        def profile(seq):
            with torch.no_grad():
                return model(one_hot_encode(seq)[None].float().cuda()).squeeze().cpu().numpy()
        bp = np.arange(OUTW) - OUTW // 2
        name, ylabel = "ChromBPNet (dermis)", "expected counts / bp"
    else:
        from alphagenome_pytorch import AlphaGenome
        W = 524288
        model = AlphaGenome.from_pretrained(
            str(config.resolve("alphagenome_pytorch_weights", must_exist=True)), device="cuda")
        q = str(fa[chrom][mid_g - W // 2:mid_g + W // 2]).upper()
        lut = np.full(256, -1, dtype=np.int64)
        for i, c in enumerate("ACGT"):
            lut[ord(c)] = i

        def profile(seq):
            idx = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
            x = torch.zeros((1, W, 4)); ok = idx >= 0
            x[0, np.where(ok)[0], idx[ok]] = 1
            with torch.no_grad():
                out = model.predict(x.cuda(), organism_index=0)
            return out["dnase"][1][0, W // 2 - HALF:W // 2 + HALF, AG_CH].float().cpu().numpy()
        bp = np.arange(2 * HALF) - HALF
        name, ylabel = "AlphaGenome (CL:0002551 dermis)", "predicted DNase"

    profiles = np.stack([profile(insert(q, OFFSETS[:k], W // 2)) for k in range(5)])
    np.savez(OUT / f"{args.model}_track_data_{args.tag}.npz", profiles=profiles, bp=bp,
             ins_bp=np.array(OFFSETS) + len(OCT_SOX) // 2, model=name, ylabel=ylabel,
             locus=args.locus)
    tot = profiles[:, (bp >= -HALF) & (bp < HALF)].sum(1)
    print(f"{name} at {args.locus}: total signal by #insertions = "
          + ", ".join(f"{t / tot[0]:.2f}" for t in tot))


if __name__ == "__main__":
    main()
