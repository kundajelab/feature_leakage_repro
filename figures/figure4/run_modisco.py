#!/usr/bin/env python
"""ISM contribution logos and TF-MoDISco motif discovery for the Figure 4 FOS/OCT4 models (Figure 4a/b, supplement a-c).

Outputs (outputs/): saturation_mutagenesis_logo_*.pdf and, with --modisco,
modisco_multitask.h5 and modisco_single_task.h5.

Run (env: featleak-dl):
    python run_modisco.py --modisco
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tangermeme.io import extract_loci  # noqa: E402
from tangermeme.ism import saturation_mutagenesis  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from common import config  # noqa: E402
from helpers_4 import train_multitask_model, train_singletask_model  # noqa: E402
from run_experiment import build_data  # noqa: E402

OUT = Path(__file__).parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ism_logo(model, ohe, target, title, fname):
    X_attr = saturation_mutagenesis(model, ohe, target=target, verbose=False)
    fig = plt.figure(figsize=(10, 2))
    ax = plt.subplot(111)
    plot_logo(X_attr[0, :, 120:220], ax=ax)
    ax.set_title(title); ax.set_ylim(-6, 6)
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


def run_modisco_on(model, peaks, hg38, name, n_peaks=1500):
    """Run modiscolite on a model's attributions over its positive peaks."""
    import modiscolite
    from tangermeme.deep_lift_shap import deep_lift_shap

    sample = peaks.sample(min(n_peaks, len(peaks)), random_state=0)
    loci, keep = extract_loci(sample, hg38, in_window=300, return_filtered=True)
    mask = loci.sum(axis=1).sum(axis=1) == loci.shape[2]
    loci = loci[mask].float()
    attr = deep_lift_shap(model, loci, target=0, device=DEVICE, batch_size=64)
    seqs = loci.numpy().transpose(0, 2, 1).astype("float32")
    contribs = (attr.numpy() * loci.numpy()).transpose(0, 2, 1).astype("float32")
    pos, neg = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=attr.numpy().transpose(0, 2, 1).astype("float32"),
        one_hot=seqs, max_seqlets_per_metacluster=20000,
        sliding_window_size=20, flank_size=5, target_seqlet_fdr=0.05,
    )
    n_pos = len(pos.seqlets) if pos else 0
    n_neg = len(neg.seqlets) if neg else 0
    print(f"    [{name}] modisco: {n_pos} pos-pattern seqlets, {n_neg} neg-pattern seqlets")
    modiscolite.io.save_hdf5(str(OUT / f"modisco_{name}.h5"), pos, neg, 20)
    return n_pos, n_neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modisco", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    hg38, peaks, labels, background = build_data()

    print("[1/3] training models ...")
    multi, _ = train_multitask_model(peaks, labels, background, hg38, relative_proportion=0.2)
    # CLEAN single-task AP1: positives = FIBROBLAST-only peaks (labels[:,0]==1),
    # negatives = random background. Crucially this NEVER sees OCT4-only peaks as
    # negatives, so it should not learn OCT-SOX. (Training on the union instead
    # -- the "leaky" setup -- is what makes a single-task model spuriously
    # respond to OCT-SOX.)
    fibro_mask = labels[:, 0] == 1
    single, _ = train_singletask_model(
        peaks[fibro_mask], labels[fibro_mask], background, hg38, task_idx=0,
        relative_proportion=0.1)

    print("[2/3] ISM logos at a joint AP1+OCT-SOX locus ...")
    joint = peaks[labels.sum(axis=1) == 2].reset_index(drop=True)
    joint_loci = extract_loci(joint, hg38, in_window=300, verbose=False)
    idx = 35 if len(joint) > 35 else 0
    chrom, start, end = joint.iloc[idx][["chrom", "start", "end"]]
    ohe = joint_loci[idx].type(torch.float32).unsqueeze(0)
    title = f"{chrom}:{start}-{end}"
    ism_logo(multi, ohe, 0, f"Multi-task AP1 task  {title}",
             "saturation_mutagenesis_logo_fibro_head_multitask.pdf")
    ism_logo(multi, ohe, 1, f"Multi-task OCT-SOX task  {title}",
             "saturation_mutagenesis_logo_esc_head_multitask.pdf")
    ism_logo(single, ohe, 0, f"Single-task AP1  {title}",
             "saturation_mutagenesis_logo_fibro_head_single_task.pdf")
    print(f"    logos saved for locus {title}")

    if args.modisco:
        print("[3/3] TF-MoDISco (Panel B) ...")
        run_modisco_on(multi, peaks[labels[:, 0] > 0], hg38, "multitask")
        run_modisco_on(single, peaks[labels[:, 0] > 0], hg38, "single_task")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
