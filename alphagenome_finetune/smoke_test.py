#!/usr/bin/env python
"""Run two heads-only AlphaGenome fine-tuning steps on a fibroblast DNase bigWig as a smoke test.

Run (env: alphagenome-ft):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        conda run -n alphagenome-ft python alphagenome_finetune/smoke_test.py
"""

from __future__ import annotations

from pathlib import Path

import jax

from alphagenome_ft import (
    create_model_with_heads,
    templates,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
)
from alphagenome.models import dna_output
from alphagenome_ft.finetune import (
    prepare_head_specs,
    validate_head_specs,
    BigWigDataModule,
    build_interval,
    train,
)

WEIGHTS = "/oak/stanford/groups/akundaje/kobbad/alphagenome/alphagenome-v1"
FASTA = Path("/oak/stanford/groups/akundaje/jmschr/common/hg38/hg38.fa")
BIGWIG = "/users/kobbad/feat_leakage/enformer_v2/ENCFF361BTT.bigWig"  # fibroblast DNase
HEAD = "fibroblast_dnase"
SEQ_LEN = 16384


def main() -> None:
    print("JAX devices:", jax.devices())

    # 1. Register a CUSTOM DNase head (1 track). Custom heads avoid the predefined
    #    metadata path, which is incompatible with the local alphagenome_research v1.
    register_custom_head(
        HEAD, templates.StandardHead,
        CustomHeadConfig(type=CustomHeadType.GENOME_TRACKS,
                         output_type=dna_output.OutputType.DNASE, num_tracks=1),
    )
    config = {"heads": [{
        "id": HEAD, "source": "custom",
        "targets": [{"path": BIGWIG, "name": "ENCFF361BTT_fibroblast_DNase"}],
    }]}
    specs = prepare_head_specs(config)
    validate_head_specs(specs)
    print(f"Head spec ready: {HEAD} <- {Path(BIGWIG).name}")

    # 2. Model: pretrained backbone + new head, backbone frozen
    model = create_model_with_heads("all_folds", heads=[HEAD], checkpoint_path=WEIGHTS)
    model.freeze_except_head(HEAD)
    print(f"Model loaded: {model.count_parameters():,} params")

    # 3. A few real intervals (train + valid), 16384 bp each
    starts = [1_000_000, 5_000_000, 20_000_000, 50_000_000, 100_000_000, 150_000_000]
    ivs = [build_interval(chromosome="chr1", start=s, end=s + SEQ_LEN) for s in starts]
    intervals = {"train": ivs[:4], "valid": ivs[4:]}
    dm = BigWigDataModule(intervals=intervals, fasta_path=FASTA, head_specs=specs,
                          batch_size=1, shuffle=True, drop_last=False)
    print(f"Data module: {len(intervals['train'])} train / {len(intervals['valid'])} valid intervals")

    # 4. 2 heads-only training steps
    train(model, dm, specs, learning_rate=1e-3, weight_decay=1e-4, num_epochs=1,
          max_train_steps=2, heads_only=True, verbose=True)
    print("\nSMOKE TEST PASSED — heads-only fine-tuning loop ran end-to-end.")


if __name__ == "__main__":
    main()
