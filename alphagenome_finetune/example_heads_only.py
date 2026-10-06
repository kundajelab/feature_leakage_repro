#!/usr/bin/env python
"""Set up heads-only fine-tuning of AlphaGenome with a new custom RNA-seq head.

Run (env: alphagenome-ft):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
        conda run -n alphagenome-ft python alphagenome_finetune/example_heads_only.py
"""

from __future__ import annotations

import jax

from alphagenome.models import dna_output
from alphagenome_ft import (
    templates,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
    create_model_with_heads,
    create_optimizer,
)

WEIGHTS = "/oak/stanford/groups/akundaje/kobbad/alphagenome/alphagenome-v1"
HEAD = "my_rna_head"


def main() -> None:
    print("JAX devices:", jax.devices())

    # 1. Register a new task-specific head (StandardHead = 1bp decoder embeddings).
    register_custom_head(
        HEAD,
        templates.StandardHead,
        CustomHeadConfig(
            type=CustomHeadType.GENOME_TRACKS,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,                 # e.g. one new RNA-seq track
        ),
    )

    # 2. Load pretrained backbone + attach head, then freeze the backbone.
    model = create_model_with_heads("all_folds", heads=[HEAD], checkpoint_path=WEIGHTS)
    model.freeze_except_head(HEAD)
    print(f"Model loaded: {model.count_parameters():,} params; head = {HEAD}")

    # 3. Heads-only optimizer (masks backbone updates).
    optimizer = create_optimizer(
        model._params,
        trainable_head_names=(HEAD,),
        learning_rate=1e-3,
        weight_decay=1e-4,
        heads_only=True,
    )
    opt_state = optimizer.init(model._params)
    print("Optimizer ready. Backbone frozen, head trainable.")

    # 4. Train.  Point these at your bigWig targets and uncomment.
    # from alphagenome_ft.finetune import train
    # train(
    #     model,
    #     train_bigwig_paths=["/path/to/train.bw"],
    #     val_bigwig_paths=["/path/to/val.bw"],
    #     heads_only=True,
    #     learning_rate=1e-3,
    #     num_epochs=10,
    # )
    # model.save_checkpoint("checkpoints/my_rna_head", save_full_model=False)

    print("Setup OK — fill in bigWig paths in step 4 to fine-tune.")


if __name__ == "__main__":
    main()
