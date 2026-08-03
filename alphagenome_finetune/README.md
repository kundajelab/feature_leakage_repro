# AlphaGenome fine-tuning environment (`alphagenome-ft`)

A working conda env for fine-tuning AlphaGenome with
[`alphagenome-ft`](https://github.com/genomicsxai/alphagenome_ft)
(blog: https://genomicsxai.github.io/blogs/2026-003/). Built fresh so it does not
touch the existing `alphagenome_research` analysis env.

## Verified working (2026-07-16)
- JAX **0.10.2** on GPU (`jax.devices() -> [CudaDevice(id=0)]`)
- `alphagenome` 0.5.1 (PyPI) + `alphagenome_research` (local `$OAK` develop install)
- `alphagenome_ft` 0.1.9 — custom head registration + `train()`
- **Loads the local AlphaGenome weights** (450,846,342 params) and merges a new head
- **End-to-end heads-only fine-tuning runs** — `smoke_test.py` trains a custom
  fibroblast-DNase head on a real bigWig (forward → loss → backward → AdamW),
  prints per-step losses + a validation metric, exits `SMOKE TEST PASSED`.

> The `train()` loop needs three small fixes to `alphagenome_ft` 0.1.9 (JAX-0.10.2
> incompatibilities + one missing import + a predefined-head metadata bug). They
> are already applied in this env; the diffs are documented in **`PATCHES.md`** so
> a rebuilt env can reapply them. Use **custom** heads (`register_custom_head`),
> not predefined heads.

## How to run it

The env needs the pip-installed NVIDIA CUDA wheel libs on `LD_LIBRARY_PATH`; a
conda `activate.d` hook does this automatically, so **use conda activation**:

```bash
# option A — activate then run
conda activate alphagenome-ft
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python your_script.py

# option B — one-shot (also runs activation, so GPU libs are found)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  conda run -n alphagenome-ft python your_script.py
```

> NOTE: calling `…/envs/alphagenome-ft/bin/python` **directly** (bypassing
> activation) will fall back to CPU, because `LD_LIBRARY_PATH` won't be set.
> Use `conda run` / `conda activate`, or export `LD_LIBRARY_PATH` yourself
> (see `etc/conda/activate.d/zz_nvidia_libs.sh`).

- **Weights:** point `checkpoint_path=` at your local AlphaGenome v1 checkpoint
  directory (see `models/README.md`); omit it to auto-download from Kaggle (needs auth).

## What was installed (for reproducibility)
```bash
conda create -n alphagenome-ft python=3.11 pip -y
pip install alphagenome-ft                    # -> jax 0.10.2 (cpu), optax, dm-haiku, flax, orbax
pip install "jax[cuda12]==0.10.2"             # GPU backend + nvidia cu12 wheels
pip install "alphagenome==0.5.1"              # matches the local alphagenome_research (v1)
# reuse local alphagenome_research (no GitHub needed):
cp <alphagenome_research env>/site-packages/_alphagenome_research.pth <alphagenome-ft env>/site-packages/
# runtime deps pulled in by the local alphagenome_research + alphagenome_ft:
pip install pyfaidx chex dm-tree tensorstore einshape einops "tensorflow-cpu==2.20.0" \
            pyranges kagglehub pyBigWig
# + etc/conda/activate.d/zz_nvidia_libs.sh puts the nvidia wheel libs on LD_LIBRARY_PATH
```

## Fine-tuning approaches (from alphagenome-ft)
1. **Heads-only** (frozen backbone) — single mid-range GPU, ~14–27 GB. See `example_heads_only.py`.
2. **LoRA-style adapters** — parameter-efficient backbone updates.
3. **Full-model** — needs ~76 GB (H100/H200).
4. **Encoder-only** — MPRA / sequences < 1 kb.

See `example_heads_only.py` for a minimal end-to-end skeleton.
