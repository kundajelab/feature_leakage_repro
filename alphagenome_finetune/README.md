# AlphaGenome fine-tuning with `alphagenome-ft` (JAX)

Helper code and an example for heads-only fine-tuning of AlphaGenome with
[`alphagenome-ft`](https://github.com/genomicsxai/alphagenome_ft) in JAX. The Figure 7
experiment uses the PyTorch implementation in `experiments/ag_deleak_finetune/` instead.

- `ft_core.py`: data loading, custom head registration and training helpers.
- `example_heads_only.py`: minimal heads-only fine-tuning example.
- `smoke_test.py`: trains a custom fibroblast DNase head for a few steps.

Environment: `alphagenome-ft` (JAX 0.10.2, `alphagenome` 0.5.1, `alphagenome_ft` 0.1.9).
`PATCHES.md` lists the changes to `alphagenome_ft` 0.1.9 needed with this JAX version.

```bash
conda activate alphagenome-ft
python smoke_test.py
```
