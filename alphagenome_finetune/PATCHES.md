# Patches applied to `alphagenome-ft` 0.1.9 (env `alphagenome-ft`)

The published `alphagenome-ft` 0.1.9 does **not** run its `train()` loop as-is
against the JAX we installed (0.10.2) — three defects surface once you get past
model load. All three are bugs in the package itself (or hard-coded to an older
JAX), not in our environment. The smoke test
(`smoke_test.py`) passes only with these applied.

File patched (both under the env's site-packages):
`…/envs/alphagenome-ft/lib/python3.11/site-packages/alphagenome_ft/finetune/train.py`

### 1. Missing `parameter_utils` import  (NameError)
`train.py` uses `parameter_utils.DictKey/…/get_head_parameter_paths` but never
imports the module (it only imports the *symbol* `create_optimizer` from
`optimizer_utils`, then shadows it with a local buggy `create_optimizer`).

Add near the other imports (after the `custom_model` import):
```python
from alphagenome_ft import parameter_utils
```

### 2. `jax.device_put_replicated` removed in JAX ≥ 0.9  (AttributeError)
`_replicate_tree()` calls the removed pmap helper. Replace its body with a
leading-device-axis broadcast that `pmap` still accepts:
```python
def _replicate_tree(tree, devices):
    import jax.numpy as jnp
    n = len(devices)
    def _rep(value):
        arr = jnp.asarray(value)
        return jnp.broadcast_to(arr[None], (n,) + arr.shape)
    return jax.tree_util.tree_map(_rep, tree)
```

### 3. `_predict(...)` missing required `requested_outputs`  (TypeError)
`custom_model.wrapped_predict()` takes a keyword-only `requested_outputs`, but
`train.py`'s `train_step`/`eval_step` call `model._predict(...)` without it.
For a **custom-head-only** model the standard-prediction branch is empty, so the
value is unused — pass an empty list. In BOTH `_predict(...)` call sites add:
```python
                requested_outputs=[],
```
(one in `train_step.loss_fn`, one in `eval_step`).

---

## Notes / limitations
- Root cause is **version mixing**: `alphagenome-ft` 0.1.9 targets an older JAX
  (pmap era) and a newer `alphagenome` metadata format than the local
  `alphagenome_research` v1 weights. These patches bridge the gap for
  single-GPU **heads-only** custom-head fine-tuning.
- **Predefined heads** (`register_predefined_head` / `source: "predefined"`)
  still hit a separate metadata incompatibility
  (`TypeError: object of type 'AlphaGenomeOutputMetadata' has no len()` in
  `custom_model._resolve_user_metadata`). Use **custom heads**
  (`register_custom_head` + `source: "custom"`) — that path works.
- Multi-GPU (`num_devices > 1`) untested here; patch #2 keeps the single-device
  pmap path working. For real multi-GPU you'd want to revisit the pmap→shard_map
  migration rather than rely on the broadcast shim.
