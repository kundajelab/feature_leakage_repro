# Changes to `alphagenome-ft` 0.1.9 for JAX 0.10.2

With JAX 0.10.2, the `train()` loop of `alphagenome-ft` 0.1.9 needs three changes to
`alphagenome_ft/finetune/train.py` in the environment's site-packages.

### 1. Missing `parameter_utils` import  (NameError)
`train.py` uses `parameter_utils.DictKey/…/get_head_parameter_paths` but never
imports the module (it only imports the *symbol* `create_optimizer` from
`optimizer_utils`, then defines a local `create_optimizer`).

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

These changes support single-GPU, heads-only fine-tuning with custom heads
(`register_custom_head`).
