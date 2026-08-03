"""Model loaders shared across figures.

Thin, config-driven wrappers so no figure script hard-codes a weight path or a
CUDA device. Import the heavy ML frameworks lazily inside each function so that
``common.config`` / ``common.motifs`` stay importable from any environment.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def get_device(prefer: str | None = None):
    """Return a torch device string. ``prefer`` overrides autodetection."""
    import torch

    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_beluga(device: str | None = None, with_counting: bool = False):
    """Load the PyTorch Beluga/DeepSEA model (2002 tasks).

    Args:
        device: torch device; autodetected if None.
        with_counting: if True, return a ``BelugaWithCounting`` (2003 tasks,
            the extra output is the raw counting logit at index 2002).
    """
    import torch

    from .beluga_model import Beluga, expand_tasks_to_2003

    device = get_device(device)
    weights = config.resolve("beluga_weights", must_exist=True)
    model = Beluga()
    model.load_state_dict(torch.load(weights, map_location=device))
    if with_counting:
        model = expand_tasks_to_2003(model)
    return model.to(device).eval()


def load_beluga_task_names() -> list[str]:
    """Return the 2002 DeepSEA/Beluga task (cell-type x assay) names."""
    path = config.resolve("beluga_task_names", must_exist=True)
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def beluga_task_index(substring: str) -> int:
    """Index of the first Beluga task whose name contains ``substring``."""
    names = load_beluga_task_names()
    for i, name in enumerate(names):
        if substring in name:
            return i
    raise ValueError(f"No Beluga task matches {substring!r}")


def load_chrombpnet(which: str = "fibroblast", device: str | None = None,
                    count_wrapper: bool = True):
    """Load a ChromBPNet DNase model via bpnet-lite.

    Args:
        which: one of ``fibroblast`` (ENCSR000ENH), ``encsr000eor``,
            ``encsr128gbn`` — resolved through :mod:`common.config`.
        count_wrapper: wrap in ``CountWrapper`` (scalar total-count output),
            as used by the ablation/marginalization effect-size code.
    """
    from bpnetlite.bpnet import BPNet, CountWrapper

    key = {
        "fibroblast": "chrombpnet_fibroblast",
        "encsr000eor": "chrombpnet_encsr000eor",
        "encsr128gbn": "chrombpnet_encsr128gbn",
    }[which]
    h5 = config.resolve(key, must_exist=True)
    model = BPNet.from_chrombpnet(str(h5))
    if count_wrapper:
        model = CountWrapper(model)
    device = get_device(device)
    return model.to(device).eval()


def load_alphagenome():
    """Load the *local* AlphaGenome v1 model (JAX). Requires the
    ``featleak-alphagenome`` env. Returns the dna_model handle.

    This deliberately uses the locally-installed weights rather than the hosted
    API, per the repo's offline-reproducibility goal.
    """
    from alphagenome_research.model import dna_model

    weights_dir = config.resolve("alphagenome_weights_dir", must_exist=True)
    return dna_model.create(str(weights_dir))


def load_grelu_model(project: str, model_name: str, device: str | None = None):
    """Load a Borzoi/Enformer model via gReLU.

    Examples:
        load_grelu_model("borzoi", "human_fold0")
        load_grelu_model("enformer", "human")
    """
    import grelu.resources

    model = grelu.resources.load_model(project=project, model_name=model_name)
    device = get_device(device)
    return model.to(device).eval()


def load_sei(device: str | None = None):
    """Load the SEI model (local weights)."""
    import sys

    import torch

    sei_dir = Path(config.resolve("sei_weights")).parent
    sys.path.insert(0, str(sei_dir))
    from sei import Sei  # noqa: PLC0415  (SEI/model/sei.py)

    device = get_device(device)
    model = Sei()
    model.load_state_dict(
        torch.load(config.resolve("sei_weights", must_exist=True),
                   map_location=device)
    )
    return model.to(device).eval()
