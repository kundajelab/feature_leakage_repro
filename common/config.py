"""Data and model path registry for the figure scripts.

Each key is resolved via a ``FEATLEAK_<KEY>`` environment variable, then
``common/paths.yaml``, then the built-in default candidates (first existing wins).
Run ``python -m common.config`` to print which resources resolve.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# repo layout
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = REPO_ROOT / "common"
FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data"

# Root of the source analysis tree.
LEGACY_ROOT = Path(os.environ.get("FEATLEAK_LEGACY_ROOT", "/users/kobbad/feat_leakage"))

# Shared data store root.
OAK_ROOT = Path(
    os.environ.get("FEATLEAK_AKUNDAJE_ROOT", "/oak/stanford/groups/akundaje")
)

# Local model-weights directory (see models/README.md); checked first for weights.
MODELS_ROOT = Path(os.environ.get("FEATLEAK_MODELS_ROOT", str(REPO_ROOT / "models")))

# --------------------------------------------------------------------------- #
# resource registry
# --------------------------------------------------------------------------- #
# Each value is a list of candidate paths; resolve() returns the first that
# exists (or the first candidate if none exist, so error messages are useful).
_REGISTRY: dict[str, list[str]] = {
    # ---- reference genome ------------------------------------------------- #
    "hg38_fasta": [
        f"{OAK_ROOT}/jmschr/common/hg38/hg38.fa",
        f"{OAK_ROOT}/kobbad/reference_genome/reference_genome/hg38.fa",
        "/users/kobbad/reference_genome/hg38.fa",
    ],
    # ---- pretrained model weights ---------------------------------------- #
    # PyTorch Beluga / DeepSEA (2002 outputs).
    "beluga_weights": [
        f"{MODELS_ROOT}/deepsea.beluga.pth",
        f"{LEGACY_ROOT}/deepSEA_negatives/deepsea.beluga.pth",
    ],
    # AlphaGenome v1 (JAX) weights directory.
    "alphagenome_weights_dir": [
        f"{MODELS_ROOT}/alphagenome-v1",
        f"{OAK_ROOT}/kobbad/alphagenome/alphagenome-v1",
    ],
    # AlphaGenome PyTorch port, all-folds weights.
    "alphagenome_pytorch_weights": [
        f"{MODELS_ROOT}/alphagenome-pytorch/model_all_folds.safetensors",
        f"{OAK_ROOT}/rrastogi/external_repos/alphagenome-pytorch/.models/model_all_folds.safetensors",
    ],
    # Sei model weights.
    "sei_weights": [
        f"{MODELS_ROOT}/sei/sei.pth",
        f"{LEGACY_ROOT}/SEI/model/sei.pth",
    ],
    # ChromBPNet, cardiac fibroblast (ENCSR000ENH)
    "chrombpnet_fibroblast": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR000ENH_chrombpnet_wo_bias.h5",
        f"{LEGACY_ROOT}/alphagenome/chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR000ENH/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # ChromBPNet, fibroblast of skin of biceps (ENCSR562ACY)
    "chrombpnet_skinbiceps": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR562ACY_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR562ACY/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # ChromBPNet, fibroblast of dermis (ENCSR000EPO)
    "chrombpnet_dermis": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR000EPO_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR000EPO/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # ChromBPNet, fibroblast of villous mesenchyme (ENCSR000EOR)
    "chrombpnet_encsr000eor": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR000EOR_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR000EOR/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # ChromBPNet, spleen (ENCSR128GBN)
    "chrombpnet_encsr128gbn": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR128GBN_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR128GBN/"
        "chrombpnet_model_encsr880cub_bias/chrombpnet_wo_bias.h5",
    ],
    # ENCODE BPNet models, K562 REST ChIP-seq (folds 0-4)
    "bpnet_rest_k562_dir": [
        f"{MODELS_ROOT}/bpnet_rest_k562",
        f"{LEGACY_ROOT}/data/bpnet_rest_k562",
    ],
    # ---- motif resources -------------------------------------------------- #
    # Vierstra motif archetypes and annotations
    "vierstra_motifs_tsv": [
        f"{DATA_DIR}/motifs/motifs.tsv",
        f"{LEGACY_ROOT}/ap1_counting/data/motifs.tsv",
    ],
    "vierstra_motif_annotations_xlsx": [
        f"{DATA_DIR}/motifs/motif_annotations.xlsx",
        f"{LEGACY_ROOT}/ap1_counting/data/motif_annotations.xlsx",
    ],
    # Beluga task names
    "beluga_task_names": [
        f"{DATA_DIR}/beluga/deepsea_beluga_names.txt",
        f"{LEGACY_ROOT}/all_ap1_motifs/deepsea_beluga_names.txt",
    ],
    # JASPAR 2022 CORE vertebrates (MEME format)
    "jaspar_meme": [
        f"{LEGACY_ROOT}/fibr_ipsc_peaks/"
        "JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    ],
    # ---- FIMO motif-hit scans (genome-wide) ------------------------------- #
    "fimo_octsox": [f"{LEGACY_ROOT}/all_ap1_motifs/OCTSOX_FIMO_HITS/fimo.tsv"],
    "fimo_ap1": [f"{LEGACY_ROOT}/all_ap1_motifs/AP1_FIMO_HITS/fimo.tsv"],
    "fimo_hnf4": [f"{LEGACY_ROOT}/all_ap1_motifs/HNF4_run_8/fimo.tsv"],
    "fimo_ctcf": [f"{LEGACY_ROOT}/all_ap1_motifs/CTCF_HITS/fimo.tsv"],
    # Affinity-scored motif hits
    "assoc_octsox": ["/users/jmschr/proj/exps/associations/oct-sox.txt"],
    "assoc_ap1": ["/users/jmschr/proj/exps/associations/ap-1.txt"],
    "assoc_hnf4": ["/users/jmschr/proj/exps/associations/hnf4.txt"],
    "assoc_ctcf": ["/users/jmschr/proj/exps/associations/ctcf.txt"],
    # ENCODE cCREs V3 (GRCh38)
    "encode_ccre_bed": [f"{LEGACY_ROOT}/data/encode_dhs/GRCh38-cCREs.bed"],
    # ---- peak / signal data ---------------------------------------------- #
    # Cardiac fibroblast DNase peaks and signal
    "peaks_fibroblast": [f"{LEGACY_ROOT}/alphagenome/ENCFF942AKV.bed.gz"],
    "bigwig_fibroblast_ag": [f"{LEGACY_ROOT}/alphagenome/ENCFF424KBD.bigWig"],
    # Cardiac fibroblast DNase signal (ENCSR000ENI)
    "bigwig_cardiac_fibroblast_eni": [f"{LEGACY_ROOT}/alphagenome/ENCFF258OVM.bigWig"],
    # Fibroblast DNase peaks (Enformer/Borzoi panels)
    "peaks_enformer_fibroblast": [f"{LEGACY_ROOT}/enformer_v2/ENCFF184KAM.bed.gz"],
    # Fibroblast of dermis DNase peaks and signal (ENCSR000EPO)
    "peaks_dermis": [f"{LEGACY_ROOT}/enformer_v2/ENCFF184KAM.bed.gz"],
    "bigwig_dermis": [f"{LEGACY_ROOT}/enformer_v2/ENCFF302JEV.bigWig"],
    "bigwig_enformer_fibroblast": [f"{LEGACY_ROOT}/enformer_v2/ENCFF361BTT.bigWig"],
    # Fibroblast of skin of biceps DNase peaks and signal
    "peaks_skin_biceps": [
        f"{LEGACY_ROOT}/skin_biceps_dnase/ENCFF246OZH.bed.gz",   # ENCSR562ACY
        f"{LEGACY_ROOT}/skin_biceps_dnase/ENCFF251OYZ.bed.gz",   # ENCSR648GDP
    ],
    "bigwig_skin_biceps": [
        f"{LEGACY_ROOT}/skin_biceps_dnase/ENCFF346TYL.bigWig",   # ENCSR562ACY
        f"{LEGACY_ROOT}/skin_biceps_dnase/ENCFF515BQL.bigWig",   # ENCSR648GDP
    ],
    "peaks_skin_biceps_alt": [f"{LEGACY_ROOT}/skin_biceps_dnase/ENCFF251OYZ.bed.gz"],
    "bigwig_skin_biceps_alt": [f"{LEGACY_ROOT}/skin_biceps_dnase/ENCFF515BQL.bigWig"],
    # Spleen DNase peaks
    "peaks_spleen": [f"{LEGACY_ROOT}/chrombpnet_2/ENCFF052RRA.bed.gz"],
    # ---- Fig 4 ChIP-seq peak sets ---------------------------------------- #
    # FOS ChIP-seq, fibroblast (ENCSR124AIG)
    "chip_fos_fibroblast": [
        f"{DATA_DIR}/figure4/ENCSR124AIG_peaks_inliers.bed.gz",
        f"{LEGACY_ROOT}/data/bed_files/fibro_ipsc/ENCSR124AIG_peaks_inliers.bed.gz",
    ],
    # POU5F1 ChIP-seq, hESC (ENCSR000BMU)
    "chip_oct4_hesc": [
        f"{DATA_DIR}/figure4/ENCSR000BMU_peaks_inliers.bed.gz",
        f"{LEGACY_ROOT}/data/bed_files/fibro_ipsc/ENCSR000BMU_peaks_inliers.bed.gz",
    ],
    # GC-matched background regions
    "gc_neg_fos": [
        f"{DATA_DIR}/figure4/ENCSR124AIG_gc_neg_only.bed.gz",
        f"{OAK_ROOT}/vir/tfatlas/processed_data/ENCSR124AIG/gc_neg_only.bed.gz",
    ],
    "gc_neg_oct4": [
        f"{DATA_DIR}/figure4/ENCSR000BMU_gc_neg_only.bed.gz",
        f"{OAK_ROOT}/vir/tfatlas/processed_data/ENCSR000BMU/gc_neg_only.bed.gz",
    ],
}


# --------------------------------------------------------------------------- #
# yaml overrides (optional)
# --------------------------------------------------------------------------- #
def _load_yaml_overrides() -> dict[str, str]:
    yaml_path = COMMON_DIR / "paths.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return {}
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh) or {}
    return {k: v for k, v in data.items() if v}


_YAML = _load_yaml_overrides()


def resolve(key: str, *, must_exist: bool = False) -> Path:
    """Resolve a registry *key* to a concrete path.

    Precedence: ``FEATLEAK_<KEY>`` env var > ``paths.yaml`` > built-in defaults
    (first existing candidate, else first candidate).

    Set ``must_exist=True`` to raise ``FileNotFoundError`` when nothing resolves
    to an existing path.
    """
    def _exists(p: Path) -> bool:
        # A candidate on a directory we lack permission to stat should be
        # skipped, not crash resolution (Path.exists() raises PermissionError).
        try:
            return p.exists()
        except OSError:
            return False

    env_key = f"FEATLEAK_{key.upper()}"
    if os.environ.get(env_key):
        path = Path(os.environ[env_key])
    elif key in _YAML:
        path = Path(_YAML[key])
    else:
        candidates = _REGISTRY.get(key)
        if not candidates:
            raise KeyError(f"Unknown resource key: {key!r}")
        path = next((Path(c) for c in candidates if _exists(Path(c))),
                    Path(candidates[0]))
    if must_exist and not _exists(path):
        raise FileNotFoundError(
            f"Resource {key!r} resolved to {path}, which does not exist. "
            f"Set ${env_key} or edit common/paths.yaml."
        )
    return path


def report() -> None:
    """Print which registry entries currently resolve to existing paths."""
    print(f"{'KEY':<32} {'EXISTS':<7} PATH")
    print("-" * 100)
    for key in _REGISTRY:
        p = resolve(key)
        try:
            ok = p.exists()
        except OSError:
            ok = False
        flag = "OK" if ok else "MISS"
        print(f"{key:<32} {flag:<7} {p}")


if __name__ == "__main__":
    report()
