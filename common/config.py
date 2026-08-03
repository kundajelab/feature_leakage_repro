"""Central configuration for the Feature-Leakage figure reproduction repo.

Every path a figure script needs is resolved here, in one place, so that porting
the repo to another machine only requires editing this file (or the environment
variables / ``paths.yaml`` it reads).

Design goals
------------
* No hard-coded absolute paths scattered across figure scripts.
* Each resource is looked up through :func:`resolve`, which tries, in order:
    1. an environment variable override (``FEATLEAK_<KEY>``),
    2. any entry for the key in ``paths.yaml`` next to this file,
    3. the built-in default(s) below (the original locations on the Kundaje-lab
       cluster; several keys list *multiple* candidates and the first that
       exists wins).
* Nothing here imports torch / jax / tensorflow, so it is safe to import from
  any environment.

Run ``python -m common.config`` to print a report of which resources resolve.
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

# Root of the original (do-not-touch) analysis tree.
LEGACY_ROOT = Path(os.environ.get("FEATLEAK_LEGACY_ROOT", "/users/kobbad/feat_leakage"))

# Shared lab store. NOTE: the interactive ``$OAK`` env var on this cluster points
# at the *user* subdirectory (.../akundaje/kobbad), but several resources live
# under the lab *group* root (jmschr/, projects/, vir/). We therefore anchor on
# the group root explicitly and address the user subdir as ``OAK_ROOT/kobbad``.
OAK_ROOT = Path(
    os.environ.get("FEATLEAK_AKUNDAJE_ROOT", "/oak/stanford/groups/akundaje")
)

# Local, git-ignored directory where an external user drops the model weights they
# download (see models/README.md). Every model-weight key below lists a
# ``MODELS_ROOT/...`` candidate FIRST, so a populated ``models/`` wins over the
# original cluster paths. Override the location with FEATLEAK_MODELS_ROOT.
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
    # PyTorch Beluga / DeepSEA (Kipoi architecture, 2002 outputs), 598 MB.
    "beluga_weights": [
        f"{MODELS_ROOT}/deepsea.beluga.pth",
        f"{LEGACY_ROOT}/deepSEA_negatives/deepsea.beluga.pth",
    ],
    # AlphaGenome v1 local weights directory (loaded via dna_model.create).
    "alphagenome_weights_dir": [
        f"{MODELS_ROOT}/alphagenome-v1",
        f"{OAK_ROOT}/kobbad/alphagenome/alphagenome-v1",
    ],
    # SEI model weights (keep sei.py architecture file alongside the .pth).
    "sei_weights": [
        f"{MODELS_ROOT}/sei/sei.pth",
        f"{LEGACY_ROOT}/SEI/model/sei.pth",
    ],
    # ChromBPNet DNase models (bpnet-lite `.h5`, load with BPNet.from_chrombpnet).
    # NOTE: ENCSR000ENH = CARDIAC fibroblast. It matches the cardiac PEAK substrate
    # (ENCFF942AKV) but NOT the readout tasks of the multitask models we compare
    # against. For feature-leakage comparisons use the EXPERIMENT-MATCHED keys below
    # (chrombpnet_skinbiceps / chrombpnet_dermis) and the shuffle-controlled
    # SPECIFICITY metric (raw log2FC is confounded by generic insertion disruption).
    "chrombpnet_fibroblast": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR000ENH_chrombpnet_wo_bias.h5",
        f"{LEGACY_ROOT}/alphagenome/chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR000ENH/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # Experiment-matched ChromBPNet (chromatin-atlas-2022, fold 0) for the exact ENCODE
    # experiment each multitask model's fibroblast DNase task uses:
    "chrombpnet_skinbiceps": [   # ENCSR562ACY -> matches AlphaGenome skin-biceps NTR:0000521
        f"{MODELS_ROOT}/chrombpnet/ENCSR562ACY_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR562ACY/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    "chrombpnet_dermis": [       # ENCSR000EPO -> matches Enformer/Borzoi dermis (ENCFF302JEV/361BTT)
        f"{MODELS_ROOT}/chrombpnet/ENCSR000EPO_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR000EPO/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # Borzoi/Enformer-matched fibroblast DNase ChromBPNet (ENCSR000EOR).
    "chrombpnet_encsr000eor": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR000EOR_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR000EOR/"
        "chrombppnet_model_encsr283tme_bias/chrombpnet_wo_bias.h5",
    ],
    # Second-tissue (spleen) DNase ChromBPNet (ENCSR128GBN).
    "chrombpnet_encsr128gbn": [
        f"{MODELS_ROOT}/chrombpnet/ENCSR128GBN_chrombpnet_wo_bias.h5",
        f"{OAK_ROOT}/projects/chromatin-atlas-2022/DNASE/ENCSR128GBN/"
        "chrombpnet_model_encsr880cub_bias/chrombpnet_wo_bias.h5",
    ],
    # ---- motif resources -------------------------------------------------- #
    # Vierstra motif-archetype tables (consensus strings + annotations).
    "vierstra_motifs_tsv": [
        f"{LEGACY_ROOT}/ap1_counting/data/motifs.tsv",
    ],
    "vierstra_motif_annotations_xlsx": [
        f"{LEGACY_ROOT}/ap1_counting/data/motif_annotations.xlsx",
    ],
    # DeepSEA/Beluga task (cell-type x assay) names, one per line, 2002 rows.
    "beluga_task_names": [
        f"{LEGACY_ROOT}/all_ap1_motifs/deepsea_beluga_names.txt",
    ],
    # JASPAR database for TF-MoDISco `report` annotation (Fig 4B).
    "jaspar_meme": [
        f"{LEGACY_ROOT}/fibr_ipsc_peaks/"
        "JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
    ],
    # ---- FIMO motif-hit scans (genome-wide) ------------------------------- #
    "fimo_octsox": [f"{LEGACY_ROOT}/all_ap1_motifs/OCTSOX_FIMO_HITS/fimo.tsv"],
    "fimo_ap1": [f"{LEGACY_ROOT}/all_ap1_motifs/AP1_FIMO_HITS/fimo.tsv"],
    "fimo_hnf4": [f"{LEGACY_ROOT}/all_ap1_motifs/HNF4_run_8/fimo.tsv"],
    "fimo_ctcf": [f"{LEGACY_ROOT}/all_ap1_motifs/CTCF_HITS/fimo.tsv"],
    # Jacob Schreiber's affinity-scored association hits (used by Fig 1 E-H
    # quartile-split scatter machinery). `score` column is the affinity.
    "assoc_octsox": ["/users/jmschr/proj/exps/associations/oct-sox.txt"],
    "assoc_ap1": ["/users/jmschr/proj/exps/associations/ap-1.txt"],
    "assoc_hnf4": ["/users/jmschr/proj/exps/associations/hnf4.txt"],
    "assoc_ctcf": ["/users/jmschr/proj/exps/associations/ctcf.txt"],
    # ---- peak / signal data ---------------------------------------------- #
    # Cardiac-fibroblast DNase peaks + signal (Fig 1/2 AlphaGenome & ChromBPNet).
    "peaks_fibroblast": [f"{LEGACY_ROOT}/alphagenome/ENCFF942AKV.bed.gz"],
    "bigwig_fibroblast_ag": [f"{LEGACY_ROOT}/alphagenome/ENCFF424KBD.bigWig"],
    # Peaks used for Borzoi/Enformer fibroblast panels.
    "peaks_enformer_fibroblast": [f"{LEGACY_ROOT}/enformer_v2/ENCFF184KAM.bed.gz"],
    "bigwig_enformer_fibroblast": [f"{LEGACY_ROOT}/enformer_v2/ENCFF361BTT.bigWig"],
    # Skin-biceps fibroblast DNase (AlphaGenome native track NTR:0000521, the OCT-SOX
    # *leaking* head we fine-tune). Two matched ENCODE single-ended DNase experiments
    # (ENCSR562ACY / ENCSR648GDP); use whichever AG correlates best with as primary.
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
    # Second-tissue peaks (spleen) for the Fig 2 supplement.
    "peaks_spleen": [f"{LEGACY_ROOT}/chrombpnet_2/ENCFF052RRA.bed.gz"],
    # ---- Fig 4 ChIP-seq peak sets ---------------------------------------- #
    # FOS/AP1 ChIP-seq in fibroblast (ENCSR124AIG).
    "chip_fos_fibroblast": [
        f"{LEGACY_ROOT}/data/bed_files/fibro_ipsc/ENCSR124AIG_peaks_inliers.bed.gz",
    ],
    # POU5F1/OCT4 ChIP-seq in hESC (ENCSR000BMU).
    "chip_oct4_hesc": [
        f"{LEGACY_ROOT}/data/bed_files/fibro_ipsc/ENCSR000BMU_peaks_inliers.bed.gz",
    ],
    # GC-matched background regions for each ChIP experiment.
    "gc_neg_fos": [
        f"{OAK_ROOT}/vir/tfatlas/processed_data/ENCSR124AIG/gc_neg_only.bed.gz",
    ],
    "gc_neg_oct4": [
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
    env_key = f"FEATLEAK_{key.upper()}"
    if os.environ.get(env_key):
        path = Path(os.environ[env_key])
    elif key in _YAML:
        path = Path(_YAML[key])
    else:
        candidates = _REGISTRY.get(key)
        if not candidates:
            raise KeyError(f"Unknown resource key: {key!r}")
        path = next((Path(c) for c in candidates if Path(c).exists()),
                    Path(candidates[0]))
    if must_exist and not path.exists():
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
        flag = "OK" if p.exists() else "MISS"
        print(f"{key:<32} {flag:<7} {p}")


if __name__ == "__main__":
    report()
