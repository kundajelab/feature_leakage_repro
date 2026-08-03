#!/usr/bin/env bash
# =============================================================================
# Reproduce all Figure 1 panels (feature leakage across models) for publication.
#
# Every figure is rendered through tools/render_with_svg.py, which forces
# Illustrator-editable fonts (pdf.fonttype=42) and writes a companion .svg next
# to each .pdf. Outputs land in figures/figure1/outputs/ (and, for the TF ChIP
# panel, experiments/tf_chip_leakage/outputs/).
#
# Usage:   bash figures/figure1/reproduce.sh [GPU]      # default GPU 0
# Envs:    featleak-dl          = torch / gReLU / Beluga / ChromBPNet
#          featleak-alphagenome = local AlphaGenome (JAX)  -- needs the AlphaGenome
#                                 checkpoint in models/alphagenome-v1 (or $OAK)
# Override env names with TORCH_ENV / AG_ENV if you named yours differently.
# =============================================================================
set -euo pipefail

GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

TORCH_ENV="${TORCH_ENV:-featleak-dl}"
AG_ENV="${AG_ENV:-featleak-alphagenome}"
LAUNCH="tools/render_with_svg.py"

run_torch () { echo ">>> [torch] $*"; conda run --no-capture-output -n "$TORCH_ENV" python "$LAUNCH" "$@"; }
run_ag    () { echo ">>> [AG]    $*"; conda run --no-capture-output -n "$AG_ENV"    python "$LAUNCH" "$@"; }

# --- 1. OCT-SOX ablation effect-size boxplots (raw effect) --------------------
run_ag    figures/figure1/run_alphagenome.py --n-loci 1000
run_torch figures/figure1/run_grelu.py --model borzoi   --n-loci 200
run_torch figures/figure1/run_grelu.py --model enformer --n-loci 200 --batch-size 4
run_torch figures/figure1/run_beluga.py --method ablation --sample-size 400

# --- 2. OCT-SOX ablation as log2FC (styled single-box; ChromBPNet vs AG) ------
run_torch figures/figure1/run_chrombpnet_logfc.py --num-loci 1000
run_torch figures/figure1/run_logfc_styled.py          # reads AG local + CBP arrays

# --- 3. Marginalization volcanoes (real-slope + shuffle-controlled specificity)
#     each run emits volcano_<model>.pdf AND volcano_<model>_specificity.pdf
run_torch figures/figure1/run_volcano_grelu.py --model enformer --n-bg 40
run_torch figures/figure1/run_volcano_grelu.py --model borzoi   --n-bg 40
run_ag    figures/figure1/run_volcano_alphagenome.py --n-bg 40
run_torch figures/figure1/run_volcano_replot.py        # clean labels from cached CSVs

# --- 4. TF ChIP-seq feature-leakage panel (REST-centric; a=log2FC, b=ISM, c=profiles)
run_ag    experiments/tf_chip_leakage/run_rest_leakage_figure.py --n-bg 12 --n-ism 8

echo "=== Figure 1 reproduction complete. PDFs+SVGs in figures/figure1/outputs/ ==="
