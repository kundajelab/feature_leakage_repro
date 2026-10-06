#!/usr/bin/env bash
# Create the conda environments needed to reproduce the Feature-Leakage figures.
#
# Two creation modes:
#   --clone     (default)  Clone the already-working lab envs. Fast and exact on
#                          the original cluster; relies on those envs existing.
#   --from-yml             Solve fresh from the exported environment.yml files.
#                          Portable to other machines but slower and subject to
#                          solver/CUDA availability.
#
# Envs created (all NEW; the script never modifies existing envs):
#   featleak-dl           PyTorch 2.7 + tangermeme 0.5 + bpnet-lite + gReLU +
#                         modiscolite + alphagenome API client. Workhorse for
#                         Figs 2 (ChromBPNet/Borzoi/Enformer), 3, 4, 5, 6, and
#                         the Beluga parts of Fig 1.
#   featleak-alphagenome  JAX + local AlphaGenome v1 model. AlphaGenome panels
#                         of Figs 1 and 2.
#   featleak-agtorch      PyTorch 2.5 + alphagenome-pytorch (all-folds v1 weights).
#                         Full-length (524,288 bp) AlphaGenome runs: Figs 1c, 2b and
#                         the Fig 7 fine-tuning.
#   featleak-tf           TensorFlow/Keras 2.15 + tf-modisco. Optional; only for
#                         the TF-Keras variants matching the Methods text.
#
# Usage:
#   bash setup_envs.sh                 # clone all four
#   bash setup_envs.sh --from-yml      # build all four from yml
#   bash setup_envs.sh --clone featleak-dl   # just one
set -euo pipefail

MODE="--clone"
declare -a WANT=(featleak-dl featleak-alphagenome featleak-agtorch featleak-tf)
declare -A SRC=(
  [featleak-dl]=alphagenome-env
  [featleak-alphagenome]=alphagenome_research
  [featleak-agtorch]=agpt
  [featleak-tf]=feat_leak
)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=()
for a in "$@"; do
  case "$a" in
    --clone|--from-yml) MODE="$a" ;;
    featleak-*) args+=("$a") ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done
[ ${#args[@]} -gt 0 ] && WANT=("${args[@]}")

for env in "${WANT[@]}"; do
  if conda env list | grep -qE "/${env}\$| ${env} "; then
    echo "== $env already exists, skipping =="
    continue
  fi
  if [ "$MODE" = "--clone" ]; then
    echo "== cloning ${SRC[$env]} -> $env =="
    conda create --clone "${SRC[$env]}" -n "$env" -y
  else
    echo "== creating $env from yml =="
    conda env create -f "$HERE/${env}.environment.yml"
  fi
done
echo "done. activate with:  conda activate featleak-dl"
