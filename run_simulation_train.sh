#!/bin/bash
# Generate simulation data and train the original 2D U-Net diffusion factor model.
#
# Usage:
#   ./run_simulation_train.sh              # paper preset, GPU 0
#   ./run_simulation_train.sh small        # small 8x8 quick test
#   ./run_simulation_train.sh paper cpu    # force CPU
#   ./run_simulation_train.sh small 0 50   # small preset, GPU 0, 50 epochs

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

PRESET="${1:-paper}"
GPU="${2:-0}"
EPOCHS="${3:-}"

OUT_DIR="simulation_experiment_data/${PRESET}"
DATA_PATH="${OUT_DIR}/training_data.npy"

python scripts/generate_simulation_data.py \
  --output_dir "${OUT_DIR}" \
  --preset "${PRESET}"

TRAIN_ARGS=(
  --data_path "${DATA_PATH}"
  --gpu "${GPU}"
)

if [[ -n "${EPOCHS}" ]]; then
  TRAIN_ARGS+=(--epochs "${EPOCHS}")
fi

python train.py "${TRAIN_ARGS[@]}"
