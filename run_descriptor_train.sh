#!/bin/bash
# Train descriptor score model on CPU or one GPU.
# Usage:
#   ./run_descriptor_train.sh           # auto GPU if available, else CPU
#   ./run_descriptor_train.sh cpu       # force CPU
#   ./run_descriptor_train.sh 0         # use GPU 0

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${1--1}"
if [[ "$GPU" == "cpu" ]]; then
  GPU=-1
fi

python scripts/generate_descriptor_data.py \
  --output_dir data/descriptor_demo \
  --descriptor_mode fixed

python train_descriptor.py \
  --data_dir data/descriptor_demo \
  --output_dir model_results/descriptor_demo \
  --descriptor_mode fixed \
  --gpu "$GPU"
