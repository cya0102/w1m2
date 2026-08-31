#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --config-path config/activitynet/final_weighted_diagnostic.json \
  --resume checkpoints/activitynet/stage1_old_v4_weighted_diagnostic/model-best.pt \
  --eval \
  --log_dir logs/activitynet \
  --tag final_weighted_diagnostic \
  --vote
