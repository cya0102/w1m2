#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-8}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PUBLIC_STATE="checkpoints/activitynet_recovery/public_initial_s${SEED}.pt"

if [[ -e "${PUBLIC_STATE}" ]]; then
  echo "Refusing to overwrite ${PUBLIC_STATE}; choose another SEED or move it." >&2
  exit 1
fi

python tools/validate_recovery_configs.py

echo "[1/3] ActivityNet recovery B: disabled baseline"
CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
  --config-path config/activitynet/recovery_b_baseline.json \
  --qcec-disabled \
  --qcec-cross-weight 0 \
  --selection-strategy nll \
  --select-on-val \
  --seed "${SEED}" \
  --save-initial-state "${PUBLIC_STATE}" \
  --tag recovery_b_s${SEED} \
  --log_dir logs/activitynet_recovery/b_s${SEED}

echo "[2/3] ActivityNet recovery A: adapter only"
CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
  --config-path config/activitynet/recovery_a_adapter.json \
  --qcec-enabled \
  --qcec-cross-weight 0 \
  --selection-strategy nll \
  --select-on-val \
  --seed "${SEED}" \
  --init-from-public "${PUBLIC_STATE}" \
  --tag recovery_a_s${SEED} \
  --log_dir logs/activitynet_recovery/a_s${SEED}

echo "[3/3] ActivityNet recovery C: adapter + crossing"
CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
  --config-path config/activitynet/recovery_c_cross.json \
  --qcec-enabled \
  --qcec-cross-weight 0.1 \
  --selection-strategy nll \
  --select-on-val \
  --seed "${SEED}" \
  --init-from-public "${PUBLIC_STATE}" \
  --tag recovery_c_s${SEED} \
  --log_dir logs/activitynet_recovery/c_s${SEED}

echo "Recovery B/A/C finished for seed ${SEED}."
echo "The final Test pass was intentionally disabled; evaluate locked checkpoints separately."
