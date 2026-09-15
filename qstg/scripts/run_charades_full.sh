#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${QSTG_PYTHON:-/home/chenyuan/miniconda3/envs/cpl/bin/python}"
cd "$ROOT"
exec "$PYTHON_BIN" train.py \
  --config-path config/charades/qstg_full.json \
  --init-weights "${QSTG_CHARADES_STAGE_B_CHECKPOINT:?set QSTG_CHARADES_STAGE_B_CHECKPOINT}" \
  --selection-strategy qstg \
  --eval-mask-mode deterministic \
  --trainable-scope all \
  --tag qstg_charades_full
