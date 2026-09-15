#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${QSTG_PYTHON:-/home/chenyuan/miniconda3/envs/cpl/bin/python}"
cd "$ROOT"
exec "$PYTHON_BIN" train.py \
  --config-path config/charades/qstg_stage_b.json \
  --init-weights "${QSTG_CHARADES_STAGE_A_CHECKPOINT:?set QSTG_CHARADES_STAGE_A_CHECKPOINT}" \
  --selection-strategy qstg \
  --eval-mask-mode deterministic \
  --trainable-scope qstg_and_proposal \
  --tag qstg_charades_stage_b
