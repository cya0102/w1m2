#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${QSTG_PYTHON:-/home/chenyuan/miniconda3/envs/cpl/bin/python}"
cd "$ROOT"
exec "$PYTHON_BIN" train.py \
  --config-path config/charades/qstg_stage_a.json \
  --init-from-baseline checkpoints/bootstrap/charades-model-best.pt \
  --selection-strategy qstg \
  --eval-mask-mode deterministic \
  --trainable-scope qstg_only \
  --tag qstg_charades_stage_a
