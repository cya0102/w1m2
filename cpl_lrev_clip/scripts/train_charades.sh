#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

python train.py \
  --config-path config/charades/main.json \
  --log_dir logs/charades \
  --tag cpl_lrev_clip_charades \
  --vote
