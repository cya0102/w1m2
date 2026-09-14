#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [extra train.py args...]" >&2
  exit 2
fi
checkpoint="$1"
shift
python train.py \
  --config-path config/activitynet/bpse.json \
  --selection-strategy bpse \
  --resume "$checkpoint" \
  --eval \
  "$@"
