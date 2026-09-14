#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --config-path config/activitynet/bpse.json \
  --selection-strategy bpse \
  --tag owlcap_activitynet_bpse \
  "$@"
