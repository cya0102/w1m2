#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python tools/build_qcec_cluster_index.py \
  --config-path config/activitynet/qcec.json \
  --splits train,val,test \
  --num-clusters 32 \
  --output data/activitynet/qcec_clusters_m32.npz

python train.py \
  --config-path config/activitynet/qcec.json \
  --log_dir logs/activitynet \
  --tag qcec_activitynet \
  --vote
