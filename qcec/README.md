# QCEC

This directory is a self-contained QCEC implementation built on the copied
CPL-LREV baseline.  Runtime imports resolve to this directory; the original
baseline is not imported or read by the code.

## Build the video-level cluster index

From this directory, configure the shared HDF5 feature path and run:

```bash
python tools/build_qcec_cluster_index.py \
  --config-path config/activitynet/qcec.json \
  --splits train,val,test \
  --num-clusters 32 \
  --output data/activitynet/qcec_clusters_m32.npz
```

The index is deterministic, deduplicated by video id, and uses the same
200-step sampling helper as the dataset.

## Train and evaluate

```bash
python train.py --config-path config/activitynet/qcec.json \
  --init-from-baseline checkpoints/bootstrap/baseline.pt
python train.py --config-path config/activitynet/qcec.json --eval \
  --resume checkpoints/activitynet/<run>/model-best.pt
```

`config/activitynet/main.json` and `config/charades/main.json` remain the
baseline configurations.  QCEC is enabled only by the corresponding
`qcec.json` configuration.  The initial experiment keeps slot priors and
inference snapping disabled; they can be enabled independently after the
global adapter and crossing loss are validated.

When snapping is enabled, inner cluster boundaries are eligible at
`snap_min_confidence`; the synthetic video edges 0/1 require the independent
`snap_edge_min_confidence` threshold.

## Tests

```bash
python -m pytest -q tests
```

The tests cover contiguous clustering, dataset metadata/role masks, the QCEC
tensor contract, proposal bias compatibility, crossing loss, snapping, and
end-to-end forward/backward behavior.

## ActivityNet recovery protocol

The recovery plan uses three paired scratch runs. B is the QCEC-disabled
baseline, A enables the adapter with crossing weight 0, and C adds crossing
weight 0.1. All three use `freeze_backbone_epochs=0`, NLL selection, and
validation-only checkpoint selection. Run them serially so B can export the
public initial state used by A/C:

```bash
./scripts/run_activitynet_recovery.sh
```

Use another seed or GPU with environment variables:

```bash
SEED=18 CUDA_VISIBLE_DEVICES=1 ./scripts/run_activitynet_recovery.sh
```

Each run writes `run_metadata.json`, `resolved_config.json`, `metrics.jsonl`,
and `metrics.csv` under
`checkpoints/activitynet_recovery/<run>/`. The final Test pass is disabled
during development; after locking the variant and checkpoint rule, evaluate
the chosen checkpoint explicitly with `--eval --resume`.

To cache a fixed Validation/Test forward pass for geometry and selector
diagnostics:

```bash
python tools/cache_recovery_diagnostics.py \
  --config-path config/activitynet/recovery_c_cross.json \
  --checkpoint checkpoints/activitynet_recovery/<run>/model-best.pt \
  --split val \
  --output diagnostics/activitynet/<run>_val.npz

python tools/analyze_recovery_cache.py \
  --cache diagnostics/activitynet/<run>_val.npz \
  --output diagnostics/activitynet/<run>_val_analysis.json
```

After seeds 8/18/28 finish, summarize only the Validation-selected rows:

```bash
python tools/summarize_recovery_runs.py \
  --run B_s8=checkpoints/activitynet_recovery/<b8-run> \
  --run A_s8=checkpoints/activitynet_recovery/<a8-run> \
  --run C_s8=checkpoints/activitynet_recovery/<c8-run> \
  --output diagnostics/activitynet/recovery_summary.json
```
