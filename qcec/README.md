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
