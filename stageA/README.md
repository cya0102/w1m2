# Stage A: event-boundary minimal-sufficient trimming

This directory is an in-place diagnostic copy of the `cpl_lrev` execution
tree.  It is intentionally not an installable or importable incremental
package; run commands from this directory so its copied `models/`,
`datasets/`, and `runners/` modules are used.  The original `cpl_lrev` source
is not modified.

The compact indices are already generated in `data/event_boundaries/`.  To
rebuild them from the source JSON:

```bash
cd /data/chenyuan/videogrounding/w1m2/stageA
PYTHONPATH=. python tools/build_event_boundary_index.py \
  --input ../CPL_baseline-main/eventBoundary/outputs/activitynet_c3d_event_boundaries.json \
  --output data/event_boundaries/activitynet_c3d_stage_a.npz \
  --min-gap-clips 2

PYTHONPATH=. python tools/build_event_boundary_index.py \
  --input ../CPL_baseline-main/eventBoundary/outputs/charades_i3d_event_boundaries.json \
  --output data/event_boundaries/charades_i3d_stage_a.npz \
  --min-gap-clips 2
```

Use `config/activitynet/stage_a.json` or `config/charades/stage_a.json` for
inference.  Stage A is eval-only, uses the same masked query as baseline, and
defaults to report-only evaluation across epsilon values `0`, `0.01`, `0.02`,
and `0.05`.

## Stage A.5

Stage A.5 is also inference-only.  First export deterministic three-mask
candidate features and the separate GT labels file, then run the Oracle
report and validation-only selector scan.  The scan never feeds labels to the
selector.

Formal reports are protocol-protected.  The exporter records `partial`,
`query_count`, `train_data`, `val_data`, `test_data`, and
`validation_is_test` in `metadata_json`.  Oracle analysis, selector scan, and
frozen evaluation reject `partial=true` by default.  Pass
`--allow-partial-smoke` only for a clearly diagnostic smoke report; it forces
`diagnostic_only=true` and `STAGE_B_GO=false`.  A Charades configuration with
`val_data == test_data` is rejected by default.  Its optional
`--allow-test-diagnostic` mode runs only the fixed shipped selector rule,
marks the output diagnostic-only, and cannot produce a Stage-B decision.

The selector scan fully evaluates the weight grid for every retained semantic
parent.  Configuration selection is gate-first: a gate-passing configuration
is preferred over any higher-objective non-passing configuration.  If no
configuration passes, the best objective is saved with `STAGE_B_GO=false`.
The selected JSON records `num_scanned_configs`,
`num_gate_passing_configs`, and `selection_reason`.

```bash
cd /data/chenyuan/videogrounding/w1m2/stageA

PYTHONPATH=. python tools/dump_stage_a5_candidates.py \
  --config-path config/activitynet/stage_a5.json \
  --checkpoint checkpoints/activitynet/<run>/model-best-r1.pt \
  --checkpoint-label activitynet_best_r1 \
  --split val --mask-seeds 8,18,28 \
  --output outputs/stage_a5/activitynet/activitynet_best_r1/val_candidates

PYTHONPATH=. python tools/analyze_stage_a5_oracle.py \
  --features outputs/stage_a5/activitynet/activitynet_best_r1/val_candidates_features.npz \
  --labels outputs/stage_a5/activitynet/activitynet_best_r1/val_candidates_labels.npz \
  --rows-output outputs/stage_a5/activitynet/activitynet_best_r1/val_oracle_rows.csv \
  --summary-output outputs/stage_a5/activitynet/activitynet_best_r1/val_oracle_summary.json \
  --bootstrap-replicates 2000

PYTHONPATH=. python tools/scan_stage_a5_selectors.py \
  --features outputs/stage_a5/activitynet/activitynet_best_r1/val_candidates_features.npz \
  --labels outputs/stage_a5/activitynet/activitynet_best_r1/val_candidates_labels.npz \
  --output outputs/stage_a5/activitynet/activitynet_best_r1/selector_scan.csv \
  --selected-config-output outputs/stage_a5/activitynet/activitynet_best_r1/selected_stage_a5_config.json
```

After the validation configuration is frozen, export the test split once and
evaluate it without scanning:

```bash
PYTHONPATH=. python tools/dump_stage_a5_candidates.py \
  --config-path config/activitynet/stage_a5.json \
  --checkpoint checkpoints/activitynet/<run>/model-best-r1.pt \
  --split test --mask-seeds 8,18,28 \
  --output outputs/stage_a5/activitynet/activitynet_best_r1/test_candidates

PYTHONPATH=. python tools/evaluate_stage_a5.py \
  --features outputs/stage_a5/activitynet/activitynet_best_r1/test_candidates_features.npz \
  --labels outputs/stage_a5/activitynet/activitynet_best_r1/test_candidates_labels.npz \
  --selected-config outputs/stage_a5/activitynet/activitynet_best_r1/selected_stage_a5_config.json \
  --output outputs/stage_a5/activitynet/activitynet_best_r1/test_report.json
```

Use the analogous `config/charades/stage_a5.json` and
`outputs/stage_a5/charades/` paths for Charades.  The shipped Charades config
retains the existing diagnostic split (`val_data == test_data`); it must not
be used for a formal validation claim.  For a formal protocol, create a
video-disjoint split and manifest with:

```bash
PYTHONPATH=. python tools/make_charades_val_split.py \
  --input data/charades/train.json \
  --train-output data/charades/train_formal_stage_a5.json \
  --val-output data/charades/val_formal_stage_a5.json \
  --manifest outputs/stage_a5/charades/charades_val_manifest.json \
  --seed 20260902
```

Run the Stage-A.5 regression suite from the project root with:

```bash
cd /data/chenyuan/videogrounding/w1m2
/home/chenyuan/miniconda3/envs/cpl/bin/python -m pytest -q stageA/tests
```
