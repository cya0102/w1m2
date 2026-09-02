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

