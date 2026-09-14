# OwlCap / BPSE

`owlcap/` is an independent copy of the `cpl_lrev` runtime with the
Bidirectional Proposal-Set Equivalence (BPSE) implementation described in
`docs/videoCaption/proposal_aaai26_owlcap_detailed.md`. The source baseline is
kept unchanged; feature HDF5 paths remain configurable in JSON.

BPSE adds rule-based query units, differentiable frame/event evidence,
completeness and purity scores, a quality head, local perturbation ranking,
optional soft proposal routing, and a `bpse` evaluation selector. It does not
add caption generation, online LLM calls, HMD data, or GRPO.

## Quick checks

```bash
cd /data/chenyuan/videogrounding/w1m2/owlcap
python -m pytest tests/test_query_units.py tests/test_bpse_module.py -q
python -m pytest tests/test_bpse_loss.py tests/test_bpse_selection.py -q
```

If `pytest` is not installed, `python -m py_compile $(find owlcap -name
'*.py')` from the repository root and the same test functions can be invoked
directly with `PYTHONPATH=owlcap`.

## Configurations

`config/activitynet/baseline.json` explicitly disables both additions and is
the parity configuration. `config/activitynet/bpse.json` enables query units
and BPSE. Equivalent Charades configs are provided; its legacy validation/test
split should be treated accordingly.

```bash
cd /data/chenyuan/videogrounding/w1m2/owlcap
python train.py --config-path config/activitynet/bpse.json \
  --selection-strategy bpse --init-from-cpl /path/to/cpl_checkpoint.pt
```

For an existing BPSE checkpoint use `--resume`; `--init-from-cpl` only loads
same-name, same-shape baseline parameters and permits missing `bpse_scorer.*`
parameters. Training never consumes ground-truth timestamps.
