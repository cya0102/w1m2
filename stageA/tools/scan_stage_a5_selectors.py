"""Validation-only hierarchical scan for the Stage-A.5 gated selector."""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from runners.stage_a5 import select_stage_a5_candidates  # noqa: E402
from tools.stage_a5_utils import (  # noqa: E402
    candidate_iou, dump_json, load_candidate_exports, ranking_metrics,
    selected_props, validate_export_protocol,
)


GEOMETRY_GRID = tuple(itertools.product(
    (0.35, 0.50, 0.65), (0.60, 0.75, 0.85), (0.10, 0.20, 0.30)))
SEMANTIC_GRID = tuple(itertools.product(
    (0.00, 0.01), (0.00, 0.01, 0.02, 0.05),
    (0.01, 0.03, 0.05), (0.50, 0.70, 0.85)))
WEIGHT_GRID = tuple(itertools.product(
    (0.5, 1.0, 2.0), (0.1, 0.25, 0.5),
    (0.25, 0.5, 1.0), (0.0, 0.1, 0.25), (0.0, 0.01, 0.02)))


def _base_config():
    return {
        'enabled': True,
        'selector': 'counterfactual_gated',
        'allowed_candidate_types': [
            'left_near', 'left_strong', 'right_near', 'right_strong'],
        'min_parent_width': 0.50,
        'min_retained_ratio': 0.75,
        'max_relative_shift': 0.20,
        'max_nll_increase': 0.00,
        'min_contrast_margin': 0.02,
        'max_contrast_std': 0.03,
        'min_boundary_percentile': 0.70,
        'lambda_contrast': 1.0,
        'lambda_edit': 0.25,
        'lambda_uncertainty': 0.5,
        'lambda_boundary': 0.1,
        'accept_margin': 0.01,
    }


def _evaluate(features, labels, config, event_weight=0.0):
    start = features['candidate_start']
    end = features['candidate_end']
    valid = features['candidate_valid']
    nll = features['candidate_nll_mean']
    nll_std = features['candidate_nll_std']
    shell = features['candidate_shell_nll_mean']
    shell_std = features['candidate_shell_nll_std']
    confidence = features['candidate_boundary_confidence']
    contrast = features['candidate_contrast_mean']
    contrast_std = features['candidate_contrast_std']
    refined, selector_score, selected_index, reasons = (
        select_stage_a5_candidates(
            start, end, valid, nll, nll_std, shell, shell_std, confidence,
            config, contrast_mean=contrast, contrast_std=contrast_std))
    gt = labels['gt_normalized']
    iou = candidate_iou(features, labels)
    original_props = np.stack([
        features['parent_start'], features['parent_end']], axis=-1)
    original_scores = nll[..., 0] - event_weight * features['parent_event_score']
    selected_nll = np.take_along_axis(
        nll, selected_index[..., None], axis=-1)[..., 0]
    selected_scores = selected_nll - event_weight * features['parent_event_score']
    stage_metrics = ranking_metrics(refined, gt, selected_scores)
    baseline_metrics = ranking_metrics(original_props, gt, original_scores)
    original_iou = iou[..., 0]
    selected_iou = np.take_along_axis(
        iou, selected_index[..., None], axis=-1)[..., 0]
    changed = selected_index != 0
    delta_iou = selected_iou - original_iou
    helpful = changed & (delta_iou > 0.01)
    harmful = changed & (delta_iou < -0.01)
    gt_width = labels['gt_normalized'][:, 1] - labels['gt_normalized'][:, 0]
    long = gt_width >= 0.60
    bucket_gate = True
    bucket_drops = {}
    for name, lower, upper in (
            ('short', 0.0, 0.15), ('medium_short', 0.15, 0.35),
            ('medium_long', 0.35, 0.60), ('long', 0.60, float('inf'))):
        bucket = (gt_width >= lower) & (gt_width < upper)
        if not bucket.any():
            continue
        base_bucket = ranking_metrics(
            original_props[bucket], gt[bucket], original_scores[bucket])
        stage_bucket = ranking_metrics(
            refined[bucket], gt[bucket], selected_scores[bucket])
        r1_drop = stage_bucket['R@1,mIoU'] - base_bucket['R@1,mIoU']
        r5_drop = stage_bucket['R@5,mIoU'] - base_bucket['R@5,mIoU']
        bucket_drops[name] = {'R1_mIoU': float(r1_drop),
                              'R5_mIoU': float(r5_drop)}
        bucket_gate &= r1_drop >= -0.01 and r5_drop >= -0.01
    if long.any():
        baseline_long = ranking_metrics(
            original_props[long], gt[long], original_scores[long])
        stage_long = ranking_metrics(
            refined[long], gt[long], selected_scores[long])
        long_penalty = max(0.0, baseline_long['R@5,mIoU'] -
                           stage_long['R@5,mIoU'])
    else:
        long_penalty = 0.0
    objective = (
        stage_metrics['R@1,mIoU'] - baseline_metrics['R@1,mIoU'] +
        0.5 * (stage_metrics['R@5,mIoU'] - baseline_metrics['R@5,mIoU']) +
        0.5 * (stage_metrics['R@1,IoU@0.5'] -
               baseline_metrics['R@1,IoU@0.5']) -
        2.0 * harmful.mean() - long_penalty)
    row = {
        'objective': float(objective),
        'changed_fraction': float(changed.mean()),
        'helpful_fraction': float(helpful.mean()),
        'harmful_fraction': float(harmful.mean()),
        'trim_precision': float(helpful.sum() /
                                max(int((helpful | harmful).sum()), 1)),
        'long_R5_penalty': float(long_penalty),
        'bucket_drops': bucket_drops,
        'baseline_R1_mIoU': baseline_metrics['R@1,mIoU'],
        'stage_R1_mIoU': stage_metrics['R@1,mIoU'],
        'baseline_R5_mIoU': baseline_metrics['R@5,mIoU'],
        'stage_R5_mIoU': stage_metrics['R@5,mIoU'],
        'baseline_R1_IoU@0.5': baseline_metrics['R@1,IoU@0.5'],
        'stage_R1_IoU@0.5': stage_metrics['R@1,IoU@0.5'],
        'stage_b_go': bool(
            stage_metrics['R@1,mIoU'] >= baseline_metrics['R@1,mIoU'] and
            stage_metrics['R@5,mIoU'] >= baseline_metrics['R@5,mIoU'] and
            (stage_metrics['R@1,IoU@0.5'] > baseline_metrics['R@1,IoU@0.5'] or
             stage_metrics['R@1,IoU@0.7'] > baseline_metrics.get(
                 'R@1,IoU@0.7', stage_metrics['R@1,IoU@0.7'])) and
            helpful.sum() > harmful.sum() and
            helpful.sum() / max(int((helpful | harmful).sum()), 1) >= 0.60 and
            long_penalty <= 0.01 and bucket_gate),
    }
    return row, refined, selected_index, reasons


def _with(config, **changes):
    result = dict(config)
    result.update(changes)
    return result


def _unique_top(rows, limit):
    rows = sorted(rows, key=lambda row: row['objective'], reverse=True)
    result = []
    seen = set()
    for row in rows:
        key = json.dumps(row['config'], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def _iter_weight_configs(parents):
    """Yield every weight-grid configuration for every semantic parent."""
    for parent in parents:
        for (lambda_contrast, lambda_edit, lambda_uncertainty,
             lambda_boundary, accept_margin) in WEIGHT_GRID:
            yield _with(
                parent['config'], lambda_contrast=lambda_contrast,
                lambda_edit=lambda_edit, lambda_uncertainty=lambda_uncertainty,
                lambda_boundary=lambda_boundary, accept_margin=accept_margin)


def _choose_selected_row(rows, diagnostic_only=False):
    """Apply gate-first selection and return the selected row metadata."""
    if not rows:
        raise ValueError('selector scan produced no configurations')
    gate_passing_rows = [row for row in rows if row['stage_b_go']]
    if diagnostic_only:
        best = max(rows, key=lambda row: row['objective'])
        selection_reason = 'diagnostic_only_best_objective_no_formal_gate'
        selected_stage_b_go = False
    elif gate_passing_rows:
        best = max(gate_passing_rows, key=lambda row: row['objective'])
        selection_reason = 'best_gate_passing_config_by_objective'
        selected_stage_b_go = True
    else:
        best = max(rows, key=lambda row: row['objective'])
        selection_reason = 'no_gate_passing_config_best_objective_with_gate_false'
        selected_stage_b_go = False
    return best, len(gate_passing_rows), selection_reason, selected_stage_b_go


def _write_scan_outputs(all_rows, metadata, diagnostic_only, args,
                        fixed_rule_diagnostic=False):
    best, num_gate_passing_configs, selection_reason, selected_stage_b_go = (
        _choose_selected_row(all_rows, diagnostic_only=diagnostic_only))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'phase', 'objective', 'changed_fraction',
        'helpful_fraction', 'harmful_fraction', 'trim_precision',
        'long_R5_penalty', 'baseline_R1_mIoU', 'stage_R1_mIoU',
        'baseline_R5_mIoU', 'stage_R5_mIoU',
        'baseline_R1_IoU@0.5', 'stage_R1_IoU@0.5',
        'stage_b_go', 'bucket_drops', 'config']
    with output.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                **row,
                'bucket_drops': json.dumps(row['bucket_drops'], sort_keys=True),
                'config': json.dumps(row['config'], sort_keys=True),
            })
    selected_path = Path(args.selected_config_output)
    summary_key = 'diagnostic_summary' if diagnostic_only else 'validation_summary'
    selected_payload = {
        'config': best['config'],
        summary_key: {key: value for key, value in best.items()
                      if key not in {'config'}},
        'num_scanned_configs': len(all_rows),
        'num_gate_passing_configs': num_gate_passing_configs,
        'selection_reason': selection_reason,
        'STAGE_B_GO': selected_stage_b_go,
        'diagnostic_only': bool(diagnostic_only),
        'protocol_metadata': metadata,
        'selection_protocol': (
            'fixed-rule diagnostic; cannot support Stage B decisions'
            if fixed_rule_diagnostic else
            'diagnostic-only hierarchical Stage-A.5 scan; cannot support '
            'Stage B decisions' if diagnostic_only else
            'validation-only hierarchical Stage-A.5 scan'),
    }
    dump_json(selected_path, selected_payload)
    print('saved selector scan to {}'.format(output))
    print('saved selected config to {}'.format(selected_path))
    return selected_payload


def scan(args):
    features, labels = load_candidate_exports(args.features, args.labels)
    metadata, diagnostic_only = validate_export_protocol(
        features, labels,
        allow_partial_smoke=getattr(args, 'allow_partial_smoke', False),
        allow_test_diagnostic=getattr(args, 'allow_test_diagnostic', False),
        operation='selector scan')
    config = _base_config()
    if metadata['validation_is_test']:
        row, _, _, _ = _evaluate(
            features, labels, config, event_weight=args.event_weight)
        row['phase'] = 'diagnostic_fixed_rule'
        row['config'] = config
        return _write_scan_outputs(
            [row], metadata, True, args, fixed_rule_diagnostic=True)
    geometry_rows = []
    for min_parent, min_ratio, max_shift in GEOMETRY_GRID:
        current = _with(
            config, min_parent_width=min_parent,
            min_retained_ratio=min_ratio, max_relative_shift=max_shift)
        row, _, _, _ = _evaluate(
            features, labels, current, event_weight=args.event_weight)
        row['phase'] = 'geometry'
        row['config'] = current
        geometry_rows.append(row)
    semantic_rows = []
    for parent in _unique_top(geometry_rows, min(args.top_geometry, len(geometry_rows))):
        for max_nll, min_contrast, max_std, min_boundary in SEMANTIC_GRID:
            current = _with(
                parent['config'], max_nll_increase=max_nll,
                min_contrast_margin=min_contrast, max_contrast_std=max_std,
                min_boundary_percentile=min_boundary)
            row, _, _, _ = _evaluate(
                features, labels, current, event_weight=args.event_weight)
            row['phase'] = 'semantic'
            row['config'] = current
            semantic_rows.append(row)
    weight_rows = []
    semantic_parents = _unique_top(
        semantic_rows, min(args.top_semantic, len(semantic_rows)))
    for current in _iter_weight_configs(semantic_parents):
        row, _, _, _ = _evaluate(
            features, labels, current, event_weight=args.event_weight)
        row['phase'] = 'weights'
        row['config'] = current
        weight_rows.append(row)
    all_rows = geometry_rows + semantic_rows + weight_rows
    return _write_scan_outputs(all_rows, metadata, diagnostic_only, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--selected-config-output', required=True)
    parser.add_argument('--event-weight', type=float, default=0.0)
    parser.add_argument('--top-geometry', type=int, default=10)
    parser.add_argument('--top-semantic', type=int, default=5)
    parser.add_argument('--allow-partial-smoke', action='store_true',
                        help='read a partial export as diagnostic-only')
    parser.add_argument('--allow-test-diagnostic', action='store_true',
                        help='read a test-as-validation export as diagnostic-only')
    args = parser.parse_args()
    if args.top_geometry < 1 or args.top_semantic < 1:
        raise ValueError('top-k scan limits must be positive')
    scan(args)


if __name__ == '__main__':
    main()
