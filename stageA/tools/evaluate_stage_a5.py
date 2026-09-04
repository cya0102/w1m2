"""Evaluate one frozen Stage-A.5 selector on an exported split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from runners.stage_a5 import CANDIDATE_NAMES, REASON_CODES, select_stage_a5_candidates  # noqa: E402
from tools.stage_a5_utils import (  # noqa: E402
    candidate_iou, dump_json, load_candidate_exports, ranking_metrics,
    selected_props, validate_export_protocol, video_cluster_bootstrap,
)


def evaluate(args):
    features, labels = load_candidate_exports(args.features, args.labels)
    metadata, diagnostic_only = validate_export_protocol(
        features, labels,
        allow_partial_smoke=getattr(args, 'allow_partial_smoke', False),
        allow_test_diagnostic=getattr(args, 'allow_test_diagnostic', False),
        operation='frozen evaluation')
    with Path(args.selected_config).open(encoding='utf8') as handle:
        selected = json.load(handle)
    config = selected.get('config', selected)
    refined, selector_score, selected_index, reasons = (
        select_stage_a5_candidates(
            features['candidate_start'], features['candidate_end'],
            features['candidate_valid'], features['candidate_nll_mean'],
            features['candidate_nll_std'], features['candidate_shell_nll_mean'],
            features['candidate_shell_nll_std'],
            features['candidate_boundary_confidence'], config,
            contrast_mean=features['candidate_contrast_mean'],
            contrast_std=features['candidate_contrast_std']))
    gt = labels['gt_normalized']
    iou = candidate_iou(features, labels)
    baseline_props = np.stack([
        features['parent_start'], features['parent_end']], axis=-1)
    parent_event_score = features['parent_event_score']
    nll = features['candidate_nll_mean']
    baseline_scores = nll[..., 0] - args.event_weight * parent_event_score
    selected_nll = np.take_along_axis(nll, selected_index[..., None], axis=-1)[..., 0]
    stage_scores = selected_nll - args.event_weight * parent_event_score
    baseline = ranking_metrics(baseline_props, gt, baseline_scores)
    stage = ranking_metrics(refined, gt, stage_scores)
    original_iou = iou[..., 0]
    selected_iou = np.take_along_axis(iou, selected_index[..., None], axis=-1)[..., 0]
    changed = selected_index != 0
    delta_iou = selected_iou - original_iou
    helpful = changed & (delta_iou > 0.01)
    harmful = changed & (delta_iou < -0.01)
    neutral = changed & ~(helpful | harmful)
    gt_width = gt[:, 1] - gt[:, 0]
    bucket = {}
    for name, lower, upper in (
            ('short', 0.0, 0.15), ('medium_short', 0.15, 0.35),
            ('medium_long', 0.35, 0.60), ('long', 0.60, float('inf'))):
        mask = (gt_width >= lower) & (gt_width < upper)
        if mask.any():
            bucket[name] = {
                'count': int(mask.sum()),
                'baseline_R1_mIoU': float(baseline['r1_iou'][mask].mean()),
                'stage_a5_R1_mIoU': float(stage['r1_iou'][mask].mean()),
                'delta_R1_mIoU': float((stage['r1_iou'][mask] -
                                        baseline['r1_iou'][mask]).mean()),
                'baseline_R5_mIoU': float(baseline['r5_iou'][mask].mean()),
                'stage_a5_R5_mIoU': float(stage['r5_iou'][mask].mean()),
                'delta_R5_mIoU': float((stage['r5_iou'][mask] -
                                        baseline['r5_iou'][mask]).mean()),
            }
    bootstrap = video_cluster_bootstrap(
        baseline_props, refined, gt, features['video_ids'],
        baseline_scores=baseline_scores, stage_scores=stage_scores,
        repeats=args.bootstrap_replicates, seed=args.bootstrap_seed)
    reason_names = {value: key for key, value in REASON_CODES.items()}
    summary = {
        'dataset': str(features['dataset'].reshape(-1)[0]),
        'split': str(features['split'].reshape(-1)[0]),
        'checkpoint_path': str(features['checkpoint_path'].reshape(-1)[0]),
        'checkpoint_sha256': str(features['checkpoint_sha256'].reshape(-1)[0]),
        'config_sha256': str(features['config_sha256'].reshape(-1)[0]),
        'mask_seeds': features['mask_seeds'].astype(int).tolist(),
        'query_count': int(len(gt)),
        'video_count': int(np.unique(features['video_ids']).size),
        'diagnostic_only': bool(diagnostic_only),
        'protocol_metadata': metadata,
        'baseline_metrics': {key: value for key, value in baseline.items()
                             if key.startswith('R@')},
        'stage_a5_metrics': {key: value for key, value in stage.items()
                             if key.startswith('R@')},
        'delta_metrics': {
            key: stage[key] - baseline[key]
            for key in ('R@1,mIoU', 'R@1,IoU@0.5', 'R@5,mIoU',
                        'R@5,IoU@0.5')},
        'changed_fraction': float(changed.mean()),
        'mean_width_before': float((baseline_props[..., 1] -
                                    baseline_props[..., 0]).mean()),
        'mean_width_after': float((refined[..., 1] - refined[..., 0]).mean()),
        'helpful_fraction': float(helpful.mean()),
        'harmful_fraction': float(harmful.mean()),
        'neutral_fraction': float(neutral.mean()),
        'trim_precision': float(helpful.sum() /
                                max(int((helpful | harmful).sum()), 1)),
        'selected_candidate_distribution': {
            CANDIDATE_NAMES[index]: float((selected_index == index).mean())
            for index in range(7)},
        'reason_distribution': {
            reason_names.get(int(index), str(int(index))): float(
                (reasons == index).mean())
            for index in np.unique(reasons)},
        'gt_width_buckets': bucket,
        'video_cluster_bootstrap': bootstrap,
        'STAGE_B_GO': bool(selected.get('STAGE_B_GO', False)) and
        not diagnostic_only,
        'stage_b_gate_note': (
            'Diagnostic-only output; partial/test-as-validation data cannot '
            'support Stage B decisions.' if diagnostic_only else
            'This report used the validation-frozen configuration; no test '
            'parameter scan was performed.'),
    }
    dump_json(args.output, summary)
    print('saved Stage-A.5 evaluation report to {}'.format(args.output))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--selected-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--event-weight', type=float, default=0.0)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260902)
    parser.add_argument('--allow-partial-smoke', action='store_true',
                        help='read a partial export as diagnostic-only')
    parser.add_argument('--allow-test-diagnostic', action='store_true',
                        help='read a test-as-validation export as diagnostic-only')
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError('--bootstrap-replicates must be positive')
    evaluate(args)


if __name__ == '__main__':
    main()
