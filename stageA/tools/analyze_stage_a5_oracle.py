"""Offline candidate Oracle, legacy-regret, and NLL diagnostics."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from runners.stage_a5 import CANDIDATE_NAMES  # noqa: E402
from tools.stage_a5_utils import (  # noqa: E402
    candidate_iou, dump_json, load_candidate_exports, ranking_metrics,
    roc_auc, selected_props, sha256_file, spearman_correlation,
    validate_export_protocol, video_cluster_bootstrap,
)


def _legacy_indices(features, epsilon):
    starts = np.asarray(features['candidate_start'])
    ends = np.asarray(features['candidate_end'])
    valid = np.asarray(features['candidate_valid']).astype(bool)
    nll = np.asarray(features['candidate_nll_mean'], dtype=np.float64)
    width = ends - starts
    eligible = valid & np.isfinite(nll) & (
        nll <= nll[..., :1] + float(epsilon))
    eligible[..., 0] = True
    index = np.zeros(width.shape[:2], dtype=np.int8)
    best_width = width[..., 0].copy()
    best_nll = nll[..., 0].copy()
    for candidate_index in range(1, 7):
        same_width = np.abs(width[..., candidate_index] - best_width) <= 1e-6
        better = eligible[..., candidate_index] & (
            (width[..., candidate_index] < best_width - 1e-6) |
            (same_width & (nll[..., candidate_index] < best_nll)))
        index = np.where(better, candidate_index, index).astype(np.int8)
        best_width = np.where(better, width[..., candidate_index], best_width)
        best_nll = np.where(better, nll[..., candidate_index], best_nll)
    return index


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float('nan')


def _bucket_report(values, iou, names, bounds):
    values = np.asarray(values)
    result = {}
    for name, (lower, upper) in zip(names, bounds):
        mask = (values >= lower) & (values < upper) & \
            np.isfinite(values) & np.isfinite(iou)
        if mask.any():
            result[name] = {
                'count': int(mask.sum()),
                'mIoU': float(np.mean(iou[mask])),
                'IoU@0.5': float(np.mean(iou[mask] >= 0.5)),
            }
    return result


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf8')
        return
    fields = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    features, labels = load_candidate_exports(args.features, args.labels)
    metadata, diagnostic_only = validate_export_protocol(
        features, labels,
        allow_partial_smoke=getattr(args, 'allow_partial_smoke', False),
        allow_test_diagnostic=getattr(args, 'allow_test_diagnostic', False),
        operation='oracle analysis')
    iou = candidate_iou(features, labels)
    valid = np.asarray(features['candidate_valid']).astype(bool)
    q, num_props, num_candidates = iou.shape
    if num_candidates != 7:
        raise ValueError('Stage-A.5 analysis expects seven candidates')
    gt = np.asarray(labels['gt_normalized'], dtype=np.float64)
    sample_ids = np.asarray(features['sample_ids']).astype(str)
    video_ids = np.asarray(features['video_ids']).astype(str)
    candidate_nll = np.asarray(features['candidate_nll_mean'], dtype=np.float64)
    candidate_nll_std = np.asarray(features['candidate_nll_std'], dtype=np.float64)
    shell_nll = np.asarray(features['candidate_shell_nll_mean'], dtype=np.float64)
    shell_nll_std = np.asarray(features['candidate_shell_nll_std'], dtype=np.float64)
    contrast = np.asarray(features['candidate_contrast_mean'], dtype=np.float64)
    contrast_std = np.asarray(features['candidate_contrast_std'], dtype=np.float64)
    boundary_confidence = np.asarray(
        features['candidate_boundary_confidence'], dtype=np.float64)

    best_index = np.argmax(iou, axis=-1).astype(np.int8)
    best_iou = np.take_along_axis(iou, best_index[..., None], axis=-1)[..., 0]
    original_iou = iou[..., 0]
    parent_gain = best_iou - original_iou
    legacy_index = np.asarray(features['legacy_selected_index']).astype(np.int8)
    legacy_iou = np.take_along_axis(
        iou, legacy_index[..., None], axis=-1)[..., 0]
    legacy_delta = legacy_iou - original_iou
    legacy_regret = best_iou - legacy_iou
    changed = legacy_index != 0
    helpful = changed & (legacy_delta > 0.01)
    harmful = changed & (legacy_delta < -0.01)
    neutral = changed & ~(helpful | harmful)
    changed_count = max(int(changed.sum()), 1)

    original_props = np.stack([
        np.asarray(features['parent_start']),
        np.asarray(features['parent_end']),
    ], axis=-1)
    baseline_scores = candidate_nll[..., 0] - args.event_weight * np.asarray(
        features['parent_event_score'])
    legacy_props = selected_props(features, legacy_index)
    legacy_scores = np.take_along_axis(candidate_nll, legacy_index[..., None],
                                        axis=-1)[..., 0] - args.event_weight * np.asarray(
                                            features['parent_event_score'])
    baseline_rank = ranking_metrics(original_props, gt, baseline_scores)
    legacy_rank = ranking_metrics(legacy_props, gt, legacy_scores)

    original_set = original_iou.max(axis=1)
    candidate_set = best_iou.max(axis=1)
    candidate_set_metrics = {
        'original_set_mIoU': float(original_set.mean()),
        'candidate_set_oracle_mIoU': float(candidate_set.mean()),
        'candidate_set_oracle_gain': float((candidate_set - original_set).mean()),
        'original_set_IoU@0.3': float((original_set >= 0.3).mean()),
        'candidate_set_oracle_IoU@0.3': float((candidate_set >= 0.3).mean()),
        'original_set_IoU@0.5': float((original_set >= 0.5).mean()),
        'candidate_set_oracle_IoU@0.5': float((candidate_set >= 0.5).mean()),
        'original_set_IoU@0.7': float((original_set >= 0.7).mean()),
        'candidate_set_oracle_IoU@0.7': float((candidate_set >= 0.7).mean()),
    }

    trim_mask = valid.copy()
    trim_mask[..., 0] = False
    delta_nll = candidate_nll - candidate_nll[..., :1]
    delta_iou = iou - original_iou[..., None]
    finite_trim = trim_mask & np.isfinite(delta_nll) & np.isfinite(delta_iou)
    helpful_trim = finite_trim & (delta_iou > 0.01)
    nll_report = {
        'spearman_delta_nll_delta_iou': spearman_correlation(
            delta_nll[finite_trim], delta_iou[finite_trim]),
        'roc_auc_neg_delta_nll_helpful_trim': roc_auc(
            -delta_nll[finite_trim], (delta_iou[finite_trim] > 0.01)),
        'valid_trim_count': int(finite_trim.sum()),
        'candidate_nll_std_mean': _safe_mean(candidate_nll_std[finite_trim]),
        'candidate_contrast_std_mean': _safe_mean(contrast_std[finite_trim]),
        'helpful_trim_fraction': float(helpful_trim.sum() /
                                       max(int(finite_trim.sum()), 1)),
    }
    decile_report = []
    finite_delta_nll = delta_nll[finite_trim]
    finite_delta_iou = delta_iou[finite_trim]
    if finite_delta_nll.size:
        edges = np.quantile(finite_delta_nll, np.linspace(0, 1, 11))
        for decile in range(10):
            mask = ((finite_delta_nll >= edges[decile]) &
                    (finite_delta_nll <= edges[decile + 1] if decile == 9 else
                     finite_delta_nll < edges[decile + 1]))
            decile_report.append({
                'decile': decile + 1,
                'count': int(mask.sum()),
                'mean_delta_nll': float(finite_delta_nll[mask].mean()) if mask.any() else None,
                'mean_delta_iou': float(finite_delta_iou[mask].mean()) if mask.any() else None,
            })
    nll_report['delta_nll_deciles'] = decile_report
    uncertainty_report = []
    finite_std = candidate_nll_std[finite_trim]
    if finite_std.size:
        std_edges = np.quantile(finite_std, [0.0, 1 / 3, 2 / 3, 1.0])
        for bucket_index in range(3):
            std_mask = ((finite_std >= std_edges[bucket_index]) &
                        (finite_std <= std_edges[bucket_index + 1]
                         if bucket_index == 2 else
                         finite_std < std_edges[bucket_index + 1]))
            if not std_mask.any():
                continue
            uncertainty_report.append({
                'bucket': bucket_index + 1,
                'count': int(std_mask.sum()),
                'mean_candidate_nll_std': float(finite_std[std_mask].mean()),
                'harmful_fraction': float(
                    (finite_delta_iou[std_mask] < -0.01).mean()),
                'helpful_fraction': float(
                    (finite_delta_iou[std_mask] > 0.01).mean()),
            })
    nll_report['candidate_nll_std_buckets'] = uncertainty_report

    epsilon_report = {}
    for epsilon in (0.0, 0.01, 0.02, 0.05):
        index = _legacy_indices(features, epsilon)
        selected = np.take_along_axis(iou, index[..., None], axis=-1)[..., 0]
        change = index != 0
        d = selected - original_iou
        h = change & (d > 0.01)
        harm = change & (d < -0.01)
        epsilon_report[str(epsilon)] = {
            'changed_fraction': float(change.mean()),
            'helpful_fraction_of_all': float(h.mean()),
            'harmful_fraction_of_all': float(harm.mean()),
            'neutral_fraction_of_all': float((change & ~(h | harm)).mean()),
            'trim_precision': float(h.sum() / max(int((h | harm).sum()), 1)),
        }

    type_distribution = {}
    for candidate_index, name in enumerate(CANDIDATE_NAMES):
        selected_type = legacy_index == candidate_index
        if selected_type.any():
            type_distribution[name] = {
                'selected_fraction': float(selected_type.mean()),
                'mean_delta_iou': float(legacy_delta[selected_type].mean()),
                'mean_contrast': _safe_mean(contrast[..., candidate_index][selected_type]),
            }

    width_bounds = ((0.0, 0.15), (0.15, 0.35), (0.35, 0.60),
                    (0.60, float('inf')))
    width_names = ('short', 'medium_short', 'medium_long', 'long')
    gt_width = gt[:, 1] - gt[:, 0]
    parent_width = original_props[..., 1] - original_props[..., 0]
    bucket_report = {
        'gt_width': {
            'baseline': _bucket_report(
                gt_width, baseline_rank['r1_iou'],
                width_names, width_bounds),
            'legacy_delta_iou': _bucket_report(
                np.repeat(gt_width, num_props), legacy_delta.reshape(-1),
                width_names, width_bounds),
        },
        'parent_width': _bucket_report(
            parent_width.reshape(-1), legacy_delta.reshape(-1),
            ('short', 'medium', 'long'),
            ((0.0, 0.35), (0.35, 0.65), (0.65, float('inf')))),
        'boundary_confidence': _bucket_report(
            boundary_confidence[..., 1:].reshape(-1),
            delta_iou[..., 1:].reshape(-1),
            ('low', 'medium', 'high'), ((0.0, 0.50), (0.50, 0.85),
                                        (0.85, float('inf')))),
    }

    bootstrap = video_cluster_bootstrap(
        original_props, legacy_props, gt, video_ids,
        baseline_scores=baseline_scores, stage_scores=legacy_scores,
        repeats=args.bootstrap_replicates, seed=args.bootstrap_seed)
    summary = {
        'schema_version': 1,
        'dataset': str(features['dataset'].reshape(-1)[0]),
        'split': str(features['split'].reshape(-1)[0]),
        'checkpoint_path': str(features['checkpoint_path'].reshape(-1)[0]),
        'checkpoint_sha256': str(features['checkpoint_sha256'].reshape(-1)[0]),
        'config_sha256': str(features['config_sha256'].reshape(-1)[0]),
        'mask_seeds': np.asarray(features['mask_seeds']).astype(int).tolist(),
        'query_count': q,
        'video_count': int(np.unique(video_ids).size),
        'diagnostic_only': bool(diagnostic_only),
        'protocol_metadata': metadata,
        'parent_count': int(q * num_props),
        'baseline_metrics': {key: value for key, value in baseline_rank.items()
                             if key.startswith('R@')},
        'legacy_metrics': {key: value for key, value in legacy_rank.items()
                           if key.startswith('R@')},
        'parent_oracle': {
            'mean_gain': float(parent_gain.mean()),
            'median_gain': float(np.median(parent_gain)),
            'gain_gt_0.01_fraction': float((parent_gain > 0.01).mean()),
            'gain_lt_neg_0.01_fraction': float((parent_gain < -0.01).mean()),
            'neutral_fraction': float((np.abs(parent_gain) <= 0.01).mean()),
            'best_candidate_distribution': {
                CANDIDATE_NAMES[index]: float((best_index == index).mean())
                for index in range(7)
            },
        },
        'candidate_set_oracle': candidate_set_metrics,
        'legacy_selector': {
            'changed_fraction': float(changed.mean()),
            'helpful_fraction_of_all': float(helpful.mean()),
            'harmful_fraction_of_all': float(harmful.mean()),
            'neutral_fraction_of_all': float(neutral.mean()),
            'trim_precision': float(helpful.sum() /
                                    max(int((helpful | harmful).sum()), 1)),
            'mean_regret': float(legacy_regret.mean()),
            'mean_delta_iou': float(legacy_delta.mean()),
        },
        'nll_discriminability': nll_report,
        'legacy_epsilon_scan': epsilon_report,
        'candidate_type': type_distribution,
        'buckets': bucket_report,
        'video_cluster_bootstrap': bootstrap,
        'STAGE_B_GO': False,
        'stage_b_gate_note': (
            'Diagnostic-only oracle; partial/test-as-validation output cannot '
            'support Stage B decisions.' if diagnostic_only else
            'Oracle/regret report only; selector validation and configuration '
            'freeze are still required.'),
    }
    rows = []
    for qi in range(q):
        for pi in range(num_props):
            bi = int(best_index[qi, pi])
            li = int(legacy_index[qi, pi])
            rows.append({
                'sample_id': sample_ids[qi],
                'video_id': video_ids[qi],
                'proposal_index': pi,
                'parent_start': original_props[qi, pi, 0],
                'parent_end': original_props[qi, pi, 1],
                'gt_start': gt[qi, 0],
                'gt_end': gt[qi, 1],
                'original_iou': original_iou[qi, pi],
                'oracle_candidate': CANDIDATE_NAMES[bi],
                'oracle_iou': best_iou[qi, pi],
                'oracle_gain': parent_gain[qi, pi],
                'legacy_candidate': CANDIDATE_NAMES[li],
                'legacy_iou': legacy_iou[qi, pi],
                'legacy_delta_iou': legacy_delta[qi, pi],
                'legacy_regret': legacy_regret[qi, pi],
                'parent_nll': candidate_nll[qi, pi, 0],
                'legacy_nll': candidate_nll[qi, pi, li],
                'candidate_contrast': contrast[qi, pi, li],
                'candidate_contrast_std': contrast_std[qi, pi, li],
                'boundary_confidence': boundary_confidence[qi, pi, li],
            })
    _write_rows(args.rows_output, rows)
    dump_json(args.summary_output, summary)
    print('saved oracle rows to {}'.format(args.rows_output))
    print('saved oracle summary to {}'.format(args.summary_output))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--rows-output', required=True)
    parser.add_argument('--summary-output', required=True)
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
    analyze(args)


if __name__ == '__main__':
    main()
