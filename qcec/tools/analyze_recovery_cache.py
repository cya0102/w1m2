"""Analyze a fixed recovery cache without another model forward pass.

This is an offline P2 diagnostic.  It compares the trained generator's outer
proposal envelope with importance-weighted and 5--95% mixture-mass intervals;
the latter two are diagnostics only and are not silently installed as a new
prediction rule.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def interval_iou(intervals, ground_truth):
    left = np.maximum(intervals[..., 0], ground_truth[:, None, 0])
    right = np.minimum(intervals[..., 1], ground_truth[:, None, 1])
    intersection = np.maximum(right - left, 0.0)
    union = np.maximum(
        np.maximum(intervals[..., 1], ground_truth[:, None, 1])
        - np.minimum(intervals[..., 0], ground_truth[:, None, 0]),
        1e-8)
    return intersection / union


def quality_intervals(component_weights, component_importance,
                      component_valid_mask):
    sample_count, num_props, _, steps = component_weights.shape
    positions = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    density = component_weights * component_importance[..., None]
    density = density * component_valid_mask[..., None].astype(np.float32)
    density = density.sum(axis=2)
    density = density / np.maximum(density.sum(axis=-1, keepdims=True), 1e-8)
    cdf = np.cumsum(density, axis=-1)
    left_index = (cdf < 0.05).sum(axis=-1).clip(max=steps - 1)
    right_index = (cdf < 0.95).sum(axis=-1).clip(max=steps - 1)
    left = positions[left_index]
    right = positions[right_index]
    return np.stack([left, np.maximum(right, left)], axis=-1)


def summarize_rule(intervals, ground_truth):
    iou = interval_iou(intervals, ground_truth)
    return {
        'mean_width': float(np.mean(intervals[..., 1] - intervals[..., 0])),
        'mIoU_best_candidate': float(np.mean(iou)),
        'R5_IoU@0.3': float(np.mean(np.max(iou[:, :5], axis=1) >= 0.3)),
        'R5_IoU@0.5': float(np.mean(np.max(iou[:, :5], axis=1) >= 0.5)),
    }


def boundary_summary(cache):
    required = {'qcec_cluster_bounds', 'qcec_cluster_mask'}
    if not required.issubset(cache):
        return {}
    bounds = cache['qcec_cluster_bounds']
    mask = cache['qcec_cluster_mask'].astype(bool)
    start = cache['ground_truth'][:, 0]
    end = cache['ground_truth'][:, 1]
    boundaries = np.concatenate([bounds[..., 0], bounds[..., 1]], axis=-1)
    valid = np.concatenate([mask, mask], axis=-1)
    start_distance = np.min(
        np.where(valid, np.abs(boundaries - start[:, None]), np.inf), axis=-1)
    end_distance = np.min(
        np.where(valid, np.abs(boundaries - end[:, None]), np.inf), axis=-1)
    result = {
        'mean_valid_boundaries': float(valid.sum(axis=-1).mean()),
        'gt_start_distance_p50': float(np.median(start_distance)),
        'gt_end_distance_p50': float(np.median(end_distance)),
    }
    for tolerance in (1 / 200, 2 / 200, 5 / 200):
        suffix = 'tol_{:.3f}'.format(tolerance)
        result['start_hit_' + suffix] = float(
            np.mean(start_distance <= tolerance))
        result['end_hit_' + suffix] = float(
            np.mean(end_distance <= tolerance))
    return result


def relevance_summary(cache):
    required = {'qcec_cluster_relevance', 'qcec_cluster_bounds',
                'qcec_cluster_mask'}
    if not required.issubset(cache):
        return {}
    relevance = cache['qcec_cluster_relevance']
    bounds = cache['qcec_cluster_bounds']
    mask = cache['qcec_cluster_mask'].astype(bool)
    gt = cache['ground_truth']
    overlap = (
        np.minimum(bounds[..., 1], gt[:, None, 1])
        - np.maximum(bounds[..., 0], gt[:, None, 0])).clip(min=0)
    overlap = overlap / np.maximum(
        np.maximum(bounds[..., 1], gt[:, None, 1])
        - np.minimum(bounds[..., 0], gt[:, None, 0]), 1e-8)
    valid_relevance = relevance[mask]
    valid_overlap = overlap[mask]
    if valid_relevance.size < 2 or np.std(valid_relevance) < 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(valid_relevance, valid_overlap)[0, 1])
    return {
        'mean_relevance': float(valid_relevance.mean())
        if valid_relevance.size else 0.0,
        'relevance_overlap_pearson': correlation,
        'relevance_p90': float(np.percentile(valid_relevance, 90))
        if valid_relevance.size else 0.0,
    }


def analyze(path):
    with np.load(path, allow_pickle=False) as loaded:
        cache = {name: loaded[name] for name in loaded.files
                 if name != 'metadata_json'}
        metadata = json.loads(str(loaded['metadata_json'].item()))
    ground_truth = cache['ground_truth']
    result = {'metadata': metadata, 'rules': {}}

    raw_center = cache['raw_center']
    raw_width = cache['raw_width']
    outer = np.stack([
        np.clip(raw_center - raw_width / 2, 0, 1),
        np.clip(raw_center + raw_width / 2, 0, 1),
    ], axis=-1)
    result['rules']['raw_generator'] = summarize_rule(outer, ground_truth)

    if {'mixture_component_centers', 'mixture_component_widths',
            'mixture_component_importance', 'mixture_component_valid_mask',
            'mixture_component_weights'}.issubset(cache):
        centers = cache['mixture_component_centers']
        widths = cache['mixture_component_widths']
        importance = cache['mixture_component_importance']
        valid = cache['mixture_component_valid_mask'].astype(bool)
        left = np.clip(centers - widths / 2, 0, 1)
        right = np.clip(centers + widths / 2, 0, 1)
        left = np.where(valid, left, 1.0)
        right = np.where(valid, right, 0.0)
        outer_components = np.stack([
            left.min(axis=-1), right.max(axis=-1),
        ], axis=-1)
        weighted_components = np.stack([
            (left * importance).sum(axis=-1),
            (right * importance).sum(axis=-1),
        ], axis=-1)
        weighted_components[..., 0] = np.clip(
            weighted_components[..., 0], 0, 1)
        weighted_components[..., 1] = np.clip(
            weighted_components[..., 1], 0, 1)
        result['rules']['component_outer'] = summarize_rule(
            outer_components, ground_truth)
        result['rules']['importance_weighted'] = summarize_rule(
            weighted_components, ground_truth)
        result['rules']['quality_mass_05_95'] = summarize_rule(
            quality_intervals(
                cache['mixture_component_weights'], importance, valid),
            ground_truth)
        outer_left_index = left.argmin(axis=-1)
        outer_right_index = right.argmax(axis=-1)
        left_importance = np.take_along_axis(
            importance, outer_left_index[..., None], axis=-1).squeeze(-1)
        right_importance = np.take_along_axis(
            importance, outer_right_index[..., None], axis=-1).squeeze(-1)
        result['outer_low_importance_fraction'] = float(np.mean(
            np.minimum(left_importance, right_importance) < 0.2))

    result['boundary_diagnostics'] = boundary_summary(cache)
    result['relevance_diagnostics'] = relevance_summary(cache)
    nll = cache['nll_score']
    raw_iou = interval_iou(outer, ground_truth)
    best_iou = raw_iou.max(axis=1)
    if nll.size > 1 and np.std(nll.reshape(-1)) > 1e-8:
        result['nll_best_iou_pearson'] = float(np.corrcoef(
            -nll.reshape(-1), raw_iou.reshape(-1))[0, 1])
    else:
        result['nll_best_iou_pearson'] = 0.0
    result['oracle_R5_IoU@0.5'] = float(np.mean(best_iou >= 0.5))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = analyze(args.cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print('saved cache analysis to {}'.format(output))


if __name__ == '__main__':
    main()
