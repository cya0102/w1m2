import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import tools.scan_stage_a5_selectors as scan_module
from tools.scan_stage_a5_selectors import (
    GEOMETRY_GRID, SEMANTIC_GRID, WEIGHT_GRID, _choose_selected_row,
    _iter_weight_configs,
)
from tools.stage_a5_utils import (
    load_candidate_exports, save_candidate_exports, validate_export_protocol,
)


def _write_exports(tmp_path, *, dataset='activitynet', partial=False,
                   validation_is_test=False):
    metadata = {
        'schema_version': 1,
        'dataset': dataset,
        'split': 'val',
        'checkpoint_label': 'test',
        'checkpoint_path': '/tmp/test.pt',
        'checkpoint_sha256': 'checkpoint-sha',
        'config_sha256': 'config-sha',
        'mask_seeds': [8, 18, 28],
        'train_data': 'data/train.json',
        'val_data': 'data/test.json' if validation_is_test else 'data/val.json',
        'test_data': 'data/test.json',
        'validation_is_test': validation_is_test,
        'query_count': 1,
        'partial': partial,
    }
    candidate_start = np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                                 dtype=np.float32)
    candidate_end = np.asarray([[[1.0, 0.9, 0.8, 0.9, 0.8, 0.8, 0.7]]],
                               dtype=np.float32)
    valid = np.ones((1, 1, 7), dtype=bool)
    nll = np.ones((1, 1, 7), dtype=np.float32)
    inf_original = np.asarray([[[np.inf, 1, 1, 1, 1, 1, 1]]],
                              dtype=np.float32)
    features = {
        'schema_version': np.asarray([1], dtype=np.int32),
        'dataset': np.asarray([dataset], dtype=str),
        'split': np.asarray(['val'], dtype=str),
        'checkpoint_path': np.asarray(['/tmp/test.pt'], dtype=str),
        'checkpoint_sha256': np.asarray(['checkpoint-sha'], dtype=str),
        'config_sha256': np.asarray(['config-sha'], dtype=str),
        'mask_seeds': np.asarray([8, 18, 28], dtype=np.int64),
        'sample_ids': np.asarray(['video:0'], dtype=str),
        'video_ids': np.asarray(['video'], dtype=str),
        'durations': np.asarray([1.0], dtype=np.float32),
        'parent_start': np.asarray([[0.0]], dtype=np.float32),
        'parent_end': np.asarray([[1.0]], dtype=np.float32),
        'parent_event_score': np.asarray([[0.0]], dtype=np.float32),
        'candidate_start': candidate_start,
        'candidate_end': candidate_end,
        'candidate_valid': valid,
        'candidate_type': np.arange(7, dtype=np.int8),
        'candidate_nll_mean': nll,
        'candidate_nll_std': np.zeros_like(nll),
        'candidate_left_boundary_score': np.zeros_like(nll),
        'candidate_right_boundary_score': np.zeros_like(nll),
        'candidate_boundary_confidence': np.zeros_like(nll),
        'candidate_shell_nll_mean': inf_original,
        'candidate_shell_nll_std': np.zeros_like(nll),
        'candidate_contrast_mean': inf_original,
        'candidate_contrast_std': np.zeros_like(nll),
        'legacy_selected_index': np.zeros((1, 1), dtype=np.int8),
        'metadata_json': np.asarray([json.dumps(metadata, sort_keys=True)],
                                    dtype=str),
    }
    labels = {
        'schema_version': np.asarray([1], dtype=np.int32),
        'dataset': np.asarray([dataset], dtype=str),
        'split': np.asarray(['val'], dtype=str),
        'sample_ids': np.asarray(['video:0'], dtype=str),
        'video_ids': np.asarray(['video'], dtype=str),
        'gt_normalized': np.asarray([[0.1, 0.9]], dtype=np.float32),
        'metadata_json': np.asarray([json.dumps(metadata, sort_keys=True)],
                                    dtype=str),
    }
    features_path = tmp_path / 'features.npz'
    labels_path = tmp_path / 'labels.npz'
    save_candidate_exports(features_path, labels_path, features, labels)
    return load_candidate_exports(features_path, labels_path)


def test_weight_grid_is_complete_for_each_semantic_parent():
    parents = [
        {'config': {'semantic_parent': 1}},
        {'config': {'semantic_parent': 2}},
    ]
    rows = list(_iter_weight_configs(parents))
    assert len(rows) == 2 * len(WEIGHT_GRID)
    counts = Counter(row['semantic_parent'] for row in rows)
    assert counts == {1: len(WEIGHT_GRID), 2: len(WEIGHT_GRID)}


def test_default_scan_row_count_matches_theoretical_grid(monkeypatch, tmp_path):
    metadata = {
        'dataset': 'activitynet', 'split': 'val', 'partial': False,
        'query_count': 1, 'validation_is_test': False,
    }
    arrays = {
        'metadata_json': np.asarray([json.dumps(metadata)], dtype=str),
        'dataset': np.asarray(['activitynet'], dtype=str),
        'split': np.asarray(['val'], dtype=str),
        'sample_ids': np.asarray(['video:0'], dtype=str),
    }
    monkeypatch.setattr(scan_module, 'load_candidate_exports',
                        lambda features, labels: (arrays, arrays))

    def fake_evaluate(features, labels, config, event_weight=0.0):
        return ({
            'objective': 0.0, 'changed_fraction': 0.0,
            'helpful_fraction': 0.0, 'harmful_fraction': 0.0,
            'trim_precision': 1.0, 'long_R5_penalty': 0.0,
            'bucket_drops': {}, 'baseline_R1_mIoU': 0.0,
            'stage_R1_mIoU': 0.0, 'baseline_R5_mIoU': 0.0,
            'stage_R5_mIoU': 0.0, 'baseline_R1_IoU@0.5': 0.0,
            'stage_R1_IoU@0.5': 0.0, 'stage_b_go': False,
        }, None, None, None)

    monkeypatch.setattr(scan_module, '_evaluate', fake_evaluate)
    args = SimpleNamespace(
        features='features.npz', labels='labels.npz',
        output=tmp_path / 'scan.csv',
        selected_config_output=tmp_path / 'selected.json', event_weight=0.0,
        top_geometry=10, top_semantic=5,
        allow_partial_smoke=False, allow_test_diagnostic=False,
    )
    selected = scan_module.scan(args)
    expected = (len(GEOMETRY_GRID) +
                10 * len(SEMANTIC_GRID) + 5 * len(WEIGHT_GRID))
    assert selected['num_scanned_configs'] == expected
    assert sum(1 for _ in open(args.output, encoding='utf8')) == expected + 1


def _scan_row(objective, stage_b_go, name):
    return {
        'objective': objective, 'changed_fraction': 0.0,
        'helpful_fraction': 0.0, 'harmful_fraction': 0.0,
        'trim_precision': 1.0, 'long_R5_penalty': 0.0,
        'bucket_drops': {}, 'baseline_R1_mIoU': 0.0,
        'stage_R1_mIoU': 0.0, 'baseline_R5_mIoU': 0.0,
        'stage_R5_mIoU': 0.0, 'baseline_R1_IoU@0.5': 0.0,
        'stage_R1_IoU@0.5': 0.0, 'stage_b_go': stage_b_go,
        'config': {'name': name},
    }


def test_gate_passing_config_wins_over_higher_objective_no_change():
    rows = [_scan_row(100.0, False, 'no-change'),
            _scan_row(1.0, True, 'gate-pass')]
    best, count, reason, stage_b_go = _choose_selected_row(rows)
    assert best['config']['name'] == 'gate-pass'
    assert count == 1
    assert reason == 'best_gate_passing_config_by_objective'
    assert stage_b_go is True


def test_no_gate_passing_config_keeps_best_objective_but_forces_no_go():
    rows = [_scan_row(100.0, False, 'best'),
            _scan_row(1.0, False, 'other')]
    best, count, reason, stage_b_go = _choose_selected_row(rows)
    assert best['config']['name'] == 'best'
    assert count == 0
    assert reason == 'no_gate_passing_config_best_objective_with_gate_false'
    assert stage_b_go is False


def test_diagnostic_selection_forces_no_go_even_if_point_gate_passes():
    best, count, reason, stage_b_go = _choose_selected_row(
        [_scan_row(1.0, True, 'gate-pass')], diagnostic_only=True)
    assert best['config']['name'] == 'gate-pass'
    assert count == 1
    assert reason == 'diagnostic_only_best_objective_no_formal_gate'
    assert stage_b_go is False


def test_partial_exports_are_rejected_unless_explicitly_diagnostic(tmp_path):
    exports = _write_exports(tmp_path, partial=True)
    with pytest.raises(ValueError, match='partial=true'):
        validate_export_protocol(*exports)
    _, diagnostic_only = validate_export_protocol(
        *exports, allow_partial_smoke=True)
    assert diagnostic_only is True


def test_charades_test_as_validation_is_rejected_unless_diagnostic(tmp_path):
    exports = _write_exports(tmp_path, dataset='charades',
                             validation_is_test=True)
    with pytest.raises(ValueError, match='reuses test_data'):
        validate_export_protocol(*exports, operation='selector scan')
    _, diagnostic_only = validate_export_protocol(
        *exports, allow_test_diagnostic=True, operation='selector scan')
    assert diagnostic_only is True


def test_charades_diagnostic_scan_uses_fixed_rule_and_no_go(monkeypatch, tmp_path):
    exports = _write_exports(tmp_path, dataset='charades',
                             validation_is_test=True)
    monkeypatch.setattr(scan_module, 'load_candidate_exports',
                        lambda features, labels: exports)
    monkeypatch.setattr(scan_module, '_evaluate',
                        lambda features, labels, config, event_weight=0.0: (
                            _scan_row(1.0, True, 'fixed'), None, None, None))
    args = SimpleNamespace(
        features='features.npz', labels='labels.npz',
        output=tmp_path / 'scan.csv',
        selected_config_output=tmp_path / 'selected.json', event_weight=0.0,
        top_geometry=10, top_semantic=5,
        allow_partial_smoke=False, allow_test_diagnostic=True,
    )
    selected = scan_module.scan(args)
    assert selected['num_scanned_configs'] == 1
    assert selected['STAGE_B_GO'] is False
    assert selected['diagnostic_only'] is True
    assert 'diagnostic_summary' in selected
    assert 'validation_summary' not in selected
    assert 'fixed-rule diagnostic' in selected['selection_protocol']
    assert 'diagnostic_fixed_rule' in args.output.read_text(encoding='utf8')


def test_gt_derived_feature_key_is_rejected(tmp_path):
    features, labels = _write_exports(tmp_path)
    bad_features = dict(features)
    bad_features['gt_width'] = np.asarray([[0.8]], dtype=np.float32)
    with pytest.raises(ValueError, match='GT-derived'):
        save_candidate_exports(tmp_path / 'bad_features.npz',
                               tmp_path / 'bad_labels.npz',
                               bad_features, labels)
