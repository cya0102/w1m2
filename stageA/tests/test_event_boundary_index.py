import json

import numpy as np
import pytest

from datasets.event_boundaries import (
    EventBoundaryIndex,
    build_boundary_index,
    merge_equal_score_plateaus,
    temporal_nms,
)


def test_plateau_merge_and_score_peak_nms():
    indices, scores = merge_equal_score_plateaus(
        [10, 11, 12, 20], [0.4, 0.4 + 1e-8, 0.4, 0.9])
    assert indices.tolist() == [11, 20]
    assert np.allclose(scores, [0.4, 0.9])

    kept_indices, kept_scores = temporal_nms(
        [10, 12, 15], [0.5, 0.9, 0.8], min_gap_clips=3)
    assert kept_indices.tolist() == [12, 15]
    assert np.allclose(kept_scores, [0.9, 0.8])


def test_build_index_has_valid_csr_and_video_lookup(tmp_path):
    source = {
        'metadata': {'dataset': 'activitynet'},
        'videos': {
            'video-a': {
                'num_clips': 200,
                'boundary_indices': [0, 10, 11, 12, 30, 199],
                'detected_boundary_indices': [10, 11, 12, 30],
                'boundary_scores': [2.0, 0.4, 0.4, 0.8, 0.5, 1.0],
            },
            'video-empty': {
                'num_clips': 200,
                'boundary_indices': [0, 199],
                'detected_boundary_indices': [],
                'boundary_scores': [1.0, 1.0],
            },
        },
    }
    source_path = tmp_path / 'source.json'
    source_path.write_text(json.dumps(source), encoding='utf8')
    index_path = tmp_path / 'boundaries.npz'
    build_boundary_index(source_path, index_path, min_gap_clips=2)

    index = EventBoundaryIndex(index_path, expected_dataset='activitynet')
    positions, scores = index.get('video-a')
    assert np.allclose(positions, [12 / 199, 30 / 199])
    assert scores.shape == (2,)
    assert index.get('video-empty')[0].size == 0
    with pytest.raises(KeyError):
        index.get('missing-video')


def test_index_loader_rejects_dataset_mismatch(tmp_path):
    path = tmp_path / 'index.npz'
    np.savez(
        path,
        video_ids=np.asarray(['v']),
        offsets=np.asarray([0, 0], dtype=np.int64),
        indices=np.empty(0, dtype=np.int16),
        positions=np.empty(0, dtype=np.float32),
        scores=np.empty(0, dtype=np.float32),
        metadata=np.asarray([json.dumps({
            'schema_version': 1,
            'dataset': 'charades',
            'num_clips': 200,
            'position_mapping': 'index / (num_clips - 1)',
            'endpoint_policy': 'internal detected boundaries only',
        })]),
    )
    with pytest.raises(ValueError, match='does not match'):
        EventBoundaryIndex(path, expected_dataset='activitynet')
