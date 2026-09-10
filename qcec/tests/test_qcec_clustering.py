import numpy as np

from datasets.base import sample_frame_features
from tools.build_qcec_cluster_index import contiguous_ward_cluster


def test_contiguous_ward_returns_continuous_deterministic_segments():
    features = np.vstack([
        np.tile([1.0, 0.0], (4, 1)),
        np.tile([0.0, 1.0], (4, 1)),
        np.tile([-1.0, 0.0], (4, 1)),
    ])
    first = contiguous_ward_cluster(features, 3)
    second = contiguous_ward_cluster(features, 3)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    ids, bounds, valid = first
    assert np.array_equal(ids, np.repeat(np.arange(3), 4))
    assert np.array_equal(valid, np.ones(3, dtype=bool))
    assert np.allclose(bounds, [[0, 4 / 12], [4 / 12, 8 / 12], [8 / 12, 1]])
    for cluster_id in range(3):
        positions = np.flatnonzero(ids == cluster_id)
        assert np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))


def test_contiguous_ward_pads_when_requested_clusters_exceed_frames():
    ids, bounds, valid = contiguous_ward_cluster(
        np.eye(3, dtype=np.float32), num_clusters=5)
    assert ids.tolist() == [0, 1, 2]
    assert valid.tolist() == [True, True, True, False, False]
    assert np.allclose(bounds[3:], 0)


def test_sampling_helper_is_fixed_length_and_mean_pooled():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    sampled = sample_frame_features(features, 3)
    assert sampled.shape == (3, 2)
    # The final endpoint follows the historical baseline sampler, which clips
    # ``len(features)`` to the last valid row before slicing.
    assert np.allclose(sampled, [[1, 2], [5, 6], [8, 9]])
