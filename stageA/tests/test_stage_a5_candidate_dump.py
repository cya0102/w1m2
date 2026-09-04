import numpy as np

from tools.dump_stage_a5_candidates import _aggregate, _output_paths


def test_candidate_aggregation_keeps_invalid_nll_infinite():
    values = np.asarray([[[[1.0, 3.0]]], [[[1.2, 4.0]]]])
    valid = np.asarray([[[True, False]]])
    mean, std = _aggregate(values, valid)
    assert np.isfinite(mean[0, 0, 0])
    assert np.isinf(mean[0, 0, 1])
    assert np.isfinite(std[0, 0, 0])


def test_output_is_a_prefix_with_separate_files():
    features, labels = _output_paths('/tmp/example_candidates')
    assert str(features).endswith('_features.npz')
    assert str(labels).endswith('_labels.npz')
