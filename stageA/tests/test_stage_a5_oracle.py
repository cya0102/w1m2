import numpy as np

from runners.stage_a5 import CANDIDATE_NAMES
from tools.stage_a5_utils import candidate_iou, ranking_metrics, roc_auc, spearman_correlation


def _synthetic_exports():
    start = np.asarray([[[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    end = np.asarray([[[0.5, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    valid = np.asarray([[[True, True, False, False, False, False, False]]])
    return {
        'candidate_start': start,
        'candidate_end': end,
        'candidate_valid': valid,
    }, {'gt_normalized': np.asarray([[0.1, 0.9]], dtype=np.float32)}


def test_candidate_oracle_never_loses_original_and_finds_known_trim():
    features, labels = _synthetic_exports()
    iou = candidate_iou(features, labels)
    assert iou[0, 0, 1] > iou[0, 0, 0]
    assert np.argmax(iou, axis=-1).item() == 1
    assert np.max(iou[..., 1:], axis=-1).item() >= iou[..., 0].item()


def test_nll_diagnostics_have_expected_direction():
    assert spearman_correlation([0, 1, 2], [0, 2, 4]) > 0.99
    assert roc_auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
