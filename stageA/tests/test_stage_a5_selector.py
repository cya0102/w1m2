import numpy as np

from runners.stage_a5 import REASON_CODES, select_stage_a5_candidates


def _inputs():
    start = np.asarray([[[0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]]], dtype=np.float32)
    end = np.asarray([[[0.9, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8]]], dtype=np.float32)
    valid = np.ones((1, 1, 7), dtype=bool)
    nll = np.asarray([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
    nll_std = np.zeros_like(nll)
    shell = np.asarray([[[np.inf, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]]])
    shell_std = np.zeros_like(shell)
    contrast = shell - nll
    contrast_std = np.zeros_like(contrast)
    confidence = np.ones_like(nll)
    confidence[..., 0] = 0
    return start, end, valid, nll, nll_std, shell, shell_std, confidence, contrast, contrast_std


def test_selector_is_conservative_and_only_opens_one_sided_candidates():
    values = _inputs()
    config = {
        'allowed_candidate_types': ['left_near', 'left_strong',
                                    'right_near', 'right_strong'],
        'min_parent_width': 0.5, 'min_retained_ratio': 0.75,
        'max_relative_shift': 0.2, 'max_nll_increase': 0.0,
        'min_contrast_margin': 0.1, 'max_contrast_std': 0.03,
        'min_boundary_percentile': 0.7, 'lambda_contrast': 1.0,
        'lambda_edit': 0.25, 'lambda_uncertainty': 0.5,
        'lambda_boundary': 0.1, 'accept_margin': 0.01,
    }
    refined, scores, index, reasons = select_stage_a5_candidates(
        *values[:8], config, contrast_mean=values[8],
        contrast_std=values[9])
    assert index.item() in (1, 2, 3, 4)
    assert reasons.item() == REASON_CODES['trim_selected']
    assert refined.shape == (1, 1, 2) and scores.item() < 0

    disabled = dict(config, allowed_candidate_types=['left_near', 'left_strong',
                                                     'right_near', 'right_strong'])
    _, _, disabled_index, disabled_reason = select_stage_a5_candidates(
        *values[:8], disabled, contrast_mean=values[8],
        contrast_std=values[9])
    assert disabled_index.item() in (1, 2, 3, 4)
    assert disabled_index.item() not in (5, 6)


def test_selector_monotonic_gates_and_edit_penalty():
    values = _inputs()
    config = {
        'allowed_candidate_types': ['left_near', 'left_strong',
                                    'right_near', 'right_strong'],
        'min_parent_width': 0.5, 'min_retained_ratio': 0.5,
        'max_relative_shift': 0.5, 'max_nll_increase': 0.0,
        'min_contrast_margin': 0.0, 'max_contrast_std': 0.03,
        'min_boundary_percentile': 0.0, 'lambda_contrast': 1.0,
        'lambda_edit': 0.1, 'lambda_uncertainty': 0.0,
        'lambda_boundary': 0.0, 'accept_margin': 0.0,
    }
    _, _, low_index, _ = select_stage_a5_candidates(
        *values[:8], config, contrast_mean=values[8],
        contrast_std=values[9])
    high = dict(config, lambda_edit=10.0)
    _, _, high_index, _ = select_stage_a5_candidates(
        *values[:8], high, contrast_mean=values[8],
        contrast_std=values[9])
    assert high_index.item() in (0, 1, 2, 3, 4)
    assert high_index.item() == 0 or high_index.item() >= low_index.item()

    strict = dict(config, min_contrast_margin=10.0)
    _, _, strict_index, _ = select_stage_a5_candidates(
        *values[:8], strict, contrast_mean=values[8],
        contrast_std=values[9])
    assert strict_index.item() == 0
