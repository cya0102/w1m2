"""Pure Stage-A.5 selector and validation helpers.

The selector in this module is deliberately independent of the dataset and
ground truth.  It can therefore be used by the online runner and by the
offline selector scan without creating a path for GT-derived features to leak
into inference.
"""

import numpy as np


CANDIDATE_NAMES = (
    "original", "left_near", "left_strong", "right_near", "right_strong",
    "both_near", "both_strong",
)

REASON_CODES = {
    "original_default": 0,
    "invalid_geometry": 1,
    "parent_not_long": 2,
    "retained_ratio_too_small": 3,
    "shift_too_large": 4,
    "insufficient_by_nll": 5,
    "weak_shell_contrast": 6,
    "high_mask_uncertainty": 7,
    "weak_boundary": 8,
    "candidate_type_disabled": 9,
    "score_margin_not_met": 10,
    "trim_selected": 11,
}


def _selector_config(config):
    if config is None:
        return {}
    config = dict(config)
    stage_a5 = config.get("stage_a5")
    if isinstance(stage_a5, dict):
        config = dict(stage_a5)
    nested = config.get("selector")
    return dict(nested) if isinstance(nested, dict) else config


def _array(value, name, ndim=3):
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError("{} must have {} dimensions".format(name, ndim))
    return result


def select_stage_a5_candidates(
        candidate_start, candidate_end, candidate_valid,
        candidate_nll_mean, candidate_nll_std,
        shell_nll_mean, shell_nll_std, boundary_confidence, config,
        contrast_mean=None, contrast_std=None):
    """Select conservative inward candidates without any GT input.

    Returns ``(refined_props, selected_score, selected_index, reason_code)``.
    The original proposal (index zero) is always the fallback.  The optional
    contrast arrays are the multi-mask summary produced by the exporter; when
    omitted, the same quantities are reconstructed from the supplied shell and
    trim summaries.
    """
    starts = _array(candidate_start, "candidate_start")
    ends = _array(candidate_end, "candidate_end")
    valid = _array(candidate_valid, "candidate_valid").astype(bool)
    nll = _array(candidate_nll_mean, "candidate_nll_mean").astype(np.float64)
    nll_std = _array(candidate_nll_std, "candidate_nll_std").astype(np.float64)
    shell = _array(shell_nll_mean, "shell_nll_mean").astype(np.float64)
    shell_std = _array(shell_nll_std, "shell_nll_std").astype(np.float64)
    confidence = _array(
        boundary_confidence, "boundary_confidence").astype(np.float64)
    arrays = (ends, valid, nll, nll_std, shell, shell_std, confidence)
    if any(item.shape != starts.shape for item in arrays):
        raise ValueError("all candidate arrays must have the same shape")
    if starts.shape[-1] != 7:
        raise ValueError("Stage A.5 expects exactly seven candidates")
    if not np.isfinite(starts[..., 0]).all() or not np.isfinite(
            ends[..., 0]).all():
        raise ValueError("original proposal geometry must be finite")

    if contrast_mean is None:
        contrast = shell - nll
    else:
        contrast = _array(contrast_mean, "contrast_mean").astype(np.float64)
        if contrast.shape != starts.shape:
            raise ValueError("contrast_mean shape does not match candidates")
    if contrast_std is None:
        uncertainty = np.sqrt(np.maximum(nll_std, 0.0) ** 2 +
                              np.maximum(shell_std, 0.0) ** 2)
    else:
        uncertainty = _array(
            contrast_std, "contrast_std").astype(np.float64)
        if uncertainty.shape != starts.shape:
            raise ValueError("contrast_std shape does not match candidates")

    cfg = _selector_config(config)
    allowed = cfg.get("allowed_candidate_types", [
        "left_near", "left_strong", "right_near", "right_strong",
    ])
    allowed = {str(name) for name in allowed}
    min_parent_width = float(cfg.get("min_parent_width", 0.50))
    min_retained_ratio = float(cfg.get("min_retained_ratio", 0.75))
    max_relative_shift = float(cfg.get("max_relative_shift", 0.20))
    max_nll_increase = float(cfg.get("max_nll_increase", 0.00))
    min_contrast_margin = float(cfg.get("min_contrast_margin", 0.02))
    max_contrast_std = float(cfg.get("max_contrast_std", 0.03))
    min_boundary_percentile = float(
        cfg.get("min_boundary_percentile", 0.70))
    lambda_contrast = float(cfg.get("lambda_contrast", 1.0))
    lambda_edit = float(cfg.get("lambda_edit", 0.25))
    lambda_uncertainty = float(cfg.get("lambda_uncertainty", 0.5))
    lambda_boundary = float(cfg.get("lambda_boundary", 0.1))
    accept_margin = float(cfg.get("accept_margin", 0.01))
    scalars = (
        min_parent_width, min_retained_ratio, max_relative_shift,
        max_nll_increase, min_contrast_margin, max_contrast_std,
        min_boundary_percentile, lambda_contrast, lambda_edit,
        lambda_uncertainty, lambda_boundary, accept_margin,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("selector thresholds and weights must be finite")
    if min_parent_width < 0 or min_retained_ratio <= 0 or \
            max_relative_shift < 0 or max_nll_increase < 0 or \
            min_contrast_margin < 0 or max_contrast_std < 0 or \
            not 0 <= min_boundary_percentile <= 1 or accept_margin < 0:
        raise ValueError("invalid Stage-A.5 selector thresholds")

    batch_size, num_props, _ = starts.shape
    widths = ends - starts
    parent_width = widths[..., 0]
    refined = np.stack([starts[..., 0], ends[..., 0]], axis=-1).copy()
    selected_score = np.zeros((batch_size, num_props), dtype=np.float32)
    selected_index = np.zeros((batch_size, num_props), dtype=np.int8)
    reasons = np.full(
        (batch_size, num_props), REASON_CODES["original_default"],
        dtype=np.int8)

    for batch_index in range(batch_size):
        for proposal_index in range(num_props):
            base_width = parent_width[batch_index, proposal_index]
            best_score = 0.0
            best_index = 0
            first_rejection = None
            for candidate_index in range(1, 7):
                name = CANDIDATE_NAMES[candidate_index]
                code = None
                if name not in allowed:
                    code = REASON_CODES["candidate_type_disabled"]
                elif not valid[batch_index, proposal_index, candidate_index]:
                    code = REASON_CODES["invalid_geometry"]
                elif (not np.isfinite(widths[batch_index, proposal_index,
                                             candidate_index]) or
                      widths[batch_index, proposal_index, candidate_index] <= 0):
                    code = REASON_CODES["invalid_geometry"]
                elif (not np.isfinite(base_width) or
                      base_width < min_parent_width):
                    code = REASON_CODES["parent_not_long"]
                else:
                    candidate_width = widths[
                        batch_index, proposal_index, candidate_index]
                    retained_ratio = candidate_width / max(base_width, 1e-12)
                    relative_shift = max(
                        abs(starts[batch_index, proposal_index, candidate_index] -
                            starts[batch_index, proposal_index, 0]),
                        abs(ends[batch_index, proposal_index, candidate_index] -
                            ends[batch_index, proposal_index, 0])) / max(
                                base_width, 1e-12)
                    if retained_ratio < min_retained_ratio:
                        code = REASON_CODES["retained_ratio_too_small"]
                    elif relative_shift > max_relative_shift:
                        code = REASON_CODES["shift_too_large"]
                    else:
                        candidate_suff = nll[
                            batch_index, proposal_index, candidate_index] - nll[
                                batch_index, proposal_index, 0]
                        candidate_contrast = contrast[
                            batch_index, proposal_index, candidate_index]
                        candidate_uncertainty = uncertainty[
                            batch_index, proposal_index, candidate_index]
                        candidate_boundary = confidence[
                            batch_index, proposal_index, candidate_index]
                        if (not np.isfinite(candidate_suff) or
                                candidate_suff > max_nll_increase):
                            code = REASON_CODES["insufficient_by_nll"]
                        elif (not np.isfinite(candidate_contrast) or
                              candidate_contrast < min_contrast_margin):
                            code = REASON_CODES["weak_shell_contrast"]
                        elif (not np.isfinite(candidate_uncertainty) or
                              candidate_uncertainty > max_contrast_std):
                            code = REASON_CODES["high_mask_uncertainty"]
                        elif (not np.isfinite(candidate_boundary) or
                              candidate_boundary < min_boundary_percentile):
                            code = REASON_CODES["weak_boundary"]
                        else:
                            edit_ratio = 1.0 - retained_ratio
                            score = (candidate_suff - lambda_contrast *
                                     candidate_contrast + lambda_edit * edit_ratio +
                                     lambda_uncertainty * candidate_uncertainty -
                                     lambda_boundary * candidate_boundary)
                            if not np.isfinite(score) or score >= -accept_margin:
                                code = REASON_CODES["score_margin_not_met"]
                            elif (score < best_score - 1e-12 or
                                  (abs(score - best_score) <= 1e-12 and
                                   candidate_index < best_index)):
                                best_score = score
                                best_index = candidate_index
                if code is not None and first_rejection is None:
                    first_rejection = code
            if best_index != 0:
                refined[batch_index, proposal_index] = [
                    starts[batch_index, proposal_index, best_index],
                    ends[batch_index, proposal_index, best_index],
                ]
                selected_score[batch_index, proposal_index] = best_score
                selected_index[batch_index, proposal_index] = best_index
                reasons[batch_index, proposal_index] = REASON_CODES[
                    "trim_selected"]
            elif first_rejection is not None:
                reasons[batch_index, proposal_index] = first_rejection
            # Use the first rejection in candidate-priority order for a
            # reproducible fallback diagnostic.
            if best_index == 0 and first_rejection is not None:
                reasons[batch_index, proposal_index] = first_rejection
    return refined, selected_score, selected_index, reasons


def selector_reason_names():
    return {value: key for key, value in REASON_CODES.items()}


__all__ = [
    "CANDIDATE_NAMES", "REASON_CODES", "select_stage_a5_candidates",
    "selector_reason_names",
]
