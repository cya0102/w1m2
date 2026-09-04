"""NPZ schema, metrics, and paired bootstrap helpers for Stage A.5."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
FEATURE_KEYS = {
    "schema_version", "dataset", "split", "checkpoint_path",
    "checkpoint_sha256", "config_sha256", "mask_seeds", "sample_ids",
    "video_ids", "durations", "parent_start", "parent_end",
    "parent_event_score", "candidate_start", "candidate_end",
    "candidate_valid", "candidate_type", "candidate_nll_mean",
    "candidate_nll_std", "candidate_left_boundary_score",
    "candidate_right_boundary_score", "candidate_boundary_confidence",
    "candidate_shell_nll_mean", "candidate_shell_nll_std",
    "candidate_contrast_mean", "candidate_contrast_std",
    "legacy_selected_index", "metadata_json",
}
LABEL_KEYS = {"schema_version", "dataset", "split", "sample_ids",
              "video_ids", "gt_normalized", "metadata_json"}


def _metadata_bool(value, name):
    if isinstance(value, bool):
        return value
    raise ValueError("metadata {} must be a boolean".format(name))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_scalar_string(value, name):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("{} must contain one scalar".format(name))
    return str(value.reshape(-1)[0])


def _validate_common(features, labels):
    if "gt_normalized" in features.files or any(
            "gt" in key.lower() for key in features.files):
        raise ValueError("GT-derived arrays must be kept in the labels file")
    missing = FEATURE_KEYS.difference(features.files)
    if missing:
        raise ValueError("features file is missing {}".format(sorted(missing)))
    missing = LABEL_KEYS.difference(labels.files)
    if missing:
        raise ValueError("labels file is missing {}".format(sorted(missing)))
    if int(np.asarray(features["schema_version"]).reshape(-1)[0]) != SCHEMA_VERSION:
        raise ValueError("unsupported Stage-A.5 feature schema")
    if int(np.asarray(labels["schema_version"]).reshape(-1)[0]) != SCHEMA_VERSION:
        raise ValueError("unsupported Stage-A.5 label schema")
    for name in ("dataset", "split"):
        if as_scalar_string(features[name], name) != as_scalar_string(labels[name], name):
            raise ValueError("features and labels disagree on {}".format(name))
    feature_ids = np.asarray(features["sample_ids"]).astype(str)
    label_ids = np.asarray(labels["sample_ids"]).astype(str)
    if feature_ids.ndim != 1 or label_ids.ndim != 1 or \
            not np.array_equal(feature_ids, label_ids):
        raise ValueError("features and labels sample ids are misaligned")
    if np.unique(feature_ids).size != feature_ids.size:
        raise ValueError("sample ids must be unique within a split")
    if np.asarray(features["video_ids"]).shape != feature_ids.shape or \
            np.asarray(labels["video_ids"]).shape != feature_ids.shape:
        raise ValueError("video ids must have one item per sample")
    gt = np.asarray(labels["gt_normalized"], dtype=np.float32)
    if gt.shape != (feature_ids.size, 2) or not np.isfinite(gt).all():
        raise ValueError("labels gt_normalized must have shape [Q, 2]")
    if np.any(gt[:, 0] > gt[:, 1]):
        raise ValueError("ground-truth intervals must be ordered")
    q = feature_ids.size
    candidate_start = np.asarray(features["candidate_start"])
    candidate_end = np.asarray(features["candidate_end"])
    candidate_valid = np.asarray(features["candidate_valid"])
    if candidate_start.ndim != 3 or candidate_start.shape != candidate_end.shape or \
            candidate_start.shape[0] != q or candidate_start.shape[-1] != 7 or \
            candidate_valid.shape != candidate_start.shape:
        raise ValueError("candidate arrays must have shape [Q, N, 7]")
    if not candidate_valid[:, :, 0].all():
        raise ValueError("candidate zero must be valid for every parent")
    for name in ("parent_start", "parent_end", "parent_event_score"):
        value = np.asarray(features[name])
        if value.shape != candidate_start.shape[:2]:
            raise ValueError("{} must have shape [Q, N]".format(name))
    durations = np.asarray(features["durations"], dtype=np.float64)
    if durations.shape != (q,) or not np.isfinite(durations).all() or \
            np.any(durations <= 0):
        raise ValueError("durations must be positive and finite")


def load_candidate_exports(features_path, labels_path):
    with np.load(features_path, allow_pickle=False) as features, \
            np.load(labels_path, allow_pickle=False) as labels:
        _validate_common(features, labels)
        return ({key: np.asarray(features[key]) for key in features.files},
                {key: np.asarray(labels[key]) for key in labels.files})


def export_metadata(features, labels):
    """Read and validate the protocol metadata shared by both NPZ files."""
    try:
        feature_metadata = json.loads(
            as_scalar_string(features["metadata_json"], "metadata_json"))
        label_metadata = json.loads(
            as_scalar_string(labels["metadata_json"], "metadata_json"))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Stage-A.5 metadata_json") from exc
    if not isinstance(feature_metadata, dict) or not isinstance(label_metadata, dict):
        raise ValueError("Stage-A.5 metadata_json must contain objects")
    if feature_metadata != label_metadata:
        raise ValueError("features and labels metadata are not identical")
    required = ("dataset", "split", "partial", "query_count",
                "validation_is_test")
    missing = [key for key in required if key not in feature_metadata]
    if missing:
        raise ValueError("metadata is missing {}".format(sorted(missing)))
    partial = _metadata_bool(feature_metadata["partial"], "partial")
    validation_is_test = _metadata_bool(
        feature_metadata["validation_is_test"], "validation_is_test")
    query_count = feature_metadata["query_count"]
    if isinstance(query_count, bool) or not isinstance(query_count, int):
        raise ValueError("metadata query_count must be an integer")
    if query_count != len(np.asarray(features["sample_ids"])):
        raise ValueError("metadata query_count does not match exported rows")
    if feature_metadata["dataset"] != as_scalar_string(features["dataset"], "dataset"):
        raise ValueError("metadata dataset disagrees with features")
    if feature_metadata["split"] != as_scalar_string(features["split"], "split"):
        raise ValueError("metadata split disagrees with features")
    return feature_metadata


def validate_export_protocol(features, labels, *, allow_partial_smoke=False,
                             allow_test_diagnostic=False, operation="analysis"):
    """Enforce formal-vs-diagnostic boundaries for offline Stage-A.5 tools.

    Returns ``(metadata, diagnostic_only)``.  Partial exports and a Charades
    validation split that is the test split are never formal evidence.  The
    explicit flags are intentionally required at every CLI boundary.
    """
    metadata = export_metadata(features, labels)
    partial = metadata["partial"]
    validation_is_test = metadata["validation_is_test"]
    if partial and not allow_partial_smoke:
        raise ValueError(
            "partial=true export is diagnostic-only; pass "
            "--allow-partial-smoke to override")
    if validation_is_test and not allow_test_diagnostic:
        raise ValueError(
            "validation split reuses test_data and cannot support formal "
            "{}; pass --allow-test-diagnostic for fixed-rule diagnostics".format(
                operation))
    diagnostic_only = partial or validation_is_test
    return metadata, diagnostic_only


def save_candidate_exports(features_path, labels_path, features, labels):
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype == object for value in features.values()) or \
            any(np.asarray(value).dtype == object for value in labels.values()):
        raise ValueError("Stage-A.5 exports must not contain object arrays")
    np.savez_compressed(features_path, **features)
    np.savez_compressed(labels_path, **labels)
    with np.load(features_path, allow_pickle=False) as feature_data, \
            np.load(labels_path, allow_pickle=False) as label_data:
        _validate_common(feature_data, label_data)


def interval_iou(props, gt):
    props = np.asarray(props, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    inter = np.maximum(
        0.0, np.minimum(props[..., 1], gt[..., 1]) -
        np.maximum(props[..., 0], gt[..., 0]))
    union = np.maximum(
        0.0, np.maximum(props[..., 1], gt[..., 1]) -
        np.minimum(props[..., 0], gt[..., 0]))
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-12)


def candidate_iou(features, labels):
    start = np.asarray(features["candidate_start"], dtype=np.float64)
    end = np.asarray(features["candidate_end"], dtype=np.float64)
    gt = np.asarray(labels["gt_normalized"], dtype=np.float64)
    props = np.stack([start, end], axis=-1)
    iou = interval_iou(props, gt[:, None, None, :])
    valid = np.asarray(features["candidate_valid"]).astype(bool)
    return np.where(valid, iou, -np.inf)


def selected_props(features, index):
    start = np.take_along_axis(
        np.asarray(features["candidate_start"]), index[..., None], axis=-1)[..., 0]
    end = np.take_along_axis(
        np.asarray(features["candidate_end"]), index[..., None], axis=-1)[..., 0]
    return np.stack([start, end], axis=-1)


def ranking_metrics(props, gt, scores=None):
    props = np.asarray(props)
    gt = np.asarray(gt)
    if props.ndim != 3 or props.shape[-1] != 2 or gt.shape != (props.shape[0], 2):
        raise ValueError("props/gt shape mismatch")
    if scores is None:
        order = np.arange(props.shape[1])[None, :].repeat(props.shape[0], axis=0)
    else:
        scores = np.asarray(scores)
        if scores.shape != props.shape[:2]:
            raise ValueError("scores shape mismatch")
        order = np.argsort(scores, axis=1, kind="stable")
    ranked = np.take_along_axis(props, order[..., None].repeat(2, axis=-1), axis=1)
    ious = interval_iou(ranked, gt[:, None, :])
    r1 = ious[:, 0]
    r5 = ious[:, :min(5, ious.shape[1])].max(axis=1)
    result = {
        "R@1,mIoU": float(r1.mean()),
        "R@5,mIoU": float(r5.mean()),
        "r1_iou": r1,
        "r5_iou": r5,
        "order": order,
    }
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        result["R@1,IoU@{:.1f}".format(threshold)] = float(
            (r1 >= threshold).mean())
        result["R@5,IoU@{:.1f}".format(threshold)] = float(
            (r5 >= threshold).mean())
    return result


def spearman_correlation(x, y):
    """Return tie-aware Spearman correlation without a SciPy dependency."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")

    def rank(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(values.size, dtype=np.float64)
        result[order] = np.arange(values.size, dtype=np.float64)
        unique, inverse, counts = np.unique(values, return_inverse=True,
                                             return_counts=True)
        for group, count in enumerate(counts):
            if count > 1:
                result[inverse == group] = result[inverse == group].mean()
        return result

    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) /
                 (positives.size * negatives.size))


def metric_vector(props, gt, scores=None):
    result = ranking_metrics(props, gt, scores)
    return np.asarray([
        result["R@1,mIoU"], result["R@1,IoU@0.5"],
        result["R@5,mIoU"], result["R@5,IoU@0.5"],
    ], dtype=np.float64)


def video_cluster_bootstrap(baseline_props, stage_props, gt, video_ids,
                            baseline_scores=None, stage_scores=None,
                            repeats=2000, seed=20260902):
    """Paired percentile CIs by resampling videos, not individual queries."""
    baseline_props = np.asarray(baseline_props)
    stage_props = np.asarray(stage_props)
    gt = np.asarray(gt)
    video_ids = np.asarray(video_ids).astype(str)
    if baseline_props.shape != stage_props.shape or \
            baseline_props.shape[0] != gt.shape[0] or video_ids.shape != gt.shape[:1]:
        raise ValueError("bootstrap inputs have inconsistent shapes")
    unique_videos = np.unique(video_ids)
    if unique_videos.size == 0:
        return {}
    groups = [np.flatnonzero(video_ids == video) for video in unique_videos]
    rng = np.random.default_rng(seed)
    deltas = np.empty((int(repeats), 4), dtype=np.float64)
    for repeat in range(int(repeats)):
        selected_groups = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in selected_groups])
        base = metric_vector(
            baseline_props[rows], gt[rows],
            baseline_scores[rows] if baseline_scores is not None else None)
        stage = metric_vector(
            stage_props[rows], gt[rows],
            stage_scores[rows] if stage_scores is not None else None)
        deltas[repeat] = stage - base
    point = metric_vector(baseline_props, gt, baseline_scores)
    point_stage = metric_vector(stage_props, gt, stage_scores)
    names = ("R@1,mIoU", "R@1,IoU@0.5", "R@5,mIoU", "R@5,IoU@0.5")
    return {
        name: {
            "baseline": float(point[index]),
            "stage_a5": float(point_stage[index]),
            "delta": float(point_stage[index] - point[index]),
            "ci95": [float(np.percentile(deltas[:, index], 2.5)),
                     float(np.percentile(deltas[:, index], 97.5))],
        }
        for index, name in enumerate(names)
    }


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def sanitize(item):
        if isinstance(item, dict):
            return {key: sanitize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(value) for value in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    with path.open("w", encoding="utf8") as handle:
        json.dump(sanitize(value), handle, indent=2, sort_keys=True,
                  allow_nan=False)


__all__ = [
    "FEATURE_KEYS", "LABEL_KEYS", "SCHEMA_VERSION", "candidate_iou",
    "dump_json", "export_metadata", "interval_iou", "load_candidate_exports",
    "ranking_metrics", "roc_auc", "save_candidate_exports", "selected_props",
    "sha256_file", "spearman_correlation", "validate_export_protocol",
    "video_cluster_bootstrap",
]
