"""Compact event-boundary index used by the inference-only Stage-A path.

The source event-boundary files contain a considerable amount of per-video
metadata.  Stage A only needs the internal boundary positions and scores, so
this module stores them in a small CSR/ragged NPZ and loads the arrays once per
process.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np


SCHEMA_VERSION = 1
NUM_CLIPS = 200
PLATEAU_TOLERANCE = 1e-7


def _as_1d_array(values, dtype, name):
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    return array


def merge_equal_score_plateaus(indices, scores, tolerance=PLATEAU_TOLERANCE):
    """Merge consecutive equal-score boundary detections.

    The merged position is the rounded mean index and the merged score is the
    maximum score in the plateau.  The input order is preserved; callers that
    receive unsorted detections should sort them before calling this helper.
    """
    indices = _as_1d_array(indices, np.int64, "indices")
    scores = _as_1d_array(scores, np.float64, "scores")
    if indices.size != scores.size:
        raise ValueError("indices and scores must have the same length")
    if not np.isfinite(scores).all():
        raise ValueError("boundary scores must be finite")
    if indices.size == 0:
        return indices.astype(np.int64), scores.astype(np.float32)

    merged_indices = []
    merged_scores = []
    start = 0
    for end in range(1, indices.size + 1):
        continues = (
            end < indices.size
            and indices[end] == indices[end - 1] + 1
            and abs(scores[end] - scores[end - 1]) <= tolerance
        )
        if continues:
            continue
        group_indices = indices[start:end]
        group_scores = scores[start:end]
        # All indices are non-negative.  ``floor(x + 0.5)`` implements the
        # documented four舍五入 behavior, unlike NumPy's ties-to-even rint.
        merged_indices.append(int(np.floor(group_indices.mean() + 0.5)))
        merged_scores.append(float(group_scores.max()))
        start = end
    return (np.asarray(merged_indices, dtype=np.int64),
            np.asarray(merged_scores, dtype=np.float32))


def temporal_nms(indices, scores, min_gap_clips=2):
    """Apply score-prioritized one-dimensional temporal NMS."""
    indices = _as_1d_array(indices, np.int64, "indices")
    scores = _as_1d_array(scores, np.float64, "scores")
    if indices.size != scores.size:
        raise ValueError("indices and scores must have the same length")
    if int(min_gap_clips) != min_gap_clips or min_gap_clips < 0:
        raise ValueError("min_gap_clips must be a non-negative integer")
    if not np.isfinite(scores).all():
        raise ValueError("boundary scores must be finite")

    # Stable ordering makes equal-score ties deterministic and favors the
    # earlier input detection before the final temporal sort.
    order = np.argsort(-scores, kind="stable")
    kept = []
    for candidate in order:
        index = int(indices[candidate])
        if all(abs(index - int(indices[other])) >= min_gap_clips
               for other in kept):
            kept.append(int(candidate))
    kept.sort(key=lambda item: int(indices[item]))
    return (indices[np.asarray(kept, dtype=np.int64)].astype(np.int64),
            scores[np.asarray(kept, dtype=np.int64)].astype(np.float32))


def _extract_detected_boundaries(video):
    """Read detected indices and align scores from one source JSON record."""
    if int(video.get("num_clips", -1)) != NUM_CLIPS:
        raise ValueError("event-boundary source must use num_clips=200")
    detected = _as_1d_array(
        video.get("detected_boundary_indices", []), np.int64,
        "detected_boundary_indices")
    boundary_indices = _as_1d_array(
        video.get("boundary_indices", []), np.int64, "boundary_indices")
    boundary_scores = _as_1d_array(
        video.get("boundary_scores", []), np.float64, "boundary_scores")

    if boundary_scores.size == detected.size:
        scores = boundary_scores
    elif boundary_scores.size == boundary_indices.size:
        score_by_index = {
            int(index): float(score)
            for index, score in zip(boundary_indices, boundary_scores)
        }
        try:
            scores = np.asarray(
                [score_by_index[int(index)] for index in detected],
                dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                "detected boundary index missing from boundary_indices") from exc
    elif detected.size == 0 and boundary_scores.size == 0:
        scores = np.empty(0, dtype=np.float64)
    else:
        raise ValueError("cannot align boundary indices and scores")

    # Endpoints are delimiters, never trim targets.
    internal = (detected > 0) & (detected < NUM_CLIPS - 1)
    indices = detected[internal]
    scores = scores[internal]
    if indices.size:
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        scores = scores[order]
    return merge_equal_score_plateaus(indices, scores)


def _validate_index_arrays(video_ids, offsets, indices, positions, scores,
                           metadata, expected_dataset=None):
    """Validate the on-disk schema and all numeric invariants."""
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported event-boundary index schema")
    if int(metadata.get("num_clips", -1)) != NUM_CLIPS:
        raise ValueError("event-boundary index num_clips does not equal 200")
    dataset = metadata.get("dataset")
    if dataset not in {"activitynet", "charades"}:
        raise ValueError("event-boundary index has an invalid dataset")
    if expected_dataset is not None and dataset != str(expected_dataset).lower():
        raise ValueError(
            "event-boundary index dataset {} does not match {}".format(
                dataset, expected_dataset))
    if metadata.get("position_mapping") != "index / (num_clips - 1)":
        raise ValueError("unsupported event-boundary position mapping")
    if metadata.get("endpoint_policy") != "internal detected boundaries only":
        raise ValueError("unsupported event-boundary endpoint policy")

    video_ids = _as_1d_array(video_ids, object, "video_ids")
    offsets = _as_1d_array(offsets, np.int64, "offsets")
    indices = _as_1d_array(indices, np.int64, "indices")
    positions = _as_1d_array(positions, np.float32, "positions")
    scores = _as_1d_array(scores, np.float32, "scores")
    if offsets.size != video_ids.size + 1:
        raise ValueError("offsets must have one more item than video_ids")
    if offsets.size == 0 or offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != indices.size:
        raise ValueError("offsets are not a valid CSR index")
    if not (indices.size == positions.size == scores.size):
        raise ValueError("boundary arrays must have equal lengths")
    if indices.size and (np.any(indices <= 0) or
                         np.any(indices >= NUM_CLIPS - 1)):
        raise ValueError("stored boundary indices must be internal")
    if not np.isfinite(scores).all() or not np.isfinite(positions).all():
        raise ValueError("stored boundary values must be finite")
    if indices.size and not np.allclose(
            positions, indices.astype(np.float32) / (NUM_CLIPS - 1),
            rtol=0.0, atol=1e-6):
        raise ValueError("boundary positions do not match indices")
    for start, end in zip(offsets[:-1], offsets[1:]):
        local = indices[start:end]
        if local.size and np.any(local[1:] <= local[:-1]):
            raise ValueError("boundary indices must be strictly increasing")


class EventBoundaryIndex:
    """Read-only CSR event-boundary lookup indexed by video id."""

    def __init__(self, path, expected_dataset=None, strict=True):
        self.path = _resolve_index_path(path)
        self.strict = bool(strict)
        with np.load(self.path, allow_pickle=False) as data:
            required = {"video_ids", "offsets", "indices", "positions",
                        "scores", "metadata"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError("event-boundary index missing {}".format(
                    sorted(missing)))
            metadata = _load_metadata(data["metadata"])
            _validate_index_arrays(
                data["video_ids"], data["offsets"], data["indices"],
                data["positions"], data["scores"], metadata,
                expected_dataset=expected_dataset)
            self.video_ids = np.asarray(data["video_ids"]).astype(str)
            self.offsets = np.asarray(data["offsets"], dtype=np.int64)
            self.indices = np.asarray(data["indices"], dtype=np.int64)
            self.positions = np.asarray(data["positions"], dtype=np.float32)
            self.scores = np.asarray(data["scores"], dtype=np.float32)
            self.metadata = metadata
        self._lookup = {
            video_id: index for index, video_id in enumerate(self.video_ids)
        }

    def get(self, video_id):
        """Return ``(positions, scores)`` for a video id."""
        key = str(video_id)
        row = self._lookup.get(key)
        if row is None:
            if self.strict:
                raise KeyError(
                    "video id {!r} is absent from {}".format(key, self.path))
            return (np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        start, end = self.offsets[row:row + 2]
        return self.positions[start:end], self.scores[start:end]


def _load_metadata(value):
    if np.asarray(value).size != 1:
        raise ValueError("metadata must contain one JSON string")
    raw = np.asarray(value).reshape(-1)[0]
    try:
        metadata = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("event-boundary metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("event-boundary metadata must be an object")
    return metadata


def _resolve_index_path(path):
    path = Path(path)
    if path.is_absolute():
        resolved = path
    else:
        # Relative paths are configuration paths relative to the Stage-A
        # project root, rather than the caller's current working directory.
        resolved = Path(__file__).resolve().parents[1] / path
    if not resolved.is_file():
        raise FileNotFoundError("event-boundary index not found: {}".format(
            resolved))
    return resolved


@lru_cache(maxsize=None)
def load_event_boundary_index(path, expected_dataset=None, strict=True):
    """Load and cache one compact index per resolved path/configuration."""
    return EventBoundaryIndex(
        path, expected_dataset=expected_dataset, strict=strict)


def build_boundary_index(input_path, output_path, min_gap_clips=2):
    """Convert a source boundary JSON into a validated CSR NPZ atomically."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    with input_path.open(encoding="utf8") as handle:
        source = json.load(handle)
    if not isinstance(source, dict) or not isinstance(source.get("videos"), dict):
        raise ValueError("source event-boundary JSON must contain videos")
    source_metadata = source.get("metadata", {})
    dataset = str(source_metadata.get("dataset", "")).lower()
    if dataset not in {"activitynet", "charades"}:
        raise ValueError("source metadata must identify activitynet or charades")

    video_ids = []
    offsets = [0]
    all_indices = []
    all_positions = []
    all_scores = []
    for video_id, video in source["videos"].items():
        indices, scores = _extract_detected_boundaries(video)
        indices, scores = temporal_nms(indices, scores, min_gap_clips)
        video_ids.append(str(video_id))
        all_indices.extend(indices.tolist())
        all_positions.extend((indices.astype(np.float32) /
                              (NUM_CLIPS - 1)).tolist())
        all_scores.extend(scores.astype(np.float32).tolist())
        offsets.append(len(all_indices))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_json": str(input_path),
        "dataset": dataset,
        "num_clips": NUM_CLIPS,
        "position_mapping": "index / (num_clips - 1)",
        "plateau_tolerance": PLATEAU_TOLERANCE,
        "min_gap_clips": int(min_gap_clips),
        "endpoint_policy": "internal detected boundaries only",
    }
    arrays = {
        "video_ids": np.asarray(video_ids, dtype=str),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "indices": np.asarray(all_indices, dtype=np.int16),
        "positions": np.asarray(all_positions, dtype=np.float32),
        "scores": np.asarray(all_scores, dtype=np.float32),
        "metadata": np.asarray([json.dumps(metadata, sort_keys=True)],
                                dtype=str),
    }
    _validate_index_arrays(
        arrays["video_ids"], arrays["offsets"], arrays["indices"],
        arrays["positions"], arrays["scores"], metadata,
        expected_dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=str(output_path.parent), prefix=output_path.name + ".",
                suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        # Re-open the temporary file before publication so a corrupt write can
        # never replace a valid existing index.
        EventBoundaryIndex(temp_path, expected_dataset=dataset)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return output_path
