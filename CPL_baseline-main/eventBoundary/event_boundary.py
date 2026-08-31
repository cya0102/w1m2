"""Paper-faithful temporal event-boundary extraction.

The implementation follows Sec. III-E.1 and Eq. (6) of
"Mismatched Pairs Dynamic Correction for Cross-Modal Alignment in Video
Moment Retrieval".  The endpoint handling and normalized event-span
construction follow the official EaTR implementation cited by that section.

Only NumPy is required for the boundary calculation.  In particular, the
implementation does not materialize the full ``T x T`` temporal
self-similarity matrix when computing scores.  Eq. (6)'s kernel is the outer
product ``[1, 1, 0, -1, -1]^T [1, 1, 0, -1, -1]``; consequently, the score on
the TSM diagonal has a mathematically equivalent O(TD) form rather than an
O(T^2) materialization.  ``temporal_self_similarity`` remains available for
inspection and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_KERNEL_VECTOR = np.asarray([1.0, 1.0, 0.0, -1.0, -1.0], dtype=np.float32)
CONTRASTIVE_KERNEL = np.outer(_KERNEL_VECTOR, _KERNEL_VECTOR).astype(np.float32)


def _validate_features(features: np.ndarray) -> np.ndarray:
    """Return finite, two-dimensional float32 clip features."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            "video features must have shape (num_clips, feature_dim); "
            "received {}".format(array.shape)
        )
    if array.shape[0] == 0:
        raise ValueError("video features must contain at least one clip")
    if array.shape[1] == 0:
        raise ValueError("video features must have a non-zero feature dimension")
    if not np.isfinite(array).all():
        raise ValueError("video features contain NaN or infinity")
    return array


def l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each clip feature using the reference implementation's epsilon."""

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")
    array = _validate_features(features)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / (norms + np.float32(eps))


def temporal_self_similarity(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Construct the cosine temporal self-similarity matrix ``S``."""

    normalized = l2_normalize(features, eps=eps)
    return np.matmul(normalized, normalized.T)


def compute_boundary_scores(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute Eq. (6)'s contrastive-kernel score along the TSM diagonal.

    This is mathematically equivalent to zero-padding the temporal
    self-similarity matrix by two cells, applying the fixed 5x5 kernel as a
    2-D correlation, and taking the main diagonal of the result.  Floating-
    point reduction order can cause tiny round-off differences from an
    explicitly materialized convolution.

    Args:
        features: Array with shape ``(num_clips, feature_dim)``.
        eps: Epsilon used by clip-wise L2 normalization.

    Returns:
        A float32 vector with one raw boundary score per clip.
    """

    normalized = l2_normalize(features, eps=eps)
    num_clips, feature_dim = normalized.shape

    # Z = w^T w, so diag(conv(TSM, Z))[t] = ||sum_i w_i x[t+i-2]||^2.
    # Padding the features with zeros is identical to zero-padding the TSM.
    padded = np.zeros((num_clips + 4, feature_dim), dtype=np.float32)
    padded[2 : 2 + num_clips] = normalized
    contrast = (
        padded[0:num_clips]
        + padded[1 : num_clips + 1]
        - padded[3 : num_clips + 3]
        - padded[4 : num_clips + 4]
    )
    return np.einsum("td,td->t", contrast, contrast, dtype=np.float32)


def resample_features_like_cpl(features: np.ndarray, num_clips: int) -> np.ndarray:
    """Reproduce ``BaseDataset._sample_frame_features`` from the CPL baseline.

    CPL feeds exactly ``max_num_frames`` clips to the model.  Using this helper
    before boundary extraction therefore keeps offline boundary indices aligned
    with the tensors consumed by CPL, including its rounding and endpoint
    behavior.
    """

    array = _validate_features(features)
    if num_clips <= 0:
        raise ValueError("num_clips must be a positive integer")

    keep_idx = np.arange(num_clips + 1, dtype=np.float64)
    keep_idx = keep_idx / float(num_clips) * len(array)
    keep_idx = np.round(keep_idx).astype(np.int64)
    keep_idx[keep_idx >= len(array)] = len(array) - 1

    sampled = []
    for start, end in zip(keep_idx[:-1], keep_idx[1:]):
        if start > end:
            raise RuntimeError("CPL resampling produced a reversed interval")
        if start == end:
            sampled.append(array[start])
        else:
            sampled.append(array[start:end].mean(axis=0))
    return np.stack(sampled, axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class EventBoundaryResult:
    """Boundary scores, selected indices, and event spans for one video."""

    scores: np.ndarray
    threshold: float
    boundary_indices: np.ndarray

    @property
    def num_clips(self) -> int:
        return int(self.scores.shape[0])

    def event_intervals_indices(self) -> List[List[int]]:
        """Return reference-style consecutive boundary-point pairs."""

        if len(self.boundary_indices) < 2:
            return []
        return [
            [int(start), int(end)]
            for start, end in zip(self.boundary_indices[:-1], self.boundary_indices[1:])
        ]

    def detected_boundary_indices(self) -> List[int]:
        """Return score-selected internal peaks, excluding forced endpoints."""

        last_index = self.num_clips - 1
        return [
            int(index)
            for index in self.boundary_indices
            if int(index) not in (0, last_index)
        ]

    def segmentation_cuts_indices(self) -> List[int]:
        """Return cuts suitable for complete, non-overlapping ``[start, end)`` events."""

        return [0] + self.detected_boundary_indices() + [self.num_clips]

    def event_intervals_indices_half_open(self) -> List[List[int]]:
        """Return full-coverage, half-open clip intervals used for mean pooling."""

        cuts = self.segmentation_cuts_indices()
        return [[start, end] for start, end in zip(cuts[:-1], cuts[1:]) if end > start]

    def event_spans_normalized_cw(self) -> List[List[float]]:
        """Return cleaned reference-style ``(center, width) / num_clips`` spans."""

        denominator = float(self.num_clips)
        return [
            [(start + end) / (2.0 * denominator), (end - start) / denominator]
            for start, end in self.event_intervals_indices()
        ]

    def event_spans_half_open_normalized_cw(self) -> List[List[float]]:
        """Return normalized spans for the full-coverage half-open intervals."""

        denominator = float(self.num_clips)
        return [
            [(start + end) / (2.0 * denominator), (end - start) / denominator]
            for start, end in self.event_intervals_indices_half_open()
        ]

    def boundary_positions_normalized(self) -> List[float]:
        """Map boundary indices onto [0, 1], with the last clip at 1."""

        if self.num_clips == 1:
            return [0.0 for _ in self.boundary_indices]
        denominator = float(self.num_clips - 1)
        return [float(index) / denominator for index in self.boundary_indices]

    def to_record(
        self,
        source_num_clips: int,
        duration: Optional[float] = None,
        include_all_scores: bool = False,
    ) -> Dict[str, object]:
        """Convert the result into the JSON schema used by the dataset CLI."""

        selected_scores = [float(self.scores[index]) for index in self.boundary_indices]
        record: Dict[str, object] = {
            "source_num_clips": int(source_num_clips),
            "num_clips": self.num_clips,
            "threshold": float(self.threshold),
            "boundary_indices": [int(index) for index in self.boundary_indices],
            "detected_boundary_indices": self.detected_boundary_indices(),
            "boundary_scores": selected_scores,
            "boundary_positions_normalized": self.boundary_positions_normalized(),
            "event_intervals_boundary_coordinates": self.event_intervals_indices(),
            "event_spans_normalized_cw": self.event_spans_normalized_cw(),
            "segmentation_cuts_indices": self.segmentation_cuts_indices(),
            "event_intervals_indices_half_open": self.event_intervals_indices_half_open(),
            "pooling_event_spans_normalized_cw": (
                self.event_spans_half_open_normalized_cw()
            ),
        }
        if include_all_scores:
            record["all_boundary_scores"] = [float(value) for value in self.scores]

        if duration is not None:
            duration = float(duration)
            if not np.isfinite(duration) or duration < 0:
                raise ValueError("duration must be a finite non-negative number")
            positions = self.boundary_positions_normalized()
            times = [position * duration for position in positions]
            cut_times = [
                float(cut) / float(self.num_clips) * duration
                for cut in self.segmentation_cuts_indices()
            ]
            record["duration"] = duration
            record["boundary_times_seconds"] = times
            record["event_intervals_boundary_seconds"] = [
                [start, end] for start, end in zip(times[:-1], times[1:])
            ]
            record["event_intervals_seconds"] = [
                [start, end] for start, end in zip(cut_times[:-1], cut_times[1:])
            ]
        return record


def detect_event_boundaries(features: np.ndarray, eps: float = 1e-8) -> EventBoundaryResult:
    """Detect event boundaries using the paper's complete post-processing.

    The raw score mean is used as the threshold.  Clips below it are removed,
    and a size-3 sliding maximum keeps local maxima (ties are retained, matching
    max-filter semantics and the cited implementation).  The first and last
    valid clip are always inserted as event delimiters; the official reference
    implementation realizes this by assigning them a score of 100 before the
    local-maximum test.
    """

    scores = compute_boundary_scores(features, eps=eps)
    # The official PyTorch path keeps the score tensor and its mean in float32.
    threshold = float(scores.mean(dtype=np.float32))
    selection_scores = scores.copy()

    # Cosine inputs bound the raw score well below 100 (there are 16 non-zero
    # kernel cells), so this reproduces the cited implementation exactly.
    selection_scores[0] = np.float32(100.0)
    selection_scores[-1] = np.float32(100.0)

    previous = np.roll(selection_scores, 1)
    following = np.roll(selection_scores, -1)
    keep = (
        (previous <= selection_scores)
        & (following <= selection_scores)
        & (selection_scores >= np.float32(threshold))
    )
    boundary_indices = np.flatnonzero(keep).astype(np.int64)

    return EventBoundaryResult(
        scores=scores,
        threshold=threshold,
        boundary_indices=boundary_indices,
    )
