"""Event-boundary extraction for Charades-STA and ActivityNet Captions."""

from .event_boundary import (
    CONTRASTIVE_KERNEL,
    EventBoundaryResult,
    compute_boundary_scores,
    detect_event_boundaries,
    l2_normalize,
    resample_features_like_cpl,
    temporal_self_similarity,
)

__all__ = [
    "CONTRASTIVE_KERNEL",
    "EventBoundaryResult",
    "compute_boundary_scores",
    "detect_event_boundaries",
    "l2_normalize",
    "resample_features_like_cpl",
    "temporal_self_similarity",
]
