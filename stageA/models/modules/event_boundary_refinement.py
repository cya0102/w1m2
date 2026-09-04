"""Parameter-free geometry and soft-mask utilities for Stage-A trimming."""

import torch


class EventBoundaryRefiner:
    """Generate inward-only boundary candidates for existing proposals."""

    CANDIDATE_NAMES = (
        "original",
        "left_near",
        "left_strong",
        "right_near",
        "right_strong",
        "both_near",
        "both_strong",
    )

    def __init__(self, num_clips=200, min_boundary_margin_clips=1,
                 min_candidate_width=0.02, min_retained_ratio=0.25,
                 soft_window_temperature=0.01):
        if int(num_clips) != num_clips or int(num_clips) < 2:
            raise ValueError("num_clips must be an integer >= 2")
        if min_boundary_margin_clips < 0:
            raise ValueError("min_boundary_margin_clips must be non-negative")
        if min_candidate_width <= 0 or min_retained_ratio <= 0:
            raise ValueError("candidate width constraints must be positive")
        if soft_window_temperature <= 0:
            raise ValueError("soft_window_temperature must be positive")
        self.num_clips = int(num_clips)
        self.min_boundary_margin_clips = float(min_boundary_margin_clips)
        self.min_candidate_width = float(min_candidate_width)
        self.min_retained_ratio = float(min_retained_ratio)
        self.soft_window_temperature = float(soft_window_temperature)

    def build_candidates(self, center, width, boundary_positions,
                         boundary_scores, boundary_mask,
                         return_boundary_confidence=False,
                         return_boundary_scores=False):
        """Return seven inward candidates and their validity mask.

        ``candidate_valid`` is intentionally conservative: duplicate
        coordinates keep the lower candidate index, and candidates failing
        either minimum-width rule are invalidated.  Candidate zero is always
        the original proposal and is the fallback used by the selector.
        """
        if center.ndim != 2 or width.shape != center.shape:
            raise ValueError("center and width must both have shape [B, N]")
        if boundary_positions.ndim != 2:
            raise ValueError("boundary tensors must have shape [B, J]")
        if (boundary_scores.shape != boundary_positions.shape or
                boundary_mask.shape != boundary_positions.shape or
                boundary_positions.size(0) != center.size(0)):
            raise ValueError("boundary tensor shapes do not match proposals")
        if not (torch.isfinite(center).all() and torch.isfinite(width).all()):
            raise ValueError("proposal geometry must be finite")

        original_start = (center - width / 2).clamp(0.0, 1.0)
        original_end = (center + width / 2).clamp(0.0, 1.0)
        if torch.any(original_end <= original_start):
            raise ValueError("proposals must have positive width")
        batch_size, num_props = center.shape
        starts = original_start.unsqueeze(-1).expand(
            batch_size, num_props, 7).clone()
        ends = original_end.unsqueeze(-1).expand(
            batch_size, num_props, 7).clone()
        valid = torch.zeros(
            batch_size, num_props, 7, dtype=torch.bool, device=center.device)
        valid[..., 0] = True
        boundary_confidence = torch.zeros(
            batch_size, num_props, 7, dtype=center.dtype,
            device=center.device)
        left_boundary_score = torch.zeros_like(boundary_confidence)
        right_boundary_score = torch.zeros_like(boundary_confidence)
        margin = self.min_boundary_margin_clips / float(self.num_clips - 1)

        # The loops are over proposals, not frames.  This keeps the selection
        # rules transparent and the candidate tensor itself fully batched.
        for batch_index in range(batch_size):
            positions = boundary_positions[batch_index]
            scores = boundary_scores[batch_index]
            available = boundary_mask[batch_index].bool()
            if torch.any(available & ~torch.isfinite(positions)):
                raise ValueError("boundary positions must be finite")
            if torch.any(available & ~torch.isfinite(scores)):
                raise ValueError("boundary scores must be finite")
            available_scores = scores[available]

            def score_percentile(score):
                if available_scores.numel() <= 1:
                    return score.new_ones(())
                # A strict-lower rank gives equal scores the same deterministic
                # percentile and avoids depending on torch sort stability.
                return (available_scores < score).to(score.dtype).sum() / (
                    available_scores.numel() - 1)

            for proposal_index in range(num_props):
                start = original_start[batch_index, proposal_index]
                end = original_end[batch_index, proposal_index]
                midpoint = (start + end) / 2
                left_pool = (available & (positions > start + margin) &
                             (positions <= midpoint))
                right_pool = (available & (positions >= midpoint) &
                              (positions < end - margin))

                left_near = left_strong = None
                right_near = right_strong = None
                if torch.any(left_pool):
                    left_values = positions[left_pool]
                    left_near = left_values.min()
                    left_scores = scores.masked_fill(~left_pool,
                                                     -torch.inf)
                    left_strong = positions[torch.argmax(left_scores)]
                if torch.any(right_pool):
                    right_values = positions[right_pool]
                    right_near = right_values.max()
                    right_scores = scores.masked_fill(~right_pool,
                                                      -torch.inf)
                    right_strong = positions[torch.argmax(right_scores)]

                selections = (
                    (left_near, None),
                    (left_strong, None),
                    (None, right_near),
                    (None, right_strong),
                    (left_near, right_near),
                    (left_strong, right_strong),
                )
                for candidate_index, (candidate_start, candidate_end) in \
                        enumerate(selections, start=1):
                    if candidate_start is not None:
                        starts[batch_index, proposal_index, candidate_index] = \
                            candidate_start
                    if candidate_end is not None:
                        ends[batch_index, proposal_index, candidate_index] = \
                            candidate_end
                    # A left/right candidate requires the corresponding pool;
                    # a both candidate requires both sides.
                    needed = ((candidate_start is not None) or
                              (candidate_end is not None))
                    valid[batch_index, proposal_index, candidate_index] = needed
                left_near_conf = (score_percentile(
                    scores[(positions == left_near).nonzero(as_tuple=False)[0, 0]])
                    if left_near is not None else None)
                left_strong_conf = (score_percentile(
                    scores[(positions == left_strong).nonzero(as_tuple=False)[0, 0]])
                    if left_strong is not None else None)
                right_near_conf = (score_percentile(
                    scores[(positions == right_near).nonzero(as_tuple=False)[0, 0]])
                    if right_near is not None else None)
                right_strong_conf = (score_percentile(
                    scores[(positions == right_strong).nonzero(as_tuple=False)[0, 0]])
                    if right_strong is not None else None)
                confidence_values = (
                    left_near_conf, left_strong_conf,
                    right_near_conf, right_strong_conf,
                    (torch.minimum(left_near_conf, right_near_conf)
                     if left_near_conf is not None and right_near_conf is not None
                     else None),
                    (torch.minimum(left_strong_conf, right_strong_conf)
                     if left_strong_conf is not None and right_strong_conf is not None
                     else None),
                )
                for candidate_index, confidence in enumerate(
                        confidence_values, start=1):
                    if confidence is not None:
                        boundary_confidence[
                            batch_index, proposal_index, candidate_index] = confidence
                score_values = (
                    (scores[(positions == left_near).nonzero(as_tuple=False)[0, 0]]
                     if left_near is not None else None),
                    (scores[(positions == left_strong).nonzero(as_tuple=False)[0, 0]]
                     if left_strong is not None else None),
                    (scores[(positions == right_near).nonzero(as_tuple=False)[0, 0]]
                     if right_near is not None else None),
                    (scores[(positions == right_strong).nonzero(as_tuple=False)[0, 0]]
                     if right_strong is not None else None),
                )
                for candidate_index, score in enumerate(
                        score_values, start=1):
                    if score is not None:
                        left_boundary_score[batch_index, proposal_index,
                                            candidate_index] = score if candidate_index < 3 else 0
                        right_boundary_score[batch_index, proposal_index,
                                             candidate_index] = score if candidate_index >= 3 else 0
                if score_values[0] is not None and score_values[2] is not None:
                    left_boundary_score[batch_index, proposal_index, 5] = score_values[0]
                    right_boundary_score[batch_index, proposal_index, 5] = score_values[2]
                if score_values[1] is not None and score_values[3] is not None:
                    left_boundary_score[batch_index, proposal_index, 6] = score_values[1]
                    right_boundary_score[batch_index, proposal_index, 6] = score_values[3]

        candidate_width = ends - starts
        minimum_width = torch.maximum(
            center.new_full(center.shape, self.min_candidate_width),
            self.min_retained_ratio * (original_end - original_start))
        finite_geometry = (torch.isfinite(starts) & torch.isfinite(ends) &
                           (starts >= 0) & (ends <= 1) &
                           (starts < ends))
        valid &= finite_geometry & (candidate_width >= minimum_width.unsqueeze(-1))

        # Candidate order is the priority order.  In particular, original is
        # preferred to any exact duplicate and near is preferred to strong.
        duplicate_tolerance = 1e-7
        for candidate_index in range(1, 7):
            duplicate = torch.zeros_like(valid[..., candidate_index])
            for previous in range(candidate_index):
                duplicate |= (
                    valid[..., previous]
                    & (torch.abs(starts[..., candidate_index] -
                                 starts[..., previous]) <= duplicate_tolerance)
                    & (torch.abs(ends[..., candidate_index] -
                                 ends[..., previous]) <= duplicate_tolerance))
            valid[..., candidate_index] &= ~duplicate

        candidate_type = torch.arange(7, device=center.device,
                                      dtype=torch.long)
        if return_boundary_confidence:
            boundary_confidence = boundary_confidence.masked_fill(~valid, 0)
            left_boundary_score = left_boundary_score.masked_fill(~valid, 0)
            right_boundary_score = right_boundary_score.masked_fill(~valid, 0)
            if return_boundary_scores:
                return (starts, ends, valid, candidate_type,
                        boundary_confidence, left_boundary_score,
                        right_boundary_score)
            return (starts, ends, valid, candidate_type,
                    boundary_confidence)
        return starts, ends, valid, candidate_type

    def build_candidate_masks(self, original_mask, candidate_start,
                              candidate_end, candidate_valid=None):
        """Build normalized soft inward masks; candidate zero is untouched."""
        masks, mask_valid = self.build_candidate_masks_with_validity(
            original_mask, candidate_start, candidate_end, candidate_valid)
        return masks

    def build_candidate_masks_with_validity(
            self, original_mask, candidate_start, candidate_end,
            candidate_valid=None):
        if original_mask.ndim != 3:
            raise ValueError("original_mask must have shape [B, N, L]")
        if candidate_start.shape != candidate_end.shape or \
                candidate_start.ndim != 3 or candidate_start.size(-1) != 7:
            raise ValueError("candidate geometry must have shape [B, N, 7]")
        if candidate_start.shape[:2] != original_mask.shape[:2]:
            raise ValueError("candidate geometry does not match original mask")
        if original_mask.size(-1) < 2:
            raise ValueError("soft masks need at least two time positions")
        if not torch.isfinite(original_mask).all():
            raise ValueError("original mask must be finite")
        if candidate_valid is None:
            candidate_valid = torch.ones(
                candidate_start.shape, dtype=torch.bool,
                device=candidate_start.device)
        if candidate_valid.shape != candidate_start.shape:
            raise ValueError("candidate_valid shape does not match geometry")

        masks, _, mask_valid, _ = self._build_trim_shell_masks(
            original_mask, candidate_start, candidate_end, candidate_valid)
        return masks, mask_valid

    def build_shell_masks_with_validity(
            self, original_mask, candidate_start, candidate_end,
            candidate_valid=None):
        """Return normalized removed-shell masks and their validity.

        The shell is computed from the same unnormalized soft window as trim:
        ``original * (1 - window)``.  It is normalized independently, so the
        complement is formed before either side is rescaled.
        """
        _, shell_masks, _, shell_valid = self._build_trim_shell_masks(
            original_mask, candidate_start, candidate_end, candidate_valid)
        return shell_masks, shell_valid

    def build_candidate_masks_with_shell_validity(
            self, original_mask, candidate_start, candidate_end,
            candidate_valid=None):
        """Return ``(trim, shell, trim_valid, shell_valid)`` together."""
        return self._build_trim_shell_masks(
            original_mask, candidate_start, candidate_end, candidate_valid)

    def build_candidate_masks_with_shell(
            self, original_mask, candidate_start, candidate_end,
            candidate_valid=None):
        """Compatibility convenience wrapper returning normalized masks."""
        trim, shell, _, _ = self._build_trim_shell_masks(
            original_mask, candidate_start, candidate_end, candidate_valid)
        return trim, shell

    def build_raw_trim_shell_masks(
            self, original_mask, candidate_start, candidate_end):
        """Return raw (not independently normalized) trim and shell masks."""
        self._validate_mask_inputs(
            original_mask, candidate_start, candidate_end,
            torch.ones(candidate_start.shape, dtype=torch.bool,
                       device=candidate_start.device))
        sequence_length = original_mask.size(-1)
        positions = torch.linspace(
            0.0, 1.0, sequence_length, device=candidate_start.device,
            dtype=candidate_start.dtype).view(1, 1, 1, sequence_length)
        temperature = candidate_start.new_tensor(self.soft_window_temperature)
        window = torch.sigmoid(
            (positions - candidate_start.unsqueeze(-1)) / temperature) * \
            torch.sigmoid(
                (candidate_end.unsqueeze(-1) - positions) / temperature)
        raw_trim = original_mask.unsqueeze(2) * window
        raw_shell = original_mask.unsqueeze(2) * (1.0 - window)
        return raw_trim, raw_shell

    def _validate_mask_inputs(self, original_mask, candidate_start,
                              candidate_end, candidate_valid):
        if original_mask.ndim != 3:
            raise ValueError("original_mask must have shape [B, N, L]")
        if candidate_start.shape != candidate_end.shape or \
                candidate_start.ndim != 3 or candidate_start.size(-1) != 7:
            raise ValueError("candidate geometry must have shape [B, N, 7]")
        if candidate_start.shape[:2] != original_mask.shape[:2]:
            raise ValueError("candidate geometry does not match original mask")
        if original_mask.size(-1) < 2:
            raise ValueError("soft masks need at least two time positions")
        if not torch.isfinite(original_mask).all():
            raise ValueError("original mask must be finite")
        if candidate_valid.shape != candidate_start.shape:
            raise ValueError("candidate_valid shape does not match geometry")

    def _build_trim_shell_masks(self, original_mask, candidate_start,
                                candidate_end, candidate_valid=None):
        if candidate_valid is None:
            candidate_valid = torch.ones(
                candidate_start.shape, dtype=torch.bool,
                device=candidate_start.device)
        self._validate_mask_inputs(
            original_mask, candidate_start, candidate_end, candidate_valid)
        batch_size, num_props, _, sequence_length = (
            candidate_start.size(0), candidate_start.size(1),
            candidate_start.size(2), original_mask.size(2))
        raw_trim, raw_shell = self.build_raw_trim_shell_masks(
            original_mask, candidate_start, candidate_end)
        trim_masks = original_mask.new_zeros(
            batch_size, num_props, 7, sequence_length)
        shell_masks = original_mask.new_zeros(
            batch_size, num_props, 7, sequence_length)
        trim_masks[..., 0, :] = original_mask
        trim_valid = candidate_valid.clone()
        trim_valid[..., 0] = True
        shell_valid = candidate_valid.clone()
        shell_valid[..., 0] = False

        trim_max = raw_trim.amax(dim=-1, keepdim=True)
        shell_max = raw_shell.amax(dim=-1, keepdim=True)
        trim_validity = (torch.isfinite(raw_trim).all(dim=-1) &
                         torch.isfinite(trim_max.squeeze(-1)) &
                         (trim_max.squeeze(-1) >= 1e-6))
        shell_validity = (torch.isfinite(raw_shell).all(dim=-1) &
                          torch.isfinite(shell_max.squeeze(-1)) &
                          (shell_max.squeeze(-1) >= 1e-6))
        trim_normalized = raw_trim / trim_max.clamp_min(1e-6)
        shell_normalized = raw_shell / shell_max.clamp_min(1e-6)
        trim_normalized = torch.where(
            torch.isfinite(trim_normalized), trim_normalized,
            torch.zeros_like(trim_normalized))
        shell_normalized = torch.where(
            torch.isfinite(shell_normalized), shell_normalized,
            torch.zeros_like(shell_normalized))
        trim_masks[..., 1:, :] = trim_normalized[..., 1:, :]
        shell_masks[..., 1:, :] = shell_normalized[..., 1:, :]
        trim_valid[..., 1:] &= trim_validity[..., 1:]
        shell_valid[..., 1:] &= shell_validity[..., 1:]
        trim_masks = trim_masks.masked_fill(~trim_valid.unsqueeze(-1), 0)
        shell_masks = shell_masks.masked_fill(~shell_valid.unsqueeze(-1), 0)
        return trim_masks, shell_masks, trim_valid, shell_valid


__all__ = ["EventBoundaryRefiner"]
