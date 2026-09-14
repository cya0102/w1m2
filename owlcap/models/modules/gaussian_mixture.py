"""Reconstructor-aware Gaussian mixture proposals.

This module adapts PPS's Gaussian mixture proposal idea to CPL.  Proposal k
contains min(k + 1, max_components) Gaussian components.  Components inside a
proposal learn independent centers and a shared width; their importance is
predicted from features produced by the query reconstructor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureProposalGenerator(nn.Module):
    """Generate a small set of variable-complexity mixture proposals."""

    def __init__(self, hidden_size, num_proposals, max_components=5,
                 sigma=4.0, importance_temperature=1.0,
                 boundary_mode="outer", boundary_shrink=0.0, eps=1e-6):
        super().__init__()
        if num_proposals < 1:
            raise ValueError("num_proposals must be positive")
        if max_components < 1:
            raise ValueError("max_components must be positive")
        if sigma <= 0:
            raise ValueError("mixture sigma must be positive")
        if importance_temperature <= 0:
            raise ValueError("importance temperature must be positive")
        if boundary_mode not in {"outer", "weighted"}:
            raise ValueError(
                "boundary mode must be either 'outer' or 'weighted'")
        if not 0 <= boundary_shrink < 1:
            raise ValueError("boundary shrink must be in [0, 1)")

        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        self.max_components = max_components
        self.sigma = float(sigma)
        self.importance_temperature = float(importance_temperature)
        self.boundary_mode = boundary_mode
        self.boundary_shrink = float(boundary_shrink)
        self.eps = eps

        self.component_counts = [
            min(index + 1, max_components)
            for index in range(num_proposals)
        ]
        self.total_components = sum(self.component_counts)

        self.center_head = nn.Linear(hidden_size, self.total_components)
        self.width_head = nn.Linear(hidden_size, num_proposals)

        # PPS computes component importance from reconstructor features rather
        # than directly from the proposal generator.  Keep the same separation
        # here: this scorer is called only after component-conditioned query
        # reconstruction has produced one feature per Gaussian component.
        self.importance_projection = nn.Linear(
            hidden_size, hidden_size, bias=False)
        self.importance_score = nn.Linear(hidden_size, 1, bias=False)

        valid = torch.zeros(
            num_proposals, max_components, dtype=torch.bool)
        for proposal_index, count in enumerate(self.component_counts):
            valid[proposal_index, :count] = True
        self.register_buffer("component_valid_mask", valid)

    def predict_components(self, global_feature, sequence_length):
        """Predict component centers, shared widths and Gaussian masks.

        Returns flattened component tensors.  Components are ordered proposal
        by proposal, matching ``component_counts``.
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        centers = torch.sigmoid(self.center_head(global_feature))
        proposal_widths = torch.sigmoid(self.width_head(global_feature))
        widths = torch.cat([
            proposal_widths[:, index:index + 1].expand(-1, count)
            for index, count in enumerate(self.component_counts)
        ], dim=1)

        position = torch.linspace(
            0, 1, sequence_length, device=centers.device,
            dtype=centers.dtype).view(1, 1, -1)
        standard_deviation = widths.unsqueeze(-1).clamp_min(1e-2) / self.sigma
        component_weights = torch.exp(
            -0.5 * ((position - centers.unsqueeze(-1))
                    / standard_deviation) ** 2)
        component_weights = component_weights / component_weights.amax(
            dim=-1, keepdim=True).clamp_min(self.eps)
        return centers, widths, component_weights

    def combine(self, centers, widths, component_weights,
                reconstructor_features):
        """Combine components using reconstruction-aware importance weights."""
        batch_size, total_components = centers.shape
        if total_components != self.total_components:
            raise ValueError("component count does not match generator layout")
        if reconstructor_features.shape[:2] != centers.shape:
            raise ValueError(
                "reconstructor features do not match Gaussian components")

        scores = self.importance_score(torch.tanh(
            self.importance_projection(reconstructor_features))).squeeze(-1)
        scores = scores / self.importance_temperature

        mixture_weights = []
        proposal_centers = []
        proposal_widths = []
        context_centers = []
        context_widths = []
        padded_centers = []
        padded_widths = []
        padded_component_weights = []
        padded_importance = []

        offset = 0
        for count in self.component_counts:
            component_slice = slice(offset, offset + count)
            group_scores = scores[:, component_slice]
            group_importance = torch.softmax(group_scores, dim=-1)
            group_centers = centers[:, component_slice]
            group_widths = widths[:, component_slice]
            group_weights = component_weights[:, component_slice]

            mixture = (
                group_importance.unsqueeze(-1) * group_weights).sum(dim=1)
            mixture = mixture / mixture.amax(
                dim=-1, keepdim=True).clamp_min(self.eps)

            component_left = (group_centers - group_widths / 2).clamp(0, 1)
            component_right = (group_centers + group_widths / 2).clamp(0, 1)
            outer_left = component_left.min(dim=-1)[0]
            outer_right = component_right.max(dim=-1)[0]
            weighted_left = (
                group_importance * component_left).sum(dim=-1)
            weighted_right = (
                group_importance * component_right).sum(dim=-1)
            if self.boundary_mode == "outer":
                left, right = outer_left, outer_right
            else:
                left, right = weighted_left, weighted_right

            # Keep the uncontracted component envelope for background mining.
            # This is PPS's outer-hull geometry: every positive component is
            # excluded from the left/right contextual negatives.
            context_centers.append((outer_left + outer_right) / 2)
            context_widths.append(
                (outer_right - outer_left).clamp_min(self.eps))

            # Optional contraction is retained only as an explicit ablation.
            # The fixed V4 configuration uses zero contraction so that the
            # returned interval represents the same evidence as the mask.
            raw_width = (right - left).clamp_min(self.eps)
            contraction = self.boundary_shrink * raw_width / 2
            left = (left + contraction).clamp(0, 1)
            right = (right - contraction).clamp(0, 1)
            final_width = (right - left).clamp_min(self.eps)

            mixture_weights.append(mixture)
            proposal_centers.append((left + right) / 2)
            proposal_widths.append(final_width)

            pad_components = self.max_components - count
            padded_centers.append(F.pad(group_centers, (0, pad_components)))
            padded_widths.append(F.pad(group_widths, (0, pad_components)))
            padded_component_weights.append(F.pad(
                group_weights, (0, 0, 0, pad_components)))
            padded_importance.append(F.pad(
                group_importance, (0, pad_components)))
            offset += count

        mixture_weights = torch.stack(mixture_weights, dim=1)
        proposal_centers = torch.stack(proposal_centers, dim=1)
        proposal_widths = torch.stack(proposal_widths, dim=1)
        context_centers = torch.stack(context_centers, dim=1)
        context_widths = torch.stack(context_widths, dim=1)

        return {
            "mixture_weights": mixture_weights,
            "proposal_centers": proposal_centers,
            "proposal_widths": proposal_widths,
            "context_centers": context_centers,
            "context_widths": context_widths,
            "component_centers": torch.stack(padded_centers, dim=1),
            "component_widths": torch.stack(padded_widths, dim=1),
            "component_weights": torch.stack(
                padded_component_weights, dim=1),
            "component_importance": torch.stack(padded_importance, dim=1),
            "component_valid_mask": self.component_valid_mask.unsqueeze(0)
                .expand(batch_size, -1, -1),
        }

    def extra_repr(self):
        return (
            "num_proposals={}, counts={}, sigma={}, boundary={}, shrink={}".format(
                self.num_proposals, self.component_counts, self.sigma,
                self.boundary_mode, self.boundary_shrink))
