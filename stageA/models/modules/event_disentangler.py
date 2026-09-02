"""Stable low-rank event subspaces for weakly supervised grounding."""

import torch
import torch.nn as nn


class LowRankEventDisentangler(nn.Module):
    """Build and persist a positive-eigenvalue contrastive event subspace.

    V1 applied SVD to an indefinite contrastive covariance matrix. SVD orders
    directions by the absolute eigenvalue, so large negative (background-
    dominant) directions were frequently selected. V2 uses ``eigh`` and keeps
    only directions with positive contrastive variance. Covariances are trace
    normalized and accumulated with EMA to make the basis stable across
    batches and available at inference time.
    """

    def __init__(self, hidden_size, rank=8, cpca_alpha=1.0,
                 covariance_ema=0.95, normalize_covariance=True, eps=1e-6):
        super().__init__()
        if rank < 1 or rank > hidden_size:
            raise ValueError("event rank must be in [1, hidden_size]")
        if cpca_alpha < 0:
            raise ValueError("cPCA alpha must be non-negative")
        if not 0 <= covariance_ema < 1:
            raise ValueError("covariance EMA must be in [0, 1)")
        self.hidden_size = hidden_size
        self.rank = rank
        self.cpca_alpha = cpca_alpha
        self.covariance_ema = covariance_ema
        self.normalize_covariance = normalize_covariance
        self.eps = eps

        self.register_buffer(
            "running_event_cov", torch.zeros(hidden_size, hidden_size))
        self.register_buffer(
            "running_background_cov", torch.zeros(hidden_size, hidden_size))
        self.register_buffer(
            "event_basis", torch.zeros(hidden_size, rank))
        self.register_buffer(
            "subspace_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "positive_rank", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "largest_contrastive_eigenvalue", torch.zeros(()))
        self.register_buffer(
            "smallest_selected_eigenvalue", torch.zeros(()))

    def _weighted_pool(self, features, weights, mask):
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

    def _covariance(self, samples):
        samples = samples.float()
        centered = samples - samples.mean(dim=0, keepdim=True)
        covariance = centered.t().mm(centered) / max(centered.size(0) - 1, 1)
        covariance = 0.5 * (covariance + covariance.t())
        if self.normalize_covariance:
            covariance = covariance / torch.trace(covariance).clamp_min(self.eps)
        return covariance

    def _positive_basis(self, event_covariance, background_covariance):
        contrastive = event_covariance - self.cpca_alpha * background_covariance
        contrastive = 0.5 * (contrastive + contrastive.t())
        eigenvalues, eigenvectors = torch.linalg.eigh(contrastive)
        order = eigenvalues.argsort(descending=True)[:self.rank]
        top_values = eigenvalues[order]
        basis = eigenvectors[:, order]
        # Zero columns whose contrastive variance is not positive. This avoids
        # reintroducing precisely the background directions cPCA should remove.
        positive = top_values > self.eps
        basis = basis * positive.to(basis.dtype).unsqueeze(0)
        return basis, top_values, positive

    @torch.no_grad()
    def update_subspace(self, event_samples, background_samples):
        event_covariance = self._covariance(event_samples.detach())
        background_covariance = self._covariance(background_samples.detach())
        if self.subspace_updates.item() == 0:
            self.running_event_cov.copy_(event_covariance)
            self.running_background_cov.copy_(background_covariance)
        else:
            decay = self.covariance_ema
            self.running_event_cov.mul_(decay).add_(
                event_covariance, alpha=1.0 - decay)
            self.running_background_cov.mul_(decay).add_(
                background_covariance, alpha=1.0 - decay)
        basis, top_values, positive = self._positive_basis(
            self.running_event_cov, self.running_background_cov)
        self.event_basis.copy_(basis.to(self.event_basis.dtype))
        self.positive_rank.copy_(positive.sum())
        self.largest_contrastive_eigenvalue.copy_(top_values[0])
        if positive.any():
            self.smallest_selected_eigenvalue.copy_(top_values[positive][-1])
        else:
            self.smallest_selected_eigenvalue.zero_()
        self.subspace_updates.add_(1)

    def forward(self, proposal_features, proposal_mask, positive_weights,
                negative_weights, selection_weights, update_subspace=True):
        """
        Args:
            proposal_features: ``(B*K, T, D)`` query-conditioned frames.
            proposal_mask: ``(B*K, T)`` valid-frame mask.
            positive_weights: ``(B*K, T)`` positive Gaussian masks.
            negative_weights: ``(B*K, 2, T)`` context masks.
            selection_weights: ``(B, K)`` detached soft pseudo-labels.
            update_subspace: update EMA statistics in this forward pass.
        """
        positive = self._weighted_pool(
            proposal_features, positive_weights, proposal_mask)
        negative = torch.stack([
            self._weighted_pool(
                proposal_features, negative_weights[:, side], proposal_mask)
            for side in range(negative_weights.size(1))
        ], dim=1)
        difference = positive - negative.mean(dim=1)

        batch_size, num_props = selection_weights.shape
        if batch_size * num_props != difference.size(0):
            raise ValueError("selection weights do not match proposal features")
        difference_by_proposal = difference.view(batch_size, num_props, -1)
        negative_by_proposal = negative.view(
            batch_size, num_props, negative.size(1), -1)
        selected_difference = (
            difference_by_proposal * selection_weights.unsqueeze(-1)).sum(dim=1)
        selected_background = (
            negative_by_proposal
            * selection_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        if self.training and update_subspace:
            self.update_subspace(
                selected_difference,
                selected_background.reshape(-1, selected_background.size(-1)))

        basis = self.event_basis.to(dtype=difference.dtype)
        event = difference.mm(basis).mm(basis.t())
        return positive, negative, event

    def extra_repr(self):
        return "hidden_size={}, rank={}, alpha={}, ema={}".format(
            self.hidden_size, self.rank, self.cpca_alpha, self.covariance_ema)
