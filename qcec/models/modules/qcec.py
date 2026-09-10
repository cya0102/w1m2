"""Query-conditioned coherent event clusters.

The QCEC implementation deliberately keeps the temporal structure outside of
the transformer.  Cluster assignments and visual transition scores are fixed
metadata, while the pooled projected frame states and the query-conditioned
fusion remain differentiable.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_enhanced_pool(
    frame_states: torch.Tensor,
    frame_mask: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_bounds: torch.Tensor,
    cluster_mask: torch.Tensor,
    edge_boost: float = 1.0,
    edge_power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pool projected frame states into fixed contiguous clusters.

    ``cluster_ids`` and the masks are treated as structural metadata.  The
    reduction itself is differentiable with respect to ``frame_states``.  A
    scatter reduction is used instead of materialising a ``[B, M, T, D]``
    tensor, which keeps the memory cost small for the 200-frame input.
    """
    if frame_states.dim() != 3:
        raise ValueError("frame_states must have shape [B, T, D]")
    if frame_mask.dim() != 2 or frame_mask.shape != frame_states.shape[:2]:
        raise ValueError("frame_mask must have shape [B, T]")
    if cluster_ids.dim() != 2 or cluster_ids.shape != frame_states.shape[:2]:
        raise ValueError("cluster_ids must have shape [B, T]")
    if cluster_bounds.dim() != 3 or cluster_bounds.size(-1) != 2:
        raise ValueError("cluster_bounds must have shape [B, M, 2]")
    if cluster_mask.shape != cluster_bounds.shape[:2]:
        raise ValueError("cluster_mask must have shape [B, M]")
    if edge_boost < 0:
        raise ValueError("edge_boost must be non-negative")
    if edge_power <= 0:
        raise ValueError("edge_power must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    batch_size, num_frames, hidden_size = frame_states.shape
    num_clusters = cluster_bounds.size(1)
    device = frame_states.device

    ids = cluster_ids.to(device=device, dtype=torch.long)
    valid_clusters = cluster_mask.to(device=device, dtype=torch.bool)
    valid_frames = frame_mask.to(device=device, dtype=torch.bool)
    safe_ids = ids.clamp(min=0, max=max(num_clusters - 1, 0))

    # Convert normalized half-open bounds back to sampled-frame indices.  This
    # gives the first and last frame of a cluster local coordinates 0 and 1;
    # singleton clusters use the neutral coordinate 0.5.
    bounds = cluster_bounds.to(
        device=device, dtype=torch.float32).detach()
    start_indices = torch.round(bounds[..., 0] * float(num_frames)).long()
    end_indices = torch.round(bounds[..., 1] * float(num_frames)).long()
    selected_start = torch.gather(start_indices, 1, safe_ids)
    selected_end = torch.gather(end_indices, 1, safe_ids)
    selected_length = (selected_end - selected_start).clamp_min(1)
    denominator = (selected_length - 1).clamp_min(1).to(torch.float32)
    frame_position = torch.arange(
        num_frames, device=device, dtype=torch.float32).view(1, -1)
    local_position = (
        frame_position - selected_start.to(torch.float32)) / denominator
    singleton = selected_length <= 1
    local_position = torch.where(
        singleton, torch.full_like(local_position, 0.5), local_position)
    local_position = local_position.clamp(0.0, 1.0)

    assigned_cluster_valid = torch.gather(valid_clusters, 1, safe_ids)
    valid_frames = valid_frames & (ids >= 0) & (ids < num_clusters)
    valid_frames = valid_frames & assigned_cluster_valid
    edge_profile = (2.0 * (local_position - 0.5).abs()).pow(edge_power)
    frame_weights = (1.0 + float(edge_boost) * edge_profile)
    frame_weights = frame_weights * valid_frames.to(frame_weights.dtype)

    # Accumulate in float32 under AMP and cast only the final tokens back to
    # the state dtype.  ``index_add_`` propagates gradients through values.
    states_float = frame_states.float()
    weighted_states = states_float * frame_weights.unsqueeze(-1)
    batch_offsets = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        .view(-1, 1) * num_clusters)
    flat_index = (batch_offsets + safe_ids).reshape(-1)
    flat_states = weighted_states.reshape(-1, hidden_size)
    flat_weights = frame_weights.reshape(-1, 1)
    pooled = states_float.new_zeros(batch_size * num_clusters, hidden_size)
    denominators = states_float.new_zeros(batch_size * num_clusters, 1)
    pooled.index_add_(0, flat_index, flat_states)
    denominators.index_add_(0, flat_index, flat_weights)
    pooled = pooled / denominators.clamp_min(float(eps))
    pooled = pooled.view(batch_size, num_clusters, hidden_size)

    # A structurally valid but empty cluster is also padding for the online
    # interaction.  The multiplication keeps its value and gradient zero.
    has_frames = denominators.view(batch_size, num_clusters, 1) > float(eps)
    pooled = pooled * (valid_clusters.unsqueeze(-1) & has_frames).to(
        pooled.dtype)
    return pooled.to(dtype=frame_states.dtype)


def pool_query_roles(
    query_states: torch.Tensor,
    query_mask: torch.Tensor,
    query_role_mask: torch.Tensor,
    query_role_valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return predicate/entity/sentence query units with safe fallbacks.

    Role masks are intersected with the ordinary query mask after collation,
    so OOV and padding tokens cannot contribute.  A role with no usable token
    (or with ``query_role_valid=False``) falls back to all valid sentence
    tokens.  The returned tensor has shape ``[B, 3, D]``.
    """
    if query_states.dim() != 3:
        raise ValueError("query_states must have shape [B, W, D]")
    batch_size, query_length, hidden_size = query_states.shape
    if query_mask.shape != (batch_size, query_length):
        raise ValueError("query_mask must have shape [B, W]")
    if query_role_mask.shape != (batch_size, 3, query_length):
        raise ValueError("query_role_mask must have shape [B, 3, W]")
    if query_role_valid.shape != (batch_size, 3):
        raise ValueError("query_role_valid must have shape [B, 3]")

    query_valid = query_mask.to(dtype=torch.bool, device=query_states.device)
    role_mask = query_role_mask.to(
        dtype=torch.bool, device=query_states.device)
    role_valid = query_role_valid.to(
        dtype=torch.bool, device=query_states.device)
    role_tokens = role_mask & query_valid.unsqueeze(1)
    usable_role = role_tokens.any(dim=-1) & role_valid
    fallback = query_valid.unsqueeze(1).expand(-1, 3, -1)
    effective_mask = torch.where(
        usable_role.unsqueeze(-1), role_tokens, fallback)

    weights = effective_mask.to(dtype=torch.float32)
    states = query_states.float()
    pooled = torch.bmm(weights.view(batch_size * 3, 1, query_length),
                       states.unsqueeze(1).expand(-1, 3, -1, -1).reshape(
                           batch_size * 3, query_length, hidden_size)).squeeze(1)
    denominator = weights.sum(dim=-1).reshape(batch_size * 3, 1)
    pooled = pooled / denominator.clamp_min(float(eps))
    return pooled.view(batch_size, 3, hidden_size).to(query_states.dtype)


def compute_transition_scores(
    cluster_tokens: torch.Tensor,
    cluster_mask: torch.Tensor,
    transition_threshold: float = 0.2,
    transition_temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute query-independent visual transition scores.

    The cosine distance is explicitly detached: QCEC must not move the fixed
    visual event structure by changing the projected frame representation.
    """
    if cluster_tokens.dim() != 3:
        raise ValueError("cluster_tokens must have shape [B, M, D]")
    if cluster_mask.shape != cluster_tokens.shape[:2]:
        raise ValueError("cluster_mask must have shape [B, M]")
    if transition_temperature <= 0:
        raise ValueError("transition_temperature must be positive")
    batch_size, num_clusters, _ = cluster_tokens.shape
    mask = cluster_mask.to(
        device=cluster_tokens.device, dtype=torch.bool)
    if num_clusters < 2:
        empty = cluster_tokens.new_zeros(batch_size, 0)
        return empty, mask.new_zeros(batch_size, 0)

    normalized = F.normalize(cluster_tokens.detach().float(), dim=-1, eps=1e-6)
    distance = 1.0 - (
        normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1)
    distance = distance.clamp(min=-1.0, max=2.0)
    score = torch.sigmoid(torch.clamp(
        (distance - float(transition_threshold))
        / float(transition_temperature), min=-30.0, max=30.0))
    transition_mask = mask[:, :-1] & mask[:, 1:]
    score = score * transition_mask.to(score.dtype)
    return score.to(cluster_tokens.dtype), transition_mask


def compute_directional_hints(
    relevance: torch.Tensor,
    transition_score: torch.Tensor,
    transition_mask: torch.Tensor,
    relevance_change_margin: float = 0.05,
    cluster_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build left/right boundary hints and directional barriers."""
    if relevance.dim() != 2:
        raise ValueError("relevance must have shape [B, M]")
    batch_size, num_clusters = relevance.shape
    expected_transition_shape = (batch_size, max(num_clusters - 1, 0))
    if transition_score.shape != expected_transition_shape:
        raise ValueError("transition_score has an incompatible shape")
    if transition_mask.shape != expected_transition_shape:
        raise ValueError("transition_mask has an incompatible shape")
    if relevance_change_margin < 0:
        raise ValueError("relevance_change_margin must be non-negative")
    if cluster_mask is None:
        valid_clusters = torch.ones(
            relevance.shape, dtype=torch.bool, device=relevance.device)
    else:
        if cluster_mask.shape != relevance.shape:
            raise ValueError("cluster_mask must have shape [B, M]")
        valid_clusters = cluster_mask.to(
            device=relevance.device, dtype=torch.bool)

    if num_clusters == 0:
        empty = relevance.new_zeros(batch_size, 0)
        return empty, empty, empty, empty

    score = transition_score.to(dtype=relevance.dtype)
    transition_valid = transition_mask.to(
        device=relevance.device, dtype=torch.bool)
    score = score * transition_valid.to(score.dtype)
    if num_clusters == 1:
        zeros = relevance.new_zeros(batch_size, 0)
        edge = relevance * valid_clusters.to(relevance.dtype)
        return edge, edge, zeros, zeros

    left_inner = score * F.relu(
        relevance[:, 1:] - relevance[:, :-1]
        - float(relevance_change_margin))
    right_inner = score * F.relu(
        relevance[:, :-1] - relevance[:, 1:]
        - float(relevance_change_margin))
    left_hint = torch.cat([relevance[:, :1], left_inner], dim=1)
    right_hint = torch.cat([right_inner, relevance[:, -1:]], dim=1)
    left_hint = left_hint * valid_clusters.to(left_hint.dtype)
    right_hint = right_hint * valid_clusters.to(right_hint.dtype)

    # ``lr`` means relevant on the left and irrelevant on the right (an
    # endpoint should not cross to the right); ``rl`` is the reverse.  The
    # boundary hints use the opposite orientation for their endpoint side:
    # entering a relevant cluster is a left hint, leaving one is a right hint.
    barrier_lr = right_inner
    barrier_rl = left_inner
    return left_hint, right_hint, barrier_lr, barrier_rl


class QCECProposalAdapter(nn.Module):
    """Zero-initialized residual adapter for the original proposal feature."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.input_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, base_feature: torch.Tensor, global_hint: torch.Tensor
    ) -> torch.Tensor:
        if base_feature.shape != global_hint.shape:
            raise ValueError("base feature and QCEC hint must have the same shape")
        update = self.input_projection(torch.cat(
            [base_feature, global_hint], dim=-1))
        return base_feature + self.output_projection(F.gelu(update))


class QueryConditionedCoherentEventClusters(nn.Module):
    """Query-conditioned interaction over fixed contiguous event clusters."""

    def __init__(
        self,
        hidden_size: int,
        num_proposals: int,
        max_components: int,
        edge_boost: float = 1.0,
        edge_power: float = 2.0,
        attention_dim: Optional[int] = None,
        relevance_temperature: float = 0.1,
        relevance_change_margin: float = 0.05,
        transition_threshold: float = 0.2,
        transition_temperature: float = 0.05,
        prior_bias_scale: float = 0.25,
        eps: float = 1e-6,
        use_slot_prior: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_proposals < 1:
            raise ValueError("num_proposals must be positive")
        if max_components < 1:
            raise ValueError("max_components must be positive")
        if edge_boost < 0:
            raise ValueError("edge_boost must be non-negative")
        if edge_power <= 0:
            raise ValueError("edge_power must be positive")
        if attention_dim is None:
            attention_dim = hidden_size
        if attention_dim < 1:
            raise ValueError("attention_dim must be positive")
        if relevance_temperature <= 0:
            raise ValueError("relevance_temperature must be positive")
        if transition_temperature <= 0:
            raise ValueError("transition_temperature must be positive")
        if relevance_change_margin < 0:
            raise ValueError("relevance_change_margin must be non-negative")
        if transition_threshold < 0:
            raise ValueError("transition_threshold must be non-negative")
        if prior_bias_scale < 0:
            raise ValueError("prior_bias_scale must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        self.max_components = max_components
        self.total_components = sum(
            min(index + 1, max_components)
            for index in range(num_proposals))
        self.edge_boost = float(edge_boost)
        self.edge_power = float(edge_power)
        self.attention_dim = int(attention_dim)
        self.relevance_temperature = float(relevance_temperature)
        self.relevance_change_margin = float(relevance_change_margin)
        self.transition_threshold = float(transition_threshold)
        self.transition_temperature = float(transition_temperature)
        self.prior_bias_scale = float(prior_bias_scale)
        self.eps = float(eps)
        self.use_slot_prior = bool(use_slot_prior)

        self.role_embeddings = nn.Parameter(torch.zeros(3, hidden_size))
        self.cluster_attention_projection = nn.Linear(
            hidden_size, attention_dim, bias=False)
        self.query_attention_projection = nn.Linear(
            hidden_size, attention_dim, bias=False)
        self.relevance_cluster_projection = nn.Linear(
            hidden_size, attention_dim, bias=False)
        self.relevance_query_projection = nn.Linear(
            hidden_size, attention_dim, bias=False)
        self.cluster_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cluster_fusion_norm = nn.LayerNorm(hidden_size)
        self.relevance_bias = nn.Parameter(torch.zeros(()))
        self.global_projection = nn.Linear(hidden_size * 3 + 2, hidden_size)

        if self.use_slot_prior:
            # The slot's coarse temporal location is useful context for the
            # bias heads in addition to the visual cluster representation.
            # The final layers below are still zero initialized, so this
            # positional path cannot perturb a baseline warm start.
            self.slot_position_projection = nn.Linear(2, hidden_size)
            self.center_bias_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, max_components),
            )
            self.width_bias_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.single_bias_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 2),
            )
            self._zero_initialize_prior_heads()

    def _zero_initialize_prior_heads(self) -> None:
        for head in (
            self.center_bias_head,
            self.width_bias_head,
            self.single_bias_head,
        ):
            output = head[-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _weighted_feature(
        features: torch.Tensor,
        weights: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        weights = weights.to(dtype=features.dtype)
        mass = weights.sum(dim=1, keepdim=True)
        value = (features * weights.unsqueeze(-1)).sum(dim=1)
        value = value / mass.clamp_min(float(eps))
        if fallback is not None:
            value = torch.where(
                (mass > float(eps)).expand_as(value), fallback, value)
        return value

    def _slot_outputs(
        self,
        slot_features: torch.Tensor,
        slot_valid: torch.Tensor,
        slot_prior_center: torch.Tensor,
        slot_prior_width: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_slot_prior:
            return None, None, None
        slot_features = slot_features * slot_valid.unsqueeze(-1).to(
            slot_features.dtype)
        slot_position = torch.stack([
            slot_prior_center, slot_prior_width,
        ], dim=-1).to(dtype=slot_features.dtype, device=slot_features.device)
        slot_features_with_position = slot_features + self.slot_position_projection(
            slot_position)
        center_by_slot = self.center_bias_head(slot_features_with_position)
        center_bias = torch.cat([
            center_by_slot[:, index, :min(index + 1, self.max_components)]
            for index in range(self.num_proposals)
        ], dim=-1)
        width_bias = self.width_bias_head(slot_features_with_position).squeeze(-1)
        single_bias = self.single_bias_head(slot_features_with_position)
        valid = slot_valid.unsqueeze(-1).to(center_bias.dtype)
        center_bias = center_bias * torch.cat([
            valid[:, index].expand(
                -1, min(index + 1, self.max_components))
            for index in range(self.num_proposals)
        ], dim=-1)
        width_bias = width_bias * slot_valid.to(width_bias.dtype)
        single_bias = single_bias * slot_valid.unsqueeze(-1).to(
            single_bias.dtype)
        return (
            center_bias * self.prior_bias_scale,
            width_bias * self.prior_bias_scale,
            single_bias * self.prior_bias_scale,
        )

    def forward(
        self,
        frame_states: torch.Tensor,
        frame_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        query_role_mask: torch.Tensor,
        query_role_valid: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_bounds: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if frame_states.dim() != 3:
            raise ValueError("frame_states must have shape [B, T, D]")
        batch_size, _, hidden_size = frame_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError("frame state hidden size does not match QCEC")
        if query_states.size(0) != batch_size or query_states.size(-1) != hidden_size:
            raise ValueError("query states must share batch and hidden dimensions")
        if cluster_bounds.size(0) != batch_size:
            raise ValueError("cluster metadata must share the batch dimension")
        num_clusters = cluster_bounds.size(1)
        if num_clusters < self.num_proposals:
            raise ValueError(
                "QCEC requires at least as many clusters as proposals")
        if cluster_mask.shape != (batch_size, num_clusters):
            raise ValueError("cluster_mask has an incompatible shape")
        if cluster_ids.shape != frame_states.shape[:2]:
            raise ValueError("cluster_ids has an incompatible shape")

        cluster_mask = cluster_mask.to(
            device=frame_states.device, dtype=torch.bool).detach()
        cluster_bounds = cluster_bounds.to(
            device=frame_states.device, dtype=frame_states.dtype).detach()
        cluster_tokens = boundary_enhanced_pool(
            frame_states,
            frame_mask,
            cluster_ids,
            cluster_bounds,
            cluster_mask,
            edge_boost=self.edge_boost,
            edge_power=self.edge_power,
            eps=self.eps,
        )

        query_units = pool_query_roles(
            query_states, query_mask, query_role_mask, query_role_valid,
            eps=self.eps)
        query_units = query_units + self.role_embeddings.unsqueeze(0).to(
            dtype=query_units.dtype, device=query_units.device)

        role_valid = query_role_valid.to(
            device=frame_states.device, dtype=torch.bool).clone()
        # The sentence unit is always available to the cross-modal attention;
        # its pooled value is zero only when the input query itself is empty.
        role_valid[:, 2] = True
        cluster_projection = self.cluster_attention_projection(cluster_tokens)
        query_projection = self.query_attention_projection(query_units)
        attention_logits = torch.einsum(
            "bma,bla->bml", cluster_projection, query_projection)
        attention_logits = attention_logits / float(self.attention_dim) ** 0.5
        attention_logits = attention_logits.masked_fill(
            ~role_valid.unsqueeze(1), -1e4)
        cluster_attention = torch.softmax(attention_logits, dim=-1)
        cluster_attention = cluster_attention * cluster_mask.unsqueeze(-1).to(
            cluster_attention.dtype)

        attended_query = torch.einsum(
            "bml,bld->bmd", cluster_attention, query_units)
        fusion_input = torch.cat([
            cluster_tokens,
            attended_query,
            cluster_tokens * attended_query,
        ], dim=-1)
        enhanced_clusters = self.cluster_fusion_norm(
            cluster_tokens + self.cluster_fusion(fusion_input))
        enhanced_clusters = enhanced_clusters * cluster_mask.unsqueeze(-1).to(
            enhanced_clusters.dtype)

        relevance_cluster = self.relevance_cluster_projection(enhanced_clusters)
        relevance_query = self.relevance_query_projection(query_units[:, 2])
        relevance_cosine = F.cosine_similarity(
            relevance_cluster,
            relevance_query.unsqueeze(1),
            dim=-1,
            eps=1e-6,
        )
        relevance = torch.sigmoid(torch.clamp(
            (relevance_cosine - self.relevance_bias)
            / self.relevance_temperature,
            min=-30.0,
            max=30.0,
        ))
        relevance = relevance * cluster_mask.to(relevance.dtype)

        transition_score, transition_mask = compute_transition_scores(
            cluster_tokens,
            cluster_mask,
            transition_threshold=self.transition_threshold,
            transition_temperature=self.transition_temperature,
        )
        left_hint, right_hint, barrier_lr, barrier_rl = (
            compute_directional_hints(
                relevance,
                transition_score,
                transition_mask,
                relevance_change_margin=self.relevance_change_margin,
                cluster_mask=cluster_mask,
            ))
        if num_clusters > 1:
            transition_positions = cluster_bounds[:, :-1, 1]
            transition_positions = transition_positions * transition_mask.to(
                transition_positions.dtype)
        else:
            transition_positions = cluster_bounds.new_zeros(batch_size, 0)

        relevance_feature = self._weighted_feature(
            enhanced_clusters, relevance, eps=self.eps)
        left_feature = self._weighted_feature(
            enhanced_clusters, left_hint, fallback=relevance_feature,
            eps=self.eps)
        right_feature = self._weighted_feature(
            enhanced_clusters, right_hint, fallback=relevance_feature,
            eps=self.eps)

        starts = cluster_bounds[..., 0]
        ends = cluster_bounds[..., 1]
        relevance_mass = relevance.sum(dim=-1, keepdim=True)
        fallback_index = relevance.argmax(dim=-1)
        fallback_start = torch.gather(starts, 1, fallback_index.unsqueeze(-1))
        fallback_end = torch.gather(ends, 1, fallback_index.unsqueeze(-1))
        left_mass = left_hint.sum(dim=-1, keepdim=True)
        right_mass = right_hint.sum(dim=-1, keepdim=True)
        soft_start = (left_hint * starts).sum(dim=-1, keepdim=True) / (
            left_mass.clamp_min(self.eps))
        soft_end = (right_hint * ends).sum(dim=-1, keepdim=True) / (
            right_mass.clamp_min(self.eps))
        soft_start = torch.where(
            left_mass > self.eps, soft_start, fallback_start.detach())
        soft_end = torch.where(
            right_mass > self.eps, soft_end, fallback_end.detach())
        # The fallback is also well-defined for an all-padding metadata row:
        # the corresponding bounds are zero, while the mask remains explicit.
        soft_start = soft_start * (relevance_mass >= 0).to(soft_start.dtype)
        soft_end = soft_end * (relevance_mass >= 0).to(soft_end.dtype)

        global_input = torch.cat([
            relevance_feature,
            left_feature,
            right_feature,
            soft_start.to(relevance_feature.dtype),
            soft_end.to(relevance_feature.dtype),
        ], dim=-1)
        global_hint = self.global_projection(global_input)

        # Top-k is only used for the optional proposal prior.  A tiny stable
        # index term makes ties deterministic without materially changing the
        # relevance values used by the differentiable global path.
        tie_break = torch.arange(
            num_clusters, device=frame_states.device,
            dtype=relevance.dtype).view(1, -1)
        tie_break = (float(num_clusters) - tie_break) * 1e-8
        top_scores = relevance.masked_fill(~cluster_mask, -1e4) + tie_break
        top_indices = torch.topk(
            top_scores, self.num_proposals, dim=-1, largest=True,
            sorted=True).indices
        slot_valid = torch.gather(cluster_mask, 1, top_indices)
        slot_features = torch.gather(
            enhanced_clusters,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        slot_features = slot_features * slot_valid.unsqueeze(-1).to(
            slot_features.dtype)
        slot_prior_center = torch.gather(
            (starts + ends) / 2.0, 1, top_indices)
        slot_prior_width = torch.gather(
            (ends - starts).clamp_min(0.0), 1, top_indices)
        slot_prior_center = slot_prior_center * slot_valid.to(
            slot_prior_center.dtype)
        slot_prior_width = slot_prior_width * slot_valid.to(
            slot_prior_width.dtype)
        center_logit_bias, width_logit_bias, single_logit_bias = (
            self._slot_outputs(
                slot_features, slot_valid, slot_prior_center, slot_prior_width))

        raw_cluster_ids = cluster_ids.to(
            device=frame_states.device, dtype=torch.long)
        safe_ids = raw_cluster_ids.clamp(
            min=0, max=max(num_clusters - 1, 0))
        frame_hints = torch.gather(
            torch.stack([relevance, left_hint, right_hint], dim=-1),
            1,
            safe_ids.unsqueeze(-1).expand(-1, -1, 3),
        )
        frame_hints = frame_hints * (
            (raw_cluster_ids >= 0) & (raw_cluster_ids < num_clusters)
        ).to(frame_hints.dtype).unsqueeze(-1)

        return {
            "cluster_tokens": cluster_tokens,
            "enhanced_clusters": enhanced_clusters,
            "cluster_attention": cluster_attention,
            "cluster_relevance": relevance,
            "left_boundary_hint": left_hint,
            "right_boundary_hint": right_hint,
            "transition_score": transition_score,
            "transition_positions": transition_positions,
            "transition_mask": transition_mask,
            "barrier_lr": barrier_lr,
            "barrier_rl": barrier_rl,
            "frame_hints": frame_hints,
            "global_hint": global_hint,
            "slot_features": slot_features,
            "slot_prior_center": slot_prior_center,
            "slot_prior_width": slot_prior_width,
            "center_logit_bias": center_logit_bias,
            "width_logit_bias": width_logit_bias,
            "single_logit_bias": single_logit_bias,
            "slot_valid": slot_valid,
        }


def snap_proposal_boundaries(
    center: torch.Tensor,
    width: torch.Tensor,
    cluster_bounds: torch.Tensor,
    cluster_relevance: torch.Tensor,
    left_boundary_hint: torch.Tensor,
    right_boundary_hint: torch.Tensor,
    cluster_mask: torch.Tensor,
    frame_lengths: torch.Tensor,
    radius_frames: int = 2,
    min_confidence: float = 0.15,
    min_relevance: float = 0.5,
    min_width: float = 0.01,
    edge_min_confidence: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Snap proposal endpoints to nearby confident cluster boundaries."""
    if center.dim() != 2 or width.shape != center.shape:
        raise ValueError("center and width must have shape [B, N]")
    if cluster_bounds.dim() != 3 or cluster_bounds.size(-1) != 2:
        raise ValueError("cluster_bounds must have shape [B, M, 2]")
    batch_size, num_proposals = center.shape
    if cluster_bounds.size(0) != batch_size:
        raise ValueError("cluster metadata must share the batch dimension")
    if cluster_relevance.shape != cluster_bounds.shape[:2]:
        raise ValueError("cluster_relevance has an incompatible shape")
    if left_boundary_hint.shape != cluster_relevance.shape:
        raise ValueError("left_boundary_hint has an incompatible shape")
    if right_boundary_hint.shape != cluster_relevance.shape:
        raise ValueError("right_boundary_hint has an incompatible shape")
    if cluster_mask.shape != cluster_relevance.shape:
        raise ValueError("cluster_mask has an incompatible shape")
    if radius_frames < 0:
        raise ValueError("radius_frames must be non-negative")
    if min_confidence < 0 or min_relevance < 0:
        raise ValueError("snap thresholds must be non-negative")
    if min_width < 0:
        raise ValueError("min_width must be non-negative")
    if edge_min_confidence is not None and edge_min_confidence < 0:
        raise ValueError("edge_min_confidence must be non-negative")

    device = center.device
    dtype = center.dtype
    bounds = cluster_bounds.to(device=device, dtype=dtype)
    relevance = cluster_relevance.to(device=device, dtype=dtype)
    left_hint = left_boundary_hint.to(device=device, dtype=dtype)
    right_hint = right_boundary_hint.to(device=device, dtype=dtype)
    valid_clusters = cluster_mask.to(device=device, dtype=torch.bool)
    lengths = frame_lengths.to(device=device, dtype=dtype).view(batch_size)
    radius = float(radius_frames) / lengths.clamp_min(1.0)

    raw_start = (center - width / 2.0).clamp(0.0, 1.0)
    raw_end = (center + width / 2.0).clamp(0.0, 1.0)
    candidate_start = bounds[..., 0]
    candidate_end = bounds[..., 1]
    start_valid = valid_clusters & (left_hint >= float(min_confidence)) & (
        relevance >= float(min_relevance))
    end_valid = valid_clusters & (right_hint >= float(min_confidence)) & (
        relevance >= float(min_relevance))
    # The synthetic boundaries at 0 and 1 are useful for the global hint but
    # are poor default snapping targets.  Permit them only when the caller
    # explicitly supplies an independent, usually higher, edge threshold.
    edge_threshold = (
        float("inf") if edge_min_confidence is None
        else float(edge_min_confidence))
    if cluster_bounds.size(1) > 0:
        start_valid[:, 0] = start_valid[:, 0] & (
            left_hint[:, 0] >= edge_threshold)
        end_valid[:, -1] = end_valid[:, -1] & (
            right_hint[:, -1] >= edge_threshold)

    def nearest(
        endpoint: torch.Tensor,
        candidates: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidates.size(-1) == 0:
            return endpoint, torch.zeros_like(endpoint, dtype=torch.bool)
        distance = (candidates.unsqueeze(1) - endpoint.unsqueeze(-1)).abs()
        allowed = candidate_valid.unsqueeze(1) & (
            distance <= radius.unsqueeze(1).unsqueeze(-1))
        distance = distance.masked_fill(~allowed, float("inf"))
        min_distance, index = distance.min(dim=-1)
        found = torch.isfinite(min_distance)
        selected = torch.gather(
            candidates.unsqueeze(1).expand(-1, num_proposals, -1),
            -1,
            index.unsqueeze(-1),
        ).squeeze(-1)
        return torch.where(found, selected, endpoint), found

    snapped_start, start_found = nearest(
        raw_start, candidate_start, start_valid)
    snapped_end, end_found = nearest(raw_end, candidate_end, end_valid)
    legal = (snapped_end - snapped_start) >= float(min_width)
    legal = legal & (snapped_end >= snapped_start)
    snapped_start = torch.where(legal, snapped_start, raw_start)
    snapped_end = torch.where(legal, snapped_end, raw_end)
    snapped_center = (snapped_start + snapped_end) / 2.0
    snapped_width = (snapped_end - snapped_start).clamp_min(0.0)

    start_delta = snapped_start - raw_start
    end_delta = snapped_end - raw_end
    diagnostics = {
        "start_delta": start_delta,
        "end_delta": end_delta,
        "start_abs_delta": start_delta.abs(),
        "end_abs_delta": end_delta.abs(),
        "start_moved": start_found & legal,
        "end_moved": end_found & legal,
        "valid": legal,
        # Explicit aliases make the output convenient for runner diagnostics.
        "snap_start_delta": start_delta,
        "snap_end_delta": end_delta,
    }
    return snapped_center, snapped_width, diagnostics


# Short alias for callers that refer to the component by the specification's
# informal ``QCECModule`` name.
QCECModule = QueryConditionedCoherentEventClusters
