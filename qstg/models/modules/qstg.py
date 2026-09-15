"""Query-Subgraph Temporal Grounder modules.

The implementation deliberately keeps the three integration points separate:
query/temporal encoding, component enhancement, and proposal scoring.  This
matches the proposal's CPL call order and makes the graph usable in small CPU
unit tests without constructing a full CPL model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qstg_ops import (
    analytic_soft_box,
    build_barrier,
    build_relation_kernel,
    build_temporal_adjacency,
    compute_connectivity,
    compute_pair_relation_satisfaction,
    compute_relation_satisfaction,
    compute_transition_score,
    connected_refine,
    masked_log_mean_exp,
    probabilistic_coverage,
    resample_to_nodes,
    temporal_pool,
)


def _zero_init(module):
    """Zero the output of a linear adapter while preserving its input path."""
    if not isinstance(module, nn.Linear):
        raise TypeError("zero initialization expects nn.Linear")
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _masked_mean(values, mask, dim, eps=1e-6):
    """Mean over valid entries, performing the reduction in float32."""
    values32 = values.float()
    mask32 = mask.to(values32.dtype)
    numerator = (values32 * mask32).sum(dim=dim)
    denominator = mask32.sum(dim=dim).clamp_min(eps)
    result = numerator / denominator
    return result.to(values.dtype)


def _required_or_valid(phrase_valid, phrase_required):
    required = phrase_valid.bool() & phrase_required.bool()
    has_required = required.any(dim=-1, keepdim=True)
    return torch.where(has_required, required, phrase_valid.bool())


def _masked_node_softmax(logits, node_mask, temperature, eps=1e-6):
    """Softmax over valid temporal nodes with a finite empty-row fallback."""
    valid = node_mask.bool().unsqueeze(1)
    scaled = logits.float() / max(float(temperature), eps)
    scaled = scaled.masked_fill(~valid, -1e4)
    result = torch.softmax(scaled, dim=-1)
    result = result * valid.to(result.dtype)
    result = result / result.sum(-1, keepdim=True).clamp_min(eps)
    return result.to(logits.dtype)


def _smooth_phrase_relevance(binding_logits, phrase_mask, temperature,
                             bias=0.0):
    """Normalized smooth max over phrase slots for every temporal node."""
    # [B, P, M] -> [B, M, P]
    bsz, _, num_nodes = binding_logits.shape
    phrase_mask = phrase_mask.bool()
    expanded = phrase_mask.unsqueeze(1).expand(bsz, num_nodes, -1)
    smooth = masked_log_mean_exp(
        binding_logits.transpose(1, 2), expanded, temperature)
    return torch.sigmoid(smooth - bias)


class QueryPhraseGraphEncoder(nn.Module):
    """Pool token states into a fixed query phrase graph."""

    def __init__(self, hidden_size, num_phrase_types=6, num_edge_types=5,
                 num_heads=4, dropout=0.1):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_heads < 1 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.type_embedding = nn.Embedding(num_phrase_types, hidden_size)
        self.graph_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.graph_norm = nn.LayerNorm(hidden_size)

    def forward(self, query_states, query_mask, phrase_token_mask,
                phrase_type, phrase_valid, query_edge_type):
        if query_states.dim() != 3:
            raise ValueError("query_states must be [B,W,D]")
        bsz, num_words, hidden = query_states.shape
        if phrase_token_mask.shape[-1] != num_words:
            raise ValueError("phrase token mask does not match query states")
        if phrase_type.shape[:2] != phrase_valid.shape[:2]:
            raise ValueError("phrase type/valid shapes do not match")
        if phrase_token_mask.shape[:2] != phrase_valid.shape:
            raise ValueError("phrase token mask/valid shapes do not match")
        valid = phrase_valid.bool()
        token_mask = phrase_token_mask.bool() & query_mask.bool().unsqueeze(1)
        token_mask = token_mask & valid.unsqueeze(-1)

        token_values = query_states.unsqueeze(1)
        token_values = token_values.expand(-1, phrase_token_mask.size(1), -1,
                                           -1)
        pooled = _masked_mean(token_values, token_mask.unsqueeze(-1), 2)
        # ``_masked_mean`` above returns [B,P,D]; empty phrase slots are
        # explicitly zeroed before type embeddings are added.
        pooled = pooled * valid.unsqueeze(-1).to(pooled.dtype)
        phrase_features = pooled + self.type_embedding(
            phrase_type.long().clamp_min(0))
        phrase_features = phrase_features * valid.unsqueeze(-1).to(
            phrase_features.dtype)

        # Restrict attention to graph neighbours and always add a self edge
        # for valid phrases.  A 3-D mask is accepted by torch 1.13 when its
        # first dimension is batch * heads.
        phrase_count = phrase_valid.size(1)
        edge = query_edge_type.long() > 0
        edge = edge & valid.unsqueeze(1) & valid.unsqueeze(2)
        diagonal = torch.eye(
            phrase_count, device=query_states.device, dtype=torch.bool)
        edge = edge | (diagonal.unsqueeze(0) & valid.unsqueeze(-1))
        attn_mask = torch.zeros(
            bsz, phrase_count, phrase_count, device=query_states.device,
            dtype=phrase_features.dtype)
        attn_mask = attn_mask.masked_fill(~edge, float('-inf'))
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        key_padding_mask = ~valid
        updated, _ = self.graph_attention(
            phrase_features, phrase_features, phrase_features,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        phrase_features = self.graph_norm(phrase_features + updated)
        phrase_features = phrase_features * valid.unsqueeze(-1).to(
            phrase_features.dtype)

        # Relation words remain valid phrase nodes for binding and message
        # passing, but relation satisfaction is defined only on semantic
        # content phrases (action/entity/attribute) at both ends.
        phrase_type = phrase_type.long()
        content = valid & ((phrase_type == 2) | (phrase_type == 3)
                           | (phrase_type == 4))
        content_edge_mask = edge & content.unsqueeze(1) & content.unsqueeze(2)
        return {
            'phrase_features': phrase_features,
            'phrase_mask': valid,
            'content_edge_mask': content_edge_mask,
        }


class TemporalRelationGraphLayer(nn.Module):
    """Sparse temporal message passing over a fixed neighbour table."""

    def __init__(self, hidden_size, num_edge_types=4, num_heads=4,
                 dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)
        self.edge_projection = nn.Linear(hidden_size, num_heads, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ffn2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, node_mask, node_gate, neighbor_index,
                neighbor_type, neighbor_mask, relation_edge_bias=None):
        bsz, num_nodes, hidden = nodes.shape
        degree = neighbor_index.size(-1)
        index = neighbor_index.unsqueeze(-1).expand(
            bsz, num_nodes, degree, hidden)
        neighbour_nodes = nodes.unsqueeze(1).expand(
            bsz, num_nodes, num_nodes, hidden).gather(2, index)

        q = self.query(nodes).unsqueeze(2)
        k = self.key(neighbour_nodes)
        value = self.value(neighbour_nodes)
        logits = (q * k).sum(-1) / math.sqrt(float(hidden))
        edge_value = self.edge_embedding(neighbor_type.long())
        logits = logits + self.edge_projection(edge_value).mean(-1)
        if relation_edge_bias is not None:
            rel = relation_edge_bias.gather(2, neighbor_index)
            logits = logits + rel
        neighbour_gate = node_gate.unsqueeze(1).expand(
            bsz, num_nodes, num_nodes).gather(2, neighbor_index)
        logits = logits + torch.log(neighbour_gate.float().clamp_min(1e-6))
        logits = logits.float().masked_fill(~neighbor_mask.bool(), -1e4)
        logits = logits.masked_fill(~node_mask.bool().unsqueeze(-1), -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * neighbor_mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        message = (weights.to(value.dtype).unsqueeze(-1) * value).sum(2)
        message = self.output(message)
        gate = node_gate.clamp(0.0, 1.0).unsqueeze(-1).to(message.dtype)
        nodes = self.norm1(nodes + self.dropout(gate * message))
        feedforward = self.ffn2(F.gelu(self.ffn1(nodes)))
        nodes = self.norm2(nodes + self.dropout(feedforward))
        return nodes * node_mask.unsqueeze(-1).to(nodes.dtype), weights


class TemporalEvidenceGraphEncoder(nn.Module):
    """Construct and propagate the fixed-stride temporal evidence graph."""

    def __init__(self, hidden_size, node_stride=4, num_layers=2,
                 num_heads=4, max_temporal_hop=4, similar_topk=2,
                 similarity_threshold=0.5, dropout=0.1,
                 transition_threshold=0.2, transition_temperature=0.05):
        super().__init__()
        if node_stride < 1:
            raise ValueError("node_stride must be >= 1")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.node_stride = int(node_stride)
        self.max_temporal_hop = int(max_temporal_hop)
        self.similar_topk = int(similar_topk)
        self.similarity_threshold = float(similarity_threshold)
        self.transition_threshold = float(transition_threshold)
        self.transition_temperature = float(transition_temperature)
        self.node_fusion = nn.Linear(hidden_size * 3, hidden_size)
        _zero_init(self.node_fusion)
        self.node_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([
            TemporalRelationGraphLayer(
                hidden_size, num_edge_types=4, num_heads=num_heads,
                dropout=dropout)
            for _ in range(num_layers)
        ])

    def encode_nodes(self, visual_states, grounded_states, frame_mask):
        visual_nodes, node_mask, node_bounds, node_centers = temporal_pool(
            visual_states, frame_mask, self.node_stride)
        grounded_nodes, grounded_mask, _, _ = temporal_pool(
            grounded_states, frame_mask, self.node_stride)
        node_mask = node_mask & grounded_mask
        fusion_input = torch.cat([
            visual_nodes, grounded_nodes, visual_nodes * grounded_nodes,
        ], dim=-1)
        graph_nodes = self.node_norm(
            grounded_nodes + self.node_fusion(fusion_input))
        graph_nodes = graph_nodes * node_mask.unsqueeze(-1).to(
            graph_nodes.dtype)
        transition = compute_transition_score(
            visual_nodes, node_mask, self.transition_threshold,
            self.transition_temperature)
        neighbor_index, neighbor_type, neighbor_mask = \
            build_temporal_adjacency(
                visual_nodes, node_mask, self.max_temporal_hop,
                self.similar_topk, self.similarity_threshold)
        return {
            'visual_nodes': visual_nodes,
            'grounded_nodes': grounded_nodes,
            'graph_nodes_initial': graph_nodes,
            'node_bounds': node_bounds,
            'node_centers': node_centers,
            'node_mask': node_mask,
            'transition_score': transition,
            'neighbor_index': neighbor_index,
            'neighbor_type': neighbor_type,
            'neighbor_mask': neighbor_mask,
        }

    def propagate(self, node_state, node_mask, node_gate, neighbor_index,
                  neighbor_type, neighbor_mask, relation_edge_bias=None):
        state = node_state
        diagnostics = None
        for layer in self.layers:
            state, diagnostics = layer(
                state, node_mask, node_gate, neighbor_index, neighbor_type,
                neighbor_mask, relation_edge_bias=relation_edge_bias)
        if not self.layers:
            diagnostics = state.new_zeros(
                state.size(0), state.size(1), neighbor_index.size(-1))
        return state, diagnostics

    def forward(self, visual_states, grounded_states, frame_mask,
                phrase_features=None, phrase_valid=None, phrase_required=None,
                query_edge_type=None, node_gate=None,
                relation_edge_bias=None):
        state = self.encode_nodes(visual_states, grounded_states, frame_mask)
        if node_gate is None:
            node_gate = state['node_mask'].to(state['graph_nodes_initial'].dtype)
        graph_nodes, attention = self.propagate(
            state['graph_nodes_initial'], state['node_mask'], node_gate,
            state['neighbor_index'], state['neighbor_type'],
            state['neighbor_mask'], relation_edge_bias)
        state['graph_nodes'] = graph_nodes
        state['graph_attention'] = attention
        return state


class QSTGProposalQualityHead(nn.Module):
    """Learned proposal quality scorer; its initial output is exactly zero."""

    def __init__(self, input_size=10, hidden_size=64, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        _zero_init(self.network[-1])

    def forward(self, features):
        return self.network(features).squeeze(-1)


class QuerySubgraphTemporalGrounder(nn.Module):
    """QSTG's three-stage module used by :class:`models.cpl.CPL`."""

    def __init__(self, hidden_size, num_proposals, max_components,
                 component_counts=None, max_phrases=8, node_stride=4,
                 num_graph_layers=2, num_graph_heads=4, keep_ratio=0.5,
                 gate_floor=0.05, gate_temperature=0.1,
                 binding_temperature=0.1, final_binding_temperature=0.1,
                 relation_bias_scale=0.5, relation_decay=4.0,
                 simultaneous_decay=1.5, transition_threshold=0.2,
                 transition_temperature=0.05, proposal_prior_scale=0.0,
                 importance_bias_scale=0.0, component_residual_enabled=False,
                 quality_head_enabled=True, membership_threshold=0.1,
                 soft_box_temperature=0.02, connected_refine=False,
                 hard_barrier_threshold=0.65,
                 relation_connect_threshold=0.4,
                 coverage_keep_ratio=0.9, max_refine_shift=0.1,
                 min_refine_width=0.01, return_diagnostics=False,
                 similarity_threshold=0.5, max_temporal_hop=4,
                 similar_topk=2, graph_dropout=0.1, eps=1e-6, **kwargs):
        super().__init__()
        if max_phrases < 2:
            raise ValueError("max_phrases must be >= 2")
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        for name, value in {
                'binding_temperature': binding_temperature,
                'final_binding_temperature': final_binding_temperature,
                'gate_temperature': gate_temperature}.items():
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
        self.hidden_size = hidden_size
        self.num_proposals = int(num_proposals)
        self.max_components = int(max_components)
        self.max_phrases = int(max_phrases)
        self.keep_ratio = float(keep_ratio)
        self.gate_floor = float(gate_floor)
        self.gate_temperature = float(gate_temperature)
        self.binding_temperature = float(binding_temperature)
        self.final_binding_temperature = float(final_binding_temperature)
        self.relation_bias_scale = float(relation_bias_scale)
        self.relation_decay = float(relation_decay)
        self.simultaneous_decay = float(simultaneous_decay)
        self.proposal_prior_scale = float(proposal_prior_scale)
        self.importance_bias_scale = float(importance_bias_scale)
        self.component_residual_enabled = bool(component_residual_enabled)
        self.quality_head_enabled = bool(quality_head_enabled)
        self.membership_threshold = float(membership_threshold)
        self.soft_box_temperature = float(soft_box_temperature)
        self.connected_refine_enabled = bool(connected_refine)
        self.hard_barrier_threshold = float(hard_barrier_threshold)
        self.relation_connect_threshold = float(relation_connect_threshold)
        self.coverage_keep_ratio = float(coverage_keep_ratio)
        self.max_refine_shift = float(max_refine_shift)
        self.min_refine_width = float(min_refine_width)
        self.return_diagnostics = bool(return_diagnostics)
        self.eps = float(eps)
        self.component_counts = (
            list(component_counts) if component_counts is not None else None)

        self.query_encoder = QueryPhraseGraphEncoder(
            hidden_size, num_phrase_types=6, num_edge_types=5,
            num_heads=num_graph_heads, dropout=graph_dropout)
        self.temporal_encoder = TemporalEvidenceGraphEncoder(
            hidden_size, node_stride=node_stride, num_layers=num_graph_layers,
            num_heads=num_graph_heads, max_temporal_hop=max_temporal_hop,
            similar_topk=similar_topk,
            similarity_threshold=similarity_threshold,
            dropout=graph_dropout, transition_threshold=transition_threshold,
            transition_temperature=transition_temperature)
        self.pre_query_projection = nn.Linear(hidden_size, hidden_size)
        self.pre_visual_projection = nn.Linear(hidden_size, hidden_size)
        self.final_query_projection = nn.Linear(hidden_size, hidden_size)
        self.final_node_projection = nn.Linear(hidden_size, hidden_size)
        self.relevance_bias = nn.Parameter(torch.zeros(1))
        self.global_adapter = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        _zero_init(self.global_adapter[-1])

        self.quality_head = QSTGProposalQualityHead(
            input_size=10, hidden_size=64, dropout=graph_dropout)
        self.center_bias_heads = nn.ModuleList()
        self.width_bias_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(self.num_proposals)
        ])
        for head in self.width_bias_heads:
            _zero_init(head)
        self.single_bias_heads = nn.ModuleList([
            nn.Linear(hidden_size, 2) for _ in range(self.num_proposals)
        ])
        for head in self.single_bias_heads:
            _zero_init(head)
        if self.component_counts is not None:
            if len(self.component_counts) != self.num_proposals:
                raise ValueError("component_counts must match num_proposals")
            self.center_bias_heads = nn.ModuleList([
                nn.Linear(hidden_size, int(count))
                for count in self.component_counts
            ])
            for head in self.center_bias_heads:
                _zero_init(head)
        self.importance_bias_head = nn.Linear(hidden_size, 1)
        _zero_init(self.importance_bias_head)
        self.component_adapter = nn.Sequential(
            nn.Linear(hidden_size * 2 + max_phrases + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        _zero_init(self.component_adapter[-1])

    def _relation_affinity(self, binding_prob, content_edge_mask,
                           query_edge_type, node_centers):
        bsz, _, num_nodes = binding_prob.shape
        kernel = build_relation_kernel(
            node_centers, relation_decay=self.relation_decay,
            simultaneous_decay=self.simultaneous_decay)
        affinity = binding_prob.new_zeros(bsz, num_nodes, num_nodes)
        edge_count = binding_prob.new_zeros(bsz)
        batch_index = torch.arange(bsz, device=binding_prob.device)
        for source in range(binding_prob.size(1)):
            for target in range(binding_prob.size(1)):
                if source == target:
                    continue
                active = content_edge_mask[:, source, target]
                if not active.any():
                    continue
                outer = binding_prob[:, source, :].unsqueeze(-1) \
                    * binding_prob[:, target, :].unsqueeze(-2)
                typed_kernel = kernel[
                    batch_index, query_edge_type[:, source, target]]
                affinity = affinity + outer * typed_kernel \
                    * active.to(affinity.dtype).view(bsz, 1, 1)
                edge_count = edge_count + active.to(edge_count.dtype)
        affinity = affinity / edge_count.view(bsz, 1, 1).clamp_min(1.0)
        return affinity, edge_count > 0, kernel

    def _node_gate(self, binding_logits, phrase_mask, node_mask):
        relevance = _smooth_phrase_relevance(
            binding_logits, phrase_mask, self.gate_temperature,
            bias=self.relevance_bias)
        relevance = relevance * node_mask.to(relevance.dtype)
        valid_count = node_mask.sum(-1).clamp_min(1)
        keep_count = (valid_count.to(torch.float32) * self.keep_ratio).ceil()
        keep_count = keep_count.clamp_min(1).clamp_max(node_mask.size(-1))
        sorted_relevance = relevance.masked_fill(~node_mask.bool(), -1e4) \
            .sort(dim=-1, descending=True)[0]
        threshold_index = (keep_count.long() - 1).view(-1, 1)
        kappa = sorted_relevance.gather(1, threshold_index).detach()
        gate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(
            (relevance - kappa) / self.gate_temperature)
        gate = gate * node_mask.to(gate.dtype)
        return relevance, gate, kappa.squeeze(-1)

    def _pair_node_score(self, pair_binding_logits, phrase_required,
                         phrase_valid, node_mask):
        q_count, v_count, phrase_count, num_nodes = pair_binding_logits.shape
        required = _required_or_valid(phrase_valid, phrase_required)
        node_valid = node_mask.bool().unsqueeze(0).unsqueeze(2).expand(
            q_count, v_count, phrase_count, num_nodes)
        phrase_score = masked_log_mean_exp(
            pair_binding_logits.reshape(-1, num_nodes),
            node_valid.reshape(-1, num_nodes), self.binding_temperature)
        phrase_score = phrase_score.view(q_count, v_count, phrase_count)
        phrase_score = (phrase_score * required.to(phrase_score.dtype)
                        .unsqueeze(1)).sum(-1) / required.sum(-1).clamp_min(1)
        # Reverse direction: aggregate phrase evidence per node and average
        # only the strongest valid node fraction.
        phrase_mask = required.unsqueeze(1).unsqueeze(-1).expand(
            q_count, v_count, phrase_count, num_nodes)
        reverse = masked_log_mean_exp(
            pair_binding_logits.transpose(-1, -2).reshape(
                -1, phrase_count), phrase_mask.transpose(-1, -2).reshape(
                    -1, phrase_count), self.binding_temperature)
        reverse = reverse.view(q_count, v_count, num_nodes)
        reverse = reverse.masked_fill(
            ~node_mask.bool().unsqueeze(0), -1e4)
        keep = max(1, int(math.ceil(self.keep_ratio * num_nodes)))
        strongest = reverse.topk(min(keep, num_nodes), dim=-1).values
        return 0.5 * (phrase_score + strongest.mean(-1))

    def _global_summary(self, graph_nodes, node_relevance, transition,
                        phrase_features, phrase_type, binding_prob,
                        node_mask):
        dtype = graph_nodes.dtype
        relevance = node_relevance * node_mask.to(dtype)
        g_rel = (graph_nodes.float() * relevance.float().unsqueeze(-1)).sum(1) \
            / relevance.float().sum(1, keepdim=True).clamp_min(1e-6)
        g_rel = g_rel.to(dtype)
        type_ids = phrase_type.long()

        def phrase_summary(type_id):
            mask = type_ids == type_id
            weight = binding_prob * mask.unsqueeze(-1).to(dtype)
            weight = weight * node_mask.unsqueeze(1).to(dtype)
            summed = torch.einsum('bpm,bmd->bd', weight, graph_nodes)
            denominator = weight.sum(dim=(1, 2), keepdim=False).clamp_min(1e-6)
            return summed / denominator.unsqueeze(-1), mask.any(-1)

        g_act, has_action = phrase_summary(2)
        g_ent, has_entity = phrase_summary(3)
        g_act = torch.where(has_action.unsqueeze(-1), g_act, g_rel)
        g_ent = torch.where(has_entity.unsqueeze(-1), g_ent, g_rel)
        relevance_diff = (relevance[:, :-1] - relevance[:, 1:]).abs()
        boundary_weight = transition[:, :-1] * relevance_diff
        boundary_nodes = (graph_nodes[:, :-1].float()
                          + graph_nodes[:, 1:].float()) * 0.5
        g_boundary = (boundary_nodes
                      * boundary_weight.float().unsqueeze(-1)).sum(1) \
            / boundary_weight.float().sum(1, keepdim=True).clamp_min(1e-6)
        g_boundary = g_boundary.to(dtype)
        has_boundary = boundary_weight.sum(-1) > 1e-6
        g_boundary = torch.where(has_boundary.unsqueeze(-1), g_boundary, g_rel)
        summary = torch.cat([g_rel, g_act, g_ent, g_boundary], dim=-1)
        # Keep the adapter input compact: graph summary is projected to D.
        global_hint = summary.view(summary.size(0), 4, self.hidden_size).mean(1)
        return global_hint, {
            'global_relevance_summary': g_rel,
            'global_action_summary': g_act,
            'global_entity_summary': g_ent,
            'global_boundary_summary': g_boundary,
        }

    def _proposal_biases(self, graph_nodes, node_relevance, node_bounds,
                         node_mask):
        bsz, num_nodes, hidden = graph_nodes.shape
        sorted_indices = node_relevance.masked_fill(~node_mask.bool(), -1e4) \
            .argsort(dim=-1, descending=True)
        # Radius-2 temporal NMS supplies distinct proposal slots when the
        # relevance map contains several nearby peaks.  If there are fewer
        # separated peaks than proposals, fall back to the remaining highest
        # peak and record the duplicate explicitly.
        available = node_mask.bool().clone()
        selected = []
        duplicate = []
        positions = torch.arange(num_nodes, device=graph_nodes.device)
        for _ in range(self.num_proposals):
            available_scores = node_relevance.masked_fill(~available, -1e4)
            candidate = available_scores.argmax(-1)
            has_available = available.any(-1)
            fallback = sorted_indices[:, 0]
            candidate = torch.where(has_available, candidate, fallback)
            selected.append(candidate)
            duplicate.append(~has_available)
            suppress = (positions.view(1, -1) - candidate.unsqueeze(-1)).abs() <= 2
            available = available & ~suppress
        slot_index = torch.stack(selected, dim=1)
        slot_duplicate = torch.stack(duplicate, dim=1)
        slot_feature = graph_nodes.gather(
            1, slot_index.unsqueeze(-1).expand(-1, -1, hidden))
        slot_center = node_bounds[:, :, :].mean(-1).gather(1, slot_index)
        slot_width = node_bounds[:, :, 1].sub(node_bounds[:, :, 0]).gather(
            1, slot_index).clamp_min(2.0 / max(num_nodes, 1))
        center_bias = None
        if self.component_counts is not None:
            center_bias = torch.cat([
                self.center_bias_heads[index](slot_feature[:, index])
                for index in range(self.num_proposals)
            ], dim=-1) * self.proposal_prior_scale
        width_bias = torch.cat([
            self.width_bias_heads[index](slot_feature[:, index])
            for index in range(self.num_proposals)
        ], dim=-1) * self.proposal_prior_scale
        single_bias = torch.stack([
            self.single_bias_heads[index](slot_feature[:, index])
            for index in range(self.num_proposals)
        ], dim=1) * self.proposal_prior_scale
        return slot_index, slot_feature, slot_center, slot_width, center_bias, \
            width_bias, single_bias, slot_duplicate

    def forward_pre_proposal(self, global_feature, visual_states,
                             grounded_states, frame_mask, query_states,
                             query_mask, phrase_token_mask, phrase_type,
                             phrase_valid, phrase_required, query_edge_type,
                             video_group_id=None):
        phrase = self.query_encoder(
            query_states, query_mask, phrase_token_mask, phrase_type,
            phrase_valid, query_edge_type)
        temporal = self.temporal_encoder.encode_nodes(
            visual_states, grounded_states, frame_mask)
        phrase_features = phrase['phrase_features']
        valid_phrases = phrase['phrase_mask']
        required = _required_or_valid(phrase_valid, phrase_required)
        pre_query = F.normalize(
            self.pre_query_projection(phrase_features), dim=-1)
        pre_visual = F.normalize(
            self.pre_visual_projection(temporal['visual_nodes']), dim=-1)
        pair_binding_logits = torch.einsum(
            'bpd,vmd->bvpm', pre_query, pre_visual)
        pair_binding_logits = pair_binding_logits / self.binding_temperature
        pair_binding_logits = pair_binding_logits.masked_fill(
            ~valid_phrases.bool().unsqueeze(1).unsqueeze(-1), -1e4)
        pair_binding_logits = pair_binding_logits.masked_fill(
            ~temporal['node_mask'].bool().unsqueeze(0).unsqueeze(2), -1e4)
        batch_index = torch.arange(
            global_feature.size(0), device=global_feature.device)
        pre_binding_logits = pair_binding_logits[batch_index, batch_index]
        pre_binding_prob = torch.sigmoid(pre_binding_logits)
        pre_relevance, node_gate, gate_threshold = self._node_gate(
            pre_binding_logits, required, temporal['node_mask'])
        pre_relation, _, _ = self._relation_affinity(
            pre_binding_prob, phrase['content_edge_mask'], query_edge_type,
            temporal['node_centers'])
        node_gate = node_gate * temporal['node_mask'].to(node_gate.dtype)
        graph_nodes, graph_attention = self.temporal_encoder.propagate(
            temporal['graph_nodes_initial'], temporal['node_mask'], node_gate,
            temporal['neighbor_index'], temporal['neighbor_type'],
            temporal['neighbor_mask'],
            relation_edge_bias=self.relation_bias_scale * pre_relation)
        final_query = F.normalize(
            self.final_query_projection(phrase_features), dim=-1)
        final_node = F.normalize(
            self.final_node_projection(graph_nodes), dim=-1)
        binding_logits = torch.einsum(
            'bpd,bmd->bpm', final_query, final_node)
        binding_logits = binding_logits / self.final_binding_temperature
        binding_logits = binding_logits.masked_fill(
            ~valid_phrases.unsqueeze(-1), -1e4)
        binding_logits = binding_logits.masked_fill(
            ~temporal['node_mask'].unsqueeze(1), -1e4)
        binding_prob = torch.sigmoid(binding_logits)
        node_relevance = _smooth_phrase_relevance(
            binding_logits, required, self.gate_temperature,
            bias=self.relevance_bias)
        node_relevance = node_relevance * temporal['node_mask'].to(
            node_relevance.dtype)
        relation_affinity, relation_valid, relation_kernel = \
            self._relation_affinity(
                binding_prob, phrase['content_edge_mask'], query_edge_type,
                temporal['node_centers'])
        global_hint, global_parts = self._global_summary(
            graph_nodes, node_relevance, temporal['transition_score'],
            phrase_features, phrase_type, binding_prob, temporal['node_mask'])
        adapted_global = global_feature + self.global_adapter(torch.cat([
            global_feature, global_hint,
        ], dim=-1))
        (slot_index, slot_feature, slot_center, slot_width, center_bias,
         width_bias, single_bias, slot_duplicate) = self._proposal_biases(
             graph_nodes, node_relevance, temporal['node_bounds'],
             temporal['node_mask'])
        node_pair_score = self._pair_node_score(
            pair_binding_logits, phrase_required, phrase_valid,
            temporal['node_mask'])
        state = {
            **phrase,
            **temporal,
            'phrase_features': phrase_features,
            'phrase_valid': valid_phrases,
            'phrase_required': required,
            'query_edge_type': query_edge_type,
            'video_group_id': video_group_id,
            'graph_nodes': graph_nodes,
            'graph_attention': graph_attention,
            'pre_relation_edge_bias': pre_relation,
            'relation_edge_affinity': relation_affinity,
            'relation_valid': relation_valid,
            'relation_kernel': relation_kernel,
            'pair_binding_logits': pair_binding_logits,
            'pre_binding_logits': pre_binding_logits,
            'pre_binding_prob': pre_binding_prob,
            'binding_logits': binding_logits,
            'binding_prob': binding_prob,
            'pre_node_relevance': pre_relevance,
            'node_relevance': node_relevance,
            'node_gate': node_gate,
            'gate_threshold': gate_threshold,
            'global_hint': global_hint,
            'global_parts': global_parts,
            'adapted_global_feature': adapted_global,
            'slot_node_index': slot_index,
            'slot_feature': slot_feature,
            'slot_prior_center': slot_center,
            'slot_prior_width': slot_width,
            'slot_duplicate_mask': slot_duplicate,
            'center_logit_bias': center_bias,
            'width_logit_bias': width_bias,
            'single_logit_bias': single_bias,
            'node_pair_score': node_pair_score,
        }
        return state

    def enhance_components(self, component_summary, component_centers,
                            component_widths, qstg_state):
        if component_summary is None:
            return {
                'enhanced_component_summary': None,
                'importance_logit_bias': None,
            }
        node_centers = qstg_state['node_centers']
        node_mask = qstg_state['node_mask']
        membership = analytic_soft_box(
            component_centers, component_widths, node_centers,
            self.soft_box_temperature)
        membership = membership * node_mask.unsqueeze(1).to(membership.dtype)
        membership = membership / membership.amax(
            -1, keepdim=True).clamp_min(self.eps)
        phrase_logits = qstg_state['binding_logits']
        evidence = _masked_node_softmax(
            phrase_logits, node_mask, self.final_binding_temperature)
        evidence = evidence * qstg_state['phrase_valid'].unsqueeze(-1).to(
            evidence.dtype)
        phrase_score = torch.einsum('bkm,bpm->bkp', membership, evidence)
        assign_mask = _required_or_valid(
            qstg_state['phrase_valid'], qstg_state['phrase_required'])
        assign_logits = phrase_score.masked_fill(
            ~assign_mask.unsqueeze(1), -1e4)
        assignment = torch.softmax(assign_logits, dim=-1)
        assignment = assignment * assign_mask.unsqueeze(1).to(
            assignment.dtype)
        assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(
            self.eps)
        relevance = qstg_state['node_relevance']
        weighted_relevance = membership * relevance.unsqueeze(1)
        graph_summary = torch.einsum(
            'bkm,bmd->bkd', weighted_relevance,
            qstg_state['graph_nodes']) / weighted_relevance.sum(
                -1, keepdim=True).clamp_min(self.eps)
        exclusivity = weighted_relevance.sum(-1) / membership.sum(
            -1).clamp_min(self.eps)
        adapter_input = torch.cat([
            component_summary, graph_summary, assignment, exclusivity.unsqueeze(-1)
        ], dim=-1)
        delta = self.component_adapter(adapter_input)
        enhanced = component_summary + (
            delta if self.component_residual_enabled else delta * 0.0)
        importance_bias = self.importance_bias_head(enhanced).squeeze(-1)
        importance_bias = importance_bias * self.importance_bias_scale
        return {
            'component_node_membership': membership,
            'component_graph_summary': graph_summary,
            'component_phrase_score': phrase_score,
            'component_assignment': assignment,
            'component_exclusivity': exclusivity,
            'enhanced_component_summary': enhanced,
            'importance_logit_bias': importance_bias,
        }

    def _boundary_relevance(self, start, end, node_centers, relevance):
        left_weight = torch.exp(-(
            node_centers.unsqueeze(1) - start.unsqueeze(-1)).square() / 0.01)
        right_weight = torch.exp(-(
            node_centers.unsqueeze(1) - end.unsqueeze(-1)).square() / 0.01)
        weighted_relevance = relevance.unsqueeze(1)
        left = (left_weight * weighted_relevance).sum(-1) / left_weight.sum(
            -1).clamp_min(self.eps)
        right = (right_weight * weighted_relevance).sum(-1) / right_weight.sum(
            -1).clamp_min(self.eps)
        return left, right

    def _pair_quality(self, pair_binding_logits, proposal_membership,
                      proposal_soft_box, qstg_state):
        q_count, v_count, phrase_count, num_nodes = pair_binding_logits.shape
        valid_phrases = qstg_state['phrase_valid']
        required = _required_or_valid(valid_phrases,
                                      qstg_state['phrase_required'])
        node_mask = qstg_state['node_mask']
        pair_prob = torch.sigmoid(pair_binding_logits)
        pair_evidence = _masked_node_softmax(
            pair_binding_logits.reshape(q_count * v_count, phrase_count,
                                        num_nodes),
            node_mask.unsqueeze(0).expand(q_count, -1, -1).reshape(
                q_count * v_count, num_nodes),
            self.final_binding_temperature).view(
                q_count, v_count, phrase_count, num_nodes)
        pair_coverage_per_phrase = torch.einsum(
            'qvpm,vnm->qvnp', pair_evidence, proposal_membership)
        coverage = (pair_coverage_per_phrase * required[:, None, None, :]
                    .to(pair_coverage_per_phrase.dtype)).sum(-1) / \
            required.sum(-1).clamp_min(1).view(q_count, 1, 1)
        pair_relevance = _smooth_phrase_relevance(
            pair_binding_logits.reshape(q_count * v_count, phrase_count,
                                        num_nodes),
            required.unsqueeze(1).expand(q_count, v_count, phrase_count)
            .reshape(q_count * v_count, phrase_count),
            self.gate_temperature, bias=self.relevance_bias).view(
                q_count, v_count, num_nodes)
        inside = proposal_membership * (
            proposal_membership >= self.membership_threshold).to(
                proposal_membership.dtype)
        exclusivity = (inside.unsqueeze(0) * pair_relevance.unsqueeze(2)).sum(-1) \
            / inside.unsqueeze(0).sum(-1).clamp_min(self.eps)
        relation, relation_valid = compute_pair_relation_satisfaction(
            proposal_membership, pair_prob, qstg_state['content_edge_mask'],
            qstg_state['query_edge_type'], qstg_state['relation_kernel'],
            qstg_state['neighbor_index'], qstg_state['neighbor_mask'])
        barrier = qstg_state['qstg_barrier']
        pair_connectivity = compute_connectivity(
            proposal_soft_box, barrier).unsqueeze(0).expand(
                q_count, -1, -1)
        relation_valid_f = relation_valid.view(q_count, 1, 1).to(
            coverage.dtype)
        relation_for_score = relation * relation_valid_f + (1.0 - relation_valid_f)
        relation_valid_feature = relation_valid_f.expand_as(coverage)
        width = proposal_soft_box.new_zeros(v_count, proposal_membership.size(1))
        width = width + proposal_soft_box.max(-1).values * 0.0
        # The normalized width proxy is filled by the caller's soft box span;
        # its exact center/width is not needed by the pair contrastive loss.
        width = proposal_membership.sum(-1) / float(max(num_nodes, 1))
        disagreement = (proposal_membership - proposal_soft_box).abs().mean(-1)
        features = torch.stack([
            coverage, exclusivity, relation_for_score, relation_valid_feature,
            pair_connectivity, pair_relevance.unsqueeze(2)
            .expand(-1, -1, proposal_membership.size(1), -1).mean(-1),
            pair_relevance[..., :1].unsqueeze(2).expand(
                -1, -1, proposal_membership.size(1), -1).squeeze(-1),
            pair_relevance[..., -1:].unsqueeze(2).expand(
                -1, -1, proposal_membership.size(1), -1).squeeze(-1),
            width.unsqueeze(0).expand(q_count, -1, -1),
            disagreement.unsqueeze(0).expand(q_count, -1, -1),
        ], dim=-1)
        logits = self.quality_head(features) if self.quality_head_enabled \
            else features[..., :1].sum(-1) * 0.0
        return logits, features

    def score_proposals(self, gauss_weight, center, width, qstg_state):
        bsz = qstg_state['graph_nodes'].size(0)
        num_props = self.num_proposals
        if gauss_weight.dim() == 2:
            props_len = gauss_weight.size(-1)
            membership = resample_to_nodes(
                gauss_weight.view(bsz, num_props, props_len),
                qstg_state['node_centers'])
        else:
            membership = resample_to_nodes(
                gauss_weight, qstg_state['node_centers'])
        membership = membership * qstg_state['node_mask'].unsqueeze(1).to(
            membership.dtype)
        membership = membership / membership.amax(-1, keepdim=True).clamp_min(
            self.eps)
        center = center.view(bsz, num_props)
        width = width.view(bsz, num_props)
        soft_box = analytic_soft_box(
            center, width, qstg_state['node_centers'],
            self.soft_box_temperature)
        required = _required_or_valid(
            qstg_state['phrase_valid'], qstg_state['phrase_required'])
        evidence = _masked_node_softmax(
            qstg_state['binding_logits'], qstg_state['node_mask'],
            self.final_binding_temperature)
        evidence = evidence * qstg_state['phrase_valid'].unsqueeze(-1).to(
            evidence.dtype)
        coverage_per_phrase, coverage = probabilistic_coverage(
            membership, evidence, required)
        quality_membership = membership * (
            membership >= self.membership_threshold).to(membership.dtype)
        exclusivity = (quality_membership * qstg_state['node_relevance']
                       .unsqueeze(1)).sum(-1) / quality_membership.sum(
                           -1).clamp_min(self.eps)
        relation, relation_valid = compute_relation_satisfaction(
            membership, qstg_state['binding_prob'],
            qstg_state['content_edge_mask'], qstg_state['query_edge_type'],
            qstg_state['relation_kernel'], qstg_state['neighbor_index'],
            qstg_state['neighbor_mask'])
        relevance = qstg_state['node_relevance']
        rel_adj = qstg_state['relation_edge_affinity']
        transition = qstg_state['transition_score']
        barrier = build_barrier(
            transition, relevance, rel_adj, detach=True)
        qstg_state['qstg_barrier'] = barrier
        connectivity = compute_connectivity(soft_box, barrier)
        start = (center - width * 0.5).clamp(0, 1)
        end = (center + width * 0.5).clamp(0, 1)
        left_relevance, right_relevance = self._boundary_relevance(
            start, end, qstg_state['node_centers'], relevance)
        mean_inside = (membership * relevance.unsqueeze(1)).sum(-1) / \
            membership.sum(-1).clamp_min(self.eps)
        disagreement = (membership - soft_box).abs().mean(-1)
        relation_valid_f = relation_valid.to(membership.dtype).unsqueeze(-1)
        relation_weight = 0.20 * relation_valid_f
        no_relation_extra = 0.20 * (1.0 - relation_valid_f) * 0.5
        c_weight = 0.35 + no_relation_extra
        x_weight = 0.30 + no_relation_extra
        analytic = c_weight * coverage + x_weight * exclusivity \
            + relation_weight * relation + 0.15 * connectivity
        quality_features = torch.stack([
            coverage, exclusivity, relation, relation_valid_f.expand_as(coverage),
            connectivity, mean_inside, left_relevance, right_relevance,
            width.clamp(0, 1), disagreement,
        ], dim=-1)
        quality_logit = self.quality_head(quality_features) \
            if self.quality_head_enabled else analytic * 0.0
        pair_quality_logits, pair_quality_features = self._pair_quality(
            qstg_state['pair_binding_logits'], membership, soft_box,
            {**qstg_state, 'qstg_barrier': barrier})
        output = {
            'proposal_node_membership': membership,
            'proposal_soft_box': soft_box,
            'proposal_coverage_per_phrase': coverage_per_phrase,
            'proposal_coverage': coverage,
            'proposal_exclusivity': exclusivity,
            'proposal_relation': relation,
            'proposal_relation_valid': relation_valid,
            'proposal_connectivity': connectivity,
            'proposal_analytic_score': analytic,
            'proposal_quality_logit': quality_logit,
            'pair_quality_logits': pair_quality_logits,
            'pair_quality_features': pair_quality_features,
            'qstg_barrier': barrier,
        }
        if (not self.training) and self.connected_refine_enabled:
            refined = connected_refine(
                qstg_state['node_bounds'], qstg_state['node_mask'], membership,
                soft_box, evidence, required, relevance, barrier, rel_adj,
                qstg_state['binding_prob'], qstg_state['content_edge_mask'],
                qstg_state['query_edge_type'], qstg_state['relation_kernel'],
                qstg_state['neighbor_index'], qstg_state['neighbor_mask'],
                center, width, membership_threshold=self.membership_threshold,
                soft_box_threshold=0.5,
                hard_barrier_threshold=self.hard_barrier_threshold,
                relation_connect_threshold=self.relation_connect_threshold,
                coverage_keep_ratio=self.coverage_keep_ratio,
                max_refine_shift=self.max_refine_shift,
                min_refine_width=self.min_refine_width)
            output.update({
                'qstg_eval_center': refined['eval_center'],
                'qstg_eval_width': refined['eval_width'],
                'qstg_refine_mask': refined['refine_mask'],
                'qstg_refine_shift': refined['refine_shift'],
            })
        else:
            output.update({
                'qstg_eval_center': None,
                'qstg_eval_width': None,
                'qstg_refine_mask': None,
                'qstg_refine_shift': None,
            })
        return output
