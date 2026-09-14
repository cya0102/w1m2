"""Bidirectional Proposal-Set Equivalence (BPSE) scoring.

The module intentionally contains no dataset, loss, runner, or NumPy code so
that its geometric and semantic behavior can be tested on small CPU tensors.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_shape(name, value, rank):
    if value is None or value.dim() != rank:
        actual = None if value is None else tuple(value.shape)
        raise ValueError('{} must have rank {}, got {}'.format(
            name, rank, actual))


def _safe_normalize(value, dim=-1, eps=1e-6):
    return value / value.norm(dim=dim, keepdim=True).clamp_min(eps)


class QueryUnitPooler(nn.Module):
    """Pool query states according to sparse token-to-unit masks."""

    def forward(self, query_states, query_mask, unit_token_mask,
                unit_valid_mask):
        _check_shape('query_states', query_states, 3)
        _check_shape('query_mask', query_mask, 2)
        _check_shape('unit_token_mask', unit_token_mask, 3)
        _check_shape('unit_valid_mask', unit_valid_mask, 2)
        if query_states.size(0) != unit_token_mask.size(0):
            raise ValueError('query and unit batch sizes must match')
        if query_states.size(1) != unit_token_mask.size(2):
            raise ValueError('unit mask token axis must match query states')
        token_mask = unit_token_mask.to(query_states.dtype)
        token_mask = token_mask * query_mask[:, None, :].to(query_states.dtype)
        numerator = torch.bmm(token_mask, query_states)
        denominator = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        units = numerator / denominator
        return units * unit_valid_mask.unsqueeze(-1).to(units.dtype)


class SalientEventTokenizer(nn.Module):
    """Create one differentiable visual event token per fixed time bin."""

    def __init__(self, num_event_tokens=16, motion_weight=1.0,
                 detail_weight=0.5, saliency_floor=0.1,
                 pool_temperature=0.1, eps=1e-6):
        super().__init__()
        if num_event_tokens < 1:
            raise ValueError('num_event_tokens must be positive')
        if pool_temperature <= 0:
            raise ValueError('pool_temperature must be positive')
        self.num_event_tokens = int(num_event_tokens)
        self.motion_weight = float(motion_weight)
        self.detail_weight = float(detail_weight)
        self.saliency_floor = float(saliency_floor)
        self.pool_temperature = float(pool_temperature)
        self.eps = float(eps)

    def forward(self, visual_states, frame_mask):
        _check_shape('visual_states', visual_states, 3)
        _check_shape('frame_mask', frame_mask, 2)
        batch_size, time_steps, hidden_size = visual_states.shape
        if frame_mask.shape != (batch_size, time_steps):
            raise ValueError('frame_mask must match visual_states')
        device = visual_states.device
        dtype = visual_states.dtype
        valid_frames = frame_mask.bool()
        normalized = _safe_normalize(visual_states.float(), dim=-1)
        if time_steps:
            previous = torch.cat([normalized[:, :1], normalized[:, :-1]], dim=1)
            motion = (normalized - previous).norm(dim=-1)
            global_mean = (normalized * valid_frames.unsqueeze(-1)).sum(dim=1)
            global_mean = global_mean / valid_frames.sum(
                dim=1, keepdim=True).clamp_min(1).float()
            detail = (normalized - global_mean.unsqueeze(1)).norm(dim=-1)
            saliency = (self.motion_weight * motion +
                        self.detail_weight * detail + self.saliency_floor)
        else:
            saliency = visual_states.new_zeros(batch_size, 0).float()

        boundaries = torch.linspace(
            0, time_steps, self.num_event_tokens + 1, device=device).long()
        event_tokens = visual_states.new_zeros(
            batch_size, self.num_event_tokens, hidden_size)
        event_strength = visual_states.new_zeros(
            batch_size, self.num_event_tokens)
        pool_weights = visual_states.new_zeros(
            batch_size, self.num_event_tokens, time_steps)
        event_valid = torch.zeros(
            batch_size, self.num_event_tokens, dtype=torch.bool, device=device)

        for event_index in range(self.num_event_tokens):
            start = int(boundaries[event_index].item())
            end = int(boundaries[event_index + 1].item())
            if end <= start:
                continue
            local_valid = valid_frames[:, start:end]
            event_valid[:, event_index] = local_valid.any(dim=-1)
            logits = saliency[:, start:end] / self.pool_temperature
            logits = logits.masked_fill(~local_valid, -1e4)
            weights = torch.softmax(logits.float(), dim=-1)
            weights = weights * local_valid.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pool_weights[:, event_index, start:end] = weights.to(dtype)
            event_tokens[:, event_index] = torch.bmm(
                weights.to(dtype).unsqueeze(1), visual_states[:, start:end]
            ).squeeze(1)
            count = local_valid.sum(dim=-1).clamp_min(1).float()
            event_strength[:, event_index] = (
                saliency[:, start:end] * local_valid.float()).sum(dim=-1) / count

        event_tokens = event_tokens * event_valid.unsqueeze(-1).to(dtype)
        event_strength = event_strength * event_valid.to(dtype)
        return {
            'event_tokens': event_tokens,
            'event_strength': event_strength,
            'event_pool_weights': pool_weights,
            'event_valid_mask': event_valid,
        }


def build_soft_box_masks(spans, frame_mask, temperature=0.02):
    """Return differentiable frame membership for normalized [start, end]."""
    _check_shape('spans', spans, 3)
    _check_shape('frame_mask', frame_mask, 2)
    if spans.size(0) != frame_mask.size(0):
        raise ValueError('spans and frame_mask batch sizes must match')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    batch_size, _, time_steps = frame_mask.shape[0], spans.size(1), frame_mask.size(1)
    if time_steps == 0:
        return spans.new_zeros(batch_size, spans.size(1), 0)
    positions = torch.linspace(
        0, 1, time_steps, device=spans.device, dtype=spans.dtype)
    start = spans[..., 0].unsqueeze(-1)
    end = spans[..., 1].unsqueeze(-1)
    # Compute the sigmoid in fp32 for narrow intervals under autocast/fp16.
    logits_left = (positions - start).float() / float(temperature)
    logits_right = (end - positions).float() / float(temperature)
    membership = (torch.sigmoid(logits_left) *
                   torch.sigmoid(logits_right)).to(spans.dtype)
    membership = membership * frame_mask[:, None, :].to(spans.dtype)
    return membership.clamp(0, 1)


def build_proposal_groups(base_spans, offsets=(0.05,), min_width=0.02,
                          dedup_eps=1e-4):
    """Construct local perturbations used only as quality-head supervision."""
    _check_shape('base_spans', base_spans, 3)
    if base_spans.size(-1) != 2:
        raise ValueError('base_spans must have final dimension 2')
    variants = [base_spans]
    for offset in offsets:
        offset = float(offset)
        if offset <= 0:
            raise ValueError('perturbation offsets must be positive')
        start, end = base_spans[..., 0], base_spans[..., 1]
        variants.extend([
            torch.stack([(start + offset).clamp(0, 1), end], dim=-1),
            torch.stack([start, (end - offset).clamp(0, 1)], dim=-1),
            torch.stack([(start - offset).clamp(0, 1), end], dim=-1),
            torch.stack([start, (end + offset).clamp(0, 1)], dim=-1),
            torch.stack([(start + offset).clamp(0, 1),
                         (end - offset).clamp(0, 1)], dim=-1),
            torch.stack([(start - offset).clamp(0, 1),
                         (end + offset).clamp(0, 1)], dim=-1),
            torch.stack([(start - offset).clamp(0, 1),
                         (end - offset).clamp(0, 1)], dim=-1),
            torch.stack([(start + offset).clamp(0, 1),
                         (end + offset).clamp(0, 1)], dim=-1),
        ])
    groups = torch.stack(variants, dim=2)
    widths = groups[..., 1] - groups[..., 0]
    valid = widths >= float(min_width)
    for index in range(groups.size(2)):
        if index == 0:
            continue
        duplicate = (groups[:, :, :index] - groups[:, :, index:index + 1]).abs()
        duplicate = duplicate.amax(dim=-1).amin(dim=-1) <= float(dedup_eps)
        valid[:, :, index] &= ~duplicate
    return groups, valid


class BPSEScorer(nn.Module):
    """Compute completeness, purity, equivalence, and learned quality."""

    def __init__(self, hidden_size, max_query_units=12, num_event_tokens=16,
                 box_temperature=0.02, coverage_pool_temperature=0.1,
                 unit_match_temperature=0.1, shell_width=0.05,
                 quality_hidden_size=64, quality_dropout=0.1,
                 quality_detach_inputs=True, eps=1e-6,
                 motion_saliency_weight=1.0, detail_saliency_weight=0.5,
                 saliency_floor=0.1, event_pool_temperature=0.1,
                 min_perturb_width=0.02, span_loss_detach_evidence=True):
        super().__init__()
        if hidden_size < 1:
            raise ValueError('hidden_size must be positive')
        self.hidden_size = int(hidden_size)
        self.max_query_units = int(max_query_units)
        self.box_temperature = float(box_temperature)
        self.coverage_pool_temperature = float(coverage_pool_temperature)
        self.unit_match_temperature = float(unit_match_temperature)
        self.shell_width = float(shell_width)
        self.quality_detach_inputs = bool(quality_detach_inputs)
        self.eps = float(eps)
        self.min_perturb_width = float(min_perturb_width)
        self.span_loss_detach_evidence = bool(span_loss_detach_evidence)
        self.unit_pooler = QueryUnitPooler()
        self.event_tokenizer = SalientEventTokenizer(
            num_event_tokens=num_event_tokens,
            motion_weight=motion_saliency_weight,
            detail_weight=detail_saliency_weight,
            saliency_floor=saliency_floor,
            pool_temperature=event_pool_temperature,
            eps=eps)
        self.visual_projection = nn.Linear(hidden_size, hidden_size)
        self.grounded_projection = nn.Linear(hidden_size, hidden_size)
        self.unit_projection = nn.Linear(hidden_size, hidden_size)
        self.event_projection = nn.Linear(hidden_size, hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.grounded_norm = nn.LayerNorm(hidden_size)
        self.unit_norm = nn.LayerNorm(hidden_size)
        self.event_norm = nn.LayerNorm(hidden_size)
        self._init_identity_projection(self.visual_projection)
        self._init_identity_projection(self.grounded_projection)
        self._init_identity_projection(self.unit_projection)
        self._init_identity_projection(self.event_projection)
        self.quality_head = nn.Sequential(
            nn.Linear(6, quality_hidden_size),
            nn.GELU(),
            nn.Dropout(quality_dropout),
            nn.Linear(quality_hidden_size, 1),
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    @staticmethod
    def _init_identity_projection(layer):
        nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)

    def _coverage_evidence(self, grounded_states, unit_states, frame_mask,
                           unit_valid_mask):
        grounded = _safe_normalize(self.grounded_norm(
            self.grounded_projection(grounded_states)), dim=-1, eps=self.eps)
        units = _safe_normalize(self.unit_norm(
            self.unit_projection(unit_states)), dim=-1, eps=self.eps)
        evidence = ((torch.bmm(grounded, units.transpose(1, 2)) + 1.0) / 2.0)
        evidence = evidence.clamp(0, 1)
        evidence = evidence * frame_mask[:, :, None].to(evidence.dtype)
        evidence = evidence * unit_valid_mask[:, None, :].to(evidence.dtype)
        return evidence

    def _score_spans(self, spans, frame_mask, coverage_evidence,
                     event_pool_weights, event_strength, event_valid_mask,
                     event_explainability, unit_valid_mask, unit_weights,
                     return_unit_coverage=False):
        membership = build_soft_box_masks(
            spans, frame_mask, temperature=self.box_temperature)
        batch_size, num_spans, time_steps = membership.shape
        dtype = membership.dtype
        valid_frames = frame_mask.bool()
        log_membership = membership.float().clamp_min(self.eps).log()
        coverage_logits = (
            coverage_evidence.float().unsqueeze(1) + log_membership.unsqueeze(-1)
        ) / self.coverage_pool_temperature
        coverage_logits = coverage_logits.masked_fill(
            ~valid_frames[:, None, :, None], -1e4)
        alpha = torch.softmax(coverage_logits, dim=2)
        unit_coverage = (alpha * coverage_evidence.float().unsqueeze(1)).sum(dim=2)
        unit_coverage = unit_coverage.to(dtype)
        unit_coverage = unit_coverage * unit_valid_mask[:, None, :].to(dtype)
        weighted_units = unit_weights * unit_valid_mask.to(unit_weights.dtype)
        weighted_units = weighted_units / weighted_units.sum(
            dim=-1, keepdim=True).clamp_min(self.eps)
        completeness = (unit_coverage * weighted_units[:, None, :]).sum(dim=-1)
        has_units = unit_valid_mask.any(dim=-1)
        completeness = completeness * has_units[:, None].to(dtype)

        event_membership = torch.einsum(
            'bjt,bgt->bgj', event_pool_weights, membership)
        event_mass = (event_membership * event_strength[:, None, :].to(dtype) *
                      event_valid_mask[:, None, :].to(dtype)).sum(dim=-1)
        numerator = (event_membership * event_strength[:, None, :].to(dtype) *
                     event_explainability[:, None, :].to(dtype) *
                     event_valid_mask[:, None, :].to(dtype)).sum(dim=-1)
        purity = numerator / event_mass.clamp_min(self.eps)
        purity = purity * (event_mass > self.eps).to(dtype)
        purity = purity * has_units[:, None].to(dtype)
        completeness = completeness.clamp(0, 1)
        purity = purity.clamp(0, 1)
        equivalence = (2 * completeness * purity / (
            completeness + purity + self.eps)).clamp(0, 1)

        query_support = coverage_evidence.max(dim=-1).values
        query_support = query_support * frame_mask.to(query_support.dtype)
        inner_mass = membership.sum(dim=-1)
        inner_score = (membership * query_support[:, None, :]).sum(dim=-1)
        inner_score = inner_score / inner_mass.clamp_min(self.eps)
        expanded = spans.clone()
        expanded[..., 0] = (expanded[..., 0] - self.shell_width).clamp(0, 1)
        expanded[..., 1] = (expanded[..., 1] + self.shell_width).clamp(0, 1)
        expanded_membership = build_soft_box_masks(
            expanded, frame_mask, temperature=self.box_temperature)
        shell_membership = (expanded_membership - membership).clamp_min(0)
        shell_mass = shell_membership.sum(dim=-1)
        shell_score = (shell_membership * query_support[:, None, :]).sum(dim=-1)
        shell_score = shell_score / shell_mass.clamp_min(self.eps)
        shell_valid = shell_mass > self.eps
        contrast = torch.where(shell_valid, inner_score - shell_score,
                               torch.zeros_like(inner_score))
        if return_unit_coverage:
            return (completeness, purity, equivalence, contrast, event_mass,
                    unit_coverage)
        return completeness, purity, equivalence, contrast, event_mass

    def _event_explainability(self, event_tokens, unit_states,
                              unit_valid_mask):
        events = _safe_normalize(self.event_norm(
            self.event_projection(event_tokens)), dim=-1, eps=self.eps)
        units = _safe_normalize(self.unit_norm(
            self.unit_projection(unit_states)), dim=-1, eps=self.eps)
        evidence = ((torch.bmm(events, units.transpose(1, 2)) + 1.0) / 2.0)
        evidence = evidence.clamp(0, 1)
        evidence = evidence.masked_fill(~unit_valid_mask[:, None, :], 0)
        logits = evidence.float() / self.unit_match_temperature
        logits = logits.masked_fill(~unit_valid_mask[:, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        explainability = (weights * evidence.float()).sum(dim=-1)
        explainability = explainability * unit_valid_mask.any(
            dim=-1, keepdim=True).to(explainability.dtype)
        return evidence.to(event_tokens.dtype), explainability.to(event_tokens.dtype)

    def forward(self, visual_states, grounded_states, frame_mask,
                query_states, query_mask, unit_token_mask,
                unit_valid_mask, unit_weights, base_spans,
                build_perturbations=False, perturb_offsets=(0.05,)):
        _check_shape('visual_states', visual_states, 3)
        _check_shape('grounded_states', grounded_states, 3)
        _check_shape('query_states', query_states, 3)
        _check_shape('base_spans', base_spans, 3)
        if visual_states.shape != grounded_states.shape:
            raise ValueError('visual and grounded states must have same shape')
        if visual_states.size(0) != query_states.size(0):
            raise ValueError('state batch sizes must match')
        if base_spans.size(-1) != 2:
            raise ValueError('base_spans must have final dimension 2')
        if unit_token_mask.size(1) != self.max_query_units:
            raise ValueError('unit count {} does not match max_query_units {}'.format(
                unit_token_mask.size(1), self.max_query_units))
        unit_states = self.unit_pooler(
            query_states, query_mask, unit_token_mask, unit_valid_mask)
        coverage_evidence = self._coverage_evidence(
            grounded_states, unit_states, frame_mask, unit_valid_mask)
        event_data = self.event_tokenizer(visual_states, frame_mask)
        event_unit_evidence, event_explainability = self._event_explainability(
            event_data['event_tokens'], unit_states, unit_valid_mask)

        scored = self._score_spans(
            base_spans, frame_mask, coverage_evidence,
            event_data['event_pool_weights'], event_data['event_strength'],
            event_data['event_valid_mask'], event_explainability,
            unit_valid_mask, unit_weights, return_unit_coverage=True)
        completeness, purity, equivalence, contrast, event_mass, unit_coverage = scored
        width = (base_spans[..., 1] - base_spans[..., 0]).clamp_min(0)
        features = torch.stack([
            completeness, purity, equivalence, width, contrast,
            torch.log1p(event_mass.clamp_min(0)),
        ], dim=-1)
        quality_features = features.detach() if self.quality_detach_inputs else features
        quality_logits = self.quality_head(quality_features).squeeze(-1)
        quality_probs = torch.sigmoid(quality_logits)

        span_evidence = (coverage_evidence.detach()
                         if self.span_loss_detach_evidence else coverage_evidence)
        span_explainability = (event_explainability.detach()
                               if self.span_loss_detach_evidence
                               else event_explainability)
        span_event_strength = (event_data['event_strength'].detach()
                               if self.span_loss_detach_evidence
                               else event_data['event_strength'])
        span_event_pool_weights = (event_data['event_pool_weights'].detach()
                                   if self.span_loss_detach_evidence
                                   else event_data['event_pool_weights'])
        span_scored = self._score_spans(
            base_spans, frame_mask, span_evidence,
            span_event_pool_weights, span_event_strength,
            event_data['event_valid_mask'], span_explainability,
            unit_valid_mask, unit_weights)

        output = {
            'quality_logits': quality_logits,
            'quality_probs': quality_probs,
            'completeness': completeness,
            'purity': purity,
            'equivalence': equivalence,
            'inside_shell_contrast': contrast,
            'event_mass': event_mass,
            'unit_coverage': unit_coverage,
            'span_completeness': span_scored[0],
            'span_purity': span_scored[1],
            'span_equivalence': span_scored[2],
            'event_unit_evidence': event_unit_evidence,
            'event_explainability': event_explainability,
            'event_membership': torch.einsum(
                'bjt,bgt->bgj', event_data['event_pool_weights'],
                build_soft_box_masks(base_spans, frame_mask,
                                     self.box_temperature)),
            'event_tokens': event_data['event_tokens'],
            'event_strength': event_data['event_strength'],
            'event_pool_weights': event_data['event_pool_weights'],
            'event_valid_mask': event_data['event_valid_mask'],
        }
        if build_perturbations:
            group_spans, group_valid = build_proposal_groups(
                base_spans.detach(), offsets=perturb_offsets,
                min_width=self.min_perturb_width)
            batch_size, num_props, num_groups, _ = group_spans.shape
            flat_spans = group_spans.reshape(batch_size, num_props * num_groups, 2)
            group_scored = self._score_spans(
                flat_spans, frame_mask, coverage_evidence,
                event_data['event_pool_weights'], event_data['event_strength'],
                event_data['event_valid_mask'], event_explainability,
                unit_valid_mask, unit_weights)
            group_features = torch.stack([
                group_scored[0], group_scored[1], group_scored[2],
                (flat_spans[..., 1] - flat_spans[..., 0]).clamp_min(0),
                group_scored[3], torch.log1p(group_scored[4].clamp_min(0)),
            ], dim=-1)
            group_logits = self.quality_head(
                group_features.detach() if self.quality_detach_inputs
                else group_features).squeeze(-1)
            output.update({
                'group_quality_logits': group_logits.view(
                    batch_size, num_props, num_groups),
                'group_completeness': group_scored[0].view(
                    batch_size, num_props, num_groups),
                'group_purity': group_scored[1].view(
                    batch_size, num_props, num_groups),
                'group_equivalence': group_scored[2].view(
                    batch_size, num_props, num_groups),
                'group_event_mass': group_scored[4].view(
                    batch_size, num_props, num_groups),
                'group_valid_mask': group_valid,
                'group_spans': group_spans,
            })
        else:
            output.update({
                'group_quality_logits': None,
                'group_completeness': None,
                'group_purity': None,
                'group_equivalence': None,
                'group_event_mass': None,
                'group_valid_mask': None,
                'group_spans': None,
            })
        return output
