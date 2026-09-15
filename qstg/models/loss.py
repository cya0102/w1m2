import math

import torch
import torch.nn.functional as F
import pdb

from models.modules.qstg_ops import (
    build_same_video_negative_mask,
    masked_log_mean_exp,
)


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss(words_logit, words_id, words_mask, num_props, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    nll_loss = nll_loss.view(bsz, num_props)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask) 
        final_loss = final_loss + ref_nll_loss.mean()
        final_loss = final_loss / 2
    
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': min_nll_loss.mean().item(),
    }
    if ref_words_logit is not None:
        loss_dict.update({
            'ref_nll_loss': ref_nll_loss.mean().item(),
            })

    return final_loss, loss_dict


def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = ref_loss.mean()
    else:
        rank_loss = min_nll_loss.mean()
    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    loss = kwargs['alpha_1'] * rank_loss

    gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(num_props).unsqueeze(0).cuda() * kwargs["lambda"]
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    loss = loss + kwargs['alpha_2'] * div_loss.mean()

    return loss, {
        'ivc_loss': loss.item(),
        'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        'neg_active_fraction_1': (
            (neg_loss_1 > 0).float().mean().item()
            if neg_words_logit_1 is not None else 0.0),
        'neg_active_fraction_2': (
            (neg_loss_2 > 0).float().mean().item()
            if neg_words_logit_2 is not None else 0.0),
        'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
        'div_loss': div_loss.mean().item()
    }


def mixture_pull_push_loss(words_logit, num_props,
                           mixture_component_centers=None,
                           mixture_component_weights=None,
                           mixture_component_importance=None,
                           mixture_component_valid_mask=None,
                           gauss_weight=None, **kwargs):
    """PPS-style regularization for components inside mixture proposals.

    Pulling keeps the farthest components of one proposal close enough to
    describe a coherent event. Intra-pushing prevents those components from
    collapsing onto exactly the same temporal evidence. Inter-pushing keeps
    different proposals from converging to a single reconstruction winner.
    Their weights are configured separately so that coherence does not
    overwhelm diversity.
    """
    zero = words_logit.sum() * 0.0
    if mixture_component_centers is None:
        return zero, {
            'mixture_loss': 0.0,
            'mixture_pull_loss': 0.0,
            'mixture_intra_push_loss': 0.0,
            'mixture_inter_push_loss': 0.0,
            'mixture_component_spread': 0.0,
            'mixture_component_similarity': 0.0,
            'mixture_importance_entropy': 0.0,
        }

    valid = mixture_component_valid_mask.bool()
    valid_float = valid.to(mixture_component_centers.dtype)
    component_count = valid.sum(dim=-1)
    has_multiple = component_count > 1

    positive_infinity = torch.finfo(
        mixture_component_centers.dtype).max
    minimum_center = mixture_component_centers.masked_fill(
        ~valid, positive_infinity).min(dim=-1)[0]
    maximum_center = mixture_component_centers.masked_fill(
        ~valid, -positive_infinity).max(dim=-1)[0]
    spread = (maximum_center - minimum_center).clamp_min(0)
    pull_per_proposal = spread.square() * has_multiple.to(spread.dtype)
    # Match PPS: sum proposal-wise terms, then average over the batch.
    pull_loss = pull_per_proposal.sum(dim=-1).mean()

    normalized_components = mixture_component_weights / (
        mixture_component_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6))
    component_gram = torch.matmul(
        normalized_components, normalized_components.transpose(-1, -2))
    max_components = mixture_component_centers.size(-1)
    identity = torch.eye(
        max_components, device=component_gram.device,
        dtype=component_gram.dtype).view(1, 1, max_components, max_components)
    intra_target = kwargs.get('mixture_intra_push_target', 0.15) * identity
    valid_pairs = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    intra_error = (component_gram - intra_target).square()
    intra_error = intra_error * valid_pairs.to(intra_error.dtype)
    intra_error = intra_error.sum(dim=(-1, -2))
    intra_error = intra_error * has_multiple.to(intra_error.dtype)
    intra_push_loss = intra_error.sum(dim=-1).mean()

    batch_size = mixture_component_centers.size(0)
    mixture_masks = gauss_weight.view(batch_size, num_props, -1)
    normalized_mixtures = mixture_masks / mixture_masks.sum(
        dim=-1, keepdim=True).clamp_min(1e-6)
    mixture_gram = torch.matmul(
        normalized_mixtures, normalized_mixtures.transpose(1, 2))
    proposal_identity = torch.eye(
        num_props, device=mixture_gram.device,
        dtype=mixture_gram.dtype).unsqueeze(0)
    inter_target = (
        kwargs.get('mixture_inter_push_target', 0.15)
        * proposal_identity)
    inter_push_loss = (mixture_gram - inter_target).square().sum(
        dim=(-1, -2)).mean()

    l2_components = F.normalize(
        mixture_component_weights, p=2, dim=-1)
    cosine_gram = torch.matmul(
        l2_components, l2_components.transpose(-1, -2))
    off_diagonal = valid_pairs & ~torch.eye(
        max_components, device=valid.device,
        dtype=torch.bool).view(1, 1, max_components, max_components)
    component_similarity = (
        cosine_gram.masked_select(off_diagonal).mean()
        if off_diagonal.any() else zero)

    importance = mixture_component_importance.clamp_min(1e-8)
    entropy = -(importance * importance.log() * valid_float).sum(dim=-1)
    normalizer = component_count.clamp_min(2).to(entropy.dtype).log()
    normalized_entropy = entropy / normalizer
    importance_entropy = (
        normalized_entropy.masked_select(has_multiple).mean()
        if has_multiple.any() else zero)
    mean_spread = (
        spread.masked_select(has_multiple).mean()
        if has_multiple.any() else zero)

    loss = (
        kwargs.get('mixture_pull_weight', 0.0) * pull_loss
        + kwargs.get('mixture_intra_push_weight', 0.0) * intra_push_loss
        + kwargs.get('mixture_inter_push_weight', 0.0) * inter_push_loss)

    return loss, {
        'mixture_loss': loss.item(),
        'mixture_pull_loss': pull_loss.item(),
        'mixture_intra_push_loss': intra_push_loss.item(),
        'mixture_inter_push_loss': inter_push_loss.item(),
        'mixture_component_spread': mean_spread.item(),
        'mixture_component_similarity': component_similarity.item(),
        'mixture_importance_entropy': importance_entropy.item(),
    }


def event_disentanglement_loss(words_logit, words_id, words_mask, num_props,
                               event_pos_feat=None, event_neg_feat=None,
                               event_vector=None, event_text_feat=None,
                               event_selection_weights=None, center=None,
                               width=None, event_schedule=1.0,
                               event_positive_rank=0,
                               event_largest_eigenvalue=0.0,
                               event_smallest_selected_eigenvalue=0.0,
                               **kwargs):
    """Boundary-aware soft Event Vector loss (BECL).

    Besides text alignment and positive/background separation, BECL requires
    pseudo-positive proposals to leave enough observable context and penalizes
    pairwise temporal overlap. This blocks V1's shortcut in which all positive
    proposals expand toward the full video and squeeze negatives to endpoints.
    Reconstruction-derived soft weights replace the discontinuous hard argmin.
    """
    if event_vector is None:
        zero = words_logit.sum() * 0.0
        return zero, {'event_loss': 0.0, 'event_sep_loss': 0.0,
                      'event_text_loss': 0.0, 'event_context_loss': 0.0,
                      'event_overlap_loss': 0.0, 'event_schedule': 0.0}

    bsz = words_logit.size(0) // num_props
    positive = event_pos_feat.view(bsz, num_props, -1)
    negative = event_neg_feat.view(
        bsz, num_props, event_neg_feat.size(1), -1)
    event = event_vector.view(bsz, num_props, -1)
    selection = event_selection_weights / event_selection_weights.sum(
        dim=-1, keepdim=True).clamp_min(1e-6)

    event_direction = F.normalize(event, dim=-1)
    positive_score = F.cosine_similarity(positive, event_direction, dim=-1)
    negative_score = F.cosine_similarity(
        negative, event_direction.unsqueeze(2), dim=-1)
    margin = kwargs.get('event_margin', 0.2)
    separation_per_proposal = F.relu(
        margin - positive_score.unsqueeze(-1) + negative_score).mean(dim=-1)

    text_for_proposals = event_text_feat.unsqueeze(1).expand(
        bsz, num_props, -1)
    text_per_proposal = 1.0 - F.cosine_similarity(
        event, text_for_proposals, dim=-1)

    center = center.view(bsz, num_props)
    width = width.view(bsz, num_props)
    left_context = (center - width / 2).clamp_min(0)
    right_context = (1 - center - width / 2).clamp_min(0)
    available_context = left_context + right_context
    minimum_context = kwargs.get('event_min_context', 0.15)
    context_per_proposal = F.relu(minimum_context - available_context)

    # Proposals without enough context are unreliable event/background pairs.
    # Detaching reliability prevents a new shortcut through the loss weight;
    # the explicit context term remains responsible for boundary gradients.
    reliability = (available_context / max(minimum_context, 1e-6)).clamp(
        max=1.0).detach()
    reliable_selection = selection * reliability

    separation_loss = (reliable_selection * separation_per_proposal).sum(
        dim=-1).mean()
    text_loss = (reliable_selection * text_per_proposal).sum(dim=-1).mean()
    context_loss = (selection * context_per_proposal).sum(dim=-1).mean()

    if num_props > 1:
        proposal_start = (center - width / 2).clamp_min(0)
        proposal_end = (center + width / 2).clamp_max(1)
        intersection = (
            torch.minimum(proposal_end.unsqueeze(2), proposal_end.unsqueeze(1))
            - torch.maximum(proposal_start.unsqueeze(2), proposal_start.unsqueeze(1))
        ).clamp_min(0)
        union = (
            torch.maximum(proposal_end.unsqueeze(2), proposal_end.unsqueeze(1))
            - torch.minimum(proposal_start.unsqueeze(2), proposal_start.unsqueeze(1))
        ).clamp_min(1e-6)
        pairwise_iou = intersection / union
        off_diagonal = ~torch.eye(
            num_props, dtype=torch.bool,
            device=pairwise_iou.device).unsqueeze(0)
        maximum_overlap = kwargs.get('event_max_overlap', 0.70)
        overlap_loss = F.relu(
            pairwise_iou.masked_select(off_diagonal) - maximum_overlap).mean()
    else:
        overlap_loss = width.sum() * 0.0

    semantic_loss = (
        kwargs.get('event_sep_weight', 1.0) * separation_loss
        + kwargs.get('event_text_weight', 1.0) * text_loss)
    boundary_loss = (
        kwargs.get('event_context_weight', 1.0) * context_loss
        + kwargs.get('event_overlap_weight', 0.5) * overlap_loss)
    # Warm up only the semantic Event objective. Boundary anti-collapse terms
    # must be active from epoch 1; V2's first run multiplied all four terms by
    # zero and allowed widths to reach 0.98 before the Event branch started.
    loss = kwargs.get('event_alpha', 0.0) * (
        float(event_schedule) * semantic_loss + boundary_loss)

    selection_entropy = -(
        selection * selection.clamp_min(1e-8).log()).sum(dim=-1).mean()

    return loss, {
        'event_loss': loss.item(),
        'event_sep_loss': separation_loss.item(),
        'event_text_loss': text_loss.item(),
        'event_context_loss': context_loss.item(),
        'event_overlap_loss': overlap_loss.item(),
        'event_semantic_loss': semantic_loss.item(),
        'event_boundary_loss': boundary_loss.item(),
        'event_selection_entropy': selection_entropy.item(),
        'event_mean_width': width.mean().item(),
        'event_context_violation': (
            available_context < minimum_context).float().mean().item(),
        'event_schedule': float(event_schedule),
        'event_positive_rank': float(event_positive_rank),
        'event_largest_eigenvalue': float(event_largest_eigenvalue),
        'event_smallest_selected_eigenvalue': float(
            event_smallest_selected_eigenvalue),
    }


def _qstg_metric(value, default=0.0):
    """Convert a scalar tensor to a detached Python float for logging."""
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        return float(value.detach().mean().item())
    return float(value)


def _qstg_symmetric_infonce(scores, valid_mask=None):
    """Symmetric diagonal InfoNCE with safe no-negative handling."""
    batch = scores.size(0)
    if batch < 2:
        return scores.sum() * 0.0, scores.new_zeros(()), 0
    if valid_mask is None:
        valid_mask = torch.ones(
            batch, batch, dtype=torch.bool, device=scores.device)
    positive = torch.arange(batch, device=scores.device)
    valid_negative = valid_mask.clone()
    valid_negative[positive, positive] = False
    row_has_negative = valid_negative.any(-1)
    col_has_negative = valid_negative.any(0)

    def one_direction(logits, active):
        safe = logits.masked_fill(~valid_mask, -1e4)
        value = -safe[positive, positive] + torch.logsumexp(safe, dim=-1)
        value = value.masked_fill(~active, 0.0)
        return value.sum() / active.sum().clamp_min(1)

    row_loss = one_direction(scores, row_has_negative)
    # Transpose changes the positive direction but the same symmetric mask is
    # valid because same-video exclusion is symmetric.
    col_loss = one_direction(scores.transpose(0, 1), col_has_negative)
    active = row_has_negative.to(scores.dtype).sum() \
        + col_has_negative.to(scores.dtype).sum()
    return 0.5 * (row_loss + col_loss), active, int(active.detach().item())


def _qstg_pair_node_scores(pair_binding_logits, phrase_required,
                           phrase_valid, temperature=0.1, keep_ratio=0.5):
    """Build query/video node matching scores from pair binding logits."""
    q_count, v_count, phrase_count, num_nodes = pair_binding_logits.shape
    required = phrase_valid.bool() & phrase_required.bool()
    has_required = required.any(-1, keepdim=True)
    required = torch.where(has_required, required, phrase_valid.bool())
    node_mask = pair_binding_logits > -5e3
    node_mask = node_mask.any(2)
    phrase_mask = node_mask.unsqueeze(2).expand(
        q_count, v_count, phrase_count, num_nodes)
    flat = pair_binding_logits.reshape(-1, num_nodes)
    flat_mask = phrase_mask.reshape(-1, num_nodes)
    scaled = (flat.float() / max(temperature, 1e-6)).masked_fill(
        ~flat_mask, -1e4)
    smooth = temperature * (
        torch.logsumexp(scaled, dim=-1)
        - flat_mask.sum(-1).clamp_min(1).float().log())
    phrase_scores = smooth.view(q_count, v_count, phrase_count)
    phrase_scores = (phrase_scores * required.unsqueeze(1).to(
        phrase_scores.dtype)).sum(-1) / required.sum(-1).clamp_min(1).view(
            q_count, 1)

    # The reverse direction is a smooth max over query phrases, averaged over
    # the strongest valid node fraction.
    flat_reverse = pair_binding_logits.transpose(-1, -2).reshape(
        -1, phrase_count)
    phrase_mask_reverse = required.unsqueeze(1).unsqueeze(-1).expand(
        q_count, v_count, phrase_count, num_nodes).transpose(-1, -2).reshape(
            -1, phrase_count)
    scaled_reverse = (flat_reverse.float() / max(temperature, 1e-6)).masked_fill(
        ~phrase_mask_reverse, -1e4)
    reverse = temperature * (
        torch.logsumexp(scaled_reverse, dim=-1)
        - phrase_mask_reverse.sum(-1).clamp_min(1).float().log())
    reverse = reverse.view(q_count, v_count, num_nodes)
    reverse = reverse.masked_fill(~node_mask, -1e4)
    keep = max(1, min(num_nodes, int(math.ceil(keep_ratio * num_nodes))))
    return 0.5 * (phrase_scores + reverse.topk(keep, -1).values.mean(-1))


def qstg_loss(
        words_logit, num_props, proposal_reconstruction_nll=None,
        pair_binding_logits=None, node_pair_score=None,
        pair_quality_logits=None, proposal_node_membership=None,
        binding_logits=None, binding_prob=None, phrase_required=None,
        phrase_valid=None, video_group_id=None, node_gate=None,
        transition_score=None, proposal_connectivity=None,
        component_assignment=None, mixture_component_valid_mask=None,
        **kwargs):
    """Weakly supervised QSTG objective and diagnostics.

    No timestamp or IoU target is accepted here.  The reconstruction selector
    is detached by default and all missing branches return a connected zero,
    so this function can remain in the runner for baseline-compatible runs.
    """
    zero = words_logit.sum() * 0.0
    names = [
        'qstg_loss', 'qstg_mc_loss', 'qstg_proposal_loss', 'qstg_fa_loss',
        'qstg_explain_loss', 'qstg_gate_loss', 'qstg_connect_loss',
        'qstg_role_loss', 'qstg_valid_negative_ratio',
        'qstg_selector_entropy', 'qstg_selector_confident_ratio',
        'qstg_gate_mean', 'qstg_gate_active_ratio', 'qstg_binding_mean',
        'qstg_quality_mean', 'qstg_coverage_mean', 'qstg_exclusivity_mean',
        'qstg_connectivity_mean',
    ]
    metrics = {name: 0.0 for name in names}
    if all(value is None for value in (
            pair_binding_logits, node_pair_score, pair_quality_logits,
            proposal_node_membership, binding_logits, node_gate,
            proposal_connectivity, component_assignment)):
        return zero, metrics

    eps = float(kwargs.get('qstg_eps', 1e-6))
    mc_temperature = float(kwargs.get('qstg_mc_temperature', 0.07))
    rec_temperature = float(
        kwargs.get('qstg_reconstruction_temperature', 0.1))
    fa_temperature = float(kwargs.get('qstg_fa_temperature', 0.1))
    fa_margin = float(kwargs.get('qstg_fa_margin', 0.2))
    keep_ratio = float(kwargs.get('qstg_keep_ratio', 0.5))
    if mc_temperature <= 0 or rec_temperature <= 0 or fa_temperature <= 0:
        raise ValueError('QSTG loss temperatures must be positive')

    same_video_mask = build_same_video_negative_mask(video_group_id)
    mc_loss = zero
    mc_active = 0
    if pair_binding_logits is not None and phrase_required is not None \
            and phrase_valid is not None:
        node_scores = _qstg_pair_node_scores(
            pair_binding_logits, phrase_required, phrase_valid,
            temperature=mc_temperature, keep_ratio=keep_ratio)
        mc_loss, active_count, mc_active = _qstg_symmetric_infonce(
            node_scores / mc_temperature, same_video_mask)
        off_diagonal = torch.ones_like(node_scores, dtype=torch.bool)
        off_diagonal.fill_diagonal_(False)
        if same_video_mask is None:
            usable_negative = off_diagonal
        else:
            usable_negative = off_diagonal & same_video_mask
        denominator = off_diagonal.sum().clamp_min(1)
        metrics['qstg_valid_negative_ratio'] = float(
            (usable_negative.sum().float() / denominator).detach().item())

    selector = None
    selector_entropy = zero
    confident_ratio = zero
    if proposal_reconstruction_nll is not None:
        nll = proposal_reconstruction_nll
        selector = torch.softmax(-nll / rec_temperature, dim=-1)
        selector_entropy = -(selector * selector.clamp_min(eps).log()).sum(-1).mean()
        if selector.size(-1) > 1:
            ordered = nll.sort(dim=-1).values
            margin = ordered[:, 1] - ordered[:, 0]
            confidence_margin = float(
                kwargs.get('qstg_selector_confidence_margin', 0.05))
            confident = margin >= confidence_margin
            confident_ratio = confident.float().mean()
        else:
            confident = torch.ones(
                nll.size(0), dtype=torch.bool, device=nll.device)
            confident_ratio = confident.float().mean()
        if kwargs.get('qstg_detach_selector', True):
            selector = selector.detach()

    proposal_loss = zero
    if pair_quality_logits is not None:
        pair_quality = pair_quality_logits
        if selector is not None:
            # pair_quality is [query, video, proposal].  The reconstruction
            # selector belongs to the video sample whose proposal is being
            # scored, so it weights the video/proposal axes as [1, video,
            # proposal] before reducing to [query, video].
            pair_quality = (
                pair_quality * selector.unsqueeze(0)).sum(-1)
        else:
            pair_quality = pair_quality.mean(-1)
        proposal_loss, _, _ = _qstg_symmetric_infonce(
            pair_quality / mc_temperature, same_video_mask)
    elif node_pair_score is not None:
        proposal_loss, _, _ = _qstg_symmetric_infonce(
            node_pair_score / mc_temperature, same_video_mask)

    fa_loss = zero
    explain_loss = zero
    coverage = None
    exclusivity = None
    if proposal_node_membership is not None and binding_logits is not None \
            and phrase_required is not None and phrase_valid is not None:
        membership = proposal_node_membership.clamp(0, 1)
        prob = torch.sigmoid(binding_logits) if binding_prob is None \
            else binding_prob
        required = phrase_valid.bool() & phrase_required.bool()
        has_required = required.any(-1, keepdim=True)
        required = torch.where(has_required, required, phrase_valid.bool())
        req_float = required.to(membership.dtype)
        # Selector confidence is intentionally a detached reliability weight.
        if selector is None:
            selector = membership.new_full(
                (membership.size(0), membership.size(1)),
                1.0 / max(membership.size(1), 1))
            confident = torch.ones(
                membership.size(0), dtype=torch.bool,
                device=membership.device)
        else:
            confident = confident.to(membership.device)
        sample_weight = torch.where(
            confident, torch.ones_like(confident, dtype=membership.dtype),
            torch.full_like(confident, 0.25, dtype=membership.dtype))
        sample_weight = sample_weight.unsqueeze(-1).unsqueeze(-1)

        inside = (binding_logits.unsqueeze(1).float()
                  + membership.unsqueeze(2).float().clamp_min(eps).log())
        outside = (binding_logits.unsqueeze(1).float()
                   + (1.0 - membership).unsqueeze(2).clamp_min(eps).log())
        score_in = fa_temperature * torch.logsumexp(
            inside / fa_temperature, dim=-1)
        score_out = fa_temperature * torch.logsumexp(
            outside / fa_temperature, dim=-1)
        ranking = F.softplus(
            (fa_margin - score_in + score_out) / fa_temperature)
        ranking_weight = selector.unsqueeze(-1) * req_float.unsqueeze(1) \
            * sample_weight
        fa_loss = (ranking * ranking_weight).sum() / \
            ranking_weight.sum().clamp_min(eps)

        # Required phrase smooth max for every node; this uses probabilities
        # for a bounded and interpretable reverse-explainability penalty.
        explainability = torch.sigmoid(masked_log_mean_exp(
            binding_logits.transpose(1, 2),
            required.unsqueeze(1).expand_as(
                binding_logits.transpose(1, 2)),
            fa_temperature)).to(membership.dtype)
        explain_loss = ((membership * (1.0 - explainability.unsqueeze(1)))
                        .sum(-1) / membership.sum(-1).clamp_min(eps)
                        * selector).mean()
        evidence = torch.softmax(binding_logits.float(), dim=-1)
        evidence = evidence * torch.ones_like(evidence)
        coverage = torch.einsum('bnm,bpm->bnp', membership, evidence)
        coverage = (coverage * req_float.unsqueeze(1)).sum(-1) / \
            req_float.sum(-1).clamp_min(1).unsqueeze(-1)
        exclusivity = (membership * (membership >= 0.1).to(
            membership.dtype) * explainability.unsqueeze(1)).sum(-1) / \
            (membership * (membership >= 0.1).to(membership.dtype)).sum(
                -1).clamp_min(eps)

    gate_loss = zero
    if node_gate is not None:
        if binding_logits is not None:
            node_valid = (binding_logits > -5e3).any(dim=1)
        else:
            node_valid = torch.ones_like(node_gate, dtype=torch.bool)
        valid_gate = node_valid.to(node_gate.dtype)
        gate_mean = (node_gate.float() * valid_gate.float()).sum() / \
            valid_gate.float().sum().clamp_min(eps)
        gate_min = float(kwargs.get('qstg_gate_min_ratio', 0.1))
        gate_max = float(kwargs.get('qstg_gate_max_ratio', 0.7))
        budget = F.relu(gate_min - gate_mean) + F.relu(gate_mean - gate_max)
        smooth = zero
        if node_gate.size(-1) > 1:
            transition = (transition_score if transition_score is not None
                          else torch.zeros_like(node_gate))
            if kwargs.get('qstg_detach_barrier', True):
                transition = transition.detach()
            smooth_mask = valid_gate[:, :-1] * valid_gate[:, 1:]
            smooth = (((1.0 - transition[:, :-1]).clamp_min(0.0)
                       * (node_gate[:, :-1] - node_gate[:, 1:]).abs()
                       * smooth_mask).sum()
                      / smooth_mask.sum().clamp_min(eps))
        gate_loss = budget + 0.5 * smooth
        metrics['qstg_gate_mean'] = _qstg_metric(gate_mean)
        metrics['qstg_gate_active_ratio'] = _qstg_metric(
            (node_gate > 0.1).float())

    connect_loss = zero
    if proposal_connectivity is not None:
        if selector is None:
            connect_loss = (1.0 - proposal_connectivity).mean()
        else:
            connect_loss = ((1.0 - proposal_connectivity) * selector).mean()

    role_loss = zero
    if component_assignment is not None and \
            mixture_component_valid_mask is not None:
        assignment = component_assignment
        valid_components = mixture_component_valid_mask.bool()
        role_terms = []
        cursor = 0
        if assignment.dim() == 4:
            # [B, N, C, P]
            for proposal in range(valid_components.size(1)):
                current = assignment[:, proposal]
                valid = valid_components[:, proposal]
                if current.size(1) < 2:
                    continue
                cosine = F.normalize(current, dim=-1).matmul(
                    F.normalize(current, dim=-1).transpose(-1, -2))
                off_diag = ~torch.eye(
                    current.size(1), dtype=torch.bool,
                    device=current.device).unsqueeze(0)
                pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2) & off_diag
                if pair_valid.any():
                    role_terms.append(cosine.masked_select(pair_valid).mean())
        else:
            # Flat component order is the generator's proposal-major order.
            for proposal in range(valid_components.size(1)):
                count = int(valid_components[:, proposal].sum(-1).max().item())
                current = assignment[:, cursor:cursor + count]
                cursor += count
                if current.size(1) < 2:
                    continue
                cosine = F.normalize(current, dim=-1).matmul(
                    F.normalize(current, dim=-1).transpose(-1, -2))
                off_diag = ~torch.eye(
                    current.size(1), dtype=torch.bool,
                    device=current.device).unsqueeze(0)
                role_terms.append(cosine.masked_select(off_diag).mean())
        if role_terms:
            role_loss = torch.stack(role_terms).mean()

    # Stage B/C start proposal-facing weak losses only after a short warmup.
    # Stage A intentionally remains mc/pq-only; an explicit qstg_ramp_factor
    # is accepted for controlled ablations and test-time auditing.
    ramp_factor = float(kwargs.get('qstg_ramp_factor', 1.0))
    stage = str(kwargs.get('qstg_stage', '')).upper()
    ramp_epoch = kwargs.get('qstg_epoch')
    if (ramp_epoch is not None and stage in {'B', 'C'}
            and kwargs.get('qstg_ramp_enabled', True)):
        warmup = int(kwargs.get('qstg_ramp_warmup', 1))
        ramp_epochs = int(kwargs.get('qstg_ramp_epochs', 3))
        if ramp_epochs <= 0:
            ramp_factor = 1.0 if int(ramp_epoch) > warmup else 0.0
        else:
            ramp_factor = min(max(
                (int(ramp_epoch) - warmup) / float(ramp_epochs), 0.0), 1.0)
    ramp_factor = min(max(ramp_factor, 0.0), 1.0)
    configured_weights = {
        'fa': float(kwargs.get('qstg_fa_weight', 0.1)),
        'explain': float(kwargs.get('qstg_explain_weight', 0.03)),
        'connect': float(kwargs.get('qstg_connect_weight', 0.05)),
        'role': float(kwargs.get('qstg_role_weight', 0.02)),
    }
    effective_weights = {
        name: value * ramp_factor
        for name, value in configured_weights.items()
    }

    weighted_loss = (
        float(kwargs.get('qstg_total_weight', 1.0)) * (
            float(kwargs.get('qstg_mc_weight', 0.05)) * mc_loss
            + float(kwargs.get('qstg_proposal_weight', 0.05)) * proposal_loss
            + effective_weights['fa'] * fa_loss
            + effective_weights['explain'] * explain_loss
            + float(kwargs.get('qstg_gate_weight', 0.01)) * gate_loss
            + effective_weights['connect'] * connect_loss
            + effective_weights['role'] * role_loss))
    metrics.update({
        'qstg_loss': _qstg_metric(weighted_loss),
        'qstg_mc_loss': _qstg_metric(mc_loss),
        'qstg_proposal_loss': _qstg_metric(proposal_loss),
        'qstg_fa_loss': _qstg_metric(fa_loss),
        'qstg_explain_loss': _qstg_metric(explain_loss),
        'qstg_gate_loss': _qstg_metric(gate_loss),
        'qstg_connect_loss': _qstg_metric(connect_loss),
        'qstg_role_loss': _qstg_metric(role_loss),
        'qstg_selector_entropy': _qstg_metric(selector_entropy),
        'qstg_selector_confident_ratio': _qstg_metric(confident_ratio),
        'qstg_binding_mean': _qstg_metric(
            torch.sigmoid(binding_logits) if binding_logits is not None else None),
        'qstg_quality_mean': _qstg_metric(
            pair_quality_logits if pair_quality_logits is not None else node_pair_score),
        'qstg_coverage_mean': _qstg_metric(coverage),
        'qstg_exclusivity_mean': _qstg_metric(exclusivity),
        'qstg_connectivity_mean': _qstg_metric(proposal_connectivity),
        'qstg_ramp_factor': float(ramp_factor),
        'qstg_configured_fa_weight': configured_weights['fa'],
        'qstg_configured_explain_weight': configured_weights['explain'],
        'qstg_configured_connect_weight': configured_weights['connect'],
        'qstg_configured_role_weight': configured_weights['role'],
        'qstg_effective_fa_weight': effective_weights['fa'],
        'qstg_effective_explain_weight': effective_weights['explain'],
        'qstg_effective_connect_weight': effective_weights['connect'],
        'qstg_effective_role_weight': effective_weights['role'],
    })
    return weighted_loss, metrics
