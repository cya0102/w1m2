import torch
import torch.nn.functional as F
import pdb


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
