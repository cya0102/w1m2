"""Stateless tensor operations for the QSTG temporal evidence graph.

Every function here is a pure tensor computation: no parameters, no
``.cuda()`` calls, no NumPy.  Each one is unit-testable on CPU float32 and
behaves identically on CUDA float32 (proposal section 12.4).
"""

import math

import torch
import torch.nn.functional as F

# Neighbour edge types inside the temporal evidence graph.
TEMPORAL_EDGE_SELF = 0
TEMPORAL_EDGE_NEXT = 1
TEMPORAL_EDGE_PREV = 2
TEMPORAL_EDGE_SIMILAR = 3

# Query edge types (mirrors datasets.base; duplicated as plain ints so this
# module stays dependency-free).
EDGE_ASSOC = 1
EDGE_BEFORE = 2
EDGE_AFTER = 3
EDGE_WHILE = 4


def temporal_pool(states, frame_mask, node_stride, eps=1e-6):
    """Pool per-frame states into fixed contiguous temporal nodes.

    Args:
        states: ``[B, T, D]`` per-frame states.
        frame_mask: ``[B, T]`` bool/byte validity (prefix mask).
        node_stride: frames per node; node ``m`` covers frames
            ``[m*stride, min((m+1)*stride, valid_len))`` (proposal 4.1).

    Returns:
        ``(pooled [B,M,D], node_mask [B,M], node_bounds [B,M,2],
        node_centers [B,M])`` with bounds normalized to ``[0, 1]`` by the
        per-sample valid length.  Invalid nodes are zeroed.
    """
    if node_stride < 1:
        raise ValueError('node_stride must be >= 1')
    bsz, num_frames, dim = states.shape
    num_nodes = int(math.ceil(num_frames / node_stride))
    pad = num_nodes * node_stride - num_frames
    mask = frame_mask.to(states.dtype)
    if pad:
        states = F.pad(states, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad))
    mask = mask.view(bsz, num_nodes, node_stride)
    denom = mask.sum(-1).clamp_min(eps)
    pooled = (states.view(bsz, num_nodes, node_stride, dim)
              * mask.unsqueeze(-1)).sum(2) / denom.unsqueeze(-1)
    node_mask = mask.sum(-1) > 0
    pooled = pooled * node_mask.unsqueeze(-1).to(pooled.dtype)

    valid_len = frame_mask.sum(-1).clamp_min(1).to(states.dtype)
    starts = torch.arange(
        num_nodes, device=states.device, dtype=states.dtype) * node_stride
    node_start = (starts.unsqueeze(0) / valid_len.unsqueeze(1)).clamp(max=1.0)
    node_end = (torch.minimum(starts.unsqueeze(0) + node_stride,
                              valid_len.unsqueeze(1))
                / valid_len.unsqueeze(1)).clamp(min=0.0, max=1.0)
    bounds = torch.stack([node_start, node_end], dim=-1)
    bounds = bounds * node_mask.unsqueeze(-1).to(bounds.dtype)
    centers = (node_start + node_end) * 0.5
    return pooled, node_mask, bounds, centers


def compute_transition_score(visual_nodes, node_mask, threshold=0.2,
                             temperature=0.05, eps=1e-6):
    """Cosine change between adjacent nodes, standardized by a sigmoid.

    Returns ``[B, M]`` where entry ``m`` scores the change between nodes
    ``m`` and ``m+1`` (the last entry is 0).  Fully detached: the score is a
    structural prior only and the model must not collapse visual features to
    remove barriers (proposal 4.2).
    """
    v = F.normalize(visual_nodes.detach().float(), dim=-1, eps=eps)
    change = 1.0 - (v[:, :-1] * v[:, 1:]).sum(-1)  # [B, M-1]
    score = torch.sigmoid((change - threshold) / max(temperature, eps))
    last = torch.zeros_like(score[:, :1])
    full = torch.cat([score, last], dim=1)  # [B, M]
    full = full * node_mask.to(full.dtype)
    return full.to(visual_nodes.dtype)


def build_temporal_adjacency(visual_nodes, node_mask, max_temporal_hop=4,
                             similar_topk=2, similarity_threshold=0.5,
                             eps=1e-6):
    """Self + adjacent + local-similarity neighbour lists of fixed degree.

    Returns ``(neighbor_index [B,M,K], neighbor_type [B,M,K],
    neighbor_mask [B,M,K])`` with ``K = 3 + similar_topk``.  Missing
    neighbours are padded with index 0 and masked out.  Similar edges only
    connect nodes with ``1 < |i-j| <= max_temporal_hop`` and cosine at or
    above ``similarity_threshold`` on the detached clean nodes (proposal 4.3).
    """
    bsz, num_nodes, _ = visual_nodes.shape
    degree = 3 + max(int(similar_topk), 0)
    device = visual_nodes.device
    positions = torch.arange(num_nodes, device=device)

    neighbor_index = torch.zeros(bsz, num_nodes, degree, dtype=torch.long,
                                 device=device)
    neighbor_type = torch.zeros(bsz, num_nodes, degree, dtype=torch.long,
                                device=device)
    neighbor_mask = torch.zeros(bsz, num_nodes, degree, dtype=torch.bool,
                                device=device)

    valid = node_mask.to(torch.bool)
    neighbor_index[:, :, 0] = positions.view(1, num_nodes)
    neighbor_mask[:, :, 0] = valid

    if num_nodes > 1:
        neighbor_index[:, :-1, 1] = positions[1:].view(1, num_nodes - 1)
        neighbor_type[:, :-1, 1] = TEMPORAL_EDGE_NEXT
        neighbor_mask[:, :-1, 1] = valid[:, :-1] & valid[:, 1:]
        neighbor_index[:, 1:, 2] = positions[:-1].view(1, num_nodes - 1)
        neighbor_type[:, 1:, 2] = TEMPORAL_EDGE_PREV
        neighbor_mask[:, 1:, 2] = valid[:, 1:] & valid[:, :-1]

    if similar_topk > 0 and num_nodes > 2:
        v = F.normalize(visual_nodes.detach().float(), dim=-1, eps=eps)
        cosine = torch.bmm(v, v.transpose(1, 2))  # [B, M, M]
        dist = (positions.view(1, num_nodes, 1)
                - positions.view(1, 1, num_nodes)).abs()
        # ``dist`` already has a leading singleton batch dimension because
        # both position views are shaped ``[1, M, 1]``/``[1, 1, M]``.
        # Adding another singleton here makes ``pair_valid`` four
        # dimensional and broadcasts the batch dimension of ``cosine`` a
        # second time (the subsequent top-k indices then have the wrong
        # rank).  Keep the hop mask at ``[1, M, M]`` so it broadcasts only
        # across the real batch dimension.
        hop_ok = (dist > 1) & (dist <= max_temporal_hop)
        pair_valid = (valid.unsqueeze(2) & valid.unsqueeze(1)
                      & hop_ok)
        # Keep invalid entries well below the valid cosine range so a real
        # cosine of -1.0 is still eligible when callers use a low threshold.
        cosine = cosine.masked_fill(~pair_valid, -1e4)
        cosine = cosine.masked_fill(cosine < similarity_threshold, -1e4)
        k = min(int(similar_topk), num_nodes - 2)
        values, indices = torch.topk(cosine, k=k, dim=-1)
        for offset in range(int(similar_topk)):
            slot = 3 + offset
            if offset < k:
                ok = values[:, :, offset] >= similarity_threshold
                idx = torch.where(ok, indices[:, :, offset],
                                  torch.zeros_like(indices[:, :, offset]))
                neighbor_index[:, :, slot] = idx
                neighbor_type[:, :, slot] = TEMPORAL_EDGE_SIMILAR
                neighbor_mask[:, :, slot] = ok
    return neighbor_index, neighbor_type, neighbor_mask


def masked_log_mean_exp(logits, mask, temperature, fill=-1e9, eps=1e-6):
    """``tau * (logsumexp(logits/tau over mask) - log |mask|)`` per row.

    The normalized smooth maximum from proposal sections 5.2/7.3.  Rows with
    an empty mask return ``fill/temperature``-scale large negative values so
    downstream sigmoids stay finite.  ``mask`` has the same shape as
    ``logits`` with the reduction over the last dimension.
    """
    mask = mask.to(torch.bool)
    # Reducing in float32 avoids turning the large masked fill into -inf for
    # fp16 inputs before the empty-row guard is applied.
    work = logits.float()
    count = mask.sum(-1).to(work.dtype)
    masked = work.masked_fill(~mask, float(fill))
    lse = torch.logsumexp(masked / max(float(temperature), eps), dim=-1)
    out = temperature * (lse - count.clamp_min(1).log())
    empty = count <= 0
    if empty.any():
        out = torch.where(empty, torch.full_like(out, -1e4),
                          out)
    return out.to(logits.dtype)


def probabilistic_coverage(membership, phrase_evidence, phrase_required,
                           eps=1e-6):
    """Query->proposal coverage from node memberships.

    Args:
        membership: ``[B, N, M]`` proposal node membership (peak-normalized).
        phrase_evidence: ``[B, P, M]`` time-normalized phrase distribution
            (sums to 1 over valid nodes).
        phrase_required: ``[B, P]`` bool.

    Returns ``(per_phrase [B,N,P], proposal [B,N])``.
    """
    per_phrase = torch.einsum('bnm,bpm->bnp', membership, phrase_evidence)
    required = phrase_required.to(membership.dtype)
    total = (per_phrase * required.unsqueeze(1)).sum(-1) / (
        required.sum(-1).clamp_min(eps)).unsqueeze(-1)
    return per_phrase, total


def resample_to_nodes(weights, node_centers, eps=1e-6):
    """Linearly resample proposal weights from the uniform grid to nodes.

    Args:
        weights: ``[B, N, Tp]`` defined on ``linspace(0, 1, Tp)``.
        node_centers: ``[B, M]`` in ``[0, 1]``.

    Returns ``[B, N, M]`` peak-normalized membership (proposal 7.1/7.2).
    """
    bsz, num_props, grid = weights.shape
    scaled = node_centers.clamp(0.0, 1.0) * float(grid - 1)
    left = torch.floor(scaled).long().clamp(0, grid - 1)
    right = (left + 1).clamp(max=grid - 1)
    frac = (scaled - left.to(scaled.dtype)).clamp(0.0, 1.0)
    idx_left = left.unsqueeze(1).expand(bsz, num_props, scaled.size(-1))
    idx_right = right.unsqueeze(1).expand_as(idx_left)
    out = (weights.gather(2, idx_left) * (1.0 - frac).unsqueeze(1)
           + weights.gather(2, idx_right) * frac.unsqueeze(1))
    out = out / out.amax(dim=-1, keepdim=True).clamp_min(eps)
    return out


def analytic_soft_box(center, width, node_centers, temperature, eps=1e-6):
    """Differentiable box membership ``[B, N, M]`` (proposal 7.1)."""
    start = (center - width * 0.5).unsqueeze(-1)
    end = (center + width * 0.5).unsqueeze(-1)
    positions = node_centers.unsqueeze(1)
    inside = torch.sigmoid((positions - start) / max(float(temperature), eps)) \
        * torch.sigmoid((end - positions) / max(float(temperature), eps))
    return inside


def build_relation_kernel(node_centers, relation_decay=4.0,
                          simultaneous_decay=1.5, eps=1e-6):
    """Per-sample typed relation kernels ``[B, num_types, M, M]``.

    Index 0 is unused (no edge).  Type 1 decays with distance, type 2
    (before) additionally requires ``j >= i``, type 3 (after) requires
    ``j <= i`` and type 4 (simultaneous) uses a shorter decay (proposal 5.3).
    """
    positions = node_centers  # [B, M]
    dist = (positions.unsqueeze(-1) - positions.unsqueeze(-2)).abs()
    decay = torch.exp(-dist / max(float(relation_decay), eps))
    simultaneous = torch.exp(
        -dist / max(float(simultaneous_decay), eps))
    num_nodes = positions.size(-1)
    kernel = positions.new_zeros(positions.size(0), 5, num_nodes, num_nodes)
    kernel[:, EDGE_ASSOC] = decay
    forward = (positions.unsqueeze(-2) <= positions.unsqueeze(-1)).to(
        decay.dtype)  # [B, M(i), M(j)]: j >= i
    backward = (positions.unsqueeze(-2) >= positions.unsqueeze(-1)).to(
        decay.dtype)
    kernel[:, EDGE_BEFORE] = decay * forward
    kernel[:, EDGE_AFTER] = decay * backward
    kernel[:, EDGE_WHILE] = simultaneous
    return kernel


def _gather_neighbor_values(values, neighbor_index):
    """Gather ``[..., M]`` values at neighbour positions -> ``[..., M, K]``.

    ``neighbor_index`` is ``[Bv, M, K]``; it broadcasts against the leading
    dimensions of ``values`` whose last dimension is the video batch.
    """
    degree = neighbor_index.size(-1)
    num_nodes = neighbor_index.size(-2)
    expanded = values.unsqueeze(-1).expand(
        *values.shape[:-1], values.size(-1), degree)
    idx = neighbor_index
    while idx.dim() < expanded.dim():
        idx = idx.unsqueeze(0)
    idx = idx.expand(*values.shape[:-1], num_nodes, degree)
    return expanded.gather(-2, idx)


def compute_relation_satisfaction(membership, binding_prob, content_edge_mask,
                                  query_edge_type, kernel, neighbor_index,
                                  neighbor_mask, eps=1e-6):
    """Diagonal relation satisfaction ``R_n`` (proposal 7.4).

    Args:
        membership: ``[B, N, M]`` proposal node membership.
        binding_prob: ``[B, P, M]`` final binding probability (padding
            phrases already zeroed).
        content_edge_mask: ``[B, P, P]`` bool.
        query_edge_type: ``[B, P, P]`` long.
        kernel: ``[B, T, M, M]`` from :func:`build_relation_kernel`.
        neighbor_index / neighbor_mask: temporal adjacency.

    Returns ``(satisfaction [B, N], valid [B])``.  Samples without any
    content edge return satisfaction 1 and ``valid=False``.
    """
    bsz, num_props, num_nodes = membership.shape
    num_phrases = binding_prob.size(1)
    degree = neighbor_index.size(-1)
    device = membership.device
    dtype = membership.dtype
    batch_idx = torch.arange(bsz, device=device)

    neighbor_mask_f = neighbor_mask.to(dtype)
    aj = membership.unsqueeze(-1).expand(
        bsz, num_props, num_nodes, degree).gather(
            2, neighbor_index.unsqueeze(1).expand(
                bsz, num_props, num_nodes, degree))

    num = membership.new_zeros(bsz, num_props)
    den = membership.new_zeros(bsz, num_props)
    edge_count = membership.new_zeros(bsz)
    for p in range(num_phrases):
        for q in range(num_phrases):
            if p == q:
                continue
            emask = content_edge_mask[:, p, q]
            if not emask.any():
                continue
            p_j = _gather_neighbor_values(binding_prob[:, q, :],
                                          neighbor_index)
            w_i = membership * binding_prob[:, p, :].unsqueeze(1)
            w_j = aj * p_j.unsqueeze(1) * neighbor_mask_f.unsqueeze(1)
            prod = w_i.unsqueeze(-1) * w_j  # [B, N, M, K]
            kij = kernel[batch_idx, query_edge_type[:, p, q]]
            kij = kij.gather(2, neighbor_index) * neighbor_mask_f
            num = num + (prod * kij.unsqueeze(1)).sum((-1, -2)) \
                * emask.to(dtype).unsqueeze(-1)
            den = den + prod.sum((-1, -2)) * emask.to(dtype).unsqueeze(-1)
            edge_count = edge_count + emask.to(dtype)

    # ``num``/``den`` is already the weighted average over all active
    # phrase-pair edges.  Dividing by ``edge_count`` again would make a
    # sample with two valid relations look half as satisfied as the same
    # sample with one relation.
    satisfaction = num / den.clamp_min(eps)
    valid = edge_count > 0
    satisfaction = torch.where(valid.unsqueeze(-1), satisfaction,
                               torch.ones_like(satisfaction))
    return satisfaction, valid


def compute_pair_relation_satisfaction(membership, binding_prob,
                                       content_edge_mask, query_edge_type,
                                       kernel, neighbor_index, neighbor_mask,
                                       eps=1e-6):
    """Cross-batch relation satisfaction for pair quality (proposal 7.7).

    Args:
        membership: ``[Bv, N, M]``.
        binding_prob: ``[Bq, Bv, P, M]``.
        content_edge_mask / query_edge_type: ``[Bq, P, P]`` (query side).
        kernel: ``[Bv, T, M, M]`` (video side).

    Returns ``(satisfaction [Bq, Bv, N], valid [Bq])``.
    """
    num_queries = binding_prob.size(0)
    num_videos = membership.size(0)
    num_props = membership.size(1)
    num_nodes = membership.size(2)
    num_phrases = binding_prob.size(2)
    degree = neighbor_index.size(-1)
    device = membership.device
    dtype = membership.dtype

    neighbor_mask_f = neighbor_mask.to(dtype)
    neighbor_nb = neighbor_index.unsqueeze(0).expand(
        num_queries, num_videos, num_nodes, degree)
    aj = membership.unsqueeze(-1).expand(
        num_videos, num_props, num_nodes, degree).gather(
            2, neighbor_index.unsqueeze(1).expand(
                num_videos, num_props, num_nodes, degree))
    aj = aj.unsqueeze(0) * neighbor_mask_f.unsqueeze(0).unsqueeze(2)

    num = membership.new_zeros(num_queries, num_videos, num_props)
    den = membership.new_zeros(num_queries, num_videos, num_props)
    edge_count = membership.new_zeros(num_queries)
    for p in range(num_phrases):
        for q in range(num_phrases):
            if p == q:
                continue
            emask = content_edge_mask[:, p, q]
            if not emask.any():
                continue
            p_j = _gather_neighbor_values(binding_prob[:, :, q, :],
                                          neighbor_index)
            w_i = membership.unsqueeze(0) \
                * binding_prob[:, :, p, :].unsqueeze(2)  # [Bq,Bv,N,M]
            w_j = aj * p_j.unsqueeze(2) * neighbor_mask_f \
                .unsqueeze(0).unsqueeze(2)  # [Bq,Bv,N,M,K]
            prod = w_i.unsqueeze(-1) * w_j
            # kij[a, v, i, j] = kernel[v, type_a(p, q), i, j]
            kij = kernel.index_select(1, query_edge_type[:, p, q]) \
                .transpose(0, 1)  # [Bq, Bv, M, M]
            kij = kij.gather(3, neighbor_nb) * neighbor_mask_f \
                .unsqueeze(0)  # [Bq, Bv, M, K]
            num = num + (prod * kij.unsqueeze(2)).sum((-1, -2)) \
                * emask.to(dtype).view(num_queries, 1, 1)
            den = den + prod.sum((-1, -2)) \
                * emask.to(dtype).view(num_queries, 1, 1)
            edge_count = edge_count + emask.to(dtype)

    satisfaction = num / den.clamp_min(eps)
    valid = edge_count > 0
    satisfaction = torch.where(valid.view(-1, 1, 1), satisfaction,
                               torch.ones_like(satisfaction))
    return satisfaction, valid


def compute_connectivity(soft_box, barrier, eps=1e-6):
    """Connectivity ``H_n`` from the analytic soft box and barriers (7.5)."""
    pair = soft_box[:, :, :-1] * soft_box[:, :, 1:]  # [B, N, M-1]
    cross = pair * barrier[:, :-1].unsqueeze(1)
    return torch.exp(-cross.sum(-1) / pair.sum(-1).clamp_min(eps))


def build_barrier(transition_score, node_relevance, relation_edge_affinity,
                  detach=True, eps=1e-6):
    """Adjacent-node barrier ``beta_m`` (proposal 7.5).

    ``beta_m = d_hat_m * |E_m - E_{m+1}| * (1 - A^rel_{m,m+1})``.  Returns
    ``[B, M]`` with the final entry zero.  Detached by default so the
    connectivity loss moves proposal geometry instead of erasing barriers.
    """
    if detach:
        transition_score = transition_score.detach()
        node_relevance = node_relevance.detach()
        relation_edge_affinity = relation_edge_affinity.detach()
    num_nodes = transition_score.size(-1)
    # affinity between adjacent nodes m and m+1
    idx = torch.arange(num_nodes - 1, device=relation_edge_affinity.device)
    rel_adjacent = relation_edge_affinity[:, idx, idx + 1]
    relevance = node_relevance
    diff = (relevance[:, :-1] - relevance[:, 1:]).abs()
    barrier = transition_score[:, :-1] * diff * (1.0 - rel_adjacent)
    barrier = barrier.clamp(min=0.0, max=1.0)
    last = torch.zeros_like(barrier[:, :1])
    return torch.cat([barrier, last], dim=1)


def build_same_video_negative_mask(video_group_id):
    """Mask for cross-batch contrastives (proposal 7.8).

    Returns ``[B, B]`` bool where ``True`` marks usable score positions:
    the diagonal (positive) and videos from other groups (negatives).
    Same-video off-diagonal pairs are excluded.  ``None`` input disables
    the mask.
    """
    if video_group_id is None:
        return None
    group = video_group_id.view(-1, 1)
    same_video = group == group.transpose(0, 1)
    eye = torch.eye(group.size(0), dtype=torch.bool, device=group.device)
    return (~same_video) | eye


def dense_temporal_adjacency(neighbor_index, neighbor_mask, dtype=torch.float32):
    """Dense ``[B, M, M]`` adjacency from neighbour lists."""
    num_nodes = neighbor_index.size(1)
    adj = torch.zeros(neighbor_index.size(0), num_nodes, num_nodes,
                      device=neighbor_index.device, dtype=dtype)
    adj.scatter_(2, neighbor_index, neighbor_mask.to(dtype))
    return (adj > 0).to(dtype)


def connected_refine(node_bounds, node_mask, membership, soft_box,
                     phrase_evidence, phrase_required, node_relevance,
                     barrier, relation_edge_affinity, binding_prob,
                     content_edge_mask, query_edge_type, kernel,
                     neighbor_index, neighbor_mask, center, width,
                     membership_threshold=0.1, soft_box_threshold=0.5,
                     hard_barrier_threshold=0.65,
                     relation_connect_threshold=0.4,
                     coverage_keep_ratio=0.9, max_refine_shift=0.1,
                     min_refine_width=0.01, move_penalty=0.1,
                     score_weights=(0.40, 0.30, 0.20, 0.10), eps=1e-6):
    """Search the best connected sub-span inside every proposal (9.2).

    Eval-only refinement: keeps the proposal count fixed, cuts each
    proposal's support at hard barriers that the query relation graph does
    not support, scores every resulting connected component with
    ``0.40C + 0.30X + 0.20R + 0.10H - lambda_move * shift`` and falls back
    to the raw span whenever no component is acceptable.
    """
    with torch.no_grad():
        bsz, num_props, num_nodes = membership.shape
        num_phrases = phrase_evidence.size(1)
        device = membership.device
        dtype = membership.dtype
        w_cov, w_ex, w_rel, w_conn = score_weights

        support = ((membership >= membership_threshold)
                   | (soft_box >= soft_box_threshold)) \
            & node_mask.unsqueeze(1)
        positions = torch.arange(num_nodes, device=device)
        idx = torch.arange(num_nodes - 1, device=device)
        rel_adjacent = relation_edge_affinity[:, idx, idx + 1]
        cut = ((barrier[:, :-1] >= hard_barrier_threshold)
               & (rel_adjacent < relation_connect_threshold))  # [B, M-1]
        no_break = support[:, :, :-1] & support[:, :, 1:] \
            & ~cut.unsqueeze(1)
        run_start = support.clone()
        run_start[:, :, 1:] = support[:, :, 1:] & ~no_break
        run_id = torch.where(
            support, (run_start.cumsum(-1) - 1),
            torch.full_like(run_start, -1, dtype=torch.long))
        num_runs = run_start.sum(-1)  # [B, N]
        max_runs = max(int(num_runs.max().item()), 1)

        cand_start = torch.zeros(bsz, num_props, max_runs, dtype=torch.long,
                                 device=device)
        cand_end = torch.zeros_like(cand_start)
        cand_valid = torch.zeros(bsz, num_props, max_runs, dtype=torch.bool,
                                 device=device)
        for run in range(max_runs):
            hit = run_id == run
            any_hit = hit.any(-1)
            s = torch.where(hit, positions.view(1, 1, -1),
                            torch.full_like(hit, num_nodes, dtype=torch.long)) \
                .min(-1)[0]
            e = torch.where(hit, positions.view(1, 1, -1),
                            torch.full_like(hit, -1, dtype=torch.long)) \
                .max(-1)[0]
            cand_start[:, :, run] = s
            cand_end[:, :, run] = e
            cand_valid[:, :, run] = any_hit & (s <= e)

        pos_f = positions.to(dtype).view(1, 1, 1, num_nodes)
        in_seg = ((pos_f >= cand_start.unsqueeze(-1).to(dtype))
                  & (pos_f <= cand_end.unsqueeze(-1).to(dtype))
                  & cand_valid.unsqueeze(-1)
                  & node_mask.view(bsz, 1, 1, num_nodes))  # [B, N, R, M]
        in_seg_f = in_seg.to(dtype)

        required = phrase_required.to(dtype)
        req_total = required.sum().clamp_min(eps)
        seg_phrase = torch.einsum('bpm,bnrm->bnrp', phrase_evidence, in_seg_f)
        c_seg = (seg_phrase * required.view(bsz, 1, 1, num_phrases)) \
            .sum(-1) / req_total
        # The acceptance baseline is the raw Gaussian/mixed membership, not
        # the thresholded support used to discover connected candidates.
        raw_phrase = torch.einsum('bpm,bnm->bnp', phrase_evidence, membership)
        c_raw = (raw_phrase * required.unsqueeze(1)).sum(-1) / req_total

        seg_count = in_seg_f.sum(-1).clamp_min(eps)
        x_seg = torch.einsum('bm,bnrm->bnr', node_relevance, in_seg_f) \
            / seg_count

        inside = in_seg[:, :, :, :-1] & in_seg[:, :, :, 1:]
        bar_sum = (barrier[:, :-1].view(bsz, 1, 1, num_nodes - 1)
                   * inside.to(dtype)).sum(-1)
        pair_count = inside.sum(-1).to(dtype)
        h_seg = torch.exp(-bar_sum / pair_count.clamp_min(1.0))

        adj = dense_temporal_adjacency(neighbor_index, neighbor_mask, dtype)
        batch_idx = torch.arange(bsz, device=device)
        r_seg = membership.new_zeros(bsz, num_props, max_runs)
        edge_count = membership.new_zeros(bsz)
        for p in range(num_phrases):
            for q in range(num_phrases):
                if p == q:
                    continue
                emask = content_edge_mask[:, p, q]
                if not emask.any():
                    continue
                outer = binding_prob[:, p, :].unsqueeze(2) \
                    * binding_prob[:, q, :].unsqueeze(1)  # [B, M, M]
                den_e = outer * adj
                kij = kernel[batch_idx, query_edge_type[:, p, q]]
                num_e = den_e * kij
                seg_num = torch.einsum('bij,bnri,bnrj->bnr', num_e,
                                       in_seg_f, in_seg_f)
                seg_den = torch.einsum('bij,bnri,bnrj->bnr', den_e,
                                       in_seg_f, in_seg_f)
                r_edge = seg_num / seg_den.clamp_min(eps)
                r_seg = r_seg + r_edge * emask.to(dtype).view(bsz, 1, 1)
                edge_count = edge_count + emask.to(dtype)
        r_seg = r_seg / edge_count.clamp_min(1.0).view(bsz, 1, 1)
        r_seg = torch.where((edge_count > 0).view(bsz, 1, 1), r_seg,
                            torch.ones_like(r_seg))

        # Empty runs use ``num_nodes``/``-1`` sentinels.  They are already
        # excluded by ``cand_valid``; clamp only for this geometry gather so
        # the sentinel cannot address outside the node table.
        safe_start = cand_start.clamp(0, num_nodes - 1)
        safe_end = cand_end.clamp(0, num_nodes - 1)
        start_pos = node_bounds[:, :, 0].gather(
            1, safe_start.view(bsz, -1)).view(bsz, num_props, max_runs)
        end_pos = node_bounds[:, :, 1].gather(
            1, safe_end.view(bsz, -1)).view(bsz, num_props, max_runs)
        raw_start = (center - width * 0.5).unsqueeze(-1)
        raw_end = (center + width * 0.5).unsqueeze(-1)
        move = (start_pos - raw_start).abs() + (end_pos - raw_end).abs()

        score = (w_cov * c_seg + w_ex * x_seg + w_rel * r_seg
                 + w_conn * h_seg - move_penalty * move)
        acceptable = cand_valid & (c_seg >= coverage_keep_ratio
                                   * c_raw.unsqueeze(-1))
        score = score.masked_fill(~acceptable, -1e9)
        best = score.argmax(-1)  # [B, N]
        has_candidate = acceptable.any(-1)

        gather = best.unsqueeze(-1)
        best_start = cand_start.gather(2, gather).squeeze(-1)
        best_end = cand_end.gather(2, gather).squeeze(-1)
        best_move = move.gather(2, gather).squeeze(-1)
        new_start = node_bounds[:, :, 0].gather(1, best_start)
        new_end = node_bounds[:, :, 1].gather(1, best_end)
        new_center = (new_start + new_end) * 0.5
        new_width = (new_end - new_start).clamp_min(eps)

        usable = has_candidate & (new_width >= min_refine_width) \
            & (best_move <= max_refine_shift)
        eval_center = torch.where(usable, new_center, center)
        eval_width = torch.where(usable, new_width, width.clamp_min(eps))
        shift = torch.stack(
            [new_start - raw_start.squeeze(-1),
             new_end - raw_end.squeeze(-1)], dim=-1)
        shift = shift * usable.unsqueeze(-1).to(shift.dtype)
        return {
            'eval_center': eval_center,
            'eval_width': eval_width,
            'refine_mask': usable,
            'refine_shift': shift,
        }
