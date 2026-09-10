import torch

from models.modules.qcec import (
    QCECProposalAdapter,
    QueryConditionedCoherentEventClusters,
    boundary_enhanced_pool,
    compute_directional_hints,
    compute_transition_scores,
    pool_query_roles,
    snap_proposal_boundaries,
)


def make_metadata(batch_size=2, frames=20, clusters=4):
    ids = (torch.arange(frames).view(1, frames) * clusters // frames)
    ids = ids.expand(batch_size, -1).long()
    bounds = torch.tensor([
        [[0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    ]).expand(batch_size, -1, -1).clone()
    mask = torch.ones(batch_size, clusters, dtype=torch.bool)
    return ids, bounds, mask


def test_boundary_pooling_boosts_cluster_edges_and_masks_padding():
    source_leaf = torch.arange(4.0, requires_grad=True)
    source = source_leaf.view(1, 4, 1)
    pooled = boundary_enhanced_pool(
        source,
        torch.ones(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.long),
        torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        torch.tensor([[True, False]]),
    )
    pooled.sum().backward()
    assert source_leaf.grad is not None
    assert source_leaf.grad[0] > source_leaf.grad[1]
    assert torch.allclose(
        boundary_enhanced_pool(
            torch.ones(1, 4, 2), torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([[-1, -1, -1, -1]]),
            torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
            torch.tensor([[False, False]]),
        ),
        torch.zeros(1, 2, 2),
    )


def test_query_role_fallback_is_finite():
    query = torch.randn(2, 5, 8)
    query_mask = torch.ones(2, 5, dtype=torch.bool)
    roles = torch.zeros(2, 3, 5, dtype=torch.bool)
    roles[:, 2] = True
    valid = torch.tensor([[False, False, True], [False, True, True]])
    units = pool_query_roles(query, query_mask, roles, valid)
    assert units.shape == (2, 3, 8)
    assert torch.isfinite(units).all()
    assert torch.allclose(units[:, 0], units[:, 2])


def test_directional_barrier_points_from_relevant_left_to_irrelevant_right():
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    transition, mask = compute_transition_scores(
        tokens, torch.ones(1, 2, dtype=torch.bool))
    relevance = torch.tensor([[0.9, 0.1]], requires_grad=True)
    left, right, barrier_lr, barrier_rl = compute_directional_hints(
        relevance, transition, mask, cluster_mask=torch.ones(1, 2, dtype=torch.bool))
    assert barrier_lr.item() > barrier_rl.item()
    assert left.shape == right.shape == (1, 2)


def test_qcec_contract_and_backward():
    batch_size, frames, clusters, words, hidden, proposals = 2, 20, 4, 6, 8, 3
    module = QueryConditionedCoherentEventClusters(
        hidden, proposals, 3, attention_dim=8, use_slot_prior=True)
    frame_states = torch.randn(batch_size, frames, hidden, requires_grad=True)
    query_states = torch.randn(batch_size, words, hidden, requires_grad=True)
    ids, bounds, cluster_mask = make_metadata(batch_size, frames, clusters)
    output = module(
        frame_states, torch.ones(batch_size, frames, dtype=torch.bool),
        query_states, torch.ones(batch_size, words, dtype=torch.bool),
        torch.ones(batch_size, 3, words, dtype=torch.bool),
        torch.ones(batch_size, 3, dtype=torch.bool),
        ids, bounds, cluster_mask)
    assert output['cluster_tokens'].shape == (batch_size, clusters, hidden)
    assert output['global_hint'].shape == (batch_size, hidden)
    assert output['center_logit_bias'].shape == (batch_size, 6)
    assert output['width_logit_bias'].shape == (batch_size, proposals)
    assert output['single_logit_bias'].shape == (batch_size, proposals, 2)
    objective = output['global_hint'].sum() + output['cluster_relevance'].sum()
    objective.backward()
    assert torch.isfinite(frame_states.grad).all()
    assert torch.isfinite(query_states.grad).all()
    assert torch.isfinite(module.relevance_bias.grad)


def test_adapter_is_identity_at_initialization():
    adapter = QCECProposalAdapter(8)
    base = torch.randn(3, 8)
    hint = torch.randn(3, 8)
    assert torch.equal(adapter(base, hint), base)


def test_snapping_uses_nearest_confident_boundaries_and_preserves_width():
    _, bounds, mask = make_metadata(1, 20, 4)
    center, width, diagnostics = snap_proposal_boundaries(
        torch.tensor([[0.40, 0.80]]), torch.tensor([[0.30, 0.20]]),
        bounds,
        torch.tensor([[0.9, 0.8, 0.1, 0.1]]),
        torch.tensor([[0.9, 0.8, 0.1, 0.1]]),
        torch.tensor([[0.1, 0.1, 0.1, 0.1]]), mask,
        torch.tensor([20]), radius_frames=2,
        min_confidence=0.5, min_relevance=0.5)
    assert center.shape == width.shape == (1, 2)
    assert torch.isfinite(center).all()
    assert torch.all(width > 0)
    assert torch.all(diagnostics['start_abs_delta'] <= 0.1)


def test_snapping_does_not_use_video_edges_without_edge_threshold():
    _, bounds, mask = make_metadata(1, 20, 4)
    center, width, diagnostics = snap_proposal_boundaries(
        torch.tensor([[0.04]]), torch.tensor([[0.08]]), bounds,
        torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4), mask,
        torch.tensor([20]), radius_frames=2, min_confidence=0.5,
        min_relevance=0.5)
    assert torch.allclose(center, torch.tensor([[0.04]]))
    assert torch.allclose(width, torch.tensor([[0.08]]))
    assert not diagnostics['start_moved'].item()
