import torch

from models.modules.qstg_ops import (
    build_temporal_adjacency,
    build_relation_kernel,
    compute_relation_satisfaction,
    compute_transition_score,
    masked_log_mean_exp,
    temporal_pool,
)


def test_stride_pool_and_sparse_adjacency():
    torch.manual_seed(4)
    states = torch.randn(2, 10, 6)
    mask = torch.tensor([[1] * 10, [1] * 7 + [0] * 3], dtype=torch.bool)
    pooled, node_mask, bounds, centers = temporal_pool(states, mask, 4)
    assert pooled.shape == (2, 3, 6)
    assert node_mask.tolist() == [[True, True, True], [True, True, False]]
    assert bounds.shape == (2, 3, 2)
    assert centers.shape == (2, 3)
    index, edge_type, edge_mask = build_temporal_adjacency(
        pooled, node_mask, max_temporal_hop=2, similar_topk=2,
        similarity_threshold=-1.0)
    assert index.shape == (2, 3, 5)
    assert edge_type.shape == edge_mask.shape == index.shape
    assert edge_mask[:, :, 0].equal(node_mask)
    assert torch.isfinite(compute_transition_score(pooled, node_mask)).all()


def test_adjacency_topk_keeps_indices_in_range_for_nontrivial_batch():
    nodes = torch.randn(3, 50, 8)
    mask = torch.ones(3, 50, dtype=torch.bool)
    index, _, valid = build_temporal_adjacency(nodes, mask, similar_topk=2)
    assert index.min() >= 0 and index.max() < 50
    assert valid[:, :, 0].all()


def test_multiple_relation_edges_are_averaged_once():
    batch, proposals, phrases, nodes = 1, 1, 3, 4
    membership = torch.ones(batch, proposals, nodes)
    binding = torch.ones(batch, phrases, nodes)
    content_edges = torch.zeros(batch, phrases, phrases, dtype=torch.bool)
    content_edges[:, 0, 1] = True
    content_edges[:, 1, 0] = True
    edge_types = torch.zeros(batch, phrases, phrases, dtype=torch.long)
    edge_types[content_edges] = 1
    centers = torch.linspace(0.125, 0.875, nodes).view(1, nodes)
    kernel = build_relation_kernel(centers)
    neighbor_index = torch.arange(nodes).view(1, nodes, 1).expand(
        batch, nodes, nodes)
    neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool)
    satisfaction, valid = compute_relation_satisfaction(
        membership, binding, content_edges, edge_types, kernel,
        neighbor_index, neighbor_mask)
    assert valid.tolist() == [True]
    assert torch.allclose(satisfaction, torch.ones_like(satisfaction))


def test_masked_log_mean_exp_is_finite_for_half_and_empty_rows():
    logits = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float16)
    mask = torch.tensor([[True, True], [False, False]])
    result = masked_log_mean_exp(logits, mask, temperature=0.1)
    assert torch.isfinite(result).all()
    assert result.dtype == logits.dtype
