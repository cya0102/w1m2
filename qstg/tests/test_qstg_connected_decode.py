import torch

from models.modules.qstg_ops import connected_refine


def test_connected_refine_is_finite_and_preserves_proposal_count():
    batch, props, nodes, phrases = 1, 2, 6, 2
    bounds = torch.stack([torch.arange(nodes, dtype=torch.float32) / nodes,
                          torch.arange(1, nodes + 1, dtype=torch.float32) / nodes], -1).unsqueeze(0)
    node_mask = torch.ones(batch, nodes, dtype=torch.bool)
    membership = torch.zeros(batch, props, nodes)
    membership[:, :, 1:5] = 1
    soft_box = membership.clone()
    evidence = torch.full((batch, phrases, nodes), 1 / nodes)
    required = torch.ones(batch, phrases, dtype=torch.bool)
    relevance = torch.ones(batch, nodes)
    barrier = torch.zeros(batch, nodes)
    relation = torch.zeros(batch, nodes, nodes)
    binding = torch.ones(batch, phrases, nodes)
    edge = torch.zeros(batch, phrases, phrases, dtype=torch.bool)
    edge[:, 0, 1] = True
    edge[:, 1, 0] = True
    edge_type = torch.zeros(batch, phrases, phrases, dtype=torch.long)
    edge_type[:, 0, 1] = 1
    edge_type[:, 1, 0] = 1
    kernel = torch.zeros(batch, 5, nodes, nodes)
    neighbor_index = torch.arange(nodes).view(1, nodes, 1).expand(batch, -1, 1)
    neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool)
    result = connected_refine(
        bounds, node_mask, membership, soft_box, evidence, required,
        relevance, barrier, relation, binding, edge, edge_type, kernel,
        neighbor_index, neighbor_mask, torch.full((batch, props), .5),
        torch.full((batch, props), .6))
    assert result['eval_center'].shape == (batch, props)
    assert result['eval_width'].shape == (batch, props)
    assert result['refine_mask'].shape == (batch, props)
    assert torch.isfinite(result['eval_center']).all()
    assert torch.all(result['eval_width'] > 0)
