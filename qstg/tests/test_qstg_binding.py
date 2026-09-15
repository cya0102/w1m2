import torch

from models.modules.qstg import QuerySubgraphTemporalGrounder


def _inputs(batch=2, dim=8, words=6, phrases=4, frames=20):
    torch.manual_seed(12)
    token_mask = torch.zeros(batch, phrases, words, dtype=torch.bool)
    token_mask[:, 0] = True
    token_mask[:, 1, :2] = True
    token_mask[:, 2, 2:4] = True
    token_mask[:, 3, 4] = True
    phrase_type = torch.tensor([[1, 2, 3, 5]] * batch)
    valid = torch.ones(batch, phrases, dtype=torch.bool)
    required = torch.tensor([[0, 1, 1, 1]] * batch, dtype=torch.bool)
    edges = torch.zeros(batch, phrases, phrases, dtype=torch.long)
    edges[:, 1, 2] = edges[:, 2, 1] = 1
    return (torch.randn(batch, dim), torch.randn(batch, frames, dim),
            torch.randn(batch, frames, dim), torch.ones(batch, frames, dtype=torch.bool),
            torch.randn(batch, words, dim), torch.ones(batch, words, dtype=torch.bool),
            token_mask, phrase_type, valid, required, edges)


def test_pre_proposal_binding_and_gate_shapes_are_finite():
    model = QuerySubgraphTemporalGrounder(
        hidden_size=8, num_proposals=3, max_components=3,
        component_counts=[1, 2, 3], max_phrases=4, node_stride=4,
        num_graph_layers=1, num_graph_heads=2, graph_dropout=0.0)
    args = _inputs()
    output = model.forward_pre_proposal(*args, video_group_id=torch.tensor([0, 1]))
    assert output['pair_binding_logits'].shape == (2, 2, 4, 5)
    assert output['binding_logits'].shape == (2, 4, 5)
    assert output['node_gate'].shape == (2, 5)
    for name in ('pre_binding_logits', 'binding_prob', 'node_relevance', 'global_hint'):
        assert torch.isfinite(output[name]).all()
    assert output['node_gate'].min() >= 0
    assert output['node_gate'].max() <= 1
