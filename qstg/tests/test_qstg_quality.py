import torch

from models.modules.qstg import QuerySubgraphTemporalGrounder
from test_qstg_binding import _inputs


def test_quality_outputs_use_actual_mask_and_keep_proposal_count():
    model = QuerySubgraphTemporalGrounder(
        hidden_size=8, num_proposals=3, max_components=3,
        component_counts=[1, 2, 3], max_phrases=4, node_stride=4,
        num_graph_layers=1, num_graph_heads=2, graph_dropout=0.0)
    state = model.forward_pre_proposal(*_inputs(), video_group_id=torch.tensor([0, 1]))
    weights = torch.rand(6, 5)
    centers = torch.sigmoid(torch.randn(6))
    widths = torch.sigmoid(torch.randn(6))
    result = model.score_proposals(weights, centers, widths, state)
    assert result['proposal_node_membership'].shape == (2, 3, 5)
    assert result['proposal_coverage_per_phrase'].shape == (2, 3, 4)
    assert result['proposal_quality_logit'].shape == (2, 3)
    assert result['pair_quality_logits'].shape == (2, 2, 3)
    assert torch.isfinite(result['proposal_analytic_score']).all()
    assert torch.isfinite(result['proposal_connectivity']).all()


def test_zero_initialized_quality_and_global_adapters_are_identity():
    model = QuerySubgraphTemporalGrounder(
        hidden_size=8, num_proposals=2, max_components=2,
        component_counts=[1, 2], max_phrases=4, num_graph_heads=2,
        graph_dropout=0.0)
    args = _inputs()
    state = model.forward_pre_proposal(*args, video_group_id=torch.tensor([0, 1]))
    assert torch.allclose(
        state['adapted_global_feature'], args[0], atol=0, rtol=0)
    assert torch.equal(state['center_logit_bias'], torch.zeros_like(
        state['center_logit_bias']))
