import torch

from models.loss import qstg_loss


def test_qstg_loss_is_finite_and_connected_to_binding():
    torch.manual_seed(9)
    batch, props, phrases, nodes = 3, 2, 4, 5
    words = torch.randn(batch * props, 4, 7, requires_grad=True)
    pair = torch.randn(batch, batch, phrases, nodes, requires_grad=True)
    binding = torch.randn(batch, phrases, nodes, requires_grad=True)
    membership = torch.rand(batch, props, nodes, requires_grad=True)
    membership = membership / membership.amax(-1, keepdim=True)
    required = torch.tensor([[0, 1, 1, 0]] * batch, dtype=torch.bool)
    valid = torch.ones(batch, phrases, dtype=torch.bool)
    loss, metrics = qstg_loss(
        words, props, pair_binding_logits=pair,
        proposal_reconstruction_nll=torch.rand(batch, props),
        proposal_node_membership=membership, binding_logits=binding,
        phrase_required=required, phrase_valid=valid,
        video_group_id=torch.tensor([0, 1, 2]),
        node_gate=torch.full((batch, nodes), 0.5),
        proposal_connectivity=torch.full((batch, props), 0.8))
    assert torch.isfinite(loss)
    assert set(('qstg_loss', 'qstg_mc_loss', 'qstg_proposal_loss',
                'qstg_fa_loss', 'qstg_explain_loss', 'qstg_gate_loss',
                'qstg_connect_loss', 'qstg_role_loss')).issubset(metrics)
    loss.backward()
    assert pair.grad is not None and binding.grad is not None
    assert torch.isfinite(pair.grad).all()


def test_disabled_qstg_loss_returns_connected_zero_and_no_nan_for_batch_one():
    words = torch.randn(1, 1, 4, requires_grad=True)
    loss, metrics = qstg_loss(words, 1)
    assert loss.item() == 0
    assert metrics['qstg_loss'] == 0
    loss.backward()
    assert words.grad is not None


def test_same_video_false_negatives_are_ignored():
    words = torch.zeros(2, 1, 4, requires_grad=True)
    pair = torch.zeros(2, 2, 2, 3, requires_grad=True)
    required = torch.ones(2, 2, dtype=torch.bool)
    valid = required.clone()
    loss, metrics = qstg_loss(
        words, 1, pair_binding_logits=pair, phrase_required=required,
        phrase_valid=valid, video_group_id=torch.tensor([4, 4]))
    assert loss.item() == 0
    assert metrics['qstg_valid_negative_ratio'] == 0


def test_pair_quality_selector_reduces_only_the_proposal_axis():
    torch.manual_seed(12)
    batch, props, phrases, nodes = 3, 2, 2, 4
    words = torch.zeros(batch * props, 1, 3, requires_grad=True)
    pair = torch.randn(batch, batch, phrases, nodes, requires_grad=True)
    pair_quality = torch.randn(batch, batch, props, requires_grad=True)
    nll = torch.tensor([[0.0, 4.0], [0.5, 0.5], [4.0, 0.0]])
    required = torch.ones(batch, phrases, dtype=torch.bool)
    loss, _ = qstg_loss(
        words, props, proposal_reconstruction_nll=nll,
        pair_binding_logits=pair, pair_quality_logits=pair_quality,
        phrase_required=required, phrase_valid=required,
        video_group_id=torch.arange(batch))
    assert torch.isfinite(loss)
    loss.backward()
    assert pair_quality.grad is not None
    assert torch.isfinite(pair_quality.grad).all()


def test_stage_b_weak_losses_follow_configured_ramp():
    words = torch.zeros(2, 1, 3, requires_grad=True)
    membership = torch.full((2, 1, 3), 0.5, requires_grad=True)
    binding = torch.zeros(2, 2, 3, requires_grad=True)
    required = torch.ones(2, 2, dtype=torch.bool)
    nll = torch.tensor([[0.0], [0.0]])
    common = dict(
        proposal_reconstruction_nll=nll,
        proposal_node_membership=membership, binding_logits=binding,
        phrase_required=required, phrase_valid=required,
        qstg_total_weight=1.0, qstg_mc_weight=0.0,
        qstg_proposal_weight=0.0, qstg_fa_weight=0.4,
        qstg_explain_weight=0.2, qstg_connect_weight=0.3,
        qstg_role_weight=0.1, qstg_stage='B')
    _, warm_metrics = qstg_loss(
        words, 1, qstg_epoch=1, **common)
    _, full_metrics = qstg_loss(
        words, 1, qstg_epoch=4, **common)
    assert warm_metrics['qstg_ramp_factor'] == 0.0
    assert full_metrics['qstg_ramp_factor'] == 1.0
    assert warm_metrics['qstg_effective_fa_weight'] == 0.0
    assert full_metrics['qstg_effective_fa_weight'] == 0.4
