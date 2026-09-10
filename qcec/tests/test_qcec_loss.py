import torch

from models.loss import qcec_coherence_loss


def make_loss_inputs(center):
    words_logit = torch.zeros(1, 3, 5, requires_grad=True)
    width = torch.full_like(center, 0.4)
    positions = torch.tensor([[0.5, 0.8]])
    barrier_lr = torch.tensor([[1.0, 0.0]])
    barrier_rl = torch.zeros_like(barrier_lr)
    mask = torch.ones_like(barrier_lr, dtype=torch.bool)
    return dict(
        words_logit=words_logit, num_props=1, center=center, width=width,
        qcec_transition_positions=positions,
        qcec_barrier_lr=barrier_lr, qcec_barrier_rl=barrier_rl,
        qcec_transition_mask=mask, qcec_cross_weight=1.0,
        qcec_cross_temperature=0.02,
    )


def test_crossing_loss_increases_when_endpoint_moves_past_barrier():
    inside, inside_metrics = qcec_coherence_loss(
        **make_loss_inputs(torch.tensor([[0.35]], requires_grad=True)))
    crossing, crossing_metrics = qcec_coherence_loss(
        **make_loss_inputs(torch.tensor([[0.65]], requires_grad=True)))
    assert crossing.item() > inside.item()
    assert crossing_metrics['qcec_cross_raw'] > inside_metrics['qcec_cross_raw']


def test_crossing_loss_has_proposal_gradient_and_detaches_barrier_by_default():
    center = torch.tensor([[0.65]], requires_grad=True)
    barrier = torch.tensor([[1.0]], requires_grad=True)
    loss, _ = qcec_coherence_loss(
        words_logit=torch.zeros(1, 3, 5, requires_grad=True), num_props=1,
        center=center, width=torch.tensor([[0.4]]),
        qcec_transition_positions=torch.tensor([[0.5]]),
        qcec_barrier_lr=barrier, qcec_barrier_rl=torch.zeros_like(barrier),
        qcec_transition_mask=torch.ones(1, 1, dtype=torch.bool),
        qcec_cross_weight=1.0, qcec_detach_barrier=True)
    loss.backward()
    assert center.grad is not None and torch.isfinite(center.grad).all()
    assert barrier.grad is None


def test_disabled_qcec_loss_is_graph_connected_zero():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    loss, metrics = qcec_coherence_loss(
        words_logit=logits, num_props=3, qcec_cross_weight=1.0)
    assert loss.item() == 0.0
    assert all(value == 0.0 for value in metrics.values())
    loss.backward()
    assert logits.grad is not None
