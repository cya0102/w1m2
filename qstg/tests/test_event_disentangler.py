import torch

from models.loss import event_disentanglement_loss
from models.modules.event_disentangler import LowRankEventDisentangler


def make_event_inputs(batch_size=4, num_props=3, time=6, dim=8):
    torch.manual_seed(3)
    features = torch.randn(
        batch_size * num_props, time, dim, requires_grad=True)
    mask = torch.ones(batch_size * num_props, time, dtype=torch.uint8)
    positive_weights = torch.softmax(
        torch.randn(batch_size * num_props, time), dim=-1)
    negative_weights = torch.softmax(
        torch.randn(batch_size * num_props, 2, time), dim=-1)
    selection = torch.softmax(torch.randn(batch_size, num_props), dim=-1)
    return features, mask, positive_weights, negative_weights, selection


def test_positive_cpca_basis_is_persistent_and_differentiable():
    batch_size, num_props, dim = 4, 3, 8
    inputs = make_event_inputs(batch_size, num_props, dim=dim)
    features = inputs[0]
    module = LowRankEventDisentangler(
        hidden_size=dim, rank=3, cpca_alpha=1.0,
        covariance_ema=0.5, normalize_covariance=True)
    module.train()
    positive, negative, event = module(*inputs, update_subspace=True)

    assert positive.shape == (batch_size * num_props, dim)
    assert negative.shape == (batch_size * num_props, 2, dim)
    assert event.shape == (batch_size * num_props, dim)
    assert module.subspace_updates.item() == 1
    contrastive = (
        module.running_event_cov - module.cpca_alpha
        * module.running_background_cov)
    active = module.event_basis.norm(dim=0) > 0
    rayleigh = torch.diagonal(
        module.event_basis.t().mm(contrastive).mm(module.event_basis))
    assert torch.all(rayleigh[active] > 0)

    event.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    state = module.state_dict()
    assert "event_basis" in state
    assert "running_event_cov" in state
    assert "running_background_cov" in state

    module.eval()
    updates = module.subspace_updates.item()
    with torch.no_grad():
        module(*make_event_inputs(batch_size, num_props, dim=dim),
               update_subspace=False)
    assert module.subspace_updates.item() == updates


def call_event_loss(width, center=None, **overrides):
    torch.manual_seed(5)
    batch_size, num_props, words, vocab, dim = 2, 3, 4, 9, 6
    if center is None:
        center = torch.tensor([[0.2, 0.5, 0.8], [0.2, 0.5, 0.8]])
    logits = torch.randn(batch_size * num_props, words, vocab)
    event = torch.randn(batch_size * num_props, dim, requires_grad=True)
    loss, metrics = event_disentanglement_loss(
        words_logit=logits,
        words_id=torch.randint(0, vocab, (batch_size, words)),
        words_mask=torch.ones(batch_size, words, dtype=torch.uint8),
        num_props=num_props,
        event_pos_feat=torch.randn(batch_size * num_props, dim),
        event_neg_feat=torch.randn(batch_size * num_props, 2, dim),
        event_vector=event,
        event_text_feat=torch.randn(batch_size, dim),
        event_selection_weights=torch.full(
            (batch_size, num_props), 1.0 / num_props),
        center=center,
        width=width,
        event_schedule=1.0,
        event_alpha=0.1,
        event_margin=0.2,
        event_text_weight=1.0,
        event_min_context=0.15,
        event_context_weight=1.0,
        event_max_overlap=0.7,
        event_overlap_weight=0.5,
        **overrides
    )
    return loss, metrics, event


def test_boundary_aware_loss_penalizes_full_video_shortcut():
    narrow_width = torch.full((6,), 0.4)
    wide_width = torch.full((6,), 0.95, requires_grad=True)
    narrow_loss, narrow_metrics, _ = call_event_loss(narrow_width)
    wide_loss, wide_metrics, wide_event = call_event_loss(wide_width)

    assert wide_metrics["event_context_loss"] > narrow_metrics["event_context_loss"]
    assert wide_metrics["event_context_violation"] > narrow_metrics[
        "event_context_violation"]
    assert wide_loss.ndim == 0
    wide_loss.backward()
    assert wide_event.grad is not None
    # Positive width gradients mean gradient descent will shrink the shortcut.
    assert wide_width.grad is not None
    assert wide_width.grad.mean() > 0


def test_warmup_keeps_boundary_regularization_active():
    loss, metrics, _ = call_event_loss(torch.full((6,), 0.8))
    zero_loss, zero_metrics = event_disentanglement_loss(
        words_logit=torch.randn(6, 4, 9),
        words_id=torch.randint(0, 9, (2, 4)),
        words_mask=torch.ones(2, 4, dtype=torch.uint8),
        num_props=3,
        event_pos_feat=torch.randn(6, 6),
        event_neg_feat=torch.randn(6, 2, 6),
        event_vector=torch.randn(6, 6),
        event_text_feat=torch.randn(2, 6),
        event_selection_weights=torch.full((2, 3), 1 / 3),
        center=torch.full((6,), 0.5),
        width=torch.full((6,), 0.8),
        event_schedule=0.0,
        event_alpha=0.1,
    )
    assert loss.item() >= 0
    assert zero_loss.item() > 0
    assert zero_metrics["event_schedule"] == 0
    assert zero_metrics["event_boundary_loss"] > 0


def test_separation_weight_can_be_ablated_independently():
    width = torch.full((6,), 0.8)
    full_loss, full_metrics, _ = call_event_loss(width)
    no_sep_loss, no_sep_metrics, _ = call_event_loss(
        width, event_sep_weight=0.0)
    expected_delta = 0.1 * full_metrics["event_sep_loss"]
    assert torch.isclose(
        full_loss - no_sep_loss,
        torch.tensor(expected_delta), atol=1e-6)
    assert no_sep_metrics["event_sep_loss"] == full_metrics["event_sep_loss"]
