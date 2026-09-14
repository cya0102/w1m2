import torch

from models.loss import bpse_loss, ivc_loss, rec_loss


def _group_tensors():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[0.9, 0.4, 0.1, 0.1]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    return logits, target, valid


def test_pairwise_loss_has_gradient_and_targets_are_detached():
    logits, target, valid = _group_tensors()
    words = torch.zeros(1, 1, 3, requires_grad=True)
    loss, metrics = bpse_loss(
        words_logit=words,
        bpse_quality_logits=torch.zeros(1, 1, requires_grad=True),
        bpse_completeness=torch.ones(1, 1),
        bpse_purity=torch.ones(1, 1),
        bpse_equivalence=torch.ones(1, 1),
        bpse_span_completeness=torch.ones(1, 1),
        bpse_span_equivalence=torch.ones(1, 1),
        bpse_group_quality_logits=logits,
        bpse_group_equivalence=target,
        bpse_group_valid_mask=valid,
        bpse_rank_weight=1.0, bpse_abs_weight=.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert metrics['bpse_valid_pairs'] > 0


def test_no_pair_returns_graph_connected_zero():
    logits = torch.zeros(1, 1, 3, requires_grad=True)
    target = torch.ones(1, 1, 3) * .5
    words = torch.zeros(1, 1, 3, requires_grad=True)
    loss, metrics = bpse_loss(
        words_logit=words,
        bpse_quality_logits=torch.zeros(1, 1, requires_grad=True),
        bpse_completeness=torch.ones(1, 1), bpse_purity=torch.ones(1, 1),
        bpse_equivalence=torch.ones(1, 1),
        bpse_span_completeness=torch.ones(1, 1),
        bpse_group_quality_logits=logits,
        bpse_group_equivalence=target,
        bpse_group_valid_mask=torch.ones_like(target, dtype=torch.bool),
        bpse_pair_margin=.5)
    assert metrics['bpse_valid_pairs'] == 0
    loss.backward()
    assert logits.grad is not None


def test_weighted_rec_loss_is_finite_and_legacy_none_path_exists():
    words_logit = torch.randn(4, 3, 5, requires_grad=True)
    words_id = torch.randint(0, 5, (2, 3))
    words_mask = torch.ones(2, 3)
    weighted, _ = rec_loss(
        words_logit, words_id, words_mask, num_props=2,
        proposal_weights=torch.tensor([[.8, .2], [.2, .8]]))
    legacy, _ = rec_loss(words_logit, words_id, words_mask, num_props=2)
    assert torch.isfinite(weighted) and torch.isfinite(legacy)


def test_weighted_ivc_path_is_finite():
    words_logit = torch.randn(4, 3, 5, requires_grad=True)
    neg1 = torch.randn_like(words_logit)
    neg2 = torch.randn_like(words_logit)
    ids = torch.randint(0, 5, (2, 3))
    mask = torch.ones(2, 3)
    loss, _ = ivc_loss(
        words_logit, ids, mask, num_props=2,
        neg_words_logit_1=neg1, neg_words_logit_2=neg2,
        proposal_weights=torch.tensor([[.8, .2], [.2, .8]]),
        gauss_weight=torch.rand(4, 8), margin_1=.1, margin_2=.2,
        **{'lambda': .1, 'alpha_1': 1., 'alpha_2': 0.})
    assert torch.isfinite(loss)
