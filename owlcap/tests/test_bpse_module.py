import torch

from models.modules.bpse import (
    BPSEScorer,
    SalientEventTokenizer,
    build_proposal_groups,
    build_soft_box_masks,
)


def _inputs(batch=2, frames=32, words=6, hidden=16, props=5, units=12):
    unit_mask = torch.zeros(batch, units, words)
    unit_mask[:, 0, :words // 2] = 1
    unit_mask[:, 1, words // 2:] = 1
    valid_units = torch.zeros(batch, units, dtype=torch.bool)
    valid_units[:, :2] = True
    unit_weights = torch.zeros(batch, units)
    unit_weights[:, :2] = 0.5
    spans = torch.tensor([[
        [.05, .25], [.15, .40], [.30, .50], [.50, .75], [.0, 1.0]
    ] for _ in range(batch)])
    return (
        torch.randn(batch, frames, hidden), torch.randn(batch, frames, hidden),
        torch.ones(batch, frames, dtype=torch.bool),
        torch.randn(batch, words, hidden), torch.ones(batch, words, dtype=torch.bool),
        unit_mask, valid_units, unit_weights, spans)


def test_soft_box_and_groups_shapes_and_bounds():
    spans = torch.tensor([[[.1, .4], [.0, 1.0]]]).expand(2, -1, -1)
    masks = build_soft_box_masks(spans, torch.ones(2, 32, dtype=torch.bool))
    assert masks.shape == (2, 2, 32)
    assert masks.min() >= 0 and masks.max() <= 1
    groups, valid = build_proposal_groups(spans, offsets=(.05,))
    assert groups.shape == (2, 2, 9, 2)
    assert valid.shape == (2, 2, 9)


def test_event_tokenizer_masks_padding_and_static_inputs():
    tokenizer = SalientEventTokenizer(num_event_tokens=8)
    states = torch.zeros(2, 9, 4)
    frame_mask = torch.ones(2, 9, dtype=torch.bool)
    frame_mask[1, 4:] = False
    result = tokenizer(states, frame_mask)
    assert result['event_tokens'].shape == (2, 8, 4)
    assert torch.isfinite(result['event_tokens']).all()
    assert result['event_pool_weights'][1, :, 4:].sum() == 0
    valid = result['event_valid_mask']
    assert torch.allclose(
        result['event_pool_weights'].sum(-1)[valid],
        torch.ones_like(result['event_pool_weights'].sum(-1)[valid]),
        atol=1e-5)


def test_bpse_forward_is_finite_and_has_train_groups():
    args = _inputs()
    scorer = BPSEScorer(16, max_query_units=12, num_event_tokens=8,
                        quality_hidden_size=12, quality_dropout=0)
    result = scorer(*args, build_perturbations=True, perturb_offsets=(.05,))
    assert result['quality_logits'].shape == (2, 5)
    assert result['group_quality_logits'].shape == (2, 5, 9)
    assert result['unit_coverage'].shape == (2, 5, 12)
    for value in result.values():
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all()


def test_span_mass_increases_when_interval_expands():
    frame_mask = torch.ones(1, 64, dtype=torch.bool)
    narrow = build_soft_box_masks(torch.tensor([[[.3, .4]]]), frame_mask)
    wide = build_soft_box_masks(torch.tensor([[[.2, .5]]]), frame_mask)
    assert wide.sum() >= narrow.sum()
