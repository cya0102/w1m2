import numpy as np
import torch

from models.modules.event_boundary_refinement import EventBoundaryRefiner
from runners.main_runner import select_minimal_sufficient_candidates
from datasets.base import build_collate_data


def test_inward_geometry_and_duplicate_priority():
    refiner = EventBoundaryRefiner(
        min_boundary_margin_clips=1,
        min_candidate_width=0.02,
        min_retained_ratio=0.25)
    center = torch.tensor([[0.5, 0.5]])
    width = torch.tensor([[0.8, 0.2]])
    positions = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
    scores = torch.tensor([[0.2, 0.9, 0.8, 0.1]])
    mask = torch.ones_like(positions, dtype=torch.bool)
    starts, ends, valid, _ = refiner.build_candidates(
        center, width, positions, scores, mask)

    original_start = starts[..., :1]
    original_end = ends[..., :1]
    assert torch.all(starts[valid] >= original_start.expand_as(starts)[valid])
    assert torch.all(ends[valid] <= original_end.expand_as(ends)[valid])
    assert torch.all(valid[..., 0])
    assert torch.allclose(starts[0, 0, 1], torch.tensor(0.2))
    assert torch.allclose(starts[0, 0, 2], torch.tensor(0.4))
    assert torch.allclose(ends[0, 0, 3], torch.tensor(0.8))
    assert torch.all(valid[0, 0, 1:])
    assert not torch.any(valid[0, 1, 1:])

    duplicate_positions = torch.tensor([[0.2, 0.8]])
    duplicate_scores = torch.tensor([[0.5, 0.5]])
    _, _, duplicate_valid, _ = refiner.build_candidates(
        torch.tensor([[0.5]]), torch.tensor([[0.8]]), duplicate_positions,
        duplicate_scores, torch.ones_like(duplicate_positions, dtype=torch.bool))
    assert duplicate_valid[0, 0].tolist() == [True, True, False, True,
                                               False, True, False]


def test_soft_mask_preserves_original_and_normalizes_trimmed_masks():
    refiner = EventBoundaryRefiner(soft_window_temperature=0.01)
    center = torch.tensor([[0.5]])
    width = torch.tensor([[0.8]])
    positions = torch.tensor([[0.25, 0.75]])
    scores = torch.tensor([[0.5, 0.6]])
    valid_boundary = torch.ones_like(positions, dtype=torch.bool)
    starts, ends, valid, _ = refiner.build_candidates(
        center, width, positions, scores, valid_boundary)
    original = torch.linspace(0.1, 1.0, 50).view(1, 1, 50)
    masks, mask_valid = refiner.build_candidate_masks_with_validity(
        original, starts, ends, valid)
    assert torch.equal(masks[..., 0, :], original)
    trimmed_max = masks[..., 1:, :].amax(dim=-1)
    assert torch.allclose(trimmed_max[mask_valid[..., 1:]],
                          torch.ones_like(trimmed_max[mask_valid[..., 1:]]))
    assert torch.isfinite(masks).all()
    assert torch.all(mask_valid[..., 0])
    assert torch.all(masks[..., 1, :].amax(dim=-1) <= 1)


def test_selector_uses_width_then_nll_and_falls_back_to_original():
    starts = torch.tensor([[[0.1, 0.12, 0.1, 0.1, 0.1, 0.2, 0.2]]])
    ends = torch.tensor([[[0.9, 0.85, 0.84, 0.86, 0.86, 0.75, 0.75]]])
    nll = torch.tensor([[[1.0, 1.01, 1.20, 1.01, 1.01, 1.02, 1.03]]])
    valid = torch.ones_like(nll, dtype=torch.bool)
    props, selected_nll, selected = select_minimal_sufficient_candidates(
        starts, ends, nll, valid, max_nll_increase=0.02)
    assert selected.item() == 5
    assert np.allclose(props.numpy(), [[[0.2, 0.75]]])
    assert torch.allclose(selected_nll, torch.tensor([[1.02]]))

    nll[..., 5] = 1.5
    _, _, selected = select_minimal_sufficient_candidates(
        starts, ends, nll, valid, max_nll_increase=0.02)
    assert selected.item() == 1


def test_collate_pads_boundaries_and_keeps_all_empty_batches_nonzero_width():
    collate = build_collate_data(4, 3, 2, 2)
    common = {
        'frames_feat': np.ones((4, 2), dtype=np.float32),
        'words_feat': np.ones((2, 2), dtype=np.float32),
        'words_id': [1],
        'weights': [1.0],
        'raw': ['v', 1.0, [0.0, 1.0], 'word'],
    }
    samples = [
        dict(common, event_boundary_positions=np.asarray([.2, .5], dtype=np.float32),
             event_boundary_scores=np.asarray([.4, .9], dtype=np.float32)),
        dict(common, event_boundary_positions=np.empty(0, dtype=np.float32),
             event_boundary_scores=np.empty(0, dtype=np.float32)),
    ]
    batch = collate(samples)
    assert batch['net_input']['event_boundary_positions'].shape == (2, 2)
    assert batch['net_input']['event_boundary_mask'].tolist() == [[True, True],
                                                                   [False, False]]

    empty_batch = collate([dict(common), dict(common)])
    assert empty_batch['net_input']['event_boundary_positions'].shape == (2, 1)
    assert not torch.any(empty_batch['net_input']['event_boundary_mask'])
