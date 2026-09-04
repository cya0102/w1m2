import torch
from unittest.mock import patch

from models.modules.event_boundary_refinement import EventBoundaryRefiner
from models.cpl import CPL, deterministic_eval_word_mask
from tests.test_stage_a_integration import _config, _inputs


def test_raw_trim_and_shell_are_complements_before_normalization():
    refiner = EventBoundaryRefiner(soft_window_temperature=0.02)
    candidate_start = torch.tensor([[[0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]]])
    candidate_end = torch.tensor([[[0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8]]])
    original = torch.linspace(0.2, 1.0, 50).view(1, 1, 50)
    raw_trim, raw_shell = refiner.build_raw_trim_shell_masks(
        original, candidate_start, candidate_end)
    assert torch.allclose(
        raw_trim + raw_shell, original.unsqueeze(2), atol=1e-6, rtol=0)

    valid = torch.ones(candidate_start.shape, dtype=torch.bool)
    trim, shell, trim_valid, shell_valid = refiner._build_trim_shell_masks(
        original, candidate_start, candidate_end, valid)
    assert torch.equal(trim[..., 0, :], original)
    assert not shell_valid[..., 0].item()
    assert torch.isfinite(trim).all() and torch.isfinite(shell).all()
    assert torch.allclose(trim[..., 1:, :].amax(-1)[trim_valid[..., 1:]],
                          torch.ones_like(trim[..., 1:, :].amax(-1)[trim_valid[..., 1:]]))
    assert torch.allclose(shell[..., 1:, :].amax(-1)[shell_valid[..., 1:]],
                          torch.ones_like(shell[..., 1:, :].amax(-1)[shell_valid[..., 1:]]))


def test_boundary_confidence_and_boundary_scores_are_exportable():
    refiner = EventBoundaryRefiner()
    result = refiner.build_candidates(
        torch.tensor([[0.5]]), torch.tensor([[0.8]]),
        torch.tensor([[0.2, 0.4, 0.7]]), torch.tensor([[0.1, 0.8, 0.3]]),
        torch.tensor([[True, True, True]]), return_boundary_confidence=True,
        return_boundary_scores=True)
    assert len(result) == 7
    _, _, valid, _, confidence, left_score, right_score = result
    assert confidence.shape == (1, 1, 7)
    assert torch.all(confidence[valid] >= 0)
    assert torch.all(confidence[valid] <= 1)
    assert torch.allclose(left_score[0, 0, 2], torch.tensor(0.8))
    assert torch.allclose(right_score[0, 0, 3], torch.tensor(0.3))


def test_score_shell_false_does_not_run_shell_decoder():
    config = _config('single_gaussian')
    config['event_boundary_refinement']['stage_a5'] = {
        'enabled': True, 'score_shell': False,
    }
    model = CPL(config).eval()
    inputs = _inputs()
    inputs['run_stage_a5'] = True
    inputs['eval_word_mask'] = deterministic_eval_word_mask(
        ['v:0', 'v:1'], inputs['words_len'], inputs['words_feat'].size(1) - 1,
        8, weights=inputs['weights'])
    calls = []
    original = model.score_stage_a_candidates
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    with patch.object(model, 'score_stage_a_candidates', counted):
        output = model(**inputs)
    assert calls == [1]
    assert torch.isinf(output['stage_a_candidate_shell_nll']).all()
