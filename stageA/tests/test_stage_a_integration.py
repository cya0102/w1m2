from unittest.mock import patch
from copy import deepcopy

import numpy as np
import torch

from models.cpl import CPL


def _config(proposal_type):
    return {
        'frames_input_size': 4,
        'words_input_size': 3,
        'hidden_size': 8,
        'vocab_size': 11,
        'use_negative': False,
        'num_props': 3,
        'sigma': 9,
        'gamma': 0,
        'dropout': 0.0,
        'max_epoch': 30,
        'proposal_generator': {
            'type': proposal_type,
            'max_components': 3,
            'component_sigma': 4.0,
            'importance_temperature': 1.0,
            'boundary_mode': 'outer',
            'boundary_shrink': 0.0,
        },
        'event_boundary_refinement': {
            'enabled': True,
            'decode_chunk_size': 1,
            'min_candidate_width': 0.02,
            'min_retained_ratio': 0.25,
            'soft_window_temperature': 0.01,
        },
        'DualTransformer': {
            'd_model': 8,
            'num_heads': 2,
            'num_decoder_layers1': 1,
            'num_decoder_layers2': 1,
            'dropout': 0.0,
        },
    }


def _inputs():
    return {
        'frames_feat': torch.randn(2, 20, 4),
        'frames_len': torch.tensor([20, 18]),
        'words_id': torch.randint(0, 11, (2, 20)),
        'words_feat': torch.randn(2, 21, 3),
        'words_len': torch.tensor([5, 6]),
        'weights': torch.tensor([
            [.2] * 5 + [0.] * 15,
            [1 / 6] * 6 + [0.] * 14]),
        'epoch': 0,
        'run_stage_a': True,
        'event_boundary_positions': torch.tensor([
            [.2, .4, .7], [.1, .5, 0.]]),
        'event_boundary_scores': torch.tensor([
            [.2, .8, .3], [.4, .5, 0.]]),
        'event_boundary_mask': torch.tensor([
            [True, True, True], [True, True, False]]),
    }


def test_stage_a_disabled_has_no_new_state_tensors_and_old_checkpoint_loads():
    disabled = _config('single_gaussian')
    disabled['event_boundary_refinement']['enabled'] = False
    model = CPL(disabled)
    assert not any(name.startswith('stage_a') for name in model.state_dict())
    clone = CPL(disabled)
    result = clone.load_state_dict(model.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_stage_a_runs_for_single_and_mixture_generators():
    with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
        for proposal_type in ('single_gaussian', 'gaussian_mixture'):
            torch.manual_seed(4)
            model = CPL(_config(proposal_type)).eval()
            assert not any(name.startswith('stage_a')
                           for name in model.state_dict())
            output = model(**_inputs())
            assert output['stage_a_candidate_start'].shape == (2, 3, 7)
            assert output['stage_a_candidate_end'].shape == (2, 3, 7)
            assert output['stage_a_candidate_valid'].shape == (2, 3, 7)
            assert output['stage_a_candidate_nll'].shape == (2, 3, 7)
            assert torch.isfinite(
                output['stage_a_candidate_nll'][
                    output['stage_a_candidate_valid']]).all()
            assert torch.equal(
                output['stage_a_candidate_nll'][..., 0],
                model.proposal_reconstruction_nll(
                    output['words_logit'], output['words_id'],
                    output['words_mask']))


def test_candidate_chunk_size_does_not_change_nll_and_mask_words_runs_once():
    config_small = _config('single_gaussian')
    config_large = deepcopy(config_small)
    config_large['event_boundary_refinement']['decode_chunk_size'] = 64
    torch.manual_seed(13)
    small = CPL(config_small).eval()
    large = CPL(config_large).eval()
    large.load_state_dict(small.state_dict(), strict=True)
    small_calls = []
    large_calls = []
    original_small = small._mask_words
    original_large = large._mask_words
    small._mask_words = lambda *args, **kwargs: (
        small_calls.append(1) or original_small(*args, **kwargs))
    large._mask_words = lambda *args, **kwargs: (
        large_calls.append(1) or original_large(*args, **kwargs))
    inputs = _inputs()
    with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
        torch.manual_seed(14)
        np.random.seed(15)
        small_output = small(**{
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in inputs.items()})
        torch.manual_seed(14)
        np.random.seed(15)
        large_output = large(**{
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in inputs.items()})
    assert small_calls == [1]
    assert large_calls == [1]
    assert torch.allclose(
        small_output['stage_a_candidate_nll'],
        large_output['stage_a_candidate_nll'], atol=1e-6, rtol=0)
