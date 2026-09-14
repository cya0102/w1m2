import random

import numpy as np
import torch

from models.cpl import CPL
from runners.main_runner import MainRunner
from utils import isolated_rng, stable_sample_seed


def small_config(qcec_enabled=False):
    config = {
        'frames_input_size': 4,
        'words_input_size': 3,
        'hidden_size': 8,
        'vocab_size': 11,
        'use_negative': True,
        'num_props': 3,
        'sigma': 9,
        'gamma': 0,
        'dropout': 0.0,
        'max_epoch': 4,
        'proposal_generator': {
            'type': 'gaussian_mixture',
            'max_components': 3,
            'component_sigma': 4.0,
            'importance_temperature': 1.0,
            'boundary_mode': 'weighted',
            'boundary_shrink': 0.0,
        },
        'event_disentanglement': {'enabled': False},
        'DualTransformer': {
            'd_model': 8,
            'num_heads': 2,
            'num_decoder_layers1': 1,
            'num_decoder_layers2': 1,
            'dropout': 0.0,
        },
    }
    if qcec_enabled:
        config['qcec'] = {
            'enabled': True,
            'num_clusters': 4,
            'attention_dim': 8,
            'init_seed': 20260911,
            'freeze_backbone_epochs': 0,
            'use_slot_prior': False,
        }
    return config


class FakeScheduler:
    def __init__(self):
        self.updates = []

    def step_update(self, value):
        self.updates.append(value)
        return 0.0


def test_public_initial_state_keeps_disabled_and_qcec_public_weights_equal(tmp_path):
    torch.manual_seed(8)
    baseline = CPL(small_config(False))
    state_path = tmp_path / 'public.pt'
    torch.save({'model_parameters': baseline.state_dict()}, state_path)

    torch.manual_seed(8)
    target = CPL(small_config(True))
    runner = MainRunner.__new__(MainRunner)
    runner.args = {
        'seed': 8,
        'dataset': {},
        'train': {},
        'tag': 'test',
    }
    runner.model = target
    runner.model_saved_path = str(tmp_path / 'run')
    runner.lr_scheduler = FakeScheduler()
    runner.selection_strategy = 'nll'
    runner.selection_temperature = 0.1
    runner.eval_seed = 100
    runner.initialization_source = 'random'
    runner.initialization_kind = 'random'
    runner.initialization_valid = False
    runner._load_public_initial_state(str(state_path))

    for name, value in baseline.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)

    batch_size, frames, words = 2, 20, 6
    common = {
        'frames_feat': torch.randn(batch_size, frames, 4),
        'frames_len': torch.tensor([frames, frames - 2]),
        'words_id': torch.randint(0, 11, (batch_size, words)),
        'words_feat': torch.randn(batch_size, words + 1, 3),
        'words_len': torch.tensor([4, 5]),
        'weights': torch.full((batch_size, words), 1 / words),
        'eval_mask_seeds': torch.tensor([11, 22]),
        'epoch': 1,
    }
    qcec_inputs = {
        'query_role_mask': torch.ones(batch_size, 3, words, dtype=torch.bool),
        'query_role_valid': torch.ones(batch_size, 3, dtype=torch.bool),
        'qcec_cluster_ids': (torch.arange(frames).view(1, frames) * 4
                             // frames).expand(batch_size, -1),
        'qcec_cluster_bounds': torch.tensor([
            [[0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
        ]).expand(batch_size, -1, -1),
        'qcec_cluster_mask': torch.ones(batch_size, 4, dtype=torch.bool),
    }
    baseline.eval()
    target.eval()
    with torch.no_grad():
        baseline_output = baseline(**{
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in common.items()})
        target_output = target(**{
            **{
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in common.items()},
            **qcec_inputs,
        })
    for key in ('center', 'width', 'gauss_weight', 'words_logit'):
        assert torch.allclose(
            baseline_output[key], target_output[key], atol=1e-6, rtol=1e-5)


def test_qcec_freeze_rejects_uninitialized_scratch():
    runner = MainRunner.__new__(MainRunner)
    runner.model = type('Model', (), {
        'use_qcec': True,
        'qcec_freeze_backbone_epochs': 1,
    })()
    runner.initialization_valid = False
    runner.initialization_kind = 'random'
    runner.start_epoch = 0
    try:
        runner._validate_training_protocol()
    except ValueError as error:
        assert 'freeze_backbone_epochs' in str(error)
    else:
        raise AssertionError('random QCEC-only scratch start was accepted')


def test_eval_mask_seed_is_stable_and_sample_specific():
    model = CPL(small_config(False))
    # _mask_words receives features after the model's word projection.
    words_feat = torch.randn(2, 7, 8)
    words_len = torch.tensor([5, 4])
    weights = torch.zeros(2, 6)
    weights[0, :5] = 0.2
    weights[1, :4] = 0.25
    seeds = torch.tensor([
        stable_sample_seed('v1', 'a person runs'),
        stable_sample_seed('v2', 'a person sits'),
    ])
    first = model._mask_words(
        words_feat, words_len, weights=weights, mask_seeds=seeds)[1]
    second = model._mask_words(
        words_feat, words_len, weights=weights, mask_seeds=seeds)[1]
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_isolated_rng_restores_all_streams():
    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    with isolated_rng(123):
        random.random()
        np.random.rand()
        torch.rand(1)
    restored_values = (random.random(), np.random.rand(), torch.rand(1))

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    expected_values = (random.random(), np.random.rand(), torch.rand(1))
    assert restored_values[0] == expected_values[0]
    assert restored_values[1] == expected_values[1]
    assert torch.equal(restored_values[2], expected_values[2])
