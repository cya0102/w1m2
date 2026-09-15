from types import SimpleNamespace
from unittest.mock import patch

import torch

from runners.main_runner import MainRunner
from tests.test_qstg_integration import _batch, _config


def _fake_build_dataset(runner):
    runner.train_set = SimpleNamespace(vocab_size=11)
    runner.test_set = SimpleNamespace(vocab_size=11)
    runner.val_set = None

    def make_batch():
        net_input = _batch()
        net_input.pop('epoch')
        return {
            'net_input': net_input,
            'raw': [
                ('video-a', 10.0, [1.0, 5.0]),
                ('video-b', 12.0, [3.0, 7.0]),
            ],
        }

    runner.train_loader = [make_batch(), make_batch()]
    runner.test_loader = [make_batch()]
    runner.val_loader = None


def test_runner_qstg_train_and_eval_smoke():
    model_config = _config('gaussian_mixture')
    args = {
        'dataset': {'dataset': 'ActivityNet'},
        'train': {
            'batch_size': 2,
            'max_num_epochs': 1,
            'model_saved_path': '/tmp/qstg-runner-smoke',
            'optimizer': {
                'lr': 2e-4,
                'weight_decay': 0,
                'warmup_updates': 4,
                'warmup_init_lr': 1e-7,
            },
        },
        'model': {'name': 'CPL', 'config': model_config},
        'loss': {
            'margin_1': 0.1,
            'margin_2': 0.15,
            'lambda': 0.125,
            'alpha_1': 0.0,
            'alpha_2': 0.0,
            'mixture_pull_weight': 0.0,
            'mixture_intra_push_weight': 0.0,
            'mixture_inter_push_weight': 0.0,
            'qstg_total_weight': 1.0,
            'qstg_mc_weight': 0.05,
            'qstg_proposal_weight': 0.05,
            'qstg_gate_weight': 0.01,
        },
        'selection_strategy': 'qstg',
        'trainable_scope': 'qstg_only',
        'qstg_stage': 'A',
    }

    with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
        with patch.object(MainRunner, '_build_dataset', _fake_build_dataset):
            runner = MainRunner(args)
            runner._train_one_epoch(1)
            results = runner.eval(loader=runner.test_loader, split='Smoke')

    assert 'R@1,mIoU' in results
    assert 'R@3,mIoU' in results
    assert all(torch.isfinite(parameter).all() for parameter in
               runner.model.qstg.parameters())
