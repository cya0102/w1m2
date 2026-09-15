#!/usr/bin/env python
"""Compare disabled and zero-residual QSTG forward outputs on one batch."""

import argparse
import copy
import pickle
import numpy as np
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import ActivityNet, CharadesSTA
from models.cpl import CPL
from utils import load_json


def _make_dataset(config):
    dataset_args = config['dataset']
    with open(dataset_args['vocab_path'], 'rb') as handle:
        vocab = pickle.load(handle)
    cls = {'ActivityNet': ActivityNet,
           'CharadesSTA': CharadesSTA}[dataset_args['dataset']]
    return cls(dataset_args['train_data'], vocab, dataset_args)


def _load(model, path, strict):
    state = torch.load(path, map_location='cpu')
    parameters = state.get('model_parameters', state)
    result = model.load_state_dict(parameters, strict=strict)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--config-path', dest='config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--mask-mode', choices=['legacy', 'deterministic', 'both'],
                        default='both')
    args = parser.parse_args()
    config = load_json(args.config)
    dataset = _make_dataset(config)
    samples = [dataset[index] for index in range(min(args.batch_size, len(dataset)))]
    batch = dataset.collate_data(samples)

    enabled_config = copy.deepcopy(config['model']['config'])
    enabled_config['vocab_size'] = dataset.vocab_size
    enabled_config['max_epoch'] = config['train']['max_num_epochs']
    enabled_config.setdefault('qstg', {})['enabled'] = True
    enabled_config['qstg']['proposal_prior_scale'] = 0.0
    enabled_config['qstg']['importance_bias_scale'] = 0.0
    enabled_config['qstg']['component_residual_enabled'] = False
    enabled_config['qstg']['allow_global_fallback'] = False
    disabled_config = copy.deepcopy(enabled_config)
    disabled_config.pop('qstg', None)

    # The copied baseline model uses the project's historical CUDA calls; an
    # explicit patch here lets parity also run on a CPU-only host.
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda tensor, *unused, **ignored: tensor
    try:
        disabled = CPL(disabled_config)
        enabled = CPL(enabled_config)
        _load(disabled, args.checkpoint, strict=True)
        result = _load(enabled, args.checkpoint, strict=False)
        missing = [name for name in result.missing_keys
                   if not name.startswith('qstg.')]
        if missing or result.unexpected_keys:
            raise RuntimeError('unexpected parity keys: missing={}, unexpected={}'
                               .format(missing, result.unexpected_keys))
        disabled.eval()
        enabled.eval()
        net_input = batch['net_input']
        modes = ('legacy', 'deterministic') if args.mask_mode == 'both' \
            else (args.mask_mode,)
        mode_errors = {}
        for mode in modes:
            kwargs = dict(epoch=0, mask_mode=mode, eval_seed=0)
            # Legacy masking consumes the global NumPy RNG. Restore its state
            # so both models receive exactly the same random mask.
            np_state = np.random.get_state()
            disabled_output = disabled(**kwargs, **{
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in net_input.items()})
            np.random.set_state(np_state)
            enabled_output = enabled(**kwargs, **{
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in net_input.items()})

            values = ('center', 'width', 'gauss_weight', 'words_logit')
            errors = {}
            for name in values:
                left = disabled_output[name].float()
                right = enabled_output[name].float()
                errors[name] = float((left - right).abs().max().item())
            mode_errors[mode] = errors
    finally:
        torch.Tensor.cuda = original_cuda

    errors = mode_errors
    maximum = max(max(values.values()) for values in mode_errors.values())
    print('parity errors:', errors)
    print('max_abs_err:', maximum)
    if maximum > 1e-6:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
