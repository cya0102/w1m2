#!/usr/bin/env python
"""Inspect phrase-to-node bindings and proposal quality without GT targets."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.main_runner import MainRunner, move_to_cuda
from utils import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--config-path', dest='config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--limit', type=int, default=4)
    args = parser.parse_args()
    config = load_json(args.config)
    config['selection_strategy'] = 'qstg'
    config.setdefault('eval_mask_mode', 'deterministic')
    runner = MainRunner(config)
    runner._load_model_parameters(args.checkpoint)
    runner.model.eval()
    with torch.no_grad():
        for bid, batch in enumerate(runner.val_loader or runner.test_loader):
            if bid >= args.limit:
                break
            output = runner.model(
                epoch=0, mask_mode=config['eval_mask_mode'],
                eval_seed=config.get('seed', 0),
                **move_to_cuda(batch['net_input']))
            binding = output['binding_prob']
            bounds = output['node_bounds']
            top = binding.topk(min(5, binding.size(-1)), dim=-1).indices
            print('batch', bid + 1)
            for sample in range(binding.size(0)):
                print(' sample', sample, 'gate_mean',
                      float(output['node_gate'][sample].mean()))
                for phrase in range(binding.size(1)):
                    nodes = top[sample, phrase].tolist()
                    print('  phrase', phrase, 'nodes', [
                        (node, [round(float(value), 4) for value in bounds[sample, node]])
                        for node in nodes])
                print('  proposal C/X/R/H:', [
                    [round(float(value), 4) for value in output[name][sample]]
                    for name in ('proposal_coverage', 'proposal_exclusivity',
                                 'proposal_relation', 'proposal_connectivity')])


if __name__ == '__main__':
    main()
