"""Cache fixed-checkpoint candidates for ActivityNet recovery diagnostics.

The cache is generated from one forward pass per batch under the same
deterministic evaluation protocol used by ``MainRunner.eval``.  It contains
enough information to compare candidate geometry, NLL ranking and QCEC
relevance without rerunning the model or changing the word mask.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.loss import cal_nll_loss  # noqa: E402
from runners.main_runner import MainRunner, move_to_cuda  # noqa: E402
from utils import isolated_rng, load_json, sha256_file  # noqa: E402


def install_cpu_cuda_shim():
    """Let the CUDA-only model run on a CPU smoke-test host."""
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self


def build_runner(config_path, seed, eval_seed, batch_size=None):
    config_path = Path(config_path).resolve()
    args = load_json(str(config_path))
    if batch_size is not None:
        args['train']['batch_size'] = int(batch_size)
    args['config_path'] = str(config_path)
    args['seed'] = int(seed)
    args['eval_seed'] = int(eval_seed)
    args['tag'] = 'recovery_cache'
    args['run_timestamp'] = 'diagnostic_cache'
    args['selection_strategy'] = 'nll'
    args['selection_temperature'] = 0.1
    args['select_on_val'] = True
    args['final_test'] = False
    return MainRunner(args)


def _as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_cache(runner, loader, epoch=0, max_batches=None):
    runner.model.eval()
    arrays = {
        'video_ids': [],
        'sample_keys': [],
        'durations': [],
        'ground_truth': [],
        'nll_score': [],
        'event_score': [],
        'raw_center': [],
        'raw_width': [],
        'eval_center': [],
        'eval_width': [],
    }
    optional_names = (
        'mixture_context_center', 'mixture_context_width',
        'mixture_component_centers', 'mixture_component_widths',
        'mixture_component_weights', 'mixture_component_importance',
        'mixture_component_valid_mask', 'qcec_cluster_bounds',
        'qcec_cluster_mask', 'qcec_cluster_relevance',
        'qcec_transition_positions', 'qcec_transition_mask',
        'qcec_transition_score', 'qcec_barrier_lr', 'qcec_barrier_rl',
    )
    optional = {name: [] for name in optional_names}

    with isolated_rng(runner.eval_seed):
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, 1):
                if max_batches is not None and batch_index > max_batches:
                    break
                raw = batch['raw']
                durations = np.asarray([item[1] for item in raw], dtype=np.float32)
                gt = np.asarray([item[2] for item in raw], dtype=np.float32)
                net_input = move_to_cuda(batch['net_input'])
                output = runner.model(epoch=epoch, **net_input)
                batch_size = len(raw)
                num_props = runner.model.num_props
                words_mask = output['words_mask'].unsqueeze(1).expand(
                    batch_size, num_props, -1).contiguous().view(
                        batch_size * num_props, -1)
                words_id = output['words_id'].unsqueeze(1).expand(
                    batch_size, num_props, -1).contiguous().view(
                        batch_size * num_props, -1)
                nll, _ = cal_nll_loss(
                    output['words_logit'], words_id, words_mask)

                eval_center = output.get('qcec_eval_center')
                eval_width = output.get('qcec_eval_width')
                if eval_center is None or eval_width is None:
                    eval_center = output['center']
                    eval_width = output['width']
                arrays['video_ids'].extend(str(item[0]) for item in raw)
                arrays['sample_keys'].extend(
                    '{}\0{}'.format(item[0], item[3]) for item in raw)
                arrays['durations'].append(durations)
                arrays['ground_truth'].append(gt / durations[:, None])
                arrays['nll_score'].append(
                    _as_numpy(nll.view(batch_size, num_props)))
                event_score = output.get('event_score')
                if event_score is None:
                    event_score = torch.zeros_like(nll)
                arrays['event_score'].append(
                    _as_numpy(event_score.view(batch_size, num_props)))
                for name, output_name in (
                        ('raw_center', 'center'), ('raw_width', 'width'),
                        ('eval_center', None), ('eval_width', None)):
                    value = output[output_name] if output_name else (
                        eval_center if name == 'eval_center' else eval_width)
                    arrays[name].append(
                        _as_numpy(value.view(batch_size, num_props)))
                for name in optional_names:
                    value = output.get(name)
                    if value is None:
                        value = batch['net_input'].get(name)
                    if value is not None:
                        optional[name].append(_as_numpy(value))

    result = {
        'video_ids': np.asarray(arrays['video_ids'], dtype=str),
        'sample_keys': np.asarray(arrays['sample_keys'], dtype=str),
    }
    for name in arrays:
        if name in {'video_ids', 'sample_keys'}:
            continue
        result[name] = np.concatenate(arrays[name], axis=0)
    for name, values in optional.items():
        if values:
            result[name] = np.concatenate(values, axis=0)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--eval-seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument(
        '--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == 'cpu' or (
            args.device == 'auto' and not torch.cuda.is_available()):
        install_cpu_cuda_shim()
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA but CUDA is unavailable')
    eval_seed = (
        args.eval_seed if args.eval_seed is not None
        else args.seed + 1000033)
    runner = build_runner(
        args.config_path, args.seed, eval_seed, batch_size=args.batch_size)
    runner._load_model_parameters(args.checkpoint)
    loader = runner.val_loader if args.split == 'val' else runner.test_loader
    cache = collect_cache(
        runner, loader, epoch=args.epoch, max_batches=args.max_batches)
    metadata = {
        'config_path': str(Path(args.config_path).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'split': args.split,
        'epoch': args.epoch,
        'seed': args.seed,
        'eval_seed': eval_seed,
        'batch_size': args.batch_size or runner.args['train']['batch_size'],
        'sample_count': int(len(cache['video_ids'])),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output), metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True)), **cache)
    print('saved {} samples to {}'.format(len(cache['video_ids']), output))


if __name__ == '__main__':
    main()
