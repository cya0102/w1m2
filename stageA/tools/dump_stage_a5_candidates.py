"""GPU export of deterministic Stage-A.5 candidate features and labels."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cpl import deterministic_eval_word_mask  # noqa: E402
from runners import MainRunner  # noqa: E402
from runners.main_runner import (  # noqa: E402
    move_to_cuda, select_minimal_sufficient_candidates,
)
from tools.stage_a5_utils import (  # noqa: E402
    SCHEMA_VERSION, save_candidate_exports, sha256_file,
)
from utils import load_json  # noqa: E402


def parse_mask_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('--mask-seeds must contain one or more unique integers')
    return seeds


def install_cpu_cuda_shim():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _output_paths(prefix):
    prefix = Path(prefix)
    if prefix.suffix == '.npz':
        prefix = prefix.with_suffix('')
    return (Path(str(prefix) + '_features.npz'),
            Path(str(prefix) + '_labels.npz'))


def _aggregate(values, valid):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid).astype(bool)
    mean = np.full(valid.shape, np.inf, dtype=np.float32)
    std = np.full(valid.shape, np.inf, dtype=np.float32)
    for candidate_index in range(values.shape[-1]):
        rows = valid[..., candidate_index]
        if not rows.any():
            continue
        candidate_values = values[:, ..., candidate_index]
        finite = np.isfinite(candidate_values).all(axis=0) & rows
        if finite.any():
            mean[..., candidate_index][finite] = candidate_values[:, finite].mean(
                axis=0).astype(np.float32)
            std[..., candidate_index][finite] = candidate_values[:, finite].std(
                axis=0).astype(np.float32)
    return mean, std


def export_candidates(args):
    config_path = _resolve(args.config_path)
    checkpoint_path = _resolve(args.checkpoint)
    config = load_json(str(config_path))
    if not config['model']['config'].get('event_boundary_refinement', {}).get(
            'stage_a5', {}).get('enabled', False):
        raise ValueError('config must enable event_boundary_refinement.stage_a5')
    mask_seeds = parse_mask_seeds(
        args.mask_seeds or ','.join(str(seed) for seed in config[
            'model']['config']['event_boundary_refinement']['stage_a5'].get(
                'eval_mask_seeds', [8, 18, 28])))

    if args.device == 'cpu' or (args.device == 'auto' and
                                not torch.cuda.is_available()):
        install_cpu_cuda_shim()
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested --device cuda, but CUDA is unavailable')
    os.chdir(str(PROJECT_ROOT))

    config_for_runner = load_json(str(config_path))
    config_for_runner['tag'] = 'stage_a5_dump'
    config_for_runner['run_timestamp'] = 'stage_a5_dump'
    config_for_runner['vote'] = False
    config_for_runner['selection_strategy'] = 'nll'
    config_for_runner['selection_temperature'] = 0.1
    config_for_runner['select_on_val'] = True
    if args.batch_size is not None:
        config_for_runner['train']['batch_size'] = args.batch_size
    runner = MainRunner(config_for_runner)
    runner._load_model_parameters(str(checkpoint_path))
    loader = runner.val_loader if args.split == 'val' else runner.test_loader
    if loader is None:
        raise ValueError('requested split is absent from the config')
    runner.model.eval()
    device = next(runner.model.parameters()).device

    rows = {key: [] for key in (
        'sample_ids', 'video_ids', 'durations', 'parent_start', 'parent_end',
        'parent_event_score', 'candidate_start', 'candidate_end',
        'candidate_valid', 'candidate_nll_mean', 'candidate_nll_std',
        'candidate_left_boundary_score', 'candidate_right_boundary_score',
        'candidate_boundary_confidence', 'candidate_shell_nll_mean',
        'candidate_shell_nll_std', 'candidate_contrast_mean',
        'candidate_contrast_std', 'legacy_selected_index',
    )}
    gt_rows = []
    candidate_type = None
    seen_sample_ids = set()
    batch_count = 0
    with torch.no_grad():
        for batch_count, batch in enumerate(loader, 1):
            if args.max_batches is not None and batch_count > args.max_batches:
                break
            raw = batch['raw']
            sample_ids = [str(item[4]) if len(item) >= 5 else None
                          for item in raw]
            if any(item is None for item in sample_ids):
                raise ValueError('Stage A.5 export requires raw sample_id at raw[4]')
            if seen_sample_ids.intersection(sample_ids):
                raise ValueError('sample ids are not unique in the exported split')
            seen_sample_ids.update(sample_ids)
            video_ids = [str(item[0]) for item in raw]
            durations = np.asarray([float(item[1]) for item in raw], dtype=np.float32)
            gt = np.asarray([item[2] for item in raw], dtype=np.float32)
            gt = gt / durations[:, None]
            gt_rows.append(gt)
            cpu_net_input = batch['net_input']
            per_seed = []
            for mask_seed in mask_seeds:
                net_input = move_to_cuda(cpu_net_input)
                net_input['run_stage_a'] = True
                net_input['run_stage_a5'] = True
                mask = deterministic_eval_word_mask(
                    sample_ids, cpu_net_input['words_len'],
                    cpu_net_input['words_feat'].size(1) - 1, mask_seed,
                    weights=cpu_net_input['weights'])
                net_input['eval_word_mask'] = mask.to(device)
                output = runner.model(epoch=args.epoch, **net_input)
                required = (
                    'stage_a_candidate_start', 'stage_a_candidate_end',
                    'stage_a_candidate_valid', 'stage_a_candidate_type',
                    'stage_a_candidate_nll', 'stage_a_candidate_shell_nll',
                    'stage_a_candidate_boundary_confidence',
                    'stage_a_candidate_left_boundary_score',
                    'stage_a_candidate_right_boundary_score',
                )
                missing = [key for key in required if key not in output]
                if missing:
                    raise RuntimeError('model output is missing {}'.format(missing))
                current = {
                    key: output[key].detach().cpu().numpy()
                    for key in required
                }
                expected_nll = runner.model.proposal_reconstruction_nll(
                    output['words_logit'], output['words_id'],
                    output['words_mask']).detach().cpu().numpy()
                if not np.allclose(current['stage_a_candidate_nll'][..., 0],
                                   expected_nll, atol=1e-6, rtol=0):
                    raise AssertionError('candidate zero NLL differs from baseline')
                event_score = output.get('event_score')
                current['event_score'] = (
                    event_score.detach().view(len(raw), runner.model.num_props)
                    .cpu().numpy() if event_score is not None else
                    np.zeros((len(raw), runner.model.num_props), dtype=np.float32))
                per_seed.append(current)

            reference = per_seed[0]
            for current in per_seed[1:]:
                for key in ('stage_a_candidate_start', 'stage_a_candidate_end',
                            'stage_a_candidate_valid',
                            'stage_a_candidate_type',
                            'stage_a_candidate_boundary_confidence',
                            'stage_a_candidate_left_boundary_score',
                            'stage_a_candidate_right_boundary_score'):
                    if not np.array_equal(reference[key], current[key]):
                        raise AssertionError(
                            '{} changed across mask seeds'.format(key))
            valid = reference['stage_a_candidate_valid'].astype(bool)
            original_start = reference['stage_a_candidate_start'][..., 0]
            original_end = reference['stage_a_candidate_end'][..., 0]
            if not np.all(reference['stage_a_candidate_end'] >
                          reference['stage_a_candidate_start']):
                raise AssertionError('candidate intervals must have positive width')
            broadcast_start = np.broadcast_to(
                original_start[..., None], reference['stage_a_candidate_start'].shape)
            broadcast_end = np.broadcast_to(
                original_end[..., None], reference['stage_a_candidate_end'].shape)
            if not np.all(reference['stage_a_candidate_start'][valid] >=
                          broadcast_start[valid] - 1e-6) or \
                    not np.all(reference['stage_a_candidate_end'][valid] <=
                               broadcast_end[valid] + 1e-6):
                raise AssertionError('valid candidates must be inward-only')
            nll_values = np.stack([
                current['stage_a_candidate_nll'] for current in per_seed], axis=0)
            shell_values = np.stack([
                current['stage_a_candidate_shell_nll'] for current in per_seed], axis=0)
            nll_mean, nll_std = _aggregate(nll_values, valid)
            trim_valid = valid & (np.arange(7)[None, None, :] > 0)
            shell_mean, shell_std = _aggregate(shell_values, trim_valid)
            shell_mean[..., 0] = np.inf
            shell_std[..., 0] = 0.0
            # Invalid candidates use ``inf`` placeholders for both terms.
            # Avoid inf-inf -> NaN; only compute contrast where both
            # counterfactual losses are finite and keep all other entries at
            # the existing invalid sentinel.
            contrast_values = np.full(
                shell_values.shape,
                np.inf,
                dtype=np.result_type(shell_values.dtype, nll_values.dtype,
                                     np.float32))
            finite_contrast = np.isfinite(shell_values) & np.isfinite(nll_values)
            np.subtract(shell_values, nll_values, out=contrast_values,
                        where=finite_contrast)
            contrast_mean, contrast_std = _aggregate(
                contrast_values, trim_valid)
            contrast_mean[..., 0] = np.inf
            contrast_std[..., 0] = np.inf
            legacy_epsilon = float(config_for_runner['model']['config'][
                'event_boundary_refinement'].get('max_nll_increase', 0.02))
            _, _, legacy_index = select_minimal_sufficient_candidates(
                torch.from_numpy(reference['stage_a_candidate_start']),
                torch.from_numpy(reference['stage_a_candidate_end']),
                torch.from_numpy(nll_mean), torch.from_numpy(valid),
                legacy_epsilon)

            rows['sample_ids'].extend(sample_ids)
            rows['video_ids'].extend(video_ids)
            rows['durations'].extend(durations.tolist())
            rows['parent_start'].append(reference['stage_a_candidate_start'][..., 0])
            rows['parent_end'].append(reference['stage_a_candidate_end'][..., 0])
            rows['parent_event_score'].append(np.mean([
                current['event_score'] for current in per_seed], axis=0))
            for name, value in (
                    ('candidate_start', reference['stage_a_candidate_start']),
                    ('candidate_end', reference['stage_a_candidate_end']),
                    ('candidate_valid', valid),
                    ('candidate_nll_mean', nll_mean),
                    ('candidate_nll_std', nll_std),
                    ('candidate_left_boundary_score',
                     reference['stage_a_candidate_left_boundary_score']),
                    ('candidate_right_boundary_score',
                     reference['stage_a_candidate_right_boundary_score']),
                    ('candidate_boundary_confidence',
                     reference['stage_a_candidate_boundary_confidence']),
                    ('candidate_shell_nll_mean', shell_mean),
                    ('candidate_shell_nll_std', shell_std),
                    ('candidate_contrast_mean', contrast_mean),
                    ('candidate_contrast_std', contrast_std),
                    ('legacy_selected_index', legacy_index.numpy().astype(np.int8))):
                rows[name].append(value)
            current_type = reference['stage_a_candidate_type'].astype(np.int8)
            if candidate_type is None:
                candidate_type = current_type
            elif not np.array_equal(candidate_type, current_type):
                raise AssertionError('candidate type changed across batches')

    if batch_count == 0:
        raise ValueError('no batches were exported')
    partial = args.max_batches is not None
    q = len(rows['sample_ids'])
    if not partial and q != len(loader.dataset):
        raise AssertionError('export row count does not equal dataset length')
    metadata = {
        'schema_version': SCHEMA_VERSION,
        'dataset': str(config_for_runner['dataset']['dataset']).lower().replace(
            'activitynet', 'activitynet').replace('charadessta', 'charades'),
        'split': args.split,
        'checkpoint_label': args.checkpoint_label,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'config_sha256': sha256_file(config_path),
        'mask_seeds': list(mask_seeds),
        'train_data': str(config_for_runner['dataset']['train_data']),
        'val_data': str(config_for_runner['dataset']['val_data']),
        'test_data': str(config_for_runner['dataset']['test_data']),
        'validation_is_test': os.path.normpath(str(
            config_for_runner['dataset']['val_data'])) == os.path.normpath(str(
                config_for_runner['dataset']['test_data'])),
        'query_count': q,
        'partial': partial,
    }
    dataset = 'activitynet' if 'activity' in metadata['dataset'] else 'charades'
    metadata['dataset'] = dataset
    features = {
        'schema_version': np.asarray([SCHEMA_VERSION], dtype=np.int32),
        'dataset': np.asarray([dataset], dtype=str),
        'split': np.asarray([args.split], dtype=str),
        'checkpoint_path': np.asarray([str(checkpoint_path)], dtype=str),
        'checkpoint_sha256': np.asarray([metadata['checkpoint_sha256']], dtype=str),
        'config_sha256': np.asarray([metadata['config_sha256']], dtype=str),
        'mask_seeds': np.asarray(mask_seeds, dtype=np.int64),
        'sample_ids': np.asarray(rows['sample_ids'], dtype=str),
        'video_ids': np.asarray(rows['video_ids'], dtype=str),
        'durations': np.asarray(rows['durations'], dtype=np.float32),
        'parent_start': np.concatenate(rows['parent_start'], axis=0).astype(np.float32),
        'parent_end': np.concatenate(rows['parent_end'], axis=0).astype(np.float32),
        'parent_event_score': np.concatenate(
            rows['parent_event_score'], axis=0).astype(np.float32),
        'candidate_start': np.concatenate(rows['candidate_start'], axis=0).astype(np.float32),
        'candidate_end': np.concatenate(rows['candidate_end'], axis=0).astype(np.float32),
        'candidate_valid': np.concatenate(rows['candidate_valid'], axis=0).astype(bool),
        'candidate_type': candidate_type.astype(np.int8),
        'candidate_nll_mean': np.concatenate(rows['candidate_nll_mean'], axis=0).astype(np.float32),
        'candidate_nll_std': np.concatenate(rows['candidate_nll_std'], axis=0).astype(np.float32),
        'candidate_left_boundary_score': np.concatenate(
            rows['candidate_left_boundary_score'], axis=0).astype(np.float32),
        'candidate_right_boundary_score': np.concatenate(
            rows['candidate_right_boundary_score'], axis=0).astype(np.float32),
        'candidate_boundary_confidence': np.concatenate(
            rows['candidate_boundary_confidence'], axis=0).astype(np.float32),
        'candidate_shell_nll_mean': np.concatenate(
            rows['candidate_shell_nll_mean'], axis=0).astype(np.float32),
        'candidate_shell_nll_std': np.concatenate(
            rows['candidate_shell_nll_std'], axis=0).astype(np.float32),
        'candidate_contrast_mean': np.concatenate(
            rows['candidate_contrast_mean'], axis=0).astype(np.float32),
        'candidate_contrast_std': np.concatenate(
            rows['candidate_contrast_std'], axis=0).astype(np.float32),
        'legacy_selected_index': np.concatenate(
            rows['legacy_selected_index'], axis=0).astype(np.int8),
        'metadata_json': np.asarray([json.dumps(metadata, sort_keys=True)], dtype=str),
    }
    labels = {
        'schema_version': np.asarray([SCHEMA_VERSION], dtype=np.int32),
        'dataset': np.asarray([dataset], dtype=str),
        'split': np.asarray([args.split], dtype=str),
        'sample_ids': np.asarray(rows['sample_ids'], dtype=str),
        'video_ids': np.asarray(rows['video_ids'], dtype=str),
        'gt_normalized': np.concatenate(gt_rows, axis=0).astype(np.float32),
        'metadata_json': np.asarray([json.dumps(metadata, sort_keys=True)], dtype=str),
    }
    features_path, labels_path = _output_paths(args.output)
    save_candidate_exports(features_path, labels_path, features, labels)
    print('saved features to {}'.format(features_path))
    print('saved labels to {}'.format(labels_path))
    return features_path, labels_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-label', default=None)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--mask-seeds', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    export_candidates(args)


if __name__ == '__main__':
    main()
