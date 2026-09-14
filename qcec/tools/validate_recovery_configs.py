"""Validate the ActivityNet B/A/C recovery protocol before training."""

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    with open(path, encoding='utf8') as handle:
        return json.load(handle)


def validate(config_paths):
    configs = [_load(path) for path in config_paths]
    if [config.get('experiment', {}).get('recovery_id') for config in configs] != [
            'B', 'A', 'C']:
        raise ValueError('configs must be ordered B, A, C')

    common_dataset = (
        'dataset', 'feature_path', 'vocab_size', 'word_dim', 'frame_dim',
        'max_num_words', 'max_num_frames', 'target_stride', 'train_data',
        'test_data', 'val_data', 'vocab_path')
    for key in common_dataset:
        values = [config['dataset'].get(key) for config in configs]
        if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
            raise ValueError('dataset field {} differs across B/A/C'.format(key))

    common_train = ('batch_size', 'max_num_epochs', 'model_saved_path')
    for key in common_train:
        values = [config['train'].get(key) for config in configs]
        if key != 'model_saved_path' and len(set(values)) != 1:
            raise ValueError('train field {} differs across B/A/C'.format(key))

    def public_model_config(config):
        value = json.loads(json.dumps(config['model']['config']))
        value.pop('qcec', None)
        return value

    model_signatures = [json.dumps(public_model_config(config), sort_keys=True)
                        for config in configs]
    if len(set(model_signatures)) != 1:
        raise ValueError('public model configuration differs across B/A/C')

    public_loss_keys = sorted({
        key for config in configs for key in config.get('loss', {})
        if not key.startswith('qcec_')
    })
    for key in public_loss_keys:
        values = [config.get('loss', {}).get(key) for config in configs]
        if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
            raise ValueError('public loss field {} differs across B/A/C'.format(key))

    for config in configs:
        if config.get('selection_use_event_score', True):
            raise ValueError('selection_use_event_score must be false')
        qcec = config.get('model', {}).get('config', {}).get('qcec', {})
        if config['experiment']['recovery_id'] == 'B':
            if qcec.get('enabled', False):
                raise ValueError('B must disable QCEC')
            continue
        if not qcec.get('enabled', False):
            raise ValueError('{} must enable QCEC'.format(
                config['experiment']['recovery_id']))
        if qcec.get('freeze_backbone_epochs') != 0:
            raise ValueError('QCEC scratch freeze must be exactly 0')
        if qcec.get('use_slot_prior', False) or qcec.get('snap_enabled', False):
            raise ValueError('slot prior and snapping must remain disabled')
        index_path = config['dataset'].get('qcec_cluster_index_path')
        if not index_path:
            raise ValueError('QCEC config has no cluster index path')
        resolved = Path(index_path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError('missing cluster index {}'.format(resolved))

    cross_values = [
        config.get('loss', {}).get('qcec_cross_weight', 0.0)
        for config in configs]
    if cross_values != [0.0, 0.0, 0.1]:
        raise ValueError('expected crossing weights [0.0, 0.0, 0.1], got {}'.format(
            cross_values))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'configs', nargs='*',
        default=[
            'config/activitynet/recovery_b_baseline.json',
            'config/activitynet/recovery_a_adapter.json',
            'config/activitynet/recovery_c_cross.json',
        ])
    args = parser.parse_args()
    paths = [
        str((PROJECT_ROOT / path).resolve())
        if not os.path.isabs(path) else path
        for path in args.configs
    ]
    configs = validate(paths)
    print('validated recovery configs: {}'.format(', '.join(
        config['experiment']['recovery_id'] for config in configs)))


if __name__ == '__main__':
    main()
