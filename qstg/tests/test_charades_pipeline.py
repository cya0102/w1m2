import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _config(stage):
    return json.loads((ROOT / 'config' / 'charades' /
                       ('qstg_{}.json'.format(stage))).read_text())


def test_charades_stage_configs_form_a_serial_pipeline():
    stage_a = _config('stage_a')
    stage_b = _config('stage_b')
    stage_c = _config('full')

    for config in (stage_a, stage_b, stage_c):
        dataset = config['dataset']
        assert dataset['dataset'] == 'CharadesSTA'
        assert dataset['train_data'] == 'data/charades/train_qstg.json'
        assert dataset['val_data'] == 'data/charades/val_qstg.json'
        assert dataset['test_data'] == 'data/charades/test.json'
        assert dataset['feature_path'].startswith('/data/')
        assert config['model']['config']['qstg']['enabled'] is True
        assert config['selection_strategy'] == 'qstg'

    assert stage_a['qstg_stage'] == 'A'
    assert stage_a['trainable_scope'] == 'qstg_only'
    assert stage_a['detach_base_states'] is True
    assert stage_a['train']['max_num_epochs'] == 5
    assert stage_a['train']['optimizer']['lr'] == 2e-4
    assert stage_a['model']['config']['qstg']['component_residual_enabled'] is False

    assert stage_b['qstg_stage'] == 'B'
    assert stage_b['trainable_scope'] == 'qstg_and_proposal'
    assert stage_b['detach_base_states'] is True
    assert stage_b['train']['max_num_epochs'] == 10
    assert stage_b['train']['optimizer']['lr'] == 1e-4
    assert stage_b['model']['config']['qstg']['component_residual_enabled'] is True

    assert stage_c['qstg_stage'] == 'C'
    assert stage_c['trainable_scope'] == 'all'
    assert stage_c['detach_base_states'] is False
    assert stage_c['train']['max_num_epochs'] == 10
    assert stage_c['train']['optimizer']['lr'] == 5e-5


def test_charades_stage_scripts_use_the_matching_handoffs():
    scripts = {
        'stage_a': ('config/charades/qstg_stage_a.json',
                    'charades-model-best.pt'),
        'stage_b': ('config/charades/qstg_stage_b.json',
                    'QSTG_CHARADES_STAGE_A_CHECKPOINT'),
        'full': ('config/charades/qstg_full.json',
                 'QSTG_CHARADES_STAGE_B_CHECKPOINT'),
    }
    for stage, expected in scripts.items():
        text = (ROOT / 'scripts' /
                ('run_charades_{}.sh'.format(stage))).read_text()
        assert expected[0] in text
        assert expected[1] in text
