import torch

from models.cpl import CPL
from runners.main_runner import MainRunner


def small_config(enabled=True):
    return {
        'frames_input_size': 4, 'words_input_size': 3, 'hidden_size': 8,
        'vocab_size': 11, 'use_negative': True, 'num_props': 3,
        'sigma': 9, 'gamma': 0, 'dropout': 0.0, 'max_epoch': 4,
        'proposal_generator': {
            'type': 'gaussian_mixture', 'max_components': 3,
            'component_sigma': 4.0, 'importance_temperature': 1.0,
            'boundary_mode': 'weighted', 'boundary_shrink': 0.0,
        },
        'event_disentanglement': {
            'enabled': False,
        },
        'qcec': {
            'enabled': enabled, 'num_clusters': 4, 'attention_dim': 8,
            'use_slot_prior': False, 'snap_enabled': False,
        },
        'DualTransformer': {
            'd_model': 8, 'num_heads': 2,
            'num_decoder_layers1': 1, 'num_decoder_layers2': 1,
            'dropout': 0.0,
        },
    }


class FakeScheduler:
    def __init__(self):
        self.updates = []

    def step_update(self, update):
        self.updates.append(update)
        return 0.0


def make_runner(model, tmp_path):
    runner = MainRunner.__new__(MainRunner)
    runner.args = {'tag': 'checkpoint-test'}
    runner.model = model
    runner.model_saved_path = str(tmp_path)
    runner.lr_scheduler = FakeScheduler()
    runner.num_updates = 0
    runner.start_epoch = 0
    return runner


def test_baseline_warm_start_only_allows_qcec_missing_keys(tmp_path):
    torch.manual_seed(3)
    baseline = CPL(small_config(enabled=False))
    checkpoint_path = tmp_path / 'baseline.pt'
    torch.save({'model_parameters': baseline.state_dict()}, checkpoint_path)

    target = CPL(small_config(enabled=True))
    runner = make_runner(target, tmp_path)
    runner._load_baseline_model(str(checkpoint_path))

    baseline_state = baseline.state_dict()
    target_state = target.state_dict()
    for name, value in baseline_state.items():
        assert torch.equal(target_state[name], value)
    assert runner.num_updates == 0
    assert runner.start_epoch == 0
    assert runner.lr_scheduler.updates[-1] == 0


def test_qcec_checkpoint_saves_and_resumes_epoch(tmp_path):
    model = CPL(small_config(enabled=True))
    runner = make_runner(model, tmp_path)
    runner.num_updates = 17
    runner.current_epoch = 3
    checkpoint_path = tmp_path / 'qcec.pt'
    runner._save_model(str(checkpoint_path))

    resumed = make_runner(CPL(small_config(enabled=True)), tmp_path)
    resumed._load_model(str(checkpoint_path))
    assert resumed.num_updates == 17
    assert resumed.start_epoch == 3
    assert resumed.lr_scheduler.updates[-1] == 17


def test_qcec_stage_freezes_then_restores_baseline_parameters(tmp_path):
    model = CPL(small_config(enabled=True))
    runner = make_runner(model, tmp_path)
    qcec_parameters = {
        id(parameter) for parameter in model.qcec.parameters()
    }
    qcec_parameters.update(id(parameter) for parameter in model.qcec_adapter.parameters())

    model.set_qcec_training_stage(True)
    assert all(
        parameter.requires_grad == (id(parameter) in qcec_parameters)
        for parameter in model.parameters())
    model.set_qcec_training_stage(False)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_disabled_baseline_state_is_strictly_compatible():
    source = CPL(small_config(enabled=False))
    target = CPL(small_config(enabled=False))
    target.load_state_dict(source.state_dict(), strict=True)
