import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

import train
from runners.main_runner import MainRunner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/activitynet/main.json"


def test_v4_inherits_current_v3_cli_defaults():
    argv = ["train.py", "--config-path", str(CONFIG), "--vote"]
    with patch.object(sys, "argv", argv):
        args = train.parse_args()
    assert args.seed == 8
    assert args.select_on_val is True
    assert args.vote is True
    assert train.resolve_selection_strategy(
        args.vote, args.selection_strategy) == "semantic_vote"


def test_v4_inherits_current_v3_losses_and_adds_mixture():
    config = json.loads(CONFIG.read_text())
    loss = config["loss"]
    assert loss["alpha_1"] == 2.0
    assert loss["alpha_2"] == 0.0
    assert loss["event_alpha"] == 0.1
    assert loss["event_sep_weight"] == 1.0
    assert loss["event_text_weight"] == 1.0
    assert loss["event_context_weight"] == 2.0
    assert loss["event_overlap_weight"] == 1.0
    assert "event_semantic_weight" not in loss
    assert loss["mixture_pull_weight"] == 0.05
    assert loss["mixture_intra_push_weight"] == 0.05
    assert loss["mixture_inter_push_weight"] == 0.1


def test_v4_replaces_only_the_proposal_representation():
    config = json.loads(CONFIG.read_text())["model"]["config"]
    assert config["num_props"] == 5
    proposal = config["proposal_generator"]
    assert proposal == {
        "type": "gaussian_mixture",
        "max_components": 5,
        "component_sigma": 4.0,
        "importance_temperature": 1.0,
        "boundary_mode": "weighted",
        "boundary_shrink": 0.0,
    }
    event = config["event_disentanglement"]
    assert event["rank"] == 8
    assert event["cpca_alpha"] == 1.0
    assert event["inference_event_weight"] == 0.0
    assert event["inference_vote_event_weight"] == 0.0


def test_stage1_ablation_configs_change_only_boundary_and_pull_push():
    expected = {
        "stage1_a1_weighted_current.json": ("weighted", 0.05, 0.05, 0.1),
        "stage1_a2_outer_pps.json": ("outer", 0.2, 0.01, 0.1),
        "stage1_a3_weighted_pps.json": ("weighted", 0.2, 0.01, 0.1),
    }
    for filename, values in expected.items():
        config = json.loads(
            (ROOT / "config/activitynet" / filename).read_text())
        proposal = config["model"]["config"]["proposal_generator"]
        loss = config["loss"]
        actual = (
            proposal["boundary_mode"],
            loss["mixture_pull_weight"],
            loss["mixture_intra_push_weight"],
            loss["mixture_inter_push_weight"],
        )
        assert actual == values
        assert proposal["boundary_shrink"] == 0.0


def test_v3_warm_start_loads_backbone_but_resets_event_and_schedule():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(3, 2)
            self.event_disentangler = nn.Linear(2, 2)
            self.mixture_generator = nn.Linear(2, 4)

    class ToySchedule:
        def __init__(self):
            self.updates = []

        def step_update(self, update):
            self.updates.append(update)

    runner = object.__new__(MainRunner)
    runner.model = ToyModel()
    runner.num_updates = 999
    runner.lr_scheduler = ToySchedule()
    original_event = {
        name: value.clone()
        for name, value in runner.model.event_disentangler.state_dict().items()
    }
    original_mixture = {
        name: value.clone()
        for name, value in runner.model.mixture_generator.state_dict().items()
    }

    source = {
        "shared.weight": torch.full_like(runner.model.shared.weight, 3.0),
        "shared.bias": torch.full_like(runner.model.shared.bias, 3.0),
        "event_disentangler.weight": torch.full_like(
            runner.model.event_disentangler.weight, 7.0),
        "event_disentangler.bias": torch.full_like(
            runner.model.event_disentangler.bias, 7.0),
        "fc_gauss.weight": torch.randn(16, 2),
        "fc_gauss.bias": torch.randn(16),
    }
    checkpoint = {
        "num_updates": 7020,
        "config": {"tag": "abla_no_diversity_s8"},
        "model_parameters": source,
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v3.pt"
        torch.save(checkpoint, path)
        runner._load_pretrained_model(path)

    assert torch.all(runner.model.shared.weight == 3.0)
    assert torch.all(runner.model.shared.bias == 3.0)
    for name, value in runner.model.event_disentangler.state_dict().items():
        assert torch.equal(value, original_event[name])
    for name, value in runner.model.mixture_generator.state_dict().items():
        assert torch.equal(value, original_mixture[name])
    assert runner.num_updates == 0
    assert runner.lr_scheduler.updates == [0]
