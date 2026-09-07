import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from runners.main_runner import (
    MainRunner,
    calculate_candidate_diagnostics,
    calculate_checkpoint_selection_scores,
)


def test_checkpoint_selection_scores_track_rank1_rank5_and_composite():
    results = {
        "R@1,mIoU": SimpleNamespace(avg=0.4),
        "R@5,mIoU": SimpleNamespace(avg=0.6),
    }
    assert calculate_checkpoint_selection_scores(results) == {
        "r1": 0.4,
        "r5": 0.6,
        "composite": 0.5,
    }


def test_training_saves_all_three_selection_objectives():
    runner = object.__new__(MainRunner)
    runner.args = {
        "select_on_val": False,
        "train": {"max_num_epochs": 2},
    }
    runner.val_loader = None
    runner.test_loader = object()
    runner.model = SimpleNamespace(use_event_disentanglement=False)
    runner._train_one_epoch = lambda epoch: None
    evaluation_results = [
        {
            "R@1,mIoU": SimpleNamespace(avg=0.4),
            "R@5,mIoU": SimpleNamespace(avg=0.5),
        },
        {
            "R@1,mIoU": SimpleNamespace(avg=0.3),
            "R@5,mIoU": SimpleNamespace(avg=0.7),
        },
    ]

    with tempfile.TemporaryDirectory() as directory:
        runner.model_saved_path = directory
        runner._save_model = lambda path: Path(path).write_bytes(b"model")
        runner.eval = lambda **kwargs: evaluation_results.pop(0)
        runner.train()

        assert (Path(directory) / "model-best-r1.pt").exists()
        assert (Path(directory) / "model-best-r5.pt").exists()
        assert (Path(directory) / "model-best-composite.pt").exists()
        assert (Path(directory) / "model-best.pt").exists()
def test_candidate_diagnostics_report_scale_coverage_and_boundary_gap():
    raw = np.array([
        [[0.10, 0.20], [0.05, 0.35], [0.50, 0.80]],
        [[0.50, 0.60], [0.40, 0.70], [0.00, 1.00]],
    ])
    # Already NLL-sorted for this synthetic example.
    sorted_props = raw.copy()
    gt = np.array([[0.10, 0.20], [0.45, 0.65]])
    context_width = np.array([
        [0.15, 0.50, 0.50],
        [0.15, 0.50, 1.00],
    ])

    diagnostics = calculate_candidate_diagnostics(
        raw, sorted_props, gt, context_width=context_width)

    assert np.isclose(
        diagnostics["proposal1_mean_width"][0], np.mean([0.1, 0.1]))
    assert diagnostics["nll_top3_IoU@0.5"][0] == 1.0
    assert diagnostics["gt_short_R5_IoU@0.5"][1] == 1
    assert diagnostics["gt_medium_short_R5_IoU@0.5"][1] == 1
    assert diagnostics["negative_localization_width_gap"][0] > 0
