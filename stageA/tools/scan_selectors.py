"""Offline Rank-1 selector sweep for CPL-LRRV V4 stage1.5.

This script keeps the trained proposal generator fixed and only changes the
inference-time proposal scoring/selection rule.  It is intended to answer one
question before adding new training losses:

    Do the strong Rank-5 candidates already contain a good Rank-1 proposal,
    and can a better selector recover it?
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.loss import cal_nll_loss  # noqa: E402
from runners import MainRunner  # noqa: E402
from runners.main_runner import (  # noqa: E402
    select_proposal_by_strategy,
    top_1_metric,
    top_n_metric,
)
from utils import load_json  # noqa: E402


METRIC_KEYS = [
    "R@1,mIoU",
    "R@1,IoU@0.1",
    "R@1,IoU@0.3",
    "R@1,IoU@0.5",
    "R@1,IoU@0.7",
    "R@1,IoU@0.9",
    "R@5,mIoU",
    "R@5,IoU@0.1",
    "R@5,IoU@0.3",
    "R@5,IoU@0.5",
    "R@5,IoU@0.7",
    "R@5,IoU@0.9",
]


def parse_float_list(value):
    if value is None or value == "":
        return []
    return [float(item) for item in value.split(",")]


def parse_splits(value):
    splits = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"val", "test"}
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError("unknown split(s): {}".format(", ".join(unknown)))
    return splits


def update_meter(meters, name, value, count):
    total, seen = meters.get(name, (0.0, 0))
    meters[name] = (total + float(value) * count, seen + count)


def finalize_meters(meters):
    return {
        name: (total / max(count, 1))
        for name, (total, count) in meters.items()
    }


def selector_grid(event_betas, temperatures, width_gammas):
    selectors = []
    for width_gamma in width_gammas:
        for event_beta in event_betas:
            selectors.append({
                "name": "nll_beta{}_wg{}".format(event_beta, width_gamma),
                "strategy": "nll",
                "temperature": 0.0,
                "event_beta": event_beta,
                "width_gamma": width_gamma,
            })
            for temperature in temperatures:
                selectors.append({
                    "name": "semantic_t{}_beta{}_wg{}".format(
                        temperature, event_beta, width_gamma),
                    "strategy": "semantic_vote",
                    "temperature": temperature,
                    "event_beta": event_beta,
                    "width_gamma": width_gamma,
                })
    selectors.append({
        "name": "geometric_vote",
        "strategy": "geometric_vote",
        "temperature": 0.1,
        "event_beta": 0.0,
        "width_gamma": 0.0,
    })
    return selectors


def install_cpu_cuda_shim():
    """Let the original CUDA-only CPL code run on CPU for smoke tests."""
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self


def build_runner(config_path, tag, seed, batch_size=None):
    import random

    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    np.random.seed(seed + 1)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = load_json(config_path)
    if batch_size is not None:
        args["train"]["batch_size"] = batch_size
    args["tag"] = tag
    args["run_timestamp"] = "selector_scan"
    args["vote"] = False
    args["selection_strategy"] = "nll"
    args["selection_temperature"] = 0.1
    args["select_on_val"] = True
    return MainRunner(args)


def gather_batch_candidates(runner, batch, epoch):
    durations = np.asarray([item[1] for item in batch["raw"]])
    gt = np.asarray([item[2] for item in batch["raw"]])
    gt = gt / durations[:, np.newaxis]

    net_input = move_to_cuda(batch["net_input"])
    output = runner.model(epoch=epoch, **net_input)
    bsz = len(durations)
    num_props = runner.model.num_props
    k = min(num_props, 5)

    words_mask = output["words_mask"].unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
    words_id = output["words_id"].unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)

    nll_loss, _ = cal_nll_loss(output["words_logit"], words_id, words_mask)
    nll_score = nll_loss.view(bsz, num_props)

    raw_width = output["width"].view(bsz, num_props)
    raw_center = output["center"].view(bsz, num_props)
    raw_props = torch.stack([
        torch.clamp(raw_center - raw_width / 2, min=0),
        torch.clamp(raw_center + raw_width / 2, max=1),
    ], dim=-1)

    event_score = output.get("event_score")
    if event_score is None:
        event_score = torch.zeros_like(nll_score)
    else:
        event_score = event_score.view(bsz, num_props)

    return {
        "gt": gt,
        "nll_score": nll_score,
        "event_score": event_score,
        "raw_width": raw_width,
        "raw_props": raw_props,
        "top_k": k,
        "batch_size": bsz,
    }


def evaluate_selectors(runner, loader, split, selectors, epoch, prior_width,
                       max_batches=None):
    runner.model.eval()
    meters_by_selector = {selector["name"]: {} for selector in selectors}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            if max_batches is not None and batch_index > max_batches:
                break
            candidates = gather_batch_candidates(runner, batch, epoch)
            gt = candidates["gt"]
            bsz = candidates["batch_size"]
            k = candidates["top_k"]

            for selector in selectors:
                width_penalty = torch.abs(
                    candidates["raw_width"] - prior_width)
                proposal_score = (
                    candidates["nll_score"]
                    - selector["event_beta"] * candidates["event_score"]
                    + selector["width_gamma"] * width_penalty)
                idx = proposal_score.argsort(dim=-1)
                sorted_score = proposal_score.gather(
                    index=idx, dim=-1).cpu().numpy()
                sorted_props = candidates["raw_props"].gather(
                    index=idx.unsqueeze(-1).expand(-1, -1, 2),
                    dim=1).cpu().numpy()

                selected_index = select_proposal_by_strategy(
                    sorted_props,
                    sorted_score,
                    strategy=selector["strategy"],
                    temperature=max(selector["temperature"], 1e-6),
                    charades_anchor=False)
                rank1 = top_1_metric(
                    sorted_props[np.arange(bsz), selected_index], gt)
                rank5 = top_n_metric(
                    sorted_props[:, :k].transpose(1, 0, 2), gt)

                meters = meters_by_selector[selector["name"]]
                for key, value in rank1.items():
                    update_meter(meters, "R@1," + key, value, bsz)
                for key, value in rank5.items():
                    update_meter(meters, "R@{},".format(k) + key, value, bsz)

    rows = []
    for selector in selectors:
        metrics = finalize_meters(meters_by_selector[selector["name"]])
        row = {
            "split": split,
            "selector": selector["name"],
            "strategy": selector["strategy"],
            "temperature": selector["temperature"],
            "event_beta": selector["event_beta"],
            "width_gamma": selector["width_gamma"],
            "prior_width": prior_width,
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        if row["R@5,mIoU"] != "" and row["R@1,mIoU"] != "":
            row["oracle_gap_mIoU"] = row["R@5,mIoU"] - row["R@1,mIoU"]
        else:
            row["oracle_gap_mIoU"] = ""
        rows.append(row)
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_label",
        "checkpoint",
        "epoch",
        "split",
        "selector",
        "strategy",
        "temperature",
        "event_beta",
        "width_gamma",
        "prior_width",
    ] + METRIC_KEYS + ["oracle_gap_mIoU"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows, split):
    subset = [row for row in rows if row["split"] == split]
    if not subset:
        return
    print("\nTop selectors on {}:".format(split))
    for metric in ("R@1,mIoU", "R@1,IoU@0.3", "R@1,IoU@0.5"):
        best = max(subset, key=lambda row: float(row[metric]))
        print("{}: {} = {:.4f}, R5@0.3 = {:.4f}, R5@0.5 = {:.4f}".format(
            metric,
            best["selector"],
            float(best[metric]),
            float(best["R@5,IoU@0.3"]),
            float(best["R@5,IoU@0.5"])))


def move_to_cuda(sample):
    def _move(value):
        if torch.is_tensor(value):
            return value.cuda()
        if isinstance(value, dict):
            return {key: _move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_move(item) for item in value]
        return value

    return _move(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--event-betas", default="0,0.1,0.25,0.5,1.0")
    parser.add_argument("--temperatures", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--width-gammas", default="0")
    parser.add_argument("--prior-width", type=float, default=0.30)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="use cuda when available; cpu is intended for smoke tests")
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="debug only: evaluate at most this many batches per split")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.device == "cpu" or (
            args.device == "auto" and not torch.cuda.is_available()):
        install_cpu_cuda_shim()
        print("Running selector scan on CPU. Use --device cuda on the GPU machine "
              "for full evaluation.")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda, but CUDA is not available")

    label = args.checkpoint_label or Path(args.checkpoint).parent.name
    runner = build_runner(
        args.config_path,
        tag="selector_scan_" + label,
        seed=args.seed,
        batch_size=args.batch_size)
    runner._load_model_parameters(args.checkpoint)

    selectors = selector_grid(
        parse_float_list(args.event_betas),
        parse_float_list(args.temperatures),
        parse_float_list(args.width_gammas))

    all_rows = []
    for split in parse_splits(args.splits):
        loader = runner.val_loader if split == "val" else runner.test_loader
        rows = evaluate_selectors(
            runner, loader, split, selectors, args.epoch, args.prior_width,
            max_batches=args.max_batches)
        for row in rows:
            row["checkpoint_label"] = label
            row["checkpoint"] = args.checkpoint
            row["epoch"] = args.epoch
        all_rows.extend(rows)
        print_summary(rows, split)

    output = Path(args.output)
    write_rows(output, all_rows)
    print("\nSaved selector scan to {}".format(output))


if __name__ == "__main__":
    main()
