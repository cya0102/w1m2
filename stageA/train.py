import argparse
import time
import os
from pathlib import Path

from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='config file path')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        '--resume', type=str, default=None,
        help='resume a structurally identical V4 checkpoint')
    checkpoint_group.add_argument(
        '--init-from-v3', type=str, default=None,
        help='initialize compatible weights from a V3 checkpoint')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    parser.add_argument('--tag', default='lrrv_v4', type=str, help='experiment tag')
    parser.add_argument(
        '--vote', action='store_true',
        help='use semantic weighted voting during inference')
    parser.add_argument(
        '--selection-strategy', default=None,
        choices=['nll', 'geometric_vote', 'semantic_vote'],
        help='proposal selector; overrides the backward-compatible --vote flag')
    parser.add_argument(
        '--selection-temperature', default=0.1, type=float,
        help='softmax temperature for semantic weighted voting')
    parser.add_argument('--seed', default=8, type=int, help='random seed')
    parser.add_argument('--alpha-1', default=None, type=float,
                        help='override CPL intra-video ranking weight')
    parser.add_argument('--alpha-2', default=None, type=float,
                        help='override CPL Gaussian diversity weight')
    parser.add_argument('--event-alpha', default=None, type=float,
                        help='override the total BECL weight')
    parser.add_argument('--event-sep-weight', default=None, type=float,
                        help='override low-rank event/background separation weight')
    parser.add_argument('--event-text-weight', default=None, type=float,
                        help='override Event/text alignment weight')
    parser.add_argument('--event-context-weight', default=None, type=float,
                        help='override minimum-context boundary weight')
    parser.add_argument('--event-overlap-weight', default=None, type=float,
                        help='override interval-overlap weight')
    parser.add_argument('--mixture-pull-weight', default=None, type=float,
                        help='override within-mixture pulling weight')
    parser.add_argument('--mixture-intra-push-weight', default=None, type=float,
                        help='override within-mixture pushing weight')
    parser.add_argument('--mixture-inter-push-weight', default=None, type=float,
                        help='override between-mixture pushing weight')
    parser.add_argument('--select-on-val', dest='select_on_val',
                        action='store_true', default=True,
                        help='select checkpoints on validation (stage01 default)')
    parser.add_argument('--legacy-select-on-test', dest='select_on_val',
                        action='store_false',
                        help='legacy diagnostic only: select checkpoints on test')

    return parser.parse_args()


def resolve_selection_strategy(vote, explicit_strategy=None):
    """Map the legacy flag to the fixed V4 proposal selector."""
    if explicit_strategy is not None:
        return explicit_strategy
    return 'semantic_vote' if vote else 'nll'


def main(kargs):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    if kargs.selection_temperature <= 0:
        raise ValueError('--selection-temperature must be positive')

    seed = kargs.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate the timestamp once so logs and checkpoints from the same run
    # share an identical, easy-to-match name.
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if kargs.log_dir:
        Path(kargs.log_dir).mkdir(parents=True, exist_ok=True)
        log_filename = os.path.join(
            kargs.log_dir, "{}_{}.log".format(kargs.tag, run_timestamp))
    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    args = load_json(kargs.config_path)
    loss_overrides = {
        'alpha_1': kargs.alpha_1,
        'alpha_2': kargs.alpha_2,
        'event_alpha': kargs.event_alpha,
        'event_sep_weight': kargs.event_sep_weight,
        'event_text_weight': kargs.event_text_weight,
        'event_context_weight': kargs.event_context_weight,
        'event_overlap_weight': kargs.event_overlap_weight,
        'mixture_pull_weight': kargs.mixture_pull_weight,
        'mixture_intra_push_weight': kargs.mixture_intra_push_weight,
        'mixture_inter_push_weight': kargs.mixture_inter_push_weight,
    }
    active_overrides = {
        key: value for key, value in loss_overrides.items()
        if value is not None
    }
    args['loss'].update(active_overrides)
    args['ablation_overrides'] = active_overrides
    args['tag'] = kargs.tag
    args['run_timestamp'] = run_timestamp
    args['vote'] = kargs.vote
    args['selection_strategy'] = resolve_selection_strategy(
        kargs.vote, kargs.selection_strategy)
    args['selection_temperature'] = kargs.selection_temperature
    args['select_on_val'] = kargs.select_on_val
    logging.info(str(args))

    runner = MainRunner(args)

    if kargs.init_from_v3:
        runner._load_pretrained_model(kargs.init_from_v3)
    if kargs.resume:
        runner._load_model(kargs.resume)
    if kargs.eval:
        runner.eval()
        return
    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
