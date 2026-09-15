import logging

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

logger = logging.getLogger(__name__)

class MeterBase(object):
    """
    Interface
    """

    def __init__(self, cfg, writer: SummaryWriter, mode: str):
        """
        build meter
        :param cfg: config
        :param writer: initialized tensorboard summary writer
        :param mode: train, val or test mode, for tagging tensorboard, etc.
        """
        assert mode in ["train", "test", "val"], f"mode is invalid: {mode}"
        self.cfg = cfg
        self.writer = writer
        self.mode = mode

    def set_mode(self, mode: str):
        assert mode in ["train", "test", "val"], f"mode is invalid: {mode}"
        self.mode = mode

    @torch.no_grad()
    def update(self, inputs, outputs, global_step=None):
        """
        call on each step
        update inner status based on the input
        :param inputs: the dataloader outputs/model inputs
        :param outputs: the model outputs
        :param global_step: global step, use `self.step` if step is None
        """
        raise NotImplementedError

    @torch.no_grad()
    def summary(self, epoch):
        """
        call at the end of the epoch
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class DummySummaryWriter:
    """
    Issue: https://github.com/pytorch/pytorch/issues/24236
    """
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

class DummyMeter(MeterBase):

    def update(self, inputs, outputs, n=None, global_step=None):
        pass

    def summary(self, epoch):
        pass

    def reset(self):
        pass

def get_writer(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        return SummaryWriter(*args, **kwargs)
    else:
        return DummySummaryWriter()

def build_meter(cfg, writer: SummaryWriter, mode: str):
    logger.warning("Meter is not specified!")
    return DummyMeter(cfg=cfg, writer=writer, mode=mode)