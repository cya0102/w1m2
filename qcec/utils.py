import contextlib
import hashlib
import json
import os
import random
import time

import numpy as np


def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)


@contextlib.contextmanager
def isolated_rng(seed):
    """Run a deterministic block without consuming the caller's RNG state.

    Evaluation uses this context so a validation pass cannot perturb the
    random streams used by the next training epoch.  ``seed`` controls all
    RNGs that can be touched by the CUDA-only training code.  CUDA states are
    saved only when CUDA is available, which keeps the helper usable in the
    CPU smoke-test environment as well.
    """
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None)
    try:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def stable_sample_seed(video_id, sentence):
    """Return a process-independent seed for one video/query pair."""
    payload = '{}\0{}'.format(video_id, sentence).encode('utf8')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    # Keep the value in the signed int64 range used by torch collate tensors.
    return int.from_bytes(digest, byteorder='little', signed=False) & ((1 << 63) - 1)


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_digest(root):
    """Return a stable digest for the source/config files in ``root``.

    Runtime outputs, datasets and Python bytecode are deliberately excluded;
    the result is intended to identify the implementation used by an
    experiment rather than its generated artifacts.
    """
    root = os.path.abspath(os.fspath(root))
    included_suffixes = {'.py', '.json', '.sh', '.txt'}
    excluded_parts = {
        '__pycache__', '.git', 'data', 'logs', 'checkpoints',
    }
    digest = hashlib.sha256()
    paths = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in excluded_parts and not name.startswith('.'))
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.splitext(filename)[1].lower() in included_suffixes:
                paths.append(path)
    for path in sorted(paths):
        relative = os.path.relpath(path, root).replace(os.sep, '/')
        digest.update(relative.encode('utf8'))
        digest.update(b'\0')
        with open(path, 'rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b'\0')
    return digest.hexdigest()


def iou(pred, gt):
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * (inter + 1e-10) / (union + 1e-10)
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n
