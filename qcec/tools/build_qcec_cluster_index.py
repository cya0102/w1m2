"""Build the video-level contiguous Ward index used by QCEC.

The command intentionally performs clustering once per unique video rather
than once per query.  It uses the exact sampling helper from
``datasets.base`` and writes a portable, non-pickle NPZ file.
"""

import argparse
import heapq
import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.base import sample_frame_features  # noqa: E402
from utils import load_json  # noqa: E402


CLUSTER_ALGORITHM_VERSION = "contiguous_ward_v1"


def contiguous_ward_cluster(features, num_clusters, l2_normalize=True):
    """Return deterministic contiguous Ward clusters.

    Args:
        features: NumPy array with shape ``[T, D]``.
        num_clusters: Requested number of clusters.  If ``T`` is smaller,
            exactly ``T`` valid singleton clusters are returned and the
            remaining output slots are padding.
        l2_normalize: Normalize each frame before computing Ward costs.

    Returns:
        ``cluster_ids [T]``, ``cluster_bounds [M, 2]`` and
        ``cluster_mask [M]``.  Bounds are normalized half-open intervals.
    """
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [T, D]")
    num_frames, feature_dim = values.shape
    if num_frames < 1 or feature_dim < 1:
        raise ValueError("features must be non-empty")
    if int(num_clusters) < 1:
        raise ValueError("num_clusters must be positive")
    requested = int(num_clusters)
    valid_count = min(num_frames, requested)
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")

    if l2_normalize:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, 1e-12)

    # Each active node represents one contiguous segment.  We retain the
    # segment's sum and count, so Ward costs can be updated in O(D) after a
    # merge.  Heap tuple ordering gives a fixed left-to-right tie break.
    segment_sum = values.copy()
    segment_count = np.ones(num_frames, dtype=np.int64)
    segment_start = np.arange(num_frames, dtype=np.int64)
    segment_end = np.arange(num_frames, dtype=np.int64) + 1
    previous = np.arange(num_frames, dtype=np.int64) - 1
    following = np.arange(num_frames, dtype=np.int64) + 1
    following[-1] = -1
    active = np.ones(num_frames, dtype=bool)
    version = np.zeros(num_frames, dtype=np.int64)

    def ward_cost(left, right):
        left_count = float(segment_count[left])
        right_count = float(segment_count[right])
        left_mean = segment_sum[left] / left_count
        right_mean = segment_sum[right] / right_count
        difference = left_mean - right_mean
        return (left_count * right_count / (left_count + right_count)
                * float(np.dot(difference, difference)))

    heap = []
    for left in range(num_frames - 1):
        right = left + 1
        heapq.heappush(
            heap,
            (ward_cost(left, right), left, right,
             int(version[left]), int(version[right])),
        )

    active_count = num_frames
    while active_count > valid_count:
        while heap:
            _, left, right, left_version, right_version = heapq.heappop(heap)
            if (active[left] and active[right]
                    and following[left] == right
                    and version[left] == left_version
                    and version[right] == right_version):
                break
        else:
            raise RuntimeError("Ward heap exhausted before reaching M clusters")

        right_after = following[right]
        segment_sum[left] += segment_sum[right]
        segment_count[left] += segment_count[right]
        segment_end[left] = segment_end[right]
        following[left] = right_after
        if right_after >= 0:
            previous[right_after] = left
        active[right] = False
        version[left] += 1
        version[right] += 1
        active_count -= 1

        left_neighbor = previous[left]
        if left_neighbor >= 0:
            heapq.heappush(
                heap,
                (ward_cost(left_neighbor, left), left_neighbor, left,
                 int(version[left_neighbor]), int(version[left])),
            )
        if right_after >= 0:
            heapq.heappush(
                heap,
                (ward_cost(left, right_after), left, right_after,
                 int(version[left]), int(version[right_after])),
            )

    segments = []
    cursor = int(np.flatnonzero(active)[0])
    while cursor >= 0:
        segments.append((int(segment_start[cursor]), int(segment_end[cursor])))
        cursor = int(following[cursor])
    if len(segments) != valid_count:
        raise RuntimeError("invalid active Ward segment count")

    cluster_ids = np.full(num_frames, -1, dtype=np.int64)
    bounds = np.zeros((requested, 2), dtype=np.float32)
    mask = np.zeros(requested, dtype=bool)
    for cluster_index, (start, end) in enumerate(segments):
        cluster_ids[start:end] = cluster_index
        bounds[cluster_index] = (
            float(start) / float(num_frames),
            float(end) / float(num_frames),
        )
        mask[cluster_index] = True

    if (cluster_ids < 0).any():
        raise RuntimeError("Ward clustering left unassigned frames")
    return cluster_ids, bounds, mask


def _resolve_path(path, config_path):
    path = os.fspath(path)
    if os.path.isabs(path):
        return path
    current_path = os.path.abspath(path)
    if os.path.exists(current_path):
        return current_path
    project_path = os.path.join(str(PROJECT_ROOT), path)
    if os.path.exists(project_path):
        return os.path.abspath(project_path)
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(config_path)), path))


def _split_names(value):
    if isinstance(value, str):
        value = [value]
    names = []
    for group in value:
        names.extend(item.strip() for item in group.split(',') if item.strip())
    if not names:
        raise ValueError("at least one split is required")
    return names


def _video_ids_from_split(path):
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get('data', data.get('items', []))
    result = []
    for item in data:
        if not isinstance(item, (list, tuple)) or not item:
            raise ValueError("split {} contains an invalid sample".format(path))
        result.append(str(item[0]))
    return result


def _read_video_features(handle, dataset_name, video_id):
    if video_id not in handle:
        raise KeyError(
            "video id {} is missing from feature file".format(video_id))
    node = handle[video_id]
    if str(dataset_name).lower() == 'activitynet':
        if 'c3d_features' not in node:
            raise KeyError(
                "ActivityNet video {} has no c3d_features dataset".format(
                    video_id))
        node = node['c3d_features']
    return np.asarray(node, dtype=np.float32)


def build_index(config_path, split_names, num_clusters, output, overwrite=False):
    config_path = os.path.abspath(config_path)
    config = load_json(config_path)
    dataset_args = config['dataset']
    dataset_name = dataset_args['dataset']
    max_num_frames = int(dataset_args['max_num_frames'])
    l2_normalize = bool(dataset_args.get('qcec_l2_normalize', True))
    feature_path = _resolve_path(dataset_args['feature_path'], config_path)
    split_paths = []
    for split in split_names:
        key = '{}_data'.format(split)
        if key not in dataset_args or not dataset_args[key]:
            raise ValueError("config has no data path for split {}".format(split))
        split_paths.append(_resolve_path(dataset_args[key], config_path))

    if os.path.exists(output) and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing QCEC index {}; pass --overwrite"
            .format(output))
    output = os.path.abspath(output)
    requested_clusters = int(num_clusters)
    if requested_clusters < 1:
        raise ValueError("num_clusters must be positive")

    video_ids = sorted({
        video_id
        for split_path in split_paths
        for video_id in _video_ids_from_split(split_path)
    })
    if not video_ids:
        raise ValueError("the selected splits contain no videos")

    all_ids = []
    all_bounds = []
    all_masks = []
    feature_dim = None
    with h5py.File(feature_path, 'r') as handle:
        for video_id in video_ids:
            features = _read_video_features(handle, dataset_name, video_id)
            sampled = sample_frame_features(features, max_num_frames)
            if feature_dim is None:
                feature_dim = int(sampled.shape[1])
            elif sampled.shape[1] != feature_dim:
                raise ValueError("feature dimensions differ across videos")
            ids, bounds, mask = contiguous_ward_cluster(
                sampled, requested_clusters, l2_normalize=l2_normalize)
            all_ids.append(ids)
            all_bounds.append(bounds)
            all_masks.append(mask)

    metadata = {
        'algorithm_version': CLUSTER_ALGORITHM_VERSION,
        'dataset': dataset_name,
        'max_num_frames': max_num_frames,
        'num_clusters': requested_clusters,
        'feature_dim': feature_dim,
        'feature_path': feature_path,
        'l2_normalize': l2_normalize,
        'sampling': 'BaseDataset.sample_frame_features/v1',
    }
    output_parent = os.path.dirname(output) or os.curdir
    os.makedirs(output_parent, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb', suffix='.npz', dir=output_parent, delete=False) as fp:
            temporary_path = fp.name
            np.savez_compressed(
                fp,
                video_ids=np.asarray(video_ids, dtype=str),
                cluster_ids=np.stack(all_ids, axis=0),
                cluster_bounds=np.stack(all_bounds, axis=0),
                cluster_mask=np.stack(all_masks, axis=0),
                metadata_json=np.asarray(json.dumps(
                    metadata, sort_keys=True)),
            )
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return metadata, len(video_ids)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--splits', nargs='+', default=['train,val,test'])
    parser.add_argument('--num-clusters', type=int, default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(args.config_path)
    configured_clusters = config['dataset'].get('qcec_num_clusters', 32)
    num_clusters = (
        configured_clusters if args.num_clusters is None else args.num_clusters)
    metadata, video_count = build_index(
        args.config_path,
        _split_names(args.splits),
        num_clusters,
        args.output,
        overwrite=args.overwrite,
    )
    print("wrote {} videos to {} (M={}, T={})".format(
        video_count, os.path.abspath(args.output),
        metadata['num_clusters'], metadata['max_num_frames']))


if __name__ == '__main__':
    main()
