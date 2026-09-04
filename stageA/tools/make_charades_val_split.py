"""Create the registered video-disjoint Charades validation split."""

import argparse
import hashlib
import json
from pathlib import Path


def source_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_split(args):
    input_path = Path(args.input).resolve()
    with input_path.open(encoding='utf8') as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError('Charades source must be a list of query records')
    video_ids = sorted({str(record[0]) for record in data})
    if not video_ids:
        raise ValueError('Charades source has no videos')
    val_count = max(1, int(round(len(video_ids) * args.val_fraction)))
    ranked = sorted(
        video_ids,
        key=lambda video: hashlib.sha256(
            '{}:{}'.format(args.seed, video).encode('utf8')).digest())
    val_videos = set(ranked[:val_count])
    train = [record for record in data if str(record[0]) not in val_videos]
    val = [record for record in data if str(record[0]) in val_videos]
    if not train or not val:
        raise ValueError('split would produce an empty train or validation set')
    train_videos = {str(record[0]) for record in train}
    val_videos_from_rows = {str(record[0]) for record in val}
    if train_videos.intersection(val_videos_from_rows):
        raise AssertionError('Charades split is not video-disjoint')
    for path, rows in ((args.train_output, train), (args.val_output, val)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf8') as handle:
            json.dump(rows, handle, indent=2)
    manifest = {
        'dataset': 'charades',
        'seed': args.seed,
        'val_fraction': args.val_fraction,
        'source_path': str(input_path),
        'source_sha256': source_sha256(input_path),
        'source_query_count': len(data),
        'source_video_count': len(video_ids),
        'train_query_count': len(train),
        'val_query_count': len(val),
        'train_video_count': len(train_videos),
        'val_video_count': len(val_videos_from_rows),
        'train_videos': sorted(train_videos),
        'val_videos': sorted(val_videos_from_rows),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print('saved Charades train split to {}'.format(args.train_output))
    print('saved Charades val split to {}'.format(args.val_output))
    print('saved split manifest to {}'.format(args.manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--train-output', required=True)
    parser.add_argument('--val-output', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seed', type=int, default=20260902)
    parser.add_argument('--val-fraction', type=float, default=0.10)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be between zero and one')
    make_split(args)


if __name__ == '__main__':
    main()
