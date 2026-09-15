#!/usr/bin/env python
"""Create a deterministic video-disjoint Charades validation split."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _read(path):
    with path.open(encoding='utf8') as handle:
        return json.load(handle)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def _video_is_validation(video_id, fraction, salt):
    digest = hashlib.sha1(
        '{}\0{}'.format(salt, video_id).encode('utf8')).digest()
    bucket = int.from_bytes(digest[:8], byteorder='big')
    return bucket / float(1 << 64) < fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/charades/train.json')
    parser.add_argument('--test', default='data/charades/test.json')
    parser.add_argument('--train-output', default='data/charades/train_qstg.json')
    parser.add_argument('--val-output', default='data/charades/val_qstg.json')
    parser.add_argument('--manifest-output',
                        default='data/charades/qstg_split_manifest.json')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--salt', default='qstg-charades-val-v1')
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be in (0, 1)')

    source_path = Path(args.source)
    rows = _read(source_path)
    test_rows = _read(Path(args.test)) if args.test else []
    source_videos = sorted({row[0] for row in rows})
    test_videos = {row[0] for row in test_rows}
    val_videos = {
        video for video in source_videos
        if _video_is_validation(video, args.val_fraction, args.salt)
    }
    if not val_videos:
        raise RuntimeError('deterministic rule selected no validation videos')
    if val_videos & test_videos:
        raise RuntimeError('validation/test video overlap detected')

    train_rows = [row for row in rows if row[0] not in val_videos]
    val_rows = [row for row in rows if row[0] in val_videos]
    if not train_rows or not val_rows:
        raise RuntimeError('split produced an empty train or validation set')
    if {row[0] for row in train_rows} & {row[0] for row in val_rows}:
        raise RuntimeError('train/validation video overlap detected')

    _write(Path(args.train_output), train_rows)
    _write(Path(args.val_output), val_rows)
    _write(Path(args.manifest_output), {
        'source': str(source_path),
        'salt': args.salt,
        'val_fraction': args.val_fraction,
        'source_queries': len(rows),
        'train_queries': len(train_rows),
        'val_queries': len(val_rows),
        'source_videos': len(source_videos),
        'train_videos': len({row[0] for row in train_rows}),
        'val_videos': len(val_videos),
        'test_videos': len(test_videos),
        'query_counts_by_video': {
            'train': Counter(row[0] for row in train_rows).most_common(3),
            'val': Counter(row[0] for row in val_rows).most_common(3),
        },
    })
    print('Charades QSTG split:', {
        'train_queries': len(train_rows),
        'val_queries': len(val_rows),
        'train_videos': len({row[0] for row in train_rows}),
        'val_videos': len(val_videos),
        'test_videos': len(test_videos),
        'salt': args.salt,
    })


if __name__ == '__main__':
    main()
