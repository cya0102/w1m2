import json
from pathlib import Path


def test_charades_qstg_split_is_video_disjoint_and_complete():
    root = Path(__file__).resolve().parents[1] / 'data' / 'charades'
    source = json.load((root / 'train.json').open(encoding='utf8'))
    train = json.load((root / 'train_qstg.json').open(encoding='utf8'))
    validation = json.load((root / 'val_qstg.json').open(encoding='utf8'))
    test = json.load((root / 'test.json').open(encoding='utf8'))
    manifest = json.load(
        (root / 'qstg_split_manifest.json').open(encoding='utf8'))

    train_videos = {row[0] for row in train}
    validation_videos = {row[0] for row in validation}
    test_videos = {row[0] for row in test}
    assert len(train) + len(validation) == len(source)
    assert not train_videos & validation_videos
    assert not validation_videos & test_videos
    assert manifest['train_queries'] == len(train)
    assert manifest['val_queries'] == len(validation)
    assert manifest['train_videos'] == len(train_videos)
    assert manifest['val_videos'] == len(validation_videos)
