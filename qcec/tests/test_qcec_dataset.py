import json
from collections import Counter
import os

import numpy as np
import torch

from datasets.base import BaseDataset, build_collate_data


class FixedFeatureDataset(BaseDataset):
    def _load_frame_features(self, vid):
        return np.arange(12, dtype=np.float32).reshape(6, 2)


def make_vocab():
    return {
        'counter': Counter({'a': 2, 'man': 2, 'running': 2, 'quickly': 2}),
        'w2id': {'a': 1, 'man': 2, 'running': 3, 'quickly': 4},
        'id2vec': [np.zeros(3, dtype=np.float32)] + [
            np.full(3, index, dtype=np.float32) for index in range(1, 5)
        ],
    }


def test_roles_are_aligned_after_oov_filtering(tmp_path):
    data_path = tmp_path / 'data.json'
    data_path.write_text(json.dumps([
        ['video', 10.0, [1.0, 2.0], 'A missing man running quickly']
    ]))
    dataset = FixedFeatureDataset(
        str(data_path), make_vocab(), {
            'max_num_frames': 4, 'max_num_words': 8, 'vocab_size': 8,
            'dataset': 'Synthetic', 'feature_path': 'unused',
            'frame_dim': 2, 'word_dim': 3, 'return_query_roles': True,
        })
    sample = dataset[0]
    assert sample['words_id'] == [1, 2, 3, 4]
    assert sample['query_role_mask'].shape == (3, 4)
    assert sample['query_role_valid'].tolist() == [True, True, True]
    # running and its adjacent adverb are predicate tokens; a/man are nouns.
    assert sample['query_role_mask'][0].tolist() == [False, False, True, True]
    assert sample['query_role_mask'][1].tolist() == [True, True, False, False]


def test_collate_returns_optional_qcec_contract():
    sample = {
        'raw': ['video', 10.0, [1.0, 2.0], 'sentence'],
        'frames_feat': np.zeros((4, 2), dtype=np.float32),
        'words_feat': [np.zeros(3, dtype=np.float32)] * 3,
        'words_id': [1, 2], 'weights': [1, 1],
        'query_role_mask': np.ones((3, 2), dtype=bool),
        'query_role_valid': np.ones(3, dtype=bool),
        'qcec_cluster_ids': np.array([0, 0, 1, 1], dtype=np.int64),
        'qcec_cluster_bounds': np.array(
            [[0.0, 0.5], [0.5, 1.0], [0.0, 0.0]], dtype=np.float32),
        'qcec_cluster_mask': np.array([True, True, False]),
    }
    batch = build_collate_data(
        4, 4, 2, 3, qcec_num_clusters=3,
        return_query_roles=True)([sample])
    net_input = batch['net_input']
    assert net_input['query_role_mask'].shape == (1, 3, 2)
    assert net_input['query_role_mask'].dtype == torch.bool
    assert net_input['qcec_cluster_ids'].shape == (1, 4)
    assert net_input['qcec_cluster_bounds'].shape == (1, 3, 2)
    assert net_input['qcec_cluster_mask'].dtype == torch.bool


def test_dataset_validates_and_indexes_video_cluster_metadata(tmp_path):
    data_path = tmp_path / 'data.json'
    data_path.write_text(json.dumps([
        ['video', 10.0, [1.0, 2.0], 'a man running']
    ]))
    index_path = tmp_path / 'clusters.npz'
    metadata = json.dumps({
        'dataset': 'Synthetic', 'max_num_frames': 4,
        'num_clusters': 3, 'feature_dim': 2,
        'feature_path': 'unused', 'l2_normalize': True,
    })
    np.savez_compressed(
        str(index_path),
        video_ids=np.asarray(['video']),
        cluster_ids=np.asarray([[0, 0, 1, 2]], dtype=np.int64),
        cluster_bounds=np.asarray([[
            [0.0, 0.5], [0.5, 0.75], [0.75, 1.0]
        ]], dtype=np.float32),
        cluster_mask=np.asarray([[True, True, True]]),
        metadata_json=np.asarray(metadata),
    )
    dataset = FixedFeatureDataset(
        str(data_path), make_vocab(), {
            'max_num_frames': 4, 'max_num_words': 8, 'vocab_size': 8,
            'dataset': 'Synthetic', 'feature_path': 'unused',
            'frame_dim': 2, 'word_dim': 3,
            'qcec_cluster_index_path': os.fspath(index_path),
            'qcec_num_clusters': 3,
        })
    sample = dataset[0]
    assert sample['qcec_cluster_ids'].tolist() == [0, 0, 1, 2]
    assert sample['qcec_cluster_mask'].tolist() == [True, True, True]
