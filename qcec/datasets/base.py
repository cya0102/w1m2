import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_json, stable_sample_seed
try:
    import nltk
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    nltk = None


def _fallback_tag_tokens(sentence):
    """Small dependency-free tokenizer/POS fallback for smoke environments."""
    import re

    tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", sentence)
    verbs = {
        'am', 'are', 'be', 'being', 'been', 'can', 'come', 'cut', 'do',
        'does', 'doing', 'drink', 'drive', 'eat', 'fall', 'get', 'give',
        'go', 'hold', 'jump', 'make', 'move', 'open', 'pick', 'play',
        'put', 'read', 'run', 'sit', 'stand', 'take', 'talk', 'turn',
        'use', 'walk', 'watch', 'wear', 'work',
    }
    tagged = []
    for token in tokens:
        lowered = token.lower()
        if lowered in verbs or lowered.endswith(('ing', 'ed')):
            tag = 'VB'
        elif lowered.endswith('ly'):
            tag = 'RB'
        elif lowered.endswith(('ous', 'ful', 'ive', 'al')):
            tag = 'JJ'
        else:
            tag = 'NN' if token[:1].isalnum() else '.'
        tagged.append((token, tag))
    return tagged


def _tag_sentence(sentence):
    if nltk is None:
        return _fallback_tag_tokens(sentence)
    try:
        tokens = nltk.tokenize.word_tokenize(sentence)
        return nltk.pos_tag(tokens)
    except LookupError:
        # NLTK is installed in many training environments but its punkt/tagger
        # data is not always provisioned in lightweight CI images.
        return _fallback_tag_tokens(sentence)


def sample_frame_features(frames_feat, num_clips):
    """Mean-pool a variable-length feature sequence into ``num_clips`` steps.

    This is the single sampling implementation used by both the dataset and
    the offline QCEC index builder.  Keeping it at module scope prevents the
    cluster boundaries from drifting away from the features seen by ``CPL``.
    """
    frames_feat = np.asarray(frames_feat)
    if frames_feat.ndim != 2:
        raise ValueError("frames_feat must have shape [T, D]")
    if len(frames_feat) < 1:
        raise ValueError("cannot sample an empty feature sequence")
    if int(num_clips) < 1:
        raise ValueError("num_clips must be positive")

    num_clips = int(num_clips)
    keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)
    keep_idx = np.round(keep_idx).astype(np.int64)
    keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
    frames_feat1 = []
    for j in range(num_clips):
        s, e = keep_idx[j], keep_idx[j + 1]
        if s > e:
            raise ValueError("sampling boundaries must be monotonic")
        if s == e:
            frames_feat1.append(frames_feat[s])
        else:
            frames_feat1.append(frames_feat[s:e].mean(axis=0))
    return np.stack(frames_feat1, 0).astype(np.float32, copy=False)


def _decode_metadata(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf8")
    if not isinstance(value, str):
        raise ValueError("QCEC metadata_json must contain a JSON string")
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid QCEC metadata_json") from exc


def _paths_match(first, second):
    if first is None or second is None:
        return True
    first = os.fspath(first)
    second = os.fspath(second)
    if first == second:
        return True
    return os.path.realpath(first) == os.path.realpath(second)

class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.return_query_roles = bool(args.get('return_query_roles', False))
        self.qcec_cluster_index_path = args.get('qcec_cluster_index_path')
        self.qcec_num_clusters = args.get('qcec_num_clusters')
        if (self.qcec_num_clusters is not None
                and int(self.qcec_num_clusters) < 1):
            raise ValueError('qcec_num_clusters must be positive')

        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(args['vocab_size']):
            self.keep_vocab[w] = self.vocab_size

        self._qcec_cluster_ids = None
        self._qcec_cluster_bounds = None
        self._qcec_cluster_mask = None
        self._qcec_video_to_index = None
        if self.qcec_cluster_index_path:
            self._load_qcec_cluster_index(self.qcec_cluster_index_path)

    def _load_qcec_cluster_index(self, path):
        """Load and validate the immutable video-level QCEC metadata."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                "QCEC cluster index does not exist: {}".format(path))

        with np.load(path, allow_pickle=False) as index:
            required = {
                'video_ids', 'cluster_ids', 'cluster_bounds',
                'cluster_mask', 'metadata_json',
            }
            missing = sorted(required.difference(index.files))
            if missing:
                raise ValueError(
                    "QCEC cluster index {} is missing {}".format(
                        path, ', '.join(missing)))
            video_ids = np.asarray(index['video_ids'])
            cluster_ids = np.asarray(index['cluster_ids'])
            cluster_bounds = np.asarray(index['cluster_bounds'])
            cluster_mask = np.asarray(index['cluster_mask'])
            metadata = _decode_metadata(index['metadata_json'])

        if video_ids.ndim != 1:
            raise ValueError("QCEC video_ids must have shape [V]")
        if cluster_ids.ndim != 2:
            raise ValueError("QCEC cluster_ids must have shape [V, T]")
        if cluster_bounds.ndim != 3 or cluster_bounds.shape[-1] != 2:
            raise ValueError("QCEC cluster_bounds must have shape [V, M, 2]")
        if cluster_mask.ndim != 2:
            raise ValueError("QCEC cluster_mask must have shape [V, M]")
        if (cluster_ids.shape[0] != len(video_ids)
                or cluster_bounds.shape[0] != len(video_ids)
                or cluster_mask.shape[0] != len(video_ids)):
            raise ValueError("QCEC index arrays disagree on video count")
        if cluster_ids.shape[1] != self.max_num_frames:
            raise ValueError(
                "QCEC index T={} does not match configured max_num_frames={}"
                .format(cluster_ids.shape[1], self.max_num_frames))
        if cluster_bounds.shape[1] != cluster_mask.shape[1]:
            raise ValueError("QCEC bounds and masks disagree on cluster count")
        if self.qcec_num_clusters is not None and int(
                self.qcec_num_clusters) != cluster_bounds.shape[1]:
            raise ValueError(
                "QCEC index M={} does not match configured qcec_num_clusters={}"
                .format(cluster_bounds.shape[1], self.qcec_num_clusters))
        if not np.isfinite(cluster_bounds).all():
            raise ValueError("QCEC cluster bounds contain non-finite values")
        if ((cluster_bounds < 0).any() or (cluster_bounds > 1).any()):
            raise ValueError("QCEC cluster bounds must be normalized to [0, 1]")
        if ((cluster_bounds[..., 1] < cluster_bounds[..., 0]).any()):
            raise ValueError("QCEC cluster bounds must have start <= end")
        num_clusters = cluster_bounds.shape[1]
        if num_clusters < 1:
            raise ValueError("QCEC cluster index must contain at least one cluster")
        if ((cluster_ids < -1).any()
                or (cluster_ids >= num_clusters).any()):
            raise ValueError("QCEC cluster ids must be -1 or valid cluster indices")
        valid_cluster_bounds = cluster_mask & (
            cluster_bounds[..., 1] > cluster_bounds[..., 0])
        if (cluster_mask & ~valid_cluster_bounds).any():
            raise ValueError(
                "valid QCEC clusters must have positive-width bounds")
        safe_ids = np.clip(cluster_ids, 0, num_clusters - 1)
        assigned_mask = np.take_along_axis(
            cluster_mask[:, None, :], safe_ids[..., None], axis=2).squeeze(2)
        if ((cluster_ids >= 0) & ~assigned_mask).any():
            raise ValueError(
                "QCEC frames may only reference valid cluster slots")

        metadata_t = metadata.get('max_num_frames')
        if metadata_t is not None and int(metadata_t) != self.max_num_frames:
            raise ValueError(
                "QCEC metadata max_num_frames={} does not match {}".format(
                    metadata_t, self.max_num_frames))
        metadata_m = metadata.get('num_clusters')
        if metadata_m is not None and int(metadata_m) != cluster_bounds.shape[1]:
            raise ValueError(
                "QCEC metadata num_clusters={} does not match index M={}"
                .format(metadata_m, cluster_bounds.shape[1]))
        metadata_feature_dim = metadata.get('feature_dim')
        configured_feature_dim = self.args.get('frame_dim')
        if (metadata_feature_dim is not None and configured_feature_dim is not None
                and int(metadata_feature_dim) != int(configured_feature_dim)):
            raise ValueError(
                "QCEC metadata feature_dim={} does not match {}".format(
                    metadata_feature_dim, configured_feature_dim))
        configured_normalize = self.args.get('qcec_l2_normalize')
        if (configured_normalize is not None
                and 'l2_normalize' in metadata
                and bool(configured_normalize) != bool(metadata['l2_normalize'])):
            raise ValueError(
                "QCEC index normalization does not match dataset config")
        metadata_dataset = metadata.get('dataset')
        configured_dataset = self.args.get('dataset')
        if (metadata_dataset is not None and configured_dataset is not None
                and str(metadata_dataset) != str(configured_dataset)):
            raise ValueError(
                "QCEC metadata dataset={} does not match {}".format(
                    metadata_dataset, configured_dataset))
        metadata_feature_path = metadata.get('feature_path')
        if metadata_feature_path and not _paths_match(
                metadata_feature_path, self.args.get('feature_path')):
            raise ValueError(
                "QCEC metadata feature_path={} does not match {}".format(
                    metadata_feature_path, self.args.get('feature_path')))

        ids = [
            value.decode('utf8') if isinstance(value, bytes) else str(value)
            for value in video_ids.tolist()
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("QCEC cluster index contains duplicate video ids")
        self._qcec_video_to_index = {
            video_id: row for row, video_id in enumerate(ids)}
        self._qcec_cluster_ids = cluster_ids.astype(np.int64, copy=False)
        self._qcec_cluster_bounds = cluster_bounds.astype(
            np.float32, copy=False)
        self._qcec_cluster_mask = cluster_mask.astype(bool, copy=False)

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _sample_frame_features(self, frames_feat):
        return sample_frame_features(frames_feat, self.num_clips)

    @property
    def num_clips(self):
        return self.max_num_frames

    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)

        weights = [] # Probabilities to be masked
        tagged_words = []
        for word, tag in _tag_sentence(sentence):
            word = word.lower()
            if word in self.keep_vocab:
                if 'NN' in tag:
                    weights.append(2)
                elif 'VB' in tag:
                    weights.append(2)
                elif 'JJ' in tag or 'RB' in tag:
                    weights.append(2)
                else:
                    weights.append(1)
                tagged_words.append((word, tag))
        words = [word for word, _ in tagged_words]
        query_role_mask = None
        query_role_valid = None
        if self.return_query_roles:
            tags = [tag for _, tag in tagged_words]
            predicate = [tag.startswith('VB') for tag in tags]
            entity = [tag.startswith('NN') for tag in tags]
            for token_index, tag in enumerate(tags):
                if tag.startswith('RB'):
                    adjacent_predicate = (
                        (token_index > 0 and predicate[token_index - 1])
                        or (token_index + 1 < len(tags)
                            and predicate[token_index + 1]))
                    predicate[token_index] = adjacent_predicate
                if tag.startswith('JJ'):
                    adjacent_entity = (
                        (token_index > 0 and entity[token_index - 1])
                        or (token_index + 1 < len(tags)
                            and entity[token_index + 1]))
                    entity[token_index] = adjacent_entity
            sentence_role = [True for _ in words]
            query_role_mask = np.asarray(
                [predicate, entity, sentence_role], dtype=bool)
            query_role_valid = np.asarray(
                [any(predicate), any(entity), True], dtype=bool)
        words_id = [self.keep_vocab[w] for w in words]
        words_feat = [self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)] # placeholder for the start token
        words_feat.extend([self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words])
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        
        sample = {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            # Evaluation masks are derived from the sample identity rather
            # than the DataLoader order.  This makes a fixed checkpoint
            # comparable across batch sizes and worker counts.
            'eval_mask_seed': stable_sample_seed(vid, sentence),
            'raw': [vid, duration, timestamps, sentence]
        }
        if self.return_query_roles:
            sample['query_role_mask'] = query_role_mask
            sample['query_role_valid'] = query_role_valid
        if self.qcec_cluster_index_path:
            row = self._qcec_video_to_index.get(str(vid))
            if row is None:
                raise KeyError(
                    "video id {} at dataset index {} is missing from QCEC "
                    "cluster index {}".format(
                        vid, index, self.qcec_cluster_index_path))
            sample['qcec_cluster_ids'] = self._qcec_cluster_ids[row].copy()
            sample['qcec_cluster_bounds'] = self._qcec_cluster_bounds[row].copy()
            sample['qcec_cluster_mask'] = self._qcec_cluster_mask[row].copy()
        return sample


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim,
                       qcec_num_clusters=None, return_query_roles=False):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []
        words_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_id']), max_num_words))

        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max(words_len) + 1, word_dim]).astype(np.float32)
        words_id = np.zeros([bsz, max(words_len)]).astype(np.int64)
        weights = np.zeros([bsz, max(words_len)]).astype(np.float32)
        if return_query_roles:
            query_role_mask = np.zeros(
                [bsz, 3, max(words_len)], dtype=bool)
            query_role_valid = np.zeros([bsz, 3], dtype=bool)
        if qcec_num_clusters is not None:
            qcec_cluster_ids = np.full(
                [bsz, max_num_frames], -1, dtype=np.int64)
            qcec_cluster_bounds = np.zeros(
                [bsz, int(qcec_num_clusters), 2], dtype=np.float32)
            qcec_cluster_mask = np.zeros(
                [bsz, int(qcec_num_clusters)], dtype=bool)
        eval_mask_seeds = np.zeros([bsz], dtype=np.int64)
        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / max(np.sum(tmp), 1e-12)
            eval_mask_seeds[i] = int(sample.get(
                'eval_mask_seed',
                stable_sample_seed(sample['raw'][0], sample['raw'][3])))
            if return_query_roles:
                if 'query_role_mask' not in sample:
                    raise ValueError(
                        "return_query_roles=True but sample has no role mask")
                role_mask = np.asarray(sample['query_role_mask'], dtype=bool)
                role_valid = np.asarray(sample['query_role_valid'], dtype=bool)
                if role_mask.shape[0] != 3 or role_mask.shape[1] < keep:
                    raise ValueError("sample query role mask has wrong shape")
                query_role_mask[i, :, :keep] = role_mask[:, :keep]
                query_role_valid[i] = role_valid
            if qcec_num_clusters is not None:
                if ('qcec_cluster_ids' not in sample
                        or 'qcec_cluster_bounds' not in sample
                        or 'qcec_cluster_mask' not in sample):
                    raise ValueError(
                        "qcec_num_clusters is set but sample has no cluster metadata")
                sample_ids = np.asarray(sample['qcec_cluster_ids'])
                sample_bounds = np.asarray(sample['qcec_cluster_bounds'])
                sample_mask = np.asarray(sample['qcec_cluster_mask'], dtype=bool)
                if sample_ids.shape[0] != max_num_frames:
                    raise ValueError("sample cluster ids do not match max_num_frames")
                if sample_bounds.shape != (int(qcec_num_clusters), 2):
                    raise ValueError("sample cluster bounds do not match M")
                if sample_mask.shape != (int(qcec_num_clusters),):
                    raise ValueError("sample cluster mask does not match M")
                qcec_cluster_ids[i] = sample_ids[:max_num_frames]
                qcec_cluster_bounds[i] = sample_bounds
                qcec_cluster_mask[i] = sample_mask

        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_id': torch.from_numpy(words_id),
                'weights': torch.from_numpy(weights),
                'words_len': torch.from_numpy(np.asarray(words_len)),
                'eval_mask_seeds': torch.from_numpy(eval_mask_seeds),
            }
        })
        if return_query_roles:
            batch['net_input'].update({
                'query_role_mask': torch.from_numpy(query_role_mask),
                'query_role_valid': torch.from_numpy(query_role_valid),
            })
        if qcec_num_clusters is not None:
            batch['net_input'].update({
                'qcec_cluster_ids': torch.from_numpy(qcec_cluster_ids),
                'qcec_cluster_bounds': torch.from_numpy(qcec_cluster_bounds),
                'qcec_cluster_mask': torch.from_numpy(qcec_cluster_mask),
            })
        return batch

    return collate_data
