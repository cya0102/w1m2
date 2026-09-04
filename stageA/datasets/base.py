import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_json
from datasets.event_boundaries import load_event_boundary_index

try:
    import nltk
except ImportError:  # Keep lightweight geometry/model tests dependency-free.
    import re

    class _NltkFallback:
        @staticmethod
        def pos_tag(tokens):
            return [(token, 'NN') for token in tokens]

        class tokenize:
            @staticmethod
            def word_tokenize(sentence):
                return re.findall(r"\w+|[^\w\s]", sentence,
                                  flags=re.UNICODE)

    nltk = _NltkFallback()

class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.event_boundary_index = None
        boundary_path = args.get('event_boundary_path')
        if boundary_path:
            dataset_name = str(args.get('dataset', '')).lower()
            expected_dataset = (
                'activitynet' if 'activity' in dataset_name else
                'charades' if 'charades' in dataset_name else None)
            self.event_boundary_index = load_event_boundary_index(
                boundary_path,
                expected_dataset=expected_dataset,
                strict=args.get('event_boundary_strict', True))

        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(args['vocab_size']):
            self.keep_vocab[w] = self.vocab_size

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _sample_frame_features(self, frames_feat):
        num_clips = self.num_clips
        keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)
        keep_idx = np.round(keep_idx).astype(np.int64)
        keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
        frames_feat1 = []
        for j in range(num_clips):
            s, e = keep_idx[j], keep_idx[j + 1]
            assert s <= e
            if s == e:
                frames_feat1.append(frames_feat[s])
            else:
                frames_feat1.append(frames_feat[s:e].mean(axis=0))
        return np.stack(frames_feat1, 0)

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
        words = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
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
                words.append(word)
        words_id = [self.keep_vocab[w] for w in words]
        words_feat = [self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)] # placeholder for the start token
        words_feat.extend([self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words])
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        
        sample = {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            # Keep the historical raw[0:4] contract and append a stable id for
            # deterministic evaluation masks.  The id is independent of
            # Python's process-randomized hash implementation.
            'raw': [vid, duration, timestamps, sentence,
                    '{}:{}'.format(vid, index)]
        }
        if self.event_boundary_index is not None:
            boundary_positions, boundary_scores = self.event_boundary_index.get(vid)
            sample.update({
                'event_boundary_positions': boundary_positions,
                'event_boundary_scores': boundary_scores,
            })
        return sample


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim):
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
        boundary_lengths = [
            len(sample.get('event_boundary_positions', []))
            for sample in samples
        ]
        max_boundaries = max(1, max(boundary_lengths, default=0))
        boundary_positions = np.zeros(
            [bsz, max_boundaries], dtype=np.float32)
        boundary_scores = np.zeros(
            [bsz, max_boundaries], dtype=np.float32)
        boundary_mask = np.zeros(
            [bsz, max_boundaries], dtype=np.bool_)
        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)
            sample_positions = np.asarray(
                sample.get('event_boundary_positions', []), dtype=np.float32)
            sample_scores = np.asarray(
                sample.get('event_boundary_scores', []), dtype=np.float32)
            if sample_positions.size != sample_scores.size:
                raise ValueError(
                    'event boundary positions and scores must have equal length')
            count = min(sample_positions.size, max_boundaries)
            if count:
                boundary_positions[i, :count] = sample_positions[:count]
                boundary_scores[i, :count] = sample_scores[:count]
                boundary_mask[i, :count] = True

        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_id': torch.from_numpy(words_id),
                'weights': torch.from_numpy(weights),
                'words_len': torch.from_numpy(np.asarray(words_len)),
                'event_boundary_positions': torch.from_numpy(boundary_positions),
                'event_boundary_scores': torch.from_numpy(boundary_scores),
                'event_boundary_mask': torch.from_numpy(boundary_mask),
            }
        })
        return batch

    return collate_data
