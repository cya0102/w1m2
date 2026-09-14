import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_json
try:
    import nltk
except ImportError:  # Keep pure query-unit tests dependency-light.
    nltk = None


UNIT_ACTION = 0
UNIT_ENTITY = 1
UNIT_RELATION = 2
UNIT_WHOLE = 3


def _is_content_tag(tag):
    return (tag.startswith('NN') or tag.startswith('PRP') or
            tag.startswith('VB') or tag.startswith('JJ') or
            tag.startswith('RB') or tag == 'CD')


def _same_segment(pos_tags, left, right):
    """Return whether a token span contains no punctuation boundary."""
    if left > right:
        left, right = right, left
    return not any(tag in {'.', ',', ':', ';', '!', '?', '(', ')'}
                   for tag in pos_tags[left:right + 1])


def build_query_units(words, pos_tags, max_units, include_whole_query=True,
                      type_weights=None):
    """Build deterministic sparse semantic units over retained query tokens.

    The returned token axis is the *post-vocabulary-filter* token axis.  This
    is important because the first query state in ``CPL`` is a learned start
    state and is not part of a unit mask.
    """
    if len(words) != len(pos_tags):
        raise ValueError('words and pos_tags must have the same length')
    if max_units < 1:
        raise ValueError('max_units must be positive')
    type_weights = type_weights or {
        UNIT_ACTION: 1.5,
        UNIT_ENTITY: 1.0,
        UNIT_RELATION: 1.0,
        UNIT_WHOLE: 0.25,
    }
    num_words = len(words)
    candidates = []

    def add(indices, unit_type):
        indices = sorted(set(i for i in indices if 0 <= i < num_words))
        if not indices:
            return
        mask = tuple(indices)
        for item_index, item in enumerate(candidates):
            if item[0] == mask:
                # Keep the explicit whole-query fallback even when a short
                # query makes an action/relation mask identical to it.
                if unit_type == UNIT_WHOLE:
                    candidates[item_index] = (mask, unit_type)
                return
        candidates.append((mask, unit_type))

    # Action units include nearby adverbs/particles and the nearest nominal
    # arguments in the same punctuation segment.  The sparse mask keeps the
    # rule useful for both short grounding queries and longer descriptions.
    for index, tag in enumerate(pos_tags):
        if not tag.startswith('VB'):
            continue
        indices = [index]
        cursor = index - 1
        while cursor >= 0 and pos_tags[cursor].startswith(('RB', 'RP')):
            indices.append(cursor)
            cursor -= 1
        cursor = index + 1
        while cursor < num_words and pos_tags[cursor].startswith(('RB', 'RP')):
            indices.append(cursor)
            cursor += 1
        for direction in (-1, 1):
            cursor = index + direction
            while 0 <= cursor < num_words:
                if not _same_segment(pos_tags, index, cursor):
                    break
                if pos_tags[cursor].startswith(('NN', 'PRP')):
                    indices.append(cursor)
                    break
                cursor += direction
        add(indices, UNIT_ACTION)

    # Entity/detail units are maximal adjective/number/noun spans.
    index = 0
    while index < num_words:
        if not (pos_tags[index].startswith(('JJ', 'NN')) or
                pos_tags[index] == 'CD'):
            index += 1
            continue
        end = index + 1
        while end < num_words and (
                pos_tags[end].startswith(('JJ', 'NN')) or
                pos_tags[end] == 'CD'):
            end += 1
        add(range(index, end), UNIT_ENTITY)
        index = end

    # Relation units bind a preposition/particle to its closest content
    # token(s), again without crossing punctuation.
    for index, tag in enumerate(pos_tags):
        if tag not in {'IN', 'TO', 'RP'}:
            continue
        indices = [index]
        for direction in (-1, 1):
            cursor = index + direction
            while 0 <= cursor < num_words:
                if not _same_segment(pos_tags, index, cursor):
                    break
                if _is_content_tag(pos_tags[cursor]):
                    indices.append(cursor)
                    break
                cursor += direction
        add(indices, UNIT_RELATION)

    if include_whole_query and num_words:
        add(range(num_words), UNIT_WHOLE)

    # A query without a usable content unit still gets a stable whole-query
    # fallback.  Empty queries intentionally remain all-padding.
    if num_words and not candidates:
        add(range(num_words), UNIT_WHOLE)

    priority = {UNIT_ACTION: 0, UNIT_RELATION: 1, UNIT_ENTITY: 2,
                UNIT_WHOLE: 3}
    candidates.sort(key=lambda item: (priority[item[1]], item[0]))
    if include_whole_query:
        whole = [item for item in candidates if item[1] == UNIT_WHOLE]
        non_whole = [item for item in candidates if item[1] != UNIT_WHOLE]
        candidates = non_whole[:max(0, max_units - len(whole))] + whole
    else:
        candidates = candidates[:max_units]
    candidates = candidates[:max_units]

    token_mask = np.zeros((max_units, num_words), dtype=np.float32)
    type_ids = np.zeros((max_units,), dtype=np.int64)
    valid_mask = np.zeros((max_units,), dtype=np.bool_)
    weights = np.zeros((max_units,), dtype=np.float32)
    for unit_index, (indices, unit_type) in enumerate(candidates):
        token_mask[unit_index, list(indices)] = 1.0
        type_ids[unit_index] = unit_type
        valid_mask[unit_index] = True
        weights[unit_index] = float(type_weights.get(unit_type, 1.0))
    valid_weight_sum = weights.sum()
    if valid_weight_sum > 0:
        weights /= valid_weight_sum
    return {
        'unit_token_mask': token_mask,
        'unit_type_ids': type_ids,
        'unit_valid_mask': valid_mask,
        'unit_weights': weights,
    }

class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.query_unit_config = args.get('query_units', {}) or {}
        self.query_units_enabled = bool(
            self.query_unit_config.get('enabled', False))
        self.query_unit_max = int(
            self.query_unit_config.get('max_units', 12))

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
        if nltk is None:
            raise ImportError('NLTK is required to load the grounding dataset')
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)

        weights = [] # Probabilities to be masked
        words = []
        kept_tags = []
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
                kept_tags.append(tag)
        words_id = [self.keep_vocab[w] for w in words]
        if words:
            start_feature = self.vocab['id2vec'][self.vocab['w2id'][words[0]]]
        else:
            start_feature = np.zeros((self.vocab['id2vec'].shape[1],),
                                     dtype=np.float32)
        words_feat = [np.asarray(start_feature, dtype=np.float32)] # start
        words_feat.extend([self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words])
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        sample = {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            'raw': [vid, duration, timestamps, sentence]
        }
        if self.query_units_enabled:
            config_weights = {
                UNIT_ACTION: self.query_unit_config.get('action_weight', 1.5),
                UNIT_ENTITY: self.query_unit_config.get('entity_weight', 1.0),
                UNIT_RELATION: self.query_unit_config.get('relation_weight', 1.0),
                UNIT_WHOLE: self.query_unit_config.get('whole_weight', 0.25),
            }
            sample.update(build_query_units(
                words, kept_tags, self.query_unit_max,
                include_whole_query=self.query_unit_config.get(
                    'include_whole_query', True),
                type_weights=config_weights))
        return sample


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim,
                       query_unit_config=None):
    query_unit_config = query_unit_config or {}
    query_units_enabled = bool(query_unit_config.get('enabled', False))
    max_units = int(query_unit_config.get('max_units', 12))

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
        if query_units_enabled:
            unit_token_mask = np.zeros(
                [bsz, max_units, max(words_len)], dtype=np.float32)
            unit_type_ids = np.zeros([bsz, max_units], dtype=np.int64)
            unit_valid_mask = np.zeros([bsz, max_units], dtype=np.bool_)
            unit_weights = np.zeros([bsz, max_units], dtype=np.float32)
        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            if keep:
                weights[i, :keep] = tmp / np.sum(tmp)
            if query_units_enabled:
                sample_units = sample.get('unit_token_mask')
                if sample_units is not None:
                    unit_keep = min(sample_units.shape[0], max_units)
                    word_keep = min(sample_units.shape[1], max_num_words,
                                    words_id.shape[1])
                    unit_token_mask[i, :unit_keep, :word_keep] = \
                        sample_units[:unit_keep, :word_keep]
                    unit_type_ids[i, :unit_keep] = sample[
                        'unit_type_ids'][:unit_keep]
                    unit_valid_mask[i, :unit_keep] = sample[
                        'unit_valid_mask'][:unit_keep]
                    unit_weights[i, :unit_keep] = sample[
                        'unit_weights'][:unit_keep]
                    # A unit that became empty after truncation is padding.
                    unit_valid_mask[i] &= (unit_token_mask[i].sum(axis=-1) > 0)
                    unit_weights[i, ~unit_valid_mask[i]] = 0
                    total = unit_weights[i].sum()
                    if total > 0:
                        unit_weights[i] /= total

        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_id': torch.from_numpy(words_id),
                'weights': torch.from_numpy(weights),
                'words_len': torch.from_numpy(np.asarray(words_len)),
            }
        })
        if query_units_enabled:
            batch['net_input'].update({
                'unit_token_mask': torch.from_numpy(unit_token_mask),
                'unit_type_ids': torch.from_numpy(unit_type_ids),
                'unit_valid_mask': torch.from_numpy(unit_valid_mask),
                'unit_weights': torch.from_numpy(unit_weights),
            })
        return batch

    return collate_data
