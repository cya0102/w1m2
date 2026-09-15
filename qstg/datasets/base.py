import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_json
import nltk


# Phrase type ids (see proposal section 3.2).
PHRASE_TYPE_PAD = 0
PHRASE_TYPE_GLOBAL = 1
PHRASE_TYPE_ACTION = 2
PHRASE_TYPE_ENTITY = 3
PHRASE_TYPE_ATTRIBUTE = 4
PHRASE_TYPE_RELATION = 5

# Query edge type ids (see proposal section 3.2).
EDGE_NONE = 0
EDGE_ASSOC = 1        # undirected co-occurrence
EDGE_BEFORE = 2       # source evidence happens before target evidence
EDGE_AFTER = 3        # source evidence happens after target evidence
EDGE_WHILE = 4        # simultaneous

_RELATION_SURFACE_TYPES = {
    'before': EDGE_BEFORE,
    'then': EDGE_BEFORE,
    'after': EDGE_AFTER,
    'while': EDGE_WHILE,
    'during': EDGE_WHILE,
    'with': EDGE_WHILE,
}
# Surface words that always introduce a relation phrase.  Relation
# satisfaction only fires on these explicit triggers (proposal 20.2).
_RELATION_SURFACE_WORDS = {
    'before', 'after', 'then', 'while', 'during', 'with', 'into', 'out', 'of',
}
_RELATION_TAGS = ('IN', 'TO', 'RP')
_REVERSE_DIRECTIONAL_EDGE = {EDGE_BEFORE: EDGE_AFTER, EDGE_AFTER: EDGE_BEFORE}


def _span_distance(a_start, a_end, b_start, b_end):
    """Token distance between two half-open spans; 0 when they touch/overlap."""
    if a_end <= b_start:
        return b_start - a_end + 1
    if b_end <= a_start:
        return a_start - b_end + 1
    return 0


def _reverse_edge_type(edge_type):
    return _REVERSE_DIRECTIONAL_EDGE.get(edge_type, edge_type)


def build_query_phrase_graph(words, tags, max_phrases, max_words,
                             max_actions=2, max_entities=3, max_relations=1):
    """Build the fixed-capacity query phrase graph from POS-tagged tokens.

    ``words``/``tags`` must already be OOV-filtered and truncated to
    ``max_words`` by the caller so phrase token indices stay aligned with the
    tokens that reach the model (proposal section 3.2).  The function is pure
    and deterministic: identical inputs produce identical tensors.
    """
    if max_phrases < 2:
        raise ValueError('max_phrases must be >= 2')
    if max_actions < 0 or max_entities < 0 or max_relations < 0:
        raise ValueError('phrase capacity limits must be non-negative')
    num_words = len(words)
    if num_words > max_words:
        raise ValueError(
            'build_query_phrase_graph expects tokens truncated to max_words; '
            'got {} > {}'.format(num_words, max_words))
    if num_words < 1:
        raise ValueError(
            'build_query_phrase_graph expects at least one token; callers '
            'must insert the internal <unk> placeholder for empty queries')

    consumed = [False] * num_words

    # ------------------------------------------------------------------
    # 1. ACTION candidates: adjacent VB* tokens merge into one phrase and
    #    absorb immediately neighbouring RB*/RP tokens unless the modifier
    #    is an explicit relation word (that word must stay available for
    #    the relation pass below).
    # ------------------------------------------------------------------
    action_spans = []
    index = 0
    while index < num_words:
        if not consumed[index] and tags[index].startswith('VB'):
            start = index
            end = index + 1
            while end < num_words and not consumed[end] \
                    and tags[end].startswith('VB'):
                end += 1
            left = start
            while left - 1 >= 0 and not consumed[left - 1] \
                    and (tags[left - 1].startswith('RB')
                         or tags[left - 1] == 'RP') \
                    and words[left - 1] not in _RELATION_SURFACE_WORDS:
                left -= 1
            right = end
            while right < num_words and not consumed[right] \
                    and (tags[right].startswith('RB')
                         or tags[right] == 'RP') \
                    and words[right] not in _RELATION_SURFACE_WORDS:
                right += 1
            for token in range(left, right):
                consumed[token] = True
            action_spans.append((left, right))
            index = right
        else:
            index += 1

    # ------------------------------------------------------------------
    # 2. ENTITY candidates: maximal contiguous JJ*/NN*/CD chunks that
    #    contain at least one NN*.  Chunks without a noun stay unconsumed
    #    so their modifiers can still become attributes.
    # ------------------------------------------------------------------
    entity_spans = []
    index = 0
    while index < num_words:
        if not consumed[index] and (
                tags[index].startswith('NN')
                or tags[index].startswith('JJ')
                or tags[index].startswith('CD')):
            start = index
            end = index + 1
            while end < num_words and not consumed[end] and (
                    tags[end].startswith('NN')
                    or tags[end].startswith('JJ')
                    or tags[end].startswith('CD')):
                end += 1
            if any(tags[token].startswith('NN')
                   for token in range(start, end)):
                for token in range(start, end):
                    consumed[token] = True
                entity_spans.append((start, end))
            index = end
        else:
            index += 1

    # ------------------------------------------------------------------
    # 3. RELATION candidates: explicit surface words first, then any
    #    unconsumed IN/TO/RP token.
    # ------------------------------------------------------------------
    relation_items = []
    for index in range(num_words):
        if consumed[index]:
            continue
        edge_type = None
        if words[index] in _RELATION_SURFACE_WORDS:
            edge_type = _RELATION_SURFACE_TYPES.get(words[index], EDGE_ASSOC)
        elif tags[index] in _RELATION_TAGS:
            edge_type = EDGE_ASSOC
        if edge_type is not None:
            consumed[index] = True
            relation_items.append((index, edge_type))

    # ------------------------------------------------------------------
    # 4. ATTRIBUTE candidates: unconsumed JJ*/RB* runs adjacent to an
    #    action or entity span.  Only kept when slots remain (enforced
    #    during capacity selection below).
    # ------------------------------------------------------------------
    attribute_spans = []
    index = 0
    while index < num_words:
        if not consumed[index] and (
                tags[index].startswith('JJ')
                or tags[index].startswith('RB')):
            start = index
            end = index + 1
            while end < num_words and not consumed[end] and (
                    tags[end].startswith('JJ')
                    or tags[end].startswith('RB')):
                end += 1
            adjacent = any(
                _span_distance(start, end, span[0], span[1]) <= 1
                for span in action_spans + entity_spans)
            if adjacent:
                attribute_spans.append((start, end))
            index = end
        else:
            index += 1

    # ------------------------------------------------------------------
    # 5. Capacity selection: GLOBAL > ACTION > ENTITY > RELATION >
    #    ATTRIBUTE, same priority ordered by sentence position.
    # ------------------------------------------------------------------
    selected = [(PHRASE_TYPE_GLOBAL, 0, num_words)]
    budget = max_phrases - 1

    def take_spans(spans, capacity, phrase_type):
        nonlocal budget
        for span in spans:
            if budget <= 0 or capacity <= 0:
                return
            selected.append((phrase_type, span[0], span[1]))
            budget -= 1
            capacity -= 1

    take_spans(action_spans[:max_actions], max_actions, PHRASE_TYPE_ACTION)
    take_spans(entity_spans[:max_entities], max_entities, PHRASE_TYPE_ENTITY)
    kept_relations = 0
    relation_types = {}
    for token, edge_type in relation_items:
        if budget <= 0 or kept_relations >= max_relations:
            break
        selected.append((PHRASE_TYPE_RELATION, token, token + 1))
        relation_types[token] = edge_type
        kept_relations += 1
        budget -= 1
    take_spans(attribute_spans, budget, PHRASE_TYPE_ATTRIBUTE)

    has_content = len(selected) > 1

    # ------------------------------------------------------------------
    # 6. Fixed tensors.
    # ------------------------------------------------------------------
    phrase_token_mask = np.zeros((max_phrases, num_words), dtype=bool)
    phrase_type = np.full(max_phrases, PHRASE_TYPE_PAD, dtype=np.int64)
    phrase_valid = np.zeros(max_phrases, dtype=bool)
    phrase_required = np.zeros(max_phrases, dtype=bool)
    query_edge_type = np.full((max_phrases, max_phrases), EDGE_NONE,
                              dtype=np.int64)

    phrase_type[0] = PHRASE_TYPE_GLOBAL
    phrase_valid[0] = True
    phrase_token_mask[0, :] = True

    content_slots = []
    action_slots = []
    entity_slots = []
    relation_slots = {}
    attribute_slots = []
    for slot, (ptype, start, end) in enumerate(selected[1:], start=1):
        phrase_type[slot] = ptype
        phrase_valid[slot] = True
        phrase_required[slot] = True
        phrase_token_mask[slot, start:end] = True
        content_slots.append(slot)
        if ptype == PHRASE_TYPE_ACTION:
            action_slots.append(slot)
        elif ptype == PHRASE_TYPE_ENTITY:
            entity_slots.append(slot)
        elif ptype == PHRASE_TYPE_RELATION:
            relation_slots[slot] = start
        elif ptype == PHRASE_TYPE_ATTRIBUTE:
            attribute_slots.append(slot)
    if not has_content:
        phrase_required[0] = True

    def span_of(slot):
        return selected[slot][1], selected[slot][2]

    def set_edge(source, target, edge_type):
        query_edge_type[source, target] = edge_type
        query_edge_type[target, source] = _reverse_edge_type(edge_type)

    # GLOBAL <-> every content phrase: undirected, propagation only.
    for slot in content_slots:
        set_edge(0, slot, EDGE_ASSOC)

    # ACTION <-> nearest ENTITY to the right (fall back to the left).
    for action in action_slots:
        a_start, a_end = span_of(action)
        right = [
            (_span_distance(a_start, a_end, span_of(slot)[0], span_of(slot)[1]), slot)
            for slot in entity_slots if span_of(slot)[0] >= a_end]
        left = [
            (_span_distance(a_start, a_end, span_of(slot)[0], span_of(slot)[1]), slot)
            for slot in entity_slots if span_of(slot)[1] <= a_start]
        pool = right if right else left
        if pool:
            _, entity = min(pool)
            set_edge(action, entity, EDGE_ASSOC)

    # ATTRIBUTE <-> nearest ENTITY/ACTION host.
    for attribute in attribute_slots:
        a_start, a_end = span_of(attribute)
        hosts = [
            (_span_distance(a_start, a_end, span_of(slot)[0], span_of(slot)[1]), slot)
            for slot in action_slots + entity_slots]
        if hosts:
            _, host = min(hosts)
            set_edge(attribute, host, EDGE_ASSOC)

    # RELATION: link the relation word to its left/right content phrases and
    # mediate a typed content-content edge between them.
    for relation, r_token in relation_slots.items():
        edge_type = relation_types[r_token]
        left = [
            (_span_distance(span_of(slot)[0], span_of(slot)[1],
                            r_token, r_token + 1), slot)
            for slot in action_slots + entity_slots
            if span_of(slot)[1] <= r_token]
        right = [
            (_span_distance(span_of(slot)[0], span_of(slot)[1],
                            r_token, r_token + 1), slot)
            for slot in action_slots + entity_slots
            if span_of(slot)[0] > r_token]
        if left:
            _, left_slot = min(left)
            set_edge(relation, left_slot, EDGE_ASSOC)
        if right:
            _, right_slot = min(right)
            set_edge(relation, right_slot, EDGE_ASSOC)
        if left and right:
            _, left_slot = min(left)
            _, right_slot = min(right)
            set_edge(left_slot, right_slot, edge_type)

    return {
        'phrase_token_mask': phrase_token_mask,
        'phrase_type': phrase_type,
        'phrase_valid': phrase_valid,
        'phrase_required': phrase_required,
        'query_edge_type': query_edge_type,
    }


class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']

        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(args['vocab_size']):
            self.keep_vocab[w] = self.vocab_size

        phrase_config = args.get('phrase_graph', {})
        self.phrase_graph_enabled = bool(phrase_config.get('enabled', False))
        self.max_phrases = int(args.get('max_phrases', 8))
        self.max_actions = int(phrase_config.get('max_actions', 2))
        self.max_entities = int(phrase_config.get('max_entities', 3))
        self.max_relations = int(phrase_config.get('max_relations', 1))
        # Stable per-split integer id per unique video, used for same-video
        # negative masking in the contrastive losses (proposal 7.8).
        self.video_group_ids = {}
        for row in self.data:
            vid = row[0]
            if vid not in self.video_group_ids:
                self.video_group_ids[vid] = len(self.video_group_ids)

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

    def tokenize_query(self, sentence):
        """Tokenize/POS-tag/filter a query exactly as ``__getitem__`` does.

        Shared with the diagnostic tools so phrase-graph inspection cannot
        drift from the training-time token stream.  Returns ``(words, tags,
        weights, words_id, fallback)``; all lists are truncated to
        ``max_num_words`` and OOV tokens are dropped.
        """
        weights = []
        words = []
        tags = []
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
                tags.append(tag)
        words_id = [self.keep_vocab[w] for w in words]
        fallback = False
        if len(words) < 1:
            # Degenerate query: every token was out-of-vocabulary.  Insert an
            # internal <unk> placeholder (words_id 0, zero vector, weight 1)
            # so downstream shapes stay well-defined; the phrase graph falls
            # back to a single required GLOBAL phrase.
            words = ['<unk>']
            tags = ['UNK']
            words_id = [0]
            weights = [1]
            fallback = True
        limit = self.max_num_words
        words = words[:limit]
        tags = tags[:limit]
        weights = weights[:limit]
        words_id = words_id[:limit]
        return words, tags, weights, words_id, fallback

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)

        words, tags, weights, words_id, fallback = self.tokenize_query(sentence)
        if fallback:
            words_feat = [np.zeros_like(self.vocab['id2vec'][0])]
        else:
            words_feat = [self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)]
            words_feat.extend(
                [self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32)
                 for w in words])
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))

        sample = {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            'raw': [vid, duration, timestamps, sentence],
            'sample_uid': int(index),
            'video_group_id': int(self.video_group_ids.get(vid, 0)),
            'query_fallback': int(fallback),
        }
        if self.phrase_graph_enabled:
            sample.update(build_query_phrase_graph(
                words, tags, self.max_phrases, self.max_num_words,
                max_actions=self.max_actions,
                max_entities=self.max_entities,
                max_relations=self.max_relations))
        return sample


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim,
                       max_phrases=8, phrase_graph_enabled=False):
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
        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)

        net_input = {
            'frames_feat': torch.from_numpy(frames_feat),
            'frames_len': torch.from_numpy(np.asarray(frames_len)),
            'words_feat': torch.from_numpy(words_feat),
            'words_id': torch.from_numpy(words_id),
            'weights': torch.from_numpy(weights),
            'words_len': torch.from_numpy(np.asarray(words_len)),
            'sample_uid': torch.from_numpy(np.asarray(
                [sample['sample_uid'] for sample in samples], dtype=np.int64)),
            'video_group_id': torch.from_numpy(np.asarray(
                [sample['video_group_id'] for sample in samples],
                dtype=np.int64)),
            'query_fallback': torch.from_numpy(np.asarray(
                [sample['query_fallback'] for sample in samples],
                dtype=np.int64)),
        }

        if phrase_graph_enabled:
            num_words = words_id.shape[1]
            phrase_token_mask = np.zeros(
                [bsz, max_phrases, num_words], dtype=bool)
            phrase_type = np.zeros([bsz, max_phrases], dtype=np.int64)
            phrase_valid = np.zeros([bsz, max_phrases], dtype=bool)
            phrase_required = np.zeros([bsz, max_phrases], dtype=bool)
            query_edge_type = np.zeros(
                [bsz, max_phrases, max_phrases], dtype=np.int64)
            for i, sample in enumerate(samples):
                mask = sample['phrase_token_mask']
                phrase_token_mask[i, :, :mask.shape[1]] = mask
                phrase_type[i] = sample['phrase_type']
                phrase_valid[i] = sample['phrase_valid']
                phrase_required[i] = sample['phrase_required']
                query_edge_type[i] = sample['query_edge_type']
            net_input.update({
                'phrase_token_mask': torch.from_numpy(phrase_token_mask),
                'phrase_type': torch.from_numpy(phrase_type),
                'phrase_valid': torch.from_numpy(phrase_valid),
                'phrase_required': torch.from_numpy(phrase_required),
                'query_edge_type': torch.from_numpy(query_edge_type),
            })

        batch.update({'net_input': net_input})
        return batch

    return collate_data
