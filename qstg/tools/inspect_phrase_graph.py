#!/usr/bin/env python
"""Print and summarize the deterministic POS-rule phrase graph."""

import argparse
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import ActivityNet, CharadesSTA
from models.modules.qstg_ops import EDGE_AFTER, EDGE_ASSOC, EDGE_BEFORE, EDGE_WHILE
from utils import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--config-path', dest='config', required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=8)
    args = parser.parse_args()
    config = load_json(args.config)
    dataset_args = config['dataset']
    with open(dataset_args['vocab_path'], 'rb') as handle:
        vocab = pickle.load(handle)
    dataset_class = {'ActivityNet': ActivityNet,
                     'CharadesSTA': CharadesSTA}[dataset_args['dataset']]
    dataset = dataset_class(dataset_args['train_data'], vocab, dataset_args)
    random.seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:min(args.limit, len(indices))]
    stats = {'sampled': len(indices), 'no_verb': 0, 'no_noun': 0,
             'truncated': 0, 'global_only': 0, 'fallback': 0}
    edge_names = {EDGE_ASSOC: 'assoc', EDGE_BEFORE: 'before',
                  EDGE_AFTER: 'after', EDGE_WHILE: 'while'}
    for index in indices:
        row = dataset.data[index]
        words, tags, _, _, fallback = dataset.tokenize_query(row[3])
        graph = dataset[index]
        if not any(tag.startswith('VB') for tag in tags):
            stats['no_verb'] += 1
        if not any(tag.startswith('NN') for tag in tags):
            stats['no_noun'] += 1
        if fallback:
            stats['fallback'] += 1
        if len(words) >= dataset.max_num_words:
            stats['truncated'] += 1
        valid = graph['phrase_valid']
        content = valid & (graph['phrase_type'] > 1)
        if not content.any():
            stats['global_only'] += 1
        phrases = []
        for slot in range(dataset.max_phrases):
            if not valid[slot]:
                continue
            token_indices = graph['phrase_token_mask'][slot].nonzero()[0]
            phrases.append({
                'slot': slot,
                'type': int(graph['phrase_type'][slot]),
                'required': bool(graph['phrase_required'][slot]),
                'tokens': [words[int(i)] for i in token_indices],
            })
        edges = []
        for source in range(dataset.max_phrases):
            for target in range(dataset.max_phrases):
                edge = int(graph['query_edge_type'][source, target])
                if edge and source < target:
                    edges.append((source, target, edge_names.get(edge, str(edge))))
        print('\n#{:04d} {}'.format(index, row[3]))
        print('tokens:', ' '.join(words))
        print('phrases:', phrases)
        print('edges:', edges)
    print('\nstatistics:', stats)


if __name__ == '__main__':
    main()
