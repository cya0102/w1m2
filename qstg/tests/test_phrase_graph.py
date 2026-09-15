import numpy as np

from datasets.base import build_query_phrase_graph


def test_phrase_graph_is_fixed_and_aligned():
    words = ['man', 'quickly', 'walks', 'into', 'the', 'room']
    tags = ['NN', 'RB', 'VBZ', 'IN', 'DT', 'NN']
    result = build_query_phrase_graph(words, tags, max_phrases=8, max_words=20)
    assert result['phrase_token_mask'].shape == (8, len(words))
    assert result['phrase_token_mask'].dtype == np.bool_
    assert result['phrase_type'].dtype == np.int64
    assert result['phrase_valid'][0]
    assert result['phrase_required'].any()
    assert result['phrase_type'][1] == 2  # ACTION
    assert result['phrase_type'][2] == 3  # ENTITY
    assert 5 in result['phrase_type']  # RELATION
    valid = result['phrase_valid']
    edges = result['query_edge_type']
    assert np.all(edges[~valid, :] == 0)
    assert np.all(edges[:, ~valid] == 0)


def test_directional_relations_and_global_fallback_are_deterministic():
    first = build_query_phrase_graph(
        ['woman', 'sits', 'before', 'standing', 'up'],
        ['NN', 'VBZ', 'IN', 'VBG', 'RP'], 8, 20)
    second = build_query_phrase_graph(
        ['woman', 'sits', 'before', 'standing', 'up'],
        ['NN', 'VBZ', 'IN', 'VBG', 'RP'], 8, 20)
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert 2 in first['query_edge_type']
    fallback = build_query_phrase_graph(['<unk>'], ['UNK'], 8, 20)
    assert fallback['phrase_valid'][0]
    assert fallback['phrase_required'][0]
    assert not fallback['phrase_valid'][1:].any()


def test_phrase_capacity_and_truncation_contract():
    words = ['a', 'b', 'c', 'd']
    tags = ['NN', 'VB', 'NN', 'VB']
    result = build_query_phrase_graph(words, tags, max_phrases=3, max_words=4)
    assert result['phrase_token_mask'].shape == (3, 4)
    assert result['phrase_valid'].sum() <= 3
