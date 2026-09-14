import numpy as np

from datasets.base import (
    UNIT_ACTION,
    UNIT_ENTITY,
    UNIT_RELATION,
    UNIT_WHOLE,
    build_query_units,
)


def test_query_units_have_expected_types_and_normalized_weights():
    result = build_query_units(
        ['man', 'quickly', 'opens', 'the', 'red', 'door', 'in', 'room'],
        ['NN', 'RB', 'VBZ', 'DT', 'JJ', 'NN', 'IN', 'NN'],
        max_units=12)
    assert result['unit_token_mask'].shape == (12, 8)
    assert result['unit_valid_mask'].sum() >= 3
    assert UNIT_ACTION in result['unit_type_ids'][result['unit_valid_mask']]
    assert UNIT_ENTITY in result['unit_type_ids'][result['unit_valid_mask']]
    assert UNIT_RELATION in result['unit_type_ids'][result['unit_valid_mask']]
    assert UNIT_WHOLE in result['unit_type_ids'][result['unit_valid_mask']]
    assert np.isclose(result['unit_weights'].sum(), 1.0)


def test_query_units_deduplicate_and_keep_whole_query():
    result = build_query_units(['opens'], ['VBZ'], max_units=2)
    valid = result['unit_valid_mask']
    assert valid.sum() == 1
    assert result['unit_type_ids'][0] == UNIT_WHOLE


def test_query_units_respect_priority_and_max_units():
    result = build_query_units(
        ['person', 'moves', 'to', 'the', 'small', 'red', 'chair'],
        ['NN', 'VBZ', 'TO', 'DT', 'JJ', 'JJ', 'NN'],
        max_units=3)
    types = result['unit_type_ids'][result['unit_valid_mask']]
    assert len(types) == 3
    assert UNIT_WHOLE in types
    assert types[0] == UNIT_ACTION


def test_query_units_empty_query_is_padding():
    result = build_query_units([], [], max_units=4)
    assert not result['unit_valid_mask'].any()
    assert result['unit_token_mask'].sum() == 0
    assert result['unit_weights'].sum() == 0
