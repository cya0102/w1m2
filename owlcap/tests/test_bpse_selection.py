import numpy as np

from runners.main_runner import select_proposal_by_strategy


def test_bpse_quality_order_is_descending():
    props = np.asarray([[[.0, .1], [.2, .5], [.6, .7]]])
    # Already sorted by selection cost = -quality.
    cost = np.asarray([[-2.0, -1.0, 0.0]])
    selected = select_proposal_by_strategy(props, cost, strategy='bpse')
    assert selected.tolist() == [0]


def test_old_selectors_remain_available():
    props = np.asarray([[[.0, .1], [.2, .5], [.6, .7]]])
    scores = np.asarray([[0.0, 1.0, 2.0]])
    assert select_proposal_by_strategy(props, scores, strategy='nll').tolist() == [0]
    assert select_proposal_by_strategy(
        props, scores, strategy='semantic_vote').shape == (1,)
