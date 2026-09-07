import numpy as np

from runners.main_runner import select_proposal_by_strategy


def test_semantic_vote_protects_strong_proposal_from_weak_cluster():
    # Proposal 0 is the NLL winner. Proposals 1--4 form a mutually overlapping
    # background cluster, so legacy geometric voting selects that cluster.
    proposals = np.array([[[
        0.05, 0.20],
        [0.55, 0.75],
        [0.56, 0.76],
        [0.57, 0.77],
        [0.58, 0.78],
    ]])
    scores = np.array([[1.0, 2.0, 2.1, 2.2, 2.3]])

    geometric = select_proposal_by_strategy(
        proposals, scores, strategy="geometric_vote")
    semantic = select_proposal_by_strategy(
        proposals, scores, strategy="semantic_vote", temperature=0.1)
    nll = select_proposal_by_strategy(
        proposals, scores, strategy="nll")

    assert geometric[0] != 0
    assert semantic[0] == 0
    assert nll[0] == 0


def test_explicit_selector_overrides_legacy_vote_flag():
    assert train_strategy(True, "nll") == "nll"
    assert train_strategy(False, "geometric_vote") == "geometric_vote"


def train_strategy(vote, explicit):
    # Local import avoids importing the training entry point before numpy in
    # lightweight test environments.
    from train import resolve_selection_strategy
    return resolve_selection_strategy(vote, explicit)
