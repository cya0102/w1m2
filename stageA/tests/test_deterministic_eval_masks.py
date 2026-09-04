import inspect

import torch

from models.cpl import CPL, deterministic_eval_word_mask
from tests.test_stage_a_integration import _config


def test_eval_mask_is_stable_across_batch_order_and_seed_changes():
    ids = ['video-a:3', 'video-b:9']
    lengths = torch.tensor([6, 4])
    weights = torch.ones(2, 6)
    first = deterministic_eval_word_mask(ids, lengths, 6, 18, weights)
    second = deterministic_eval_word_mask(ids, lengths, 6, 18, weights)
    reordered = deterministic_eval_word_mask(
        [ids[1], ids[0]], torch.tensor([4, 6]), 6, 18,
        weights[[1, 0]])
    assert torch.equal(first, second)
    assert torch.equal(first, reordered[[1, 0]])
    assert not torch.equal(first, deterministic_eval_word_mask(
        ids, lengths, 6, 28, weights))
    assert not first[:, 0].any()
    assert first[0].sum().item() == 2
    assert first[1].sum().item() == 1
    assert not first[0, 7:].any()


def test_mask_override_does_not_use_rng_and_training_rejects_it():
    model = CPL(_config('single_gaussian')).eval()
    # _mask_words is called after the input word projection in CPL.
    words_feat = torch.randn(2, 7, 8)
    words_len = torch.tensor([5, 4])
    override = deterministic_eval_word_mask(
        ['a:0', 'b:1'], words_len, 6, 8)
    torch.manual_seed(1)
    first, first_mask = model._mask_words(words_feat.clone(), words_len,
                                          mask_override=override)
    torch.manual_seed(999)
    second, second_mask = model._mask_words(words_feat.clone(), words_len,
                                            mask_override=override)
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first, second)
    model.train()
    try:
        model._mask_words(words_feat, words_len, mask_override=override)
    except ValueError:
        pass
    else:
        raise AssertionError('training must reject deterministic mask overrides')


def test_selector_signature_has_no_ground_truth_argument():
    from runners.stage_a5 import select_stage_a5_candidates
    assert 'gt' not in inspect.signature(
        select_stage_a5_candidates).parameters
    assert 'iou' not in inspect.signature(
        select_stage_a5_candidates).parameters
