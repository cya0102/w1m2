from unittest.mock import patch

import torch

from models.cpl import CPL
from models.modules.gaussian_mixture import GaussianMixtureProposalGenerator


def test_optional_generator_bias_none_matches_legacy_call():
    torch.manual_seed(31)
    generator = GaussianMixtureProposalGenerator(8, 3, max_components=3)
    feature = torch.randn(2, 8)
    old = generator.predict_components(feature, 12)
    new = generator.predict_components(feature, 12, None, None)
    for first, second in zip(old, new):
        assert torch.equal(first, second)
    summary = torch.randn(2, generator.total_components, 8)
    old_combined = generator.combine(*old, summary)
    new_combined = generator.combine(*new, summary, importance_logit_bias=None)
    for key in old_combined:
        if torch.is_tensor(old_combined[key]):
            assert torch.equal(old_combined[key], new_combined[key])


def test_zero_residual_enabled_cpl_keeps_baseline_proposals():
    base = {
        'frames_input_size': 4, 'words_input_size': 3, 'hidden_size': 8,
        'vocab_size': 11, 'use_negative': True, 'num_props': 2, 'sigma': 9,
        'gamma': 0, 'dropout': 0., 'max_epoch': 10,
        'proposal_generator': {'type': 'single_gaussian'},
        'event_disentanglement': {'enabled': False},
        'DualTransformer': {'d_model': 8, 'num_heads': 2,
                            'num_decoder_layers1': 1,
                            'num_decoder_layers2': 1, 'dropout': 0.},
    }
    enabled = dict(base)
    enabled['qstg'] = {'enabled': True, 'max_phrases': 4, 'num_graph_layers': 1,
                       'num_graph_heads': 2, 'proposal_prior_scale': 0.,
                       'importance_bias_scale': 0., 'graph_dropout': 0.,
                       'allow_global_fallback': False}
    with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
        torch.manual_seed(33)
        baseline = CPL(base)
        torch.manual_seed(34)
        qstg = CPL(enabled)
        qstg.load_state_dict(baseline.state_dict(), strict=False)
        inputs = {
            'frames_feat': torch.randn(2, 20, 4),
            'frames_len': torch.tensor([20, 18]),
            'words_id': torch.randint(0, 11, (2, 20)),
            'words_feat': torch.randn(2, 21, 3),
            'words_len': torch.tensor([5, 6]),
            'weights': torch.cat([torch.full((1, 20), .2),
                                  torch.cat([torch.full((1, 6), 1 / 6), torch.zeros(1, 14)], 1)]),
            'epoch': 0, 'sample_uid': torch.tensor([1, 2]),
            'video_group_id': torch.tensor([0, 1]),
        }
        phrase_mask = torch.zeros(2, 4, 20, dtype=torch.bool)
        phrase_mask[:, 0] = True
        phrase_mask[:, 1, :2] = True
        phrase_mask[:, 2, 2:4] = True
        phrase_mask[:, 3, 4] = True
        qinputs = dict(inputs, phrase_token_mask=phrase_mask,
                        phrase_type=torch.tensor([[1, 2, 3, 5]] * 2),
                        phrase_valid=torch.ones(2, 4, dtype=torch.bool),
                        phrase_required=torch.tensor([[0, 1, 1, 1]] * 2, dtype=torch.bool),
                        query_edge_type=torch.zeros(2, 4, 4, dtype=torch.long))
        baseline.eval(); qstg.eval()
        first = baseline(**inputs, mask_mode='deterministic', eval_seed=0)
        second = qstg(**qinputs, mask_mode='deterministic', eval_seed=0)
        for key in ('center', 'width', 'gauss_weight', 'words_logit'):
            assert torch.equal(first[key], second[key])
