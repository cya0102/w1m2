from unittest.mock import patch

import torch

from models.cpl import CPL


def _config(proposal_type):
    return {
        'frames_input_size': 4, 'words_input_size': 3, 'hidden_size': 8,
        'vocab_size': 11, 'use_negative': True, 'num_props': 3,
        'sigma': 9, 'gamma': 0, 'dropout': 0., 'max_epoch': 10,
        'proposal_generator': {
            'type': proposal_type, 'max_components': 3,
            'component_sigma': 4., 'importance_temperature': 1.,
            'boundary_mode': 'outer', 'boundary_shrink': 0.,
        },
        'event_disentanglement': {'enabled': False},
        'DualTransformer': {
            'd_model': 8, 'num_heads': 2, 'num_decoder_layers1': 1,
            'num_decoder_layers2': 1, 'dropout': 0.,
        },
        'qstg': {
            'enabled': True, 'max_phrases': 4, 'node_stride': 4,
            'num_graph_layers': 1, 'num_graph_heads': 2,
            'similar_topk': 1, 'similarity_threshold': .1,
            'proposal_prior_scale': 0., 'importance_bias_scale': 0.,
            'component_residual_enabled': True, 'quality_head_enabled': True,
            'allow_global_fallback': False, 'graph_dropout': 0.,
        },
    }


def _batch():
    batch, frames, words, phrases = 2, 20, 20, 4
    token_mask = torch.zeros(batch, phrases, words, dtype=torch.bool)
    token_mask[:, 0] = True
    token_mask[:, 1, :2] = True
    token_mask[:, 2, 2:4] = True
    token_mask[:, 3, 4] = True
    return dict(
        frames_feat=torch.randn(batch, frames, 4),
        frames_len=torch.tensor([20, 18]),
        words_id=torch.randint(0, 11, (batch, words)),
        words_feat=torch.randn(batch, words + 1, 3),
        words_len=torch.tensor([5, 6]),
        weights=torch.cat([torch.full((1, words), .2),
                           torch.cat([torch.full((1, 6), 1 / 6), torch.zeros(1, 14)], 1)]),
        epoch=1, sample_uid=torch.tensor([0, 1]),
        video_group_id=torch.tensor([0, 1]), phrase_token_mask=token_mask,
        phrase_type=torch.tensor([[1, 2, 3, 5]] * batch),
        phrase_valid=torch.ones(batch, phrases, dtype=torch.bool),
        phrase_required=torch.tensor([[0, 1, 1, 1]] * batch, dtype=torch.bool),
        query_edge_type=torch.zeros(batch, phrases, phrases, dtype=torch.long),
    )


def test_single_and_mixture_qstg_forward_backward():
    with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
        for proposal_type in ('single_gaussian', 'gaussian_mixture'):
            torch.manual_seed(21)
            model = CPL(_config(proposal_type))
            model.train()
            output = model(**_batch())
            assert output['proposal_reconstruction_nll'].shape == (2, 3)
            assert output['proposal_coverage'].shape == (2, 3)
            assert torch.isfinite(output['words_logit']).all()
            objective = output['words_logit'].mean() + output['proposal_analytic_score'].mean()
            objective.backward()
            assert model.qstg.pre_query_projection.weight.grad is not None
