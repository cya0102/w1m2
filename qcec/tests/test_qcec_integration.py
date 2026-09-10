import torch

from models.cpl import CPL
from models.loss import (
    event_disentanglement_loss,
    ivc_loss,
    mixture_pull_push_loss,
    qcec_coherence_loss,
    rec_loss,
)


def small_config(proposal='gaussian_mixture', enabled=True, slots=False):
    return {
        'frames_input_size': 4, 'words_input_size': 3, 'hidden_size': 8,
        'vocab_size': 11, 'use_negative': True, 'num_props': 3,
        'sigma': 9, 'gamma': 0, 'dropout': 0.0, 'max_epoch': 30,
        'proposal_generator': {
            'type': proposal, 'max_components': 3,
            'component_sigma': 4.0, 'importance_temperature': 1.0,
            'boundary_mode': 'weighted', 'boundary_shrink': 0.0,
        },
        'event_disentanglement': {
            'enabled': True, 'rank': 3, 'cpca_alpha': 1.0,
            'covariance_ema': 0.5, 'normalize_covariance': True,
            'selection_temperature': 0.1, 'warmup_epochs': 0,
            'ramp_epochs': 0, 'score_separation_weight': 0.5,
        },
        'qcec': {
            'enabled': enabled, 'num_clusters': 4, 'attention_dim': 8,
            'use_slot_prior': slots, 'prior_bias_scale': 0.25,
            'snap_enabled': False,
        },
        'DualTransformer': {
            'd_model': 8, 'num_heads': 2,
            'num_decoder_layers1': 1, 'num_decoder_layers2': 1,
            'dropout': 0.0,
        },
    }


def batch_inputs(batch_size=2, frames=20, words=20):
    cluster_ids = (torch.arange(frames).view(1, frames) * 4 // frames)
    return {
        'frames_feat': torch.randn(batch_size, frames, 4),
        'frames_len': torch.tensor([frames, frames - 2]),
        'words_id': torch.randint(0, 11, (batch_size, words)),
        'words_feat': torch.randn(batch_size, words + 1, 3),
        'words_len': torch.tensor([5, 6]),
        'weights': torch.cat([
            torch.full((1, words), 0.2),
            torch.cat([torch.full((1, 6), 1 / 6), torch.zeros(1, words - 6)], 1),
        ]),
        'query_role_mask': torch.ones(batch_size, 3, words, dtype=torch.bool),
        'query_role_valid': torch.ones(batch_size, 3, dtype=torch.bool),
        'qcec_cluster_ids': cluster_ids.expand(batch_size, -1).long(),
        'qcec_cluster_bounds': torch.tensor([
            [[0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
        ]).expand(batch_size, -1, -1).clone(),
        'qcec_cluster_mask': torch.ones(batch_size, 4, dtype=torch.bool),
        'epoch': 6,
    }


def test_qcec_forward_all_losses_backward():
    model = CPL(small_config())
    model.train()
    output = model(**batch_inputs())
    common = {
        'margin_1': 0.1, 'margin_2': 0.15, 'lambda': 0.125,
        'alpha_1': 2.0, 'alpha_2': 0.0, 'event_alpha': 0.1,
        'event_margin': 0.2, 'event_sep_weight': 1.0,
        'event_text_weight': 1.0, 'event_min_context': 0.15,
        'event_context_weight': 2.0, 'event_max_overlap': 0.7,
        'event_overlap_weight': 1.0, 'mixture_pull_weight': 0.05,
        'mixture_intra_push_weight': 0.05, 'mixture_inter_push_weight': 0.1,
        'mixture_intra_push_target': 0.15, 'mixture_inter_push_target': 0.15,
        'qcec_cross_weight': 0.1, 'qcec_cross_temperature': 0.02,
        'qcec_detach_barrier': True,
    }
    losses = [
        rec_loss(**output, num_props=model.num_props, **common)[0],
        ivc_loss(**output, num_props=model.num_props, **common)[0],
        event_disentanglement_loss(
            **output, num_props=model.num_props, **common)[0],
        mixture_pull_push_loss(
            **output, num_props=model.num_props, **common)[0],
        qcec_coherence_loss(
            **output, num_props=model.num_props, **common)[0],
    ]
    total = sum(losses)
    assert torch.isfinite(total)
    total.backward()
    assert model.qcec_adapter.output_projection.weight.grad is not None
    assert torch.isfinite(
        model.qcec_adapter.output_projection.weight.grad).all()


def test_disabled_model_has_no_qcec_parameters_and_accepts_old_inputs():
    model = CPL(small_config(enabled=False))
    assert not model.use_qcec
    assert not any(name.startswith('qcec') for name, _ in model.named_parameters())
    output = model(
        **{key: value for key, value in batch_inputs().items()
           if not key.startswith('qcec_') and not key.startswith('query_role_')})
    assert output['qcec_transition_positions'] is None
    assert output['qcec_eval_center'] is None

