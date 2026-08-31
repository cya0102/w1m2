from unittest.mock import patch

import torch

from models.cpl import CPL
from models.loss import (
    event_disentanglement_loss,
    ivc_loss,
    mixture_pull_push_loss,
    rec_loss,
)


def make_small_config():
    return {
        "frames_input_size": 4,
        "words_input_size": 3,
        "hidden_size": 8,
        "vocab_size": 11,
        "use_negative": True,
        "num_props": 3,
        "sigma": 9,
        "gamma": 0,
        "dropout": 0.0,
        "max_epoch": 30,
        "proposal_generator": {
            "type": "gaussian_mixture",
            "max_components": 3,
            "component_sigma": 4.0,
            "importance_temperature": 1.0,
            "boundary_mode": "weighted",
            "boundary_shrink": 0.0,
        },
        "event_disentanglement": {
            "enabled": True,
            "rank": 3,
            "cpca_alpha": 1.0,
            "covariance_ema": 0.5,
            "normalize_covariance": True,
            "selection_temperature": 0.1,
            "warmup_epochs": 0,
            "ramp_epochs": 0,
            "score_separation_weight": 0.5,
        },
        "DualTransformer": {
            "d_model": 8,
            "num_heads": 2,
            "num_decoder_layers1": 1,
            "num_decoder_layers2": 1,
            "dropout": 0.0,
        },
    }


def test_v4_forward_and_all_losses_are_differentiable_on_cpu():
    torch.manual_seed(23)
    model = CPL(make_small_config())
    model.train()
    captured_negative_geometry = {}
    original_negative_mining = model.negative_proposal_mining

    def capture_negative_mining(props_len, center, width, epoch):
        captured_negative_geometry["center"] = center.detach().clone()
        captured_negative_geometry["width"] = width.detach().clone()
        return original_negative_mining(
            props_len, center, width, epoch)

    model.negative_proposal_mining = capture_negative_mining
    batch_size = 2
    frames = torch.randn(batch_size, 20, 4)
    frame_lengths = torch.tensor([20, 18])
    word_ids = torch.randint(0, 11, (batch_size, 20))
    word_features = torch.randn(batch_size, 21, 3)
    word_lengths = torch.tensor([5, 6])
    word_weights = torch.zeros(batch_size, 20)
    word_weights[0, :5] = 1.0 / 5
    word_weights[1, :6] = 1.0 / 6

    # The original CPL code constructs masks with Tensor.cuda().  Patching it
    # to identity lets this integration test exercise the complete graph on a
    # CPU-only test host without changing the training implementation.
    with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
        output = model(
            frames_feat=frames,
            frames_len=frame_lengths,
            words_id=word_ids,
            words_feat=word_features,
            words_len=word_lengths,
            weights=word_weights,
            epoch=6,
        )

        assert output["words_logit"].shape[:2] == (
            batch_size * model.num_props, 20)
        assert output["gauss_weight"].shape == (
            batch_size * model.num_props, 5)
        assert output["mixture_component_centers"].shape == (
            batch_size, model.num_props, 3)
        assert torch.all(
            output["mixture_context_width"] + 1e-6 >= output["width"])
        assert torch.any(
            output["mixture_context_width"] > output["width"] + 1e-5)
        assert torch.allclose(
            captured_negative_geometry["center"],
            output["mixture_context_center"])
        assert torch.allclose(
            captured_negative_geometry["width"],
            output["mixture_context_width"])
        assert output["event_vector"] is not None

        common_loss = {
            "margin_1": 0.1,
            "margin_2": 0.15,
            "lambda": 0.125,
            "alpha_1": 2.0,
            "alpha_2": 0.0,
            "event_alpha": 0.1,
            "event_margin": 0.2,
            "event_sep_weight": 1.0,
            "event_text_weight": 1.0,
            "event_min_context": 0.15,
            "event_context_weight": 2.0,
            "event_max_overlap": 0.7,
            "event_overlap_weight": 1.0,
            "mixture_pull_weight": 0.05,
            "mixture_intra_push_weight": 0.05,
            "mixture_inter_push_weight": 0.1,
            "mixture_intra_push_target": 0.15,
            "mixture_inter_push_target": 0.15,
        }
        reconstruction, _ = rec_loss(
            **output, num_props=model.num_props, **common_loss)
        ranking, _ = ivc_loss(
            **output, num_props=model.num_props, **common_loss)
        event, _ = event_disentanglement_loss(
            **output, num_props=model.num_props, **common_loss)
        mixture, mixture_metrics = mixture_pull_push_loss(
            **output, num_props=model.num_props, **common_loss)
        total = reconstruction + ranking + event + mixture
        total.backward()

    assert mixture_metrics["mixture_loss"] > 0
    assert model.mixture_generator.center_head.weight.grad is not None
    assert model.mixture_generator.importance_score.weight.grad is not None
    assert model.mixture_importance_token.grad is not None
    assert torch.isfinite(
        model.mixture_generator.center_head.weight.grad).all()
