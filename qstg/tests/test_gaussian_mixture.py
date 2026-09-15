import torch

from models.loss import mixture_pull_push_loss
from models.modules.gaussian_mixture import GaussianMixtureProposalGenerator


def test_variable_component_mixture_shapes_and_gradients():
    torch.manual_seed(17)
    batch_size, hidden_size, num_proposals = 2, 8, 5
    generator = GaussianMixtureProposalGenerator(
        hidden_size=hidden_size,
        num_proposals=num_proposals,
        max_components=5,
        sigma=4.0,
        boundary_mode="outer",
        boundary_shrink=0.0,
    )
    global_feature = torch.randn(
        batch_size, hidden_size, requires_grad=True)
    centers, widths, component_weights = generator.predict_components(
        global_feature, sequence_length=24)
    reconstruction_feature = torch.randn(
        batch_size, generator.total_components, hidden_size,
        requires_grad=True)
    output = generator.combine(
        centers, widths, component_weights, reconstruction_feature)

    assert generator.component_counts == [1, 2, 3, 4, 5]
    assert centers.shape == (batch_size, 15)
    assert output["mixture_weights"].shape == (
        batch_size, num_proposals, 24)
    assert output["component_weights"].shape == (
        batch_size, num_proposals, 5, 24)
    assert output["component_valid_mask"][0].sum(dim=-1).tolist() == [
        1, 2, 3, 4, 5]
    assert torch.allclose(
        output["mixture_weights"].amax(dim=-1),
        torch.ones(batch_size, num_proposals), atol=1e-6)

    valid = output["component_valid_mask"]
    importance_sum = (
        output["component_importance"] * valid).sum(dim=-1)
    assert torch.allclose(
        importance_sum, torch.ones_like(importance_sum), atol=1e-6)
    assert torch.all(output["proposal_centers"] >= 0)
    assert torch.all(output["proposal_centers"] <= 1)
    assert torch.all(output["proposal_widths"] > 0)
    assert torch.all(output["proposal_widths"] <= 1)
    assert torch.allclose(
        output["proposal_centers"], output["context_centers"])
    assert torch.allclose(
        output["proposal_widths"], output["context_widths"])

    # Every component in one proposal shares the same width, following PPS.
    offset = 0
    for count in generator.component_counts:
        group_width = widths[:, offset:offset + count]
        assert torch.allclose(
            group_width, group_width[:, :1].expand_as(group_width))
        offset += count

    objective = (
        output["mixture_weights"].sum()
        + output["proposal_centers"].sum()
        + output["proposal_widths"].sum())
    objective.backward()
    assert global_feature.grad is not None
    assert reconstruction_feature.grad is not None
    assert torch.isfinite(global_feature.grad).all()
    assert torch.isfinite(reconstruction_feature.grad).all()


def test_outer_boundary_expands_with_component_spread():
    generator = GaussianMixtureProposalGenerator(
        hidden_size=4,
        num_proposals=2,
        max_components=2,
        boundary_mode="outer",
        boundary_shrink=0.0,
    )
    widths = torch.full((1, 3), 0.2)
    component_weights = torch.ones(1, 3, 8)
    reconstruction_features = torch.zeros(1, 3, 4)

    near = torch.tensor([[0.5, 0.45, 0.55]])
    far = torch.tensor([[0.5, 0.10, 0.90]])
    near_output = generator.combine(
        near, widths, component_weights, reconstruction_features)
    far_output = generator.combine(
        far, widths, component_weights, reconstruction_features)

    assert far_output["proposal_widths"][0, 1] > near_output[
        "proposal_widths"][0, 1]
    assert torch.allclose(
        far_output["proposal_widths"], far_output["context_widths"])
    assert torch.allclose(
        far_output["proposal_centers"], far_output["context_centers"])


def test_weighted_localization_is_decoupled_from_outer_context():
    generator = GaussianMixtureProposalGenerator(
        hidden_size=4,
        num_proposals=2,
        max_components=2,
        boundary_mode="weighted",
        boundary_shrink=0.0,
    )
    # Zero features produce equal component importance. The two-component
    # proposal has a narrow importance-weighted localization boundary while
    # its context exclusion boundary still spans both components.
    centers = torch.tensor([[0.5, 0.10, 0.90]])
    widths = torch.full((1, 3), 0.2)
    component_weights = torch.ones(1, 3, 8)
    reconstruction_features = torch.zeros(1, 3, 4)
    output = generator.combine(
        centers, widths, component_weights, reconstruction_features)

    assert output["proposal_widths"][0, 1] < output[
        "context_widths"][0, 1]
    assert torch.allclose(
        output["proposal_widths"][0, 1], torch.tensor(0.2), atol=1e-6)
    assert torch.allclose(
        output["context_widths"][0, 1], torch.tensor(1.0), atol=1e-6)


def call_pull_push(centers, component_weights):
    batch_size, num_proposals, max_components = centers.shape
    valid = torch.ones(
        batch_size, num_proposals, max_components, dtype=torch.bool)
    importance = torch.full_like(
        centers, 1.0 / max_components)
    mixture = component_weights.mean(dim=2)
    words_logit = torch.zeros(1, 1, 2, requires_grad=True)
    return mixture_pull_push_loss(
        words_logit=words_logit,
        num_props=num_proposals,
        mixture_component_centers=centers,
        mixture_component_weights=component_weights,
        mixture_component_importance=importance,
        mixture_component_valid_mask=valid,
        gauss_weight=mixture.view(batch_size * num_proposals, -1),
        mixture_pull_weight=0.05,
        mixture_intra_push_weight=0.05,
        mixture_inter_push_weight=0.1,
        mixture_intra_push_target=0.15,
    )


def test_pull_penalizes_scattered_components():
    weights = torch.ones(1, 1, 2, 10)
    near_centers = torch.tensor([[[0.45, 0.55]]])
    far_centers = torch.tensor([[[0.10, 0.90]]])
    _, near_metrics = call_pull_push(near_centers, weights)
    _, far_metrics = call_pull_push(far_centers, weights)
    assert far_metrics["mixture_pull_loss"] > near_metrics[
        "mixture_pull_loss"]


def test_intra_push_penalizes_collapsed_component_masks():
    centers = torch.tensor([[[0.45, 0.55]]])
    identical = torch.ones(1, 1, 2, 10)
    separated = torch.zeros(1, 1, 2, 10)
    separated[:, :, 0, :5] = 1
    separated[:, :, 1, 5:] = 1
    _, identical_metrics = call_pull_push(centers, identical)
    _, separated_metrics = call_pull_push(centers, separated)
    assert identical_metrics["mixture_intra_push_loss"] > separated_metrics[
        "mixture_intra_push_loss"]


def test_single_gaussian_path_has_zero_mixture_loss():
    logits = torch.randn(2, 3, 7, requires_grad=True)
    loss, metrics = mixture_pull_push_loss(
        words_logit=logits, num_props=2)
    assert loss.item() == 0
    assert metrics["mixture_loss"] == 0
