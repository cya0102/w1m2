import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
from models.modules import (
    EventBoundaryRefiner,
    GaussianMixtureProposalGenerator,
    LowRankEventDisentangler,
)
import math


def deterministic_eval_word_mask(sample_ids, words_len, max_words,
                                  eval_mask_seed, weights=None,
                                  device=None):
    """Build a stable weighted word-mask override for evaluation.

    The seed is derived from the sample id and the requested mask seed, so the
    result is independent of batch order, DataLoader workers, and process-wide
    NumPy RNG state.  Position zero is the learned start token and is never
    masked; positions ``1..word_len`` are the only eligible query words.
    """
    if len(sample_ids) != len(words_len):
        raise ValueError('sample_ids and words_len must have equal length')
    if int(max_words) < 0:
        raise ValueError('max_words must be non-negative')
    if weights is not None and (not torch.is_tensor(weights) or
                                weights.ndim != 2 or
                                weights.shape[0] != len(sample_ids)):
        raise ValueError('weights must have shape [batch, max_words]')
    result = torch.zeros(
        (len(sample_ids), int(max_words) + 1), dtype=torch.bool,
        device=device if device is not None else (
            weights.device if torch.is_tensor(weights) else None))
    for row, (sample_id, length_value) in enumerate(zip(sample_ids, words_len)):
        length = int(length_value)
        if length < 0 or length > max_words:
            raise ValueError('word length is outside max_words')
        if length == 0:
            continue
        count = max(length // 3, 1)
        digest = hashlib.sha256(
            '{}:{}'.format(sample_id, int(eval_mask_seed)).encode('utf8')
        ).digest()
        sample_seed = int.from_bytes(digest[:8], 'little')
        rng = np.random.default_rng(sample_seed)
        probabilities = None
        if weights is not None:
            probabilities = weights[row, :length].detach().cpu().numpy()
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if (not np.isfinite(probabilities).all() or
                    np.any(probabilities < 0) or
                    probabilities.sum() <= 0):
                probabilities = None
            else:
                probabilities = probabilities / probabilities.sum()
        choices = rng.choice(
            np.arange(1, length + 1), size=count, replace=False,
            p=probabilities)
        result[row, torch.from_numpy(np.asarray(choices, dtype=np.int64)).to(
            result.device)] = True
    return result


class CPL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']
        event_config = config.get('event_disentanglement', {})
        self.use_event_disentanglement = event_config.get('enabled', False)
        self.event_selection_temperature = event_config.get(
            'selection_temperature', 0.1)
        self.event_warmup_epochs = event_config.get('warmup_epochs', 5)
        self.event_ramp_epochs = event_config.get('ramp_epochs', 5)
        self.event_score_separation_weight = event_config.get(
            'score_separation_weight', 0.5)
        stage_a_config = config.get('event_boundary_refinement', {})
        self.stage_a_config = dict(stage_a_config)
        self.stage_a_enabled = bool(stage_a_config.get('enabled', False))
        self.stage_a5_config = dict(stage_a_config.get('stage_a5', {}))
        self.stage_a5_enabled = bool(self.stage_a5_config.get('enabled', False))
        self.stage_a5_score_shell = bool(
            self.stage_a5_config.get('score_shell', True))
        if self.stage_a5_enabled and not self.stage_a_enabled:
            raise ValueError('Stage A.5 requires Stage A to be enabled')
        self.stage_a_refiner = None
        if self.stage_a_enabled:
            candidate_policy = stage_a_config.get(
                'candidate_policy', 'near_and_strong')
            if candidate_policy != 'near_and_strong':
                raise ValueError(
                    'Stage A only supports candidate_policy=near_and_strong')
            self.stage_a_refiner = EventBoundaryRefiner(
                num_clips=stage_a_config.get('num_clips', 200),
                min_boundary_margin_clips=stage_a_config.get(
                    'min_boundary_margin_clips', 1),
                min_candidate_width=stage_a_config.get(
                    'min_candidate_width', 0.02),
                min_retained_ratio=stage_a_config.get(
                    'min_retained_ratio', 0.25),
                soft_window_temperature=stage_a_config.get(
                    'soft_window_temperature', 0.01),
            )
            self.stage_a_decode_chunk_size = int(
                stage_a_config.get('decode_chunk_size', 64))
            if self.stage_a_decode_chunk_size < 1:
                raise ValueError('decode_chunk_size must be positive')
        proposal_config = config.get('proposal_generator', {})
        self.proposal_generator_type = proposal_config.get(
            'type', 'single_gaussian')
        self.use_gaussian_mixture = (
            self.proposal_generator_type == 'gaussian_mixture')
        if self.proposal_generator_type not in {
                'single_gaussian', 'gaussian_mixture'}:
            raise ValueError(
                'unknown proposal generator: {}'.format(
                    self.proposal_generator_type))
        if self.use_event_disentanglement and not self.use_negative:
            raise ValueError('event disentanglement requires negative proposals')
        if self.event_selection_temperature <= 0:
            raise ValueError('event selection temperature must be positive')

        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        if self.use_gaussian_mixture:
            self.mixture_generator = GaussianMixtureProposalGenerator(
                hidden_size=config['hidden_size'],
                num_proposals=self.num_props,
                max_components=proposal_config.get('max_components', 5),
                sigma=proposal_config.get('component_sigma', 4.0),
                importance_temperature=proposal_config.get(
                    'importance_temperature', 1.0),
                boundary_mode=proposal_config.get(
                    'boundary_mode', 'outer'),
                boundary_shrink=proposal_config.get(
                    'boundary_shrink', 0.0),
            )
            # Appended after the masked query for component-conditioned
            # reconstruction. Under the causal query decoder this token sees
            # every preceding word and acts like PPS's reconstruction summary.
            self.mixture_importance_token = nn.Parameter(
                torch.zeros(config['hidden_size']))
        else:
            self.fc_gauss = nn.Linear(
                config['hidden_size'], self.num_props * 2)
 
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)
        if self.use_event_disentanglement:
            self.event_disentangler = LowRankEventDisentangler(
                hidden_size=config['hidden_size'],
                rank=event_config.get('rank', 8),
                cpca_alpha=event_config.get('cpca_alpha', 1.0),
                covariance_ema=event_config.get('covariance_ema', 0.95),
                normalize_covariance=event_config.get(
                    'normalize_covariance', True),
            )

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        run_stage_a5 = bool(kwargs.get('run_stage_a5', False))
        if run_stage_a5 and not self.stage_a5_enabled:
            raise RuntimeError(
                'run_stage_a5=True but event_boundary_refinement.stage_a5 is disabled')
        run_stage_a = bool(kwargs.get('run_stage_a', False)) or run_stage_a5
        if run_stage_a:
            if not self.stage_a_enabled:
                raise RuntimeError(
                    'run_stage_a=True but event_boundary_refinement is disabled')
            if self.training:
                raise RuntimeError('Stage A is inference-only and requires eval()')
            boundary_names = (
                'event_boundary_positions', 'event_boundary_scores',
                'event_boundary_mask')
            missing = [name for name in boundary_names
                       if kwargs.get(name) is None]
            if missing:
                raise ValueError(
                    'Stage A requires boundary tensors: {}'.format(
                        ', '.join(missing)))
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.to(words_feat.device)
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # generate Gaussian masks
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        query_feat = None
        if self.use_event_disentanglement:
            # Pool contextualized query tokens (excluding the learned start
            # token). This is the textual event anchor.
            query_mask = words_mask[:, 1:]
            query_feat = enc_out[:, 1:]
            query_feat = (query_feat * query_mask.unsqueeze(-1).to(query_feat.dtype)).sum(dim=1)
            query_feat = query_feat / query_mask.sum(dim=1, keepdim=True).clamp_min(1).to(query_feat.dtype)
        proposal_generator_feature = h[:, -1]
        if not self.use_gaussian_mixture:
            gauss_param = torch.sigmoid(
                self.fc_gauss(proposal_generator_feature)).view(
                    bsz * self.num_props, 2)
            gauss_center = gauss_param[:, 0]
            gauss_width = gauss_param[:, 1]

        # downsample for effeciency
        props_len = n_frames//4
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        event_props_feat = None
        if self.use_event_disentanglement:
            # Keep the Event Vector visual-only. V1 used query-conditioned
            # states and could minimize text alignment through query leakage.
            event_props_feat = props_feat
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)

        # semantic completion
        words_feat, masked_words = self._mask_words(
            words_feat, words_len, weights=weights,
            mask_override=kwargs.get('eval_word_mask'))
        masked_query_base = words_feat + words_pos
        masked_query_base = masked_query_base[:, :-1]
        query_mask_base = words_mask[:, :-1]
        words_feat = masked_query_base
        words_mask = query_mask_base

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        mixture_component_centers = None
        mixture_component_widths = None
        mixture_component_weights = None
        mixture_component_importance = None
        mixture_component_valid_mask = None
        mixture_context_center = None
        mixture_context_width = None
        if self.use_gaussian_mixture:
            flat_centers, flat_widths, flat_component_weights = (
                self.mixture_generator.predict_components(
                    proposal_generator_feature, props_len))
            total_components = self.mixture_generator.total_components

            component_props_feat = frames_feat.unsqueeze(1).expand(
                bsz, total_components, -1, -1).contiguous().view(
                    bsz * total_components, props_len, -1)
            component_props_mask = frames_mask.unsqueeze(1).expand(
                bsz, total_components, -1).contiguous().view(
                    bsz * total_components, -1)
            importance_token = self.mixture_importance_token.view(
                1, 1, -1).expand(bsz, 1, -1)
            importance_words_feat = torch.cat(
                [words_feat, importance_token], dim=1)
            importance_words_mask = torch.cat([
                words_mask,
                words_mask.new_ones(bsz, 1),
            ], dim=1)
            component_words_mask = importance_words_mask.unsqueeze(1).expand(
                bsz, total_components, -1).contiguous().view(
                    bsz * total_components, -1)
            component_words_feat = importance_words_feat.unsqueeze(1).expand(
                bsz, total_components, -1, -1).contiguous().view(
                    bsz * total_components, importance_words_mask.size(1), -1)
            component_mask = flat_component_weights.view(
                bsz * total_components, props_len)

            _, component_hidden = self.trans(
                component_props_feat, component_props_mask,
                component_words_feat, component_words_mask,
                decoding=2, gauss_weight=component_mask)
            component_summary = component_hidden[:, -1].view(
                bsz, total_components, -1)

            mixture = self.mixture_generator.combine(
                flat_centers, flat_widths, flat_component_weights,
                component_summary)
            gauss_weight = mixture['mixture_weights'].view(
                bsz * self.num_props, props_len)
            gauss_center = mixture['proposal_centers'].reshape(-1)
            gauss_width = mixture['proposal_widths'].reshape(-1)
            mixture_context_center = mixture['context_centers'].reshape(-1)
            mixture_context_width = mixture['context_widths'].reshape(-1)
            mixture_component_centers = mixture['component_centers']
            mixture_component_widths = mixture['component_widths']
            mixture_component_weights = mixture['component_weights']
            mixture_component_importance = mixture['component_importance']
            mixture_component_valid_mask = mixture['component_valid_mask']
        else:
            gauss_weight = self.generate_gauss_weight(
                props_len, gauss_center, gauss_width)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)

        stage_a_output = None
        if run_stage_a:
            original_mask = pos_weight.view(bsz, self.num_props, props_len)
            center_2d = gauss_center.view(bsz, self.num_props)
            width_2d = gauss_width.view(bsz, self.num_props)
            boundary_positions = kwargs['event_boundary_positions'].to(
                center_2d.device)
            boundary_scores = kwargs['event_boundary_scores'].to(
                center_2d.device)
            boundary_mask = kwargs['event_boundary_mask'].to(
                center_2d.device).bool()
            candidate_start, candidate_end, candidate_valid, candidate_type, \
                candidate_boundary_confidence, candidate_left_score, \
                candidate_right_score = (
                self.stage_a_refiner.build_candidates(
                    center_2d, width_2d, boundary_positions, boundary_scores,
                    boundary_mask, return_boundary_confidence=True,
                    return_boundary_scores=True))
            candidate_masks, mask_valid = (
                self.stage_a_refiner.build_candidate_masks_with_validity(
                    original_mask, candidate_start, candidate_end,
                    candidate_valid))
            candidate_valid &= mask_valid
            original_nll = self.proposal_reconstruction_nll(
                words_logit, words_id, words_mask)
            candidate_nll = self.score_stage_a_candidates(
                frames_feat=frames_feat,
                frames_mask=frames_mask,
                masked_query_base=masked_query_base,
                query_mask_base=query_mask_base,
                words_id=words_id,
                original_masks=original_mask,
                candidate_masks=candidate_masks,
                candidate_valid=candidate_valid,
                original_nll=original_nll,
                chunk_size=self.stage_a_decode_chunk_size)
            stage_a_output = {
                'stage_a_candidate_start': candidate_start,
                'stage_a_candidate_end': candidate_end,
                'stage_a_candidate_valid': candidate_valid,
                'stage_a_candidate_nll': candidate_nll,
                'stage_a_candidate_type': candidate_type,
                'stage_a_candidate_boundary_confidence':
                    candidate_boundary_confidence,
                'stage_a_candidate_left_boundary_score': candidate_left_score,
                'stage_a_candidate_right_boundary_score': candidate_right_score,
            }
            if run_stage_a5 and self.stage_a5_score_shell:
                shell_masks, shell_valid = (
                    self.stage_a_refiner.build_shell_masks_with_validity(
                        original_mask, candidate_start, candidate_end,
                        candidate_valid))
                candidate_shell_nll = self.score_stage_a_candidates(
                    frames_feat=frames_feat,
                    frames_mask=frames_mask,
                    masked_query_base=masked_query_base,
                    query_mask_base=query_mask_base,
                    words_id=words_id,
                    original_masks=original_mask,
                    candidate_masks=shell_masks,
                    candidate_valid=shell_valid,
                    original_nll=original_nll,
                    chunk_size=self.stage_a_decode_chunk_size,
                    include_original=False)
                stage_a_output.update({
                    'stage_a_candidate_shell_nll': candidate_shell_nll,
                    'stage_a_candidate_contrast': (
                        candidate_shell_nll - candidate_nll),
                })
            elif run_stage_a5:
                # Keep a shape-stable diagnostic output while ensuring that a
                # score_shell=false run does not invoke the shell decoder.
                no_shell = original_nll.new_full(
                    (bsz, self.num_props, 7), float('inf'))
                stage_a_output.update({
                    'stage_a_candidate_shell_nll': no_shell,
                    'stage_a_candidate_contrast': no_shell.clone(),
                })

        event_pos_feat = None
        event_neg_feat = None
        event_vector = None
        event_selection_weights = None
        event_score = None
        event_schedule = 0.0
        if self.use_negative:
            # Mixture masks may contain several separated components. Mine
            # background outside their complete outer envelope, rather than
            # outside an importance-averaged interval that can cut through a
            # positive component.
            negative_center = (
                mixture_context_center
                if mixture_context_center is not None else gauss_center)
            negative_width = (
                mixture_context_width
                if mixture_context_width is not None else gauss_width)
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(
                props_len, negative_center, negative_width, kwargs['epoch'])

            if self.use_event_disentanglement:
                proposal_nll = self.proposal_reconstruction_nll(
                    words_logit, words_id, words_mask)
                event_selection_weights = torch.softmax(
                    -proposal_nll / self.event_selection_temperature,
                    dim=-1).detach()
                event_schedule = self.event_loss_schedule(kwargs['epoch'])
                negative_weights = torch.stack([neg_1_weight, neg_2_weight], dim=1)
                event_pos_feat, event_neg_feat, event_vector = self.event_disentangler(
                    event_props_feat, props_mask, pos_weight, negative_weights,
                    event_selection_weights,
                    update_subspace=self.training and event_schedule > 0)

                event_direction = F.normalize(event_vector, dim=-1)
                positive_score = F.cosine_similarity(
                    event_pos_feat, event_direction, dim=-1)
                negative_score = F.cosine_similarity(
                    event_neg_feat, event_direction.unsqueeze(1), dim=-1)
                text_for_proposals = query_feat.unsqueeze(1).expand(
                    bsz, self.num_props, -1).contiguous().view(
                        bsz * self.num_props, -1)
                text_score = F.cosine_similarity(
                    event_vector, text_for_proposals, dim=-1)
                separation_score = positive_score - negative_score.max(dim=1)[0]
                event_score = text_score + (
                    self.event_score_separation_weight * separation_score)
            
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
  
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        output = {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'mixture_component_centers': mixture_component_centers,
            'mixture_component_widths': mixture_component_widths,
            'mixture_component_weights': mixture_component_weights,
            'mixture_component_importance': mixture_component_importance,
            'mixture_component_valid_mask': mixture_component_valid_mask,
            'mixture_context_center': mixture_context_center,
            'mixture_context_width': mixture_context_width,
            'event_pos_feat': event_pos_feat,
            'event_neg_feat': event_neg_feat,
            'event_vector': event_vector,
            'event_text_feat': query_feat if event_vector is not None else None,
            'event_selection_weights': event_selection_weights,
            'event_score': event_score,
            'event_schedule': event_schedule,
            'event_positive_rank': int(
                self.event_disentangler.positive_rank.item())
            if self.use_event_disentanglement else 0,
            'event_largest_eigenvalue': float(
                self.event_disentangler.largest_contrastive_eigenvalue.item())
            if self.use_event_disentanglement else 0.0,
            'event_smallest_selected_eigenvalue': float(
                self.event_disentangler.smallest_selected_eigenvalue.item())
            if self.use_event_disentanglement else 0.0,
        }
        if stage_a_output is not None:
            output.update(stage_a_output)
        return output

    @staticmethod
    def reconstruction_nll_from_logits(words_logit, words_id, words_mask):
        """Calculate the baseline label-smoothed reconstruction NLL.

        This is the same equation as ``models.loss.cal_nll_loss`` without the
        accuracy side channel.  Keeping it here lets candidate chunks use the
        exact baseline objective while avoiding any candidate-logit storage.
        """
        if words_logit.ndim != 3 or words_id.ndim != 2 or words_mask.ndim != 2:
            raise ValueError('invalid reconstruction tensor ranks')
        if words_logit.shape[:2] != words_id.shape or \
                words_id.shape != words_mask.shape:
            raise ValueError('reconstruction tensor shapes do not match')
        log_probs = words_logit.log_softmax(dim=-1)
        nll = -log_probs.gather(
            dim=-1, index=words_id.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.sum(dim=-1)
        nll = 0.9 * nll + 0.1 / words_logit.size(-1) * smooth
        mask = words_mask.to(nll.dtype)
        nll = nll * mask
        return nll.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)

    def proposal_reconstruction_nll(self, words_logit, words_id, words_mask):
        bsz = words_id.size(0)
        expanded_mask = words_mask.unsqueeze(1).expand(
            bsz, self.num_props, -1).contiguous().view(
                bsz * self.num_props, -1)
        expanded_id = words_id.unsqueeze(1).expand(
            bsz, self.num_props, -1).contiguous().view(
                bsz * self.num_props, -1)
        nll = self.reconstruction_nll_from_logits(
            words_logit, expanded_id, expanded_mask)
        return nll.view(bsz, self.num_props)

    def score_stage_a_candidates(
            self, frames_feat, frames_mask, masked_query_base,
            query_mask_base, words_id, original_masks, candidate_masks,
            candidate_valid, original_nll, chunk_size,
            include_original=True):
        """Decode valid trim candidates in bounded chunks.

        Candidate zero reuses ``original_nll``.  Only the visual/query rows
        selected by the validity mask are materialized for each chunk, and
        logits are discarded as soon as their NLL has been written.
        """
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        batch_size, num_props, num_candidates, sequence_length = \
            candidate_masks.shape
        if num_candidates != 7:
            raise ValueError('Stage A expects exactly seven candidates')
        if frames_feat.shape[:2] != (batch_size, sequence_length):
            raise ValueError('frames_feat shape does not match candidate masks')
        if frames_mask.shape != (batch_size, sequence_length):
            raise ValueError('frames_mask shape does not match candidate masks')
        if original_masks.shape != (batch_size, num_props, sequence_length):
            raise ValueError('original_masks shape does not match candidates')
        if include_original and not torch.equal(
                candidate_masks[..., 0, :], original_masks):
            raise ValueError('candidate zero mask must equal original mask')
        if candidate_valid.shape != (batch_size, num_props, 7):
            raise ValueError('candidate_valid shape does not match masks')
        if original_nll.shape != (batch_size, num_props):
            raise ValueError('original_nll shape does not match candidates')
        if masked_query_base.size(0) != batch_size or \
                query_mask_base.shape != words_id.shape or \
                query_mask_base.size(0) != batch_size:
            raise ValueError('query tensors do not match candidate batch')

        candidate_nll = original_nll.new_full(
            (batch_size, num_props, 7), float('inf'))
        if include_original:
            candidate_nll[..., 0] = original_nll
            valid_rows = candidate_valid[..., 1:].nonzero(as_tuple=False)
            candidate_offset = 1
        else:
            valid_rows = candidate_valid.nonzero(as_tuple=False)
            candidate_offset = 0
        # Columns are batch, proposal, candidate for shell masks, and
        # candidate-1 for the original/trim mask path.
        for chunk_start in range(0, valid_rows.size(0), chunk_size):
            rows = valid_rows[chunk_start:chunk_start + chunk_size]
            batch_index = rows[:, 0]
            proposal_index = rows[:, 1]
            candidate_index = rows[:, 2] + candidate_offset
            visual = frames_feat[batch_index]
            visual_mask = frames_mask[batch_index]
            query = masked_query_base[batch_index]
            query_mask = query_mask_base[batch_index]
            ids = words_id[batch_index]
            gauss_mask = candidate_masks[
                batch_index, proposal_index, candidate_index]
            _, hidden = self.trans(
                visual, visual_mask, query, query_mask,
                decoding=2, gauss_weight=gauss_mask)
            logits = self.fc_comp(hidden)
            chunk_nll = self.reconstruction_nll_from_logits(
                logits, ids, query_mask)
            candidate_nll[batch_index, proposal_index, candidate_index] = \
                chunk_nll
            del logits, hidden, chunk_nll
        return candidate_nll

    def event_loss_schedule(self, epoch):
        if not self.training:
            return 1.0
        if epoch <= self.event_warmup_epochs:
            return 0.0
        if self.event_ramp_epochs <= 0:
            return 1.0
        return min(
            (epoch - self.event_warmup_epochs) / self.event_ramp_epochs, 1.0)
    
    
    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]


    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None,
                    mask_override=None):
        token = self.mask_vec.to(words_feat.device).unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        if mask_override is not None:
            if self.training:
                raise ValueError('mask_override is only valid during evaluation')
            if not torch.is_tensor(mask_override) or mask_override.shape != (
                    words_feat.size(0), words_feat.size(1)):
                raise ValueError(
                    'mask_override must have shape [batch, max_words + 1]')
            masked_words = mask_override.to(
                device=words_feat.device, dtype=torch.bool).clone()
            for i, length in enumerate(words_len):
                length = int(length)
                if torch.any(masked_words[i, :1]) or torch.any(
                        masked_words[i, length + 1:]):
                    raise ValueError('mask_override contains an illegal position')
                expected = max(length // 3, 1) if length > 0 else 0
                if int(masked_words[i].sum()) != expected:
                    raise ValueError('mask_override has an incorrect mask count')
            masked_words = masked_words.unsqueeze(-1)
            masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
            masked_words_vec = masked_words_vec.masked_fill_(
                masked_words == 0, 0)
            words_feat1 = words_feat.masked_fill(
                masked_words == 1, 0) + masked_words_vec
            return words_feat1, masked_words

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros(
                [words_feat.size(1)], dtype=torch.bool,
                device=words_feat.device))
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            # The copied transformer uses ``1 - mask``; uint8 preserves the
            # original byte-mask semantics while remaining device-safe.
            mask.append(torch.zeros(
                [x.size(1)], dtype=torch.uint8, device=x.device))
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
