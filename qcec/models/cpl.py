import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
from models.modules import (
    GaussianMixtureProposalGenerator,
    LowRankEventDisentangler,
)
from models.modules.qcec import (
    QCECProposalAdapter,
    QueryConditionedCoherentEventClusters,
    snap_proposal_boundaries,
)
import math

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

        qcec_config = config.get('qcec', {})
        self.use_qcec = bool(qcec_config.get('enabled', False))
        self.qcec_snap_enabled = bool(qcec_config.get('snap_enabled', False))
        self.qcec_snap_radius_frames = int(
            qcec_config.get('snap_radius_frames', 2))
        self.qcec_snap_min_confidence = float(
            qcec_config.get('snap_min_confidence', 0.15))
        self.qcec_snap_min_relevance = float(
            qcec_config.get('snap_min_relevance', 0.5))
        self.qcec_snap_min_width = float(
            qcec_config.get('snap_min_width', 0.01))
        self.qcec_snap_edge_min_confidence = qcec_config.get(
            'snap_edge_min_confidence', None)
        if self.qcec_snap_edge_min_confidence is not None:
            self.qcec_snap_edge_min_confidence = float(
                self.qcec_snap_edge_min_confidence)
        self.qcec_freeze_backbone_epochs = int(
            qcec_config.get('freeze_backbone_epochs', 0))
        self.qcec_return_debug_tensors = bool(
            qcec_config.get('return_debug_tensors', False))
        self.qcec_use_slot_prior = bool(
            qcec_config.get('use_slot_prior', False))
        self.qcec_num_clusters = int(qcec_config.get('num_clusters', 32))
        self.qcec_prior_bias_scale = float(
            qcec_config.get('prior_bias_scale', 0.25))
        if self.use_qcec:
            if self.qcec_num_clusters < self.num_props:
                raise ValueError(
                    'QCEC num_clusters must be at least num_props')
            if self.qcec_num_clusters < 1:
                raise ValueError('QCEC num_clusters must be positive')
            if self.qcec_freeze_backbone_epochs < 0:
                raise ValueError('QCEC freeze_backbone_epochs must be non-negative')
            if self.qcec_prior_bias_scale < 0:
                raise ValueError('QCEC prior_bias_scale must be non-negative')
            if self.qcec_snap_radius_frames < 0:
                raise ValueError('QCEC snap_radius_frames must be non-negative')
            if self.qcec_snap_min_confidence < 0 or self.qcec_snap_min_relevance < 0:
                raise ValueError('QCEC snap thresholds must be non-negative')
            if self.qcec_snap_min_width < 0:
                raise ValueError('QCEC snap_min_width must be non-negative')
            if (self.qcec_snap_edge_min_confidence is not None
                    and self.qcec_snap_edge_min_confidence < 0):
                raise ValueError(
                    'QCEC snap_edge_min_confidence must be non-negative')

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

        if self.use_qcec:
            self.qcec = QueryConditionedCoherentEventClusters(
                hidden_size=config['hidden_size'],
                num_proposals=self.num_props,
                max_components=proposal_config.get('max_components', 5),
                edge_boost=qcec_config.get('edge_boost', 1.0),
                edge_power=qcec_config.get('edge_power', 2.0),
                attention_dim=qcec_config.get('attention_dim'),
                relevance_temperature=qcec_config.get(
                    'relevance_temperature', 0.1),
                relevance_change_margin=qcec_config.get(
                    'relevance_change_margin', 0.05),
                transition_threshold=qcec_config.get(
                    'transition_threshold', 0.2),
                transition_temperature=qcec_config.get(
                    'transition_temperature', 0.05),
                prior_bias_scale=self.qcec_prior_bias_scale,
                use_slot_prior=self.qcec_use_slot_prior,
            )
            self.qcec_adapter = QCECProposalAdapter(config['hidden_size'])
        self._qcec_parameter_ids = set()
        if self.use_qcec:
            self._qcec_parameter_ids.update(
                id(parameter) for parameter in self.qcec.parameters())
            self._qcec_parameter_ids.update(
                id(parameter) for parameter in self.qcec_adapter.parameters())
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
        self._original_requires_grad = {
            id(parameter): parameter.requires_grad
            for parameter in self.parameters()
        }
        self.qcec_only_training = False

    def forward(
            self,
            frames_feat,
            frames_len,
            words_id,
            words_feat,
            words_len,
            weights,
            query_role_mask=None,
            query_role_valid=None,
            qcec_cluster_ids=None,
            qcec_cluster_bounds=None,
            qcec_cluster_mask=None,
            **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)
        highres_frame_states = frames_feat[:, :n_frames]
        highres_frame_mask = frames_mask[:, :n_frames].bool()

        words_feat[:, 0] = self.start_vec.to(
            device=words_feat.device, dtype=words_feat.dtype)
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
        base_proposal_feature = h[:, -1]
        qcec_output = None
        center_logit_bias = None
        width_logit_bias = None
        single_logit_bias = None
        if self.use_qcec:
            qcec_inputs = {
                'query_role_mask': query_role_mask,
                'query_role_valid': query_role_valid,
                'qcec_cluster_ids': qcec_cluster_ids,
                'qcec_cluster_bounds': qcec_cluster_bounds,
                'qcec_cluster_mask': qcec_cluster_mask,
            }
            missing = [name for name, value in qcec_inputs.items()
                       if value is None]
            if missing:
                raise ValueError(
                    'QCEC is enabled but missing input(s): {}'.format(
                        ', '.join(missing)))
            if qcec_cluster_ids.shape != (bsz, n_frames):
                raise ValueError(
                    'QCEC cluster_ids must have shape [B, input_frames]')
            if (qcec_cluster_bounds.dim() != 3
                    or qcec_cluster_bounds.shape[0] != bsz
                    or qcec_cluster_bounds.shape[1] != self.qcec_num_clusters
                    or qcec_cluster_bounds.shape[2] != 2):
                raise ValueError(
                    'QCEC cluster_bounds do not match configured cluster count')
            if qcec_cluster_mask.shape != (bsz, self.qcec_num_clusters):
                raise ValueError(
                    'QCEC cluster_mask does not match configured cluster count')
            qcec_output = self.qcec(
                frame_states=highres_frame_states,
                frame_mask=highres_frame_mask,
                query_states=enc_out[:, 1:],
                query_mask=words_mask[:, 1:].bool(),
                query_role_mask=query_role_mask.bool(),
                query_role_valid=query_role_valid.bool(),
                cluster_ids=qcec_cluster_ids.long(),
                cluster_bounds=qcec_cluster_bounds,
                cluster_mask=qcec_cluster_mask.bool(),
            )
            proposal_generator_feature = self.qcec_adapter(
                base_proposal_feature, qcec_output['global_hint'])
            if self.qcec_use_slot_prior:
                center_logit_bias = qcec_output.get('center_logit_bias')
                width_logit_bias = qcec_output.get('width_logit_bias')
                single_logit_bias = qcec_output.get('single_logit_bias')
        else:
            proposal_generator_feature = base_proposal_feature
        if not self.use_gaussian_mixture:
            gauss_logits = self.fc_gauss(proposal_generator_feature).view(
                bsz, self.num_props, 2)
            if single_logit_bias is not None:
                if single_logit_bias.shape != gauss_logits.shape:
                    raise ValueError(
                        'QCEC single proposal bias has an invalid shape')
                gauss_logits = gauss_logits + single_logit_bias.to(
                    device=gauss_logits.device, dtype=gauss_logits.dtype)
            gauss_param = torch.sigmoid(gauss_logits).view(
                bsz * self.num_props, 2)
            gauss_center = gauss_param[:, 0]
            gauss_width = gauss_param[:, 1]

        # downsample for effeciency
        props_len = max(n_frames//4, 1)
        keep_idx = torch.linspace(
            0, n_frames-1, steps=props_len,
            device=frames_feat.device).long()
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
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

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
                    proposal_generator_feature,
                    props_len,
                    center_logit_bias=center_logit_bias,
                    width_logit_bias=width_logit_bias,
                ))
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

        qcec_eval_center = None
        qcec_eval_width = None
        qcec_snap_start_delta = None
        qcec_snap_end_delta = None
        if (self.use_qcec and not self.training and self.qcec_snap_enabled):
            with torch.no_grad():
                qcec_eval_center, qcec_eval_width, snap_diagnostics = (
                    snap_proposal_boundaries(
                        gauss_center.view(bsz, self.num_props),
                        gauss_width.view(bsz, self.num_props),
                        qcec_cluster_bounds,
                        qcec_output['cluster_relevance'],
                        qcec_output['left_boundary_hint'],
                        qcec_output['right_boundary_hint'],
                        qcec_cluster_mask.bool(),
                        frames_len,
                        radius_frames=self.qcec_snap_radius_frames,
                        min_confidence=self.qcec_snap_min_confidence,
                        min_relevance=self.qcec_snap_min_relevance,
                        min_width=self.qcec_snap_min_width,
                        edge_min_confidence=(
                            self.qcec_snap_edge_min_confidence),
                    ))
            qcec_snap_start_delta = snap_diagnostics['start_delta']
            qcec_snap_end_delta = snap_diagnostics['end_delta']
            qcec_eval_center = qcec_eval_center.reshape(-1)
            qcec_eval_width = qcec_eval_width.reshape(-1)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)

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
                    update_subspace=(
                        self.training and event_schedule > 0
                        and not self.qcec_only_training))

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
            'qcec_transition_positions': (
                qcec_output['transition_positions']
                if qcec_output is not None else None),
            'qcec_transition_mask': (
                qcec_output['transition_mask']
                if qcec_output is not None else None),
            'qcec_transition_score': (
                qcec_output['transition_score']
                if qcec_output is not None else None),
            'qcec_barrier_lr': (
                qcec_output['barrier_lr']
                if qcec_output is not None else None),
            'qcec_barrier_rl': (
                qcec_output['barrier_rl']
                if qcec_output is not None else None),
            'qcec_cluster_relevance': (
                qcec_output['cluster_relevance']
                if qcec_output is not None else None),
            'qcec_left_boundary_hint': (
                qcec_output['left_boundary_hint']
                if qcec_output is not None else None),
            'qcec_right_boundary_hint': (
                qcec_output['right_boundary_hint']
                if qcec_output is not None else None),
            'qcec_cluster_bounds': (
                qcec_cluster_bounds if qcec_output is not None else None),
            'qcec_cluster_mask': (
                qcec_cluster_mask if qcec_output is not None else None),
            'qcec_eval_center': qcec_eval_center,
            'qcec_eval_width': qcec_eval_width,
            'qcec_snap_start_delta': qcec_snap_start_delta,
            'qcec_snap_end_delta': qcec_snap_end_delta,
        }
        if self.use_qcec and self.qcec_return_debug_tensors:
            output.update({
                'qcec_cluster_tokens': qcec_output['cluster_tokens'],
                'qcec_enhanced_clusters': qcec_output['enhanced_clusters'],
                'qcec_cluster_attention': qcec_output['cluster_attention'],
                'qcec_frame_hints': qcec_output['frame_hints'],
                'qcec_global_hint': qcec_output['global_hint'],
                'qcec_slot_features': qcec_output['slot_features'],
            })
        return output

    def set_qcec_training_stage(self, qcec_only):
        """Freeze or restore the baseline parameters for staged QCEC tuning."""
        qcec_only = bool(qcec_only) and self.use_qcec
        self.qcec_only_training = qcec_only
        for parameter in self.parameters():
            parameter.requires_grad = (
                id(parameter) in self._qcec_parameter_ids
                if qcec_only
                else self._original_requires_grad.get(
                    id(parameter), parameter.requires_grad))

    def proposal_reconstruction_nll(self, words_logit, words_id, words_mask):
        bsz = words_id.size(0)
        expanded_mask = words_mask.unsqueeze(1).expand(
            bsz, self.num_props, -1).contiguous().view(
                bsz * self.num_props, -1)
        expanded_id = words_id.unsqueeze(1).expand(
            bsz, self.num_props, -1).contiguous().view(
                bsz * self.num_props, -1)
        log_probs = words_logit.log_softmax(dim=-1)
        nll = -log_probs.gather(
            dim=-1, index=expanded_id.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.sum(dim=-1)
        nll = 0.9 * nll + 0.1 / words_logit.size(-1) * smooth
        nll = nll.masked_fill(expanded_mask == 0, 0)
        nll = nll.sum(dim=-1) / expanded_mask.sum(
            dim=-1).clamp_min(1)
        return nll.view(bsz, self.num_props)

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

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.to(
            device=words_feat.device, dtype=words_feat.dtype).unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros(
                [words_feat.size(1)], dtype=torch.uint8,
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
