from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGuidedIterativeQueryFormer(nn.Module):
    """Graph-guided Iterative Query Former.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 5,
        memory_size: int = 8,
        num_iterations: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.memory_size = memory_size
        self.num_iterations = num_iterations

        self.language_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.context_memory = nn.Parameter(torch.randn(memory_size, hidden_dim) * 0.02)
        self.visual_graph_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.textual_query_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.visual_language_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.visual_graph_norm = nn.LayerNorm(hidden_dim)
        self.context_query_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.memory_update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        graph_features: torch.Tensor,
        text_queries: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if visual_features is None:
            visual_features = graph_features

        num_frames, num_slots, hidden_dim = graph_features.shape
        if text_queries is None:
            text_queries = self._language_queries(num_slots, graph_features.device)
            text_queries = text_queries.unsqueeze(0).expand(num_frames, -1, -1)
        else:
            text_queries = self._match_slots(text_queries, num_slots)

        memory = self.context_memory.unsqueeze(0).expand(num_frames, -1, -1)
        f_vg = graph_features
        f_cq = text_queries
        f_vl = text_queries
        memory_states = []
        for _ in range(self.num_iterations):
            f_vg_delta, _ = self.visual_graph_attn(
                self.graph_norm(f_vg), visual_features, visual_features)
            f_vg = self.visual_graph_norm(f_vg + f_vg_delta)

            f_cq_delta, _ = self.textual_query_attn(
                self.query_norm(f_cq), memory, memory)
            f_cq = self.context_query_norm(f_cq + f_cq_delta)

            f_vl_delta, _ = self.visual_language_attn(f_vg, f_cq, f_cq)
            f_vl = self.output_norm(f_vg + f_vl_delta)

            pooled = f_vl.mean(dim=1)
            flat_memory = memory.reshape(-1, hidden_dim)
            update = pooled[:, None].expand(-1, self.memory_size, -1).reshape(-1, hidden_dim)
            memory = self.memory_update(update, flat_memory).reshape(
                num_frames, self.memory_size, hidden_dim)
            memory_states.append(memory)

        return {
            'visual_graph_features': f_vg,
            'context_query_features': f_cq,
            'visual_language_features': f_vl,
            'context_memory': memory_states[-1] if memory_states else memory,
        }

    def _match_slots(self, tensor: torch.Tensor, num_slots: int) -> torch.Tensor:
        if tensor.shape[1] == num_slots:
            return tensor
        if tensor.shape[1] > num_slots:
            return tensor[:, :num_slots]
        repeat = num_slots // tensor.shape[1] + 1
        return tensor.repeat(1, repeat, 1)[:, :num_slots]

    def _language_queries(self, num_slots: int, device: torch.device) -> torch.Tensor:
        if num_slots <= self.num_queries:
            return self.language_queries[:num_slots].to(device)
        repeat = num_slots // self.num_queries + 1
        return self.language_queries.repeat(repeat, 1)[:num_slots].to(device)


class FineGrainedMaskLinguisticDecoder(nn.Module):
    """Fine-grained Mask-linguistic Decoder with FA and MC losses."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 5,
        memory_size: int = 8,
        num_iterations: int = 2,
        mc_loss_weight: float = 0.1,
        temperature_init: float = 0.07,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gi_qformer = GraphGuidedIterativeQueryFormer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            memory_size=memory_size,
            num_iterations=num_iterations,
            dropout=dropout,
        )
        self.embedding_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.referring_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_queries),
        )
        self.mc_loss_weight = mc_loss_weight
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))

    def forward(
        self,
        pred_embeddings_list_video: Sequence[torch.Tensor],
        frame_graph_features: Optional[Sequence[torch.Tensor]],
        visual_features: Optional[Sequence[torch.Tensor]] = None,
        alignment_targets: Optional[Sequence[torch.Tensor]] = None,
        seg_token_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if frame_graph_features is None or len(frame_graph_features) != len(pred_embeddings_list_video):
            return {
                'pred_embeddings_list_video': pred_embeddings_list_video,
                'loss_fa': None,
                'loss_mc': None,
                'loss_mc_raw': None,
            }

        pred_embeddings = torch.stack(list(pred_embeddings_list_video), dim=0)
        if pred_embeddings.numel() == 0:
            zero = pred_embeddings.sum() * 0.0
            return {
                'pred_embeddings_list_video': pred_embeddings_list_video,
                'loss_fa': zero,
                'loss_mc': zero,
                'loss_mc_raw': zero,
            }

        graph_features = torch.stack([
            self._match_slots(features, pred_embeddings.shape[1])
            for features in frame_graph_features
        ], dim=0)
        graph_features = graph_features.to(
            device=pred_embeddings.device,
            dtype=next(self.parameters()).dtype,
        )
        visual_features = self._prepare_visual_features(
            visual_features=visual_features,
            frame_graph_features=frame_graph_features,
            num_slots=pred_embeddings.shape[1],
            device=pred_embeddings.device,
            dtype=next(self.parameters()).dtype,
        )
        pred_for_qformer = pred_embeddings.to(next(self.parameters()).dtype)

        qformer_outputs = self.gi_qformer(
            graph_features=graph_features,
            text_queries=pred_for_qformer,
            visual_features=visual_features,
        )
        f_vg = qformer_outputs['visual_graph_features']
        f_vl = qformer_outputs['visual_language_features']
        fused = self.embedding_fusion(torch.cat([pred_for_qformer, f_vl], dim=-1))
        fused = pred_for_qformer + fused

        referring_logits = self.referring_head(f_vg)
        fa_target = self._prepare_alignment_target(
            alignment_targets,
            referring_logits.shape,
            device=referring_logits.device,
            dtype=referring_logits.dtype,
        )
        loss_fa = F.binary_cross_entropy_with_logits(referring_logits, fa_target)
        loss_mc_raw = self._multi_entity_contrastive_loss(fused, f_vl, fa_target)
        loss_mc = loss_mc_raw * self.mc_loss_weight

        return {
            'pred_embeddings_list_video': [
                item.to(pred_embeddings.dtype) for item in fused.unbind(dim=0)
            ],
            'referring_logits': referring_logits,
            'alignment_target': fa_target,
            'loss_fa': loss_fa,
            'loss_mc': loss_mc,
            'loss_mc_raw': loss_mc_raw,
            'mc_loss_weight': self.mc_loss_weight,
            **qformer_outputs,
        }

    def _match_slots(self, features: torch.Tensor, num_slots: int) -> torch.Tensor:
        if features.shape[0] == num_slots:
            return features
        if features.shape[0] > num_slots:
            return features[:num_slots]
        repeat = num_slots // features.shape[0] + 1
        return features.repeat(repeat, 1)[:num_slots]

    def _prepare_visual_features(
        self,
        visual_features: Optional[Sequence[torch.Tensor]],
        frame_graph_features: Sequence[torch.Tensor],
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if visual_features is None:
            visual_features = frame_graph_features
        return torch.stack([
            self._match_slots(features, num_slots)
            for features in visual_features
        ], dim=0).to(device=device, dtype=dtype)

    def _prepare_alignment_target(self, alignment_targets, shape, device, dtype):
        if alignment_targets is None or len(alignment_targets) != shape[0]:
            return self._build_alignment_target(shape, device, dtype)
        target = torch.stack([
            self._fit_alignment_target(torch.as_tensor(target), shape[1], shape[2])
            for target in alignment_targets
        ], dim=0)
        return target.to(device=device, dtype=dtype)

    def _fit_alignment_target(self, target: torch.Tensor, num_masks: int, caption_slots: int):
        target = target[:num_masks, :caption_slots]
        if target.shape == (num_masks, caption_slots):
            return target
        padded = target.new_zeros(num_masks, caption_slots)
        padded[:target.shape[0], :target.shape[1]] = target
        return padded

    def _build_alignment_target(self, shape, device, dtype):
        num_frames, num_masks, caption_slots = shape
        target = torch.zeros(shape, device=device, dtype=dtype)
        diag = min(num_masks, caption_slots)
        idx = torch.arange(diag, device=device)
        target[:, idx, idx] = 1.0
        return target

    def _multi_entity_contrastive_loss(self, mask_embeddings, word_embeddings, target):
        mask_embeddings = F.normalize(mask_embeddings, dim=-1)
        word_embeddings = F.normalize(word_embeddings, dim=-1)
        logits = torch.bmm(mask_embeddings, word_embeddings.transpose(1, 2))
        tau = self.log_temperature.exp().clamp_min(1e-4)
        logits = logits / tau
        log_prob_m2w = F.log_softmax(logits, dim=-1)
        log_prob_w2m = F.log_softmax(logits.transpose(1, 2), dim=-1)

        positive_m2w = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
        positive_w2m = target.transpose(1, 2)
        positive_w2m = positive_w2m / positive_w2m.sum(dim=-1, keepdim=True).clamp_min(1.0)

        loss_m2w = -(positive_m2w * log_prob_m2w).sum(dim=-1)
        loss_w2m = -(positive_w2m * log_prob_w2m).sum(dim=-1)
        valid_m2w = target.sum(dim=-1) > 0
        valid_w2m = target.sum(dim=1) > 0
        loss_m2w = loss_m2w[valid_m2w].mean() if valid_m2w.any() else logits.sum() * 0.0
        loss_w2m = loss_w2m[valid_w2m].mean() if valid_w2m.any() else logits.sum() * 0.0
        return loss_m2w + loss_w2m

    def compose_total_loss(
        self,
        caption_loss: torch.Tensor,
        mask_loss: torch.Tensor,
        loss_fa: Optional[torch.Tensor],
        loss_mc_raw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute L_caption + (L_mask + L_FA) + lambda * L_MC."""
        total = caption_loss + mask_loss
        if loss_fa is not None:
            total = total + loss_fa
        if loss_mc_raw is not None:
            total = total + self.mc_loss_weight * loss_mc_raw
        return total
