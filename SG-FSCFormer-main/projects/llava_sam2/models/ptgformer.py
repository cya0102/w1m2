from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalSceneGraph:
    """Lightweight container for prompt-guided temporal scene graph features."""

    frame_graph_features: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    adjacency_ps: torch.Tensor
    adjacency_po: torch.Tensor
    prompt_boxes: torch.Tensor


def _box_to_tensor(boxes, device, dtype):
    if boxes is None or len(boxes) == 0:
        return None
    box = boxes[0] if isinstance(boxes[0], (list, tuple)) else boxes
    if len(box) != 4:
        return None
    return torch.tensor(box, device=device, dtype=dtype)


class PromptGuidedTemporalGraphFormer(nn.Module):
    """Prompt-guided Temporal Graph Former.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_nodes: int = 9,
        num_heads: int = 8,
        temporal_window: int = 1,
        hard_keep_ratio: float = 0.75,
        sgg_feature_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.temporal_window = temporal_window
        self.hard_keep_ratio = hard_keep_ratio

        self.sgg_visual_proj = nn.Sequential(
            nn.Linear(sgg_feature_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(in_channels + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.triplet_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.triplet_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.association_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.edge_subject_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_object_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        pixel_values,
        bboxes: Sequence,
        frames_per_batch: Sequence[int],
        image_sizes: Optional[Sequence[Tuple[int, int]]] = None,
        scene_graphs: Optional[Sequence] = None,
    ) -> Dict[str, object]:
        videos = self._split_video_tensor(pixel_values, frames_per_batch)
        graphs: List[TemporalSceneGraph] = []
        frame_graph_features = []

        for i, video in enumerate(videos):
            boxes = bboxes[i] if i < len(bboxes) else None
            video_scene_graphs = scene_graphs[i] if scene_graphs is not None and i < len(scene_graphs) else None
            graph = self._forward_single_video(video, boxes, image_sizes, video_scene_graphs)
            if graph is None:
                continue
            graphs.append(graph)
            frame_graph_features.append(graph.frame_graph_features)

        if not frame_graph_features:
            return {'graphs': [], 'frame_graph_features': None}

        return {
            'graphs': graphs,
            'frame_graph_features': frame_graph_features,
        }

    def _split_video_tensor(self, pixel_values, frames_per_batch):
        if isinstance(pixel_values, list):
            return [item for item in pixel_values]
        if pixel_values.ndim == 5:
            return [pixel_values[i] for i in range(pixel_values.shape[0])]
        if pixel_values.ndim == 4:
            videos = []
            cursor = 0
            for n_frames in frames_per_batch:
                videos.append(pixel_values[cursor:cursor + n_frames])
                cursor += n_frames
            return videos
        raise NotImplementedError(f'Unsupported pixel_values shape: {pixel_values.shape}')

    def _forward_single_video(self, video, boxes, image_sizes, scene_graphs=None):
        if video.numel() == 0:
            return None
        dtype = next(self.parameters()).dtype
        video = video.to(dtype)
        device = video.device
        _, _, height, width = video.shape
        box = _box_to_tensor(boxes, device, dtype)
        if box is None:
            return None
        box = self._normalize_box(box, height, width, image_sizes)

        node_features, node_boxes = self._build_sgg_nodes(scene_graphs, box, device, dtype)
        if node_features is None:
            node_features, node_boxes = self._build_nodes(video, box)
        node_features, node_boxes = self._hard_prompt_filter(node_features, node_boxes, box)
        edge_features, adjacency_ps, adjacency_po = self._build_edges(node_features, node_boxes)
        reinforced_nodes = self._soft_filter_nodes(node_features, edge_features)
        temporal_nodes = self._temporal_update(reinforced_nodes)
        graph_features = self._aggregate_graph(
            temporal_nodes, edge_features, adjacency_ps, adjacency_po)

        return TemporalSceneGraph(
            frame_graph_features=graph_features,
            node_features=temporal_nodes,
            edge_features=edge_features,
            adjacency_ps=adjacency_ps,
            adjacency_po=adjacency_po,
            prompt_boxes=box[None].expand(video.shape[0], -1),
        )

    def _normalize_box(self, box, height, width, image_sizes):
        if box.max() <= 1.0:
            norm = box
        else:
            scale = box.new_tensor([width, height, width, height])
            norm = box / scale.clamp_min(1.0)
        x1, y1, x2, y2 = norm.unbind()
        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)
        return torch.stack([x1, y1, x2, y2]).clamp(0.0, 1.0)

    def _build_sgg_nodes(self, scene_graphs, prompt_box, device, dtype):
        if not scene_graphs:
            return None, None

        frame_features = []
        frame_boxes = []
        for graph in scene_graphs:
            if graph is None:
                return None, None
            boxes, feats = self._scene_graph_to_tensors(graph, device, dtype)
            if boxes is None or feats is None or boxes.numel() == 0 or feats.numel() == 0:
                return None, None
            boxes = self._normalize_scene_graph_boxes(boxes, graph, device, dtype)
            boxes, feats = self._select_prompt_context_nodes(boxes, feats, prompt_box)
            box_embed = boxes
            frame_features.append(self.sgg_visual_proj(torch.cat([feats, box_embed], dim=-1)))
            frame_boxes.append(boxes)

        node_features = torch.stack(frame_features, dim=0)
        node_boxes = torch.stack(frame_boxes, dim=0)
        return node_features, node_boxes

    def _scene_graph_to_tensors(self, graph, device, dtype):
        if isinstance(graph, str):
            graph = self._load_scene_graph_file(graph)
        if graph is None:
            return None, None
        if isinstance(graph, dict):
            boxes = graph.get('bbox', graph.get('boxes', graph.get('bboxes', None)))
            feats = graph.get('x', graph.get('bbox_feats', graph.get('features', None)))
        else:
            boxes = getattr(graph, 'bbox', None)
            feats = getattr(graph, 'x', None)
        if boxes is None or feats is None:
            return None, None
        boxes = torch.as_tensor(boxes, device=device, dtype=dtype)
        feats = torch.as_tensor(feats, device=device, dtype=dtype)
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or feats.ndim != 2:
            return None, None
        count = min(boxes.shape[0], feats.shape[0])
        return boxes[:count], feats[:count]

    def _load_scene_graph_file(self, path):
        if not path or not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def _normalize_scene_graph_boxes(self, boxes, graph, device, dtype):
        if boxes.max() <= 1.0:
            return boxes.clamp(0.0, 1.0)
        if isinstance(graph, str):
            graph = self._load_scene_graph_file(graph)
        image_h = None
        image_w = None
        if isinstance(graph, dict):
            image_h = graph.get('image_h', None)
            image_w = graph.get('image_w', None)
        if image_h is None or image_w is None:
            scale = boxes.new_tensor([
                boxes[:, [0, 2]].max().clamp_min(1.0),
                boxes[:, [1, 3]].max().clamp_min(1.0),
                boxes[:, [0, 2]].max().clamp_min(1.0),
                boxes[:, [1, 3]].max().clamp_min(1.0),
            ])
        else:
            image_h = float(np.asarray(image_h).reshape(-1)[0])
            image_w = float(np.asarray(image_w).reshape(-1)[0])
            scale = torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=dtype)
        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return (boxes / scale.clamp_min(1.0)).clamp(0.0, 1.0)

    def _select_prompt_context_nodes(self, boxes, feats, prompt_box):
        centers = self._box_center(boxes)
        prompt_center = self._box_center(prompt_box)
        dist = torch.norm(centers - prompt_center[None], dim=-1)
        keep = min(self.num_nodes, boxes.shape[0])
        keep_idx = torch.topk(-dist, k=keep).indices
        order = torch.argsort(dist[keep_idx])
        keep_idx = keep_idx[order]
        boxes = boxes[keep_idx]
        feats = feats[keep_idx]
        if boxes.shape[0] < self.num_nodes:
            repeat = self.num_nodes // boxes.shape[0] + 1
            boxes = boxes.repeat(repeat, 1)[:self.num_nodes]
            feats = feats.repeat(repeat, 1)[:self.num_nodes]
        return boxes, feats

    def _build_nodes(self, video, prompt_box):
        pooled = F.adaptive_avg_pool2d(video, output_size=(3, 3))
        grid_feats = pooled.permute(0, 2, 3, 1).reshape(video.shape[0], 9, video.shape[1])

        coords = torch.linspace(1 / 6, 5 / 6, 3, device=video.device, dtype=video.dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        half = torch.full_like(xx, 1 / 6)
        grid_boxes = torch.stack(
            [xx - half, yy - half, xx + half, yy + half], dim=-1).reshape(9, 4)

        prompt_feat = self._roi_pool_prompt(video, prompt_box).unsqueeze(1)
        prompt_box = prompt_box.reshape(1, 4)

        node_feats = torch.cat([prompt_feat, grid_feats[:, :self.num_nodes - 1]], dim=1)
        node_boxes = torch.cat([prompt_box, grid_boxes[:self.num_nodes - 1]], dim=0)
        box_embed = node_boxes.unsqueeze(0).expand(video.shape[0], -1, -1)
        node_feats = self.visual_proj(torch.cat([node_feats, box_embed], dim=-1))
        return node_feats, node_boxes

    def _roi_pool_prompt(self, video, box):
        _, _, height, width = video.shape
        x1, y1, x2, y2 = box
        x1 = int((x1 * width).floor().clamp(0, width - 1).item())
        x2 = int((x2 * width).ceil().clamp(x1 + 1, width).item())
        y1 = int((y1 * height).floor().clamp(0, height - 1).item())
        y2 = int((y2 * height).ceil().clamp(y1 + 1, height).item())
        return video[:, :, y1:y2, x1:x2].mean(dim=(-2, -1))

    def _hard_prompt_filter(self, node_features, node_boxes, prompt_box):
        if node_boxes.ndim == 3:
            return self._hard_prompt_filter_temporal(node_features, node_boxes, prompt_box)
        if node_features.shape[1] <= 2:
            return node_features, node_boxes
        prompt_center = self._box_center(prompt_box)
        centers = self._box_center(node_boxes)
        dist = torch.norm(centers - prompt_center[None], dim=-1)
        keep = max(2, int(node_features.shape[1] * self.hard_keep_ratio))
        keep_idx = torch.topk(-dist, k=keep).indices
        keep_idx = torch.unique(torch.cat([keep_idx.new_tensor([0]), keep_idx]), sorted=True)
        return node_features[:, keep_idx], node_boxes[keep_idx]

    def _build_edges(self, node_features, node_boxes):
        if node_boxes.ndim == 3:
            return self._build_edges_temporal(node_features, node_boxes)
        num_frames, num_nodes, _ = node_features.shape
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        edge_feats = []
        adjacency_ps = node_features.new_zeros(num_frames, len(pairs), num_nodes)
        adjacency_po = node_features.new_zeros(num_frames, len(pairs), num_nodes)
        for edge_idx, (src, dst) in enumerate(pairs):
            src_feat = node_features[:, src]
            dst_feat = node_features[:, dst]
            geom = torch.cat([node_boxes[src], node_boxes[dst]], dim=0)
            geom = geom[None].expand(num_frames, -1)
            edge_feats.append(self.edge_proj(torch.cat([src_feat, dst_feat, geom], dim=-1)))
            adjacency_ps[:, edge_idx, src] = 1
            adjacency_po[:, edge_idx, dst] = 1
        return torch.stack(edge_feats, dim=1), adjacency_ps, adjacency_po

    def _hard_prompt_filter_temporal(self, node_features, node_boxes, prompt_box):
        if node_features.shape[1] <= 2:
            return node_features, node_boxes
        prompt_center = self._box_center(prompt_box)
        centers = self._box_center(node_boxes)
        dist = torch.norm(centers - prompt_center[None, None], dim=-1).mean(dim=0)
        keep = max(2, int(node_features.shape[1] * self.hard_keep_ratio))
        keep_idx = torch.topk(-dist, k=keep).indices
        keep_idx = torch.unique(torch.cat([keep_idx.new_tensor([0]), keep_idx]), sorted=True)
        return node_features[:, keep_idx], node_boxes[:, keep_idx]

    def _build_edges_temporal(self, node_features, node_boxes):
        num_frames, num_nodes, _ = node_features.shape
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        edge_feats = []
        adjacency_ps = node_features.new_zeros(num_frames, len(pairs), num_nodes)
        adjacency_po = node_features.new_zeros(num_frames, len(pairs), num_nodes)
        for edge_idx, (src, dst) in enumerate(pairs):
            src_feat = node_features[:, src]
            dst_feat = node_features[:, dst]
            geom = torch.cat([node_boxes[:, src], node_boxes[:, dst]], dim=-1)
            edge_feats.append(self.edge_proj(torch.cat([src_feat, dst_feat, geom], dim=-1)))
            adjacency_ps[:, edge_idx, src] = 1
            adjacency_po[:, edge_idx, dst] = 1
        return torch.stack(edge_feats, dim=1), adjacency_ps, adjacency_po

    def _soft_filter_nodes(self, node_features, edge_features):
        num_frames, num_nodes, dim = node_features.shape
        edge_idx = 0
        triplets = []
        for src in range(num_nodes):
            row = []
            for dst in range(num_nodes):
                if src == dst:
                    row.append(node_features.new_zeros(num_frames, dim))
                else:
                    row.append(self.triplet_proj(torch.cat([
                        node_features[:, src],
                        edge_features[:, edge_idx],
                        node_features[:, dst],
                    ], dim=-1)))
                    edge_idx += 1
            triplets.append(torch.stack(row, dim=1))
        triplets = torch.stack(triplets, dim=1).reshape(num_frames, num_nodes * num_nodes, dim)
        attended, _ = self.triplet_attn(triplets, triplets, triplets)
        alpha = self.association_mlp(attended).reshape(num_frames, num_nodes, num_nodes)
        alpha = torch.sigmoid(alpha)
        alpha = alpha.mean(dim=-1, keepdim=True)
        return node_features * (1.0 + alpha)

    def _temporal_update(self, node_features):
        outputs = [node_features[0]]
        for t in range(1, node_features.shape[0]):
            start = max(0, t - self.temporal_window)
            memory = node_features[start:t].reshape(1, -1, node_features.shape[-1])
            query = node_features[t:t + 1]
            updated, _ = self.temporal_attn(query, memory, memory)
            outputs.append((query + updated).squeeze(0))
        return torch.stack(outputs, dim=0)

    def _aggregate_graph(self, node_features, edge_features, adjacency_ps, adjacency_po):
        subject_msg = torch.bmm(adjacency_ps.transpose(1, 2), edge_features)
        object_msg = torch.bmm(adjacency_po.transpose(1, 2), edge_features)
        graph_features = (
            node_features
            + self.edge_subject_mlp(subject_msg)
            + self.edge_object_mlp(object_msg)
        )
        return self.graph_norm(graph_features)

    @staticmethod
    def _box_center(boxes):
        return torch.stack([
            (boxes[..., 0] + boxes[..., 2]) * 0.5,
            (boxes[..., 1] + boxes[..., 3]) * 0.5,
        ], dim=-1)
