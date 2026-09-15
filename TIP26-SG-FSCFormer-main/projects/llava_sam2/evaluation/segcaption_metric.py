import json
import math
import os
import re
import tempfile
from collections import Counter

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from pycocotools import mask as mask_utils


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_binary_masks(value):
    masks = _to_numpy(value)
    if masks is None:
        return []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return [(mask > 0).astype(np.uint8) for mask in masks]


def _decode_rle_masks(rles):
    masks = []
    for rle in rles or []:
        try:
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            masks.append((mask > 0).astype(np.uint8))
        except Exception:
            continue
    return masks


def _mask_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_mask, gt_mask).sum() / union)


def _mask_boundary(mask):
    mask = mask.astype(bool)
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def _boundary_f_score(pred_mask, gt_mask):
    pred_boundary = _mask_boundary(pred_mask)
    gt_boundary = _mask_boundary(gt_mask)
    pred_count = pred_boundary.sum()
    gt_count = gt_boundary.sum()
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    overlap = np.logical_and(pred_boundary, gt_boundary).sum()
    precision = overlap / max(float(pred_count), 1.0)
    recall = overlap / max(float(gt_count), 1.0)
    return float(2 * precision * recall / max(precision + recall, 1e-6))


def _cosine_bow(text_a, text_b):
    tokens_a = re.findall(r'\w+', (text_a or '').lower())
    tokens_b = re.findall(r'\w+', (text_b or '').lower())
    if not tokens_a or not tokens_b:
        return 0.0
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    vocab = set(counts_a) | set(counts_b)
    dot = sum(counts_a[token] * counts_b[token] for token in vocab)
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    return float(dot / max(norm_a * norm_b, 1e-6))


def _token_f_scores(pred, gt):
    pred_tokens = re.findall(r'\w+', (pred or '').lower())
    gt_tokens = re.findall(r'\w+', (gt or '').lower())
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    overlap = sum(min(pred_counts[token], gt_counts[token]) for token in pred_counts)
    precision = overlap / max(float(len(pred_tokens)), 1.0)
    recall = overlap / max(float(len(gt_tokens)), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return precision, recall, f1


def _extract_phrases(caption):
    phrases = re.findall(r'<p>\s*(.*?)\s*</p>', caption or '', flags=re.IGNORECASE)
    if phrases:
        return [phrase.strip() for phrase in phrases if phrase.strip()]
    parts = re.split(r'\[SEG\]|[.;,]', caption or '')
    return [part.strip() for part in parts if part.strip()]


def _average_precision(matches, scores, num_gt):
    if num_gt <= 0 or not matches:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    tp = np.asarray(matches, dtype=np.float32)[order]
    fp = 1.0 - tp
    recalls = np.cumsum(tp) / max(float(num_gt), 1.0)
    precisions = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-6)
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for idx in range(len(precisions) - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    changed = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[changed + 1] - recalls[changed]) * precisions[changed + 1]))


class SegCaptionMetric(BaseMetric):
    """Validation metric for fine-grained video SegCaptioning."""

    default_prefix = 'segcaption'

    def __init__(self, text_sim_threshold=0.5, iou_threshold=0.5, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.text_sim_threshold = text_sim_threshold
        self.iou_threshold = iou_threshold

    def process(self, data_batch, data_samples):
        if isinstance(data_samples, dict):
            data_samples = [data_samples]
        self.results.extend(data_samples or [])

    def _compute_caption_metrics(self, results):
        pairs = [
            (idx, item.get('pred_caption', ''), item.get('gt_caption', ''))
            for idx, item in enumerate(results)
            if item.get('pred_caption') and item.get('gt_caption')
        ]
        metrics = {
            'METEOR': 0.0,
            'CIDEr': 0.0,
            'SPICE': 0.0,
            'caption_eval_samples': float(len(pairs)),
            'caption_eval_official': 0.0,
        }
        if not pairs:
            return metrics
        try:
            from pycocoevalcap.eval import COCOEvalCap
            from pycocotools.coco import COCO
            with tempfile.TemporaryDirectory() as tmp_dir:
                gt_path = os.path.join(tmp_dir, 'caption_gt.json')
                pred_path = os.path.join(tmp_dir, 'caption_pred.json')
                gt_json = {
                    'images': [{'id': idx} for idx, _, _ in pairs],
                    'annotations': [{'id': idx, 'image_id': idx, 'caption': gt} for idx, _, gt in pairs],
                    'type': 'captions',
                    'info': {},
                    'licenses': [],
                }
                pred_json = [{'image_id': idx, 'caption': pred} for idx, pred, _ in pairs]
                with open(gt_path, 'w') as f:
                    json.dump(gt_json, f)
                with open(pred_path, 'w') as f:
                    json.dump(pred_json, f)
                coco = COCO(gt_path)
                coco_result = coco.loadRes(pred_path)
                coco_eval = COCOEvalCap(coco, coco_result)
                coco_eval.params['image_id'] = coco_result.getImgIds()
                coco_eval.evaluate()
                for key in ('METEOR', 'CIDEr', 'SPICE'):
                    metrics[key] = float(coco_eval.eval.get(key, 0.0))
                metrics['caption_eval_official'] = 1.0
        except Exception:
            meteor_scores, cider_scores, spice_scores = [], [], []
            for _, pred, gt in pairs:
                precision, recall, f1 = _token_f_scores(pred, gt)
                meteor_scores.append(10 * precision * recall / max(recall + 9 * precision, 1e-6))
                cider_scores.append(_cosine_bow(pred, gt))
                spice_scores.append(f1)
            metrics['METEOR'] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
            metrics['CIDEr'] = float(np.mean(cider_scores)) if cider_scores else 0.0
            metrics['SPICE'] = float(np.mean(spice_scores)) if spice_scores else 0.0
        return metrics

    def compute_metrics(self, results):
        metrics = self._compute_caption_metrics(results)
        losses = {}
        j_scores, f_scores = [], []
        class_matches, instance_matches, scores = [], [], []
        num_gt = 0

        for item in results:
            for key, value in item.items():
                if key.startswith('loss') or key == 'llm_loss' or key == 'sg_fsc_total':
                    losses.setdefault(key, []).append(float(value))

            pred_masks = _as_binary_masks(item.get('pred_masks'))
            if not pred_masks:
                pred_masks = _decode_rle_masks(item.get('pred_masks_rle'))
            gt_masks = _as_binary_masks(item.get('gt_masks'))
            if not pred_masks or not gt_masks:
                continue

            gt_phrases = item.get('gt_phrases') or []
            pred_phrases = _extract_phrases(item.get('pred_caption', ''))
            num_gt += len(gt_masks)
            used_gt = set()
            for pred_idx, pred_mask in enumerate(pred_masks):
                ious = [_mask_iou(pred_mask, gt_mask) for gt_mask in gt_masks]
                best_gt = int(np.argmax(ious)) if ious else -1
                best_iou = ious[best_gt] if best_gt >= 0 else 0.0
                if pred_idx < len(gt_masks):
                    j_scores.append(_mask_iou(pred_mask, gt_masks[pred_idx]))
                    f_scores.append(_boundary_f_score(pred_mask, gt_masks[pred_idx]))

                is_class_match = best_iou >= self.iou_threshold and best_gt not in used_gt
                class_matches.append(1.0 if is_class_match else 0.0)
                if is_class_match:
                    used_gt.add(best_gt)
                pred_phrase = pred_phrases[pred_idx] if pred_idx < len(pred_phrases) else ''
                gt_phrase = gt_phrases[best_gt] if best_gt < len(gt_phrases) else ''
                text_match = _cosine_bow(pred_phrase, gt_phrase) > self.text_sim_threshold
                instance_matches.append(1.0 if is_class_match and text_match else 0.0)
                scores.append(1.0)

        for key, values in losses.items():
            metrics[key] = float(np.mean(values)) if values else 0.0
        metrics['J'] = float(np.mean(j_scores)) if j_scores else 0.0
        metrics['F'] = float(np.mean(f_scores)) if f_scores else 0.0
        metrics['J&F'] = (metrics['J'] + metrics['F']) / 2.0
        metrics['class_AP'] = _average_precision(class_matches, scores, num_gt)
        metrics['instance_AP'] = _average_precision(instance_matches, scores, num_gt)
        metrics['mask_eval_samples'] = float(len(j_scores))
        return metrics
