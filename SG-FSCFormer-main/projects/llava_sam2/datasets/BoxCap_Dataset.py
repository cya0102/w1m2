import copy
import json
import logging
import os


import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import pycocotools.mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset

from .encode_fn import video_lisa_encode_fn

BOXCAP_QUESTION = (
    "Based on the provided prompt box [BOX] ,Could you please give me a detailed "
    "description of the video? Please respond with interleaved segmentation masks "
    "for the corresponding parts of the answer."
)


def _format_boxes(boxes):
    if not boxes:
        return "[]"
    formatted = [f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]" for x1, y1, x2, y2 in boxes]
    return "[" + ", ".join(formatted) + "]"


def _normalize_boxes(boxes, height, width):
    if height is None or width is None:
        return boxes
    normalized_boxes = []
    for x1, y1, x2, y2 in boxes:
        normalized_boxes.append([
            float(x1) / max(float(width), 1.0),
            float(y1) / max(float(height), 1.0),
            float(x2) / max(float(width), 1.0),
            float(y2) / max(float(height), 1.0),
        ])
    return normalized_boxes


def _build_interleaved_answer(captions):
    if not captions:
        return "[SEG]."
    segments = [f"<p> {cap.strip()} </p> [SEG]" for cap in captions if cap and cap.strip()]
    if not segments:
        return "[SEG]."
    return " ".join(segments)


def _safe_decode_rle(rle, height, width):
    if rle is None:
        return np.zeros((height, width), dtype=np.uint8)
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


class VideoBoxCapDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    def __init__(
            self,
            image_folder,
            annotation_file,
            tokenizer=None,
            sampled_frames=5,
            extra_image_processor=None,
            template_map_fn=None,
            max_length=8192,
            lazy=True,
            repeats=1,
            special_tokens=None,
            arch_type='intern_vl',
            preprocessor=None,
            scene_graph_root=None,
            scene_graph_split=None,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.sampled_frames = sampled_frames
        self.max_length = max_length
        self.lazy = lazy
        self.repeats = repeats
        self.image_folder = image_folder
        self.scene_graph_root = scene_graph_root
        self.scene_graph_split = scene_graph_split

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        json_datas = self.json_file_preprocess(annotation_file)
        if self.scene_graph_root is None:
            self.scene_graph_root = self._infer_scene_graph_root(annotation_file, image_folder)
        if self.scene_graph_split is None:
            self.scene_graph_split = self._infer_scene_graph_split(annotation_file, image_folder)
        json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
        if self.lazy:
            self.text_data = build_origin_dataset(json_data, 'train')
        else:
            raise NotImplementedError

        print(f"Video boxcap dataset, include {len(self.text_data)} items.")

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    @property
    def modality_length(self):
        return [10000 for _ in range(len(self.text_data))] * self.repeats

    def json_file_preprocess(self, annotation_file):
        with open(annotation_file, 'r') as f:
            json_data = json.load(f)
        if isinstance(json_data, dict):
            return list(json_data.values())
        return json_data

    def _sample_frame_indices(self, num_frames):
        if num_frames <= 0:
            return []
        if num_frames <= self.sampled_frames:
            indices = list(range(num_frames))
            if num_frames < self.sampled_frames:
                indices += [indices[-1]] * (self.sampled_frames - num_frames)
            return indices
        step = num_frames // self.sampled_frames
        return [i * step for i in range(self.sampled_frames)]

    def _load_frames(self, file_names, indices):
        images = []
        for idx in indices:
            frame_path = os.path.join(self.image_folder, file_names[idx])
            images.append(Image.open(frame_path))
        return images

    def _infer_scene_graph_root(self, annotation_file, image_folder):
        path = f'{annotation_file} {image_folder}'.lower()
        custom_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'custom_data'))
        if 'lvvis' in path:
            return os.path.join(custom_root, 'lvvis', 'annotation')
        if 'ovis' in path:
            return os.path.join(custom_root, 'ovis', 'annotation')
        return None

    def _infer_scene_graph_split(self, annotation_file, image_folder):
        path = f'{annotation_file} {image_folder}'.lower()
        if 'train' in path or '_tr_' in path:
            return 'train'
        if 'valid' in path or 'val' in path:
            return 'val'
        if 'test' in path:
            return 'test'
        return None

    def _load_scene_graphs(self, file_names, indices):
        if self.scene_graph_root is None or self.scene_graph_split is None:
            return None
        scene_graphs = []
        for idx in indices:
            rel_path = os.path.splitext(file_names[idx])[0] + '.npz'
            feat_path = os.path.join(
                self.scene_graph_root, 'bbox_feats', self.scene_graph_split, rel_path)
            if not os.path.exists(feat_path):
                scene_graphs.append(None)
                continue
            with np.load(feat_path, allow_pickle=True) as data:
                graph = {key: data[key] for key in data.files}
            scene_graphs.append(graph)
        return scene_graphs

    def _prepare_text(self, n_frames, question, answer, num_image_tokens=256):
        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        frame_tokens = (frame_token_str + '\n') * n_frames
        frame_tokens = frame_tokens.strip()

        qa_list = [
            {'from': 'human', 'value': frame_tokens + question},
            {'from': 'gpt', 'value': answer}
        ]

        input = ''
        conversation = []
        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        conversation[0].update({'system': ''})
        return {'conversation': conversation}

    def _decode_masks(self, segmentations, frame_indices, height, width, max_objects):
        if not segmentations:
            return None
        object_masks = []
        for obj_idx in range(max_objects):
            obj_seg = segmentations[obj_idx]
            frame_masks = []
            for frame_idx in frame_indices:
                if frame_idx < len(obj_seg):
                    rle = obj_seg[frame_idx]
                else:
                    rle = None
                frame_masks.append(_safe_decode_rle(rle, height, width))
            object_masks.append(np.stack(frame_masks, axis=0))
        object_masks = np.stack(object_masks, axis=0)
        object_masks = torch.from_numpy(object_masks)
        return object_masks.flatten(0, 1)

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        file_names = data_dict['file_names']
        frame_indices = self._sample_frame_indices(len(file_names))
        images = self._load_frames(file_names, frame_indices)
        data_dict['scene_graphs'] = self._load_scene_graphs(file_names, frame_indices)

        captions = data_dict.get('caption', [])
        if isinstance(captions, str):
            captions = [captions]

        boxes = data_dict.get('box', [])
        segmentations = data_dict.get('segmentations', [])

        max_objects = min(len(captions), len(boxes), len(segmentations))
        captions = captions[:max_objects]
        boxes = boxes[:max_objects]

        question = BOXCAP_QUESTION.replace('[BOX]', _format_boxes(boxes))
        answer = _build_interleaved_answer(captions)
        data_dict['segcaption_eval'] = {
            'id': data_dict.get('id', data_dict.get('video_id', index)),
            'caption': ' '.join([cap.strip() for cap in captions if cap and cap.strip()]),
            'phrases': [cap.strip() for cap in captions if cap and cap.strip()],
            'height': data_dict.get('height'),
            'width': data_dict.get('width'),
        }

        text_dict = self._prepare_text(
            n_frames=len(frame_indices),
            question=question,
            answer=answer,
            num_image_tokens=self.patch_token,
        )

        data_dict['conversation'] = text_dict['conversation']

        height = data_dict.get('height')
        width = data_dict.get('width')
        data_dict['bboxes'] = _normalize_boxes(boxes, height, width)
        if height is not None and width is not None and max_objects > 0:
            data_dict['masks'] = self._decode_masks(
                segmentations=segmentations,
                frame_indices=frame_indices,
                height=height,
                width=width,
                max_objects=max_objects,
            )
        else:
            data_dict['masks'] = None

        if self.preprocessor is not None:
            if self.arch_type == 'qwen':
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                data_dict.update(_data_dict)
            elif self.arch_type == 'llava':
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                data_dict.update(_data_dict)
            else:
                raise NotImplementedError
        else:
            pixel_values = [self.transformer(image.convert('RGB')) for image in images]
            pixel_values = torch.stack(pixel_values, dim=0)
            data_dict['pixel_values'] = pixel_values

        if hasattr(self, 'extra_image_processor'):
            g_pixel_values = []
            for image in images:
                g_image = np.array(image.convert('RGB'))
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values.append(torch.from_numpy(g_image).permute(2, 0, 1).contiguous())
            data_dict['g_pixel_values'] = g_pixel_values

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(
            data_dict,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            with_image_token=True,
        )
        data_dict.update(result)

        data_dict['type'] = 'video'
        return data_dict
