from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from .lisa import LisaModel

from xtuner.utils import IGNORE_INDEX, PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

import numpy as np

from .internvl import InternVL_Slowfast
from .utils import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness
from .ptgformer import PromptGuidedTemporalGraphFormer
from .mldecoder import FineGrainedMaskLinguisticDecoder

class VideoLLaVASAMModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 use_fast_supervision=False,
                 # for inference
                 phi3=True,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # for inference large model
                 split_model=False,
                 # ext
                 preprocessor=None,
                 # bs
                 bs:int=0,
                 # prompt-guided temporal graph former
                 ptgformer=None,
                 # fine-grained mask-linguistic decoder
                 mldecoder=None,
                 validation_generate=True,
                 validation_max_new_tokens=128,
                 ):
        super(LisaModel, self).__init__()
        self.split_model = split_model
        if split_model:
            mllm.model_split = split_model
        if special_tokens is None:
            special_tokens = ['[SEG]']
        self.special_tokens = special_tokens
        if 'special_tokens' not in mllm.keys():
            mllm.special_tokens = special_tokens
        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        if hasattr(self.mllm, '_post_init'):
            self.mllm._post_init(
                fast_pool_size=self.fast_pool_size,
                fast_pool=self.fast_pool
            )
        else:
            print("No _post_init() in mllm !!!")

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen':
                self.mllm.model.model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
            elif self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            # self.mllm.model.language_model.base_model.model.lm_head.requires_grad_(True)
            # self.mllm.model.language_model.base_model.model.model.embed_tokens.requires_grad_(True)

        if self.arch_type == 'intern_vl':
            in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            # for llava, the hidden size is in language model
            in_dim = self.mllm.model.language_model.config.hidden_size
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        if ptgformer is not None:
            if isinstance(ptgformer, dict):
                ptgformer.setdefault('hidden_dim', out_dim)
                self.ptgformer = BUILDER.build(ptgformer)
            else:
                self.ptgformer = ptgformer
            self.ptgformer_fusion = nn.Sequential(
                nn.LayerNorm(out_dim * 2),
                nn.Linear(out_dim * 2, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.ptgformer = None
            self.ptgformer_fusion = None
        if mldecoder is not None:
            if isinstance(mldecoder, dict):
                mldecoder.setdefault('hidden_dim', out_dim)
                self.mldecoder = BUILDER.build(mldecoder)
            else:
                self.mldecoder = mldecoder
        else:
            self.mldecoder = None
        self.validation_generate = validation_generate
        self.validation_max_new_tokens = validation_max_new_tokens

        if use_fast_supervision:
            self.text_exist_fcs = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                nn.Linear(in_dim, 1), nn.Dropout(0.0)
            )

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        if use_fast_supervision:
            self.loss_exists = BUILDER.build(dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
            )

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        if fast_pool:
            self.fast_token_idx = self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0]
        else:
            self.fast_token_idx = None
        self.use_fast_supervision = use_fast_supervision

        self.phi3 = phi3
        self.template = template

        if preprocessor is None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = BUILDER.build(preprocessor)

        self.bs = bs

    def _merge_lora(self):
        # print('pre merge lora: ', self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape)
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        # print('after merge lora: ', self.mllm.model.language_model.get_input_embeddings().weight.shape)
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        return state_dict

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        # if not isinstance(self.mllm.model.language_model.get_output_embeddings(), nn.Linear):
        #     print("Change the lm_head to nn.Linear !!!")
        #     transposed = False
        #     old_lm_head = self.mllm.model.language_model.get_output_embeddings()
        #     old_num_tokens, old_lm_head_dim = (
        #         old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        #     )
        #     new_lm_head_shape = (old_lm_head_dim, len(tokenizer)) if not transposed else (
        #     len(tokenizer), old_lm_head_dim)
        #     has_new_lm_head_bias = old_lm_head.bias is not None
        #     new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias).to(self.device)
        #     new_lm_head.weight = old_lm_head.weight
        #     new_lm_head.bias = old_lm_head.bias
        #     self.mllm.model.language_model.set_output_embeddings(new_lm_head)

        # this is already done in mllm
        # if num_new_tokens > 0:
        #     self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))

        # assert isinstance(self.mllm, InternVL_Slowfast)
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
            elif self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
            raise NotImplementedError
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})

        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})

        # Step 5. others (fcs)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_exist_fcs.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'ptgformer.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'ptgformer_fusion.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mldecoder.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k or 'output' in k and 'sam2_model' not in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k or 'tok_embeddings' in k})
        return to_return

    def _get_ptgformer_frame_features(self, pixel_values, bboxes, frames_per_batch, scene_graphs=None):
        if self.ptgformer is None or bboxes is None or frames_per_batch is None:
            return None
        if len(bboxes) != len(frames_per_batch):
            return None
        try:
            graph_outputs = self.ptgformer(
                pixel_values=pixel_values,
                bboxes=bboxes,
                frames_per_batch=frames_per_batch,
                scene_graphs=scene_graphs,
            )
        except Exception as exc:
            print(f'Warning, skip PTGFormer because graph construction failed: {exc}')
            return None
        frame_graph_features = graph_outputs.get('frame_graph_features', None)
        if frame_graph_features is None or len(frame_graph_features) != len(frames_per_batch):
            return None
        flattened_features = []
        for features, n_frames in zip(frame_graph_features, frames_per_batch):
            if features.shape[0] != n_frames:
                return None
            flattened_features.extend([features[i] for i in range(n_frames)])
        return flattened_features

    def _fuse_ptgformer_embeddings(self, pred_embeddings_list_video, frame_graph_features):
        if self.ptgformer_fusion is None or frame_graph_features is None:
            return pred_embeddings_list_video
        if len(pred_embeddings_list_video) != len(frame_graph_features):
            return pred_embeddings_list_video
        fused_embeddings = []
        for pred_embeddings, graph_feature in zip(pred_embeddings_list_video, frame_graph_features):
            graph_feature = graph_feature.to(
                device=pred_embeddings.device,
                dtype=pred_embeddings.dtype,
            )
            graph_feature = graph_feature.mean(dim=0, keepdim=True).expand(
                pred_embeddings.shape[0], -1)
            fusion_input = torch.cat([pred_embeddings, graph_feature], dim=-1)
            fusion_dtype = next(self.ptgformer_fusion.parameters()).dtype
            fusion_delta = self.ptgformer_fusion(fusion_input.to(fusion_dtype))
            fused_embeddings.append(pred_embeddings + fusion_delta.to(pred_embeddings.dtype))
        return fused_embeddings

    def _decode_with_mldecoder(
        self,
        pred_embeddings_list_video,
        frame_graph_features,
        alignment_targets=None,
    ):
        if self.mldecoder is None:
            return pred_embeddings_list_video, {}
        decoder_outputs = self.mldecoder(
            pred_embeddings_list_video=pred_embeddings_list_video,
            frame_graph_features=frame_graph_features,
            alignment_targets=alignment_targets,
        )
        return decoder_outputs['pred_embeddings_list_video'], decoder_outputs

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        assert self.bs > 0
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        return_predictions = data.pop('_return_predictions', False)
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        bboxes = data.pop('bboxes', None)
        scene_graphs = data.pop('scene_graphs', None)
        alignment_targets = data.pop('alignment_targets', None)
        input_ids = data['input_ids']
        fast_exists = data.pop('fast_exists', None)
        # if self.arch_type == 'llava' and data.get('pixel_values', None) is not None:
        #     data['pixel_values'] = data['pixel_values'].to(self.torch_dtype)
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError

        frame_graph_features = self._get_ptgformer_frame_features(
            pixel_values=data.get('pixel_values', None),
            bboxes=bboxes,
            frames_per_batch=frames_per_batch,
            scene_graphs=scene_graphs,
        )
        pred_embeddings_list_video = self._fuse_ptgformer_embeddings(
            pred_embeddings_list_video,
            frame_graph_features,
        )

        if self.use_fast_supervision and fast_exists is not None:
            # gt_exists = []
            # for id_x, _fast_exists in enumerate(fast_exists):
            #     num_tot = _fast_exists.shape[0]
            #     num_conv = gt_masks[id_x].shape[0] // frames_per_batch[id_x]
            #     assert num_tot % num_conv == 0
            #     gt_exists.append(_fast_exists.reshape(num_conv, num_tot // num_conv))
            fast_flag = input_ids == self.fast_token_idx
            fast_tokens = output.hidden_states[-1][fast_flag]
            exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
            gt_exists = torch.cat(fast_exists)
            loss_exists = self.loss_exists(exists_logit, gt_exists)
        else:
            loss_exists = None

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        pred_embeddings_list_video, decoder_outputs = self._decode_with_mldecoder(
            pred_embeddings_list_video,
            frame_graph_features,
            alignment_targets=alignment_targets,
        )
        loss_fa = decoder_outputs.get('loss_fa', None)
        loss_mc = decoder_outputs.get('loss_mc', None)
        loss_mc_raw = decoder_outputs.get('loss_mc_raw', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        metric_pred_masks = pred_masks.detach()
        metric_gt_masks = gt_masks.detach()

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        if loss_exists is not None:
            loss_dict['loss_exists'] = loss_exists
        if loss_fa is not None:
            loss_dict['loss_fa'] = loss_fa * _scale
        if loss_mc is not None:
            loss_dict['loss_mc'] = loss_mc * _scale
        if self.mldecoder is not None:
            loss_dict['sg_fsc_total'] = self.mldecoder.compose_total_loss(
                caption_loss=output.loss,
                mask_loss=loss_mask + loss_dice,
                loss_fa=loss_dict.get('loss_fa', None),
                loss_mc_raw=loss_mc_raw * _scale if loss_mc_raw is not None else None,
            ).detach()
        if return_predictions:
            loss_dict['pred_masks'] = metric_pred_masks
            loss_dict['gt_masks'] = metric_gt_masks
        return loss_dict

    def _copy_data_for_validation(self, data):
        copied = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                copied[key] = value
            elif isinstance(value, list):
                copied[key] = list(value)
            elif isinstance(value, dict):
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    def _decode_label_caption(self, labels):
        label_ids = labels[labels != IGNORE_INDEX]
        if label_ids.numel() == 0:
            return ''
        return self.tokenizer.decode(label_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()

    def _build_validation_prompt_ids(self, input_ids, labels, attention_mask=None):
        valid_mask = labels == IGNORE_INDEX
        if attention_mask is not None:
            valid_mask = torch.logical_and(valid_mask, attention_mask.bool())
        return input_ids[valid_mask].detach().cpu().tolist()

    def _predict_validation_sample(self, data, sample_idx, eval_meta):
        if not self.validation_generate:
            return {}
        if not hasattr(self, 'gen_config') or not getattr(self, 'init_prediction_config', False):
            generation_kwargs = {'max_new_tokens': self.validation_max_new_tokens}
            self.preparing_for_generation({
                'template': self.template or PROMPT_TEMPLATE.phi3_chat,
                'generation_kwargs': generation_kwargs,
            })
        prompt_ids = self._build_validation_prompt_ids(
            data['input_ids'][sample_idx],
            data['labels'][sample_idx],
            data.get('attention_mask', None)[sample_idx] if data.get('attention_mask', None) is not None else None,
        )
        pixel_values = data.get('pixel_values', None)
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            pixel_values = pixel_values[sample_idx]
        g_pixel_values = data.get('g_pixel_values', None)
        frames_per_batch = data.get('frames_per_batch', None)
        if g_pixel_values is not None and frames_per_batch is not None and len(frames_per_batch) > 1:
            start = sum(frames_per_batch[:sample_idx])
            end = start + frames_per_batch[sample_idx]
            g_pixel_values = g_pixel_values[start:end]
        if pixel_values is None or g_pixel_values is None:
            return {}
        try:
            pred = self.predict_video(
                pixel_values=pixel_values,
                text_prompts=[eval_meta.get('caption', '')],
                input_ids=[prompt_ids],
                g_pixel_values=g_pixel_values,
                ori_height=eval_meta.get('height') or 0,
                ori_width=eval_meta.get('width') or 0,
            )
        except Exception as exc:
            return {'generation_error': str(exc)}
        return {
            'pred_caption': pred.get('prediction', [''])[0],
            'pred_masks_rle': pred.get('prediction_masks', [[]])[0],
        }

    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        batch_data = data['data'] if isinstance(data, dict) and 'data' in data else data
        data_samples = data.get('data_samples', None) if isinstance(data, dict) else None

        loss_data = self._copy_data_for_validation(batch_data)
        eval_meta = loss_data.pop('segcaption_eval', None)
        loss_data['_return_predictions'] = True
        with torch.no_grad():
            outputs = self.forward(loss_data, data_samples=data_samples, mode='loss')

        labels = batch_data.get('labels', None)
        batch_size = labels.shape[0] if isinstance(labels, torch.Tensor) else 1
        eval_meta = eval_meta or [None] * batch_size
        results = []
        for sample_idx in range(batch_size):
            meta = eval_meta[sample_idx] if sample_idx < len(eval_meta) and eval_meta[sample_idx] else {}
            result = {
                'id': meta.get('id', sample_idx),
                'gt_caption': meta.get('caption', ''),
                'gt_phrases': meta.get('phrases', []),
            }
            if labels is not None:
                result['teacher_caption'] = self._decode_label_caption(labels[sample_idx])
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    result[key] = value.detach().float().cpu()
            if 'pred_masks' in outputs and 'gt_masks' in outputs:
                result['pred_masks'] = outputs['pred_masks'].detach().float().cpu()
                result['gt_masks'] = outputs['gt_masks'].detach().float().cpu()
            result.update(self._predict_validation_sample(batch_data, sample_idx, meta))
            results.append(result)
        return results

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(len(pred_embeddings_list), len(frames_per_batch)))
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.mllm.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype)
        # if getattr(self, 'text_exist_fcs', None) is not None:
        #     self.text_exist_fcs.to(self.torch_dtype)

        # for sam image processor
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        if 'max_dynamic_patch' in metainfo.keys():
            self.max_dynamic_patch = metainfo['max_dynamic_patch']
        else:
            self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if self.preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            # self.preprocessor = BUILDER.build(self.preprocessor)

        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.phi3:
            self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.mllm.model.language_model)
        return

    def predict_video(self, pixel_values, text_prompts, **kwargs):
        ori_h, ori_w = kwargs['ori_height'], kwargs['ori_width']

        _input_ids = kwargs['input_ids']

        g_pixel_values = kwargs.pop('g_pixel_values', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])

        fast_pixel_values = kwargs.pop('fast_pixel_values', None)
        if fast_pixel_values is None:
            fast_token_idx = None
        else:
            fast_token_idx = self.fast_token_idx

        predictions = []
        pred_masks = []
        is_exists_list = []
        for input_ids in _input_ids:
            input_ids = torch.tensor(input_ids, device=pixel_values.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            pixel_values = pixel_values.to(dtype=self.torch_dtype)
            if fast_pixel_values is not None:
                fast_pixel_values = fast_pixel_values.to(dtype=self.torch_dtype)
            mm_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'fast_pixel_values': fast_pixel_values,
                'fast_token_idx': fast_token_idx,
            }
            if kwargs.get('image_grid_thw', None) is not None:
                mm_inputs['image_grid_thw'] = kwargs['image_grid_thw']

            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()

            # input_text = self.tokenizer.decode(mm_inputs['input_ids'][0], skip_special_tokens=False)
            # print(input_text, generate_output.sequences[0], '\n', predict, self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0])

            predictions.append(predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=self.seg_token_idx
            )

            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                pred_masks.append(torch.zeros((g_pixel_values.shape[0], ori_h, ori_w), dtype=torch.int))
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]
            seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

            seg_hidden_states = seg_hidden_states.to(dtype=torch.float32)

            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            # TODO: change 5
            if len(pixel_values) < 5:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * pixel_values.shape[0])
            else:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * 5)
            pred_mask = F.interpolate(
                pred_mask,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False,
            )
            pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.sigmoid() > 0.5
            pred_mask = pred_mask.int()
            # supervision
            if self.use_fast_supervision and (input_ids == self.fast_token_idx).sum() > 0:
                fast_flag = input_ids.squeeze(0) == self.fast_token_idx
                len_out = generate_output.sequences[0][:-1].shape[0]
                fast_tokens = last_hidden_states[:-len_out][fast_flag].to(dtype=torch.float32)
                exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
                is_exists = exists_logit.squeeze(-1).sigmoid() > 0.5
                is_exists_list.append(is_exists)
                not_exists = torch.logical_not(is_exists)
                if torch.any(not_exists):
                    pred_mask[not_exists] = pred_mask[not_exists] * 0

            pred_masks.append(pred_mask)
        assert len(pred_masks) == len(text_prompts)
        ret_dict = {
            'prediction': predictions,
            'prediction_masks': [mask_to_rle(_item.cpu().numpy()) for _item in pred_masks],
        }
        if 'id' in kwargs.keys():
            ret_dict['id'] = kwargs['id']

        if len(is_exists_list) > 0:
            ret_dict['is_exists'] = is_exists_list
        return ret_dict


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs


class VideoLLaVASAMModel_zero3(VideoLLaVASAMModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=['[SEG]', ],
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 arch_type='intern_vl',
                 # zero3
                 bs=1,
                 ptgformer=None,
                 mldecoder=None,
                 validation_generate=True,
                 validation_max_new_tokens=128,
                 ):
        super(VideoLLaVASAMModel_zero3, self).__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            grounding_encoder=grounding_encoder,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            torch_dtype=torch_dtype,
            pretrained_pth=pretrained_pth,
            frozen_sam2_decoder=frozen_sam2_decoder,
            special_tokens=special_tokens,
            loss_sample_points=loss_sample_points,
            num_points=num_points,
            # for slow fast arch
            fast_pool=fast_pool,
            fast_pool_size=fast_pool_size,
            arch_type=arch_type,
            ptgformer=ptgformer,
            mldecoder=mldecoder,
            validation_generate=validation_generate,
            validation_max_new_tokens=validation_max_new_tokens,
        )
        self.bs = bs

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        return_predictions = data.pop('_return_predictions', False)
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        bboxes = data.pop('bboxes', None)
        scene_graphs = data.pop('scene_graphs', None)
        alignment_targets = data.pop('alignment_targets', None)
        input_ids = data['input_ids']
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError
            # return {'llm_loss': output.loss, 'loss_mask': output.loss * 0.0, 'loss_dice': output.loss * 0.0}

        frame_graph_features = self._get_ptgformer_frame_features(
            pixel_values=data.get('pixel_values', None),
            bboxes=bboxes,
            frames_per_batch=frames_per_batch,
            scene_graphs=scene_graphs,
        )
        pred_embeddings_list_video = self._fuse_ptgformer_embeddings(
            pred_embeddings_list_video,
            frame_graph_features,
        )

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        pred_embeddings_list_video, decoder_outputs = self._decode_with_mldecoder(
            pred_embeddings_list_video,
            frame_graph_features,
            alignment_targets=alignment_targets,
        )
        loss_fa = decoder_outputs.get('loss_fa', None)
        loss_mc = decoder_outputs.get('loss_mc', None)
        loss_mc_raw = decoder_outputs.get('loss_mc_raw', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        # print(f"Done, {g_pixel_values.device} !!!\n\n")
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        # print(f"Done, {g_pixel_values.device} !!! {num_frames}---{num_objs}, {language_embeddings.shape}\n\n")
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)
        # pred_masks = torch.cat(pred_masks, dim=0)


        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        metric_pred_masks = pred_masks.detach()
        metric_gt_masks = gt_masks.detach()

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        if loss_fa is not None:
            loss_dict['loss_fa'] = loss_fa * _scale
        if loss_mc is not None:
            loss_dict['loss_mc'] = loss_mc * _scale
        if self.mldecoder is not None:
            loss_dict['sg_fsc_total'] = self.mldecoder.compose_total_loss(
                caption_loss=output.loss,
                mask_loss=loss_mask + loss_dice,
                loss_fa=loss_dict.get('loss_fa', None),
                loss_mc_raw=loss_mc_raw * _scale if loss_mc_raw is not None else None,
            ).detach()
        if return_predictions:
            loss_dict['pred_masks'] = metric_pred_masks
            loss_dict['gt_masks'] = metric_gt_masks
        return loss_dict
