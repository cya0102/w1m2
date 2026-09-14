import logging
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from easydict import EasyDict as edict

from models.layers.bert import BertSelfEncoder, BertLMPredictionHead

logger = logging.getLogger(__name__)
class CaptionHead(nn.Module):

    def __init__(
            self,
            word_embedding_size, visual_feature_size,
            pretrained_embedding,
            max_v_len, max_t_len, hidden_size,
            vocab_size, Fusion,
    ):
        super(CaptionHead, self).__init__()
        self.Fusion = Fusion
        self.cap_config = edict(
            word_vec_size=word_embedding_size,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=hidden_size,
            video_feature_size=visual_feature_size,
            layer_norm_eps=1e-12,  # bert layernorm
            hidden_dropout_prob=0.1,  # applies everywhere except attention
            num_hidden_layers=2,  # number of transformer layers
            num_attention_heads=8,
            share_wd_cls_weight=False,
            vocab_size=vocab_size,
            BOS_id=vocab_size - 2,
            EOS_id=vocab_size - 1,
            PAD_id=0
        )
        logger.debug("Caption Head Configuration: %s", self.cap_config)
        self.cap_sa_decoder = BertSelfEncoder(self.cap_config)
        self.prediction_head = BertLMPredictionHead(self.cap_config, self.cap_sa_decoder.word_embeddings.weight)
        self.cap_sa_decoder.word_embeddings.load_state_dict(pretrained_embedding, strict=True)
        self.prediction_head.decoder.load_state_dict(pretrained_embedding, strict=True)
        assert torch.equal(self.cap_sa_decoder.word_embeddings.weight, self.prediction_head.decoder.weight)
        # debug output cfgs

        self.step_counter = 1


    def forward(self, feature2ds, visual_output, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"
        '''fusion all feat'''
        context_feats, context_semantics_feats = self.Fusion(visual_output[0], visual_output[1], visual_output[2],
                                                             visual_output[3], visual_output[4],
                                                             visual_output[5], )
        context_feats = context_feats.unsqueeze(1) #(bsz,visual_dim)->(bsz,1,visual_dim)
        context_semantics_feats = context_semantics_feats.unsqueeze(1) #(bsz,visual_dim)->(bsz,1,visual_dim)
        if (feature2ds.size(1) + context_feats.size(1)+ context_semantics_feats.size(1)) == self.cap_config.max_v_len:
            feature2ds = feature2ds
        else:
            pad_len = self.cap_config.max_v_len -context_feats.size(1)-context_semantics_feats.size(1)- feature2ds.size(1)
            feature2ds = torch.cat([feature2ds,context_feats.repeat(1,pad_len,1)],dim=1)
        input_types = torch.cat(
            [
                torch.full((feature2ds.size(0), feature2ds.size(1)),
                           fill_value=1, dtype=torch.long, device=feature2ds.device),
                torch.full((context_feats.size(0), context_feats.size(1)),
                           fill_value=0, dtype=torch.long, device=context_feats.device),
                torch.full((context_semantics_feats.size(0), context_semantics_feats.size(1)),
                           fill_value=0, dtype=torch.long, device=context_semantics_feats.device),
                torch.full((input_ids.size(0), input_ids.size(1)),
                           fill_value=2, dtype=torch.long, device=input_ids.device)
            ], dim=1
        )
        visual_output = torch.cat([feature2ds, context_feats,context_semantics_feats], dim=1)
        input_mask = torch.cat(
            [
                torch.ones(size=(visual_output.size(0), visual_output.size(1)),
                           dtype=torch.long, device=visual_output.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_output, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        # logger.debug("GT  : %s", self.ids2text(input_ids))
        # logger.debug("Pred: %s", self.probability2text(prediction_scores))
        # if self.step_counter % self.log_interval == 0:
        #     logger.debug("GT  : %s", self.ids2text(input_ids))
        #     logger.debug("Pred: %s", self.probability2text(prediction_scores))
        # self.step_counter += 1
        return prediction_scores


    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionHead.ids2text(predict_ids)

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids: Union[np.ndarray, torch.Tensor]):
        if isinstance(gt_ids, np.ndarray) or isinstance(gt_ids, torch.Tensor):
            assert 0 < len(gt_ids.shape) <= 2, f"gt_ids should be a 1 dim or 2 dim array/tensor, got {gt_ids.shape}"
        else:
            raise ValueError("gt_ids should be np.ndarray or torch.Tensor")
        if isinstance(gt_ids, torch.Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()
        if len(gt_ids.shape) == 1:
            return convert_ids_to_sentence(gt_ids.tolist())
        else:
            return [convert_ids_to_sentence(_gt_ids) for _gt_ids in gt_ids.tolist()]

def convert_ids_to_sentence(tokens):
    from models.layers.clip.clip import _tokenizer
    text = _tokenizer.decode(tokens)
    text_list = text.split(" ")
    new = []
    for i in range(len(text_list)):
        if i == 0:
            new.append(text_list[i].split(">")[-1])
        elif "<|endoftext|>" in text_list[i]:
            break
        else:
            new.append(text_list[i])
    return " ".join(new)