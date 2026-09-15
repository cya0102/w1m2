import logging

import torch
from torch import nn
from typing import Optional, Union

import numpy as np

from models.decoder.caption_decoder import convert_ids_to_sentence
from models.decoder.caption_module import CaptionModule
from models.layers.bert import BertSelfEncoder, BertLMPredictionHead
from models.layers.clip import clip
from models.utils import LayerNorm, PositionEncoding, gelu
from easydict import EasyDict as edict

logger = logging.getLogger(__name__)


class TAndS(nn.Module):
    def __init__(self, visual_dim, hidden_dim, semantics_dim, max_objects, vocab_size, dropout=0.1, nhead=8,
                 use_SFT=False, use_module=None, use_ham=False):
        super(TAndS, self).__init__()
        self.use_SFT = use_SFT
        self.use_module = use_module
        self.use_ham = use_ham
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)  # 词嵌入

        self.video_embeddings = nn.Sequential(
            LayerNorm(visual_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(True),
            LayerNorm(hidden_dim, eps=1e-12),
        )# visaul_dim 到 hidden_dim
        self.object_embeddings = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, visual_dim)
        self.position_embeddings = PositionEncoding(n_filters=hidden_dim, max_len=1000)
        # self.token_type_embeddings = nn.Embedding(3, hidden_dim)  # 3->跟你输入的特征形式有关,有几种形式填数字几
        self.LayerNorm = LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        '''teacher use encoder+decoder+ham'''
        '''student use ham'''
        " case:1 use_sft=true trans+ham; ham"
        if self.use_module is not None:
            if self.use_module=='transformer':
                self.dense = nn.Linear(hidden_dim, hidden_dim)
                self.intermediate_act_fn = gelu
                self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
                self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
            elif self.use_module=='lstm':
                self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2,
                                    batch_first=True, bidirectional=True)

        " case:2 use_sft=false ham"
        if self.use_ham:
            total_visual_dim = 0
            total_semantics_dim = 0
            setattr(self, 'Ua', nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, 'ba', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
            setattr(self, 'wa', nn.Linear(hidden_dim, 1, bias=False))
            total_visual_dim += hidden_dim

            setattr(self, 'Uv', nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, 'bv', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
            setattr(self, 'wv', nn.Linear(hidden_dim, 1, bias=False))
            total_visual_dim += hidden_dim
            total_semantics_dim += semantics_dim
            '''如果跑lstm，total_visual_dim, hidden_dim'''
            '''如果跑bert，total_visual_dim, visual_dim'''
            setattr(self, 'linear_visual_layer', nn.Linear(hidden_dim, visual_dim))
            # setattr(self, 'linear_semantics_layer', nn.Linear(semantics_dim, visual_dim))
            self.bilstm = nn.LSTM(input_size=hidden_dim + hidden_dim+ hidden_dim,
                                hidden_size=hidden_dim // 2,
                                num_layers=1, bidirectional=True, batch_first=True)

    def encoder(self, key, mask):
        memory = key
        memory = self.self_attn(memory, memory, value=memory, attn_mask=None,
                                key_padding_mask=None)[0]
        memory1 = self.dense(memory)
        memory1 = self.dropout(memory1)
        memory_output = self.LayerNorm(memory1 + memory)
        # intermediate
        intermediate_memory_output = self.dense(memory_output)
        intermediate_memory_output = self.intermediate_act_fn(intermediate_memory_output)
        # output
        intermediate_memory_output = self.dense(intermediate_memory_output)
        intermediate_memory_output = self.dropout(intermediate_memory_output)
        memory_hidden_states = self.LayerNorm(intermediate_memory_output + memory_output)
        return memory_hidden_states

    def decoder(self, query, key, value, mask):
        # query_embed = query_embed.unsqueeze(1).repeat(1, query.shape[1], 1)
        tgt = self.self_attn(query , query , value=query, attn_mask=None,
                             key_padding_mask=None)[0]
        tgt = self.dense(tgt)
        tgt = self.dropout(tgt)
        tgt = self.LayerNorm(tgt + query)
        tgt1 = self.multihead_attn(query=tgt,
                                   key=key,
                                   value=value, attn_mask=None,
                                   key_padding_mask=None)[0]
        tgt1 = self.dense(tgt1)
        tgt1 = self.dropout(tgt1)
        tgt_output = self.LayerNorm(tgt + tgt1)
        # intermediate
        intermediate_tgt_output = self.dense(tgt_output)
        intermediate_tgt_output = self.intermediate_act_fn(intermediate_tgt_output)
        # output
        intermediate_tgt_output = self.dense(intermediate_tgt_output)
        intermediate_tgt_output = self.dropout(intermediate_tgt_output)
        tgt_hidden_states = self.LayerNorm(intermediate_tgt_output + tgt_output)
        return tgt_hidden_states

    def ham(self, action_feats, video_feats):

        U_act = self.Ua(action_feats) if hasattr(self, 'Ua') else None
        U_video = self.Uv(video_feats) if hasattr(self, 'Uv') else None

        attn_feat = U_video.unsqueeze(2)+U_act.unsqueeze(1) + self.ba
        attn_weights = self.wa(torch.tanh(attn_feat))
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, max_objects, 1)
        attn1 = attn_weights * attn_feat  # (bsz, max_objects, hidden_dim)
        attn1 = attn1.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        attn_feat = U_video.unsqueeze(2)+attn1.unsqueeze(1) + self.bv
        attn_weights = self.wv(torch.tanh(attn_feat))
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, 1)
        attn_video = attn_weights * attn_feat  # (bsz, sample_numb, hidden_dim)
        attn_video = attn_video.sum(dim=-2)  # (bsz, hidden_dim)

        features = torch.cat([video_feats, attn_video, attn1], dim=-1)  # (bsz, sample_numb, hidden_dim * 3)
        output, states = self.bilstm(features)  # (bsz, sample_numb, hidden_dim)
        video = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        video_features = video  # (bsz, hidden_dim)
        # video_semantics = self.fc_layer(video)  # (bsz, semantics_dim)

        # feats_list = []
        # if attn_video is not None:
        #     feats_list.append(attn_video)
        # if attn_objects is not None:
        #     feats_list.append(attn_objects)
        # visual_feats = torch.cat(feats_list, dim=-1)
        video_features = self.linear_visual_layer(video_features) if hasattr(self, 'linear_visual_layer') else video_features
        # for semantic features
        # semantics_list = []
        # attn_weights = self.wos(torch.tanh(U_objs + self.bos))
        # attn_weights = attn_weights.softmax(dim=1)  # (bsz, max_objects, 1)
        # attn_objs = attn_weights * object_semantics  # (bsz, max_objects, emb_dim)
        # attn_objs = attn_objs.sum(dim=1)  # (bsz, emb_dim)
        # semantics_list.append(attn_objs)
        #
        # semantics_list.append(video_semantics)
        # semantics_feats = torch.cat(semantics_list, dim=-1) if len(semantics_list) > 0 else None
        # video_semantics = self.linear_semantics_layer(video_semantics) if video_semantics is not None else None
        return video_features

    def forward(self, visual_feats, subject_feats, predict_feats, sentence_feats, input_mask,TorS):
        visual_feats = self.video_embeddings(visual_feats)
        # visual_semantics = self.fc_layer(torch.max(visual_feats, dim=1)[0])
        if subject_feats is not None:
            subject_feats = self.word_embeddings(subject_feats)
            object_hidden_states = self.object_embeddings(subject_feats)
        else:
            object_hidden_states = self.object_embeddings(visual_feats)
            object_semantics = self.fc_layer(object_hidden_states)

        '''---subject---'''
        input_mask = input_mask[:, :object_hidden_states.shape[1]]
        input_mask = input_mask.flatten(1)
        if self.use_SFT and TorS=='T':##Trans/lstm+ham
            if self.use_module=='transformer':
                content_vectors = torch.max(visual_feats, dim=1)[0]  # (bsz, hidden_dim)
                feat_2d_tgt = content_vectors[None, ...].repeat(object_hidden_states.shape[1], 1, 1)
                memory = object_hidden_states.permute(1, 0, 2)
                for i in range(6):
                    memory = self.encoder(memory, input_mask)
                    action_features = self.decoder(feat_2d_tgt, memory, memory, input_mask)

                action_features = action_features.permute(1, 0, 2)
                action_semantics = self.fc_layer(action_features)
            elif self.use_module == 'lstm':
                action_features, _ = self.lstm(object_hidden_states)  # (bsz, sample_numb, hidden_dim)
                action_semantics = self.fc_layer(action_features)  # (bsz, sample_numb, semantics_dim)
        else:
            action_features = object_hidden_states
            action_semantics = self.fc_layer(action_features)

        '''
        1. when teacher use ham,input is global_features,which is after trans 
        2. when student use ham ,input is global_features,which is equal to object_hidden_states
        '''
        if self.use_ham:
            video_features = self.ham(action_features, visual_feats)
        else:
            video_features = torch.max(action_features, dim=1)
            # video_semantics = torch.max(action_semantics, dim=1)
        return video_features, object_semantics, action_semantics

class CaptionDecoder(nn.Module):
    def __init__(
            self,
            word_embedding_size, visual_feature_size,
            pretrained_embedding,
            max_v_len, max_t_len, hidden_size,
            vocab_size,
    ):
        super(CaptionDecoder, self).__init__()

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

    def forward(self, feature2ds, context_feats, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"
        context_feats = context_feats.unsqueeze(1)  # (bsz,visual_dim)->(bsz,1,visual_dim)
        # context_semantics_feats = context_semantics_feats.unsqueeze(1)  # (bsz,visual_dim)->(bsz,1,visual_dim)

        input_types = torch.cat(
            [
                torch.full((feature2ds.size(0), feature2ds.size(1)),
                           fill_value=1, dtype=torch.long, device=feature2ds.device),
                torch.full((context_feats.size(0), context_feats.size(1)),
                           fill_value=0, dtype=torch.long, device=context_feats.device),
                torch.full((input_ids.size(0), input_ids.size(1)),
                           fill_value=2, dtype=torch.long, device=input_ids.device)
            ], dim=1
        )
        visual_output = torch.cat([feature2ds, context_feats], dim=1)
        input_mask = torch.cat(
            [
                torch.ones(size=(visual_output.size(0), visual_output.size(1)),
                           dtype=torch.long, device=visual_output.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_output, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        return prediction_scores

    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionDecoder.ids2text(predict_ids)

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

class SFTNET(CaptionModule):
    def __init__(self, cfgs):
        super().__init__()
        """
        1.处理数据
        """
        self.Use_SFT = cfgs.SFT
        if self.Use_SFT:
            self.sft_num = cfgs.sft_num
        # self.query_embed = nn.Embedding(cfgs.sample_numb, cfgs.decoder.hidden_dim)
        '''2.字幕头'''

        # 获取configs（clip、bert）
        clip_weights = cfgs.data.clip_weights
        pretrained_model, _ = clip.load(name=clip_weights)
        state_dict = pretrained_model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]  #768
        vocab_size = state_dict["token_embedding.weight"].shape[0] #49408
        transformer_width = state_dict["ln_final.weight"].shape[0] #768
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        # self.visual_layer = nn.Linear(1536, embed_dim)
        self.caption_head = CaptionDecoder(word_embedding_size=transformer_width,  # 768
                                           visual_feature_size=embed_dim,  # 768
                                           pretrained_embedding=pretrained_embedding,
                                           max_v_len=cfgs.sample_numb + 1,  #17
                                           max_t_len=cfgs.test.max_caption_len, #22
                                           hidden_size=embed_dim,  # 768
                                           vocab_size=vocab_size,  # 49408
                                           )
        if self.Use_SFT:#Tnet and Snet is shared,so need to put them in similiar part
            '''teacher use transformer and ham'''
            self.SFTNet = TAndS(cfgs                                                                                                                                                         .encoder.visual_dim, cfgs.decoder.hidden_dim, cfgs.encoder.semantics_dim,
                                    cfgs.encoder.max_objects, vocab_size=vocab_size, use_SFT=self.Use_SFT, use_module=cfgs.use_module,
                                    use_ham=cfgs.use_ham)
        elif cfgs.Only_T_or_S=='T':# Tnet and Snet is not shared,so need two part
            '''teacher use transformer and ham'''
            self.SNet = TAndS(cfgs.encoder.visual_dim, cfgs.decoder.hidden_dim, embed_dim,
                                    cfgs.encoder.max_objects, vocab_size=vocab_size,
                                    use_SFT=self.Use_SFT,
                                    use_module=cfgs.use_module,
                                    use_ham=cfgs.use_ham)
            '''student no transformer but have ham'''
        else:
            self.SNet = TAndS(cfgs.encoder.visual_dim, cfgs.decoder.hidden_dim, embed_dim,
                                    cfgs.encoder.max_objects, vocab_size=vocab_size, use_SFT=self.Use_SFT, use_module=None,
                                    use_ham=cfgs.use_ham)
    def forward(self, objects_feats, objects_mask, feature2ds, caption_ids, caption_mask, T_caption_ids=None, T_caption_mask=None):
        device = feature2ds.device
        mask = objects_mask.to(device).bool()
        # feature2ds = self.visual_layer(feature2ds)
        if self.Use_SFT:
            '''teachers'''
            T_global_features, T_object_feats, T_action_feats = self.SFTNet(
                visual_feats=feature2ds,
                subject_feats=None,
                # query_embed=self.query_embed.weight,
                predict_feats=None,
                sentence_feats=None,
                input_mask=mask,
                TorS = 'T'
            )
            '''students'''
            S_global_features, S_object_feats, S_action_feats = self.SFTNet(
                visual_feats=feature2ds,
                subject_feats=None,
                # query_embed=self.query_embed.weight,
                predict_feats=None,
                sentence_feats=None,
                input_mask=mask,
                TorS = 'S'
            )
            T_prediction_scores = self.caption_head(
                feature2ds,
                T_global_features,
                T_caption_ids if T_caption_ids is not None else caption_ids,
                T_caption_mask if T_caption_mask is not None else caption_mask,
            )
            S_prediction_scores = self.caption_head(
                feature2ds,
                S_global_features,
                caption_ids,
                caption_mask,
            )
            T_global_prediction_scores = self.caption_head.prediction_head(T_global_features)
            S_global_prediction_scores = self.caption_head.prediction_head(S_global_features)
            O_prediction_scores = self.caption_head.prediction_head(S_object_feats)
            A_prediction_scores = self.caption_head.prediction_head(S_action_feats)

            return {"T_prediction_scores": T_prediction_scores,
                    "S_prediction_scores": S_prediction_scores,
                    "O_prediction_scores": O_prediction_scores,
                    "A_prediction_scores": A_prediction_scores,
                    "T_global_prediction_scores": T_global_prediction_scores,
                    "S_global_prediction_scores": S_global_prediction_scores}
        else:
            S_global_features, S_object_feats, S_action_feats = self.SNet(
                visual_feats=feature2ds,
                subject_feats=None,
                # query_embed=self.query_embed.weight,
                predict_feats=None,
                sentence_feats=None,
                input_mask=mask,
                TorS='S'
            )
            S_prediction_scores = self.caption_head(
                feature2ds,
                S_global_features,
                caption_ids,
                caption_mask,
            )
            O_prediction_scores = self.caption_head.prediction_head(S_object_feats)
            A_prediction_scores = self.caption_head.prediction_head(S_action_feats)
            return {"prediction_scores": S_prediction_scores,
                    "O_prediction_scores": O_prediction_scores,
                    "A_prediction_scores": A_prediction_scores,
                    }




