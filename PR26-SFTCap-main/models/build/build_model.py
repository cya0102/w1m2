import logging
import math
import pickle
import random

import torch
from nltk import word_tokenize, pos_tag
from torch import nn

from models.decoder.caption_decoder import CaptionHead
from models.decoder.caption_module import CaptionModule
from models.decoder.decoder import Decoder
from models.encoder.SFT import SFT
from models.encoder.build_encoder import Encoder_layer, FusionLevelEncoder
from models.encoder.transformer import Transformer
from models.layers.clip import clip
from models.model import SFTNET

logger = logging.getLogger(__name__)
def build_model(config, pretrained=''):
    if config.data.train_type == 'preprocess':
        model = SFT4Cap(config)
        # if config.train.save_checkpoints_path:
        #     model.load_state_dict(torch.load(config.train.save_checkpoints_path))
    elif config.data.train_type == 'e2e':
        model = SFT4CapE2E(config)
    elif config.data.train_type == 'sft':
        model = SFTNET(config)
    else:
        model = SFT4Cap4Clip(config)
    return model

class Fusion_Decoder(nn.Module):
    def __init__(self, hidden_dim, semantics_dim,num_layers, embed_dim, n_vocab):
        super(Fusion_Decoder, self).__init__()
        self.n_vocab = n_vocab
        self.num_layers = num_layers
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        setattr(self, 'Ul', nn.Linear(hidden_dim, hidden_dim, bias=False))
        setattr(self, 'bl', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
        setattr(self, 'wl', nn.Linear(hidden_dim, 1, bias=False))
        self.lstm = nn.LSTM(input_size=hidden_dim * 2 + embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers)
        self.to_word = nn.Linear(hidden_dim, embed_dim)
        self.logit = nn.Linear(embed_dim, n_vocab)
        self.__init_weight()
    def __init_weight(self):
        init_range = 0.1
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-init_range, init_range)
    def forward(self, all_visual_features, semantics_feats, embed,last_states):
        last_hidden = last_states[0][0]  # (bsz, hidden_dim)
        Wh = self.W(last_hidden)  # (bsz, hidden_dim)
        U_all = self.Ul(all_visual_features) if hasattr(self, 'Ul') else None
        attn_weights = self.wl(torch.tanh(Wh[:, None, :] + U_all + self.bl))
        attn_weights = attn_weights.softmax(dim=1)  # (bsz, max_objects, 1)
        attn_visual = attn_weights * all_visual_features  # (bsz, max_objects, hidden_dim)
        attn_visual = attn_visual.sum(dim=1)  # (bsz, hidden_dim)
        input_feats = torch.cat([attn_visual, semantics_feats,embed], dim=-1)  # (bsz, hidden_dim + embed_dim)
        output, states = self.lstm(input_feats[None, ...], last_states)
        output = output.squeeze(0)  # (bsz, hidden_dim)
        output = self.to_word(output)  # (bsz, embed_dim)
        output_prob = self.logit(output)  # (bsz, n_vocab)
        output_prob = torch.log_softmax(output_prob, dim=1)  # (bsz, n_vocab)
        return output_prob, states

class SFT4Cap(CaptionModule):
    def __init__(self, cfgs):
        super().__init__()

        self.SFT = cfgs.SFT
        transformer = Transformer(d_model=cfgs.encoder.d_model, nhead=cfgs.encoder.nheads,
                                  num_encoder_layers=cfgs.encoder.entity_encoder_layer,
                                  num_decoder_layers=cfgs.encoder.entity_encoder_layer,
                                  dim_feedforward=cfgs.encoder.dim_feedforward,
                                  dropout=cfgs.encoder.trans_dropout,
                                  activation=cfgs.encoder.transformer_activation)
        self.Encoder_layer = Encoder_layer(transformer= transformer,
                                           max_objects= cfgs.encoder.max_objects,
                                           visual_dim = cfgs.encoder.visual_dim,
                                           object_dim = cfgs.encoder.object_dim,
                                           hidden_dim = cfgs.decoder.hidden_dim,
                                           semantics_dim=cfgs.encoder.semantics_dim,
                                           embed_dim=cfgs.encoder.word_dim)
        '''2.字幕头'''
        # 获取configs（clip、bert）
        with open(cfgs.data.embedding_weights_path, 'rb') as f:
            embedding_weights = pickle.load(f)
        self.pad_idx = cfgs.dict.eos_idx
        self.eos_idx = cfgs.dict.eos_idx
        self.sos_idx = cfgs.dict.sos_idx
        self.unk_idx = cfgs.dict.unk_idx
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights),
                                                      freeze=False, padding_idx=self.pad_idx)
        self.n_vocab = embedding_weights.shape[0]
        self.max_caption_len = cfgs.test.max_caption_len
        self.beam_size = cfgs.test.beam_size
        self.temperature = cfgs.test.temperature
        self.num_layers=cfgs.decoder.num_layers
        self.caption_head = Fusion_Decoder(hidden_dim = cfgs.decoder.hidden_dim,
                                    semantics_dim = cfgs.encoder.semantics_dim,
                                    num_layers = cfgs.decoder.num_layers,
                                    embed_dim = cfgs.encoder.word_dim,
                                    n_vocab = embedding_weights.shape[0])
        self.decoder = Decoder(hidden_dim=cfgs.decoder.hidden_dim,
                                           semantics_dim=cfgs.encoder.semantics_dim,
                                           num_layers=cfgs.decoder.num_layers,
                                           embed_dim=cfgs.encoder.word_dim,
                                           n_vocab=embedding_weights.shape[0])

    def get_rnn_init_hidden(self, bsz, hidden_size, device):
        # (hidden_state, cell_state)
        return (torch.zeros(self.num_layers, bsz, hidden_size).to(device),
                torch.zeros(self.num_layers, bsz, hidden_size).to(device))

    def gen_req(self, bsz, device, n_vocab,caption_ids,objects_feats, action_feats, video_feats,
                                                      objects_semantics, action_semantics,
                                                      video_semantics,):
        state = self.get_rnn_init_hidden(bsz=bsz, hidden_size=self.decoder.hidden_dim, device=device)
        outputs = []

        for i in range(self.max_caption_len):
            if i > 0 and caption_ids[:, i].sum() == 0:
                output_word = torch.zeros([bsz, n_vocab]).cuda()
                outputs.append(output_word)
                continue

            it = caption_ids[:, i].clone()
            it_embeded = self.embedding(it)
            output_word, state = self.decoder(objects_feats, action_feats, video_feats,
                                                      objects_semantics, action_semantics,
                                                      video_semantics, it_embeded, state)
            outputs.append(output_word)

        ret_seq = torch.stack(outputs, dim=1)

        return ret_seq

    def forward(self, objects_feats, objects_mask, feature2ds, caption_ids):
        bsz = feature2ds.shape[0]
        device = objects_feats.device
        sft_num=3
        if self.SFT:
            split_num = sft_num-1
            selected_teachers=[int(i+(feature2ds.shape[1]/split_num)) for i in range(sft_num-2)]
            selected_teachers = [0]+selected_teachers+[feature2ds.shape[1]-1]
            # selected_teachers = sorted(random.sample(range(feature2ds.shape[1]), sft_num))
            feature2ds_T = feature2ds[:, selected_teachers, :]
            T_visual_features = self.Encoder_layer(
                objects=objects_feats,
                objects_mask=objects_mask,
                feature2ds=feature2ds_T
            )
            T_prediction_scores = self.gen_req( bsz, device, self.n_vocab,caption_ids,
                                                T_visual_features[0], T_visual_features[1], T_visual_features[2],
                                                      T_visual_features[3], T_visual_features[4],
                                                      T_visual_features[5],)
            selected_students1 = [i for i in range(1,selected_teachers[0]+5)]
            # selected_students1=sorted(random.sample(student_list, 3))
            feature2ds_S1 = feature2ds[:, selected_students1, :]
            S_visual_features = self.Encoder_layer(
                objects=T_visual_features[0],
                objects_mask=objects_mask,
                feature2ds=feature2ds_S1,
                teacher_action=T_visual_features[1]
            )
            S_prediction_scores = self.gen_req(bsz, device, self.n_vocab, caption_ids,
                                               S_visual_features[0], S_visual_features[1], S_visual_features[2],
                                               S_visual_features[3], S_visual_features[4],
                                               S_visual_features[5], )
            return T_prediction_scores,T_visual_features[3], T_visual_features[4],T_visual_features[5], \
                   S_prediction_scores,S_visual_features[3], S_visual_features[4],S_visual_features[5]
        else:
            all_visual_features= self.Encoder_layer(
                objects = objects_feats,
                objects_mask = objects_mask,
                feature2ds = feature2ds
            )
            prediction_scores = self.gen_req(bsz, device, self.n_vocab, caption_ids,
                                               all_visual_features[0], all_visual_features[1], all_visual_features[2],
                                               all_visual_features[3], all_visual_features[4],
                                               all_visual_features[5], )

            return prediction_scores,all_visual_features[3], all_visual_features[4], all_visual_features[5]

class SFT4Cap4Clip(CaptionModule):
    def __init__(self, cfgs):
        super().__init__()
        """
        1.处理数据
        1.1Track4Cap
        2.2SFT
        """
        self.SFT = cfgs.SFT
        transformer = Transformer(d_model=cfgs.encoder.d_model, nhead=cfgs.encoder.nheads,
                                  num_encoder_layers=cfgs.encoder.entity_encoder_layer,
                                  num_decoder_layers=cfgs.encoder.entity_encoder_layer,
                                  dim_feedforward=cfgs.encoder.dim_feedforward,
                                  dropout=cfgs.encoder.trans_dropout,
                                  activation=cfgs.encoder.transformer_activation)
        self.Encoder_layer = Encoder_layer(transformer= transformer,
                                           max_objects= cfgs.encoder.max_objects,
                                           visual_dim = cfgs.encoder.visual_dim,
                                           object_dim = cfgs.encoder.object_dim,
                                           hidden_dim = cfgs.decoder.hidden_dim,
                                           semantics_dim=cfgs.encoder.semantics_dim,
                                           embed_dim=cfgs.encoder.word_dim)
        '''2.字幕头'''
        # 获取configs（clip、bert）
        clip_weights = cfgs.data.clip_weights
        pretrained_model, _ = clip.load(name=clip_weights)
        state_dict = pretrained_model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        Fusion = FusionLevelEncoder(cfgs.decoder.hidden_dim, cfgs.encoder.semantics_dim, cfgs.encoder.visual_dim)

        self.caption_head = CaptionHead(word_embedding_size=transformer_width,#768
                                        visual_feature_size=embed_dim,#768
                                        pretrained_embedding=pretrained_embedding,
                                        max_v_len=cfgs.sample_numb + 2,
                                        max_t_len=77,
                                        hidden_size=embed_dim,#768
                                        vocab_size=vocab_size,#49408
                                        Fusion=Fusion)

    def forward(self, objects_feats, objects_mask, feature2ds, caption_ids,caption_mask):
        bsz = feature2ds.shape[0]
        device = objects_feats.device
        sft_num=3
        if self.SFT:
            split_num = sft_num-1
            selected_teachers=[int(i+(feature2ds.shape[1]/split_num)) for i in range(sft_num-2)]
            selected_teachers = [0]+selected_teachers+[feature2ds.shape[1]-1]
            # selected_teachers = sorted(random.sample(range(feature2ds.shape[1]), sft_num))
            feature2ds_T = feature2ds[:, selected_teachers, :]
            T_visual_features = self.Encoder_layer(
                objects=objects_feats,
                objects_mask=objects_mask,
                feature2ds=feature2ds_T
            )
            T_prediction_scores = self.caption_head(
            feature2ds,
            T_visual_features,
            caption_ids,
            caption_mask,
        )
            '''设置学生个数'''
            # set a stu_list,stulist=[stu1,stu2,....,stun]
            # stun=[i for i in range(selected_teachers[n]+1,selected_teachers[0]+1+stu`s num)]
            # stu`s num <=selected_teachers[n]-selected_teachers[n-1]
            selected_students1 = [i for i in range(1,selected_teachers[0]+5)]
            # selected_students1=sorted(random.sample(student_list, 3))
            feature2ds_S1 = feature2ds[:, selected_students1, :]
            S_visual_features = self.Encoder_layer(
                objects=T_visual_features[0],
                objects_mask=objects_mask,
                feature2ds=feature2ds_S1,
                teacher_action=T_visual_features[1]
            )
            S_prediction_scores = self.caption_head(
            feature2ds,
            S_visual_features,
            caption_ids,
            caption_mask,
        )
            return {"T_prediction_scores":T_prediction_scores,
                    "T_objects":T_visual_features[0],
                    "T_action":T_visual_features[1],
                    "T_video":T_visual_features[2],
                    "S_prediction_scores": S_prediction_scores,
                    "S_objects": S_visual_features[0],
                    "S_action": S_visual_features[1],
                    "S_video": S_visual_features[2],}
        else:
            all_visual_features= self.Encoder_layer(
                objects = objects_feats,
                objects_mask = objects_mask,
                feature2ds = feature2ds
            )
            prediction_scores = self.caption_head(
            feature2ds,
            all_visual_features,
            caption_ids,
            caption_mask,
        )

            return prediction_scores,all_visual_features[3], all_visual_features[4], all_visual_features[5]

class SFT4CapE2E(CaptionModule):
    def __init__(self, cfgs):
        super().__init__()
        """
        1.处理数据
        1.1Track4Cap
        2.2SFT
        """
        self.Use_SFT = cfgs.SFT
        if self.Use_SFT:
            self.sft_num = cfgs.sft_num
        self.query_embed = nn.Embedding(cfgs.encoder.max_objects, cfgs.decoder.hidden_dim)
        '''2.字幕头'''
        # 获取configs（clip、bert）
        clip_weights = cfgs.data.clip_weights
        pretrained_model, _ = clip.load(name=clip_weights)
        state_dict = pretrained_model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        Fusion = FusionLevelEncoder(cfgs.decoder.hidden_dim, cfgs.encoder.semantics_dim, cfgs.encoder.visual_dim, cfgs.SFT)

        self.caption_head = CaptionHead(word_embedding_size=transformer_width,#768
                                        visual_feature_size=embed_dim,#768
                                        pretrained_embedding=pretrained_embedding,
                                        max_v_len=cfgs.sample_numb + 2,
                                        max_t_len=cfgs.test.max_caption_len,
                                        hidden_size=embed_dim,#768
                                        vocab_size=vocab_size,#49408
                                        Fusion=Fusion)
        self.SFT = SFT(cfgs.encoder.visual_dim, cfgs.decoder.hidden_dim, cfgs.encoder.semantics_dim,
                       cfgs.encoder.max_objects, vocab_size=vocab_size, use_subject=cfgs.use_subject, use_predict=cfgs.use_predict, use_sentence=cfgs.use_sentence)

    def forward(self, objects_feats, objects_mask, feature2ds, caption_ids,caption_mask):
        bsz = feature2ds.shape[0]
        device = objects_feats.device
        mask = objects_mask.to(device).bool()
        if self.Use_SFT:
            sft_num = self.sft_num
            split_num = sft_num - 1
            selected_teachers = [int(i + (feature2ds.shape[1] / split_num)) for i in range(sft_num - 2)]
            selected_teachers = [0] + selected_teachers + [feature2ds.shape[1] - 1]
            # selected_teachers = sorted(random.sample(range(feature2ds.shape[1]), sft_num))
            feature2ds_T = feature2ds[:, selected_teachers, :]
            T_visual_features = self.SFT(
                visual_feats=feature2ds_T,
                subject_feats=None,
                query_embed=self.query_embed.weight,
                predict_feats=None,
                sentence_feats=None,
                input_mask=mask,
            )
            T_prediction_scores = self.caption_head(
                feature2ds,
                T_visual_features,
                caption_ids,
                caption_mask,
            )
            '''设置学生个数'''
            # set a stu_list,stulist=[stu1,stu2,....,stun]
            # stun=[i for i in range(selected_teachers[n]+1,selected_teachers[0]+1+stu`s num)]
            # stu`s num <=selected_teachers[n]-selected_teachers[n-1]
            selected_students1 = [i for i in range(1, selected_teachers[0] + 5)]
            # selected_students1=sorted(random.sample(student_list, 3))
            feature2ds_S1 = feature2ds[:, selected_students1, :]
            S_visual_features = self.SFT(
                visual_feats=feature2ds_S1,
                subject_feats=T_visual_features[0],
                query_embed=self.query_embed.weight,
                predict_feats=T_visual_features[1],
                sentence_feats=T_visual_features[2],
                input_mask=mask,
            )
            S_prediction_scores = self.caption_head(
                feature2ds,
                S_visual_features,
                caption_ids,
                caption_mask,
            )
            return {"T_prediction_scores": T_prediction_scores,
                    "T_objects": T_visual_features[0],
                    "T_action": T_visual_features[1],
                    "T_video": T_visual_features[2],
                    "S_prediction_scores": S_prediction_scores,
                    "S_objects": S_visual_features[0],
                    "S_action": S_visual_features[1],
                    "S_video": S_visual_features[2], }
        else:
            all_visual_features = self.SFT(
                visual_feats=feature2ds,
                subject_feats=None,
                query_embed=self.query_embed.weight,
                predict_feats=None,
                sentence_feats=None,
                input_mask=mask,
            )
            prediction_scores = self.caption_head(
                feature2ds,
                all_visual_features,
                caption_ids,
                caption_mask,
            )
            return {"prediction_scores": prediction_scores,
                    "objects": all_visual_features[0],
                    "action": all_visual_features[1],
                    "video": all_visual_features[2],}





