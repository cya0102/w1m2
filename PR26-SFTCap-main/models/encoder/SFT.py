# SFT
import math

import torch
from torch import nn

class SFT(nn.Module):
    def __init__(self, visual_dim, hidden_dim, semantics_dim, max_objects, vocab_size, dropout=0.1, nhead=8, use_subject=False, use_predict=False, use_sentence=False):
        super(SFT, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, visual_dim, padding_idx=0)  # 词嵌入
        self.word_fc = nn.Sequential(  # 300->768
            LayerNorm(visual_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(True),
            LayerNorm(hidden_dim, eps=1e-12),
        )
        self.video_embeddings = nn.Sequential(
            LayerNorm(visual_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(True),
            LayerNorm(hidden_dim, eps=1e-12),
        )
        self.position_embeddings = PositionEncoding(n_filters=hidden_dim, max_len=1000)
        self.token_type_embeddings = nn.Embedding(3, hidden_dim) #3->跟你输入的特征形式有关,有几种形式填数字几
        self.LayerNorm = LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        '''
        定义一些自己的东西
        1.self_att
        2.corss_att
        3.可学习参数
        '''
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.intermediate_act_fn = gelu
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        """
        1.subject
        2.predict
        3.sentence
        """
        self.use_subject = use_subject
        self.max_objects = max_objects
        self.use_predict = use_predict
        self.use_sentence = use_sentence
        # predict
        if self.use_predict:
            self.W = nn.Linear(hidden_dim, hidden_dim)
            self.U = nn.Linear(hidden_dim, hidden_dim)
            self.b = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
            self.w = nn.Linear(hidden_dim, 1)
            self.bilstm = nn.LSTM(input_size=hidden_dim + hidden_dim,
                                  hidden_size=hidden_dim // 2,
                                  num_layers=1, bidirectional=True, batch_first=True)
        # sentence
        # if self.use_sentence:
        self.Ws = nn.Linear(hidden_dim, hidden_dim)

        self.Uo = nn.Linear(hidden_dim, hidden_dim)
        self.Um = nn.Linear(hidden_dim, hidden_dim)

        self.bo = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.bm = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

        self.wo = nn.Linear(hidden_dim, 1)
        self.wm = nn.Linear(hidden_dim, 1)

        self.lstm = nn.LSTM(input_size=hidden_dim + hidden_dim + hidden_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

    def predict(self,visual_feats,subject_feats):
        Wf3d = self.W(visual_feats)  # (bsz, sample_numb, hidden_dim)
        Uobjs = self.U(subject_feats)  # (bsz, max_objects, hidden_dim)

        attn_feat = Wf3d.unsqueeze(2) + Uobjs.unsqueeze(1) + self.b  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.w(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([visual_feats, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 2)
        output, states = self.bilstm(features)  # (bsz, sample_numb, hidden_dim)
        action = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        action_features = output  # (bsz, sample_numb, hidden_dim)
        action_semantics = self.fc_layer(action)  # (bsz, semantics_dim)
        return action_features,action_semantics

    def sentence(self,visual_feats, subject_feats, predict_feats,part=None):
        W_f2d = self.Ws(visual_feats)
        U_objs = self.Uo(subject_feats)
        U_motion = self.Um(predict_feats)

        attn_feat = W_f2d.unsqueeze(2) + U_objs.unsqueeze(1) + self.bo  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.wo(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        attn_feat = W_f2d.unsqueeze(2) + U_motion.unsqueeze(1) + self.bm  # (bsz, sample_numb, sample_numb, hidden_dim)
        attn_weights = self.wm(torch.tanh(attn_feat))  # (bsz, sample_numb, sample_numb, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, sample_numb, 1)
        attn_motion = attn_weights * attn_feat
        attn_motion = attn_motion.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([visual_feats, attn_motion, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 3)
        output, states = self.lstm(features)  # (bsz, sample_numb, hidden_dim)
        video = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        video_features = output  # (bsz, sample_numb, hidden_dim)
        video_semantics = self.fc_layer(video)  # (bsz, semantics_dim)
        return video_features, video_semantics
        # else:
        #     salient_subjects = attn_objects
        #     object_semantics = self.fc_layer(salient_subjects)
        #     action_features = attn_motion
        #     action = torch.max(action_features, dim=1)[0]
        #     action_semantics = self.fc_layer(action)
        #     return salient_subjects, action_features, video_features, object_semantics, action_semantics, video_semantics

    def forward(self, visual_feats, subject_feats, query_embed, predict_feats, sentence_feats, input_mask):
        visual_feats = self.video_embeddings(visual_feats)

        if subject_feats is not None:
            subject_hidden_states = torch.cat((visual_feats, subject_feats), dim=1)
        else:
            subject_hidden_states = visual_feats
        if predict_feats is not None:
            predict_hidden_states = torch.cat((visual_feats, predict_feats), dim=1)
        else:
            predict_hidden_states = visual_feats
        if sentence_feats is not None:
            sentence_hidden_states = torch.cat((visual_feats, sentence_feats), dim=1)
        else:
            sentence_hidden_states = visual_feats

        subject_hidden_states = self.position_embeddings(subject_hidden_states)
        subject_hidden_states = self.LayerNorm(subject_hidden_states)
        subject_hidden_states = self.dropout(subject_hidden_states)

        predict_hidden_states = self.position_embeddings(predict_hidden_states)
        predict_hidden_states = self.LayerNorm(predict_hidden_states)
        predict_hidden_states = self.dropout(predict_hidden_states)

        sentence_hidden_states = self.position_embeddings(sentence_hidden_states)
        sentence_hidden_states = self.LayerNorm(sentence_hidden_states)
        sentence_hidden_states = self.dropout(sentence_hidden_states)

        '''---subject---'''
        memory = subject_hidden_states
        query_embed = query_embed.unsqueeze(1).repeat(1, memory.shape[0], 1)
        memory = memory.permute(1, 0, 2)
        input_mask = input_mask[:,:memory.shape[0]]
        input_mask = input_mask.flatten(1)
        memory = self.self_attn(memory, memory, value=memory, attn_mask=None,
                              key_padding_mask=input_mask)[0]
        memory1 = self.dense(memory)
        memory1 = self.dropout(memory1)
        memory_output = self.LayerNorm(memory1+memory)
        # intermediate
        intermediate_memory_output = self.dense(memory_output)
        intermediate_memory_output = self.intermediate_act_fn(intermediate_memory_output)
        #output
        intermediate_memory_output = self.dense(intermediate_memory_output)
        intermediate_memory_output = self.dropout(intermediate_memory_output)
        memory_hidden_states = self.LayerNorm(intermediate_memory_output + memory_output)
        if self.use_subject:
            '''
            1.self_att:key+value(+query_embed)
            2.multi_att:query
            '''
            content_vectors = torch.max(subject_hidden_states, dim=1)[0]  # (bsz, hidden_dim)
            feat_2d_tgt = content_vectors[None, ...].repeat(self.max_objects, 1, 1)
            tgt = self.self_attn(feat_2d_tgt+query_embed, feat_2d_tgt+query_embed, value=feat_2d_tgt, attn_mask=None,
                                  key_padding_mask=None)[0]
            tgt = self.dense(tgt)
            tgt = self.dropout(tgt)
            tgt = self.LayerNorm(tgt + feat_2d_tgt)
            tgt1 = self.multihead_attn(query=tgt+query_embed,
                                       key=memory_hidden_states,
                                       value=memory_hidden_states, attn_mask=None,
                                       key_padding_mask=input_mask)[0]
            tgt1 = self.dense(tgt1)
            tgt1 = self.dropout(tgt1)
            salient_subjects = self.LayerNorm(tgt + tgt1)
            salient_subjects = salient_subjects.permute(1, 0, 2)
            object_semantics = self.fc_layer(salient_subjects)
        else:
            salient_subjects = subject_hidden_states
            object_semantics = self.fc_layer(salient_subjects)
        if self.use_predict:
            action_features, action_semantics = self.predict(predict_hidden_states,salient_subjects)
        else:
            action = torch.max(predict_hidden_states, dim=1)[0]  # (bsz, hidden_dim)
            action_features = predict_hidden_states  # (bsz, sample_numb, hidden_dim)
            action_semantics = self.fc_layer(action)  # (bsz, semantics_dim)
            # action_features, action_semantics = self.predict(predict_hidden_states, salient_subjects)
        # if self.use_sentence:
        #     video_features, video_semantics = self.sentence(sentence_hidden_states, salient_subjects, action_features)
        # else:
        # if subject_feats is None and predict_feats is None and sentence_feats is None:
        #     salient_subjects, action_features, video_features, object_semantics, action_semantics, video_semantics = self.sentence(sentence_hidden_states, salient_subjects, action_features,'T')
        video_features, video_semantics = self.sentence(sentence_hidden_states, salient_subjects, action_features)
        return [salient_subjects, action_features, video_features,object_semantics, action_semantics,video_semantics]

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PositionEncoding(nn.Module):
    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
