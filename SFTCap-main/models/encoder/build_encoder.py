import torch
from torch import nn, Tensor

class Encoder_layer(nn.Module):
    def __init__(self, transformer, max_objects, visual_dim, object_dim, hidden_dim, semantics_dim,embed_dim):
        super(Encoder_layer, self).__init__()
        total_visual_dim = 0
        self.entity_level = EntityLevelEncoder(transformer=transformer,
                                                 max_objects=max_objects,
                                                 object_dim=object_dim,
                                                 visual_dim=visual_dim,
                                                 hidden_dim=hidden_dim,
                                                semantics_dim = semantics_dim
                                                 )
        self.predicate_level = PredicateLevelEncoder(visual_dim=visual_dim,
                                                      hidden_dim=hidden_dim,
                                                     semantics_dim = semantics_dim)
        self.sentence_level = SentenceLevelEncoder(visual_dim=visual_dim,
                                                      hidden_dim=hidden_dim,
                                                   semantics_dim = semantics_dim)
        self.fusion_level = FusionLevelEncoder(hidden_dim = hidden_dim,
                                               semantics_dim = semantics_dim,
                                               visual_dim=visual_dim,)

    def forward(self, objects, objects_mask, feature2ds,teacher_action=None):
        objects_feats, objects_semantics= self.entity_level(feature2ds, objects, objects_mask)
        action_feats, action_semantics= self.predicate_level(feature2ds, objects_feats)
        if teacher_action is not None:
            action_feats = torch.cat([teacher_action, action_feats], dim=1)
        video_feats, video_semantics = self.sentence_level(feature2ds, action_feats, objects_feats)
        # visual_feats,visual_semantics_feats = self.sentence_level(objects_feats, action_feats, video_feats,objects_semantics, action_semantics, video_semantics,)
        # all_visual_features = torch.cat([feature2ds, visual_feats.squeeze(1)], dim=1)
        return [objects_feats, action_feats, video_feats,objects_semantics, action_semantics,video_semantics]

class EntityLevelEncoder(nn.Module):
    def __init__(self, transformer, max_objects, object_dim, visual_dim, hidden_dim, semantics_dim):
        super(EntityLevelEncoder, self).__init__()
        self.max_objects = max_objects
        self.query_embed = nn.Embedding(max_objects, hidden_dim)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(object_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.input_proj_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.feature2d_proj = nn.Sequential(
            nn.Linear(visual_dim, 2 * hidden_dim),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2,
                              batch_first=True, bidirectional=True)
        self.transformer = transformer
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)


    def forward(self, visual: Tensor, objects: Tensor, objects_mask: Tensor):
        device = objects.device
        bsz, sample_numb, max_objects_per_video = visual.shape[0], visual.shape[1], objects.shape[1]

        features_2d = self.feature2d_proj(visual.view(-1, visual.shape[-1]))
        features_2d = features_2d.view(bsz, sample_numb, -1).contiguous()
        content_vectors, _ = self.bilstm(features_2d)

        content_vectors = torch.max(content_vectors, dim=1)[0]  # (bsz, hidden_dim)
        tgt = content_vectors[None, ...].repeat(self.max_objects, 1, 1)  # (max_objects, bsz, hidden_dim)
        objects_channel = objects.shape[2]
        if objects_channel == self.hidden_dim:
            objects = self.input_proj_s(objects.reshape(-1, objects.shape[-1]))
            objects = objects.view(bsz, max_objects_per_video,
                                   -1).contiguous()  # (bsz, max_objects_per_video, hidden_dim)
            mask = objects_mask.to(device).bool()
        else:
            objects = self.input_proj(objects.view(-1, objects.shape[-1]))
            objects = objects.view(bsz, max_objects_per_video, -1).contiguous()
            mask = objects_mask.to(device).bool()  # (bsz, max_objects_per_video)
        salient_objects = self.transformer(objects, tgt, mask[:, :max_objects_per_video], self.query_embed.weight)[0][0]  # (bsz, max_objects, hidden_dim)
        object_semantics = self.fc_layer(salient_objects)  # (bsz, max_objects, semantics_dim)
        return salient_objects,object_semantics

class PredicateLevelEncoder(nn.Module):
    def __init__(self, visual_dim, hidden_dim, semantics_dim):
        super(PredicateLevelEncoder, self).__init__()
        self.linear_layer = nn.Linear(visual_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.b = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.w = nn.Linear(hidden_dim, 1)
        self.inf = 1e5

        self.bilstm = nn.LSTM(input_size=hidden_dim + hidden_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)
    def forward(self, visual: Tensor, objects: Tensor):
        features3d = self.linear_layer(visual)  # (bsz, sample_numb, hidden_dim)
        Wf3d = self.W(features3d)  # (bsz, sample_numb, hidden_dim)
        Uobjs = self.U(objects)  # (bsz, max_objects, hidden_dim)

        attn_feat = Wf3d.unsqueeze(2) + Uobjs.unsqueeze(1) + self.b  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.w(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([features3d, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 2)
        output, states = self.bilstm(features)  # (bsz, sample_numb, hidden_dim)
        action = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        action_features = output  # (bsz, sample_numb, hidden_dim)
        action_semantics = self.fc_layer(action)  # (bsz, semantics_dim)

        return action_features,action_semantics

class SentenceLevelEncoder(nn.Module):
    def __init__(self, visual_dim, hidden_dim, semantics_dim):
        super(SentenceLevelEncoder, self).__init__()
        self.inf = 1e5
        self.linear_2d = nn.Linear(visual_dim, hidden_dim)

        self.W = nn.Linear(hidden_dim, hidden_dim)

        self.Uo = nn.Linear(hidden_dim, hidden_dim)
        self.Um = nn.Linear(hidden_dim, hidden_dim)

        self.bo = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.bm = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

        self.wo = nn.Linear(hidden_dim, 1)
        self.wm = nn.Linear(hidden_dim, 1)

        self.lstm = nn.LSTM(input_size=hidden_dim + hidden_dim + hidden_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)
    def forward(self, visual: Tensor, vp_features: Tensor, object_features: Tensor):
        feature2ds = self.linear_2d(visual)
        W_f2d = self.W(feature2ds)
        U_objs = self.Uo(object_features)
        U_motion = self.Um(vp_features)

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

        features = torch.cat([feature2ds, attn_motion, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 3)
        output, states = self.lstm(features)  # (bsz, sample_numb, hidden_dim)
        video = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        video_features = output  # (bsz, sample_numb, hidden_dim)
        video_semantics = self.fc_layer(video)  # (bsz, semantics_dim)
        return video_features,video_semantics

class FusionLevelEncoder(nn.Module):
    def __init__(self, hidden_dim, semantics_dim,visual_dim,Use_SFT):
        super(FusionLevelEncoder, self).__init__()
        self.Use_SFT = Use_SFT
        total_visual_dim = 0
        total_semantics_dim = 0
        setattr(self, 'Uo', nn.Linear(hidden_dim, hidden_dim, bias=False))
        setattr(self, 'bo', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
        setattr(self, 'wo', nn.Linear(hidden_dim, 1, bias=False))
        total_visual_dim += hidden_dim
        setattr(self, 'Uos', nn.Linear(semantics_dim, hidden_dim, bias=False))
        setattr(self, 'bos', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
        setattr(self, 'wos', nn.Linear(hidden_dim, 1, bias=False))
        total_semantics_dim += semantics_dim
        setattr(self, 'Um', nn.Linear(hidden_dim, hidden_dim, bias=False))
        setattr(self, 'bm', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
        setattr(self, 'wm', nn.Linear(hidden_dim, 1, bias=False))
        total_visual_dim += hidden_dim
        total_semantics_dim += semantics_dim
        setattr(self, 'Uv', nn.Linear(hidden_dim, hidden_dim, bias=False))
        setattr(self, 'bv', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
        setattr(self, 'wv', nn.Linear(hidden_dim, 1, bias=False))
        total_visual_dim += hidden_dim
        total_semantics_dim += semantics_dim
        '''如果跑lstm，total_visual_dim, hidden_dim'''
        '''如果跑bert，total_visual_dim, visual_dim'''
        setattr(self, 'linear_visual_layer', nn.Linear(total_visual_dim, visual_dim))
        setattr(self, 'linear_semantics_layer', nn.Linear(total_semantics_dim, visual_dim))

    def forward(self, objects_feats, action_feats, video_feats, object_semantics, action_semantics, video_semantics):
        if self.Use_SFT:
            # 合并特征
            U_obj = self.Uo(objects_feats) if hasattr(self, 'Uo') else None
            U_objs = self.Uos(object_semantics) if hasattr(self, 'Uos') else None
            U_action = self.Um(action_feats) if hasattr(self, 'Um') else None  # (bsz, sample_numb, hidden_dim)
            U_video = self.Uv(video_feats) if hasattr(self, 'Uv') else None

            attn_weights = self.wo(torch.tanh(U_obj + self.bo))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, max_objects, 1)
            attn_objects = attn_weights * objects_feats  # (bsz, max_objects, hidden_dim)
            attn_objects = attn_objects.sum(dim=1)  # (bsz, hidden_dim)

            attn_weights = self.wm(torch.tanh(U_action + self.bm))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, sample_numb, 1)
            attn_motion = attn_weights * action_feats  # (bsz, sample_numb, hidden_dim)
            attn_motion = attn_motion.sum(dim=1)  # (bsz, hidden_dim)

            attn_weights = self.wv(torch.tanh(U_video + self.bv))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, sample_numb, 1)
            attn_video = attn_weights * video_feats  # (bsz, sample_numb, hidden_dim)
            attn_video = attn_video.sum(dim=1)  # (bsz, hidden_dim)

            feats_list = []
            if attn_video is not None:
                feats_list.append(attn_video)
            if attn_motion is not None:
                feats_list.append(attn_motion)
            if attn_objects is not None:
                feats_list.append(attn_objects)
            visual_feats = torch.cat(feats_list, dim=-1)
            visual_feats = self.linear_visual_layer(visual_feats) if hasattr(self, 'linear_visual_layer') else visual_feats
            # for semantic features
            semantics_list = []
            attn_weights = self.wos(torch.tanh(U_objs + self.bos))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, max_objects, 1)
            attn_objs = attn_weights * object_semantics  # (bsz, max_objects, emb_dim)
            attn_objs = attn_objs.sum(dim=1)  # (bsz, emb_dim)
            semantics_list.append(attn_objs)

            semantics_list.append(action_semantics)
            semantics_list.append(video_semantics)
            semantics_feats = torch.cat(semantics_list, dim=-1) if len(semantics_list) > 0 else None
            semantics_feats = self.linear_semantics_layer(semantics_feats) if semantics_feats is not None else None
        else:
            attn_objects = objects_feats.sum(dim=1)  # (bsz, hidden_dim)
            attn_motion = action_feats.sum(dim=1)  # (bsz, hidden_dim)
            attn_video = video_feats.sum(dim=1)  # (bsz, hidden_dim)
            feats_list = []
            if attn_video is not None:
                feats_list.append(attn_video)
            if attn_motion is not None:
                feats_list.append(attn_motion)
            if attn_objects is not None:
                feats_list.append(attn_objects)
            visual_feats = torch.cat(feats_list, dim=-1)
            visual_feats = self.linear_visual_layer(visual_feats) if hasattr(self,
                                                                             'linear_visual_layer') else visual_feats
            # for semantic features
            semantics_list = []
            attn_objs = object_semantics.sum(dim=1)  # (bsz, emb_dim)
            semantics_list.append(attn_objs)
            semantics_list.append(action_semantics)
            semantics_list.append(video_semantics)
            semantics_feats = torch.cat(semantics_list, dim=-1) if len(semantics_list) > 0 else None
            semantics_feats = self.linear_semantics_layer(semantics_feats) if semantics_feats is not None else None
        return visual_feats, semantics_feats