import json
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random

from configs.settings import TotalConfigs
from models.layers.clip import clip


class CaptionDataset(Dataset):
    def __init__(self, cfgs: TotalConfigs, mode):
        """1.获取文本信息"""
        self.mode = mode
        self.dataset_name = cfgs.data.dataset
        videos_split = cfgs.data.videos_split.format(mode)
        with open(videos_split, 'rb') as f:
            video_ids = pickle.load(f)
        self.video_ids = video_ids
        # 获取训练、测试ids
        """1.获取视觉特征"""
        self.visual_path = cfgs.data.visual_features
        self.objects_visual_path = cfgs.data.object_features.format(mode)
        sample_numb = cfgs.sample_numb
        self.visual_dict = {}
        self.objects_dict = {}
        # visual dict
        for vid in tqdm(self.video_ids):
            temp_feat = np.load(os.path.join(self.visual_path, vid + '.npy'))
            sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
            self.visual_dict[vid] = temp_feat[sampled_idxs]
        # feature object dict
        with h5py.File(self.objects_visual_path, 'r') as f:
            for vid in tqdm(self.video_ids):
                temp_feat = f[vid]['feats'][()]
                self.objects_dict[vid] = temp_feat
        """2.获取文本特征"""
        # msvd\msrvtt提取方式不同
        if self.dataset_name =='msvd':
            self.vid2language_features = cfgs.data.vid2language_features
        elif self.dataset_name =='msrvtt':
            self.vid2language_features = cfgs.data.vid2language_features.format(mode)
        with open(self.vid2language_features, 'rb') as f:
            self.vid2language = pickle.load(f)
        self.total_entries = []
        self.corresponding_vid = []
        self.vid2captions = defaultdict(list)
        ##TODO：这里是caption和视频的链接
        for vid in tqdm(video_ids):
            for item in self.vid2language[vid]:
                caption, caption_ids, vp_semantics, caption_semantics, nouns, nouns_vec = item
                self.total_entries.append((caption_ids, vp_semantics, caption_semantics, nouns, nouns_vec))
                self.corresponding_vid.append(vid)
                self.vid2captions[vid].append(caption)

    def __len__(self):
        if self.mode == 'train':
            return len(self.total_entries)
        else:
            return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.corresponding_vid[idx] if (self.mode == 'train') else self.video_ids[idx]
        choose_idx = 0
        feature2d = self.visual_dict[video_id]
        objects = self.objects_dict[video_id]
        captions = self.vid2captions[video_id]
        # video_id, sentence_list = self.sentences[idx]
        # sentence = random.choice(sentence_list)
        if self.mode == 'train':
            caption_ids, vp_semantics, caption_semantics, nouns, nouns_vec = self.total_entries[idx]
        else:
            caption_ids, vp_semantics, caption_semantics, nouns, nouns_vec = self.vid2language[video_id][choose_idx][1:]
        nouns_dict = {'nouns': nouns, 'vec': torch.FloatTensor(nouns_vec)}

        return torch.FloatTensor(feature2d), torch.FloatTensor(objects), \
               torch.LongTensor(caption_ids),torch.FloatTensor(caption_semantics), captions, \
               nouns_dict, torch.FloatTensor(vp_semantics), video_id



def collate_fn_caption(batch):
    feature2ds, objects, caption_ids, caption_semantics, captions, \
    nouns_dict, vp_semantics, video_id = zip(*batch)

    bsz, obj_dim = len(feature2ds), objects[0].shape[-1]
    longest_objects_num = max([item.shape[0] for item in objects])
    ret_objects = torch.zeros([bsz, longest_objects_num, obj_dim])
    ret_objects_mask = torch.ones([bsz, longest_objects_num])
    for i in range(bsz):
        ret_objects[i, :objects[i].shape[0], :] = objects[i]
        ret_objects_mask[i, :objects[i].shape[0]] = 0.0

    feature2ds = torch.cat([item[None, ...] for item in feature2ds], dim=0)  # (bsz, sample_numb, dim_2d)
    caption_ids = torch.cat([item[None, ...] for item in caption_ids], dim=0)
    caption_masks = caption_ids > 0
    caption_semantics = torch.cat([item[None, ...] for item in caption_semantics], dim=0)
    captions = [item for item in captions]
    vp_semantics = torch.cat([item[None, ...] for item in vp_semantics], dim=0)  # (bsz, dim_sem)
    nouns_dict = list(nouns_dict)
    video_id = list(video_id)

    return feature2ds.float(), ret_objects.float(), ret_objects_mask.float(), \
           caption_ids.long(), caption_masks.float(), caption_semantics.float(), captions, \
           nouns_dict, vp_semantics.float(), video_id
