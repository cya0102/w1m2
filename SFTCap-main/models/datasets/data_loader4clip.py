import json
import os
import pickle
from collections import defaultdict
from nltk import pos_tag, word_tokenize

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random

from configs.settings import TotalConfigs
from models.layers.clip import clip


class CaptionDataset4clip(Dataset):
    def __init__(self, cfgs: TotalConfigs, mode):
        """1.获取文本信息"""
        self.mode = mode
        self.dataset_name = cfgs.data.dataset
        videos_split = cfgs.data.videos_split.format(mode)
        self.ann = json.load(open(cfgs.data.ann_root, 'r'))

        if cfgs.data.dataset == 'vatex':
            video_ids = self.ann[mode].copy()
        else:
            with open(videos_split, 'rb') as f:
                video_ids = pickle.load(f)
        self.video_ids = video_ids
        # 获取训练、测试ids
        """1.获取视觉特征"""
        self.visual_path = cfgs.data.visual_features
        self.objects_visual_path = cfgs.data.object_features.format(mode)
        sample_numb = cfgs.sample_numb
        self.sample_numb = sample_numb
        self.visual_dict = {}
        self.objects_dict = {}
        # visual dict
        if cfgs.data.dataset == 'vatex':
            mapping_file = 'E:\SFT4Caps\data/vatex\language/vatex_mapping.txt'
            real_to_custom = {}

            with open(mapping_file, 'r') as f:
                for line in f:
                    real, custom = line.strip().split()
                    real_to_custom[real] = custom

        for vid in tqdm(self.video_ids):
            if cfgs.data.dataset == 'vatex':
                # if os.path.exists(os.path.join(self.visual_path, real_to_custom[vid] + '.npy')):
                #     temp_feat = np.load(os.path.join(self.visual_path, real_to_custom[vid] + '.npy'))
                # else:
                #     self.video_ids.remove(vid)
                #     continue
                with h5py.File('E:\SFT4Caps\data/vatex/visual\CLIP_ViT-B-32.hdf5', 'r') as f:
                    if real_to_custom[vid] in f.keys():
                        temp_feat = f[real_to_custom[vid]][()]
                        sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
                        self.visual_dict[vid] = temp_feat[sampled_idxs]
                    else:
                        self.video_ids.remove(vid)
                        continue
                # # if os.path.exists(os.path.join(self.visual_path, real_to_custom[vid] + '.npy')):
                # #     temp_feat = np.load(os.path.join(self.visual_path, real_to_custom[vid] + '.npy'))
                # # elif os.path.exists(os.path.join('F:/datasets/VATEX/frame', real_to_custom[vid] + '.npy')):
                # #     temp_feat = np.load(os.path.join('F:/datasets/VATEX/frame', real_to_custom[vid] + '.npy'))
            else:
                temp_feat = np.load(os.path.join(self.visual_path, vid + '.npy'))
            '''2014/4/2注释2行'''
            sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
            self.visual_dict[vid] = temp_feat[sampled_idxs]
        self.video_ids=self.visual_dict.keys()
        # feature object dict
        '''2014/4/2将object换成concept,同步改setting'''
        # with h5py.File(os.path.join('E:/HMN-master/data/MSVD/visual','MSVD_inceptionresnetv2_'+mode+'.hdf5'), 'r') as f:
        #     for vid in tqdm(self.video_ids):
        #         temp_feat = f[vid][()]
        #         sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
        #         self.visual_dict[vid] = temp_feat[sampled_idxs]

        """2.获取文本特征"""
        self.max_words=cfgs.test.max_caption_len
        # msvd\msrvtt提取方式不同

        self.sentences = []
        best_res_filepath = os.path.join(cfgs.data.language_dir,
                                    "teacher_cap_{}.json".format(mode))
        "teacher caption"
        best_json_ref = {k: [] for k in self.video_ids}
        if os.path.exists(best_res_filepath):
            with open(best_res_filepath, "r") as f:
                best_json_ref = json.load(f)
        else:
            os.makedirs(os.path.dirname(best_res_filepath), exist_ok=True)
        json_ref = {k: [] for k in self.video_ids}
        vid2sentence = defaultdict(list)
        if cfgs.data.dataset == 'msvd':
            for item in tqdm(self.ann["metadata"]):
                if item["video_id"] in self.video_ids:
                    if mode=="train":
                        self.sentences.append([item["video_id"], [item["sentence"]]])
                    else:
                        vid2sentence[item["video_id"]].append(item["sentence"])
                        self.sentences=list(vid2sentence.items())
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item["sentence"])
                if item["video_id"] in best_json_ref:
                    if len(best_json_ref[item["video_id"]])==0:
                        teacher_cap = TFIDF(json_ref[item["video_id"]])
                        # teacher_cap = get_best_sentence(json_ref[item["video_id"]])
                        best_json_ref[item["video_id"]]=teacher_cap
            with open(best_res_filepath, "w") as f:
                json.dump(best_json_ref, f)
        elif cfgs.data.dataset == 'msrvtt':
            for item in tqdm(self.ann['sentences']):
                if item["video_id"] in self.video_ids:
                    if mode=="train":
                        self.sentences.append([item["video_id"], [item['caption']]])
                    else:
                        vid2sentence[item["video_id"]].append(item['caption'])
                        self.sentences=list(vid2sentence.items())
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item['caption'])
                if item["video_id"] in best_json_ref:
                    teacher_cap = TFIDF(json_ref[item["video_id"]])
                    best_json_ref[item["video_id"]]=teacher_cap
        elif cfgs.data.dataset == 'vatex':
            for item in tqdm(self.ann['metadata']):
                if item["video_id"] in self.visual_dict.keys():
                    if mode=="train":
                        self.sentences.append([item["video_id"], [item['sentence']]])
                    else:
                        vid2sentence[item["video_id"]].append(item['sentence'])
                        self.sentences=list(vid2sentence.items())
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item['sentence'])
                if item["video_id"] in best_json_ref:
                    teacher_cap = TFIDF(json_ref[item["video_id"]])
                    best_json_ref[item["video_id"]]=teacher_cap
        self.json_ref = json_ref
        self.best_json_ref = best_json_ref
        "save"
        # best_res_filepath = os.path.join(cfgs.train.checkpoints_dir,
        #                             "log/teacher_cap.json")
        # os.makedirs(os.path.dirname(best_res_filepath), exist_ok=True)
        # with open(res_filepath, "w") as f:
        #     json.dump(best_json_ref, f)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        # if self.mode=="train":
        #     sentence = random.choice(sentence_list)
        # else:
        #     sentence = sentence_list[0]
        sentence = random.choice(sentence_list)
        # sentence = sentence_list[0]
        captions = sentence
        feature2d = self.visual_dict[video_id]
        # objects = self.objects_dict[video_id]

        caption_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        caption_mask = torch.zeros(self.max_words, dtype=torch.long)
        caption_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1
        caption_labels = torch.cat((caption_ids[1:], torch.IntTensor([0])))
        "select video_id's caption"
        teacher_cap = self.best_json_ref[video_id]
        T_caption_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        T_caption_mask = torch.zeros(self.max_words, dtype=torch.long)
        T_caption_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1
        T_caption_labels = torch.cat((caption_ids[1:], torch.IntTensor([0])))
        '''获取分词'''
        noun_token = []
        verb_token = []
        # adjective_token = []
        lang_tokens = word_tokenize(teacher_cap)
        pos_tags = pos_tag(lang_tokens)
        # 检查标注为动词和形容词的单词
        for token, pos_tagss in pos_tags:
            if pos_tagss.startswith('NN'):
                noun_token.append(token)
                # print(f"{token}: Nouns")
            elif pos_tagss.startswith('VB'):
                verb_token.append(token)
                # print(f"{token}: Verb")
            # elif pos_tagss.startswith('JJ'):
            #     adjective_token.append(token)
            #     print(f"{token}: Adjective")
        if len(verb_token)==0:
            verb_token.append("is")
        if len(noun_token)==0:
            noun_token.append("man")
        noun_text = clip.tokenize(noun_token, context_length=self.sample_numb, truncate=True)[0]
        verb_text = clip.tokenize(verb_token, context_length=self.sample_numb, truncate=True)[0]
        objects_ids = noun_text
        objects_lable = torch.cat((noun_text[1:], torch.IntTensor([0])))
        objects_mask = torch.zeros(self.sample_numb, dtype=torch.long)
        objects_mask[:len(noun_token) + 2] = 1

        verbs_ids = verb_text
        verbs_lable = torch.cat((verb_text[1:], torch.IntTensor([0])))
        verbs_mask = torch.zeros(self.sample_numb, dtype=torch.long)
        verbs_mask[:len(verb_token) + 2] = 1
        # adjective_text = clip.tokenize(adjective_token, truncate=True).to(device)
        # clip_model,_ = clip.load(name="/media/hpc/39C3AC34579106FA/CX/CoCap/model_zoo/clip_model/ViT-B-16.pt")
        #
        # noun_text_features = clip_model.encode_text(noun_text)
        # noun_text_features = noun_text_features / noun_text_features.norm(dim=-1, keepdim=True)
        nouns_dict = {'nouns': noun_token, 'vec': noun_text}
        verbs_dict = {'verbs': verb_token, 'vec': verb_text}

        'student'


        # caption_labels = torch.cat((caption_ids[1:].to('cpu'), torch.IntTensor([0])))

        return torch.FloatTensor(feature2d), objects_ids, objects_mask, objects_lable, \
               verbs_ids, verbs_mask, verbs_lable, \
               caption_ids, caption_mask, caption_labels, captions, T_caption_ids, T_caption_mask, T_caption_labels, teacher_cap, \
               nouns_dict, verbs_dict, video_id

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def TFIDF(subtitles):
    """
    1.
    :param subtitles:
    :return: best_sentence
    """
    "step:1<get_best_sentence_for_video>"
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(subtitles)
    avg_tfidf_scores = tfidf_matrix.mean(axis=1)
    best_index = np.argmax(avg_tfidf_scores)
    best_sentence = subtitles[best_index]
    return best_sentence


# Since the user asks to use pycocoevalcap, here is the code they can use.
# But note: pycocoevalcap is not pip-installable and needs manual setup from its GitHub repo.
# This script assumes you've already set up pycocoevalcap locally and can import its scorers.

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

def get_best_sentence(subtitles):
    # Initialize scorers
    cider_scorer = Cider()
    meteor_scorer = Meteor()
    bleu_scorer = Bleu(4)
    rouge_scorer = Rouge()

    best_avg_score = -1
    best_sentence = None

    # Format: {id: [sentence]}
    gts = {i: [s] for i, s in enumerate(subtitles)}
    res = {i: [s] for i, s in enumerate(subtitles)}

    for i, candidate in enumerate(subtitles):
        bleu4_total, meteor_total, rougeL_total, cider_total = 0, 0, 0, 0
        count = 0

        # Create reference set (excluding itself)
        refs = {j: gts[j] for j in gts if j != i}
        cand = {j: res[i] for j in refs}

        # BLEU
        bleu_score, _ = bleu_scorer.compute_score(refs, cand)
        bleu4 = bleu_score[3]  # BLEU-4
        bleu4_total = sum(bleu4) / len(bleu4)

        # METEOR
        meteor_score, _ = meteor_scorer.compute_score(refs, cand)
        meteor_total = meteor_score

        # ROUGE-L
        rouge_score, _ = rouge_scorer.compute_score(refs, cand)
        rougeL_total = rouge_score

        # CIDEr
        cider_score, _ = cider_scorer.compute_score(refs, cand)
        cider_total = cider_score

        avg_score = (bleu4_total + meteor_total + rougeL_total + cider_total) / 4

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_sentence = candidate

    return best_sentence




def collate_fn_caption_clip(batch):
    feature2ds, objects, objects_mask, objects_lable, verbs_ids, verbs_mask, verbs_lable, caption_ids, caption_mask, caption_labels, captions, T_caption_ids, T_caption_mask, T_caption_labels, teacher_cap, nouns_dict, \
    verbs_dict, video_id = zip(*batch)
    '''2014/4/2注释7行'''
    # bsz, obj_dim = len(feature2ds), objects[0].shape[-1]
    #     # longest_objects_num = max([item.shape[0] for item in objects])
    #     # ret_objects = torch.zeros([bsz, longest_objects_num, obj_dim])
    #     # ret_objects_mask = torch.ones([bsz, longest_objects_num])
    #     # for i in range(bsz):
    #     #     ret_objects[i, :objects[i].shape[0], :] = objects[i]
    #     #     ret_objects_mask[i, :objects[i].shape[0]] = 0.0
    ret_objects = torch.cat([item[None, ...] for item in objects], dim=0)
    ret_objects_mask = torch.cat([item[None, ...] for item in objects_mask], dim=0)
    object_labels = torch.cat([item[None, ...] for item in objects_lable], dim=0)

    verb_labels = torch.cat([item[None, ...] for item in verbs_lable], dim=0)

    feature2ds = torch.cat([item[None, ...] for item in feature2ds], dim=0)  # (bsz, sample_numb, dim_2d)
    caption_ids = torch.cat([item[None, ...] for item in caption_ids], dim=0)
    caption_masks =torch.cat([item[None, ...] for item in caption_mask], dim=0)
    caption_labels = torch.cat([item[None, ...] for item in caption_labels], dim=0)
    # caption_masks = caption_ids > 0
    captions = [item for item in captions]
    T_caption_ids = torch.cat([item[None, ...] for item in T_caption_ids], dim=0)
    T_caption_masks = torch.cat([item[None, ...] for item in T_caption_mask], dim=0)
    T_caption_labels = torch.cat([item[None, ...] for item in T_caption_labels], dim=0)
    teacher_cap = [item for item in teacher_cap]
    # vp_semantics = torch.cat([item[None, ...] for item in vp_semantics], dim=0)  # (bsz, dim_sem)
    nouns_dict = list(nouns_dict)
    verbs_dict = list(verbs_dict)
    video_id = list(video_id)

    return feature2ds.float(), ret_objects.float(), ret_objects_mask.float(), object_labels.float(), verb_labels.float(),\
           caption_ids.long(),  caption_labels.float(), caption_masks.float(),captions, T_caption_ids.long(), T_caption_masks.float(), T_caption_labels.float(), teacher_cap,\
           nouns_dict, verbs_dict, video_id

