import json
import os
import logging
import time

import einops
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from collections import defaultdict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from typing import List, AnyStr

from configs.settings import get_timestamp
from utils.profile import Timer
from utils.train_utils import gather_object_multiple_gpu

logger = logging.getLogger(__name__)
class Translator(object):
    """Load with trained model and handle the beam search"""

    def __init__(self, checkpoint, model=None):
        self.max_v_len = checkpoint['cap_config'].max_v_len
        self.max_t_len = checkpoint['cap_config'].max_t_len
        self.PAD = checkpoint['cap_config'].PAD_id
        self.BOS = checkpoint['cap_config'].BOS_id

        self.model = model
        self.model.eval()

        self.timer = Timer(synchronize=True, history_size=500, precision=3)

    def translate_batch_single_sentence_greedy(self, objects, object_masks, feature2ds, input_ids, input_mask, model):
        inputs_ids = input_ids
        input_masks = input_mask
        max_t_len = self.max_t_len  # hard-code sentence length, for speed test, set it to 21
        inputs_ids[:, :] = 0.
        input_masks[:, :] = 0.
        assert torch.sum(input_masks[:, :]) == 0, "Initially, all text tokens should be masked"
        bsz = len(inputs_ids)
        next_symbols = torch.IntTensor([self.BOS] * bsz)  # (N, )
        logger.info(f'Model total parameters: {sum(p.numel() for p in model.parameters()):,}')

        self.timer.reset()
        # from fvcore.nn import FlopCountAnalysis
        # flops = FlopCountAnalysis(self.model, (objects, object_masks, feature2ds, input_ids, input_mask))
        # print(f"FLOPs: {flops.total()}")
        # print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
        for dec_idx in range(max_t_len):
            inputs_ids[:, dec_idx] = next_symbols.clone()
            input_masks[:, dec_idx] = 1
            start_time = time.time()
            outputs = model(objects, object_masks, feature2ds, input_ids, input_mask)
            end_time = time.time()
            print(end_time - start_time)
            if self.model.Use_SFT:
                pred_scores = outputs["S_prediction_scores"]
            else:
                pred_scores = outputs["prediction_scores"]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words.cpu()
        self.timer(stage_name="inference")

        return inputs_ids

    def translate_batch(self, objects, object_masks, feature2ds, input_ids, input_mask):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        return self.translate_batch_single_sentence_greedy(objects, object_masks, feature2ds, input_ids, input_mask, self.model)

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


def run_translate(data_loader, translator, device,epoch, opt):
    # submission template
    batch_res = {"version": "VERSION 1.0",
                 "results": defaultdict(list),
                 "external_data": {"used": "true", "details": "ay"}}
    bar = data_loader = tqdm(data_loader,
                              desc=f"Train: {epoch + 1}/{opt.train.max_epochs}",
                              dynamic_ncols=True,
                              disable=dist.is_initialized() and dist.get_rank() != 0)
    for cur_step, (
            feature2ds, objects, object_masks, objects_lable, verb_label,input_ids, input_labels, input_mask, captions, T_caption_ids, T_caption_mask, T_caption_labels, teacher_cap,
            nouns_dict, verbs_dict, video_id) in enumerate(data_loader):
        feature2ds = feature2ds.to(device)
        objects = objects.to(device)
        object_masks = object_masks.to(device)
        input_ids = input_ids.to(device)
        input_labels = input_labels.to(device)
        input_mask = input_mask.to(device)
        dec_seq = translator.translate_batch(objects, object_masks, feature2ds, input_ids, input_mask)

        # example_idx indicates which example is in the batch
        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, captions)):
            cur_data = {
                "sentence": convert_ids_to_sentence(cur_gen_sen.tolist()),
                "gt_sentence": cur_meta
            }
            batch_res["results"][video_id[example_idx]].append(cur_data)
    logger.debug(translator.timer.print())
    return batch_res


class EvalCap:
    def __init__(self, annos, rests, cls_tokenizer=PTBTokenizer,
                 use_scorers=('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr')):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annos = annos
        self.rests = rests
        self.Tokenizer = cls_tokenizer
        self.use_scorers = use_scorers

    def evaluate(self):
        res = {}
        for r in self.rests:
            res[str(r['image_id'])] = [{'caption': r['caption']}]

        gts = {}
        for imgId in self.annos:
            gts[str(imgId)] = [{'caption': c} for c in self.annos[imgId]]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = self.Tokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        use_scorers = self.use_scorers
        scorers = []
        if 'Bleu' in use_scorers:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in use_scorers:
            scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' in use_scorers:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' in use_scorers:
            scorers.append((Cider(), "CIDEr"))
        if 'SPICE' in use_scorers:
            scorers.append((Spice(), "SPICE"))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def evaluate(submission, reference):
    tokenizer = PTBTokenizer  # for English
    annos = reference
    data = submission['results']
    rests = []
    for name, value in data.items():
        rests.append({'image_id': str(name), 'caption': value[0]['sentence']})
    eval_cap = EvalCap(annos, rests, tokenizer)

    eval_cap.evaluate()

    all_score = {}
    for metric, score in eval_cap.eval.items():
        all_score[metric] = score
    return all_score


def eval_language_metrics(checkpoint, eval_data_loader, opt, model, device, eval_mode="test"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
    0, run inference
    1, Get METEOR, BLEU1-4, CIDEr scores
    2, Get vocab size, sentence length
    """
    translator = Translator(checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, device, checkpoint["epoch"], opt=opt)
    if dist.is_initialized():
        all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
        json_res['results'] = {k: v for k, v in all_results}
        logger.debug("Caption test length: %s", len(json_res["results"].items()))

    # save result tp log for debug
    if not dist.is_initialized() or dist.get_rank() == 0:
        res_filepath = os.path.join(opt.train.checkpoints_dir, "log/caption_greedy_pred_{}_epoch{}.json".format(eval_mode,checkpoint["epoch"]))
        os.makedirs(os.path.dirname(res_filepath), exist_ok=True)
        with open(res_filepath, "w") as f:
            json.dump(json_res,f)

    if not dist.is_initialized() or dist.get_rank() == 0:
        json_ref = eval_data_loader.dataset.json_ref
        return evaluate(json_res, json_ref)
    else:
        return None