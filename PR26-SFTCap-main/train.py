import json
import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import torch.distributed as dist
from tqdm import tqdm

from utils.loss import LanguageModelCriterion, CosineCriterion, SoftCriterion
from eval import eval_fn
from configs.settings import TotalConfigs
from models.hungary import HungarianMatcher


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

logger = logging.getLogger(__name__)
def train_fn(cfgs: TotalConfigs, model: nn.Module, matcher: HungarianMatcher, train_loader, valid_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=cfgs.train.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfgs.train.max_epochs, eta_min=0, last_epoch=-1)
    language_loss = LanguageModelCriterion()
    cos_loss = CosineCriterion()
    language_loss.to(device)
    best_score, cnt = None, 0
    loss_store, loss_entity, loss_predicate, loss_sentence, loss_xe, loss_soft_target = [], [], [], [], [], []

    with open(cfgs.data.idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)
    with open(cfgs.data.vid2groundtruth_path, 'rb') as f:
        vid2groundtruth = pickle.load(f)
    Tr = cfgs.train.loss_rate
    print('===================Training begin====================')
    print(cfgs.train.save_checkpoints_path)
    for epoch in range(cfgs.train.max_epochs):
        logger.debug(f"Epoch {epoch + 1}/{cfgs.train.max_epochs}")
        bar = train_loader = tqdm(train_loader,
                                  desc=f"Train: {epoch + 1}/{cfgs.train.max_epochs}",
                                  dynamic_ncols=True,
                                  disable=dist.is_initialized() and dist.get_rank() != 0)
        for i, (
                feature2ds, objects, object_masks, caption_ids, caption_masks, caption_semantics, captions,
                nouns_dict, vp_semantics, video_id) in enumerate(train_loader):
            cnt += 1
            feature2ds = feature2ds.to(device)
            # feature3ds = feature3ds.to(device)
            objects = objects.to(device)
            object_masks = object_masks.to(device)
            vp_semantics = vp_semantics.to(device)
            caption_semantics = caption_semantics.to(device)
            caption_ids = caption_ids.to(device)
            caption_masks = caption_masks.to(device)

            optimizer.zero_grad()

            # preds, objects_pending, action_pending, video_pending = model(objects, object_masks, feature2ds, feature3ds, caption_ids,average_imgs)
            # preds, objects_pending, action_pending, video_pending = model(objects, object_masks, feature2ds,
            #                                                               caption_ids, average_imgs)
            '''源代码'''
            # preds, objects_pending, action_pending, video_pending = model(objects, object_masks, feature2ds,
            #                                                               caption_ids)
            start_time=time.time()
            preds, objects_pending, action_pending, video_pending, preds_s, objects_pending_s, action_pending_s, video_pending_s = model(objects, object_masks, feature2ds,
                                                                          caption_ids)
            end_time=time.time()
            inference_time=end_time-start_time
            print(f"Inference Time: {inference_time:.6f} seconds")

            xe_loss, s_loss, ent_loss, pred_loss, sent_loss = None, None, None, None, None

            # cross entropy loss
            loss_hard = language_loss(preds, caption_ids, caption_masks, cfgs.dict.eos_idx)
            loss_hard_s = language_loss(preds, caption_ids, caption_masks, cfgs.dict.eos_idx)
            loss = Tr*loss_hard+(1-Tr)*loss_hard_s
            xe_loss = loss_hard.detach().item()

            # object module loss
            if cfgs.train.lambda_entity > 0:
                indices = matcher(objects_pending, nouns_dict)
                src_idx = _get_src_permutation_idx(indices)
                objects = objects_pending[src_idx]
                targets = torch.cat([t['vec'][i] for t, (_, i) in zip(nouns_dict, indices)], dim=0).to(device)
                if np.any(np.isnan(objects.detach().cpu().numpy())):
                    raise RuntimeError
                object_loss = cos_loss(objects, targets)

                indices_s = matcher(objects_pending_s, nouns_dict)
                src_idx_s = _get_src_permutation_idx(indices_s)
                objects_s = objects_pending_s[src_idx_s]
                targets_s = torch.cat([t['vec'][i] for t, (_, i) in zip(nouns_dict, indices_s)], dim=0).to(device)
                if np.any(np.isnan(objects_s.detach().cpu().numpy())):
                    raise RuntimeError
                object_loss_s = cos_loss(objects_s, targets_s)

                loss = loss + 0.5*object_loss * cfgs.train.lambda_entity + 0.5*object_loss_s * cfgs.train.lambda_entity
                ent_loss = object_loss.detach().item()

            # action module loss
            if cfgs.train.lambda_predicate > 0:
                action_loss = cos_loss(action_pending, vp_semantics)
                action_loss_s = cos_loss(action_pending_s, vp_semantics)
                loss = loss + 0.5*action_loss * cfgs.train.lambda_predicate + 0.5*action_loss_s * cfgs.train.lambda_predicate
                pred_loss = action_loss.detach().item()

            # video module loss
            if cfgs.train.lambda_sentence > 0:
                sent_loss = cos_loss(video_pending, caption_semantics)
                sent_loss_s = cos_loss(video_pending_s, caption_semantics)
                loss = loss + 0.5*sent_loss * cfgs.train.lambda_sentence + 0.5*sent_loss_s * cfgs.train.lambda_sentence
                sent_loss = sent_loss.detach().item()

            loss.backward()
            loss_store.append(loss.detach().item())
            loss_xe.append(xe_loss)
            loss_entity.append(ent_loss)
            loss_predicate.append(pred_loss)
            loss_sentence.append(sent_loss)
            loss_soft_target.append(s_loss)
            nn.utils.clip_grad_norm_(model.parameters(), cfgs.train.grad_clip)
            optimizer.step()

            if cnt % cfgs.train.visualize_every == 0:
                loss_store, loss_xe, loss_entity, loss_predicate, loss_sentence, loss_soft_target = \
                    loss_store[-10:], loss_xe[-10:], loss_entity[-10:], loss_predicate[-10:], loss_sentence[-10:], loss_soft_target[-10:]
                loss_value = np.array(loss_store).mean()
                xe_value = np.array(loss_xe).mean() if loss_xe[0] is not None else 0
                soft_value = np.array(loss_soft_target).mean() if loss_soft_target[0] is not None else 0
                entity_value = np.array(loss_entity).mean() if loss_entity[0] is not None else 0
                predicate_value = np.array(loss_predicate).mean() if loss_predicate[0] is not None else 0
                sentence_value = np.array(loss_sentence).mean() if loss_sentence[0] is not None else 0
                
                print('[EPOCH {};ITER {}]:loss[{:.3f}]=hard_loss[{:.3f}]*1+entity[{:.3f}]*{:.2f}+predicate[{:.3f}]*{:.2f}+sentence[{:.3f}]*{:.2f}'
                .format(epoch+1, i, loss_value, xe_value,
                        entity_value, cfgs.train.lambda_entity, 
                        predicate_value, cfgs.train.lambda_predicate, 
                        sentence_value, cfgs.train.lambda_sentence))

            if cnt % cfgs.train.save_checkpoints_every == 0:
                ckpt_path = cfgs.train.save_checkpoints_path
                scores = eval_fn(model=model, loader=valid_loader, device=device, 
                                idx2word=idx2word, save_on_disk=False, cfgs=cfgs, 
                                vid2groundtruth=vid2groundtruth)
                cider_score = scores['CIDEr']
                if best_score is None or cider_score > best_score:
                    best_score = cider_score
                    torch.save(model.state_dict(), ckpt_path)
                print('=' * 10,
                      '[EPOCH{epoch} iter{it}] :Best Cider is {bs}, Current Cider is {cs}'.
                      format(epoch=epoch, it=i, bs=best_score, cs=cider_score),
                      '=' * 10)
                log_stats={**{f'[EPOCH{epoch+1} iter{i}]_train{k}':v for k,v in scores.items()},
                          }
                with open(cfgs.train.evaluate_dir, "a") as f:
                    f.write(json.dumps(log_stats)+'\n')

        lr_scheduler.step()
    print('===================Training is finished====================')
    return model
