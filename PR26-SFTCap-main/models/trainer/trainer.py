import json
import logging
import os
import time
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm

from models.trainer.evaluate import eval_language_metrics
from models.build.build_loss import build_loss
from models.build.build_meter import get_writer, build_meter
from models.build.build_optimizer import build_optimizer
from models.layers.bert import BertLayerNorm
from models.utils import LayerNorm
from utils.checkpoint import save_checkpoint, save_model
from utils.train_utils import CudaPreFetcher

logger = logging.getLogger(__name__)
class SFTTrainer:
    def __init__(self, cfg, model, train_loader, valid_loader, device):
        self.cfg = cfg
        '''init'''
        self.model = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.loss_func = None

        self.global_step = self.epoch_start = self.epoch = 0

        self.writer = None
        self.meter = None
        self.best_score = None

        '''build'''
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # update total step for optimizer
        self.gradient_accumulation_step = cfg.bert.gradient_accumulation_steps
        epoch = cfg.train.max_epochs
        num_train_optimization_steps = (int(len(self.train_loader) + self.gradient_accumulation_step - 1)
                                        / self.gradient_accumulation_step) * epoch
        cfg.bert.t_total = num_train_optimization_steps
        self.optimizer, self.scheduler = self.prep_optimizer(cfg, self.model)
        self.loss_func = build_loss(self.cfg)

    @staticmethod
    def prep_optimizer(cfg, model):
        # based on:
        # https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
        if hasattr(model, 'module'):
            model = model.module

        decay = set()
        no_decay = set()
        all_decay = set()
        all_nodecay = set()
        pretrained_modules = [
            "caption_head.cap_sa_decoder.word_embeddings",
            "caption_head.prediction_head.decoder",
        ]
        encoder = ['Encoder_layer.entity_level',]
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
        blacklist_weight_modules = (LayerNorm, nn.LayerNorm, nn.BatchNorm2d, nn.Embedding, BertLayerNorm)
        # param_dict = {}
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # param_dict[pn] = p
                if any(fpn.startswith(p_fpn) for p_fpn in pretrained_modules):  # pretrained
                    no_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("bias_hh_l0") or pn.endswith("bias_hh_l0_reverse"):
                    no_decay.add(fpn)
                elif pn.endswith("bias_ih_l0") or pn.endswith("bias_ih_l0_reverse"):
                    no_decay.add(fpn)
                elif pn.endswith("proj"):
                    decay.add(fpn)
                elif pn.endswith("projection"):
                    decay.add(fpn)
                elif fpn.endswith("embedding"):
                    no_decay.add(fpn)
                elif pn.endswith("weight_hh_l0") or pn.endswith("weight_hh_l0_reverse"):
                    decay.add(fpn)
                elif pn.endswith("weight_ih_l0") or pn.endswith("weight_ih_l0_reverse"):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("b") or pn.endswith("ba") or pn.endswith("bm") or pn.endswith("bv") or pn.endswith("bos"):
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}
        all = [pn for pn in sorted(list(param_dict.keys())) if
               any(pn.startswith(p_pn) for p_pn in encoder)]
        all_nodecay = [pn for pn in sorted(list(no_decay)) if
                       any(pn.startswith(p_pn) for p_pn in encoder)]
        all_decay = [pn for pn in sorted(list(decay)) if
                       any(pn.startswith(p_pn) for p_pn in encoder)]
        all2=all_decay+all_nodecay
        last = [pn for pn in sorted(all) if pn not in all2]
        for pn in last:
            decay.add(pn)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                               any(pn.startswith(p_pn) for p_pn in pretrained_modules)]
        not_pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                                   not any(pn.startswith(p_pn) for p_pn in pretrained_modules)]

        # logger.debug("Parameter group decay_param: %s",
        #              "\n   " + "\n   ".join([pn for pn in sorted(list(decay))]))
        # logger.debug("Parameter group no_decay_pretrained_param: %s",
        #              "\n   " + "\n   ".join([pn for pn in sorted(list(pretrained_no_decay))]))
        # logger.debug("Parameter group no_decay_not_pretrained_param: %s",
        #              "\n   " + "\n   ".join([pn for pn in sorted(list(not_pretrained_no_decay))]))

        decay_param = [param_dict[pn] for pn in sorted(list(decay))]
        no_decay_pretrained_param = [param_dict[pn] for pn in sorted(list(pretrained_no_decay))]
        no_decay_not_pretrained_param = [param_dict[pn] for pn in sorted(list(not_pretrained_no_decay))]

        optimizer_grouped_parameters = [
            {"params": decay_param},
            {"params": no_decay_pretrained_param, "weight_decay": 0.0, "lr": cfg.bert.clip_lr},
            {"params": no_decay_not_pretrained_param, "weight_decay": 0.0}
        ]

        warmup_epoch = int(cfg.bert.warmup * cfg.train.max_epochs)
        optimizer = build_optimizer(cfg, optimizer_grouped_parameters)
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 if epoch < warmup_epoch
            else cfg.bert.lr_decay_gamma ** (epoch - warmup_epoch)
        )

        return optimizer, scheduler

    def training(self):
        #before train
        self.global_step = self.epoch_start * (len(self.train_loader) // self.gradient_accumulation_step)
        self.writer = get_writer(os.path.join(self.cfg.data.log_dir, "tensorboard"), purge_step=self.global_step)
        self.meter = build_meter(cfg=self.cfg, writer=self.writer, mode="train")
        #on train
        Tr = self.cfg.train.T_loss
        for epoch in range(self.epoch_start, self.cfg.train.max_epochs):
            """Training part"""
            self.epoch = epoch
            logger.debug(f"Epoch {epoch + 1}/{self.cfg.train.max_epochs}")
            self.model.train()
            self.meter.set_mode("train")
            torch.cuda.empty_cache()
            # if torch.cuda.is_available():
            #     logger.debug("Building CudaPreFetcher...")
            #     train_loader = CudaPreFetcher(self.train_loader)  # prefetch to GPU
            bar = train_loader = tqdm(self.train_loader,
                                    desc=f"Train: {self.epoch + 1}/{self.cfg.train.max_epochs}",
                                    dynamic_ncols=True,
                                    disable=dist.is_initialized() and dist.get_rank() != 0)
            loss_total = 0.
            logger.debug("Running train epoch for-loop...")

            for cur_step, (
                    feature2ds, objects, object_masks, objects_lable, verb_label,
                    input_ids, input_labels, input_mask, captions, T_caption_ids, T_caption_mask, T_caption_labels, teacher_cap,
                    nouns_dict, verbs_dict, video_id) in enumerate(train_loader):
                inputs = (
                    feature2ds, objects, object_masks, input_ids, input_labels, input_mask,
                    video_id, captions)
                feature2ds = feature2ds.to(self.device)
                objects = objects.to(self.device)
                object_masks = object_masks.to(self.device)
                objects_lable = objects_lable.to(self.device)

                verbs_label = verb_label.to(self.device)

                input_ids = input_ids.to(self.device)
                input_labels = input_labels.to(self.device)
                input_mask = input_mask.to(self.device)
                T_caption_ids = T_caption_ids.to(self.device)
                T_caption_mask = T_caption_mask.to(self.device)
                T_caption_labels = T_caption_labels.to(self.device)
                # from fvcore.nn import FlopCountAnalysis
                # flops = FlopCountAnalysis(self.model, (objects, object_masks, feature2ds, input_ids, input_mask, T_caption_ids, T_caption_mask))
                # print(f"FLOPs: {flops.total()}")
                # print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
                # start_time = time.time()
                outputs = self.model(objects, object_masks, feature2ds, input_ids, input_mask, T_caption_ids, T_caption_mask)
                # end_time = time.time()
                # inference_time = end_time - start_time
                # print(f"Inference Time: {inference_time:.6f} seconds")
                if self.cfg.SFT:
                    "T"
                    loss_meta_T = self.loss_func(T_caption_labels, outputs["T_prediction_scores"])
                    # backward
                    if isinstance(loss_meta_T, dict):
                        loss_T = sum([v for _, v in loss_meta_T.items()])
                    else:
                        loss_T = loss_meta_T
                    "S"
                    loss_meta_S = self.loss_func(input_labels, outputs["S_prediction_scores"])
                    # backward
                    if isinstance(loss_meta_S, dict):
                        loss_S = sum([v for _, v in loss_meta_S.items()])
                    else:
                        loss_S = loss_meta_S
                    "Obj"
                    loss_meta_O = self.loss_func(objects_lable, outputs["O_prediction_scores"])
                    # backward
                    if isinstance(loss_meta_O, dict):
                        loss_O = sum([v for _, v in loss_meta_O.items()])
                    else:
                        loss_O = loss_meta_O
                    "Act"
                    loss_meta_A = self.loss_func(objects_lable, outputs["A_prediction_scores"])
                    # backward
                    if isinstance(loss_meta_A, dict):
                        loss_A = sum([v for _, v in loss_meta_A.items()])
                    else:
                        loss_A = loss_meta_A


                    lam_o = self.cfg.train.lam_o
                    lam_a = self.cfg.train.lam_a
                    loss_meta=Tr*loss_meta_T+(1-Tr)*loss_meta_S + lam_o*loss_meta_O + lam_a*loss_meta_A
                    loss=Tr*loss_T+(1-Tr)*loss_S+lam_o*loss_O+lam_a*loss_A
                elif self.cfg.Only_T_or_S =="S":
                    "T"
                    loss_meta_T = self.loss_func(T_caption_labels, outputs["prediction_scores"])
                    # backward
                    if isinstance(loss_meta_T, dict):
                        loss_T = sum([v for _, v in loss_meta_T.items()])
                    else:
                        loss_T = loss_meta_T
                    "S"
                    loss_meta_S = self.loss_func(input_labels, outputs["prediction_scores"])
                    # backward
                    if isinstance(loss_meta_S, dict):
                        loss_S = sum([v for _, v in loss_meta_S.items()])
                    else:
                        loss_S = loss_meta_S
                    "Obj"
                    loss_meta_O = self.loss_func(objects_lable, outputs["O_prediction_scores"])
                    # backward
                    if isinstance(loss_meta_O, dict):
                        loss_O = sum([v for _, v in loss_meta_O.items()])
                    else:
                        loss_O = loss_meta_O
                    "Act"
                    loss_meta_A = self.loss_func(verbs_label, outputs["A_prediction_scores"])
                    # backward
                    if isinstance(loss_meta_A, dict):
                        loss_A = sum([v for _, v in loss_meta_A.items()])
                    else:
                        loss_A = loss_meta_A

                    lam_o = self.cfg.train.lam_o
                    lam_a = self.cfg.train.lam_a
                    loss_meta = Tr * loss_meta_T + (1 - Tr) * loss_meta_S + lam_o * loss_meta_O + lam_a * loss_meta_A
                    loss = Tr * loss_T + (1 - Tr) * loss_S + lam_o * loss_O + lam_a * loss_A
                else:
                    loss_meta = self.loss_func(input_labels, outputs["prediction_scores"])
                    # backward
                    if isinstance(loss_meta, dict):
                        loss = sum([v for _, v in loss_meta.items()])
                    else:
                        loss = loss_meta
                loss /= self.gradient_accumulation_step
                loss.backward()
                loss_total += loss.detach()
                if self.gradient_accumulation_step > 1:
                    bar.set_postfix(
                        {"Accumulation Step": (cur_step + 1) % self.gradient_accumulation_step}
                    )
                if (cur_step + 1) % self.gradient_accumulation_step == 0:
                    # optimize
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # summary
                with torch.no_grad():
                    if cur_step % self.cfg.data.log_freq == 0:
                        if dist.is_initialized():
                            dist.all_reduce(loss)
                            loss = loss / dist.get_world_size()
                            logger.debug(
                                f"loss (rank {dist.get_rank()}, step {self.global_step}): {loss.cpu().detach().numpy()}"
                            )
                        else:
                            logger.debug(f"loss (step {self.global_step}): {loss.cpu().detach().numpy()}")
                        self.writer.add_scalar("train/loss", loss_total, global_step=self.global_step)
                        if isinstance(loss_meta, dict):
                            self.writer.add_scalars("train/loss_meta", loss_meta, global_step=self.global_step)
                        self.writer.add_scalars(
                            "train/lr",
                            {
                                f"param_group_{i}": group["lr"] if self.scheduler is None else group
                                for i, group in enumerate(self.optimizer.param_groups if self.scheduler is None
                                                          else self.scheduler.get_last_lr())
                            },
                            global_step=self.global_step
                        )
                    self.meter.update(inputs=inputs, outputs=outputs, global_step=self.global_step)
                loss_total = 0.
                self.global_step += 1
            logger.debug("Train epoch for-loop finished.")
            self.optimizer.zero_grad()
            self.meter.summary(epoch=self.epoch + 1)
            self.meter.reset()
            # 保存权重
            # if (self.epoch + 1) % self.cfg.train.save_freq == 0:
            #     if not dist.is_initialized() or dist.get_rank() == 0:  # ddp is not enabled or global rank is 0
            #         save_checkpoint(ckpt_folder=os.path.join(self.cfg.train.checkpoints_dir, "checkpoint"),
            #                         epoch=self.epoch + 1,
            #                         model=self.model,
            #                         optimizer=self.optimizer,
            #                         scheduler=self.scheduler,
            #                         config=self.cfg)

            if self.scheduler is not None:
                self.scheduler.step()
            """Validing part"""
            torch.cuda.empty_cache()
            self._on_test_epoch_captioning()
            # if not dist.is_initialized() or dist.get_rank() == 0:
            #     save_model(model_file=os.path.join(self.cfg.train.checkpoints_dir, "pytorch_model.bin"), model=self.model)

    @torch.no_grad()
    def _on_test_epoch_captioning(self):
        self.model.eval()

        valid_loader = self.valid_loader
        checkpoint = {
            "epoch": self.epoch,
            "cap_config": self.model.module.caption_head.cap_config if
            hasattr(self.model, 'module') else self.model.caption_head.cap_config
        }
        metrics = eval_language_metrics(checkpoint, valid_loader, self.cfg, model=self.model, device=self.device,eval_mode = 'valid')

        if not dist.is_initialized() or dist.get_rank() == 0:
            ckpt_path = self.cfg.train.save_checkpoints_path
            # torch.save(self.model.state_dict(), ckpt_path)
            cider_score = metrics['CIDEr']
            if self.best_score is None or cider_score > self.best_score:
                self.best_score = cider_score
                torch.save(self.model.state_dict(), ckpt_path)
                save_model(model_file=os.path.join(self.cfg.train.checkpoints_dir, "pytorch_best_model.bin"),
                           model=self.model)
            logger.info('\t>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
                        format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                               metrics['CIDEr'] * 100))

            logger.info('\t>>>  Best_CIDEr: {:.2f}'.format(self.best_score * 100))

            log_stats = {**{f'[EPOCH{self.epoch+1}]_test{k}': v for k, v in metrics.items()},
                         }
            with open(self.cfg.train.evaluate_dir, "a") as f:
                f.write(json.dumps(log_stats) + '\n')

            for metric, value in metrics.items():
                self.writer.add_scalar(f"test/{metric}", value * 100, global_step=self.epoch)