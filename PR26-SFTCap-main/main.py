import os
from torchsummary import summary
import json
import time
import yaml
import torch
import logging
import numpy as np
from torch import optim
from inspect import isclass
from configs.settings import get_settings
from models.build.build_loader import build_dataset
from models.build.build_model import build_model
from models.hungary import HungarianMatcher
from models.layers.clip import clip
from models.trainer.trainer import SFTTrainer
from test import test_fn, test_fn_clip

from train import train_fn

from utils.sys_utils import set_random_seed, init_distributed

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    """1.获取参数,初始化网络、log、设备、参数等"""
    cfg = get_settings()
    # init_distributed(0, cfg)
    yaml.dump(cfg, open(os.path.join(cfg.train.checkpoints_dir, 'config.yaml'), 'w'))
    logging.basicConfig(level=getattr(logging, cfg.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    set_random_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """2.定义模型
    2.1如果使用端到端，才需要使用clip，使用端到端的用法（参照clip4clip）
    2.2如果使用预提取，就用预提取(参照preprocess（vlint）)
    """
    logger.info("Creating model")

    model = build_model(config=cfg, pretrained='')
    model = model.float()
    model = model.to(device)
    # logger.info(f'Model total parameters: {sum(p.numel() for p in model.parameters()):,}')
    # print(f'Model total parameters: {sum(p.numel() for p in model.SFTNet.parameters()):,} + {sum(p.numel() for p in model.caption_head.parameters()):,}')

    '''
    3.定义数据，处理数据使用clip的preprocess,tokenize(在数据集里引用)
    3.1端到端使用了clip的preprocess，相当于transform
    3.2不使用端到端，会预处理（1.clip的preprocess，2.clip4clip的处理）
    '''
    logger.info("Creating datasets")
    train_loader, valid_loader, test_loader = build_dataset(cfg)
    hungary_matcher = HungarianMatcher()

    """
    4.训练，需要构建数据流（主要看forward）
    """
    # trainer = Trainer(cfg, model, hungary_matcher, train_loader, val_loader, device)
    # trainer.train()
    if cfg.data.train_type == "preprocess":
        # model.load_state_dict(torch.load(
        #     'E:\SFT4Caps\output\checkpoints\msrvtt\SFT_True_checkpoint_2024-03-24T19-42-24\clip_l14_epochs_20_lr_0.0002_entity_0.6_predicate_0.3_sentence_1.0_ne_2_nd_2_max_objects_8.ckpt'))
        # model = train_fn(cfg, model, hungary_matcher, test_loader, test_loader, device)
        # model.load_state_dict(torch.load(cfg.train.save_checkpoints_path))
        # model.load_state_dict(torch.load('E:\SFT4Caps\output\checkpoints\msrvtt\SFT_True_checkpoint_2024-03-24T19-42-24\clip_l14_epochs_20_lr_0.0002_entity_0.6_predicate_0.3_sentence_1.0_ne_2_nd_2_max_objects_8.ckpt'))
        model.eval()
        test_fn(cfg, model, test_loader, device)
    else:
        # model.load_state_dict(torch.load('E:\SFT4Caps\output\checkpoints/vatex\SFT_False_checkpoint_2025-06-09T18-19-56_Tloss_0.5_KLo_0.0_KLa_0.0_DLo_0.0_DLa_0.0\clip_b32_epochs_20_lr_0.0004_useonly_S_use_transformer_module_ham_True_max_objects_4.ckpt'),strict=False)
        trainer = SFTTrainer(cfg, model, train_loader, valid_loader, device)
        trainer.training()
        # model.load_state_dict(torch.load(
        #     'E:\SFT4Caps\output\checkpoints\msrvtt\SFT_True_checkpoint_2025-05-27T10-28-10_Tloss_0.5_KLo_0.0_KLa_0.0_DLo_0.0_DLa_0.0\clip_l14_epochs_20_lr_0.0002_use_transformer_module_ham_True_max_objects_4.ckpt'),strict=False)
        model.load_state_dict(torch.load(cfg.train.save_checkpoints_path))
        model.eval()
        test_fn_clip(cfg, model, test_loader, device)
# --SFT
# --use_subject
# --use_predict
# --use_sentence
# --use_transformer
# --use_lstm
# --use_ham
#--Only_T_or_S
"""
1.SFT
--SFT
--use_ham
--use_module
transformer
"""
"""
2.T
--Only_T_or_S
T
--use_ham
--use_module
transformer
"""
"""
2.S
--Only_T_or_S
S
--use_ham
"""