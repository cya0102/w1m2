import torch
from torch.utils.data import DataLoader

from models.datasets.data_loader import CaptionDataset, collate_fn_caption
from configs.settings import TotalConfigs
from models.datasets.data_loader4clip import CaptionDataset4clip, collate_fn_caption_clip


def build_dataset(cfgs: TotalConfigs):
    if cfgs.data.train_type == 'preprocess':
        train_dataset = CaptionDataset(cfgs=cfgs, mode='train')
        valid_dataset = CaptionDataset(cfgs=cfgs, mode='valid')
        test_dataset = CaptionDataset(cfgs=cfgs, mode='test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.bsz, shuffle=True,
                              collate_fn=collate_fn_caption, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfgs.bsz, shuffle=True,
                              collate_fn=collate_fn_caption, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.bsz, shuffle=False,
                             collate_fn=collate_fn_caption, num_workers=0)
    else:
        train_dataset = CaptionDataset4clip(cfgs=cfgs, mode='train')
        if cfgs.data.dataset == 'vatex':
            valid_dataset = CaptionDataset4clip(cfgs=cfgs, mode='test')
            test_dataset=valid_dataset
        else:
            valid_dataset = CaptionDataset4clip(cfgs=cfgs, mode='valid')
            test_dataset = CaptionDataset4clip(cfgs=cfgs, mode='test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.bsz, shuffle=True,
                                  collate_fn=collate_fn_caption_clip, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfgs.bsz, shuffle=True,
                                  collate_fn=collate_fn_caption_clip, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.bsz, shuffle=False,
                                 collate_fn=collate_fn_caption_clip, num_workers=0)
    return train_loader, valid_loader,test_loader