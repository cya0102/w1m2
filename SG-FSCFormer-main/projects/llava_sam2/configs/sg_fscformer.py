from os import environ
from os.path import basename, join

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.llava_sam2.models.internvl import InternVL_Slowfast

from projects.llava_sam2.models import VideoLLaVASAMModel, SAM2TrainRunner, VideoLLaVASAMModel_zero3, PromptGuidedTemporalGraphFormer, FineGrainedMaskLinguisticDecoder
from projects.llava_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.llava_sam2.datasets import VideoChatUniViDataset, VideoQADataset, VideoBoxCapDataset
from projects.llava_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.llava_sam2.datasets import LLaVADataset
from projects.llava_sam2.datasets import ReferSegmDataset
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.evaluation.segcaption_metric import SegCaptionMetric
from projects.llava_sam2.evaluation.segcaption_val_loop import SegCaptionValLoop

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
PROJECT_NAME = 'SG-FSCFormer'

# Model
internvl_model_path = environ.get(
    'SG_FSCFORMER_INTERNVL_MODEL_PATH', './pretrained/InternVL2_5-4B')
path = environ.get(
    'SG_FSCFORMER_LLM_MODEL_PATH', './pretrained/vicuna-7b')
pretrained_pth = None

# Data
template = environ.get(
    'SG_FSCFORMER_PROMPT_TEMPLATE',
    'vicuna' if 'vicuna' in basename(path).lower() else 'phi3_chat')
prompt_template = getattr(PROMPT_TEMPLATE, template)
max_length = 8192

# Scheduler & Optimizer
batch_size = 2 # 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = [
    '<IMG_CONTEXT>', '<FAST_IMG_CONTEXT>', '<img>', '</img>',
    '[BOX]', '[SEG]', '<p>', '</p>', '<vp>', '</vp>',
]

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')
if 'vicuna' in basename(path).lower():
    tokenizer.update(pad_token='</s>')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=internvl_model_path,
        llm_model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    ptgformer=dict(
        type=PromptGuidedTemporalGraphFormer,
        num_nodes=9,
        num_heads=8,
        temporal_window=1,
        hard_keep_ratio=0.75,
    ),
    mldecoder=dict(
        type=FineGrainedMaskLinguisticDecoder,
        num_queries=5,
        num_heads=8,
        memory_size=8,
        num_iterations=2,
        mc_loss_weight=0.1,
    ),
    validation_generate=True,
    validation_max_new_tokens=128,
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = environ.get('SG_FSCFORMER_DATA_ROOT', './data/')
VIDEO_DATA_ROOT = environ.get(
    'SG_FSCFORMER_VIDEO_DATA_ROOT', join(DATA_ROOT, 'video_datas/'))

############### video res
data_root_revos = VIDEO_DATA_ROOT + 'revos/'
video_revos_image_folder = data_root_revos
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_mask_file = data_root_revos + 'mask_dict.json'

data_root_mevis = VIDEO_DATA_ROOT + 'mevis/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_mask_file = data_root_mevis + 'mask_dict.json'

data_root_refytvos = VIDEO_DATA_ROOT + 'rvos/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = data_root_refytvos + 'meta_expressions/train/meta_expressions.json'
video_refytvos_mask_file = data_root_refytvos + 'mask_dict.pkl'

video_revos_dataset = dict(
    type=VideoReVOSDataset,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    mask_file=video_revos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    mask_file=video_mevis_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    mask_file=video_refytvos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

################### Video chat
data_root_video_chatunivi = VIDEO_DATA_ROOT + 'chat_univi/'
video_chatunivi_image_folder = data_root_video_chatunivi + 'Activity_Videos/'
video_chatunivi_json_file = data_root_video_chatunivi+ 'video_chat.json'

video_chatcust_image_folder = environ.get(
    'SG_FSCFORMER_VIDEO_CHAT_IMAGE_FOLDER',
    join(DATA_ROOT, 'custom_data/ovis/train/Videos/'))
video_chatcust_json_file = environ.get(
    'SG_FSCFORMER_VIDEO_CHAT_JSON_FILE',
    join(DATA_ROOT, 'custom_data/ovis_tr_video_chat.json'))

video_qa_dataset = dict(
    type=VideoChatUniViDataset,
    image_folder=video_chatunivi_image_folder,
    json_file=video_chatunivi_json_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_qa2_dataset = dict(
    type=VideoQADataset,
    image_folder=video_chatcust_image_folder,
    json_file=video_chatcust_json_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

################## video boxcap (LVVIS/OVIS)
LVVIS_ROOT = environ.get('SG_FSCFORMER_LVVIS_ROOT', join(DATA_ROOT, 'lvvis/'))
OVIS_ROOT = environ.get('SG_FSCFORMER_OVIS_ROOT', join(DATA_ROOT, 'ovis/'))
CUSTOM_DATA_ROOT = environ.get(
    'SG_FSCFORMER_CUSTOM_DATA_ROOT', join(DATA_ROOT, 'custom_data/'))
lvvis_scene_graph_root = CUSTOM_DATA_ROOT + 'lvvis/annotation/'
ovis_scene_graph_root = CUSTOM_DATA_ROOT + 'ovis/annotation/'

lvvis_train_ann = LVVIS_ROOT + 'annotation/train_instances_boxcap_split1_3.1_v2.json'
lvvis_val_ann = LVVIS_ROOT + 'annotation/val_instances_boxcap_v2.json'
lvvis_train_frames = LVVIS_ROOT + 'frames/train/'
lvvis_val_frames = LVVIS_ROOT + 'frames/val/'

ovis_train_ann = OVIS_ROOT + 'annotation/malmm_train_instances_boxcap_v2.json'
ovis_val_ann = OVIS_ROOT + 'annotation/malmm_valid_instances_boxcap_v2.json'
ovis_frames = OVIS_ROOT + 'frames/'

lvvis_boxcap_train_dataset = dict(
    type=VideoBoxCapDataset,
    image_folder=lvvis_train_frames,
    annotation_file=lvvis_train_ann,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    scene_graph_root=lvvis_scene_graph_root,
    scene_graph_split='train',
)

lvvis_boxcap_val_dataset = dict(
    type=VideoBoxCapDataset,
    image_folder=lvvis_val_frames,
    annotation_file=lvvis_val_ann,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    scene_graph_root=lvvis_scene_graph_root,
    scene_graph_split='val',
)

ovis_boxcap_train_dataset = dict(
    type=VideoBoxCapDataset,
    image_folder=ovis_frames,
    annotation_file=ovis_train_ann,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    scene_graph_root=ovis_scene_graph_root,
    scene_graph_split='train',
)

ovis_boxcap_val_dataset = dict(
    type=VideoBoxCapDataset,
    image_folder=ovis_frames,
    annotation_file=ovis_val_ann,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    scene_graph_root=ovis_scene_graph_root,
    scene_graph_split='val',
)

################## image chat
LLAVA_ROOT = DATA_ROOT + 'llava_data/'
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path=LLAVA_ROOT + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder=LLAVA_ROOT + 'llava_images/',
)

################## image res
RES_ROOT = DATA_ROOT + 'ref_seg/'
refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=RES_ROOT + 'refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=RES_ROOT + 'refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root= RES_ROOT + 'refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

# image gcg datas
glamm_data_root = DATA_ROOT + 'glamm_data/'

refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'

grandf_image_path = glamm_data_root + 'images/grandf/train/'
grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'

flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'

psg_image_path = glamm_data_root + 'images/coco2017/'
psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder=refcocog_image_path,
    data_path=refcocog_ann_file,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=psg_ann_file,
    image_folder=psg_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

# # sam2 data
# data_sam2_folder = VIDEO_DATA_ROOT + 'segmentation_datasets/sam_v_full/'
# data_sam2_expression_file = './whole_pesudo_cap_v3/sam_v_final_v3.json'

# video_sam2_dataset = dict(
#     type=VideoSAM2Dataset,
#     sam2_folder=data_sam2_folder,
#     expression_file=data_sam2_expression_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=4,
#     special_tokens=special_tokens,
#     extra_image_processor=extra_image_processor,
#     sampled_frames=5,
#     select_number=5,
# )

# osprey
OSPREY_ROOT = DATA_ROOT + "osprey-724k/"
data_osprey_file = OSPREY_ROOT + 'Osprey-724K/osprey_conversation.json'
data_osprey_image_folders = [
    OSPREY_ROOT + 'coco/train2014/',
    OSPREY_ROOT + 'coco/val2014/',
    OSPREY_ROOT + 'coco/train2017/',
    OSPREY_ROOT + 'coco/val2017/',
]

image_osprey_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_detail_description_file = OSPREY_ROOT + 'Osprey-724K/osprey_detail_description.json'
image_osprey_description_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_detail_description_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_short_file = OSPREY_ROOT + 'Osprey-724K/osprey_short_form.json'
image_osprey_short_dataset = dict(
    type=OspreyShortDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_short_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_part_file = OSPREY_ROOT + 'Osprey-724K/osprey_part_level.json'
image_osprey_part_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_part_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_positive_neg_file = OSPREY_ROOT + 'Osprey-724K/osprey_lvis_positive_negative.json'
image_osprey_positive_neg_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_positive_neg_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # # sem seg
        # # semantic_seg_ade20k_dataset,
        # # ref seg
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # # image qa
        # llava_vqa_dataset,
        # # video res
        video_mevis_dataset, video_revos_dataset, video_refytvos_dataset,
        # # video chat
        # video_qa_dataset,
        video_qa2_dataset,
        lvvis_boxcap_train_dataset,
        ovis_boxcap_train_dataset,
        # # sam2 pesudo
        # # video_sam2_dataset,
        # # gcg data
        # glamm_psg_dataset,
        # glamm_grandf_dataset,
        # glamm_flickr_dataset,
        # glamm_refcocog_dataset,
        # # visual prompt
        # image_osprey_dataset, image_osprey_description_dataset,
        # image_osprey_part_dataset, image_osprey_short_dataset,
        # image_osprey_positive_neg_dataset,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

val_dataset = dict(
    type=ConcatDataset,
    datasets=[
        lvvis_boxcap_val_dataset,
        ovis_boxcap_val_dataset,
    ],
)

val_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=val_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=video_lisa_collate_fn),
    drop_last=False,
)
val_cfg = dict(type=SegCaptionValLoop)
val_evaluator = dict(
    type=SegCaptionMetric,
    iou_threshold=0.5,
    text_sim_threshold=0.5,
)

test_dataloader = val_dataloader
test_cfg = val_cfg
test_evaluator = val_evaluator

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
