# SFTCap
Dual-Hierarchical Knowledge Distillation for Video Captioning

## Work Introduction
In this work， we propose CapDistill, a dual hierarchical distillation framework that transfers semantic knowledge from a powerful teacher model to a lightweight student model.

✨ The framework will be shown later~ 

By releasing this code, we hope to stimulate further research and development in lightweight video captioning. If you find this work useful in your own research, please consider citing our paper as a reference.

## Environment
Clone and enter the repo:

```shell
git clone https://github.com/ccc000-png/SFTCap.git
cd SFTCap
```

We has refactored the code and tested it on: 
- `Python` 3.9
- `torch` 1.13.1

Please change the version of torch and cuda according to your hardwares.

```shell
conda create -n SFTCap python==3.9
conda activate SFTCap

# Install a proper version of torch, e.g.:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

```

## Feature Preprocessing
**1. Pre-trained CLIP** (please refer [README_PRETRAINED.md](/model_zoo/README.md))

The features of our model are extracted using pre-trained CLIP. To avoid network issues, we recommend that you download the pre-trained model in advance and put it in the `/model_zoo/clip_model` folder.

**2. Supported datasets**
- `MSVD`
- `MSRVTT`

You can download our preprocessed data from [One Drive](https://1drv.ms/u/c/00263e242b1aba9a/ERHsLCd1i9JAu7scJ7tjCdwBREjiNziEWpoO1yuGQNe1_A?e=YWs9hW), which follows the structure below:

```
└── data
    ├── msrvtt
    │   ├── language
    │   │   └── msrvtt_caption.json
    │   ├── splits
    │   │   ├── msrvtt_test_list.pkl
    │   │   ├── msrvtt_train_list.pkl
    │   │   └── msrvtt_valid_list.pkl 
    │   ├── ... 
    │   └── visual
    │      ├── clip_b16
    │      │   └── frame_feature
    │      │      ├── ... 
    │      │      └── xxx.npy
    │      ├── clip_b32
    │      │   └── frame_feature
    │      │      ├── ... 
    │      │      └── xxx.npy
    │      └── clip_l14
    │         └── frame_feature
    │            ├── ... 
    │            └── xxx.npy    
    ├── msvd
    │   ├── language
    │   │   └── msrvtt_caption.json
    │   ├── splits
    │   │   ├── msrvtt_test_list.pkl
    │   │   ├── msrvtt_train_list.pkl
    │   │   └── msrvtt_valid_list.pkl 
    │   ├── ... 
    │   └── visual
    │      ├── clip_b16
    │      │   └── frame_feature
    │      │      ├── ... 
    │      │      └── xxx.npy
    │      ├── clip_b32
    │      │   └── frame_feature
    │      │      ├── ... 
    │      │      └── xxx.npy
    │      └── clip_l14
    │         └── frame_feature
    │            ├── ... 
    │            └── xxx.npy 
    └── ...
```
You can also download raw videos for feature processing from our shared links. Please organize them as follows:
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Official Link</th>
    </tr>
    <tr align="center">
        <td>MSVD</td><td><a href="https://www.cs.utexas.edu/users/ml/clamp/videoDescription/">Link</a></td>
    </tr>
    <tr align="center">
        <td>MSRVTT</td><td><a href="http://ms-multimedia-challenge.com/2016/dataset">Link (expired)</a></td>
    </tr>
</table>
</div>

```
└── data
    ├── msrvtt
    │   └── raw_videos
    │       ├── video0.avi
    │       ├── ...
    │       └── video9999.avi
    ├── msvd
    │   └── raw_videos
    │       ├── video0.avi
    │       ├── ...
    │       └── video1969.avi
    └── ...
```
**Note:** 
- The original names of MSVD videos do not follow the format `videoXXX`, It is recommended that you use the official name in the dataset directly. And you can follow [README_DATA.md](/preprocess/README.md) to process data.


### Training
You can use the following commands to run the experiments:

- If you run the experiments on `MSRVTT`:
```
# msrvtt
python main.py --clip_name clip_b32 --dataset msrvtt --init_method tcp://localhost:2223 --learning_rate 2e-4 --train_type sft --max_objects 3 --bsz 64 --save_checkpoints_every 100 --max_epochs 20 --max_caption_len 22 --T_loss 5 --lam_o 0 --lam_a 0 --SFT --use_ham --use_module transformer
```
