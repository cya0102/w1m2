### <p align="center">Scene Graph-guided SegCaptioning Transformer with Fine-grained Alignment for Controllable Video Segmentation and Captioning
<br>
<div align="center">
  Xu&nbsp;Zhang</a> <b>&middot;</b>
  Jin&nbsp;Yuan</a> <b>&middot;</b>
  Binhong&nbsp;Yang</a> <b>&middot;</b>
  Xuan&nbsp;Liu</a> <b>&middot;</b>
  Qianjun&nbsp;Zhang</a> <b>&middot;</b>
  Yuyi&nbsp;Wang</a> <b>&middot;</b>
  Zhiyong&nbsp;Li</a> <b>&middot;</b>
  Hanwang&nbsp;Zhang</a>
  <br> <br>
</div>



### Abstract
Recent advancements in multimodal large models have significantly bridged the representation gap between diverse modalities, catalyzing the evolution of video multimodal interpretation, which enhances users' understanding of video content by generating correlated modalities. However, most existing video multimodal interpretation methods primarily concentrate on global comprehension with limited user interaction. To address this, we propose a novel task, Controllable Video Segmentation and Captioning (SegCaptioning), which empowers users to provide specific prompts, such as a bounding box around an object of interest, to simultaneously generate correlated masks and captions that precisely embody user intent. An innovative framework Scene Graph-guided Fine-grained SegCaptioning Transformer (SG-FSCFormer) is designed  that integrates a Prompt-guided Temporal Graph Former to effectively captures and represents user intent through an adaptive prompt adaptor, ensuring that the generated content well aligns with the user’s requirements. Furthermore, our model introduces a Fine-grained Mask-linguistic Decoder to collaboratively predict high-quality caption-mask pairs using a Multi-entity Contrastive loss, as well as provide fine-grained alignment between each mask and its corresponding caption tokens, thereby enhancing users' comprehension of videos. Comprehensive experiments conducted on two benchmark datasets demonstrate that SG-FSCFormer achieves remarkable performance, effectively capturing user intent and generating precise multimodal outputs tailored to user specifications.



## SG-FSCFormer model

![SG-FSCFormer](assets/network.jpg)


## Environment
Training and evaluation environment: Python 3.10, PyTorch 2.3.1, CUDA 11.8. Run the following command to install required packages.
```
pip install -r requirements.txt
```

Optional caption metrics use the COCO caption toolkit. If it is not already
available in the environment, install it from the bundled `coco-caption`
directory.

## Pretrained Models

Place pretrained weights under `pretrained/`:

- [vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)
- [sam2_hiera_large.pt](https://huggingface.co/facebook/sam2-hiera-large)


The default launcher uses `./pretrained/vicuna-7b`. InternVL2_5-4B can be
selected without code changes.

## Data Preparation

The default config expects LVVIS/OVIS-style SegCaption annotations and extracted
scene graph features:

```text
data/
  lvvis/
    annotation/
    frames/
  ovis/
    annotation/
    frames/
  custom_data/
    lvvis/annotation/
    ovis/annotation/
```

Scene graph object boxes and box features should be pre-extracted into the
dataset-specific `custom_data/*/annotation/` directories. 
You can download them from the following [link](https://drive.google.com/drive/folders/1xT03q0jq9e3D-p6yDEUY8i_S3chE6un9?usp=sharing).

## Training

Single-node multi-GPU training:

```bash
bash run_train.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 GPUS=2 bash run_train.sh
MODEL_PATH=./pretrained/InternVL2_5-4B bash run_train.sh
```


Equivalent direct command:

```bash
bash tools/dist.sh train projects/llava_sam2/configs/sg_fscformer.py 2
```

## Validation and Evaluation

The config validates once per epoch. Reported metrics include:

- Captioning: METEOR, CIDEr, and SPICE when COCOEvalCap is available.
- Video segmentation: J, F, and J&F.
- Cross-modal alignment: class-level AP and instance-level mAP.


Run evaluation on the configured LVVIS/OVIS validation sets with the default
Vicuna-7B language model:

```bash
CHECKPOINT=work_dirs/sg_fscformer/xxx.pth bash run_eval.sh
```


Multi-GPU evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=./pretrained/vicuna-7b \
bash tools/dist.sh test projects/llava_sam2/configs/sg_fscformer.py 2 \
  --checkpoint work_dirs/sg_fscformer/xxx.pth \
  --work-dir work_dirs/sg_fscformer_eval
```

## Demo

After converting or preparing an SG-FSCFormer HuggingFace checkpoint:

```bash
MODEL_PATH=./pretrained/sg_fscformer bash run_demo.sh
```

## Model Export

```bash
PTH_MODEL=work_dirs/sg_fscformer/xxx.pth \
SAVE_PATH=pretrained/finetune_models/sg_fscformer \
bash run_convert_cpk.sh
```

<!-- ## Notes

- This codebase builds on SAM2, Vicuna/InternVL, and related
  open-source components. -->
