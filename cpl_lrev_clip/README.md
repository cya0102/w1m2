# CPL-LREV with CLIP-ViP frame features

这是 CPL-LREV 的 CLIP-ViP 视觉特征版本。模型结构、损失、proposal generator、event disentanglement、训练流程和数据划分沿用当前 CPL-LREV 代码；视觉输入改为 CLIP-ViP 的逐帧特征。

- 视觉特征：每帧 512 维，读取后使用现有 `_sample_frame_features` 聚合为最多 200 帧。
- 文本特征：继续使用项目内的 300 维 GloVe；不使用 CLIP 文本编码器。
- Charades 的词表：`data/charades/glove.pkl`。
- ActivityNet 的词表：`data/activitynet/glove.pkl`。

## 特征文件

默认配置使用以下文件：

```text
Charades-STA:
/data/chenyuan/videogrounding/VGDataset/Charades-STA/clip-vip/clip_vip_b16_per_frame_video_features.hdf5

ActivityNet:
/data/chenyuan/videogrounding/VGDataset/ActivityNet/clip-vip/clip_vip_b16_per_frame_video_features.hdf5
```

两个 HDF5 文件的结构相同：视频 ID 位于根层，`file[video_id]` 直接得到 Dataset，单条数据为 `[T, 512]` 的 `float32` 数组。这里没有 `c3d_features` 子键。加载器会检查视频 ID、时间长度、维度、数据类型转换和有限值；任一数据错误都会给出数据集名称和视频 ID。

## 环境

请使用能够运行源项目的 Python、PyTorch 和 CUDA 环境，并安装：

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev_clip
pip install -r requirements.txt
```

数据集读取依赖 NLTK 的分词和词性标注资源。建议同时安装兼容不同 NLTK 版本的资源：

```bash
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

## 从头训练

旧 I3D/C3D checkpoint 的视觉层维度与本版本的 512 维输入不兼容。本目录不包含旧 checkpoint，下面的命令从随机初始化开始训练。

Charades：

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev_clip
python train.py \
  --config-path config/charades/main.json \
  --log_dir logs/charades \
  --tag cpl_lrev_clip_charades \
  --vote
```

ActivityNet：

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev_clip
python train.py \
  --config-path config/activitynet/main.json \
  --log_dir logs/activitynet \
  --tag cpl_lrev_clip_activitynet \
  --vote
```

也可以执行对应脚本：

```bash
bash scripts/train_charades.sh
bash scripts/train_activitynet.sh
```

`--vote` 与源项目一致，启用推理阶段的 semantic weighted voting。训练期间默认在 validation split 选择 checkpoint，并在训练结束后对 test split 评估 validation-selected checkpoints。

## 验证和测试

训练命令已经包含验证以及最终测试。若要单独评估一个本版本训练得到的 checkpoint，将 `<RUN_DIR>` 替换为实际目录：

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev_clip
python train.py \
  --config-path config/charades/main.json \
  --resume checkpoints/charades/<RUN_DIR>/model-best.pt \
  --eval \
  --log_dir logs/charades \
  --tag eval_cpl_lrev_clip_charades \
  --vote
```

ActivityNet 使用对应的 `config/activitynet/main.json`、`checkpoints/activitynet/<RUN_DIR>/model-best.pt` 和 `logs/activitynet`。`--resume` 只适用于本版本结构和 512 维视觉输入兼容的 checkpoint；不要加载旧 I3D/C3D checkpoint。

ActivityNet 单独测试命令如下：

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev_clip
python train.py \
  --config-path config/activitynet/main.json \
  --resume checkpoints/activitynet/<RUN_DIR>/model-best.pt \
  --eval \
  --log_dir logs/activitynet \
  --tag eval_cpl_lrev_clip_activitynet \
  --vote
```

输出位置为：

```text
checkpoints/charades/<tag>_<timestamp>/
checkpoints/activitynet/<tag>_<timestamp>/
logs/charades/<tag>_<timestamp>.log
logs/activitynet/<tag>_<timestamp>.log
```

checkpoint 路径由两个主配置中的 `train.model_saved_path` 分开设置；日志路径由命令行的 `--log_dir` 指定。运行目录应为本目录，以便配置中的 `data/...` 相对路径正确解析。

## 关键维度

```text
Charades:    frame_dim=512, frames_input_size=512, word_dim=300, words_input_size=300
ActivityNet: frame_dim=512, frames_input_size=512, word_dim=300, words_input_size=300
```

`max_num_frames=200` 和原配置中的其他模型、损失、优化器、proposal、事件解耦、训练超参数均保持不变。

## 目录

```text
models/       CPL-LREV 模型及模块
datasets/     Charades-STA / ActivityNet 数据加载
runners/      训练、验证、测试和 checkpoint 选择
optimizers/   优化器和学习率调度器
config/       两个主配置
data/         标注 JSON 和 300 维 GloVe 词表
tests/        源项目单元测试副本
tools/        checkpoint 诊断工具
scripts/      从头训练脚本
```
