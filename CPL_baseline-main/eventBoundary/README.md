# Charades / ActivityNet 事件边界计算

这里实现了论文 *Mismatched Pairs Dynamic Correction for Cross-Modal
Alignment in Video Moment Retrieval* 中 Sec. III-E.1（PDF 第 5–6 页，印刷页
12286–12287，Eq. (6)）的事件边界计算，并补齐了论文正文未说明的端点处理和
输出约定。

核心实现只依赖 NumPy；数据集批处理额外依赖 h5py。

## 算法

给定一个视频的 clip 特征 `F`，形状为 `[T, D]`：

1. 对每个 clip 特征做 L2 归一化，得到 `X`。
2. 构造余弦 Temporal Self-similarity Matrix：`S = X @ X.T`。
3. 用论文 Eq. (6) 的固定 5×5 核在 TSM 主对角线上卷积：

   ```text
    1  1  0 -1 -1
    1  1  0 -1 -1
    0  0  0  0  0
   -1 -1  0  1  1
   -1 -1  0  1  1
   ```

4. 以当前视频所有 raw score 的均值为阈值，去掉低于均值的位置。
5. 使用 size=3 的滑动最大滤波，仅保留不小于左右相邻位置的局部峰值。
6. 参考该论文直接引用的 [EaTR 官方实现](https://github.com/jinhyunj/EaTR)，
   强制把第一个和最后一个有效 clip 作为事件分隔点。

核矩阵可写成 `g g^T`，其中 `g=[1,1,0,-1,-1]`。因此主对角线上第 `t`
个分数在数学上等价于：

```text
b[t] = ||x[t-2] + x[t-1] - x[t+1] - x[t+2]||²
```

越界 clip 取零。代码利用这个恒等式把计算从显式 TSM 的 `O(T²)` 内存降为
`O(TD)`；测试会将优化结果与显式 `TSM + zero padding + 2-D convolution`
逐项比较，并允许不同浮点归约顺序产生的微小舍入误差。

批处理始终按单视频的真实有效长度计算均值和峰值，不把 batch padding 纳入阈值，
也不会跨视频连接边界。这符合论文中 `S ∈ R^(L×L)` 的单视频定义，同时避免
EaTR 批量参考代码中 padding 和全 batch `roll` 带来的实现瑕疵。

## 特征选择

默认 `--feature-source config` 读取 CPL 配置中的特征：

- Charades-STA：I3D，根节点 `/<video_id>`；
- ActivityNet Captions：C3D，节点 `/<video_id>/c3d_features`。

默认还会完全复用 CPL 的 `BaseDataset._sample_frame_features`，把每个视频变成
配置中的 `max_num_frames=200` 个 clip。这样输出的边界索引可以直接与 CPL
模型输入的 200 个时间步对齐。

论文实验中的视觉特征设置不同：ActivityNet 使用 CLIP，Charades 分别报告
VGG 和 SlowFast+CLIP。当前机器已有两个完整的 CLIP ViT-B/16、3 FPS HDF5，
可通过 `--feature-source clip` 使用。需要注意，Charades 的 CLIP-only 并不等同
于论文的 SlowFast+CLIP 拼接特征；如已有严格一致的 VGG 或 SF+C HDF5，请用
`--feature-path` 指定。

边界取决于视觉特征，所以不同 backbone 或不同采样长度得到不同边界是正常的。

## 使用方法

在 `CPL_baseline-main` 下运行。命令不依赖当前工作目录，下面只是最简写法。

按 CPL 当前配置计算两个数据集，并分别输出到 `eventBoundary/outputs/`：

```bash
python eventBoundary/compute_event_boundaries.py --dataset all
```

只计算一个数据集：

```bash
python eventBoundary/compute_event_boundaries.py --dataset charades
python eventBoundary/compute_event_boundaries.py --dataset activitynet
```

使用本机已有的 CLIP 特征，并在 HDF5 原始 3 FPS 序列上计算：

```bash
python eventBoundary/compute_event_boundaries.py \
  --dataset all \
  --feature-source clip \
  --native-length
```

使用自定义特征：

```bash
python eventBoundary/compute_event_boundaries.py \
  --dataset charades \
  --feature-path /absolute/path/to/slowfast_clip_features.hdf5 \
  --native-length \
  --output eventBoundary/outputs/charades_sfc_event_boundaries.json
```

调试时只处理前 10 个视频并保留完整 score 向量：

```bash
python eventBoundary/compute_event_boundaries.py \
  --dataset activitynet \
  --limit 10 \
  --include-all-scores \
  --pretty \
  --output /tmp/activitynet_boundary_debug.json
```

已有输出默认不会被覆盖；确认需要替换时添加 `--overwrite`。完整参数见：

```bash
python eventBoundary/compute_event_boundaries.py --help
```

## 处理范围

默认从选定的 annotation splits 中抽取 video id，再按视频去重计算，而不是按
query 行重复计算：

- Charades-STA：6,067 个标注视频；
- ActivityNet Captions：14,926 个标注视频。

`--splits train val` 可限制 split；`--all-feature-videos` 可改为处理 HDF5 中的
所有根节点。后者包含无标注视频，因此相应记录没有 `duration` 和秒数坐标。

## 输出格式

顶层包含 `metadata` 和按 video id 索引的 `videos`。每个视频记录的关键字段为：

```json
{
  "source_num_clips": 101,
  "num_clips": 200,
  "threshold": 0.42,
  "boundary_indices": [0, 47, 103, 199],
  "detected_boundary_indices": [47, 103],
  "boundary_scores": [0.12, 2.81, 3.07, 0.09],
  "boundary_positions_normalized": [0.0, 0.236, 0.518, 1.0],
  "segmentation_cuts_indices": [0, 47, 103, 200],
  "event_intervals_indices_half_open": [[0, 47], [47, 103], [103, 200]],
  "event_spans_normalized_cw": [[0.1175, 0.235], [0.375, 0.28], [0.755, 0.48]],
  "pooling_event_spans_normalized_cw": [[0.1175, 0.235], [0.375, 0.28], [0.7575, 0.485]],
  "duration": 33.67,
  "boundary_times_seconds": [0.0, 7.95, 17.43, 33.67],
  "event_intervals_seconds": [[0.0, 7.91], [7.91, 17.34], [17.34, 33.67]]
}
```

字段约定：

- `boundary_indices` 包含官方参考实现强制加入的 `0` 和 `T-1`；
- `detected_boundary_indices` 只包含由分数阈值和局部最大滤波检测到的内部峰值；
- `segmentation_cuts_indices` 使用 `[0, ..., T]`，由它形成的事件是完整覆盖且不
  重叠的半开区间 `[start, end)`，适合直接做 mean pooling；
- `event_spans_normalized_cw` 是按相邻 `[0, ..., T-1]` 边界坐标并除以
  `T` 得到的论文/EaTR 形式；
- `pooling_event_spans_normalized_cw` 是完整覆盖半开区间的归一化
  `(center, width)`，适合无重叠 mean pooling；
- `boundary_scores` 只保存选中位置的原始分数。添加 `--include-all-scores` 后才
  保存长度为 `T` 的 `all_boundary_scores`；
- 论文未规定 clip 到秒的精确映射。边界点秒数采用
  `index/(T-1)*duration`，半开事件区间采用 `cut/T*duration`，两种约定均写在
  metadata 中，避免静默混用。

## 测试

```bash
python -m unittest discover -s eventBoundary/tests -v
```

测试覆盖：固定核、优化分数与显式 TSM 卷积的数值等价性、局部峰值、事件区间
完整覆盖、CPL 采样一致性，以及 Charades/ActivityNet 两种 HDF5 布局。
