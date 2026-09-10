# QCEC / CACMI 独立工程详细实施方案

> 对应原方案：[`proposal_aaai26_cacmi.md`](proposal_aaai26_cacmi.md)  
> 对应论文：*Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction*（AAAI 2026）  
> 参考基线（只读）：`/data/chenyuan/videogrounding/w1m2/cpl_lrev`  
> 唯一实施目录：`/data/chenyuan/videogrounding/w1m2/qcec`  
> 文档性质：implementation specification，不在本文件中实施模型代码  
> 最后核对日期：2026-09-10

## 1. 目标、边界与推荐实现结论

本方案保留原 proposal 的研究意图：在高时间分辨率的视频特征上构造显式、连续的事件簇，用当前 query 判断哪些簇与目标有关，并把查询相关的事件结构送给从基线复制到 QCEC 工程内的高斯候选生成器；训练时抑制候选跨越明显无关的事件变化，推理时可把粗边界吸附到可信的簇边界。

工程实现采用以下主路径：

1. 对固定的 C3D/I3D 输入特征离线执行“只允许合并相邻片段”的 Ward 聚类，为每个视频保存 `M=32` 个连续簇的成员关系和归一化边界。聚类结果是无梯度结构先验。
2. 在 `CPL.forward` 内，使用这些固定成员关系对 `frame_fc` 后、下采样前的 200-step 特征进行边界增强池化。池化对帧特征保留梯度，但对簇分配不求梯度。
3. 从现有文本分支的上下文化 query states 中池化出谓词、实体、整句三个 query unit；不使用外部 caption 库，也不引入训练阶段独有的信息。
4. 计算每个簇的三个提示：相关性 `r`、左边界提示 `b^L`、右边界提示 `b^R`。三个提示先汇聚成一个 `D` 维全局 QCEC 表示，通过零初始化残差 adapter 修改现有的 `proposal_generator_feature = h[:, -1]`。
5. 可选地把相关性最高的 `N` 个簇编码成 proposal-slot bias，加到现有中心/宽度 head 的 logits 上。该路径仍由原 `GaussianMixtureProposalGenerator` 输出高斯参数，不替换原生成器。
6. 新增 `qcec_coherence_loss`，惩罚候选从 query 相关侧跨过强变化点伸入明显不相关侧。障碍权重默认 `detach`，避免提示分支通过把所有边界置信度降为零来逃避损失。
7. 推理阶段可选做 `±2` 个高分辨率 frame unit 范围内的边界吸附。默认先关闭，仅在 feature fusion 和 crossing loss 被验证有效后开启。

### 1.1 独立工程目录的强制约束

后续实现必须创建一个完整、可独立运行的工程目录：

```text
/data/chenyuan/videogrounding/w1m2/qcec
```

所有新增或修改后的代码、配置、脚本、测试、文档和 QCEC 预处理工具都必须位于该目录下。`cpl_lrev/` 只作为本 specification 的只读参考基线，不是 QCEC 工程运行时调用的子模块。

必须满足：

- 先把运行基线所需的源码结构复制到 `qcec/`，再只修改 `qcec/` 内的副本；
- `qcec/` 必须拥有自己的 `train.py`、`models/`、`datasets/`、`runners/`、`optimizers/`、`tools/`、`tests/`、`config/`、`scripts/`、`utils.py`、`vocab.py`、`requirements.txt` 和 `README.md`；
- 禁止在 `qcec/` 代码中使用 `from cpl_lrev...`、`import cpl_lrev...`、把 `../cpl_lrev` 加入 `sys.path/PYTHONPATH`，或在运行时动态读取 `cpl_lrev` 源文件；
- 禁止用软链接把 `qcec/models`、`qcec/datasets` 等指向 `cpl_lrev/`；QCEC 的源码必须是真实文件；
- 从项目根目录或 `qcec/` 运行时，模型、dataset、trainer、evaluation 和工具脚本只能解析到 `qcec/` 内的 Python 文件；
- 外部 HDF5 视频特征仍可通过 config 指向共享数据集路径；这属于数据依赖，不属于对 `cpl_lrev` 代码的调用；
- baseline checkpoint 可以作为一次性初始化输入，但推荐复制到 `qcec/checkpoints/bootstrap/` 后使用。QCEC 工程不能依赖 `cpl_lrev/checkpoints/...` 才能完成常规训练或评估；
- 实现过程中不得修改 `/data/chenyuan/videogrounding/w1m2/cpl_lrev` 下的任何文件。

本文后续出现的路径按以下规则理解：

- 第 2 节中的 `cpl_lrev/...` 是对现有基线的事实分析；
- 第 5、10、11、12、16 节中的实施路径均以 `qcec/...` 为准；
- 若文中讨论“原模型”或“原生成器”，是指已经复制进 `qcec/` 的基线实现，不表示跨目录 import 或调用。

### 1.2 必须修正的原方案表述

原 proposal 将“200 帧被下采样到 50”概括成候选生成器只接收 50-step 表示。当前代码并非如此：

- `cpl_lrev/models/cpl.py::CPL.forward` 先给 200 个视频特征追加一个 `pred_vec`，得到 201 个 token；
- `self.trans(..., decoding=1)` 在完整 201-token 序列上产生 `h`；
- `proposal_generator_feature = h[:, -1]` 随后直接预测中心和宽度；
- 200→50 的点采样发生在这之后，主要服务高斯掩码下的文本重构、负候选和 LREV。

因此 QCEC 的真实插入点仍应位于高分辨率分支，但“把 50-step `h` 与提示拼接后送入生成器”无法直接对应当前接口。推荐改成：

```text
200-step projected video features + contextualized query states
    -> QCEC cluster interaction
    -> qcec_global_feature [B, D]
    -> residual adapter(h[:, -1], qcec_global_feature)
    -> 原中心/宽度 head
```

这属于对原方案的工程校正，不改变“事件簇提示进入原候选生成器”的核心算法。

### 1.3 原始方案、推荐实现与可选增强的边界

| 层级 | 内容 |
|---|---|
| 原始方案 | 连续聚类、边界增强池化、query 条件的 `r/b^L/b^R`、候选初始化偏置、跨变化约束、推理边界吸附 |
| 推荐首个完整实现 | 离线连续 Ward 簇 + 在线可微池化 + 三类 query unit + 全局 residual adapter + `L_cross`；保留 slot bias 接口但用独立开关控制 |
| 可选改进 | 在线动态聚类、component-level crossing loss、边界对齐奖励、QCEC 分数参与候选排序、软聚类或可微聚类 |

不要在首轮实现中加入外部句子库、额外 VLM、caption 检索、mutual-information 对比学习或新的 proposal 数量；这些内容不是当前 QCEC 假设的必要条件。

## 2. 当前仓库的真实执行链

## 2.1 入口、模型构建和配置传递

训练/评估入口是 `cpl_lrev/train.py`：

1. `parse_args()` 读取 `--config-path`、checkpoint、候选选择策略和少量 loss override。
2. `main()` 用 `utils.load_json()` 读取 JSON 配置，将 CLI loss override 写入 `args['loss']`。
3. `MainRunner(args)` 在 `cpl_lrev/runners/main_runner.py` 中构建 dataset、model、optimizer 和 scheduler。
4. `MainRunner.__init__()` 将数据集词表大小写入 `args['model']['config']['vocab_size']`，并将最大 epoch 写入 `max_epoch`。
5. `MainRunner._build_model()` 通过 `getattr(models, model_config['name'])` 构造 `models.CPL`，随后直接调用 `.cuda()`。

当前没有 dataclass、配置 schema 或注册式 loss builder；新增 QCEC 配置应沿用嵌套字典和 `.get(default)`，并在模块构造函数中显式检查非法值。

## 2.2 Dataset、采样和 batch 形状

核心代码：

- `cpl_lrev/datasets/base.py::BaseDataset.__getitem__`
- `cpl_lrev/datasets/base.py::BaseDataset._sample_frame_features`
- `cpl_lrev/datasets/base.py::build_collate_data`
- `cpl_lrev/datasets/activitynet.py::ActivityNet`
- `cpl_lrev/datasets/charades_sta.py::CharadesSTA`

当前数据流如下：

```text
HDF5 中的可变长 C3D/I3D feature
    -> _sample_frame_features()
    -> 固定 max_num_frames=200 个均值池化 clip
    -> collate 为 frames_feat [B, 200, D_v], float32

sentence
    -> NLTK tokenize + POS tag
    -> 丢弃词表外词
    -> GloVe words_feat [B, W+1, 300], float32
    -> words_id [B, W], int64
    -> words_len [B], integer tensor
    -> weights [B, W], float32
```

ActivityNet 的 `D_v=500`，Charades-STA 的 `D_v=1024`。当前 `_sample_frame_features()` 无论原始视频 feature 有多少步，都会返回恰好 200 步，因此实际 `frames_len` 通常都是 200。ActivityNet 训练数据约 37,421 条 query，但只有约 10,009 个不同视频，说明按 query 在 forward 内重复聚类会浪费大量计算。

`raw=[vid, duration, timestamps, sentence]` 不会送入模型；GT timestamp 只在 `MainRunner.eval()` 中计算指标，训练 forward 和 loss 看不到 GT。

## 2.3 `CPL.forward` 的实际形状与调用顺序

核心文件：`cpl_lrev/models/cpl.py`。

令：

- `B`：batch size；
- `T=200`：输入视频步数；
- `D_v`：原视频特征维度；
- `D=256`：hidden size；
- `W<=20`：有效 query token 数；
- `N`：proposal 数，ActivityNet 当前为 5；
- `T_p=T//4=50`：重构分支的视频步数；
- `K_total=1+2+...+5=15`：ActivityNet mixture 总分量数。

现有 forward 为：

1. 给 `frames_feat [B,T,D_v]` 追加 `pred_vec [B,1,D_v]`。
2. dropout 后经 `frame_fc`，得到 `[B,T+1,D]`。
3. `_generate_mask(..., frames_len)` 产生 `[B,T+1]` byte mask。因为长度仍为 200，最后的 prediction token 被标成 key padding；但它仍作为 query token，经 causal self-attention 汇聚前面的帧。
4. 将 `words_feat[:,0]` 替换为 `start_vec`，加位置编码并经 `word_fc`。
5. `self.trans(..., decoding=1)` 先对文本做自注意力，再让视频 token 与文本交互，返回：
   - `enc_out [B,W+1,D]`：文本上下文表示；
   - `h [B,T+1,D]`：query-conditioned 视频表示。
6. `proposal_generator_feature=h[:,-1] [B,D]`。
7. single-Gaussian 路径由 `fc_gauss` 直接输出 `[B*N,2]`；mixture 路径稍后调用 `GaussianMixtureProposalGenerator.predict_components()`。
8. 从前 200 个 projected frame 中用 `linspace` 取 50 个位置，得到重构用 `frames_feat [B,T_p,D]`。
9. mixture 路径先预测 component centers/widths/masks，再逐 component 重构 query，使用 reconstruction summary 计算 component importance，最终合并成：
   - `gauss_weight [B*N,T_p]`；
   - `center [B*N]`；
   - `width [B*N]`；
   - padded component tensors `[B,N,K_max,...]`。
10. proposal mask 参与第二次 `self.trans(..., decoding=2)`，输出 `words_logit [B*N,W,V]`。
11. 若启用 negative/LREV，再计算负重构、reference 重构、event vectors 和 event score。

QCEC 应使用第 2 步之后、下采样之前的前 `T` 个 frame states，并使用第 5 步的 `enc_out[:,1:]`。它不得依赖后面的随机 word masking，否则训练和推理中的 query 条件结构会被随机遮蔽。

## 2.4 Gaussian mixture 候选生成器

核心文件：`cpl_lrev/models/modules/gaussian_mixture.py`。

`GaussianMixtureProposalGenerator` 当前包含：

- `center_head: Linear(D, K_total)`；
- `width_head: Linear(D, N)`；
- 每个 proposal 内所有 component 共享一个 width；
- `predict_components(global_feature, sequence_length)` 返回扁平中心、宽度和 component Gaussian mask；
- `combine(...)` 使用 component reconstruction feature 预测 importance，并生成 mixture mask；
- `boundary_mode='outer'` 时以 component 最外包络作为最终区间；`'weighted'` 时使用 importance 加权的左右边界；
- negative mining 始终可使用未收缩的 outer envelope。

QCEC 不应复制或替换该类。只需要给 `predict_components()` 增加可选的 center/width logit bias；参数为 `None` 时执行路径和数值必须与现状一致。

## 2.5 现有 loss 和训练 loop

`cpl_lrev/models/loss.py` 当前有四组 loss：

- `rec_loss()`：每个样本只使用最小 proposal NLL；
- `ivc_loss()`：reference/negative ranking 和旧 Gaussian diversity；
- `event_disentanglement_loss()`：LREV/BECL；
- `mixture_pull_push_loss()`：mixture 内外几何约束。

`MainRunner._train_one_epoch()` 的调用顺序为 forward → 四组 loss 相加 → `backward()` → gradient clipping 10 → optimizer step → scheduler step。QCEC loss 应作为第五个独立函数加入，保持其他 loss 不变，便于单独关闭和消融。

## 2.6 当前推理和评估不是 `generate()`

仓库没有独立的 `generate()` 方法，也没有自回归生成 caption 的推理 API。`MainRunner.eval()` 在 `torch.no_grad()` 下调用与训练相同的 `CPL.forward()`，仍然利用已知 query 做 masked reconstruction，然后：

1. 用 `cal_nll_loss()` 得到每个候选的 reconstruction NLL；
2. 可选减去 LREV event score；
3. 按 proposal score 排序；
4. 将 `center/width` 变为 `[start,end]`；
5. 用 `nll`、`geometric_vote` 或 `semantic_vote` 选择 Rank-1；
6. 计算 R@1/R@5 的 mIoU 和 IoU@0.1/0.3/0.5/0.7/0.9。

QCEC 的 inference 设计必须接到该路径，而不是假设存在未实现的 `generate()`。边界吸附只改变用于定位评估的 center/width，不应改变 proposal 数量或 query reconstruction decoder。

## 2.7 Checkpoint 现状

`MainRunner._save_model()` 当前保存：

```python
{
    "num_updates": self.num_updates,
    "config": self.args,
    "model_parameters": self.model.state_dict(),
}
```

重要限制：

- `_load_model()` 使用 strict `load_state_dict`，只适合结构完全一致的 checkpoint；
- `_load_pretrained_model()` 专用于 V3→V4，会主动跳过 `event_disentangler.*`，不适合从当前完整 baseline warm-start QCEC；
- optimizer state 没有保存；
- checkpoint 没有明确保存完成的 epoch，`train()` 也总是从 epoch 1 循环，因此当前 `--resume` 不是严格意义上的连续训练；
- 当前命令使用“当前 config 构建模型 + checkpoint 权重”，不会自动用 checkpoint 内 config 重建模型。

QCEC 需要单独设计 baseline warm-start 入口，详见第 9 节。

## 3. QCEC 数学定义与代码变量

## 3.1 离线连续 Ward 聚类

对每个视频采样后的固定视觉特征记为：

\[
X=[x_1,\ldots,x_T]\in\mathbb{R}^{T\times D_v}.
\]

先做 L2 normalization。初始时每一帧是一个簇，只允许合并时间上相邻的两个簇。相邻簇 `A,B` 的 Ward 代价为：

\[
\Delta(A,B)=\frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|_2^2.
\]

每次合并当前代价最小的相邻簇，直到剩余 `M` 个簇。因为只合并相邻簇，每个簇天然是一个连续区间，不会出现一个簇包含两个分离时间段。

推荐在独立工程中实现 `qcec/tools/build_qcec_cluster_index.py`，使用带双向链表和最小堆的 NumPy 算法，而不是引入当前 requirements 中不存在的 scikit-learn/scipy。每次合并后只更新新簇与左右邻居的代价；复杂度约为 `O(T log T * D_v)`，并以视频而非 query 为单位执行一次。

推荐函数接口：

```python
def contiguous_ward_cluster(
    features: np.ndarray,       # [T, D_v], float32
    num_clusters: int,
    l2_normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cluster_ids [T], bounds [M,2], valid_mask [M]."""
```

返回值约定：

- `cluster_ids [T]`：`int16` 或 `int64`，有效帧取 `0..K-1`，padding 取 `-1`；
- `cluster_bounds [M,2]`：`float32`，按时间排序；若簇覆盖离散索引 `[a,b)`，保存 `[a/T,b/T]`；
- `cluster_mask [M]`：`bool`；
- 当有效 `T<M` 时只生成 `K=T` 个簇并 padding；当前数据通常 `T=200>M`。

预计算索引建议保存为压缩 NPZ：

```text
video_ids        [V]       unicode/string
cluster_ids      [V,T]     int16
cluster_bounds   [V,M,2]   float32
cluster_mask     [V,M]     bool
metadata_json    scalar     sampling/config metadata
```

`metadata_json` 至少记录 `num_clusters`、`max_num_frames`、feature path、是否 L2 normalize、采样算法版本。Dataset 初始化时必须校验这些字段与当前配置一致，避免 100/200 帧索引混用。

### 为什么推荐离线分配、在线池化

原 proposal 写的是在 `frame_fc` 后无梯度聚类。直接照做有三个问题：

1. `frame_fc` 会训练，簇分配随 epoch 跳变；
2. 当前 forward 在输入上先做 dropout，若直接聚类会使同一视频的簇随机变化；
3. 同一视频对应多个 query，在线重复 Ward 聚类成本高。

推荐离线聚类固定 C3D/I3D feature，只固定“哪些帧属于同一簇”；模型内仍使用 `frame_fc` 后的特征计算簇 token，因此语义表示和后续交互可训练。这与 CACMI 在固定预训练视觉 feature 上发现 pseudo-event 的机制一致。

在线 `frame_fc` 后聚类可作为后续 ablation，不能作为首版默认实现。

## 3.2 边界增强的可微簇池化

模型收到：

- `frame_states F [B,T,D]`，`float32/float16`，与模型同 device；
- `frame_mask [B,T]`，`bool`；
- `cluster_ids [B,T]`，`long`；
- `cluster_bounds [B,M,2]`，与 `F` 同 dtype/device；
- `cluster_mask [B,M]`，`bool`。

对簇 `m` 内的帧 `t`，令局部归一化位置为 `u_t in [0,1]`，使用简单且稳定的边界增强权重：

\[
\omega_t=1+\beta\left(2|u_t-0.5|\right)^p,
\]

默认 `beta=1.0, p=2.0`。簇中心权重约为 1，首尾权重约为 2。簇 token 为：

\[
z_m=\frac{\sum_{t:c_t=m}\omega_t F_t}
{\sum_{t:c_t=m}\omega_t+\epsilon}.
\]

实现使用 `scatter_add_` 或 one-hot/einsum。推荐避免构造 `[B,M,T,D]` 大张量；可以构造轻量 assignment `[B,T,M]` 后用 `bmm`，或对展平的 `B*M` index 做 `scatter_add_`。

梯度约定：

- `cluster_ids/bounds/mask/omega` 不需要 gradient；
- `z_m` 对 `frame_states` 保留 gradient；
- reduction 累加在 AMP 下建议临时转 `float32`，输出再转回输入 dtype；
- 分母必须 `clamp_min(1e-6)`；padding frame 不能进入分子或分母。

推荐接口：

```python
def boundary_enhanced_pool(
    frame_states: torch.Tensor,      # [B,T,D]
    frame_mask: torch.Tensor,        # [B,T], bool
    cluster_ids: torch.Tensor,       # [B,T], long
    cluster_bounds: torch.Tensor,    # [B,M,2]
    cluster_mask: torch.Tensor,      # [B,M], bool
) -> torch.Tensor:                   # [B,M,D]
```

## 3.3 三个 query unit

当前 dataset 已经执行 POS tagging，但只把 POS 用来产生 masking probability，之后丢弃标签。QCEC 需要在 `BaseDataset.__getitem__()` 中为保留在词表中的 token 同步构造三个 mask：

1. `predicate`：POS 以 `VB` 开头的词，以及相邻/保留的 `RB` 修饰词；
2. `entity`：POS 以 `NN` 开头的词，以及 `JJ` 修饰词；
3. `sentence`：所有有效 token。

Dataset 返回 `query_role_mask [3,W_i]` 和 `query_role_valid [3]`。Collate 后分别为 `[B,3,W]`、`[B,3]`，dtype 推荐 `bool`。因为 OOV token 会被删除，role mask 必须在“保留词”循环中同步生成，不能按原句 token index 直接对齐。

在模型中使用 `enc_out[:,1:] [B,W,D]` 作为 query token states，排除 learned start token。每个 unit 为：

\[
q_l=\frac{\sum_w R_{lw}H_w}{\sum_wR_{lw}+\epsilon}+e_l,
\quad l\in\{pred,entity,sentence\},
\]

其中 `e_l` 是可学习 role embedding `[3,D]`。

若某条 query 没有保留的 verb 或 noun，对应 pooling mask 回退到 sentence mask，同时令 `query_role_valid` 的对应位置为 false。跨 role attention 可以屏蔽该重复的 fallback unit，但 sentence unit 必须始终有效。不要生成全零 mask 后直接 softmax，否则会产生 NaN。

训练和推理都由同一个 dataset 生成这些 mask，因此没有 train/inference 信息差。QCEC 应在 `_mask_words()` 之前计算，确保三个 unit 基于未随机遮蔽的完整 query。

## 3.4 Query-cluster 交互与相关性

令在线簇 token 为 `Z [B,M,D]`，三个 query unit 为 `Q [B,3,D]`。先投影并计算 scaled dot-product：

\[
A_{ml}=\frac{(W_z z_m)^T(W_q q_l)}{\sqrt{d_a}}.
\]

对无效 cluster/role 做 mask。对 role 维 softmax：

\[
\alpha_{ml}=softmax_l(A_{ml}),\qquad
\tilde q_m=\sum_l\alpha_{ml}q_l.
\]

使用独立融合层而不是修改 `DualTransformer`：

\[
\tilde z_m=LN\left(z_m+MLP([z_m;\tilde q_m;z_m\odot\tilde q_m])\right).
\]

簇相关性定义为：

\[
r_m=\sigma\left(\frac{
cos(W_r^z\tilde z_m,W_r^q q_{sentence})-b_r}{\tau_r}\right).
\]

推荐变量：

- `cluster_attention [B,M,3]`；
- `enhanced_clusters [B,M,D]`；
- `cluster_relevance [B,M]`；
- `relevance_temperature=0.1`；
- `relevance_bias` 可学习标量，初始化为 0。

`cluster_relevance` 需要 gradient，因为它参与全局提示池化和 slot feature 选择；但在 crossing loss 中使用的副本默认 detach，防止退化。

这里不引入显式 mutual-information objective。CACMI 的“cross-modal interaction”在 QCEC 中体现为 query-conditioned cluster relevance 和融合，不应仅因论文名或跨模态机制而额外假设 MI loss。

## 3.5 三个提示 `r`、`b^L`、`b^R`

相邻簇 `m` 与 `m+1` 之间的视觉变化强度先由未增强的簇 token 计算：

\[
\delta_m^{vis}=1-cos(stopgrad(z_m),stopgrad(z_{m+1})),
\]

\[
d_m=\sigma\left(\frac{\delta_m^{vis}-\theta_d}{\tau_d}\right),
\quad m=1,\ldots,M-1.
\]

默认 `transition_threshold=0.2`、`transition_temperature=0.05`；首轮实验前应在训练集抽样上检查 C3D/I3D 的 cosine-distance 分布，必要时将阈值改为分位数统计。`d [B,M-1]` 是 query-independent 变化强度，不需要 gradient。

三个提示为：

1. 相关性提示 `r_m`：目标更可能位于哪个簇；
2. 左边界提示 `b_m^L`：从前一簇进入簇 `m` 时，相关性是否明显上升；
3. 右边界提示 `b_m^R`：从簇 `m` 进入后一簇时，相关性是否明显下降。

简单公式为：

\[
b_m^L=d_{m-1}[r_m-r_{m-1}-\delta_r]_+,
\]

\[
b_m^R=d_m[r_m-r_{m+1}-\delta_r]_+.
\]

默认 `relevance_change_margin delta_r=0.05`。视频首尾可定义：

\[
b_1^L=r_1,\qquad b_M^R=r_M,
\]

但为避免所有候选被吸附到 0/1，首尾提示只用于 global summary，默认不作为 snapping candidate，除非置信度超过独立的 edge threshold。

推荐 tensor：

- `cluster_relevance [B,M]`；
- `left_boundary_hint [B,M]`；
- `right_boundary_hint [B,M]`；
- `transition_score [B,M-1]`；
- `transition_positions [B,M-1]`，取相邻簇共享边界；
- `transition_mask [B,M-1]`。

把三个簇级提示映射回 frame 仅用于可视化和诊断：

```python
frame_hints = gather(
    torch.stack([r, b_left, b_right], dim=-1),
    cluster_ids,
)  # [B,T,3]
```

不建议首版把 `frame_hints` 拼到 50-step reconstruction feature 上，因为原方案要求 reconstruction branch 保持不变，而且这会同时改变 NLL 代理任务，难以判断定位收益来自哪里。

## 3.6 三个提示如何进入原候选生成器

这是首版实现的核心接口。三类加权簇表示为：

\[
g_r=\frac{\sum_m r_m\tilde z_m}{\sum_mr_m+\epsilon},
\quad
g_L=\frac{\sum_m b_m^L\tilde z_m}{\sum_mb_m^L+\epsilon},
\quad
g_R=\frac{\sum_m b_m^R\tilde z_m}{\sum_mb_m^R+\epsilon}.
\]

同时计算软边界统计：

\[
\hat s=\frac{\sum_m b_m^L s_m}{\sum_mb_m^L+\epsilon},
\qquad
\hat e=\frac{\sum_m b_m^R e_m}{\sum_mb_m^R+\epsilon}.
\]

当某侧提示总质量小于 `1e-4` 时，用 detached fallback：相关性最高簇的起点/终点；不要让除零或无边界样本产生异常大梯度。

把 `g_r/g_L/g_R/hat_s/hat_e` 投影为：

\[
g_{qcec}=W_g[g_r;g_L;g_R;\hat s;\hat e]\in\mathbb{R}^D.
\]

现有全局候选特征记为 `g=h[:,-1]`。Residual adapter 为：

\[
g'=g+W_o GELU(W_i[g;g_{qcec}]).
\]

`W_o` 的 weight 和 bias 必须初始化为 0。这样：

- QCEC-enabled 模型从 baseline checkpoint warm-start 时初始输出不变；
- `W_o` 第一轮即可得到 gradient；
- 不需要修改 `DualTransformer`；
- 后续 gradient 可沿 adapter 回到 query-cluster interaction 和在线簇池化。

之后保持原调用：

```python
proposal_generator_feature = qcec_adapter(h[:, -1], qcec_global)
flat_centers, flat_widths, component_masks = (
    self.mixture_generator.predict_components(
        proposal_generator_feature,
        props_len,
        center_logit_bias=center_logit_bias,
        width_logit_bias=width_logit_bias,
    )
)
```

### Proposal-slot 初始化偏置

从 `r` 中选择分数最高的 `N` 个有效簇，得到：

- `slot_features [B,N,D]`；
- `slot_prior_center [B,N]`；
- `slot_prior_width [B,N]`。

用簇 feature 与 `[center,width]` 的小型位置编码共同产生零初始化的 logit residual：

```text
center_logit_bias [B,K_total]
width_logit_bias  [B,N]
```

对于 proposal `n`，只保留其前 `component_counts[n]` 个 center bias，再按现有 proposal-major 顺序展平。`GaussianMixtureProposalGenerator.predict_components()` 修改为：

```python
def predict_components(
    self,
    global_feature: torch.Tensor,              # [B,D]
    sequence_length: int,
    center_logit_bias: Optional[torch.Tensor] = None,  # [B,K_total]
    width_logit_bias: Optional[torch.Tensor] = None,   # [B,N]
):
```

内部逻辑：

```python
center_logits = self.center_head(global_feature)
width_logits = self.width_head(global_feature)
if center_logit_bias is not None:
    center_logits = center_logits + prior_bias_scale * center_logit_bias
if width_logit_bias is not None:
    width_logits = width_logits + prior_bias_scale * width_logit_bias
centers = torch.sigmoid(center_logits)
proposal_widths = torch.sigmoid(width_logits)
```

建议 `prior_bias_scale` 是 config float，而不是新的无约束可学习标量。bias head 的最后一层零初始化，`scale=0.25` 作为完整方案起点；最小验证实验设为 0，只验证 global adapter。

single-Gaussian 路径也要兼容：把 `fc_gauss(g')` reshape 为 `[B,N,2]` 后，加 `single_logit_bias [B,N,2]`，再 sigmoid。QCEC 不应只支持 ActivityNet mixture，而令 Charades single-Gaussian 路径崩溃。

## 3.7 限制候选跨越明显无关事件变化

对相邻簇边界 `tau_j`，定义两个有方向的障碍：

\[
B_j^{\rightarrow}=d_j[r_j-r_{j+1}-\delta_r]_+,
\]

表示左边与 query 相关、右边明显无关，不希望候选终点越过该边界；

\[
B_j^{\leftarrow}=d_j[r_{j+1}-r_j-\delta_r]_+,
\]

表示右边相关、左边明显无关，不希望候选起点越过该边界。

proposal `n` 的连续边界为：

\[
s_n=clip(c_n-w_n/2,0,1),\quad
e_n=clip(c_n+w_n/2,0,1).
\]

用平滑门近似“proposal 跨过边界”：

\[
I_{nj}=\sigma((\tau_j-s_n)/\eta)\,
\sigma((e_n-\tau_j)/\eta).
\]

方向性越界损失：

\[
L_{nj}^{\rightarrow}=I_{nj}B_j^{\rightarrow}[e_n-\tau_j]_+,
\]

\[
L_{nj}^{\leftarrow}=I_{nj}B_j^{\leftarrow}[\tau_j-s_n]_+.
\]

最终：

\[
L_{cross}=\frac{1}{BNJ_{valid}}
\sum_{b,n,j}mask_{bj}
(L_{bnj}^{\rightarrow}+L_{bnj}^{\leftarrow}).
\]

其中 `eta=0.02` 表示归一化视频长度约 2% 的软边界。推荐代码接口：

```python
def qcec_coherence_loss(
    words_logit: torch.Tensor,
    num_props: int,
    center: Optional[torch.Tensor] = None,             # [B*N]
    width: Optional[torch.Tensor] = None,              # [B*N]
    qcec_transition_positions: Optional[torch.Tensor] = None,  # [B,M-1]
    qcec_barrier_lr: Optional[torch.Tensor] = None,    # [B,M-1]
    qcec_barrier_rl: Optional[torch.Tensor] = None,    # [B,M-1]
    qcec_transition_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
```

实现要求：

- QCEC 关闭或任一必需 tensor 为 `None` 时，返回 `words_logit.sum()*0.0` 和全零指标；
- `center/width` 保留 gradient；
- 默认对两个 barrier tensor `.detach()`，配置为 `qcec_detach_barrier=true`；
- `transition_positions/mask` 永远不需要 gradient；
- 所有 pairwise tensor shape 为 `[B,N,M-1]`，通过 `unsqueeze` broadcast，不写 Python proposal 循环；
- loss reduction 临时使用 float32，防止 AMP 下小 barrier 与 sigmoid 相乘下溢；
- `qcec_cross_weight` 默认 0.0，QCEC 实验建议从 0.1 开始；
- 记录未乘权重的 `qcec_cross_raw`、加权后的 `qcec_loss`、active barrier ratio、proposal crossing ratio。

为什么 barrier 要 detach：若 `L_cross` 同时更新 `r`，最容易的解不是移动 proposal，而是令所有 `r` 相等，使两个方向的 barrier 都变成 0。detach 后，该损失只把当前 QCEC 判断当作结构先验，主要对候选中心/宽度和 adapter 施加梯度。QCEC relevance 自身通过原 reconstruction/ranking/event loss 经 `g'` 学习。

首版不要加入无方向的 `d_j` crossing penalty。纯视觉变化可能只是镜头运动，只有“强变化 + 两侧 query relevance 明显不同”才应该限制候选。

### 可选 component-level 版本

outer boundary 的 min/max 只给极端 component 较强梯度。若 proposal-level `L_cross` 有效但梯度稀疏，可对有效 component 的：

\[
l_{n,k}=\mu_{n,k}-w_{n,k}/2,
\quad r_{n,k}=\mu_{n,k}+w_{n,k}/2
\]

重复方向性 penalty，并用 `component_importance.detach()` 加权。该项应有独立的 `qcec_component_cross_weight`，默认 0，不能与首版混在一起。

## 3.8 推理边界吸附

边界吸附只在 `not self.training` 且 `qcec.snap_enabled=true` 时执行，必须放在 `torch.no_grad()` 逻辑中。推荐接口：

```python
def snap_proposal_boundaries(
    center: torch.Tensor,                # [B,N]
    width: torch.Tensor,                 # [B,N]
    cluster_bounds: torch.Tensor,        # [B,M,2]
    cluster_relevance: torch.Tensor,     # [B,M]
    left_boundary_hint: torch.Tensor,    # [B,M]
    right_boundary_hint: torch.Tensor,   # [B,M]
    cluster_mask: torch.Tensor,          # [B,M]
    frame_lengths: torch.Tensor,         # [B]
    radius_frames: int = 2,
    min_confidence: float = 0.15,
    min_relevance: float = 0.5,
    min_width: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
```

过程：

1. 计算原 `s/e [B,N]`；
2. start candidate 为有效簇的 `cluster_bounds[...,0]`，要求 `b^L>=min_confidence` 且 `r>=min_relevance`；
3. end candidate 为 `cluster_bounds[...,1]`，要求 `b^R>=min_confidence` 且 `r>=min_relevance`；
4. 每个 endpoint 只在 `radius_frames / frame_lengths[b]` 范围内选择最近 candidate；无 candidate 时保持不变；
5. clamp 到 `[0,1]`；若 `e-s<min_width`，回退原边界，不能交换起止点；
6. 返回 snapped center/width 和 endpoint movement 诊断。

`CPL.forward()` 应同时返回原始和推理边界：

```text
center / width                 原始训练边界，兼容现有 loss
qcec_eval_center / width       推理可选吸附后的边界；QCEC 关闭时为 None
```

`MainRunner.eval()` 使用：

```python
eval_center = output.get("qcec_eval_center")
if eval_center is None:
    eval_center = output["center"]
```

NLL 和 event score 仍按原 mask 计算并排序。由于吸附幅度仅 ±2/200，首版不重新运行 reconstruction；但必须同时报告 raw 与 snapped 指标，确认收益不是偶然边界扰动。

## 3.9 推荐 `QCECModule` 接口

在独立工程中新增 `qcec/models/modules/qcec.py`：

```python
class QueryConditionedCoherentEventClusters(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_proposals: int,
        max_components: int,
        edge_boost: float = 1.0,
        edge_power: float = 2.0,
        attention_dim: Optional[int] = None,
        relevance_temperature: float = 0.1,
        relevance_change_margin: float = 0.05,
        transition_threshold: float = 0.2,
        transition_temperature: float = 0.05,
        prior_bias_scale: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        frame_states: torch.Tensor,       # [B,T,D]
        frame_mask: torch.Tensor,         # [B,T]
        query_states: torch.Tensor,       # [B,W,D]
        query_mask: torch.Tensor,         # [B,W]
        query_role_mask: torch.Tensor,    # [B,3,W]
        query_role_valid: torch.Tensor,   # [B,3]
        cluster_ids: torch.Tensor,        # [B,T]
        cluster_bounds: torch.Tensor,     # [B,M,2]
        cluster_mask: torch.Tensor,       # [B,M]
    ) -> Dict[str, torch.Tensor]:
        ...
```

上述类型标注按项目现有 Python 3.8 环境书写，实际文件需从 `typing` 导入 `Dict`、`Optional`、`Tuple`；不要使用 Python 3.10 才支持的 `X | None`。

建议返回键：

```text
cluster_tokens               [B,M,D]
enhanced_clusters            [B,M,D]
cluster_attention            [B,M,3]
cluster_relevance            [B,M]
left_boundary_hint           [B,M]
right_boundary_hint          [B,M]
transition_score             [B,M-1]
transition_positions         [B,M-1]
transition_mask              [B,M-1]
barrier_lr                   [B,M-1]
barrier_rl                   [B,M-1]
frame_hints                  [B,T,3]
global_hint                  [B,D]
slot_features                [B,N,D]
slot_prior_center            [B,N]
slot_prior_width             [B,N]
center_logit_bias            [B,K_total]（mixture 时）
width_logit_bias             [B,N]
single_logit_bias            [B,N,2]（single-Gaussian 时）
```

不建议把这些返回值定义为不透明 tuple；字典更适合逐步 ablation 和 runner diagnostics。所有无效 cluster 对应的连续值应置零，mask 单独返回。

## 4. 端到端数据流

## 4.1 预处理阶段

```text
train/val/test JSON 中的 video id 并集
    -> 从 ActivityNet/Charades HDF5 读取固定 backbone feature
    -> 使用与 BaseDataset 完全一致的 200-step 均值采样
    -> L2 normalize
    -> contiguous Ward clustering, M=32
    -> 保存 video_id -> cluster_ids / bounds / mask 的 NPZ index
```

聚类只依赖视频，不读取 query 或 GT timestamp。必须以视频去重，不能为同一视频的每条 sentence 重复写索引。

## 4.2 训练阶段

```text
Dataset
    -> frames_feat [B,200,D_v]
    -> query features / ids / POS role masks
    -> offline cluster ids / bounds / masks

CPL.forward 原高分辨率路径
    -> frame_fc 后的 frame_states [B,200,D]
    -> DualTransformer decoding=1
    -> enc_out [B,W+1,D], h [B,201,D]

QCEC
    -> boundary-enhanced cluster tokens [B,M,D]
    -> predicate/entity/sentence query units [B,3,D]
    -> cross-modal enhanced clusters
    -> r / b_left / b_right / directional barriers
    -> global_hint [B,D] + proposal slot biases

原候选生成器
    -> g'=adapter(h[:,-1], global_hint)
    -> original center/width heads + optional zero-init slot residual
    -> Gaussian mixture masks and outer/weighted interval

原重构/LREV路径
    -> 200->50 sampling
    -> words_logit / negatives / event vectors

Loss
    -> L_rec + L_ivc + L_event + L_mixture + lambda_cross*L_cross
    -> backward / clip / optimizer / scheduler
```

训练时不执行 boundary snapping。QCEC cluster assignment、bounds、transition positions 和视觉 transition score 不求梯度；簇池化、query interaction、相关性、global adapter 和 generator 参数求梯度。

## 4.3 推理阶段

推理复用与训练相同的 dataset、簇 index、query units、QCEC forward 和原候选生成器。差异只有：

- model 为 `eval()`，dropout 关闭；
- runner 在 `torch.no_grad()` 下运行；
- 不计算或不反传 `L_cross`；
- 若 `snap_enabled`，使用 QCEC hints 对原 interval 做小范围吸附；
- runner 用原 NLL/event score 排序，再计算 raw/snapped metrics。

QCEC 不得在推理时读取 GT、训练 epoch 才存在的伪标签或训练 batch statistics。

## 5. 独立 `qcec/` 工程的文件级详细修改方案

本节所有文件都位于 `/data/chenyuan/videogrounding/w1m2/qcec`。凡标注“修改”的文件，均指修改从 baseline 复制到 `qcec/` 的副本；不得回写 `cpl_lrev/`。

### 5.0 独立工程初始目录

实现开始时先建立以下自包含结构：

```text
qcec/
├── README.md
├── requirements.txt
├── train.py
├── utils.py
├── vocab.py
├── config/
│   ├── activitynet/
│   │   ├── main.json              # 原样保留的工程内 baseline
│   │   └── qcec.json              # QCEC 实验配置
│   └── charades/
│       └── main.json
├── data/
│   ├── activitynet/               # JSON、GloVe 与生成的 cluster index
│   └── charades/
├── datasets/
├── models/
│   ├── modules/
│   │   └── qcec.py
│   └── transformer/
├── optimizers/
│   └── lr_schedulers/
├── runners/
├── scripts/
├── tools/
├── tests/
└── checkpoints/
    └── bootstrap/                 # 可选的 baseline 初始化权重
```

从 `cpl_lrev/` 复制的最小完整基线包括：`train.py`、`utils.py`、`vocab.py`、`requirements.txt`、`datasets/`、`models/`、`optimizers/`、`runners/`、`config/`、必要的 `data/` 索引和词向量、现有 tests 与有用 scripts/tools。不要复制历史 logs 和全部 epoch checkpoints；它们不是独立运行所需代码。若数据文件过大而不适合复制，允许 config 指向仓库外的共享数据集绝对路径，但不得用这种例外引用 `cpl_lrev` 下的 Python 源码。

复制完成后先在 `qcec/` 内运行 baseline smoke test，并用以下检查确保没有跨工程代码依赖：

```bash
cd /data/chenyuan/videogrounding/w1m2/qcec
rg -n "cpl_lrev|\.\./cpl_lrev|sys\.path.*cpl" \
  --glob '*.py' --glob '*.sh' --glob '*.json'
```

除用于说明来源的 README 文本外，该检查应无命中。所有后续命令都应从 `qcec/` 执行。

## 5.1 `qcec/tools/build_qcec_cluster_index.py`——新增，必须

职责：按视频预计算连续 Ward 簇。

必须实现：

- 读取一个或多个 split JSON；
- 按 `video_id` 去重；
- 按 dataset 类型读取 ActivityNet group 下 `c3d_features` 或 Charades dataset；
- 复用/抽取与 `BaseDataset._sample_frame_features()` 完全一致的采样函数；
- 实现 `contiguous_ward_cluster()`；
- 写临时文件后原子 rename，避免中断留下半成品；
- 在 metadata 中写入数据集、T、M、feature dim、normalize 和版本；
- 支持 `--overwrite`，默认已存在时拒绝覆盖；
- 固定 tie-break，保证同一输入重复构建 bitwise 相同 cluster IDs。

推荐 CLI：

```bash
python tools/build_qcec_cluster_index.py \
  --config-path config/activitynet/qcec.json \
  --splits train,val,test \
  --num-clusters 32 \
  --output data/activitynet/qcec_clusters_m32.npz
```

快速验证：随机抽 10 个视频，确认每个 cluster ID 的 frame index 连续、bounds 单调、所有 frame 恰好属于一个有效簇、相同命令两次输出一致。

## 5.2 `qcec/datasets/base.py`——修改，必须

当前职责：读取 sample、tokenize/POS、采样视频、collate。

修改点：

1. 将 `_sample_frame_features()` 中的纯 NumPy采样逻辑抽成 module-level helper，供预处理脚本复用；保留方法 wrapper，保证现有调用不变。
2. `BaseDataset.__init__()` 在配置存在 `qcec_cluster_index_path` 时加载 NPZ，并构造 `video_id -> row` 映射；配置不存在时完全不加载。
3. `__getitem__()` 在现有 POS 循环中同步构造 `query_role_mask`，严格对齐被词表保留的 `words`。
4. QCEC 开启时，按 `vid` 取 `qcec_cluster_ids/bounds/mask` 并加入 sample；找不到 vid 时抛出包含 vid 和 index path 的明确错误。
5. `build_collate_data()` 增加可选 `qcec_num_clusters=None` 与 `return_query_roles=False`，collate 相应 tensor。

新增 batch 输入建议：

```text
query_role_mask      [B,3,W]   bool
query_role_valid     [B,3]     bool
qcec_cluster_ids     [B,T]     int64
qcec_cluster_bounds  [B,M,2]   float32
qcec_cluster_mask    [B,M]     bool
```

兼容性：baseline config 不含 QCEC dataset 字段时，不返回新增 key；即使始终返回 role mask，`CPL.forward(..., **kwargs)` 也能接收，但为最大程度保证旧 pipeline 建议仅在 QCEC config 中返回。

## 5.3 `qcec/datasets/activitynet.py` 与 `qcec/datasets/charades_sta.py`——修改，必须

当前两个类只负责 feature HDF5 layout 和构造 collate closure。

修改构造 `build_collate_data(...)` 时传入：

```python
qcec_num_clusters=args.get("qcec_num_clusters"),
return_query_roles=args.get("return_query_roles", False),
```

不要在两个 dataset 子类中复制 QCEC 聚类或 role mask 逻辑。

## 5.4 `qcec/models/modules/qcec.py`——新增，必须

包含以下独立单元，便于 unit test：

- `boundary_enhanced_pool()`；
- `pool_query_roles()`；
- `compute_transition_scores()`；
- `compute_directional_hints()`；
- `QCECProposalAdapter`；
- `QueryConditionedCoherentEventClusters`；
- `snap_proposal_boundaries()`。

模块内禁止 `.cuda()`；所有新 tensor 使用输入的 `.device`、`.dtype` 或 `new_*` 创建。mask 在入口统一转 bool。所有 `softmax` 前先保证至少一个有效元素；对 fallback 路径写 unit test。

## 5.5 `qcec/models/modules/__init__.py`——修改，必须

导出：

```python
from .qcec import QueryConditionedCoherentEventClusters
```

如果 snapping 保持为 module-level helper，也一并显式导出或仅在 `cpl.py` 从 `.qcec` 直接 import；不要使用 wildcard import。

## 5.6 `qcec/models/cpl.py`——修改，必须

### `CPL.__init__`

读取 `qcec_config=config.get('qcec', {})`：

```python
self.use_qcec = qcec_config.get("enabled", False)
```

仅在 enabled 时实例化 QCEC 和 adapter。这样没有 `qcec` 配置或 `enabled=false` 时，state dict 不新增 QCEC keys，旧 checkpoint 可 strict load。

需要保存：

- `self.qcec_snap_enabled`；
- `self.qcec_prior_bias_scale`；
- `self.qcec_freeze_backbone_epochs`；
- QCEC module 和零初始化 adapter/bias heads。

### `CPL.forward`

推荐把签名扩成：

```python
def forward(
    self,
    frames_feat,
    frames_len,
    words_id,
    words_feat,
    words_len,
    weights,
    query_role_mask=None,
    query_role_valid=None,
    qcec_cluster_ids=None,
    qcec_cluster_bounds=None,
    qcec_cluster_mask=None,
    **kwargs,
):
```

具体插入顺序：

1. 保留原 append/dropout/frame_fc 计算；在覆盖变量前保存 `highres_frame_states = frames_feat[:, :n_frames]` 和 `highres_frame_mask = frames_mask[:, :n_frames].bool()`。
2. 完成原 `enc_out,h=self.trans(...,decoding=1)`。
3. 先令 `base_proposal_feature=h[:,-1]`。
4. 若 `use_qcec=false`，令 `proposal_generator_feature=base_proposal_feature`，其余行为完全不变。
5. 若 `use_qcec=true`，检查四个新增 batch tensor 均非 None，调用 QCEC：

```python
qcec_output = self.qcec(
    frame_states=highres_frame_states,
    frame_mask=highres_frame_mask,
    query_states=enc_out[:, 1:],
    query_mask=words_mask[:, 1:].bool(),
    query_role_mask=query_role_mask.bool(),
    query_role_valid=query_role_valid.bool(),
    cluster_ids=qcec_cluster_ids.long(),
    cluster_bounds=qcec_cluster_bounds,
    cluster_mask=qcec_cluster_mask.bool(),
)
proposal_generator_feature = self.qcec_adapter(
    base_proposal_feature, qcec_output["global_hint"])
```

6. single/mixture head 使用 QCEC bias；bias 为 None 时走原逻辑。
7. 在完成 final center/width 后、negative mining 前，可计算 eval snapping；训练时不要覆盖 `center/width`。
8. output dict 新增所有 loss 和 diagnostics 必需的 QCEC tensor，QCEC 关闭时相应值为 None。

推荐新 output keys：

```text
qcec_transition_positions
qcec_transition_mask
qcec_transition_score
qcec_barrier_lr
qcec_barrier_rl
qcec_cluster_relevance
qcec_left_boundary_hint
qcec_right_boundary_hint
qcec_cluster_bounds
qcec_cluster_mask
qcec_eval_center
qcec_eval_width
qcec_snap_start_delta
qcec_snap_end_delta
```

不要把所有 cluster token 返回给 runner，除非 debug config 开启；`[B,M,D]` 长期保留在 output 会增加峰值显存和 Python 引用生命周期。loss 只需要 barrier、position 和 mask。

### staged trainability

建议给 `CPL` 增加：

```python
def set_qcec_training_stage(self, qcec_only: bool) -> None:
```

当 `qcec_only=true` 时，只让参数名以 `qcec` 开头的 module、adapter 和 bias head `requires_grad=True`，其余参数 false；第二阶段恢复构造时记录的原始 trainability。不要依赖模糊字符串包含关系，最好保存 QCEC parameter object id 集合。

## 5.7 `qcec/models/modules/gaussian_mixture.py`——修改，必须

只扩展 `predict_components()` 的可选参数，不改 `combine()` 和 component layout。

必须检查：

- `center_logit_bias.shape == center_logits.shape`；
- `width_logit_bias.shape == width_logits.shape`；
- device/dtype 一致，必要时 `.to(center_logits)`；
- `None` 时不执行额外加法，保证 baseline exactness。

不要把 QCEC module import 到 Gaussian generator；generator 只接受一般化 logit bias，避免反向依赖。

## 5.8 `qcec/models/loss.py`——修改，必须

新增 `qcec_coherence_loss()`，按第 3.7 节公式实现。保持函数式风格，与其他 loss 一致。

建议返回日志：

```text
qcec_loss
qcec_cross_raw
qcec_active_barrier_fraction
qcec_mean_barrier
qcec_proposal_crossing_fraction
qcec_mean_relevance（可由 model output 传入）
```

loss 为 finite 的关键点：

- 空 transition 时返回 graph-connected zero；
- mask 后 denominator `clamp_min(1)`；
- sigmoid 输入可以 clamp 到 `[-30,30]`；
- 不用 `-inf * 0`；
- width 在原 generator 已大于 0，但 endpoint 仍需 clamp。

## 5.9 `qcec/runners/main_runner.py`——修改，必须

### Import 和训练 loss

导入 `qcec_coherence_loss`，在 mixture loss 后调用：

```python
qcec_loss, qcec_loss_dict = qcec_coherence_loss(
    **output,
    num_props=self.model.num_props,
    **self.args["loss"],
)
loss_dict.update(qcec_loss_dict)
loss = loss + qcec_loss
```

QCEC 关闭时该调用仍安全并返回 0，避免 runner 分支散落。

### 两阶段训练

在每个 epoch 开始、`self.model.train()` 后调用 `self.model.set_qcec_training_stage(...)`。若 `epoch <= freeze_backbone_epochs`，只训练 QCEC；之后联合训练。建议首版 `freeze_backbone_epochs=3`，且不要超过现有 event warmup 5，避免冻结阶段更新 LREV running subspace。

optimizer 在 epoch 前已经收集所有参数，因此只切换 `requires_grad` 即可；但需要确认自定义 Adam 对 `grad is None` 参数安全。联合阶段恢复后这些参数仍在 optimizer param groups 中。

### Eval

构造 interval 时优先使用 `qcec_eval_center/width`，同时保留 raw interval。新增 QCEC diagnostics：

- raw proposal 跨 active directional barrier 的比例；
- snapped proposal 跨 barrier 的比例；
- 平均 start/end 吸附距离；
- 每视频 active barrier 数；
- GT short (`width<0.15`) 的 R@5 指标沿用现有 diagnostics；
- 可增加 Rank-1 start MAE/end MAE，但不能替代 IoU 指标。

候选 NLL 排序和 `select_proposal_by_strategy()` 首版不修改。若后续让 QCEC score 参与排序，应作为独立 ablation，不与结构先验首轮结果混合。

## 5.10 `qcec/train.py`——修改，推荐

QCEC 主配置应放在 JSON，而不是为每个超参数增加 CLI。仅建议增加三类常用 override：

```text
--qcec-enabled / --qcec-disabled    tri-state bool，默认 None
--qcec-cross-weight FLOAT           默认 None
--qcec-snap-enabled                 tri-state，默认 None
--init-from-baseline PATH           与 --resume/--init-from-v3 互斥
```

读取位置为 `main()` 的 JSON load 后：

- enabled/snap 写入 `args['model']['config']['qcec']`；
- cross weight 写入 `args['loss']`；
- baseline checkpoint 调用新的 `_load_baseline_model()`。

如果实现者希望最小改动，可以不加前三个 override，只实现 `--init-from-baseline`；实验超参数通过独立 config 文件控制。

## 5.11 `qcec/config/activitynet/qcec.json`——新增，必须

从当前实际 `config/activitynet/main.json` 复制，而不是按 README 中已漂移的 weighted 描述重建。保持当前 baseline 的 `boundary_mode='outer'`、loss、num_props 和 event config，只新增 QCEC 字段。

不要直接修改 `main.json`，它应继续作为 baseline 配置。

## 5.12 `qcec/tests/`——新增/修改，必须

新增：

- `qcec/tests/test_qcec_clustering.py`；
- `qcec/tests/test_qcec_module.py`；
- `qcec/tests/test_qcec_loss.py`；
- `qcec/tests/test_qcec_integration.py`。

扩展：

- `test_gaussian_mixture.py`：None bias exactness、非零 bias shape/gradient；
- `test_v4_integration.py`：QCEC enabled forward + all loss backward；
- checkpoint test：旧 baseline → enabled QCEC warm-start、QCEC strict resume、disabled strict load。

## 6. 配置 specification

## 6.1 Dataset 配置

| 参数 | 类型 | 默认值 | 作用 | 读取位置 |
|---|---:|---:|---|---|
| `qcec_cluster_index_path` | str/缺省 | 缺省 | 离线 cluster NPZ；缺省时 baseline 不加载 | `BaseDataset.__init__` |
| `qcec_num_clusters` | int/缺省 | 缺省 | collate 的固定 M；QCEC config 建议 32 | dataset constructors / collate |
| `return_query_roles` | bool | false | 是否返回 `[3,W]` POS role mask | `BaseDataset.__getitem__` |

## 6.2 Model `qcec` 配置

| 参数 | 类型 | 代码默认值 | QCEC 实验建议 | 作用 |
|---|---:|---:|---:|---|
| `enabled` | bool | false | true | 总开关；false 时不实例化新参数 |
| `num_clusters` | int | 32 | 32 | 必须与离线索引一致 |
| `edge_boost` | float | 1.0 | 1.0 | 簇边缘相对增强量 |
| `edge_power` | float | 2.0 | 2.0 | 边缘权重曲线指数 |
| `attention_dim` | int/null | hidden size | 256 | cluster-query projection 维度 |
| `relevance_temperature` | float | 0.1 | 0.1 | relevance sigmoid 温度 |
| `relevance_change_margin` | float | 0.05 | 0.05 | 两侧 relevance 差异阈值 |
| `transition_threshold` | float | 0.2 | 数据校准 | cosine distance 变化阈值 |
| `transition_temperature` | float | 0.05 | 0.05 | transition sigmoid 温度 |
| `use_slot_prior` | bool | false | 第二阶段再 true | 是否给 center/width logits 加 slot bias |
| `prior_bias_scale` | float | 0.25 | 0.25 | slot bias 强度；最小实验设 0 |
| `freeze_backbone_epochs` | int | 0 | 3 | QCEC-only 训练 epoch 数 |
| `snap_enabled` | bool | false | 完整实验可 true | 推理边界吸附 |
| `snap_radius_frames` | int | 2 | 2 | 最大 endpoint 移动范围 |
| `snap_min_confidence` | float | 0.15 | 校准后决定 | b-left/right 最低置信度 |
| `snap_min_relevance` | float | 0.5 | 0.5 | snapping 簇最低相关性 |
| `snap_min_width` | float | 0.01 | 0.01 | 防止非法/零宽区间 |
| `return_debug_tensors` | bool | false | false | 是否把 `[B,M,D]` 中间量放进 output |

构造时验证：`M>=N>=1`、所有 temperature > 0、margin >= 0、radius >= 0、prior scale >= 0、hidden/attention dim 有效。

## 6.3 Loss 配置

| 参数 | 类型 | 代码默认值 | QCEC 实验建议 | 作用 |
|---|---:|---:|---:|---|
| `qcec_cross_weight` | float | 0.0 | 0.1 | proposal-level crossing loss 权重 |
| `qcec_cross_temperature` | float | 0.02 | 0.02 | crossing smooth gate 的 eta |
| `qcec_detach_barrier` | bool | true | true | 阻止 relevance 通过关闭 barrier 逃避 loss |
| `qcec_component_cross_weight` | float | 0.0 | 0.0 | 可选 component-level penalty |
| `qcec_boundary_align_weight` | float | 0.0 | 0.0 | 可选边界吸引项；首版禁用 |

## 6.4 推荐完整 JSON 片段

```json
{
  "dataset": {
    "qcec_cluster_index_path": "data/activitynet/qcec_clusters_m32.npz",
    "qcec_num_clusters": 32,
    "return_query_roles": true
  },
  "model": {
    "config": {
      "qcec": {
        "enabled": true,
        "num_clusters": 32,
        "edge_boost": 1.0,
        "edge_power": 2.0,
        "attention_dim": 256,
        "relevance_temperature": 0.1,
        "relevance_change_margin": 0.05,
        "transition_threshold": 0.2,
        "transition_temperature": 0.05,
        "use_slot_prior": false,
        "prior_bias_scale": 0.25,
        "freeze_backbone_epochs": 3,
        "snap_enabled": false,
        "snap_radius_frames": 2,
        "snap_min_confidence": 0.15,
        "snap_min_relevance": 0.5,
        "snap_min_width": 0.01,
        "return_debug_tensors": false
      }
    }
  },
  "loss": {
    "qcec_cross_weight": 0.1,
    "qcec_cross_temperature": 0.02,
    "qcec_detach_barrier": true,
    "qcec_component_cross_weight": 0.0,
    "qcec_boundary_align_weight": 0.0
  }
}
```

首个 QCEC-lite 实验应将 `qcec_cross_weight=0`、`use_slot_prior=false`、`snap_enabled=false`，只验证 global feature adapter；之后一次只打开一个机制。

## 7. 训练策略

## 7.1 推荐两阶段训练

### 阶段 A：QCEC adapter warm-up

- epoch 1–3；
- 冻结 frame/word projection、DualTransformer、reconstructor、mixture generator、LREV；
- 只训练 QCEC query-cluster interaction、global adapter 和 slot bias heads；
- `use_slot_prior=false`；
- `qcec_cross_weight=0.1` 可以开启，因为 crossing gradient 可通过固定 generator head 回到 `g'`；若训练不稳先设 0；
- 原 loss 仍计算，它们通过固定下游网络监督 adapter，但不更新 baseline 权重。

### 阶段 B：联合微调

- epoch 4 起恢复所有原本可训练参数；
- 使用较小总学习率或对 QCEC 以外参数乘 0.1 LR 是可选增强，不是首版必需；
- 先保持 slot prior 和 snapping 关闭；
- feature fusion + crossing 被验证后，再打开 slot prior；
- snapping 只在 validation/test 生效，不参与训练。

如果从随机初始化训练，QCEC-only warm-up 没有意义，因为冻结的 generator/reconstructor 尚未形成可用监督。两阶段策略默认假设使用当前 baseline checkpoint warm-start；随机初始化实验应设置 `freeze_backbone_epochs=0`。

## 7.2 不使用 GT 的原则

训练 batch 中的 `raw.timestamps` 不得送给 QCEC。cluster index、transition、role masks、relevance 和 crossing loss 均由视频/query/当前 proposal 产生。GT 只用于 validation/test metrics，保持弱监督设定。

## 8. 推理与评估细则

QCEC 首轮对比至少包含：

1. baseline；
2. QCEC-global：仅 residual adapter；
3. QCEC-global + `L_cross`；
4. 上述 + slot prior；
5. 上述 + inference snapping。

每组报告：

- R@1 mIoU；
- R@5 mIoU；
- R@1/R@5 IoU@0.3、0.5、0.7；
- GT width `<0.15` 的 R@5 mIoU 和 IoU@0.5；
- proposal 平均 width；
- pairwise IoU；
- raw/snapped crossing ratio；
- 平均 endpoint movement；
- R@5−R@1 mIoU gap。

支持假设的最低信号：短事件 R@5 mIoU 有稳定提升、R@1@0.5 不下降、active barrier crossing ratio 下降，并且提升不只体现在更宽 proposal。若 NLL 下降但短事件/高 IoU 指标不变，说明 QCEC 被重构捷径吸收；若 crossing ratio 降低但 recall 明显下降，说明 transition/relevance 过强或错误。

## 9. Checkpoint 加载、保存和 backward compatibility

## 9.1 三种加载语义必须分开

1. `--resume`：只加载结构完全一致的 QCEC checkpoint，strict=True；
2. `--init-from-v3`：保留当前 V3→V4 行为，仍跳过旧 event subspace 和不存在的 proposal head；
3. 新增 `--init-from-baseline`：从当前 V4/LREV baseline 加载所有名称和 shape 匹配的参数，只允许缺少 `qcec.*`、`qcec_adapter.*`、`qcec_*_bias.*` 等新参数。

推荐：

```python
def _load_baseline_model(self, path):
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("model_parameters", checkpoint)
    result = self.model.load_state_dict(source, strict=False)
    unexpected = result.unexpected_keys
    illegal_missing = [
        key for key in result.missing_keys
        if not key.startswith(ALLOWED_QCEC_PREFIXES)
    ]
    if unexpected or illegal_missing:
        raise ValueError(...)
    self.num_updates = 0
```

不要静默忽略任意 missing key；否则 baseline backbone 的错误配置也会被当成正常 warm-start。

## 9.2 Identity initialization

- QCEC disabled：不实例化新 module，旧 baseline checkpoint strict load；
- QCEC enabled：adapter output projection 和 slot bias output projection 零初始化，因此加载 baseline 后首个 eval 的原候选参数应与 baseline 接近/一致；
- 如果 snapping=true，即使 adapter 为 identity，eval boundary 也会变化，因此 checkpoint compatibility test 必须在 snapping=false 下执行。

## 9.3 保存 epoch 和 optimizer 的建议

为正确恢复两阶段 schedule，建议向新 checkpoint 增加：

```text
epoch
optimizer_state（推荐）
lr_scheduler_state（若当前 wrapper 支持）
qcec_cluster_metadata/hash（推荐）
```

读取旧 checkpoint 时字段缺失要有默认值。至少保存 `epoch` 并让 `train()` 从 `start_epoch+1` 开始；否则恢复训练会错误地重新进入 QCEC-only 阶段。保存 optimizer state 是推荐修改，但涉及当前 runner 的历史行为，若首版暂不做，日志中必须明确“weights-only resume”。

## 9.4 不应破坏的行为

- `qcec` 配置缺省或 false 时，模型 output、loss、选择策略与现状一致；
- baseline 的 single-Gaussian 和 Gaussian-mixture 两条路径都能运行；
- `num_props` 不变；
- NLL/semantic vote 行为首版不变；
- 原 checkpoint 文件不重写；
- 原 `main.json` 不修改，新增独立 `qcec.json`。

## 10. 代码修改清单

## 10.1 必须修改

```text
qcec/（新增独立工程根目录）
- 复制完整可运行 baseline 骨架，禁止跨目录 import、symlink 或 sys.path 注入

qcec/tools/build_qcec_cluster_index.py（新增）
- 实现视频去重、200-step 同构采样、连续 Ward 聚类、NPZ 输出和 metadata 校验

qcec/datasets/base.py
- 抽取共享采样 helper
- 加载 cluster index
- 构造 predicate/entity/sentence role masks
- collate QCEC tensors

qcec/datasets/activitynet.py
qcec/datasets/charades_sta.py
- 把 QCEC collate 参数传给 build_collate_data

qcec/models/modules/qcec.py（新增）
- 边界增强池化、query role pooling、cross-modal interaction
- 三提示、directional barriers、global hint、slot biases
- inference snapping

qcec/models/modules/__init__.py
- 导出 QCEC module

qcec/models/cpl.py
- 构造 QCEC
- 在首次 DualTransformer 后融合到 h[:, -1]
- 将可选 bias 送入 single/mixture generator
- 返回 QCEC loss/diagnostic tensors
- 推理生成 qcec_eval_center/width

qcec/models/modules/gaussian_mixture.py
- predict_components 接受可选 center/width logit bias

qcec/models/loss.py
- 新增 qcec_coherence_loss

qcec/runners/main_runner.py
- 累加 QCEC loss
- QCEC-only / joint 阶段切换
- eval 使用可选 snapped boundary 并记录 diagnostics
- 增加 baseline warm-start loader

qcec/config/activitynet/qcec.json（新增）
- 从当前实际 main.json 派生，只新增 QCEC 字段

qcec/tests/test_qcec_clustering.py（新增）
qcec/tests/test_qcec_module.py（新增）
qcec/tests/test_qcec_loss.py（新增）
qcec/tests/test_qcec_integration.py（新增）
- 覆盖聚类、shape、finite、gradient、forward/backward、compatibility
```

## 10.2 推荐修改

```text
qcec/train.py
- 增加 --init-from-baseline
- 增加少量 QCEC 常用 override

qcec/runners/main_runner.py
- checkpoint 保存 epoch/optimizer state
- raw vs snapped、crossing ratio 和 endpoint MAE diagnostics

qcec/README.md
- 补充 QCEC index 构建和训练命令
```

## 10.3 可选修改

```text
qcec/config/charades/qcec.json
- ActivityNet 验证后再扩展到 Charades

qcec/models/loss.py
- component-level crossing loss
- boundary alignment reward

qcec/tools/visualize_qcec.py
- 可视化 cluster、r/bL/bR、proposal 和 GT

qcec/models/modules/qcec.py
- 在线 frame_fc 后聚类 ablation
- soft/differentiable clustering ablation
```

## 11. Implementation Order

## Step 0：创建独立工程并锁定 baseline

目标目录：`/data/chenyuan/videogrounding/w1m2/qcec`。

操作与验证：

- 按第 5.0 节复制完整可运行 baseline 骨架到 `qcec/`，不复制历史 logs 和全部 checkpoints；
- 从此步骤起只修改 `qcec/`，把 `cpl_lrev/` 视为只读；
- 保存 `config/activitynet/main.json` 的副本 hash 和目标 baseline checkpoint config；
- 明确当前实际 main/checkpoint 都是 `boundary_mode='outer'`；
- 在 `qcec/` 中记录 baseline eval 命令和结果；
- 从 `qcec/` 运行 import/smoke test，确认 Python module 的 `__file__` 均位于 `/data/chenyuan/videogrounding/w1m2/qcec`；
- 用 `rg` 确认 QCEC 源码、配置和脚本没有 import、symlink 或路径注入到 `cpl_lrev/`；
- 当前 README/测试中存在 weighted config 与缺失文件的历史漂移，不要在 QCEC 实现中顺手修复。

下一步依赖：`qcec/` 已能在不调用 `cpl_lrev` 代码的情况下运行 baseline，且已确定 QCEC config 从哪个真实 baseline 派生。

## Step 1：实现和验证离线 cluster index

修改文件：`qcec/datasets/base.py` 的共享采样 helper、`qcec/tools/build_qcec_cluster_index.py`、`qcec/tests/test_qcec_clustering.py`。

快速验证：

- synthetic 三段特征应得到连续三簇；
- cluster IDs 覆盖 `[0,T)` 且每个 ID 只出现一个连续 run；
- bounds 与 IDs 一致；
- 重复运行 deterministic；
- 随机真实视频可成功构建 M=32 index。

下一步依赖：稳定的 `cluster_ids/bounds/mask` 文件。

## Step 2：接通 dataset 和 collate，但不改模型行为

修改文件：`qcec/datasets/base.py`、`qcec/datasets/activitynet.py`、`qcec/datasets/charades_sta.py`。

快速验证：取一个 batch，assert：

```text
frames_feat [B,200,D_v] float32
query_role_mask [B,3,W] bool
query_role_valid [B,3] bool
qcec_cluster_ids [B,200] int64
qcec_cluster_bounds [B,32,2] float32
qcec_cluster_mask [B,32] bool
```

验证每个 role 至少有 fallback token，视频 id 能正确索引。baseline config 不含 index 时 DataLoader 仍运行。

下一步依赖：模型能够收到完整 QCEC metadata。

## Step 3：实现纯 QCEC module unit tests

修改文件：`qcec/models/modules/qcec.py`、`qcec/models/modules/__init__.py`、`qcec/tests/test_qcec_module.py`。

先只实现池化、query units、`r/bL/bR`、barriers 和 global hint，不接 `CPL`。

快速验证：

- 所有输出 shape 正确；
- padding cluster 输出为 0；
- 相同簇内边缘帧 gradient 大于中心帧（在同一简单目标下）；
- 构造“左相关、右无关且视觉变化强”的样例，`barrier_lr > barrier_rl`；
- forward 和 backward 全 finite；
- 全无 verb/noun 的 fallback 不产生 NaN。

下一步依赖：稳定的 QCEC output contract。

## Step 4：只接 global residual adapter

修改文件：`qcec/models/cpl.py`、`qcec/tests/test_qcec_integration.py`。

先不开 slot prior、crossing loss 和 snapping。零初始化 adapter 后：

- baseline checkpoint 经 `strict=False` 合理加载；
- 同一输入、eval mode、固定 query mask 下，QCEC-enabled 初始 center/width 与 baseline 一致或在 `1e-6` 内；
- 将 adapter 随机化后 QCEC parameters 能收到 finite gradient；
- 原 `words_logit [B*N,W,V]` 和 `gauss_weight [B*N,50]` 不变形。

下一步依赖：三个提示已真正通过 global hint 影响原生成器。

## Step 5：接 proposal-slot bias

修改文件：`qcec/models/cpl.py`、`qcec/models/modules/gaussian_mixture.py`、相关 `qcec/tests/`。

快速验证：

- None/zero bias 与旧 generator 数值一致；
- mixture flatten 顺序与 `component_counts` 一致；
- nonzero center bias 能按预期移动对应 proposal component；
- single-Gaussian 路径也可 forward/backward；
- bias head 零初始化时 checkpoint warm-start identity 成立。

下一步依赖：完整“global + slot”候选注入机制。

## Step 6：实现 `L_cross`

修改文件：`qcec/models/loss.py`、`qcec/runners/main_runner.py`、`qcec/tests/test_qcec_loss.py`。

快速验证：

- 不跨 barrier 的 interval loss 接近 0；
- 从相关侧越过越远，loss 单调增大；
- 反向后 proposal center/width 或 generator head 有 finite gradient；
- `detach_barrier=true` 时 relevance/barrier 不从该 loss 获得 gradient；
- false 时确有 gradient，作为对照；
- QCEC disabled 返回 graph-connected zero。

下一步依赖：完整训练 objective。

## Step 7：训练阶段切换与 checkpoint

修改文件：`qcec/runners/main_runner.py`、`qcec/train.py`、`qcec/tests/` 下的 checkpoint tests。

快速验证：

- epoch 1 QCEC 参数 trainable、baseline 参数冻结；
-联合阶段全部预期参数恢复；
- baseline checkpoint warm-start 只缺 QCEC keys；
- QCEC checkpoint strict resume；
- 旧 checkpoint 缺 epoch 字段时有明确 fallback；
- resume 不会错误重复 QCEC-only 阶段。

下一步依赖：可安全启动实际训练。

## Step 8：加入 inference snapping 和 diagnostics

修改文件：`qcec/models/modules/qcec.py`、`qcec/models/cpl.py`、`qcec/runners/main_runner.py`、相关 `qcec/tests/`。

快速验证：

- 半径外 candidate 不移动；
- 半径内选择最近边界；
- 无可信 hint 不移动；
- 不产生 `start>end` 或宽度小于阈值；
- snap=false 时 eval 使用原 center/width；
- raw/snapped metrics 同时可记录。

下一步依赖：可做完整 ablation。

## Step 9：smoke train 和小 batch overfit

修改文件：主要是 `qcec/config/activitynet/qcec.json`，不再改算法。

快速验证：

- 1–2 batch 连续训练 20–100 step，loss finite；
- QCEC adapter 参数发生变化；
- `L_cross` 在合成/小样本上可下降；
- GPU memory 和 step time 在可接受范围；
- baseline/qcec-off smoke test 仍通过。

## 12. 测试与验证矩阵

## 12.1 Module/unit sanity

- 聚类连续性、覆盖性、determinism；
- boundary pool 数值手算；
- role-mask OOV 对齐和 fallback；
- query-cluster attention 对 padding mask；
- transition/barrier 方向性；
- adapter identity init；
- slot bias flatten mapping；
- snapping 最近邻和合法区间。

## 12.2 Tensor shape contract

在 `B=2,T=20,M=4,W=6,D=8,N=3,Kmax=3` 的小配置中检查：

```text
cluster_tokens            [2,4,8]
cluster_relevance         [2,4]
left/right hints          [2,4]
transition/barriers       [2,3]
global_hint               [2,8]
slot_features             [2,3,8]
center_logit_bias         [2,6]   # counts 1+2+3
width_logit_bias          [2,3]
gauss_weight              [6,5]   # 20//4
center/width              [6]
words_logit               [6,W,V]
```

## 12.3 Forward、finite 和 backward

总 loss：

```python
total = rec + ivc + event + mixture + qcec
assert torch.isfinite(total)
total.backward()
```

检查：

- adapter output projection grad；
- relevance projections grad；
- boundary pool 上游 `frame_fc` grad（联合阶段）；
- mixture center/width head grad；
- 所有非 None grad finite；
- AMP 下 forward/loss finite。

当前测试通过 patch `Tensor.cuda` 在 CPU 跑完整图；新 QCEC 代码自身必须 device-agnostic，继续兼容该测试方式。当前 shell 可能没有 `pytest` executable，应在项目要求的 `cpl` 环境中用 `python -m pytest` 或安装好的 pytest 运行，不要因此改写测试逻辑。

## 12.4 Small-batch overfit / smoke

固定 8–32 个样本：

- 关闭 dropout 或固定随机种子；
- 运行 100–300 update；
- reconstruction 总 loss 应明显下降；
- QCEC crossing raw 不应持续上升；
- relevance 不能全部饱和为 0/1；
- proposal width 不能快速塌到 0 或膨胀到 1；
- 打印 active barrier fraction，若长期为 0，先校准 transition threshold 和 relevance margin。

## 12.5 Inference

- `model.eval()` + `torch.no_grad()` 完成全 validation forward；
- snap on/off 都能运行；
- proposal count 和 selector 输入 shape 不变；
- raw 与 snapped interval 都在 `[0,1]`；
- NLL sort index 能正确 gather QCEC eval boundaries；
- QCEC 使用完整 query role states，而非训练专属 tensor。

## 12.6 Baseline compatibility

至少验证：

1. 旧 config 无 `qcec` 字段可构造、训练、eval；
2. `qcec.enabled=false` 时旧 checkpoint strict load；
3. fixed seed/eval 下 disabled 模型输出与修改前一致；
4. QCEC loss 在 disabled 时为 0；
5. QCEC dataset 字段缺省时不尝试读 cluster index；
6. Gaussian mixture bias 为 None 时现有 tests 全通过；
7. Charades single-Gaussian baseline 不受影响。
8. 从 `qcec/` 启动时，`models.__file__`、`datasets.__file__` 和 `runners.__file__` 均解析到 `/data/chenyuan/videogrounding/w1m2/qcec`；
9. 在只包含 `qcec/`、不包含 `cpl_lrev/` 的临时工作区或干净容器中，unit test 和 forward smoke test 仍能运行。

推荐命令：

```bash
cd /data/chenyuan/videogrounding/w1m2/qcec
python -m pytest -q tests/test_qcec_clustering.py
python -m pytest -q tests/test_qcec_module.py tests/test_qcec_loss.py
python -m pytest -q tests/test_qcec_integration.py tests/test_v4_integration.py
python -m pytest -q tests/test_gaussian_mixture.py tests/test_proposal_selection.py
```

## 13. 主要风险与处理方法

| 风险 | 具体表现 | 建议处理 |
|---|---|---|
| 独立工程被做成薄封装 | `qcec/` 通过 import、symlink 或 `sys.path` 调用 `cpl_lrev` | 先复制完整基线；CI/验收用 `rg` 和 module `__file__` 检查；在无 `cpl_lrev` 的临时环境运行 smoke test |
| 簇索引与采样不一致 | bounds 与模型 200-step feature 错位 | 预处理复用同一采样 helper；metadata 校验 T/M/feature path |
| cluster ID 非连续或未按时间重编号 | scatter 错位、transition 不对应相邻事件 | 保存前按 segment start 排序并重编号；unit test 每个 ID 只有一个 run |
| word/OOV 与 role mask 错位 | predicate/entity token 指向错误词 | 在 keep-vocab 循环内同步追加 role 标记，不在原 tokenizer index 上事后映射 |
| all-masked softmax | attention 产生 NaN | predicate/entity 缺失时回退 sentence；cluster mask 至少一个有效项 |
| dropout 导致聚类随机变化 | 同视频每次簇不同 | 默认离线使用固定 backbone feature；不在 dropout 后在线聚类 |
| padding 污染池化/transition | 虚假簇和末尾边界 | 所有聚合乘 frame/cluster mask；相邻 transition mask 要求两侧都有效 |
| gradient 意外截断 | QCEC 只输出诊断但不学习 | 只 detach assignment、visual transition 和 crossing barrier；不要 detach online cluster tokens/global hint |
| `L_cross` 退化 | relevance 全相等、barrier 全零 | crossing 中默认 detach barrier；监控 active barrier fraction；相关性由下游 loss 学习 |
| `L_cross` 错罚 | 快速运动/镜头切换被当事件边界 | 必须同时满足视觉变化和 relevance 方向差；阈值按数据分布校准 |
| outer min/max 梯度稀疏 | 只有极端 component 被 crossing loss 更新 | 先观察；必要时开启独立 component-level loss，不直接替换主 loss |
| slot top-k 不可微 | relevance 排名切换导致跳变 | global adapter 是主可微路径；slot prior 为可选，bias head 零初始化、较小 scale |
| proposal slots 被相邻高分簇占满 | 多个 slot 初始化几乎重复 | 后续可在 top-k 中加时间 NMS；首版记录 selected slot center 分布 |
| snapping 降低正确 proposal IoU | 不可靠 hint 拉错边界 | 默认关闭；限制 ±2 frame；低置信不动；同时报告 raw/snapped |
| 训练/推理信息不一致 | 推理缺 query phrase/cluster | role mask 和离线簇在所有 split 同样提供；禁止使用 GT 或 train-only pseudo label |
| AMP 数值下溢 | 小 barrier 乘 sigmoid 后为 0/NaN | cosine、scatter reduction 和 crossing reduction 用 float32；分母 clamp |
| 显存增加 | `[B,M,T,D]` 或 debug states 长期保留 | scatter/bmm 避免四维展开；默认不返回 `[B,M,D]` debug tensors；M=32 |
| checkpoint missing keys | baseline strict load 到 QCEC model 失败 | 新增受控 `_load_baseline_model`；只白名单 QCEC missing keys |
| 恢复训练阶段错误 | resume 后重新冻结 backbone | checkpoint 保存 epoch；旧 checkpoint fallback 需日志告警 |
| optimizer 不含恢复参数 | 先冻结后构 optimizer 导致联合阶段参数不更新 | optimizer 构建时纳入所有参数，epoch 内只切 `requires_grad` |
| distributed 行为 | 多卡指标不 reduce 或重复写 index | 当前仓库无 DDP；QCEC loss 不需要跨卡 gather。若后续 DDP，预先构建 index，rank 0 写文件并 barrier，metrics 另做 all-reduce |
| 数据 worker 内存 | 每个 worker 拷贝大索引 | NPZ 数组只读并尽量在 fork 前加载，利用 copy-on-write；压缩 NPZ 不承诺 memory-map，索引规模本身约几十 MB |
| 当前仓库历史配置漂移 | README/test 期待 weighted，但 main/checkpoint 是 outer | QCEC 以实际 main/checkpoint 为基准；不要把修复历史漂移混入实验 commit |

## 14. 当前代码中无法静默假设的事项

## 14.1 `outer` 与 `weighted` 配置不一致

当前 `config/activitynet/main.json` 和已检查的 `cpl_lverFinal.../model-best-composite.pt` config 都是 `boundary_mode='outer'`；但 `cpl_lrev/README.md` 和部分 tests 描述/断言 weighted，README 还引用当前目录不存在的 `final_weighted_diagnostic.json` 与 checkpoint 路径。

推荐选择：QCEC 实现和首轮实验以当前实际 main/checkpoint 的 outer 配置为 baseline，新增 `qcec.json`。不要在 QCEC 工作中顺手改成 weighted，否则无法区分收益来源。

备选：若导师明确指定 weighted diagnostic checkpoint，则先恢复对应 config/checkpoint，再从它派生另一份 `qcec_weighted.json`，两套结果不要混报。

## 14.2 聚类是在 raw backbone feature 还是 `frame_fc` 后

原 proposal：`frame_fc` 后聚类。  
推荐工程实现：固定 raw C3D/I3D feature 上离线决定 assignment，`frame_fc` 后在线池化。  
原因：稳定、可缓存、避免重复计算，而且 CACMI 原论文也在固定预训练视觉 feature 上聚类。

备选：实现 `cluster_source='online_projected'`，在 eval/no-dropout projected feature 上聚类，作为 ablation；不建议首版。

## 14.3 是否立即启用 slot prior

原 proposal 包含 proposal-slot 初始化。  
推荐：接口一次设计好，但最小实验先设 `use_slot_prior=false`，确认 global hints 是否有效，再独立开启。  
原因：top-k 是离散选择，且 mixture 中不同 proposal 的 component 数不同，slot prior 会同时改变 proposal 分工，解释成本高。

## 14.4 是否奖励边界靠近 transition

原 proposal 提到“跨强变化点惩罚、靠近可信 transition 奖励”。  
推荐首版只实现方向性 `L_cross`。无 GT 时对所有 proposal 强制靠近某个 transition 容易导致候选聚集或吸附到普通镜头切换。边界靠近机制先以推理期小范围 snapping 实现，训练期 align reward 设独立权重且默认 0。

## 14.5 Query phrase 的精度

当前只有 NLTK POS，没有 dependency parser。谓词/实体 unit 只能用 POS group 近似，不应声称获得了完整 semantic role。若后续发现 role unit 无收益，可以消融为“动词 token / 名词 token / 整句”，或只用整句；不要在首版引入新的重型语言模型。

## 14.6 当前 eval 的随机词遮蔽

`CPL._mask_words()` 在 `model.eval()` 下仍调用 `np.random.choice`，所以现有 reconstruction NLL 推理带随机 mask。QCEC relevance 在该函数之前、基于未 mask 的 `enc_out` 计算，因而自身可保持确定性。是否修复 eval masking 属于现有 baseline 的独立问题，不应与 QCEC 首轮改动合并；测试 QCEC raw/snap 时必须固定 NumPy seed并保持 baseline 与 QCEC 相同逻辑。

## 15. 验收标准

实现完成至少满足：

- `/data/chenyuan/videogrounding/w1m2/qcec` 是包含训练、模型、数据加载、配置、测试、工具和评估代码的完整工程，而不是对 `cpl_lrev` 的调用模块；
- `qcec/` 中不存在对 `cpl_lrev` Python 代码的 import、symlink、`sys.path` 或 `PYTHONPATH` 依赖，并能在没有 `cpl_lrev/` 的环境中完成 unit/forward smoke test；
- 离线索引可对 ActivityNet 构建，所有簇连续且 metadata 匹配；
- QCEC-enabled batch 能完整 forward/backward，所有 loss 和 grad finite；
- `r/b^L/b^R` 的 shape、范围和 mask 正确；
- 三个提示通过 `global_hint -> residual adapter -> original head` 实际影响 center/width；
- `L_cross` 对合成 directional crossing 行为单调正确；
- QCEC disabled 时旧 baseline checkpoint strict load 且原 tests/forward 可运行；
- QCEC enabled 时可从 baseline checkpoint 受控 warm-start；
- single-Gaussian 与 mixture 均不因接口扩展而崩溃；
- snap on/off 均能完整 eval，区间合法；
- 日志包含 QCEC loss、barrier 和 crossing diagnostics；
- 首轮 ablation 能独立关闭 global adapter、slot prior、cross loss 和 snapping。

## 16. 给后续 Codex 的执行说明

1. 唯一允许写入和修改的工程目录是 `/data/chenyuan/videogrounding/w1m2/qcec`。先按第 5.0 节复制完整 baseline 骨架；`/data/chenyuan/videogrounding/w1m2/cpl_lrev` 只能只读检查，任何代码改动都不得落入该目录。
2. `qcec/` 必须是完整工程，不是插件或薄封装。禁止 `import cpl_lrev`、禁止把 `../cpl_lrev` 加到路径、禁止用 symlink 复用源码。完成复制后，从 `qcec/` 执行 baseline smoke，并核对关键 module 的 `__file__`。
3. 先阅读本文件中第 2、5、11 节；真正修改的核心文件是 `qcec/models/cpl.py`、新建的 `qcec/models/modules/qcec.py`、`qcec/models/loss.py` 和 `qcec/runners/main_runner.py`。
4. 严格按 Implementation Order 实施。先创建并验证独立工程，再做离线 index 和 module unit test，最后接 `CPL.forward`；不要一开始同时加入 adapter、slot prior、loss 和 snapping。
5. 保持 QCEC 工程内的 baseline mode：`qcec` 配置缺省/false 时不实例化 QCEC 参数、不要求 dataset index、不改变 generator 调用数值、不改变 runner selector。
6. 不要重构 `DualTransformer`、自定义 attention、LREV 或现有四组 loss；QCEC 应在独立工程内部作为 `CPL` 的组成部分接到 `h[:, -1]`。这里的“组成部分”不表示调用外部 `cpl_lrev` 模块。
7. 保留 `qcec/config/activitynet/main.json` 作为工程内 baseline，创建独立 `qcec/config/activitynet/qcec.json`。不要修改源目录 `cpl_lrev/config/activitynet/main.json`；README/test 中的 weighted 历史漂移先记录，不在同一修改中处理。
8. 从 baseline 权重初始化 QCEC 时不要调用现有 `_load_pretrained_model()`，因为它会跳过 event disentangler。实现并使用只白名单 QCEC missing keys 的 `_load_baseline_model()`；建议把初始化 checkpoint 复制到 `qcec/checkpoints/bootstrap/`。
9. 新代码禁止硬编码 `.cuda()`；使用输入 device/dtype。复制进 QCEC 工程的旧代码中已有 `.cuda()` 暂不顺手重构，以免扩大算法改动。
10. 完成每一步后先跑相应小测试。最终至少跑第 12.6 节列出的测试、一个小 batch overfit、一次完整 forward/inference smoke、QCEC on/off checkpoint compatibility，以及“无 `cpl_lrev/` 环境”的独立性测试。
11. 首个可解释实验只开 global adapter；随后依次增加 `L_cross`、slot prior、snapping。一次只改变一个机制，并同时报告短事件、高 IoU、proposal width 和 crossing ratio。
12. 若实际实现发现接口与本文不符，先以复制时锁定的 baseline 真实 shape 为准，更新 specification/测试后再改算法；不要默默增加外部特征、GT supervision、额外 caption corpus或与 QCEC 无关的大规模重构。
