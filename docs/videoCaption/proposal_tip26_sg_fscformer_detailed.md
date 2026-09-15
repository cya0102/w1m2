# QSTG / SG-FSCFormer 独立工程详细实施方案

> 对应原方案：[`proposal_tip26_sg_fscformer.md`](proposal_tip26_sg_fscformer.md)  
> 对应源码：`/data/chenyuan/videogrounding/w1m2/TIP26-SG-FSCFormer-main`  
> 对应论文：*Scene Graph-Guided SegCaptioning Transformer With Fine-Grained Alignment for Controllable Video Segmentation and Captioning*（TIP 2026）  
> 事实参考基线：`/data/chenyuan/videogrounding/w1m2/cpl_lrev`  
> 推荐实施目录：`/data/chenyuan/videogrounding/w1m2/qstg`  
> 文档性质：面向后续代码实现的 specification；本文不实施模型代码  
> 最后核对日期：2026-09-14

## 1. 目标、实施边界与推荐结论

本方案把原 proposal 中的 QSTG（Query-Subgraph Temporal Grounder）细化为可以接入 `cpl_lrev` 的弱监督时序定位模块。保留的核心研究假设是：

> 目标区间应当是 query 中 action、entity、attribute、relation 等必要短语共同得到解释的一段连通时序证据，而不只是整句重构 NLL 最低的一段视频。

它直接针对 `cpl_lrev_model_issues.md` 中四类高优先级问题：

- R@5 明显高于 R@1：用独立的细粒度 proposal quality 修复排序；
- NLL 偏好宽区间：用 reverse exclusivity 和 connectivity 惩罚额外背景；
- mixture mask 与 outer span 不一致：分别评价 mask support 和连续 span，并提供候选内连通细化；
- cPCA/event 表示 query 条件不足：用 phrase-conditioned node/edge gate 产生当前 query 专属证据图。

推荐首个完整实现包含六部分：

1. 使用现有 NLTK token/POS 结果构造固定上限的 query phrase graph，不在线调用 parser、LLM 或外部句子库；
2. 将 200-step 输入特征按连续 4 帧池化为约 50 个 temporal evidence node，构造相邻边和少量局部相似边；
3. 以 phrase node 作为 prompt，预测 phrase-node binding、query-conditioned relevance gate，并做两层关系调制图传播；
4. 用 graph global summary 以零初始化 residual adapter 修改现有 `h[:, -1]`，并在 mixture 路径中把 component 与 phrase/node 的绑定送入 component importance；
5. 对每个现有 Gaussian proposal 计算 `coverage / exclusivity / relation satisfaction / connectivity`，训练 proposal quality head 并用于推理排序；
6. 推理时可在每个原 proposal 内搜索“不跨强变化、低关系边”的最佳连通子区间，保持 proposal 数量不变。

首轮最小验证只做：

~~~text
query phrase graph
    + 50-step phrase-node binding
    + 双向 coverage/exclusivity
    + 对现有 N 个 proposal 重新排序
~~~

只有最小验证能稳定提高验证集 R@1 后，才依次打开 graph propagation、global adapter、component semantic fusion 和 connected refinement。这样能区分“细粒度绑定本身有效”与“增加参数后偶然变化”。

### 1.1 推荐使用独立 `qstg/` 工程

后续实现推荐创建：

~~~text
/data/chenyuan/videogrounding/w1m2/qstg
~~~

并满足：

- `cpl_lrev/` 只作为事实参考，不在其中直接试验性修改；
- 将运行基线所需源码真实复制到 `qstg/`，不使用软链接；
- `qstg/` 拥有自己的 `train.py`、`models/`、`datasets/`、`runners/`、`optimizers/`、`config/`、`tests/`、`tools/`、`scripts/`、`utils.py`、`vocab.py`、`requirements.txt` 和 `README.md`；
- 禁止 `from cpl_lrev...`、`import cpl_lrev...` 或运行时修改 `PYTHONPATH` 指向基线；
- HDF5 视频特征和 GloVe 文件可以继续由 config 指向共享数据路径；
- baseline checkpoint 可复制到 `qstg/checkpoints/bootstrap/` 作为一次性初始化；
- `qstg.enabled=false` 时，forward、loss、候选排序和 checkpoint 加载必须保持 baseline 兼容。

如果后续明确要求直接修改 `cpl_lrev/`，本文给出的 `qstg/...` 文件路径可一一映射回基线目录，但不建议两处同时维护。

### 1.2 SG-FSCFormer 源码事实与原 proposal 的必要校正

不能把论文概念直接按名称搬到 CPL。对 `TIP26-SG-FSCFormer-main` 的核对得到以下事实：

1. `projects/llava_sam2/models/ptgformer.py::PromptGuidedTemporalGraphFormer` 仍然要求 box prompt；`_box_to_tensor()` 只取传入 boxes 的第一个 box。无 box 时整个 graph 分支返回 `None`。
2. 主路径优先读取每帧预提取的 `.npz` scene-graph box 和 2048 维 box feature；缺失时才从 RGB frame 做 3×3 grid pooling。QSTG 只有 C3D/I3D 特征，不能复用该空间节点构建器。
3. 源码先按与 prompt box 中心的距离保留最多 `num_nodes=9` 个节点，不足时重复节点；随后按 `hard_keep_ratio=0.75` 再做硬筛选。它不是一个可直接用于文本 prompt 的通用 adaptive adaptor。
4. 每帧图边是保留节点之间的完整有向边，edge feature 来自 subject、object 和两者 box 几何；`_soft_filter_nodes()` 用 triplet self-attention 得到 association gate。
5. `_temporal_update()` 只让当前帧节点 attend 过去 `temporal_window` 帧，默认窗口为 1；随后 subject/object edge message 聚合回节点。
6. `llava_sam2.py::_fuse_ptgformer_embeddings()` 先对 graph nodes 做均值，再残差加到每个 segmentation embedding；真正的 node-slot 交互主要发生在 `mldecoder.py`。
7. `GraphGuidedIterativeQueryFormer` 的“language query”是固定数量的 segmentation/object slot，不是任意 caption word token。张量第一维实际按 frame 展开，第二维是 object/graph slot。
8. `FineGrainedMaskLinguisticDecoder` 的 BCE target 形状是 `[num_frames, num_masks, caption_slots]`。当前 `VideoBoxCapDataset` 和 `video_lisa_collate_fn` 并未构造或传递 `alignment_targets`，因此正常路径会退化为对角 identity target；它不是可直接移植的真实 word-mask 标注。
9. 源码的 multi-entity contrastive 在同一样本的 mask slot 和 word/slot embedding 间双向 softmax，并未实现跨 batch 的 query-video negatives。
10. `sg_fsc_total` 在 `llava_sam2.py` 中被 `detach()`，实际可训练梯度来自单独返回的 `llm_loss/loss_mask/loss_dice/loss_fa/loss_mc`。QSTG 不应复制这个 detached aggregate 的写法。
11. 仓库同时有 `ptgformer.py` 和 `ptgformer_roi.py`；默认 `models/__init__.py`、`llava_sam2.py` 与 config 使用的是前者。后者不能被当作默认执行链来混合分析。

因此，QSTG 借鉴的是：

- prompt 先过滤再传播；
- graph、visual memory、query 迭代交互；
- 双向的多实体/多短语绑定；
- 用关系和连通性约束输出。

QSTG 不复用：

- box prompt；
- RGB/SAM2 mask decoder；
- 2048 维 SGG box feature；
- LLM caption generation；
- 没有真实 target 时的 diagonal alignment fallback。

### 1.3 原论文、原 proposal 与推荐实现的边界

| 层级 | 内容 |
|---|---|
| SG-FSCFormer 原论文/源码 | box-centric temporal scene graph；PTGFormer；graph-guided iterative query former；mask-caption 联合解码；FA BCE；MC loss |
| 原 QSTG proposal | query phrase graph；temporal evidence graph；prompt adaptor；message passing；many-to-many binding；component quality；连通区间解码 |
| 推荐首版实现 | POS 规则 phrase graph；固定连续 temporal node；soft gate；局部稀疏图；node/proposal 双向匹配；proposal rerank |
| 推荐完整实现 | global residual adapter；component graph summary；importance bias；弱监督 FA、跨 batch MC、连通约束；候选内 connected refinement |
| 可选增强 | 离线依存分析；object detector nodes；EMA teacher；分布式 negatives；动态图结构；新增 graph-only proposals |

首轮不要加入外部 detector、spaCy/Stanza、在线 VLM、caption 生成、mask supervision 或新的标注。这些都会改变当前弱监督问题设定。

## 2. 当前 `cpl_lrev` 的真实执行链

### 2.1 入口、配置与模型构建

训练/评估入口为 `cpl_lrev/train.py`：

1. `parse_args()` 读取 config、checkpoint、eval 和 proposal selector 等参数；
2. `main()` 用 `utils.load_json()` 加载 JSON，并把 CLI loss override 写回 `args['loss']`；
3. `MainRunner(args)` 构造 dataset、model、optimizer 和 scheduler；
4. runner 将真实词表大小写入 `args['model']['config']['vocab_size']`，将最大 epoch 写入 `max_epoch`；
5. `_build_model()` 通过 `getattr(models, name)` 构造 `CPL` 并直接调用 `.cuda()`。

当前工程没有通用 registry、配置 schema、dataclass 或 loss builder。QSTG 应沿用嵌套字典，并在构造函数中显式检查：

- `node_stride >= 1`；
- `max_phrases >= 2`；
- `hidden_size % num_heads == 0`；
- `0 < keep_ratio <= 1`；
- 所有温度大于 0；
- 所有 loss weight 非负；
- `connected_refine` 只在 eval 生效。

### 2.2 Dataset 与 batch

`BaseDataset.__getitem__()` 当前执行：

~~~text
HDF5 variable-length feature
    -> 均匀分成 max_num_frames=200 段并段内平均
    -> frames_feat [200,Dv]

sentence
    -> nltk.word_tokenize + nltk.pos_tag
    -> 丢弃词表外 token
    -> words_feat [W+1,300]
    -> words_id [W]
    -> POS-based masking weights [W]
~~~

collate 后：

- `frames_feat [B,200,Dv]`，`float32`；
- `frames_len [B]`，当前通常都为 200；
- `words_feat [B,W+1,300]`；
- `words_id [B,W]`；
- `words_len [B]`；
- `weights [B,W]`。

ActivityNet 使用 500 维 C3D、`num_props=5`、Gaussian mixture；Charades-STA 使用 1024 维 I3D、`num_props=8`，默认是 single Gaussian。

`raw=[vid,duration,timestamps,sentence]` 只在 runner eval 计算指标。QSTG 训练输入不得包含 `timestamps`，保持弱监督设定。

### 2.3 `CPL.forward()` 的真实顺序

令：

- `B`：batch size；
- `T=200`：视频输入步数；
- `D=256`：hidden size；
- `W<=20`：query token 数；
- `N`：proposal 数；
- `Tp=T//4=50`：重构分支时间长度；
- `Ktotal=sum(min(n+1,Kmax))`：mixture component 总数，ActivityNet 当前为 15。

forward 顺序是：

1. 追加 `pred_vec`，由 `[B,T,Dv]` 变成 `[B,T+1,Dv]`；
2. 先对输入做 dropout，再经 `frame_fc` 得到 `[B,T+1,D]`；
3. query 首位置替换为 `start_vec`，经位置编码和 `word_fc`；
4. `DualTransformer(...,decoding=1)` 返回：
   - `enc_out [B,W+1,D]`：仅由 query causal self-attention 得到；
   - `h [B,T+1,D]`：video token attend query 后的状态；
5. `proposal_generator_feature=h[:,-1]` 直接预测 Gaussian 参数；
6. 之后才从前 T 个 projected frame 点采样为 50 步；
7. mixture 路径对每个 component 做一次 query reconstruction，以末 token summary 预测 component importance；
8. 最终 proposal mask 再进行 query reconstruction，输出 `words_logit [B*N,W,V]`；
9. negative、LREV/BECL 和 reference reconstruction 在后面继续执行。

原 proposal 中“现有 50-step h”不准确。`h` 在候选生成时仍是 201-step；50-step 只服务后续 masked reconstruction。QSTG 的推荐插入点是：

~~~text
decoding=1 之后：
    clean projected visual frames [B,T,D]
    grounded video states h[:,:T] [B,T,D]
    query states enc_out[:,1:] [B,W,D]
        -> QSTG pre-proposal graph
        -> residual-adapted h[:,-1]
        -> 原 Gaussian head

mixture component reconstruction 之后：
    component_summary [B,Ktotal,D]
    component masks/centers/widths
        -> QSTG component semantic fusion
        -> 原 importance scorer

最终 center/width/mask 之后：
    -> QSTG proposal scoring
~~~

### 2.4 必须单独保留 clean visual state

基线在 `frame_fc` 前使用 dropout。若 QSTG 的离散近邻边和跨 batch contrastive 直接使用该随机状态，同一视频的 graph 会随 forward 抖动。

推荐在 QSTG 开启时额外计算：

~~~python
raw_video = frames_feat                       # [B,T,Dv]，追加 pred_vec 前
qstg_visual_states = self.frame_fc(raw_video) # [B,T,D]，无 dropout
~~~

原 baseline 分支仍执行原有 dropout 与 `frame_fc`。这样：

- QSTG 的节点、相似边和跨样本匹配较稳定；
- baseline 的数值路径不被修改；
- `frame_fc` 被两条路径共享，QSTG loss 仍可更新它；
- `qstg.enabled=false` 时不执行额外 projection。

若双投影的计算或梯度耦合导致问题，可把 `qstg_visual_proj: Linear(Dv,D)` 作为 ablation，但不是首版默认。

### 2.5 Gaussian mixture 的接口约束

`GaussianMixtureProposalGenerator` 当前：

- `center_head: Linear(D,Ktotal)`；
- `width_head: Linear(D,N)`；
- 同 proposal 内 component 共享 width；
- `predict_components()` 返回 centers、widths、Gaussian masks；
- `combine()` 从 component reconstruction summary 预测 importance；
- `boundary_mode='outer'` 取所有 component 外包络；
- `boundary_mode='weighted'` 取 importance 加权边界；
- negative mining 始终可以使用未收缩 outer envelope。

QSTG 不替换该类，只增加全部可选、默认 `None` 的残差参数：

~~~python
predict_components(
    global_feature,
    sequence_length,
    center_logit_bias=None,  # [B,Ktotal]
    width_logit_bias=None,   # [B,N]
)

combine(
    centers,
    widths,
    component_weights,
    reconstructor_features,
    importance_logit_bias=None,  # [B,Ktotal]
)
~~~

三个 bias 都为 `None` 时必须与当前数值完全一致。

### 2.6 当前 loss、推理和 checkpoint

训练当前依次相加：

- `rec_loss`；
- `ivc_loss`；
- `event_disentanglement_loss`；
- `mixture_pull_push_loss`。

QSTG 作为第五个独立 loss 加入，不改旧函数的默认语义。

推理由 `MainRunner.eval()` 完成，不存在单独 `generate()`：

1. 计算每个 proposal reconstruction NLL；
2. 可选减去 event score；
3. 按低分优先排序；
4. center/width 转换为归一化区间；
5. 用 `nll/geometric_vote/semantic_vote` 选择 Rank-1；
6. 计算 R@1 和 R@5。

QSTG 应新增 `selection_strategy='qstg'`，而不是假设存在 caption generation API。

当前 checkpoint：

- 保存 model、config、`num_updates`；
- strict resume 要求结构一致；
- `_load_pretrained_model()` 是 V3→V4 专用途径，不适合直接充当 QSTG warm start；
- 不保存 optimizer state 和 completed epoch；
- eval 仍会调用 `_mask_words()`，其中使用 NumPy 随机采样，因此 NLL 排序默认不完全确定。

QSTG 需要独立的 baseline warm-start 和 deterministic eval 约定，见第 16 节。

## 3. Query phrase graph

### 3.1 为什么首版不引入依存分析器

当前 requirements 已依赖 NLTK，dataset 已执行 POS tagging。在线加入 spaCy、Stanza 或 LLM 会带来：

- 新模型和下载依赖；
- parser token 与 GloVe 保留 token 的二次对齐；
- 多进程 DataLoader 中的初始化成本；
- train/eval 缓存和版本一致性问题。

首版使用可复现的 POS 规则。离线依存图可以作为后续增强，但必须保存与当前词表 token 对齐的索引。

### 3.2 固定张量协议

设置 `Pmax=8`。每个样本构造：

~~~text
phrase_token_mask [Pmax,W]     bool
phrase_type       [Pmax]       int64
phrase_valid      [Pmax]       bool
phrase_required   [Pmax]       bool
query_edge_type   [Pmax,Pmax]  int64
video_group_id    scalar       int64
~~~

`phrase_type`：

| id | 类型 |
|---:|---|
| 0 | PAD |
| 1 | GLOBAL |
| 2 | ACTION |
| 3 | ENTITY |
| 4 | ATTRIBUTE |
| 5 | RELATION |

`query_edge_type`：

| id | 语义 |
|---:|---|
| 0 | 无边 |
| 1 | 无方向关联/co-occur |
| 2 | source before target |
| 3 | source after target |
| 4 | simultaneous/while |

所有 mask 的 `W` 必须对应“丢弃 OOV 并截断到 `max_num_words` 后的 `words_id` 位置”，不能直接对应原句 tokenizer index。`words/tags/weights` 应在 `__getitem__()` 内同步截断，不能等 collate 分别截断。

### 3.3 可直接编码的 phrase 抽取规则

Dataset 在现有 token/POS 循环中同时保留 `kept_tags` 和原顺序。按下列顺序构造：

1. **GLOBAL**：固定为 slot 0，覆盖所有有效 token；
2. **ACTION**：每个 `VB*` token 建一个短语，并包含紧邻的 `RB*` 和 `RP`；相邻多个 `VB*` 合并；
3. **ENTITY**：最大连续 `JJ*/NN*/CD` chunk，其中必须至少有一个 `NN*`；
4. **ATTRIBUTE**：未被单独表达且修饰 entity/action 的 `JJ*` 或 `RB*`，只在仍有空 slot 时建立；
5. **RELATION**：`IN/TO/RP`，以及词面 `before/after/then/while/during/with/into/out/of`；连接其左右最近的 action/entity；
6. 若没有 action/entity/relation，GLOBAL 同时作为唯一 required phrase；
7. 若有 content phrase，GLOBAL 保持 valid，但 `phrase_required[0]=False`，避免整句节点和内容节点重复计入 coverage。

容量超过 `Pmax` 时按以下优先级保留：

~~~text
GLOBAL
> ACTION（最多 2）
> ENTITY（最多 3）
> RELATION（最多 1）
> ATTRIBUTE（剩余位置）
~~~

同优先级按原句顺序。被舍弃 token 仍包含在 GLOBAL 中。

### 3.4 Query edge 构造

固定规则：

- GLOBAL 与每个 content phrase 建双向 type-1 边，仅用于消息传播，不计入 relation satisfaction；
- ACTION 与右侧最近 ENTITY 建 type-1 边；右侧没有 entity 时连接左侧最近 entity；
- ATTRIBUTE 连接覆盖或最近的 ENTITY/ACTION；
- RELATION 连接左右最近两个 content phrase；
- `before/then` 使用 type 2；
- `after` 使用 type 3；
- `while/during/with` 使用 type 4；
- 其余 relation 使用 type 1。

relation satisfaction 只统计 content-content 边；所有 GLOBAL 边在对应 mask 中排除。

### 3.5 Phrase embedding

使用 `enc_out[:,1:] [B,W,D]`，排除 learned start token。对 phrase `p`：

\[
q_p =
\frac{\sum_w R_{pw}H_w}
{\sum_wR_{pw}+\epsilon}
+ e^{type}_p,
\]

其中 `e_type` 是 `Embedding(6,D)`。

再进行一层 query graph update：

\[
\bar q_p =
LN\left(q_p+
MHA(q_p,\{q_j:E_{pj}>0\},\{q_j:E_{pj}>0\})\right).
\]

要求：

- padding phrase 不进入 attention；
- 只有 GLOBAL 时不得产生全 mask softmax；
- query graph layer 默认 1 层；
- GLOBAL 可作为 fallback key；
- phrase pooling 在 `_mask_words()` 之前执行，避免随机遮词改变 phrase 定义。

## 4. Temporal evidence graph

### 4.1 连续 temporal node

首版不做在线聚类，按固定连续步长池化。默认：

~~~text
T = 200
node_stride = 4
M = ceil(T / node_stride) = 50
~~~

对于一般 `frames_len`，节点 `m` 覆盖：

\[
I_m=[4m,\min(4m+4,T_b)).
\]

分别池化：

\[
v_m = mean_{t\in I_m}F^{clean}_t,
\qquad
g_m = mean_{t\in I_m}H^{grounded}_t.
\]

得到：

~~~text
visual_nodes   [B,M,D]  # query-independent，供负样本与视觉边
grounded_nodes [B,M,D]  # 当前 query-conditioned，供正样本 proposal
node_bounds    [B,M,2]  # 归一化 [start,end]
node_mask      [B,M]    # bool
~~~

融合初始状态：

\[
z_m^0 =
LN\left(
g_m + W_z[v_m;g_m;v_m\odot g_m]
\right).
\]

`W_z` 最后一层零初始化，使初始 `z^0=g`。无效 node 全部置零。

### 4.2 视觉变化和边界特征

相邻节点变化：

\[
d_m =
1-\cos(stopgrad(v_m),stopgrad(v_{m+1})).
\]

在当前 batch/video feature 尺度上标准化：

\[
\hat d_m =
\sigma((d_m-\theta_d)/\tau_d).
\]

默认 `transition_threshold=0.20`、`transition_temperature=0.05`，但首次训练前必须抽样记录 C3D 和 I3D 的分布。若两个数据集分布明显不同，在各自 config 中单独设阈值。

变化分数默认 detach，只作为结构和连通先验；不得让模型通过把相邻视觉特征全部拉成相同来消除 barrier。

### 4.3 图边

每个有效节点至少有：

- self edge；
- `m -> m-1`；
- `m -> m+1`。

可选局部相似边：

1. 候选节点限制在 `1 < |i-j| <= max_temporal_hop`，默认 `max_temporal_hop=4`；
2. 在 clean `visual_nodes` 上计算 cosine；
3. 每个 source 保留最多 `similar_topk=2` 个；
4. cosine 必须不低于 `similarity_threshold=0.5`；
5. top-k index 和 adjacency detach。

将邻接表保存成固定最大度数的张量：

~~~text
neighbor_index [B,M,Kedge]  long
neighbor_type  [B,M,Kedge]  long
neighbor_mask  [B,M,Kedge]  bool
Kedge = 1(self) + 2(adjacent) + similar_topk = 5
~~~

边界位置缺少的邻居用 index 0 padding，并由 `neighbor_mask` 屏蔽。这样无需引入 torch-geometric，也不必构造昂贵的完整边特征。

边特征：

\[
e_{ij}=MLP[
v_i;
v_j;
|v_i-v_j|;
v_i\odot v_j;
\Delta t;
type_{ij};
\hat d_{ij}
].
\]

推荐用 `neighbor_index` gather `v_j`，只构造 `[B,M,Kedge,D]`。`[B,M,M]` 仅用于小型 relation kernel/诊断；禁止构造无边位置的 `[B,M,M,4D]` 大拼接张量。B=32、M=50、D=256 时，后者会额外占用数百 MB，不适合作为首版实现。

## 5. Prompt adaptor、binding 与图传播

### 5.1 跨样本 phrase-node logits

为了支持 batch 内错误 query/video negatives，不能只计算 diagonal pair。对 query 样本 `a` 和 video 样本 `b`：

\[
\ell_{abpm}
=
\frac{
\langle W_q\bar q_{ap},W_vv_{bm}\rangle
}{\tau_{bind}}.
\]

张量：

~~~text
pair_binding_logits [Bq,Bv,P,M]
~~~

默认 `binding_temperature=0.10`。计算投影后用 `einsum('bpd,vmd->bvpm')`；B=32、P=8、M=50 时约 41 万个 logit，内存可控。

`enc_out` 在 `decoding=1` 中先由 query 自注意力得到，因此可用于跨 video 匹配；video 侧必须使用 clean `visual_nodes`，不能使用已经被正确 query 条件化的 `grounded_nodes`。

同样本 diagonal：

~~~python
pre_binding_logits = pair_binding_logits[
    torch.arange(B), torch.arange(B)
]  # [B,P,M]
pre_binding_prob = torch.sigmoid(pre_binding_logits)
~~~

### 5.2 Query-conditioned relevance gate

只对 required phrase 聚合。先对 phrase 维做归一化 smooth maximum：

\[
u_m =
\tau_r\left[
\log\sum_{p\in P_{req}}\exp(B^{pre}_{pm}/\tau_r)
-\log |P_{req}|
\right],
\qquad
r_m=\sigma(u_m-b_r).
\]

`b_r` 是可学习标量，初始化为 0；`tau_r=0.1`。减去 `log |P_req|` 可避免 phrase 数量越多 relevance 天然越大。必要短语“全部被覆盖”的约束由 proposal coverage 单独负责，不能在同一 node 上对所有 phrase 概率求乘积，因为 action 和 entity 的证据可以落在相邻而非完全相同的时刻。

参考 SG-FSCFormer 的 hard filter，但首版使用无 straight-through 的 soft top-ratio：

1. 对有效 `r_m` 求第 `ceil(keep_ratio*M_valid)` 大阈值 `kappa`；
2. `kappa.detach()`；
3. 

\[
\gamma_m =
\gamma_{floor}+
(1-\gamma_{floor})
\sigma((r_m-\kappa)/\tau_g).
\]

默认：

~~~text
keep_ratio = 0.50
gate_floor = 0.05
gate_temperature = 0.10
~~~

`gate_floor` 保留背景和边界上下文。`hard_keep_ratio=0.75` 只作为源码参考，不应机械照搬，因为 temporal node 和空间 object node 的数量与含义不同。

### 5.3 Query relation 对 video edge 的调制

对每条 content query edge `p -> q`，构造 node-pair affinity：

\[
A^{pq,pre}_{ij}=P^{pre}_{pi}P^{pre}_{qj}K_{type(p,q)}(i,j).
\]

其中：

- type 1：`exp(-|i-j|/lambda_rel)`；
- type 2 before：仅 `j>=i`，并乘相同距离衰减；
- type 3 after：仅 `j<=i`；
- type 4 simultaneous：`exp(-|i-j|/lambda_sim)`，且 `lambda_sim < lambda_rel`。

多条边取均值，得到 `pre_relation_edge_bias [B,M,M]`。无 content edge 时全零，并返回 `relation_valid=false`。

该 bias 只加在已有 temporal adjacency 上，不凭 query relation 创建任意远距离边。

### 5.4 两层 temporal graph propagation

每层：

\[
a_{ij}^{(l)}
=
\frac{\langle W_Qz_i,W_Kz_j\rangle}{\sqrt{d_h}}
+w_e^Te_{ij}
+\lambda_r A^{rel}_{ij}
+\log(\gamma_j+\epsilon).
\]

对非 adjacency、无效 node 置 `-inf`，然后：

\[
\tilde z_i=\sum_j softmax_j(a_{ij})W_Vz_j,
\]

\[
z_i^{l+1}
=LN(z_i^l+\gamma_iW_O\tilde z_i),
\]

再接两层 FFN residual。默认：

~~~text
num_graph_layers = 2
num_graph_heads = 4
relation_bias_scale = 0.5
graph_dropout = 0.1
~~~

所有有效节点都有 self edge，因此不会出现全 `-inf` 行。padding node 输出必须重新置零。

### 5.5 最终 binding

图传播后，使用 phrase 和 graph node 重新计算：

\[
B_{pm} =
\frac{\langle W_b^q\bar q_p,W_b^zz_m^L\rangle}
{\tau_{final}}.
\]

用第 5.2 节同样的 normalized log-mean-exp 从 final `B` 重新计算 `node_relevance`，并用 final binding 重新计算供 quality/decode 使用的 `relation_edge_affinity`。传播前的值命名为 `pre_node_relevance` 和 `pre_relation_edge_bias`，只用于构造 `node_gate` 和 graph attention；传播后的值用于 global summary、component/proposal quality 和 connected decode。不要用一个键混合两种语义。

返回 logits 和 probability，不要只返回 sigmoid 后的概率：

~~~text
binding_logits [B,P,M]
binding_prob   [B,P,M]
pre_node_relevance [B,M]
node_relevance [B,M]
node_gate       [B,M]
graph_nodes     [B,M,D]
~~~

loss 中需要 logits 做稳定的 `logsumexp/softplus`，quality 中使用 probability。

## 6. QSTG 如何接入现有候选生成器

### 6.1 Global graph summary 与 residual adapter

计算：

\[
g_{rel}=
\frac{\sum_m r_mz_m}{\sum_mr_m+\epsilon},
\]

\[
g_{act}=
\frac{\sum_{p:type=ACTION}\sum_mP_{pm}z_m}
{\sum_{p:type=ACTION}\sum_mP_{pm}+\epsilon},
\]

\[
g_{ent}=
\frac{\sum_{p:type=ENTITY}\sum_mP_{pm}z_m}
{\sum_{p:type=ENTITY}\sum_mP_{pm}+\epsilon},
\]

\[
g_{bdry}=
\frac{\sum_m \hat d_m|r_m-r_{m+1}|(z_m+z_{m+1})/2}
{\sum_m\hat d_m|r_m-r_{m+1}|+\epsilon}.
\]

缺失 ACTION/ENTITY 时回退 `g_rel`。拼接后：

\[
g_{qstg}=W_g[g_{rel};g_{act};g_{ent};g_{bdry}].
\]

现有 proposal feature 为 `g=h[:,-1]`：

\[
g' = g+W_o GELU(W_i[g;g_{qstg}]).
\]

`W_o` 的 weight 和 bias 必须初始化为 0。这样从 baseline warm start 时，未训练 QSTG 不会立刻改变 Gaussian 参数。

### 6.2 Proposal slot prior

从 `node_relevance` 中用半径 2 的 temporal NMS 选择 N 个峰：

~~~text
slot_node_index   [B,N]
slot_feature      [B,N,D]
slot_prior_center [B,N]
slot_prior_width  [B,N]
~~~

center 使用节点 bounds 中点；width 初值取：

\[
w_n^{prior}=
clip(2/M + local\_support\_width,0.02,0.5).
\]

若有效峰少于 N，按剩余 relevance 顺序补足；仍不足时重复最高峰，但记录 `slot_duplicate_mask`。

mixture 模式为每个 proposal 建一个小 head：

~~~python
self.center_bias_heads = nn.ModuleList([
    nn.Linear(D, component_counts[n])
    for n in range(N)
])
self.width_bias_head = nn.Linear(D, 1)
~~~

拼接得到：

~~~text
center_logit_bias [B,Ktotal]
width_logit_bias  [B,N]
~~~

所有 bias head 最后一层零初始化，并乘固定 `proposal_prior_scale`，默认完整方案为 0.25，最小验证为 0。

single-Gaussian 路径：

~~~text
single_logit_bias [B,N,2]
~~~

在 `fc_gauss` 输出 reshape 为 `[B,N,2]` 后、sigmoid 前相加。

### 6.3 Component-node membership

对每个 mixture component 的 center `mu_k` 和 width `w_k`，直接在 temporal node 中点 `u_m` 上计算 Gaussian：

\[
A^{comp}_{km}
=
\exp\left[
-\frac12
\left(
\frac{u_m-\mu_k}{\max(w_k/\sigma,0.01)}
\right)^2
\right].
\]

随后按每个 component 的最大值归一化，得到：

~~~text
component_node_membership [B,Ktotal,M]
~~~

不要假设 `M==Tp`；在 `node_stride` 改变或可变长度时，按 node position 重新计算比直接 reshape 更安全。

### 6.4 Component 的 phrase 归属与 graph summary

先把每个 phrase 的 final binding logits 在时间维归一化：

\[
U_{pm}=softmax_m(B_{pm}/\tau_{cov}).
\]

component 对 phrase 的 coverage：

\[
C^{comp}_{kp}
=
\sum_m A^{comp}_{km}U_{pm}.
\]

归一化为 phrase assignment：

\[
D_{kp}
=softmax_p(C^{comp}_{kp}/\tau_{role}).
\]

softmax 只在 required phrase 上执行；有 content phrase 时 GLOBAL 和 padding 均置 `-inf`。只有 GLOBAL fallback 时 assignment 为 1。

component graph summary：

\[
s_k^{graph}
=
\frac{\sum_mA^{comp}_{km}r_mz_m}
{\sum_mA^{comp}_{km}r_m+\epsilon}.
\]

component exclusivity：

\[
X_k
=
\frac{\sum_mA^{comp}_{km}r_m}
{\sum_mA^{comp}_{km}+\epsilon}.
\]

返回：

~~~text
component_graph_summary [B,Ktotal,D]
component_phrase_score  [B,Ktotal,P]
component_assignment    [B,Ktotal,P]
component_exclusivity   [B,Ktotal]
~~~

### 6.5 残差增强 reconstruction summary

现有 `component_summary [B,Ktotal,D]` 来自 component-conditioned query reconstruction。推荐：

\[
\Delta s_k =
W_{co}\,GELU\left(
W_{ci}[s_k^{rec};s_k^{graph};D_k;X_k]
\right),
\]

\[
\hat s_k=s_k^{rec}+\Delta s_k.
\]

`W_co` 零初始化。`hat s` 继续送入原 `importance_projection/importance_score`。

同时可产生：

~~~text
importance_logit_bias [B,Ktotal]
~~~

在 `combine()` 中按 proposal slice 加到 `group_scores`。该 head 也零初始化，固定乘 `importance_bias_scale=0.25`。

建议实施顺序：

1. 只做 `component_summary` residual；
2. 验证 component importance entropy 和 R@1；
3. 再打开 direct importance bias。

### 6.6 Semantic role diversity

对一个含多个 component 的 proposal，若 query 至少有两个 required content phrase，低权重惩罚 assignment 完全相同：

\[
L_{role}
=
\frac1{|Pairs|}
\sum_{k\ne k'}
\cos(D_k,D_{k'}).
\]

约束：

- 只统计同一 proposal 内的有效 component pair；
- 用 reconstruction soft selector 给 proposal 加权；
- required phrase 少于 2 时返回零；
- 默认权重 0.02；
- 不要求严格 one-to-one，因为同一 action 可能跨多个 component。

它用于给 mixture component 语义分工，不替换现有的几何 pull/push loss。

## 7. Proposal 级双向绑定与质量

### 7.1 Proposal-node membership

最终 `gauss_weight [B*N,Tp]` reshape 为 `[B,N,Tp]`。将其线性插值到 M 个 node 中点，并归一化：

~~~text
proposal_node_membership A [B,N,M]
~~~

single 与 mixture 共用该接口。对最终边界 `[s_n,e_n]` 还需保留一个解析 soft box：

\[
\tilde A_{nm}=
\sigma((u_m-s_n)/\eta)
\cdot
\sigma((e_n-u_m)/\eta).
\]

推荐：

- coverage/exclusivity 使用实际 Gaussian/mixed mask `A`；
- connectivity/crossing 使用解析 soft box `tilde A`；
- 同时记录两者差异，防止 mixture outer hull 与稀疏 mask 语义不一致。

### 7.2 Phrase coverage

先使用第 6.4 节的时间归一化 phrase evidence distribution `U_pm`。对 required phrase：

\[
C_{np}
=\sum_m A_{nm}U_{pm}.
\]

proposal coverage：

\[
C_n=
\frac{\sum_p required_p C_{np}}
{\sum_p required_p+\epsilon}.
\]

这是 `query -> proposal` 方向：query 的每个必要语义是否在候选中有证据。

实现时先把 padding node 的 logit 置 `-inf`，再对 M 维 softmax。若实际 proposal membership 的峰值不是 1，先按 proposal 最大值归一化。`C_np` 始终位于 `[0,1]`，也不会像 50 个中等概率的 noisy-OR 一样在初始化时饱和。

### 7.3 Node explainability / exclusivity

每个 node 被某个 required phrase 解释的分数与 gate 使用相同的归一化 smooth maximum，但使用 final binding logits：

\[
E_m=
\sigma\left(
\tau_e[
\log\sum_{p\in P_{req}}\exp(B_{pm}/\tau_e)
-\log|P_{req}|
]-b_e
\right).
\]

`b_e` 可与 gate relevance bias 共享或独立；推荐独立，初始化为 0。

proposal exclusivity：

\[
X_n=
\frac{\sum_mA_{nm}E_m}
{\sum_mA_{nm}+\epsilon}.
\]

这是 `proposal -> query` 方向：候选内部的时间证据是否都能被 query 解释。宽候选包含额外背景时，`X` 应下降。

为避免将低信息、低幅度 Gaussian 尾部也当作“候选内部”，计算 `X` 前将 `A<0.1` 的 membership 置零；该阈值只作用于 quality feature，不作用于 reconstruction。

### 7.4 Relation satisfaction

对 query content edge `p->q`，只在 temporal graph 的有效邻接边 `(i,j) in E_video` 上计算：

\[
R_{n,pq}
=
\frac{
\sum_{(i,j)\in E_{video}}
A_{ni}A_{nj}P_{pi}P_{qj}K_{type}(i,j)
}{
\sum_{(i,j)\in E_{video}}
A_{ni}A_{nj}P_{pi}P_{qj}
+\epsilon
}.
\]

最终 `R_n` 是有效 content edge 的均值。用 `neighbor_index` gather 后复杂度是 `O(B*N*Pedge*M*Kedge)`，不构造 `[B,N,Pedge,M,M]`。无 content edge：

~~~text
relation_satisfaction = 1
relation_valid = 0
~~~

quality head 必须同时接收 `relation_valid`，不能把“没有关系约束”和“关系完全满足”混为一种输入。

### 7.5 Connectivity

相邻 node barrier：

\[
\beta_m=
\hat d_m
\cdot |E_m-E_{m+1}|
\cdot (1-A^{rel}_{m,m+1}).
\]

proposal 跨 barrier 的程度：

\[
cross_{nm}
=
\tilde A_{nm}\tilde A_{n,m+1}\beta_m.
\]

\[
H_n=
\exp\left(
-\frac{\sum_mcross_{nm}}
{\sum_m\tilde A_{nm}\tilde A_{n,m+1}+\epsilon}
\right).
\]

`H` 高表示内部 evidence 在 graph 上连通。只用视觉变化会误罚镜头运动，因此 barrier 同时依赖 query explainability 差和 relation affinity。

### 7.6 Proposal quality head

对每个 proposal 构造：

~~~text
[C, X, R, relation_valid, H,
 mean_inside_relevance,
 boundary_relevance_left,
 boundary_relevance_right,
 normalized_width,
 mask_span_disagreement]
~~~

输入两层 MLP：

~~~python
LayerNorm(10)
Linear(10, 64)
GELU()
Dropout(0.1)
Linear(64, 1)
~~~

得到：

~~~text
proposal_quality_logit [B,N]
proposal_analytic_score [B,N]
~~~

最后一层零初始化。解析分数默认：

\[
Q_n^{analytic}
=0.35C_n+0.30X_n+0.20R_n+0.15H_n.
\]

当无 relation edge 时，将 relation 的 0.20 权重按比例分配给 C/X，而不是无条件加满 0.20。

### 7.7 Cross-query proposal quality

训练 quality head 不能使用 GT timestamp。利用 `pair_binding_logits [Bq,Bv,P,M]`，对每个 query-video-proposal 组合计算同样的 C/X/R/H：

~~~text
pair_quality_features [Bq,Bv,N,10]
pair_quality_logits   [Bq,Bv,N]
~~~

视频 `v` 的 proposal soft selector 来自其正确 query reconstruction NLL：

\[
\pi_{vn}
=softmax(-NLL_{vn}/\tau_{rec}),
\]

并默认 detach。池化：

\[
S_{qv}^{prop}
=
\sum_n\pi_{vn}Q_{qvn}.
\]

随后用 batch 对角为正样本训练双向 InfoNCE。这样 quality head 学的是“该 proposal 是否支持这个 query”，而不是拟合不可见 GT IoU。

### 7.8 Same-video false negatives

ActivityNet 中同一个视频常有多个 query。Dataset 为每个唯一 `vid` 建稳定整数 `video_group_id`。在 `B×B` contrastive logits 中：

- 对角是正样本；
- `video_group_id` 相同但不是对角的位置设为 ignore；
- 其余视频作为负样本。

不能把同视频其他 query 默认当负样本；它可能描述重叠事件。若后续要做 same-video hard negative，必须由 proposal binding 或已知 query 互斥性产生，而不是仅按 vid 判断。

## 8. 弱监督损失

### 8.1 总损失

\[
L=
L_{rec}
+L_{ivc}
+L_{event}
+L_{mixture}
+\lambda_{qstg}L_{qstg}.
\]

\[
L_{qstg}
=
\lambda_{mc}L_{mc}
+\lambda_{pq}L_{pq}
+\lambda_{fa}L_{fa}
+\lambda_{ex}L_{ex}
+\lambda_{gate}L_{gate}
+\lambda_{conn}L_{conn}
+\lambda_{role}L_{role}.
\]

所有项在 QSTG 关闭或必要 tensor 为 `None` 时返回 `words_logit.sum()*0.0`，保证 device/dtype/gradient graph 正确。

所有 mean、entropy 和比例只统计对应 `node_mask/phrase_valid/phrase_required/component_valid_mask` 的有效位置；禁止先把 padding 置零再对完整固定长度直接求均值。

### 8.2 Node-level symmetric multi-entity contrastive `L_mc`

对 `pair_binding_logits` 构造全视频双向匹配：

1. phrase→node：每个 required phrase 对有效 node 做 temperature log-sum-exp，再对 phrase 平均；
2. node→phrase：先选 pair-specific relevance 最高的 `ceil(keep_ratio*M)` 个 node，每个 node 对 required phrase 做 log-sum-exp，再平均；
3. 两个方向均值为 `S_qv_node`。

对 `S [B,B]` 做：

\[
L_{mc}
=
\frac12
\left[
CE(S/\tau_{mc},diag)
+CE(S^T/\tau_{mc},diag)
\right].
\]

same-video off-diagonal mask 为 `-inf`。batch 中没有合法负样本时返回零并记录计数。

这是对 SG-FSCFormer multi-entity contrastive 的时序化实现；它使用跨视频 negatives，避免源码默认 diagonal target 在无标注时失去语义。

### 8.3 Proposal-level contrastive `L_pq`

使用第 7.7 节 `S_qv_prop` 做同样的对称 InfoNCE。`L_mc` 训练全视频 phrase-node 表示，`L_pq` 训练候选级 quality，不应合成一个不可消融的 loss。

### 8.4 Fine-grained alignment `L_fa`

原 SG-FSCFormer 可以用 mask-caption target 做 BCE；当前 grounding 没有 phrase-frame GT，因此推荐使用“候选内证据强于候选外证据”的 bag-level ranking，而不是伪造逐 node hard label。

对 proposal `n` 和 phrase `p`：

\[
s^{in}_{np}
=
\tau\log\sum_m
\exp((B_{pm}+\log(A_{nm}+\epsilon))/\tau),
\]

\[
s^{out}_{np}
=
\tau\log\sum_m
\exp((B_{pm}+\log(1-A_{nm}+\epsilon))/\tau).
\]

\[
L_{fa}
=
\frac{
\sum_{bnp}
\pi_{bn}required_{bp}
softplus((\delta_{fa}-s^{in}_{bnp}+s^{out}_{bnp})/\tau_{fa})
}{
\sum_{bnp}\pi_{bn}required_{bp}+\epsilon
}.
\]

`pi` 由 reconstruction NLL 产生并 detach；默认只保留 NLL 最佳与次佳差值超过 `selector_confidence_margin=0.05` 的样本，否则该样本 `L_fa` 权重降到 0.25。

这仍然是弱监督：proposal 来自模型自身。必须在 `L_mc` 已预热后再打开，避免早期错误 proposal 固化 binding。

### 8.5 Reverse explainability `L_ex`

\[
L_{ex}
=
\frac1B\sum_{bn}
\pi_{bn}
\frac{\sum_mA_{bnm}(1-E_{bm})}
{\sum_mA_{bnm}+\epsilon}.
\]

它显式惩罚 proposal 内无法由任何 required phrase 解释的节点。单独使用会促使所有 binding 变高，所以必须与跨 batch `L_mc` 联合，且默认权重不超过 0.05。

### 8.6 Gate regularization `L_gate`

包含两项：

1. gate mass 范围：

\[
L_{budget}
=
[r_{min}-mean(\gamma)]_+
+[mean(\gamma)-r_{max}]_+,
\]

默认 `r_min=0.10,r_max=0.70`；

2. 稳定区间平滑：

\[
L_{smooth}
=mean_m
(1-\hat d_m)|\gamma_m-\gamma_{m+1}|.
\]

总 `L_gate=L_budget+0.5L_smooth`。不要在强视觉变化处强制 gate 平滑。

### 8.7 Connectivity loss `L_conn`

对 `H_n`：

\[
L_{conn}=
\sum_{bn}\pi_{bn}(1-H_{bn})/B.
\]

barrier、transition score 和 relation edge affinity 默认 detach，使该 loss 主要移动 center/width 或调整 proposal adapter，而不是通过消除 barrier 逃避。

### 8.8 Role diversity `L_role`

按第 6.6 节计算。component assignment 可保留梯度；proposal selector detach。若 `use_gaussian_mixture=false`，返回零。

### 8.9 推荐初始权重

~~~text
qstg_total_weight        = 1.0
qstg_mc_weight           = 0.05
qstg_proposal_weight     = 0.05
qstg_fa_weight           = 0.10
qstg_explain_weight      = 0.03
qstg_gate_weight         = 0.01
qstg_connect_weight      = 0.05
qstg_role_weight         = 0.02
~~~

这些是起点，不是论文复现超参。`L_fa/L_ex/L_conn/L_role` 应在 warm-up 后 ramp，`L_mc` 可从第 1 个 QSTG warm-up epoch 开始。

## 9. 连通区间解码

### 9.1 为什么不能直接对两个高分簇取 outer hull

ActivityNet mixture 的 `boundary_mode='outer'` 会把最左和最右 component 包成连续区间。若两个 component 中间隔着：

- 强视觉变化；
- query explainability 明显下降；
- query relation graph 不支持连接；

outer hull 会把背景也包含进来。QSTG 不应简单取所有高 binding node 的 min/max。

### 9.2 候选内 connected refinement

默认保持 N 个原 proposal，每个 proposal 只在自己的软支持范围内细化：

1. 取 `A_nm>=0.1` 或解析 soft box `tilde A>=0.5` 的 node；
2. 在相邻 node 间，当 `barrier>=hard_barrier_threshold=0.65` 且 `relation_affinity<0.40` 时切断；
3. 得到若干连续 component；
4. 对每个连续 component `[i,j]` 重新计算 C/X/R/H；
5. 打分：

\[
score(i,j)=
0.40C+0.30X+0.20R+0.10H
-\lambda_{move}(|s'-s|+|e'-e|).
\]

默认 `lambda_move=0.1`；

6. 只接受 coverage 不低于 raw proposal coverage 的 `coverage_keep_ratio=0.9` 的 component；
7. 选最高分 component，将其首尾 node bounds 作为 refined span；
8. 若无合法 component、宽度小于 `min_width=0.01` 或移动超过 `max_refine_shift=0.10`，回退 raw span。

返回：

~~~text
qstg_eval_center  [B,N]
qstg_eval_width   [B,N]
qstg_refine_mask  [B,N]
qstg_refine_shift [B,N,2]
~~~

该过程只在 `not self.training` 且 `connected_refine=true` 时执行，并放在 `torch.no_grad()` 下。

### 9.3 首版默认关闭 refinement

实验顺序：

1. raw proposal + NLL；
2. raw proposal + QSTG rerank；
3. raw proposal + QSTG rerank + connected refinement。

必须同时报告 raw 与 refined 指标。若 refinement 提升 R@1 但显著降低 oracle R@5，说明切分过强，不应默认开启。

## 10. 推荐模块接口

### 10.1 `QueryPhraseGraphEncoder`

~~~python
class QueryPhraseGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_phrase_types: int = 6,
        num_edge_types: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        ...

    def forward(
        self,
        query_states: torch.Tensor,       # [B,W,D]
        query_mask: torch.Tensor,         # [B,W]
        phrase_token_mask: torch.Tensor,  # [B,P,W]
        phrase_type: torch.Tensor,        # [B,P]
        phrase_valid: torch.Tensor,       # [B,P]
        query_edge_type: torch.Tensor,    # [B,P,P]
    ) -> Dict[str, torch.Tensor]:
        ...
~~~

返回 `phrase_features [B,P,D]`、`phrase_mask`、`content_edge_mask`。

### 10.2 `TemporalEvidenceGraphEncoder`

~~~python
class TemporalEvidenceGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        node_stride: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        max_temporal_hop: int = 4,
        similar_topk: int = 2,
        similarity_threshold: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        ...

    def forward(
        self,
        visual_states: torch.Tensor,      # [B,T,D]
        grounded_states: torch.Tensor,    # [B,T,D]
        frame_mask: torch.Tensor,         # [B,T]
        phrase_features: torch.Tensor,    # [B,P,D]
        phrase_valid: torch.Tensor,       # [B,P]
        phrase_required: torch.Tensor,    # [B,P]
        query_edge_type: torch.Tensor,    # [B,P,P]
    ) -> Dict[str, torch.Tensor]:
        ...
~~~

### 10.3 `QSTGModule`

~~~python
class QuerySubgraphTemporalGrounder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_proposals: int,
        max_components: int,
        component_counts: Optional[Sequence[int]] = None,
        max_phrases: int = 8,
        node_stride: int = 4,
        num_graph_layers: int = 2,
        num_graph_heads: int = 4,
        keep_ratio: float = 0.5,
        gate_floor: float = 0.05,
        binding_temperature: float = 0.1,
        proposal_prior_scale: float = 0.0,
        importance_bias_scale: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        ...

    def forward_pre_proposal(
        self,
        global_feature: torch.Tensor,
        visual_states: torch.Tensor,
        grounded_states: torch.Tensor,
        frame_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        phrase_token_mask: torch.Tensor,
        phrase_type: torch.Tensor,
        phrase_valid: torch.Tensor,
        phrase_required: torch.Tensor,
        query_edge_type: torch.Tensor,
        video_group_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ...

    def enhance_components(
        self,
        component_summary: torch.Tensor,
        component_centers: torch.Tensor,
        component_widths: torch.Tensor,
        qstg_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ...

    def score_proposals(
        self,
        gauss_weight: torch.Tensor,
        center: torch.Tensor,
        width: torch.Tensor,
        qstg_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ...
~~~

不要把所有阶段塞进一个依赖二十多个 positional arguments 的 `forward()`。三个显式阶段与 CPL 当前生成顺序一致，也便于单元测试。

### 10.4 QSTG 输出字典

pre-proposal 至少返回：

~~~text
phrase_features
phrase_valid
phrase_required
query_edge_type
video_group_id
visual_nodes
grounded_nodes
graph_nodes
node_bounds
node_mask
transition_score
neighbor_index
neighbor_type
neighbor_mask
pre_relation_edge_bias
relation_edge_affinity
pair_binding_logits
pre_binding_logits
pre_binding_prob
binding_logits
binding_prob
pre_node_relevance
node_relevance
node_gate
global_hint
adapted_global_feature
center_logit_bias
width_logit_bias
single_logit_bias
node_pair_score
~~~

component 阶段追加：

~~~text
component_node_membership
component_graph_summary
component_phrase_score
component_assignment
component_exclusivity
enhanced_component_summary
importance_logit_bias
~~~

proposal 阶段追加：

~~~text
proposal_node_membership
proposal_soft_box
proposal_coverage_per_phrase
proposal_coverage
proposal_exclusivity
proposal_relation
proposal_relation_valid
proposal_connectivity
proposal_analytic_score
proposal_quality_logit
pair_quality_logits
qstg_barrier
~~~

无效 node/phrase/component 的连续值置零，并始终返回对应 mask。

## 11. 端到端数据流

### 11.1 Dataset / collate

~~~text
sentence
    -> NLTK tokenize/POS
    -> vocabulary filtering
    -> words / tags 对齐
    -> fixed-P query phrase graph
    -> phrase_token_mask / type / required / edge_type

video id
    -> stable video_group_id

HDF5 feature
    -> 原 200-step sampling

collate
    -> 原 net_input
    + phrase tensors
    + video_group_id
    + optional sample_uid
~~~

`sample_uid` 仅用于 deterministic eval masking，不包含 GT 信息。推荐用 dataset index 的稳定值；`video_group_id` 用当前 split 中 vid 的有序映射即可。若需跨 split 可比，再改为标准库 `hashlib.blake2b` 的 63-bit 稳定 hash，不能用受进程随机种子影响的 Python `hash()`。

### 11.2 Training forward

~~~text
raw frames [B,T,Dv]
    ├─ 原 dropout -> frame_fc -> frames_with_pred
    └─ QSTG enabled: clean frame_fc -> visual_states

words
    -> 原 word_fc/position
    -> DualTransformer decoding=1
       ├─ query_states = enc_out[:,1:]
       ├─ grounded_states = h[:,:T]
       └─ baseline_global = h[:,-1]

QSTG.forward_pre_proposal()
    -> phrase graph
    -> temporal nodes/edges
    -> pair binding + diagonal gate
    -> graph propagation
    -> adapted_global + optional proposal biases

原 Gaussian head
    -> component centers/widths/masks

原 per-component reconstruction
    -> component_summary

QSTG.enhance_components()
    -> enhanced_component_summary
    -> optional importance bias

原 mixture.combine()
    -> gauss_weight / center / width

QSTG.score_proposals()
    -> C/X/R/H / quality logits / pair quality

原 proposal reconstruction + negative + LREV
    -> words_logit
    -> proposal_reconstruction_nll [B,N]

runner
    -> old four losses
    -> qstg_loss
    -> backward
~~~

### 11.3 Evaluation

~~~text
同一 forward
    -> deterministic masked reconstruction
    -> reconstruction NLL
    -> event score
    -> QSTG quality/analytic score
    -> combined score
    -> sort N proposals
    -> optional connected refinement
    -> raw 与 refined R@1/R@5
~~~

### 11.4 不使用 GT 的边界

以下信息不得进入 model forward、quality target 或 connected decoder：

- GT timestamp；
- GT IoU；
- 测试集 query 对应的人工事件边界；
- 用 GT 选择 graph threshold；
- 用 test 指标选择 loss weight。

GT 仅用于 runner 的验证/测试指标。阈值和 selector weight 必须在 validation split 上选择。

## 12. 独立 `qstg/` 工程与文件级改动

### 12.1 推荐目录

~~~text
qstg/
├── README.md
├── requirements.txt
├── train.py
├── utils.py
├── vocab.py
├── config/
│   ├── activitynet/
│   │   ├── baseline.json
│   │   ├── qstg_stage_a.json
│   │   ├── qstg_stage_b.json
│   │   └── qstg_full.json
│   └── charades/
│       ├── baseline.json
│       └── qstg_full.json
├── datasets/
│   ├── __init__.py
│   ├── base.py
│   ├── activitynet.py
│   └── charades_sta.py
├── models/
│   ├── __init__.py
│   ├── cpl.py
│   ├── loss.py
│   ├── transformer/
│   └── modules/
│       ├── __init__.py
│       ├── gaussian_mixture.py
│       ├── qstg.py
│       └── qstg_ops.py
├── runners/
│   ├── __init__.py
│   └── main_runner.py
├── optimizers/
├── tools/
│   ├── inspect_phrase_graph.py
│   ├── inspect_qstg_bindings.py
│   └── compare_baseline_parity.py
├── scripts/
│   ├── run_qstg_stage_a.sh
│   ├── run_qstg_stage_b.sh
│   └── run_qstg_full.sh
├── tests/
│   ├── test_phrase_graph.py
│   ├── test_temporal_graph.py
│   ├── test_qstg_binding.py
│   ├── test_qstg_quality.py
│   ├── test_qstg_loss.py
│   ├── test_qstg_connected_decode.py
│   ├── test_qstg_integration.py
│   └── test_qstg_baseline_compat.py
└── checkpoints/
    └── bootstrap/
~~~

### 12.2 `qstg/datasets/base.py`——必须修改

新增纯函数：

~~~python
def build_query_phrase_graph(
    words: Sequence[str],
    tags: Sequence[str],
    max_phrases: int,
    max_words: int,
) -> Dict[str, np.ndarray]:
    ...
~~~

职责：

- 在 OOV 过滤后调用；
- 返回第 3 节五个 phrase tensor；
- 截断到 `max_num_words` 前后必须使用同一 token 范围；
- 每个 valid phrase 至少包含一个 token；
- 每个样本至少一个 required phrase；
- edge 端点必须 valid；
- 不修改现有 masking `weights`。

`__getitem__()` 追加：

~~~text
phrase_token_mask
phrase_type
phrase_valid
phrase_required
query_edge_type
video_group_id
sample_uid
query_fallback
~~~

`build_collate_data()`：

- 将 `phrase_token_mask` pad 为 `[B,P,max(words_len)]`；
- 其他 phrase tensor 和 `query_fallback` stack；
- 全部放入 `batch['net_input']`；
- dtype 严格采用 bool/long；
- `raw` 保持原样，不把 timestamp 混入 net_input。

还应补一个现有边界保护：若过滤后没有任何词，插入一个内部 `<unk>` 占位（`words_id=0`、300 维零向量、weight=1），phrase graph 只构造 required GLOBAL，并记录 `query_fallback=true`；不能继续访问 `words[0]` 产生无上下文 `IndexError`。训练前应统计该比例，若超过 0.1% 则停止并检查词表。

### 12.3 `qstg/datasets/activitynet.py` 与 `charades_sta.py`

构造 `collate_fn` 时传入：

~~~text
max_phrases
phrase_graph.enabled
deterministic_sample_id
~~~

两者不复制 phrase 规则。所有规则只在 `base.py` 有一份实现。

### 12.4 `qstg/models/modules/qstg_ops.py`——新增

放置不持有参数的纯 tensor 函数：

~~~python
temporal_pool(...)
build_temporal_adjacency(...)
masked_soft_or(...)
probabilistic_coverage(...)
proposal_membership_to_nodes(...)
build_relation_kernel(...)
compute_relation_satisfaction(...)
compute_connectivity(...)
build_same_video_negative_mask(...)
connected_refine(...)
~~~

每个函数都应能独立 CPU 单测，不在内部调用 `.cuda()`。

### 12.5 `qstg/models/modules/qstg.py`——新增

实现第 10 节三个类和：

- `TemporalRelationGraphLayer`；
- `QSTGProposalQualityHead`；
- component residual adapter；
- global residual adapter；
- optional prior bias heads。

实现要求：

- device 来自输入；
- 不使用 NumPy；
- pairwise reduction 在 float32 执行，输出转回模型 dtype；
- 所有 masked softmax 在至少一个 valid key 的前提下执行；
- zero-init helper 单独实现并测试；
- graph disabled 时不构造 `B×B×P×M` 张量；
- `return_diagnostics=false` 时可以不长期保留 attention map，减少显存。

### 12.6 `qstg/models/modules/__init__.py`——必须修改

显式导出：

~~~python
from .qstg import (
    QueryPhraseGraphEncoder,
    QuerySubgraphTemporalGrounder,
    TemporalEvidenceGraphEncoder,
)
~~~

不使用 wildcard import。

### 12.7 `qstg/models/cpl.py`——必须修改

#### `CPL.__init__()`

读取：

~~~python
qstg_config = config.get('qstg', {})
self.use_qstg = qstg_config.get('enabled', False)
~~~

QSTG 开启时构造模块，并从 mixture generator 读取真实 `component_counts`；single 模式传 `None`。

新增：

~~~python
def set_trainable_scope(self, scope: str) -> None:
    ...
~~~

`qstg_only` 先冻结全部参数，再只解冻 `self.qstg`；`qstg_and_proposal` 额外解冻 `fc_gauss` 或 mixture generator 及 `mixture_importance_token`；`all` 解冻所有非固定 buffer 参数。runner 必须在 `_build_optimizer()` 之前调用。Stage A 还要在 forward 接口处 detach base states，避免无意义地为冻结主干保留反向图。

#### `CPL.forward()` 输入

在签名或 `kwargs` 中接受：

~~~text
phrase_token_mask
phrase_type
phrase_valid
phrase_required
query_edge_type
video_group_id
sample_uid
~~~

QSTG 开启而 phrase tensor 缺失时直接抛出清晰错误，不允许静默退化成 GLOBAL-only；只有显式 `allow_global_fallback=true` 才允许 fallback。

#### 调用位置

在 append `pred_vec` 前保存 raw frame；在 `decoding=1` 后调用 `forward_pre_proposal()`；在 component reconstruction 后调用 `enhance_components()`；在最终 `gauss_weight/center/width` 可用后调用 `score_proposals()`。

single 路径也必须调用 `score_proposals()`，但 component 输出为 `None`。

#### 新增输出

除第 10.4 节字段外，必须输出：

~~~text
proposal_reconstruction_nll [B,N]
qstg_enabled                bool
qstg_eval_center            Optional[B,N]
qstg_eval_width             Optional[B,N]
~~~

保留原有 `center/width` 的扁平 `[B*N]` 形状，避免破坏旧 loss。QSTG 的 proposal 级输出使用 `[B,N]`。

#### 不允许的改动

- 不改变 `num_props`；
- 不把 graph nodes 拼进 reconstruction video sequence；
- 不改变 vocabulary head；
- 不在 QSTG 内调用 GT；
- 不默认替换 `rec_loss` 的 hard min；
- 不改变 negative mining 的 outer context 语义。

### 12.8 `qstg/models/modules/gaussian_mixture.py`——必须修改

`predict_components()`：

~~~python
center_logits = self.center_head(global_feature)
width_logits = self.width_head(global_feature)
if center_logit_bias is not None:
    _check_shape(center_logit_bias, center_logits)
    center_logits = center_logits + center_logit_bias
if width_logit_bias is not None:
    _check_shape(width_logit_bias, width_logits)
    width_logits = width_logits + width_logit_bias
centers = torch.sigmoid(center_logits)
proposal_widths = torch.sigmoid(width_logits)
~~~

`combine()`：

~~~python
scores = self.importance_score(...).squeeze(-1)
if importance_logit_bias is not None:
    _check_shape(importance_logit_bias, scores)
    scores = scores + importance_logit_bias
scores = scores / self.importance_temperature
~~~

注意 bias 应在 temperature 前还是后必须固定。推荐如上：bias 与原 logits 同尺度，再统一除 temperature。

### 12.9 `qstg/models/loss.py`——必须修改

新增：

~~~python
def qstg_loss(
    words_logit: torch.Tensor,
    num_props: int,
    proposal_reconstruction_nll: Optional[torch.Tensor] = None,
    pair_binding_logits: Optional[torch.Tensor] = None,
    node_pair_score: Optional[torch.Tensor] = None,
    pair_quality_logits: Optional[torch.Tensor] = None,
    proposal_node_membership: Optional[torch.Tensor] = None,
    binding_logits: Optional[torch.Tensor] = None,
    binding_prob: Optional[torch.Tensor] = None,
    phrase_required: Optional[torch.Tensor] = None,
    phrase_valid: Optional[torch.Tensor] = None,
    video_group_id: Optional[torch.Tensor] = None,
    node_gate: Optional[torch.Tensor] = None,
    transition_score: Optional[torch.Tensor] = None,
    proposal_connectivity: Optional[torch.Tensor] = None,
    component_assignment: Optional[torch.Tensor] = None,
    mixture_component_valid_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ...
~~~

必须记录：

~~~text
qstg_loss
qstg_mc_loss
qstg_proposal_loss
qstg_fa_loss
qstg_explain_loss
qstg_gate_loss
qstg_connect_loss
qstg_role_loss
qstg_valid_negative_ratio
qstg_selector_entropy
qstg_selector_confident_ratio
qstg_gate_mean
qstg_gate_active_ratio
qstg_binding_mean
qstg_quality_mean
qstg_coverage_mean
qstg_exclusivity_mean
qstg_connectivity_mean
~~~

所有日志调用 `.detach().item()`；任何分母 `clamp_min(eps)`。

### 12.10 `qstg/runners/main_runner.py`——必须修改

训练 import 和调用 `qstg_loss`：

~~~python
graph_loss, graph_loss_dict = qstg_loss(
    **output,
    num_props=self.model.num_props,
    **self.args['loss'],
)
loss = loss + graph_loss
loss_dict.update(graph_loss_dict)
~~~

调用位置在旧四项 loss 之后、`backward()` 之前。

`_build_model()` 后读取 `trainable_scope` 并调用 `model.set_trainable_scope()`，然后才构造 optimizer。启动时打印每个 scope 的参数量和关键 prefix；若 `qstg_only` 下发现 `frame_fc/trans` 可训练应直接报错。

eval：

- 接受 `selection_strategy='qstg'`；
- 用第 15 节 combined score；
- 可选读取 `qstg_eval_center/width`；
- 同时计算 raw/refined；
- 汇总 QSTG diagnostics；
- validation 选择 checkpoint，最终 test 只评估选中的 checkpoint。

### 12.11 `qstg/train.py`——必须修改

新增 CLI：

~~~text
--init-from-baseline
--selection-strategy {nll,geometric_vote,semantic_vote,qstg}
--qstg-quality-weight
--qstg-analytic-weight
--qstg-connected-refine
--eval-mask-mode {legacy,deterministic}
--trainable-scope {all,qstg_only,qstg_and_proposal}
~~~

CLI override 只覆盖非 `None` 值，并保存到 checkpoint config。

### 12.12 工具脚本

`inspect_phrase_graph.py`：

- 随机打印 100 条 query；
- 显示保留 token、phrase、type、required、edge；
- 统计无 verb、无 noun、截断比例、GLOBAL-only 比例。

`inspect_qstg_bindings.py`：

- 加载 checkpoint 和固定 validation batch；
- 导出 node bounds、每个 phrase 的 top-5 node、gate、proposal C/X/R/H；
- 不依赖 GT 也可运行；
- 可选附加 GT 只用于离线诊断，输出中明确标注。

`compare_baseline_parity.py`：

- 同 checkpoint、同 batch、同随机状态；
- 比较 QSTG disabled 与零残差 enabled；
- 报告 center/width/words_logit 最大绝对误差。

## 13. 配置 specification

### 13.1 Dataset 配置

~~~json
{
  "max_phrases": 8,
  "phrase_graph": {
    "enabled": true,
    "max_actions": 2,
    "max_entities": 3,
    "max_relations": 1,
    "allow_global_fallback": true
  }
}
~~~

### 13.2 Model 配置

~~~json
{
  "qstg": {
    "enabled": true,
    "max_phrases": 8,
    "node_stride": 4,
    "num_query_graph_layers": 1,
    "num_graph_layers": 2,
    "num_graph_heads": 4,
    "max_temporal_hop": 4,
    "similar_topk": 2,
    "similarity_threshold": 0.5,
    "keep_ratio": 0.5,
    "gate_floor": 0.05,
    "gate_temperature": 0.1,
    "binding_temperature": 0.1,
    "final_binding_temperature": 0.1,
    "relation_bias_scale": 0.5,
    "relation_decay": 4.0,
    "simultaneous_decay": 1.5,
    "transition_threshold": 0.2,
    "transition_temperature": 0.05,
    "proposal_prior_scale": 0.0,
    "importance_bias_scale": 0.0,
    "component_residual_enabled": false,
    "quality_head_enabled": true,
    "membership_threshold": 0.1,
    "soft_box_temperature": 0.02,
    "connected_refine": false,
    "hard_barrier_threshold": 0.65,
    "relation_connect_threshold": 0.4,
    "coverage_keep_ratio": 0.9,
    "max_refine_shift": 0.1,
    "min_refine_width": 0.01,
    "return_diagnostics": false
  }
}
~~~

### 13.3 Loss 配置

~~~json
{
  "qstg_total_weight": 1.0,
  "qstg_mc_weight": 0.05,
  "qstg_proposal_weight": 0.05,
  "qstg_fa_weight": 0.1,
  "qstg_explain_weight": 0.03,
  "qstg_gate_weight": 0.01,
  "qstg_connect_weight": 0.05,
  "qstg_role_weight": 0.02,
  "qstg_mc_temperature": 0.07,
  "qstg_reconstruction_temperature": 0.1,
  "qstg_fa_temperature": 0.1,
  "qstg_fa_margin": 0.2,
  "qstg_selector_confidence_margin": 0.05,
  "qstg_gate_min_ratio": 0.1,
  "qstg_gate_max_ratio": 0.7,
  "qstg_detach_selector": true,
  "qstg_detach_barrier": true
}
~~~

### 13.4 Inference 配置

~~~json
{
  "selection_strategy": "qstg",
  "qstg_quality_weight": 0.2,
  "qstg_analytic_weight": 0.1,
  "selection_temperature": 0.1,
  "eval_mask_mode": "deterministic",
  "report_qstg_raw_and_refined": true
}
~~~

`qstg_quality_weight` 和 `qstg_analytic_weight` 的量纲依赖 NLL 分布，必须在 validation 上从 `{0,0.05,0.1,0.2,0.5}` 做小网格，不能直接在 test 上选择。

阶段 config 还应在顶层保存：

~~~json
{
  "qstg_stage": "A",
  "trainable_scope": "qstg_only",
  "detach_base_states": true
}
~~~

Stage B 改为 `qstg_and_proposal/true`，因为主干仍冻结；Stage C 为 `all/false`。

### 13.5 Baseline parity 配置

基线配置应：

~~~json
{
  "phrase_graph": {"enabled": false},
  "qstg": {"enabled": false},
  "qstg_total_weight": 0.0,
  "selection_strategy": "nll",
  "eval_mask_mode": "legacy"
}
~~~

不能只把 loss weight 设零却仍执行 QSTG 并修改 proposal feature；`enabled=false` 才是严格 baseline。

## 14. 训练策略

### 14.1 为什么采用分 run 阶段训练

当前 `AdamOptimizer` 可以接收 parameter groups，但 `InverseSquareRootSchedule/FairseqOptimizer.set_lr()` 会把同一 LR 写给全部 group；同时 optimizer 构建时只收集 `requires_grad=true` 的参数。若在一个 run 中途解冻参数，新增参数不在 optimizer 中。

因此推荐每个阶段是独立 run：

1. 用当前阶段 config 构造模型并设置 trainable scope；
2. 构建 optimizer；
3. 从上一阶段 checkpoint 以“权重初始化”方式加载；
4. `num_updates` 和 scheduler 从 0 开始；
5. 每个阶段有独立 validation 选模。

不要在一个 optimizer 内临时切 `requires_grad`，除非同时重建 optimizer 和 scheduler。

### 14.2 Stage 0：baseline parity

目标：证明复制后的 `qstg/` 在不开 QSTG 时与 `cpl_lrev` 一致。

要求：

- 使用同 checkpoint；
- 固定 NumPy/Torch/CUDA seed；
- 同一个保存 batch；
- `model.eval()`；
- 先用 legacy mask 比较一次，再用 deterministic mask 各自比较；
- `center/width/gauss_weight/words_logit` 最大绝对误差不超过 `1e-6`（相同 GPU/dtype）；
- 完整 validation 指标差异只允许来自浮点累计，R@1/R@5 应一致。

### 14.3 Stage A：binding/quality warm-up

建议 3–5 epoch：

~~~text
trainable_scope = qstg_only
proposal_prior_scale = 0
importance_bias_scale = 0
component_residual_enabled = false
connected_refine = false
qstg_mc_weight = 0.05
qstg_proposal_weight = 0.05
qstg_fa/explain/connect/role = 0
~~~

baseline 参数全部冻结；进入 QSTG 的 visual/grounded/query states 在接口处 detach。训练：

- phrase encoder；
- binding projections；
- temporal graph layers；
- quality head。

原四个 loss 可以计算用于日志，但其参数冻结，不加入总 loss更节省 backward。Stage A 的 checkpoint 目标按 validation 上 QSTG rerank 的 R@1 选择。

必须监控：

- correct query-video node score 高于 shuffle；
- gate 不是全 0/全 1；
- query shuffle 后 coverage/exclusivity 下降；
- quality logits 有方差；
- 同视频 false-negative mask 生效。

### 14.4 Stage B：接入 proposal/component

建议 5–10 epoch：

~~~text
trainable_scope = qstg_and_proposal
component_residual_enabled = true
proposal_prior_scale = 0 first, then 0.25
importance_bias_scale = 0 first, then 0.25
qstg_fa/explain/connect/role = ramp
~~~

可训练：

- QSTG 全部参数；
- `fc_gauss` 或 mixture `center_head/width_head/importance_*`；
- `mixture_importance_token`；
- 可选 `h[:,-1]` 上游最后一层。

其余 DualTransformer、word head、LREV 先冻结。先只打开 component residual；确认稳定后再分别打开 center/width prior 与 importance bias，不能三项同时从非零开始。

### 14.5 Stage C：联合微调

从 Stage B 最佳 checkpoint 初始化，5–10 epoch：

- 解冻 CPL 全部原参数；
- base LR 建议是 Stage B 的 0.25–0.5；
- QSTG 所有 loss 完整开启；
- 保持 residual/bias scale 固定，不把 scale 设为无约束可学习标量；
- 梯度裁剪沿用 10；
- 若 width 快速增至接近 1，先降低 `qstg_connect_weight` 以外的 proposal adapter scale，并检查原 BECL context loss 是否工作；
- 若 gate 塌缩，先检查 contrastive negatives 和 mask，不直接增大 budget loss。

### 14.6 Loss ramp

设 Stage B/C 内局部 epoch 为 `e`：

\[
\rho(e)=clip((e-warmup)/ramp,0,1).
\]

建议：

~~~text
warmup = 1 epoch
ramp = 3 epochs
effective_fa      = rho * configured_fa
effective_explain = rho * configured_explain
effective_connect = rho * configured_connect
effective_role    = rho * configured_role
~~~

`L_mc/L_pq/L_gate` 不 ramp。日志同时打印 configured 和 effective weight。

### 14.7 推荐学习率与 batch

当前 baseline LR 为 `4e-4`、batch 32。建议起点：

| 阶段 | LR | batch | 说明 |
|---|---:|---:|---|
| A | 2e-4 | 32 | 只训练 QSTG；batch 越大 contrastive negatives 越多 |
| B | 1e-4 | 32 | QSTG + proposal |
| C | 5e-5～1e-4 | 32 | 全量微调 |

若显存因 `B×B` pair quality 增加：

1. 首先只对 node-level `L_mc` 保持全 B×B；
2. proposal pair quality 对每个 query 采样最多 8 个负 video；
3. 再考虑 batch 降为 16；
4. 不要通过 detach 所有 pair binding 来“省显存”，否则 `L_mc` 无法训练表示。

### 14.8 三随机种子与选模

完整结论至少运行 3 个 seed。每个 seed：

- validation 选择 checkpoint；
- validation 选择 QSTG score weight；
- 最终 test 只评一次选中的组合；
- 报告均值和标准差；
- 不用 test R@1 选择 connected-refine threshold。

## 15. 推理与评估细则

### 15.1 Combined proposal score

当前分数越小越好。QSTG selector：

\[
score_n =
NLL_n
-\lambda_e event_n
-\lambda_q \sigma(Q_n)
-\lambda_a Q_n^{analytic}.
\]

推荐代码：

~~~python
proposal_score = nll.view(B, N)
if event_score is not None:
    proposal_score -= event_weight * event_score.view(B, N)
proposal_score -= qstg_quality_weight * torch.sigmoid(
    qstg_quality_logit)
proposal_score -= qstg_analytic_weight * qstg_analytic_score
~~~

quality head 不直接接收 NLL，避免它只复制 reconstruction score；NLL 只在 runner 组合。

### 15.2 排序与 Rank-1

`selection_strategy='qstg'`：

1. 按 combined score 排序；
2. Rank-1 默认取排序后 index 0；
3. R@5 使用排序前 5；
4. connected refinement 只改每个 proposal 的边界，不重新排序，首版也不重新做 reconstruction；
5. 若需要 QSTG weighted vote，新增独立 `qstg_vote` 策略，不在 `qstg` 中暗中使用 medoid。

### 15.3 Deterministic eval masking

`_mask_words()` 当前使用 `np.random.choice`。推荐新增：

~~~python
def _mask_words(
    self,
    words_feat,
    words_len,
    weights=None,
    sample_uid=None,
    mask_mode='legacy',
):
    ...
~~~

`deterministic` 模式对每个 sample 使用由 `sample_uid + global_eval_seed` 构造的局部 `np.random.RandomState`，仍按原 weights 无放回采样。不能重置全局 NumPy RNG，也不能让 batch 顺序改变 mask。

训练始终使用 legacy/random；baseline 历史对比可报告 legacy eval；QSTG 主表使用 deterministic eval，并对 baseline 与 QSTG 一视同仁。

### 15.4 必须报告的指标

主指标：

- R@1 mIoU；
- R@1 IoU@0.3/0.5/0.7；
- R@5 mIoU；
- R@5 IoU@0.3/0.5/0.7。

候选诊断：

- oracle best-of-N mIoU；
- NLL top-1 与 oracle candidate 是否一致；
- QSTG top-1 与 oracle candidate 是否一致；
- raw/refined 指标差；
- mean width；
- pairwise proposal IoU；
- mixture outer span 与 mask support span 差。

QSTG 诊断：

- phrase coverage；
- exclusivity；
- relation satisfaction（仅 valid relation）；
- connectivity；
- gate mean/active ratio；
- correct-vs-shuffle node score margin；
- correct-vs-shuffle proposal score margin；
- component assignment entropy；
- component assignment off-diagonal cosine；
- refinement 接受率与平均移动。

### 15.5 Query 分组评估

使用 phrase graph 规则自动分组：

- action-only；
- entity-heavy；
- action+entity；
- relation-heavy；
- attribute-heavy；
- short event / long event（长度分组仅在评估使用 GT）；
- single-component-like / multi-stage query。

预期 QSTG 应首先在 relation-heavy、action+entity 和背景混淆样本上提升。如果只在 entity-heavy 提升、action/relation 无提升，应先改善 phrase/action binding，不继续增加 graph layer。

## 16. Checkpoint 与 backward compatibility

### 16.1 三种加载语义

必须区分：

1. **strict resume**：相同结构和同阶段；恢复 model、optimizer、scheduler、`num_updates`、completed epoch；
2. **baseline warm-start**：加载 shape-compatible baseline 参数，QSTG 新参数按规定初始化，优化状态从 0 开始；
3. **weights-only eval**：从选中 checkpoint 加载模型参数，不恢复训练状态。

新增 `--init-from-baseline`，不要复用语义为 V3→V4 的 `--init-from-v3`。

### 16.2 Warm-start 检查

baseline warm-start 时：

- 允许缺少 `qstg.*`；
- 不允许意外缺少 `frame_fc/trans/fc_comp`；
- mixture 模式必须成功加载 `mixture_generator` 原参数；
- single 模式必须加载 `fc_gauss`；
- 打印 loaded、expected missing、unexpected、shape mismatch 四类 key；
- 若 missing key 不在 QSTG allowlist 中则报错。

### 16.3 Identity initialization

必须零初始化：

- global residual adapter 输出层；
- node fusion residual 输出层；
- component residual 输出层；
- center/width prior bias head 输出；
- importance bias head输出；
- quality head最后层可零初始化，但解析分数仍可用于最小实验。

不需要把 phrase/binding projection 全部置零，否则 `L_mc` 初始无区分且各节点梯度对称。

### 16.4 保存内容

推荐 checkpoint：

~~~python
{
    'num_updates': self.num_updates,
    'completed_epoch': epoch,
    'config': self.args,
    'model_parameters': self.model.state_dict(),
    'optimizer_state': self.optimizer.state_dict(),
    'scheduler_state': {
        'num_updates': self.num_updates,
    },
    'format_version': 2,
    'qstg_stage': self.qstg_stage,
}
~~~

旧 checkpoint 没有 optimizer/epoch 时仍允许 weights-only 或 legacy resume，但必须打印提示。

### 16.5 QSTG disabled 行为

`qstg.enabled=false` 时：

- dataset 可以不构造 phrase graph；
- model 不创建 QSTG module；
- Gaussian generator 不接 bias；
- output 中 QSTG key 可缺失或为 `None`；
- `qstg_loss` 返回零；
- selector 不能设置为 qstg；
-旧 checkpoint strict load 不受新增 optional function argument 影响；
- baseline logits 和指标保持一致。

## 17. 测试与验证矩阵

### 17.1 Phrase graph 单测

至少覆盖：

- “person opens a door”；
- “man quickly walks into the room”；
- “woman sits before standing up”；
- 无 verb；
- 无 noun；
- 全部 OOV；
- 超过 20 token；
- 超过 8 phrase；
- before/after/while edge direction；
- attribute 与 entity/action 连接；
- GLOBAL-only fallback。

断言：

- shape/dtype 正确；
- valid phrase 非空；
- required 至少一个；
- edge 不连接 padding；
- truncation 后 mask 不越界；
- 同输入重复构造完全一致。

### 17.2 Temporal node/edge 单测

- T=200,stride=4 得 M=50；
- T 不能被 stride 整除；
- frames_len 小于 T；
- 最后 node 只有一个 frame；
- padding 不参与均值；
- 每个 valid node 有 self edge；
- 相邻双向边始终存在；
- similar edge 不超过 top-k/hop；
- 全相同 feature 时 transition finite；
- AMP 输入时无 NaN。

### 17.3 Binding/gate 单测

- 正确 phrase/node 人工同向时 binding 高；
- shuffle phrase 后低；
- required phrase mask 生效；
- padding logit 不参与；
- gate ratio 在预期范围；
- 只有一个 node；
- 只有 GLOBAL；
- 同视频 off-diagonal 被 ignore；
- batch=1 时 contrastive 安全返回零。

### 17.4 Coverage/exclusivity/relation sanity

构造人工 `P×M`：

1. 候选覆盖所有 phrase 证据：C 高；
2. 候选遗漏 action：C 下降；
3. 候选加入无解释背景：X 下降；
4. before 证据顺序正确：R 高；
5. before 证据反向：R 低；
6. 无 relation：R=1 且 valid=0；
7. 候选跨 barrier：H 下降；
8. 不跨 barrier：H 高。

### 17.5 Component 单测

- `component_counts=[1,2,3,4,5]` flatten 顺序正确；
- assignment 只在 valid phrase 上 softmax；
- required phrase 少于 2 时 role loss 为零；
- importance bias `None` 与旧 combine 一致；
- 全零 bias 与旧 combine 一致；
- summary residual 零初始化时输出完全等于输入；
- outer/weighted 两种 boundary mode 均通过。

### 17.6 Loss 单测

- 所有 QSTG tensor 为 `None` 返回 connected zero；
- correct diagonal score 提高时 `L_mc/L_pq` 下降；
- same-video mask 不产生 NaN；
- `selector.detach` 后 NLL 不从 QSTG loss 收 gradient；
- barrier detach 后 transition branch无梯度；
- binding 和 quality head 有非零梯度；
- 全 padding phrase 不允许进入模型；
- selector confidence 降低时 `L_fa` 样本权重下降；
- FP16/bfloat16 reduction finite。

### 17.7 Connected decode 单测

- 单连通段保持不变；
- 两段中间有 hard barrier 时选高分段；
- relation 支持连接时不错误切断；
- coverage 下降超过阈值时回退；
- shift 超限时回退；
- refined width 太小时回退；
- start/end 始终在 `[0,1]` 且 `start<end`；
- proposal 数量不变。

### 17.8 Forward/shape/backward 集成

ActivityNet mixture：

~~~text
B=2,T=200,Dv=500,W<=20,N=5,Ktotal=15,M=50,P=8
~~~

Charades single：

~~~text
B=2,T=200,Dv=1024,W<=20,N=8,M=50,P=8
~~~

断言：

- 所有第 10.4 节输出 shape；
- loss finite；
- backward 后 QSTG projection、graph、quality 有梯度；
- Stage B component adapter 有梯度；
- center/width 合法；
- old loss 仍可调用；
- `enabled=false` 不要求 phrase args。

### 17.9 Baseline compatibility

至少保留：

- QSTG disabled vs copied baseline；
- QSTG enabled、所有 residual/bias scale 0 时 Gaussian 参数初始一致；
- optional generator args 为 `None` 时严格一致；
- old ActivityNet checkpoint warm-start；
- old Charades checkpoint warm-start；
- strict QSTG resume；
- weights-only final eval。

### 17.10 Smoke 与 overfit

Smoke：

- 2 batch train + 1 batch eval；
- 单/mixture 各一次；
- peak GPU memory 记录；
- 无 NaN/OOM/key error。

Overfit：

- 固定 8–16 个样本；
- QSTG-only 训练 100–300 step；
- node/proposal contrastive loss 应下降；
- correct-vs-shuffle margin 上升；
- gate 保持非退化；
- quality logits 不全相同。

## 18. Implementation Order

### Step 0：复制并锁定 baseline

操作：

1. 创建独立 `qstg/`；
2. 复制 `cpl_lrev` 运行所需源码和 config；
3. 修正内部默认路径为 `qstg` 相对路径；
4. 运行原 tests；
5. 执行 Stage 0 parity；
6. 保存 `qstg/config/*/baseline.json` 和 parity 报告。

完成标准：QSTG 未引入前，训练 smoke 和 evaluation 均可独立运行，不 import `cpl_lrev`。

### Step 1：实现 phrase graph 数据链

只修改 dataset/collate，不改 model 行为。

完成标准：

- phrase graph 单测全部通过；
- 随机抽样人工检查至少 100 条；
- GLOBAL-only、截断、无 action/noun 统计已记录；
- 原 net_input tensor 数值不变。

### Step 2：实现纯 `qstg_ops`

实现 temporal pooling、adjacency、coverage、exclusivity、relation、connectivity。

完成标准：全部人工 tensor sanity 通过，CPU float32 与 CUDA float32 结果一致到合理误差。

### Step 3：实现 node binding，不接候选

接 `QueryPhraseGraphEncoder`、clean visual projection、temporal node 和 `pair_binding_logits`；`proposal_prior_scale=0`，不改 center/width。

完成标准：

- correct-vs-shuffle score 有可学习趋势；
- batch=1/duplicate-video 情形 finite；
- peak memory 可接受；
- baseline proposal 与零残差 enabled 仍一致。

### Step 4：实现 graph propagation 与 gate

接相邻/相似 edge、relation bias、两层传播和 gate regularization。

完成标准：

- 每个 valid node attention 行有合法 key；
- padding 不影响输出；
- gate 不塌缩；
- graph layer 0 与 2 可配置消融。

### Step 5：实现 proposal C/X/R/H 与解析 rerank

在最终 proposal 后计算 quality feature；先只用 `proposal_analytic_score`，不训练 head。

完成标准：

- query shuffle 后 C/X/analytic score 显著下降；
- validation 上 analytic rerank 与 NLL baseline 有可重复对比；
- 不改变 raw proposal。

这是原 proposal “最小验证实验”的正式落地点。若此步无收益，先停止 component 和 connected decoder 的实现，分析 action binding、候选 oracle 和 C/X 分布。

### Step 6：训练 node/proposal contrastive 与 quality head

实现 `L_mc/L_pq/L_gate` 和 Stage A。

完成标准：

- 三项 loss finite 且下降；
- quality logit 方差非零；
- learned rerank 不弱于只用 analytic score；
- same-video false negative ignore 有测试。

### Step 7：接 global residual 和 proposal prior

先只打开 global residual，再单独打开 center/width bias。

完成标准：

- scale 0 identity；
- 非零 scale 时 Gaussian head 有梯度；
- mean width、center histogram 无立即塌缩；
- single/mixture 都能训练。

### Step 8：接 component semantic fusion

实现 component membership、assignment、graph summary、summary residual；最后才打开 importance bias 和 `L_role`。

完成标准：

- component flatten/slice 顺序与 generator 一致；
- importance entropy 未塌为 0；
- 多 phrase query 中 assignment 有差异；
- ActivityNet mixture 集成测试通过。

### Step 9：实现 `L_fa/L_ex/L_conn`

接 NLL soft selector、confidence downweight 和 detach 规则，按 Stage B/C ramp。

完成标准：

- selector entropy 合理；
- 低置信样本权重下降；
- detach 单测通过；
- 宽度、gate、binding 均无退化。

### Step 10：实现 QSTG selector 和 deterministic eval

接 combined score、diagnostics 和 validation grid。

完成标准：

- 同一 checkpoint 重复评估结果一致；
- baseline 与 QSTG 使用相同 mask；
- test 不参与权重选择；
- raw proposal 的 R@1 提升可独立报告。

### Step 11：实现 connected refinement

最后实现候选内切分/搜索。

完成标准：

- 回退规则全部测试；
- raw/refined 同时报告；
- oracle R@5 不显著下降；
- refinement 收益在至少两个 seed 同方向。

### Step 12：完整实验与归档

- ActivityNet 三 seed；
- Charades 三 seed；
- 所有主 ablation；
- config、log、checkpoint、commit hash/源码快照和指标表对应；
- README 写出一键训练与评估命令。

## 19. 实验与 ablation 设计

### 19.1 主对比

| 编号 | 方案 |
|---|---|
| B0 | 原 `cpl_lrev`，NLL selector |
| B1 | B0 + phrase pooling，但只用整句相似 |
| B2 | B0 + phrase-node coverage |
| B3 | B2 + reverse exclusivity |
| B4 | B3 + relation satisfaction |
| B5 | B4 + connectivity |
| B6 | B5 + learned quality head |
| B7 | B6 + graph propagation |
| B8 | B7 + global adapter |
| B9 | B8 + component semantic fusion |
| B10 | B9 + connected refinement |

关键判据：B2→B3 检验双向 binding；B6→B7 检验图传播是否提供额外价值；B8→B9 检验 component 语义分工；B9→B10 检验非 outer-hull 解码。

### 19.2 Query graph ablation

- GLOBAL only；
- ACTION + ENTITY；
- ACTION + ENTITY + ATTRIBUTE；
- 全 phrase type 无 edge；
- 全 phrase + 无向 edge；
- 全 phrase + typed/directional edge；
- POS 规则 vs 可选离线依存 parser。

### 19.3 Temporal graph ablation

- stride 2/4/8；
- graph layer 0/1/2/4；
- only adjacent；
- adjacent + similar edge；
- max hop 2/4/8；
- clean visual node vs dropout projected node；
- clean visual node vs grounded node；
- hard top-ratio vs soft gate；
- keep ratio 0.25/0.5/0.75/1.0。

### 19.4 Binding/loss ablation

- node-level `L_mc` only；
- proposal-level `L_pq` only；
- 两者联合；
- 无 `L_fa`；
- 无 reverse `L_ex`；
- barrier detach on/off；
- selector detach on/off（仅诊断，off 有自确认风险）；
- role diversity 0/0.01/0.02/0.05。

### 19.5 Proposal 接入 ablation

- rerank only；
- global residual only；
- proposal center/width prior only；
- component summary residual only；
- importance bias only；
- residual + importance bias；
- raw outer boundary vs weighted boundary；
- connected refinement off/on。

### 19.6 支持研究假设的最低结果

至少满足：

1. phrase coverage 比整句 similarity 的 validation R@1 更高；
2. coverage + exclusivity 比单向 coverage 更高；
3. shuffle query 后 binding/quality 显著下降；
4. QSTG 主要改善 top-1 selection，而不是只靠扩大所有 proposal；
5. mean width 不出现系统性大幅上升；
6. relation-heavy 子集在加入 relation edge 后有相对收益；
7. component semantic fusion 后 assignment cosine 下降，且 R@1 不下降；
8. 至少两个 seed 的主指标提升方向一致。

若 R@1 提升但 oracle best-of-N 不变，这是合理的 rerank 收益；若 oracle 明显下降，说明 adapter/prior 破坏 candidate generation，应退回 rerank-only。

## 20. 主要风险与处理方法

### 20.1 C3D/I3D 不足以形成可靠 entity node

表现：noun/entity binding 无区分，action 反而更稳定。

处理：

- 先报告 action/entity 分类型准确性；
- entity feature 可拼接视频全局残差；
- 不立即接 detector；
- 若 entity 仍无效，首版聚焦 action-relation temporal graph，并在结论中明确。

### 20.2 POS phrase graph 解析错误

表现：短语截断、动词粒子分开、介词被误作时序关系。

处理：

- 使用 `inspect_phrase_graph.py`；
- 对高频错误写规则和回归测试；
- relation_valid 只在明确词面触发时为真；
- 不让低置信 relation 成为 hard barrier。

### 20.3 Query 与 clean visual node 尚未对齐

`enc_out` 与 `visual_nodes` 都是 256 维，但同维不等于同空间。

处理：

- 使用独立 `W_q/W_v`；
- 两侧 L2 normalize；
- Stage A 对比学习；
- 不直接用 raw cosine 当最终 score；
- 检查 correct-vs-shuffle margin。

### 20.4 Gate 全开

原因可能是 reverse explainability 或 coverage 促使所有 binding 变高。

处理：

- 确认跨 batch negatives 有效；
- 检查 same-video mask 后仍有足够 negatives；
- 使用 budget 上界；
- 降低 `L_ex`；
- 不先提高 hard threshold。

### 20.5 Gate 全关或单峰塌缩

处理：

- gate floor 保持 0.05；
- budget 下界；
- 每个 required phrase 独立 coverage；
- 降低 binding temperature 前先检查 logits 范围；
- 无合法 phrase 时 GLOBAL fallback。

### 20.6 Weak selector 自我确认

`L_fa` 使用模型 NLL 选 proposal，早期可能把错误区间当正样本。

处理：

- Stage A 不开 `L_fa`；
- selector detach；
- confidence downweight；
- temperature 不低于 0.1；
- 对所有 proposal 做 soft mixture，不用 hard argmin；
- 比较只用跨 batch contrastive 的模型。

### 20.7 Same-video false negative

处理：

- `video_group_id` mask；
- 记录每 batch 有效 negative ratio；
- batch 全为同视频时该 loss 返回零；
- 后续分布式 gather 时同步 gather group id。

### 20.8 Relation kernel 过强

口语中的词序不总是事件发生顺序。

处理：

- 只有 `before/after/then` 使用方向 kernel；
- 普通 action-entity 用无向局部 kernel；
- relation bias scale 从 0.1–0.5 搜索；
- relation edge 不单独创建远距离 video edge；
- typed relation 先作为 quality feature，再用于 propagation。

### 20.9 Graph 模块与 DualTransformer 重叠

若 graph layer 0 与 2 性能相同，说明额外传播没有贡献。

处理：

- 保留 binding/rerank，删除传播而不是强行加深；
- 检查 adjacency 与 attention entropy；
- relation-heavy 子集单独分析；
- 层数上限 2，4 层只用于消融。

### 20.10 Component role diversity 误伤单一事件

处理：

- required phrase 少于 2 不计算；
- proposal selector 加权；
- 权重不超过 0.05；
- 监控 assignment entropy 和 component spread；
- 不强制 one-hot。

### 20.11 Mixture mask 与 outer span 不一致

一个 proposal 的 Gaussian mass 可呈多峰，但输出区间是外包络。

处理：

- coverage/X 使用实际 mixture mask；
- connectivity 使用解析 span；
- 记录二者 disagreement；
- connected refinement 单独消融；
- negative mining 仍用 outer context，避免正 component 被当负样本。

### 20.12 Barrier loss 通过表示逃避

处理：

- transition、hard adjacency、barrier 默认 detach；
- connect loss主要对 proposal 几何求导；
- 分别监控 transition 分布和 width；
- 做 detach on/off 单测与消融。

### 20.13 数值下溢或 NaN

风险来自 probability product、masked softmax、FP16 log 和空 mask。

处理：

- product 转 log1p-sum-exp；
- reductions 用 float32；
- 每行至少 self edge；
- 分母 clamp；
- 无 relation/negative 时显式返回 connected zero；
- 单测覆盖全 padding、batch=1、单 node。

### 20.14 显存增加

最大新增量来自 pair binding 和 cross-query proposal quality，而非 M=50 的图本身。

处理优先级：

1. 不保存无用 attention；
2. proposal pair negatives 子采样；
3. gradient checkpoint graph layer；
4. batch 32→16；
5. 分步计算 Bq block；
6. 最后才减少 P 或 M。

### 20.15 Eval 随机遮词

处理：sample-local deterministic RNG；baseline 与 QSTG 同策略；重复 eval 一致性测试。

### 20.16 分阶段 optimizer 不完整

处理：每阶段独立 run 和 optimizer；checkpoint 保存 stage；禁止中途仅切 `requires_grad`。

### 20.17 Checkpoint key 不兼容

处理：strict resume 与 warm-start 分开；QSTG missing allowlist；shape mismatch 立即报错；保存 config 和 format version。

### 20.18 Charades validation 泄漏

当前 Charades config 的 `val_data` 与 `test_data` 都指向 `test.json`。这不能用于正式超参选择。

处理：

- 从训练集固定切分 validation，保存 split 文件；
- 所有 QSTG 权重和 threshold 在该 validation 选择；
- test 只做最终报告；
- 若无法重建 split，只将 Charades 结果标为 diagnostic，不声称严格 test。

### 20.19 双路径 `frame_fc` 梯度

clean 和 dropout 路径共享 `frame_fc`，联合训练时梯度来自两条路径。

处理：

- Stage A baseline 冻结且输入 state detach；
- Stage C 监控 `frame_fc.weight` 梯度范数；
- 可选 clean branch 使用 `F.linear(...,weight.detach())` 做消融；
- 不在未验证前增加第二套大 projection。

## 21. 当前不能静默假设的事项

下列选择已经给出推荐默认值，但实现者若改变必须在 config 和实验名中显式记录。

### 21.1 Temporal node 是固定 chunk 还是聚类

默认固定 stride=4。原 proposal 的 latent/cluster node 作为后续 ablation，不在首版在线聚类。

### 21.2 Binding 使用 clean 还是 grounded state

跨 batch negatives 使用 clean visual node；正确 pair 的 graph propagation 融合 grounded node。不能用正确 query-conditioned state 构造所有负 pair。

### 21.3 GLOBAL 是否 required

有 content phrase 时否；只有 fallback 时是。否则 GLOBAL 与细粒度 phrase 重复计权。

### 21.4 Relation 是否都是时序方向

否。仅明确 before/after/then 使用方向；普通 action-entity 是无向关联。

### 21.5 Prompt filtering 是 hard 还是 soft

默认 soft gate + floor。hard top-k 只用于推理/消融，不用 straight-through。

### 21.6 QSTG 是否立即改 proposal

否。第一完整里程碑是 rerank-only；之后才依次接 global residual、prior、component fusion。

### 21.7 Quality 用解析分数还是 learned head

两者都输出。最小实验先解析，Stage A 后比较 learned；runner 可分别设置权重。

### 21.8 ActivityNet boundary mode

以运行 config/checkpoint 为准，当前主 config 是 `outer`。不能根据旧 README 静默改成 weighted。

### 21.9 Connected refinement 是否默认开启

否。必须在 raw rerank 有效后单独验证。

### 21.10 是否使用 GT 做 alignment target

否。`L_fa` 使用弱监督 bag ranking，所有 GT timestamp 只在 metric 路径。

### 21.11 是否复制 SG-FSCFormer diagonal target

否。源码在 alignment target 缺失时的 identity 是工程 fallback，不是 QSTG 的监督依据。

### 21.12 是否做 distributed all-gather negatives

首版否。先用单进程/local batch；多 GPU 时若每卡 batch 太小，再实现带 gradient 的 gather，并同步 group id。

## 22. 验收标准

实现只有同时满足以下条件才算完成：

1. `qstg/` 可独立运行，不 import 或软链接 `cpl_lrev`；
2. QSTG disabled 与 baseline parity 通过；
3. phrase graph 对 OOV 截断后 token 严格对齐；
4. ActivityNet mixture 和 Charades single 两条路径均可 forward/backward；
5. optional Gaussian bias 为 `None` 时旧行为严格一致；
6. 所有 QSTG loss 有 finite 值，期望参数有非零梯度；
7. batch=1、同视频重复、无 relation、GLOBAL-only 不产生 NaN；
8. 训练 forward 不接收 GT timestamp；
9. query shuffle 后 node/proposal match score 显著下降；
10. gate 不全开、不全关，active ratio 有诊断；
11. coverage/exclusivity 人工 sanity 符合预期；
12. QSTG rerank 指标可与 raw NLL 独立比较；
13. connected refinement 有完整回退并保持 N 不变；
14. deterministic eval 重复运行结果一致；
15. validation 选模/选权重与 test 报告严格分离；
16. checkpoint 支持 strict resume、baseline warm-start 和 weights-only eval；
17. 三 seed 主实验与关键 ablation 已归档；
18. README 给出从数据路径、warm-start、训练到评估的完整命令。

研究层面的最低成功标准：

- R@1@0.5 或 R@1 mIoU 稳定高于对应 baseline；
- oracle best-of-N 不因 QSTG 接入明显下降；
- 双向 coverage+exclusivity 优于单向 coverage；
- relation-heavy 子集对 typed relation 有响应；
- 收益不是由 proposal 全部变宽造成。

## 23. 给后续 Codex 的执行说明

后续若根据本文实施代码，按以下约束执行：

1. 先创建独立 `qstg/`，把 `cpl_lrev` 当只读事实基线；
2. 先做 parity，再做 phrase graph，不能一开始同时改 dataset、Gaussian、loss 和 selector；
3. 每完成一个 Implementation Step 就运行对应 tests；
4. 首个可训练里程碑是 rerank-only，不默认打开 center/width prior；
5. 所有残差和 bias 必须 identity/zero 初始化；
6. 保持 single 与 mixture 兼容；
7. 保持旧 output key 的形状；
8. 不把 GT timestamp 加入 `net_input`；
9. 不从 SG-FSCFormer 源码复制 box、SAM2 或 diagonal alignment fallback；
10. 遇到本文“不能静默假设”的事项时，以推荐默认值实施；要改变则先更新 config、tests 和实验标签；
11. 不用 test split 调参，特别是先修正 Charades 的 validation；
12. 每次训练保存实际 config、seed、stage、selector、eval mask mode 和源码版本；
13. 若 Step 5 的双向 rerank 不成立，先停止更复杂模块，输出 failure analysis，不通过堆叠 graph layer 掩盖；
14. 若实现结果与本文 tensor shape 不一致，先修文档或说明原因，不能让接口事实与 specification 分叉。

建议最终 README 命令顺序：

~~~bash
python tools/compare_baseline_parity.py --config config/activitynet/baseline.json ...
python tools/inspect_phrase_graph.py --config config/activitynet/qstg_stage_a.json
python train.py --config-path config/activitynet/qstg_stage_a.json --init-from-baseline ...
python train.py --config-path config/activitynet/qstg_stage_b.json --init-from-baseline ...
python train.py --config-path config/activitynet/qstg_full.json --init-from-baseline ...
python train.py --config-path config/activitynet/qstg_full.json --resume ... --eval
~~~

其中 Stage B/C 从上一阶段初始化时，CLI 最终应使用单独的 `--init-from-qstg` 或通用 `--init-weights`；上面的 `--init-from-baseline` 仅表示 weights-only 初始化语义，实际实现时应避免让名称误导来源类型。
