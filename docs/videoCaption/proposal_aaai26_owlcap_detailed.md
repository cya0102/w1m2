# BPSE / OwlCap 详细工程实施方案

> 对应原方案：[`proposal_aaai26_owlcap.md`](proposal_aaai26_owlcap.md)  
> 对应论文：*OwlCap: Harmonizing Motion-Detail for Video Captioning via HMD-270K and Caption Set Equivalence Reward*（AAAI 2026）  
> 事实参考基线：`/data/chenyuan/videogrounding/w1m2/cpl_lrev`  
> 推荐实施目录：`/data/chenyuan/videogrounding/w1m2/owlcap`  
> 文档性质：面向代码实现的 specification；本文不实施模型代码  
> 最后核对日期：2026-09-10

## 1. 目标、实施边界与结论

本方案保留原 proposal 的核心研究意图：把 OwlCap 的 Caption Set Equivalence Reward（CSER）从“预测 caption 与 GT caption 的双向事实集合匹配”，改造成“候选时间区间与 query 的双向语义集合匹配”。

在 grounding 中，两个方向分别是：

- **完整性（completeness）**：query 中每个必要语义单元，是否都能在候选区间内部找到视觉证据；
- **纯净性（purity，亦对应 correctness/exclusivity）**：候选区间内部每个显著事件，是否都能被 query 中至少一个语义单元解释。

最终得到 BPSE（Bidirectional Proposal-Set Equivalence）质量分数。它主要解决当前 `cpl_lrev` 的四个真实问题：

1. NLL 更偏好容易重构、包含上下文的宽候选，而非高 IoU 候选；
2. `min NLL` 形成硬赢家自我确认；
3. R@5 明显高于 R@1，说明好候选存在但排序失败；
4. 最终没有独立估计“是否漏语义”和“是否包含额外事件”的质量头。

推荐的首个完整版本采用以下路径：

1. dataset 用现有 NLTK token/POS 结果构造 query unit mask，不引入在线大语言模型；
2. `CPL.forward` 保留下采样前的视觉状态和完整 query 状态；
3. mixture 合并出最终 `center/width` 后，为每个原始候选及其局部扰动候选计算完整性、纯净性和 inside-shell contrast；
4. 两层质量头输出候选质量 logit；训练标签来自停止梯度的双向等价分数；
5. 用组内高置信 pairwise ranking、绝对质量回归和可选的边界等价损失训练；
6. BPSE 开启后，以停止梯度的质量权重替代 `min NLL` 的硬赢家，用于重构和负样本排序；
7. 推理不生成 caption、不运行 RL，也不生成扰动候选，只给原有候选打分并按 BPSE 排序；
8. `bpse.enabled=false` 时必须保持当前 baseline 数值路径和 checkpoint 行为。

### 1.1 推荐使用独立 `owlcap/` 工程

当前仓库中已经存在空目录：

```text
/data/chenyuan/videogrounding/w1m2/owlcap
```

本次用户要求没有明确规定后续代码必须写入哪个目录。为避免污染可复现的 `cpl_lrev/`，本文推荐后续 Codex 将运行基线所需代码复制到 `owlcap/`，并只修改 `owlcap/` 内的副本。本文第 2 节使用 `cpl_lrev/...` 描述当前事实，第 6 节以后使用 `owlcap/...` 描述推荐的实际改动位置。

推荐的独立性规则为：

- `cpl_lrev/` 只读；
- `owlcap/` 拥有自己的 `train.py`、`models/`、`datasets/`、`runners/`、`optimizers/`、`config/`、`tests/`、`tools/`、`scripts/`、`utils.py`、`vocab.py` 和依赖文件；
- 不在 `owlcap/` 中 `import cpl_lrev`，不把 `../cpl_lrev` 加入 `sys.path`，不使用源码软链接；
- HDF5 特征可以继续通过 config 指向共享数据路径；
- baseline checkpoint 可复制到 `owlcap/checkpoints/bootstrap/` 作为一次性初始化来源。

这属于推荐的工程隔离方式，不是 OwlCap 原论文或原 proposal 的算法组成。如果后续明确要求直接修改 `cpl_lrev/`，本文中 `owlcap/...` 与 `cpl_lrev/...` 的文件结构是一一对应的，但不建议同时维护两处实现。

### 1.2 原论文、原 proposal 与推荐工程实现的边界

| 层级 | 内容 |
|---|---|
| OwlCap 原论文 | HMD-270K 数据构造；caption 原子事实分解；预测到 GT 的 correctness；GT 到预测的 completeness；CSER + GRPO |
| 原 BPSE proposal | query 原子化；帧—单元证据；完整性；显著事件纯净性；质量头；扰动候选组内偏好；BPSE 排序 |
| 推荐首版工程实现 | NLTK 规则 unit mask；可微 soft-box；分箱局部峰值 event token；高 margin pairwise + BCE；BPSE soft routing；推理按质量头排序 |
| 可选增强 | 冻结 LLM 离线原子化；MIL 跨视频校准；listwise loss；BPSE-NLL 混合；候选边界微调；分布式 in-batch negatives |

首轮实现不要加入外部 Qwen judge、HMD-270K、GRPO、caption 生成器或在线 LLM 调用。它们不是检验“候选双向集合等价能否修复 CPL 排序”的必要条件。

## 2. OwlCap 论文核心机制与迁移依据

### 2.1 论文解决的问题

OwlCap 发现详细视频描述容易在 motion 与 detail 之间失衡：动作丰富的 caption 可能遗漏人物、物体和场景细节，细节丰富的 caption 又可能遗漏事件变化。论文从数据和优化两个方面处理：

1. **HMD-270K**：先用运动导向模型产生 motion caption，再由通用 MLLM 补充静态 detail；随后把 caption 拆成最小语义单元，用 judge model 逐项核验，保留 unit accuracy 不低于 90% 的样本；
2. **CSER**：把预测 caption 拆成单位集合 `U={U_i}`，把 GT caption 拆成事实集合 `F={F_j}`，做双向 unit-to-set 验证；
3. **两阶段训练**：先在 HMD-270K 上 SFT，再在高质量子集上用 CSER 做 GRPO。

论文的两个核心分数是：

\[
S_{\text{correct}}=
\frac{1}{n}\sum_{i=1}^{n}\mathbb I(U_i\in C_{gt}),
\qquad
S_{\text{complete}}=
\frac{1}{m}\sum_{j=1}^{m}\mathbb I(F_j\in C_{pred}).
\]

correctness 检查预测有没有增加参考中无法解释的事实；completeness 检查参考事实有没有被遗漏。论文消融显示，只使用任一方向都弱于两者联合。

### 2.2 为什么机制有效，而不是简单增加一个 reward

整句相似度容易被语言流畅度、长度和高频表达主导。CSER 的有效点是改变了评价单位和方向：

- 先拆成不可再分的语义单位，避免一个正确长句掩盖局部幻觉；
- `prediction -> reference` 专门惩罚多说和错误事实；
- `reference -> prediction` 专门惩罚漏说；
- 两向联合后，模型不能只通过生成更长或更短的 caption 获益。

对应到时间区间，宽候选类似 caption 中的“多说”：它覆盖目标证据，但还包含额外事件；过窄候选类似“漏说”：它只覆盖 query 的一部分。

### 2.3 caption 与 grounding 的关键差异

不能直接复现 CSER，原因如下：

| Caption 场景 | 当前 grounding 场景 | 必须做的改造 |
|---|---|---|
| 输出是文本集合 | 输出是连续时间区间 | 用区间内的帧/事件 token 作为预测集合 |
| 有 GT caption 事实集合 | 弱监督训练不能使用 GT timestamp | query 本身作为目标语义集合，不把 timestamp 放入训练输入 |
| judge LLM 判断文本蕴含 | 当前模型只有 C3D/I3D + GloVe | 用现有跨模态状态计算软匹配，不在线调用 LLM |
| GRPO 对生成序列做策略优化 | CPL 输出连续 center/width | 用可微排序、质量回归和 soft routing，不做 RL sampling |
| correctness 约束文本幻觉 | 时间区间不会“说一句假话” | 改为候选内部额外显著事件是否可由 query 解释，即纯净性 |

因此，本方案借鉴的是双向集合等价机制，而不是 OwlCap 的模型规模、数据集或训练基础设施。

## 3. 当前 `cpl_lrev` 仓库的真实执行链

### 3.1 入口、配置和模型构建

训练与评估入口是 `cpl_lrev/train.py`：

1. `parse_args()` 读取 `--config-path`、`--resume`、`--init-from-v3`、`--eval`、选择策略和 loss override；
2. `main()` 通过 `utils.load_json()` 读取 JSON，再把 CLI override 写入 `args['loss']`；
3. `MainRunner(args)` 构建 dataset、model、optimizer 和 scheduler；
4. `MainRunner.__init__()` 将真实词表大小写入 `args['model']['config']['vocab_size']`，并设置 `max_epoch`；
5. `MainRunner._build_model()` 用 `getattr(models, model_config['name'])` 构造 `models.CPL`，随后直接 `.cuda()`。

仓库没有 dataclass 配置 schema、模块注册器、loss registry 或通用 device abstraction。BPSE 配置应继续使用嵌套字典，并用 `.get(default)` 保证旧 config 可运行。

### 3.2 Dataset、采样和 batch

真实相关代码：

- `cpl_lrev/datasets/base.py::BaseDataset.__getitem__`
- `cpl_lrev/datasets/base.py::BaseDataset._sample_frame_features`
- `cpl_lrev/datasets/base.py::build_collate_data`
- `cpl_lrev/datasets/activitynet.py::ActivityNet`
- `cpl_lrev/datasets/charades_sta.py::CharadesSTA`

当前数据流为：

```text
HDF5 可变长 C3D/I3D feature
    -> 均匀分成 200 段并逐段平均
    -> frames_feat [B, 200, Dv], float32

sentence
    -> nltk.word_tokenize + nltk.pos_tag
    -> 丢弃 vocab 外词
    -> words_feat [B, W+1, 300], float32
    -> words_id [B, W], int64
    -> words_len [B]
    -> POS 加权 masking probability [B, W]
```

ActivityNet 使用 `Dv=500` 的 C3D 特征、`num_props=5`；Charades-STA 使用 `Dv=1024` 的 I3D 特征，当前 config 中 `num_props=8`。两者 `max_num_frames=200`、`max_num_words=20`。

`raw=[vid,duration,timestamps,sentence]` 只由 runner 的 eval 使用。训练时 `net_input` 没有 timestamp，因此当前基线是弱监督定位。BPSE 必须保持这一点。

### 3.3 `CPL.forward()` 的准确顺序和形状

核心文件为 `cpl_lrev/models/cpl.py`。令：

- `B`：batch size；
- `T=200`：输入视频步数；
- `D=256`：hidden size；
- `W<=20`：query token 数；
- `N`：原始 proposal 数；
- `Tp=T//4=50`：重构路径时间步数；
- ActivityNet mixture 总 component 数为 `1+2+3+4+5=15`。

真实 forward 为：

1. 给 `frames_feat [B,T,Dv]` 追加 `pred_vec`，经 dropout 和 `frame_fc` 得到 `[B,T+1,D]`；
2. 将 query 的首 token 替换为 `start_vec`，经 `word_fc` 和位置编码；
3. `self.trans(..., decoding=1)` 返回：
   - `enc_out [B,W+1,D]`：先自注意力得到的 query states；
   - `h [B,T+1,D]`：与 query 交互后的 video states；
4. `proposal_generator_feature=h[:,-1] [B,D]`；
5. 之后才把前 200 个 projected frame 通过 `linspace` 点采样到 50 步；
6. single Gaussian 由 `fc_gauss` 直接预测参数；mixture 路径先预测 component，再逐 component 重构 query，并由 reconstruction summary 得到 component importance；
7. `GaussianMixtureProposalGenerator.combine()` 才输出最终 `center/width`；
8. 最终 Gaussian/mixed mask 参与第二次 `self.trans(...,decoding=2)`，输出 `words_logit [B*N,W,V]`；
9. negative/LREV 分支继续计算负重构、reference 重构、event vector 和 event score。

BPSE 的语义状态必须在第 3 步后保存，但对最终候选的评分必须等到第 7 步之后。推荐保存：

```text
bpse_visual_states   = projected_frames[:, :T]  # [B,T,D]，跨模态交互前
bpse_grounded_states = h[:, :T]                 # [B,T,D]，query-conditioned
bpse_query_states    = enc_out[:, 1:]            # [B,W,D]，排除 start token
bpse_frame_mask      = frames_mask[:, :T]        # [B,T]
bpse_query_mask      = words_mask[:, 1:]         # [B,W]
```

不能在 `_mask_words()` 后才构造 query unit，否则训练和推理的完整性定义会受随机遮词影响。

### 3.4 Gaussian mixture 的边界语义

`cpl_lrev/models/modules/gaussian_mixture.py::GaussianMixtureProposalGenerator` 包含：

- `center_head: Linear(D,total_components)`；
- `width_head: Linear(D,N)`；
- 一个 proposal 内 component 共享 width；
- `predict_components()` 产生 component Gaussian masks；
- `combine()` 用重构 summary 预测 component importance；
- `boundary_mode='outer'` 取所有 component 的外包络；`weighted` 取 importance 加权边界；
- negative mining 可使用未收缩 outer envelope。

当前 ActivityNet `main.json` 和检查过的 `model-best.pt` 均为 `boundary_mode='outer'`，而 README 仍描述 `weighted` 且引用仓库中不存在的 `final_weighted_diagnostic.json`。后续实现与实验记录必须以实际 config/checkpoint 为准，不得根据 README 静默切换边界模式。

### 3.5 当前 loss 和训练 loop

`cpl_lrev/models/loss.py` 有四组 loss：

- `rec_loss()`：只对每个样本的最小 proposal NLL 求均值；
- `ivc_loss()`：同样用 `argmin NLL` 选正 proposal，再与 reference/negative 比较；
- `event_disentanglement_loss()`：LREV/BECL；
- `mixture_pull_push_loss()`：mixture component 和 proposal 几何约束。

`MainRunner._train_one_epoch()` 执行：

```text
forward
  -> rec_loss
  -> ivc_loss
  -> event_disentanglement_loss
  -> mixture_pull_push_loss
  -> loss 求和
  -> backward
  -> clip_grad_norm_(10)
  -> optimizer.step
  -> scheduler.step_update
```

BPSE 应作为第五组独立 loss 接入；此外，只有在 BPSE routing 生效阶段，才给 `rec_loss/ivc_loss` 传 `proposal_weights`，以替代硬 `min/argmin`。关闭 BPSE 时必须走原分支。

### 3.6 当前 inference 不是 `generate()`

仓库没有 `generate()`。`MainRunner.eval()` 在 `torch.no_grad()` 下调用同一个 `CPL.forward()`，然后：

1. 计算每个 proposal 的 reconstruction NLL；
2. 可选减去 event score；
3. 以较小 cost 为优进行排序；
4. 把 `center/width` 转为 `[start,end]`；
5. 通过 `nll/geometric_vote/semantic_vote` 选 Rank-1；
6. 计算 R@1/R@5 的 mIoU 和 IoU@0.1/0.3/0.5/0.7/0.9。

因此“BPSE inference/generation”在本仓库中的准确含义是：仍调用 `forward()` 生成候选，但以 `-quality_logit` 作为排序 cost；不生成 caption，不做 autoregressive decoding，也不进行 GRPO sampling。

### 3.7 Checkpoint 的实际行为

`MainRunner._save_model()` 当前仅保存：

```python
{
    "num_updates": ...,
    "config": ...,
    "model_parameters": self.model.state_dict(),
}
```

没有保存 optimizer state、scheduler state、epoch 或随机数状态。`_load_model()` 名为 resume，但只恢复模型参数和 `num_updates`；它不能做到严格的优化器级续训。

`_load_pretrained_model()` 是 V3 -> V4 的特殊 warm start，会主动跳过 LREV subspace，不适合直接用于 V4 baseline -> BPSE。需要增加一个单独的兼容加载入口，保留所有同名同形状的 V4/LREV 参数，只让 `bpse.*` 随机或身份初始化。

## 4. BPSE 数学定义与代码张量

### 4.1 Query 原子单元

#### 4.1.1 推荐的无外部模型规则

当前 dataset 已经执行 NLTK tokenize 和 POS tagging，因此首版不需要新 NLP 依赖。对过滤 OOV 后、截断前的 token 序列构造以下 unit：

1. **action unit**：每个 `VB*` token，与紧邻的 `RB*`、`RP` 以及同一标点片段内最近的左右 `NN*/PRP*` 合为一个稀疏 mask；
2. **entity-detail unit**：最大连续 `JJ*/CD/NN*` span；
3. **relation unit**：`IN/TO/RP` 及其左右最近的 content token；
4. **whole-query unit**：覆盖全部保留 token，作为原子化失败时的稳定回退。

对完全相同的 mask 去重。若 unit 数超过 `Umax=12`，按 action、relation、entity-detail 的顺序保留，whole-query 永远保留。若没有可用 content unit，只保留 whole-query。

dataset 返回：

| 变量 | 形状 | dtype | 含义 |
|---|---:|---|---|
| `unit_token_mask` | `[B,Umax,Wmax]` | `float32` | unit 覆盖哪些 query token |
| `unit_valid_mask` | `[B,Umax]` | `bool` | padding unit 是否有效 |
| `unit_type_ids` | `[B,Umax]` | `long` | `0=action,1=entity_detail,2=relation,3=whole` |
| `unit_weights` | `[B,Umax]` | `float32` | 完整性聚合权重，按有效 unit 归一化 |

必须在 vocab 过滤之后构造 mask，因为 `enc_out[:,1:]` 只对应保留下来的词。collate 截断到 `max_num_words` 时，也必须同步裁剪 unit mask 并删除变成空集的 unit。

推荐默认权重为：action `1.5`、entity-detail `1.0`、relation `1.0`、whole-query `0.25`。whole unit 只用于稳定，不应压过细粒度 unit。

#### 4.1.2 Unit representation

设 query state 为 `Q in R^(B x W x D)`，unit mask 为 `A in {0,1}^(B x U x W)`：

\[
u_{b,l}=
\frac{\sum_w A_{b,l,w}Q_{b,w}}
{\sum_w A_{b,l,w}+\epsilon}.
\]

代码变量：

```text
query_units [B,U,D], float32/当前 autocast dtype, 与 query_states 同 device
unit_valid_mask [B,U], bool
```

该池化需要 gradient 到 `query_states`，但首个 quality-head warm-up 阶段 baseline 可整体冻结。

### 4.2 原始候选和可微 soft-box

现有输出为扁平 `center,width [B*N]`。先 reshape：

\[
s_{b,n}=\operatorname{clamp}(c_{b,n}-w_{b,n}/2,0,1),
\quad
e_{b,n}=\operatorname{clamp}(c_{b,n}+w_{b,n}/2,0,1).
\]

得到 `base_spans [B,N,2]`。对归一化时间坐标 `p_t=t/(T-1)` 构造可微 membership：

\[
m_{g,t}=\sigma\left(\frac{p_t-s_g}{\tau_b}\right)
\sigma\left(\frac{e_g-p_t}{\tau_b}\right),
\]

其中 `tau_b=0.02`。`g` 可以是原始候选，也可以是训练时扰动候选。

推荐接口：

```python
def build_soft_box_masks(
    spans: torch.Tensor,       # [B,G,2]
    frame_mask: torch.Tensor,  # [B,T], bool/0-1
    temperature: float = 0.02,
) -> torch.Tensor:             # [B,G,T]
    ...
```

mask 与 span 保持梯度。padding frame 必须在 sigmoid 后乘掉；归一化分母一律 `clamp_min(eps)`。

### 4.3 帧—unit 证据

推荐保留两类 video state：

- `grounded_states [B,T,D]`：第一次跨模态交互后的 `h[:,:T]`，用于完整性；
- `visual_states [B,T,D]`：`frame_fc` 后、跨模态交互前的纯视觉状态，用于纯净性 event token，降低 query 泄漏。

使用身份初始化的线性投影和 LayerNorm：

\[
\hat v_t=\operatorname{norm}(W_v v_t),
\qquad
\hat u_l=\operatorname{norm}(W_u u_l).
\]

完整性证据：

\[
E^{cov}_{t,l}=\frac{1+\cos(\hat h_t,\hat u_l)}{2}\in[0,1].
\]

代码形状为 `coverage_evidence [B,T,U]`。无效 frame/unit 在聚合前 mask，不允许先平均再 mask。

投影层应初始化为单位矩阵、bias 为 0。这样从 baseline warm start 后，语义分数不是随机的。默认 `quality_detach_inputs=true`，质量头训练不会反向修改证据场；后续可把投影解冻作为 ablation。

### 4.4 完整性计算

对每个候选 `g` 和 unit `l`，在候选内部做 soft maximum pooling：

\[
\alpha_{g,t,l}=
\operatorname{softmax}_{t}\left(
\frac{E^{cov}_{t,l}+\log(m_{g,t}+\epsilon)}{\tau_{pool}}
\right),
\]

\[
c_{g,l}=\sum_t\alpha_{g,t,l}E^{cov}_{t,l}.
\]

再按 unit 权重平均：

\[
C_g=\frac{\sum_l a_l c_{g,l}v_l}
{\sum_l a_lv_l+\epsilon},
\]

其中 `v_l` 是 `unit_valid_mask`。推荐 `tau_pool=0.1`。

直观上，`c_{g,l}` 回答第 `l` 个 query unit 能否在区间内找到至少一个支持位置，`C_g` 回答 query 是否整体被覆盖。

返回：

```text
unit_coverage [B,G,U]
completeness [B,G]
```

若使用 FP16，应先以 FP32 计算 `log(mask+eps)` 和 softmax，再 cast 回原 dtype，防止窄区间出现 `-inf/NaN`。

### 4.5 显著事件 token

原 proposal 提到“以局部峰值形成显著事件 token”。直接 `topk + NMS` 会引入离散索引、短候选无 token 和训练抖动。推荐实现为**分箱内软峰值池化**：

1. 把 `T=200` 均匀分为 `J=16` 个连续时间箱；
2. 对纯视觉状态计算 motion-detail saliency：

\[
a_t=lambda_m\|\bar v_t-\bar v_{t-1}\|_2
+\lambda_d\|\bar v_t-\mu_v\|_2+a_0,
\]

其中 `bar v` 为归一化视觉状态，`a0=0.1` 是静态细节 floor；
3. 在每个箱内对 `a_t/tau_event` 做 softmax，得到局部峰值权重 `rho_{j,t}`；
4. 池化：

\[
z_j=\sum_{t\in bin(j)}\rho_{j,t}v_t.
\]

同时保留 `event_pool_weights [B,J,T]`。它比离散 top-k 更适合当前弱监督和短事件：每个时间区域至少有一个 token，局部变化大的帧获得更高权重，静态内容也不会被完全忽略。

推荐返回：

| 变量 | 形状 | 说明 |
|---|---:|---|
| `event_tokens` | `[B,J,D]` | 纯视觉局部事件/细节 token |
| `event_strength` | `[B,J]` | 箱内 saliency，归一化后含 floor |
| `event_pool_weights` | `[B,J,T]` | 每个 event token 对 frame 的权重 |
| `event_valid_mask` | `[B,J]` | 该箱是否含有效 frame |

`event_pool_weights` 对视觉状态保留 gradient。时间箱边界是固定结构，无需 gradient。

### 4.6 纯净性计算

先做 event-to-query-unit 匹配：

\[
E^{pur}_{j,l}=\frac{1+\cos(W_z z_j,W_u u_l)}{2}.
\]

每个事件被 query 集合解释的程度，用 unit 方向的 soft maximum 表示：

\[
x_j=\sum_l
\operatorname{softmax}_{l}(E^{pur}_{j,l}/\tau_u)
E^{pur}_{j,l}.
\]

候选对 event token 的软包含度不只看箱中心，而是把 frame soft-box 通过 event pooling 权重积分：

\[
\tilde m_{g,j}=\sum_t\rho_{j,t}m_{g,t}.
\]

最后：

\[
P_g=
\frac{\sum_j \tilde m_{g,j}\,a_j\,x_j\,v_j}
{\sum_j \tilde m_{g,j}\,a_j\,v_j+\epsilon}.
\]

其中 `P_g in [0,1]` 即纯净性。宽候选若包含无法被任一 query unit 解释的显著事件，其分母增加而分子增加较少，`P_g` 会下降。

返回：

```text
event_unit_evidence [B,J,U]
event_explainability [B,J]
event_membership [B,G,J]
purity [B,G]
```

为防止极窄候选因有效 event mass 太小而得到偶然高分，额外返回：

\[
M_g=\sum_j\tilde m_{g,j}a_jv_j.
\]

质量头输入中加入 `log1p(M_g)` 或归一化 mass；不要直接把 `P_g` 在低 mass 时置零，因为这会系统性误罚短事件。pair 训练时可以过滤 `M_g < mass_min` 的候选。

### 4.7 Inside-shell contrast

只看候选内部仍可能无法分辨“刚好覆盖目标”与“边界外还有同一事件”。对候选向外扩 `shell_width=0.05`：

\[
m^{shell}_{g,t}=\operatorname{clamp}
(m^{expanded}_{g,t}-m_{g,t},0,1).
\]

令 `r_t=max_l E_cov(t,l)` 的平滑版本表示 frame 对 query 的最大支持度：

\[
I_g=\operatorname{mean}_{m_g}(r_t),
\qquad
O_g=\operatorname{mean}_{m^{shell}_g}(r_t),
\qquad
D_g=I_g-O_g.
\]

`D_g` 大表示区间内部比紧邻外壳更符合 query；若候选过窄，shell 仍含目标证据，`D_g` 会下降。该特征与纯净性互补：纯净性惩罚内部多余事件，shell contrast 检查是否漏掉边界外的必要证据。

若候选贴近视频首尾导致一侧 shell 为空，必须按实际有效 shell mass 归一化；两侧都为空时 `D_g=0`，并返回 `shell_valid=false`，不能制造伪高分。

### 4.8 双向等价分数

默认用调和平均融合完整性与纯净性：

\[
H_g=\frac{2C_gP_g}{C_g+P_g+\epsilon}.
\]

调和平均只有在两向都高时才高，能抑制：

- `C` 高、`P` 低的宽候选；
- `P` 高、`C` 低的过窄候选。

代码变量：

```text
bpse_completeness [B,G]
bpse_purity       [B,G]
bpse_equivalence  [B,G]
```

三者都为 float，理论范围 `[0,1]`。计算前后可 `clamp(0,1)`；`eps` 推荐 `1e-6`。

### 4.9 质量头

首版质量头输入为：

\[
f_g=[C_g,P_g,H_g,w_g,D_g,\log(1+M_g)].
\]

推荐结构：

```python
self.quality_head = nn.Sequential(
    nn.Linear(6, quality_hidden_size),
    nn.GELU(),
    nn.Dropout(quality_dropout),
    nn.Linear(quality_hidden_size, 1),
)
```

默认 `quality_hidden_size=64`、`quality_dropout=0.1`。最后一层权重和 bias 初始化为 0，使刚加载 baseline 时所有候选质量相同，不引入随机排序偏置。

输出：

```text
quality_logits [B,G]  # 越大越好
quality_probs  [B,G]  # sigmoid(logit)
```

默认将 `f_g.detach()` 后送入质量头，即 `quality_detach_inputs=true`。这样停止梯度的伪标签只训练质量头，不会通过“改写 C/P 特征本身”降低 BCE。证据投影是否微调作为后续 ablation 单独开启。

### 4.10 训练时的局部扰动候选组

对每个基础候选 `[s,e]` 构造一个独立组。默认 `delta=0.05`，组内含 9 个候选：

1. identity `[s,e]`；
2. trim-left `[s+delta,e]`；
3. trim-right `[s,e-delta]`；
4. expand-left `[s-delta,e]`；
5. expand-right `[s,e+delta]`；
6. shrink-both `[s+delta,e-delta]`；
7. expand-both `[s-delta,e+delta]`；
8. shift-left `[s-delta,e-delta]`；
9. shift-right `[s+delta,e+delta]`。

全部 clamp 到 `[0,1]`。宽度小于 `min_width=0.02`、与另一候选数值重复或变化量小于 `1e-4` 的条目标为 invalid。

返回：

```text
group_spans     [B,N,R,2]
group_valid     [B,N,R]
group_scores    [B,N,R]
group_logits    [B,N,R]
R = 1 + 8 * len(perturb_offsets)
```

扰动候选从 `base_spans.detach()` 构造，只负责训练质量排序器，不允许人工 perturbation 的边界梯度反向改变 proposal generator。原始候选另走一份保留梯度的 `base_spans`，供可选的边界等价损失使用。

推理阶段默认 `build_perturbations=false`，只计算 `G=N` 个原始候选，避免 9 倍候选评分开销。

推荐接口：

```python
def build_proposal_groups(
    base_spans: torch.Tensor,              # [B,N,2]
    offsets: tuple = (0.05,),
    min_width: float = 0.02,
    dedup_eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:    # spans [B,N,R,2], valid [B,N,R]
    ...
```

### 4.11 组内偏好和绝对质量损失

伪目标一律停止梯度：

```text
target_equivalence = group_equivalence.detach()
```

只比较同一基础候选的局部变体。若 `H_i-H_j >= pair_margin`，则 `i` 应排在 `j` 前：

\[
L_{rank}=\frac{1}{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}
\operatorname{softplus}\left(
-\frac{q_i-q_j}{\tau_r}
\right).
\]

默认 `pair_margin=0.05`、`rank_temperature=0.2`。valid mask、低 event mass、重复 span 都不能形成 pair。若一个 batch 没有有效 pair，返回 `quality_logits.sum()*0`，保证图连接和 backward 安全。

绝对质量项为：

\[
L_{abs}=\operatorname{BCEWithLogits}(q_g,\operatorname{stopgrad}(H_g)).
\]

推荐总质量监督：

\[
L_{quality}=\lambda_{rank}L_{rank}+\lambda_{abs}L_{abs}.
\]

默认 `lambda_rank=1.0`、`lambda_abs=0.5`。必须记录：有效 pair 数、pair 覆盖率、目标分数标准差和 pairwise accuracy。若 `H` 在组内几乎无差异，rank loss 看似 finite 但没有研究意义。

### 4.12 对基础候选的完整性/等价约束

原 proposal 写为 `lambda_cov(1-C+)`。直接 hard argmax 会重现赢家通吃，因此推荐使用停止梯度的 soft winner：

\[
\omega_n=\operatorname{softmax}
(\operatorname{stopgrad}(H_n)/\tau_s),
\]

\[
L_{cov}=\frac1B\sum_b\sum_n\omega_{b,n}(1-C_{b,n}).
\]

为使纯净性不只影响排序器，可选加入：

\[
L_{equiv}=\frac1B\sum_b\sum_n\omega_{b,n}(1-H_{b,n}).
\]

推荐首轮：`lambda_cov=0.1`、`lambda_equiv=0.0`。先验证原 proposal 的最小设计；若完整性让候选继续变宽，再开启小权重 `lambda_equiv=0.05`。

对 `L_cov/L_equiv`，推荐把 evidence tensor detach、保留 soft-box mask 对 `center/width` 的 gradient：

```text
evidence: stop-gradient
span -> soft-box membership: gradient enabled
quality soft winner omega: stop-gradient
```

这使损失主要移动候选边界，而不是把所有 frame/unit similarity 人为抬高。该行为应由配置 `span_loss_detach_evidence=true` 控制。

### 4.13 用 BPSE 替代 hard-min routing

基础候选的质量权重为：

\[
r_n=\operatorname{softmax}
(\operatorname{stopgrad}(q_n)/\tau_{route}).
\]

默认 `route_temperature=0.2`。BPSE warm-up 完成后，将它传给现有 loss：

#### Reconstruction

当前：

\[
L_{rec}=\min_n NLL_n.
\]

BPSE 路径：

\[
L_{rec}^{BPSE}=\sum_n r_n NLL_n.
\]

#### IVC negative/reference ranking

当前 `ivc_loss()` 用 `argmin NLL` 选择正候选，再 gather 对应左右负候选。BPSE 路径应改成：

\[
NLL^+=\sum_n r_nNLL_n,
\quad
NLL^-_L=\sum_n r_nNLL^-_{L,n},
\quad
NLL^-_R=\sum_n r_nNLL^-_{R,n}.
\]

再沿用现有 margin hinge。`proposal_weights=None` 时保留原始 min/argmin 分支。

质量头最初为全零，不能从第一个 step 就控制 routing。推荐 `routing_start_epoch=3`；此前 `rec_loss/ivc_loss` 仍走 baseline hard-min，或者使用均匀权重作为单独 ablation。第 3 epoch 后线性 ramp 2 个 epoch。

### 4.14 推理排序

BPSE logit 越大越好，而 runner 现有 `proposal_score` 越小越好。统一转换：

```python
selection_cost = -output["bpse_quality_logits"]  # [B,N]
idx = selection_cost.argsort(dim=-1)
```

随后对 `center/width`、cost 及所有诊断量使用同一个 `idx` gather。主实验 `selection_strategy='bpse'` 时 Rank-1 直接取排序后的第 0 个候选。

候选生成、候选数量和区间本身不因推理排序改变，因此 ActivityNet 的 R@5（使用全部 5 个候选）理论上应基本不变；主要观察 R@1。Charades 当前有 8 个候选且 R@5 只取前 5，BPSE 会同时影响 R@1 和 R@5。

可选 `bpse_nll_hybrid` 只能作为消融：分别在样本内标准化 q 和 NLL 后线性融合。主结论必须来自纯 BPSE 排序，才能检验独立质量估计假设。

## 5. 推荐模块接口

### 5.1 `QueryUnitPooler`

```python
class QueryUnitPooler(nn.Module):
    def forward(
        self,
        query_states: torch.Tensor,      # [B,W,D]
        query_mask: torch.Tensor,        # [B,W]
        unit_token_mask: torch.Tensor,   # [B,U,W]
        unit_valid_mask: torch.Tensor,   # [B,U]
    ) -> torch.Tensor:                   # [B,U,D]
        ...
```

要求：

- `query_mask` 与 `unit_token_mask` 相乘；
- 空 unit 输出 0 且在 valid mask 中为 false；
- 不在函数内部 `.cuda()`，device 跟随输入；
- 分母 `clamp_min(1)`。

### 5.2 `SalientEventTokenizer`

```python
class SalientEventTokenizer(nn.Module):
    def forward(
        self,
        visual_states: torch.Tensor,  # [B,T,D]
        frame_mask: torch.Tensor,     # [B,T]
    ) -> dict[str, torch.Tensor]:
        """Return event_tokens, event_strength, event_pool_weights,
        event_valid_mask."""
```

模块只做确定性分箱和可微池化，不维护跨 batch memory。

### 5.3 `BPSEScorer`

```python
class BPSEScorer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_query_units: int = 12,
        num_event_tokens: int = 16,
        box_temperature: float = 0.02,
        coverage_pool_temperature: float = 0.1,
        unit_match_temperature: float = 0.1,
        shell_width: float = 0.05,
        quality_hidden_size: int = 64,
        quality_dropout: float = 0.1,
        quality_detach_inputs: bool = True,
        eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        visual_states: torch.Tensor,       # [B,T,D]
        grounded_states: torch.Tensor,     # [B,T,D]
        frame_mask: torch.Tensor,          # [B,T]
        query_states: torch.Tensor,        # [B,W,D]
        query_mask: torch.Tensor,          # [B,W]
        unit_token_mask: torch.Tensor,     # [B,U,W]
        unit_valid_mask: torch.Tensor,     # [B,U]
        unit_weights: torch.Tensor,        # [B,U]
        base_spans: torch.Tensor,          # [B,N,2]
        build_perturbations: bool = False,
        perturb_offsets: tuple = (0.05,),
    ) -> dict[str, torch.Tensor]:
        ...
```

推荐返回键：

```text
quality_logits             [B,N]
quality_probs              [B,N]
completeness               [B,N]
purity                     [B,N]
equivalence                [B,N]
inside_shell_contrast      [B,N]
event_mass                 [B,N]
unit_coverage              [B,N,U]
group_quality_logits       [B,N,R] or None
group_completeness         [B,N,R] or None
group_purity               [B,N,R] or None
group_equivalence          [B,N,R] or None
group_valid_mask           [B,N,R] or None
group_spans                [B,N,R,2] or None
```

不要返回巨大的 `[B,G,T,U]` 中间张量到 runner；它们只在模块内部使用，避免显存和 checkpoint/debug 输出膨胀。

### 5.4 `bpse_loss`

```python
def bpse_loss(
    words_logit: torch.Tensor,
    bpse_quality_logits: torch.Tensor | None = None,        # [B,N]
    bpse_completeness: torch.Tensor | None = None,          # [B,N]
    bpse_purity: torch.Tensor | None = None,                # [B,N]
    bpse_equivalence: torch.Tensor | None = None,           # [B,N]
    bpse_group_quality_logits: torch.Tensor | None = None,  # [B,N,R]
    bpse_group_equivalence: torch.Tensor | None = None,     # [B,N,R]
    bpse_group_valid_mask: torch.Tensor | None = None,      # [B,N,R]
    bpse_group_spans: torch.Tensor | None = None,           # [B,N,R,2]
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

若 BPSE 未启用或字段为 `None`，返回 `words_logit.sum()*0` 和全 0 metrics。禁止创建 CPU 常量 `torch.tensor(0.)`，否则总 loss device 不一致且图断开。

### 5.5 `rec_loss/ivc_loss` 的兼容接口

```python
def rec_loss(
    ...,
    proposal_weights: torch.Tensor | None = None,  # [B,N], detached
    **kwargs,
):
    ...

def ivc_loss(
    ...,
    proposal_weights: torch.Tensor | None = None,  # [B,N], detached
    **kwargs,
):
    ...
```

`None` 必须执行当前逐行等价逻辑；传入时先检查 shape、finite、非负，并重新归一化。

## 6. 端到端数据流

### 6.1 Dataset / preprocessing

```text
JSON: [vid, duration, timestamps, sentence]
    -> 现有 NLTK tokenize/POS + vocab 过滤
    -> GloVe word tensors
    -> 新增规则 query units
    -> unit_token_mask / unit_type / unit_weight

HDF5 C3D/I3D
    -> 现有 200-step 均值采样
    -> frames_feat

collate
    -> 原有 net_input
    -> 新增 unit tensors
```

训练 timestamp 继续只留在 `raw`，不能进入 BPSE loss。

### 6.2 训练 forward

```text
frames_feat / words_feat
    -> frame_fc / word_fc
    -> 第一次 DualTransformer
    -> 保存 visual_states / grounded_states / full query_states
    -> 原 proposal generator
    -> mixture component reconstruction + combine
    -> base center/width/spans
    -> BPSE：query unit pooling
    -> BPSE：coverage evidence + completeness
    -> BPSE：salient event tokens + purity
    -> BPSE：inside-shell contrast + quality head
    -> BPSE：仅训练时构造 perturbation groups 并评分
    -> 原 proposal masked reconstruction / negative / LREV
    -> forward output dict
    -> rec + ivc + event + mixture + bpse losses
    -> backward / clipping / optimizer / scheduler
```

训练阶段特有：扰动候选、group pair mask、`L_rank/L_abs/L_cov/L_equiv`、routing schedule。

### 6.3 推理 / evaluation

```text
与训练相同的 dataset 和基础 forward
    -> 不构造 perturbation groups
    -> 原 N 个 proposal
    -> BPSE quality_logits [B,N]
    -> selection_cost = -quality_logits
    -> 降序质量排序
    -> Rank-1 / top-5 metrics
```

训练与推理共享：unit 构造规则、visual/query state、事件 token、C/P/H、质量头。推理独有：只对基础候选排序。训练独有的 perturbation 信息绝不能成为推理依赖。

### 6.4 Eval 随机遮词的处理

当前 `_mask_words()` 在 `model.eval()` 下仍使用 `np.random.choice`。BPSE 本身使用遮词前的 query states，但 mixture component importance 和最终 outer/weighted boundary 仍可能受随机遮词影响。

推荐在 `CPL._mask_words()` 增加 `eval_word_mask_mode`：

- `legacy_random`：完全复现当前行为；BPSE 关闭时默认；
- `deterministic_topk`：按 masking weight 和 token index 稳定选择；OWLCAP 主实验推荐；
- `none`：不遮词，仅作消融。

该修正不是 BPSE 核心算法，必须单独报告。不能一边使用 deterministic mask，一边把全部增益归因于 BPSE。

## 7. 独立 `owlcap/` 工程目录

推荐初始结构：

```text
owlcap/
├── README.md
├── requirements.txt
├── train.py
├── utils.py
├── vocab.py
├── config/
│   ├── activitynet/
│   │   ├── baseline.json
│   │   └── bpse.json
│   └── charades/
│       ├── baseline.json
│       └── bpse.json
├── data/
│   ├── activitynet/
│   └── charades/
├── datasets/
│   ├── __init__.py
│   ├── base.py
│   ├── activitynet.py
│   └── charades_sta.py
├── models/
│   ├── __init__.py
│   ├── cpl.py
│   ├── loss.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── bpse.py
│   │   ├── gaussian_mixture.py
│   │   └── ... baseline modules
│   └── transformer/
├── optimizers/
├── runners/
│   ├── __init__.py
│   └── main_runner.py
├── scripts/
│   ├── train_activitynet_bpse.sh
│   └── eval_activitynet_bpse.sh
├── tools/
│   └── scan_selectors.py
├── tests/
│   ├── test_query_units.py
│   ├── test_bpse_module.py
│   ├── test_bpse_loss.py
│   ├── test_bpse_selection.py
│   ├── test_bpse_integration.py
│   └── ... copied baseline tests
└── checkpoints/
    └── bootstrap/
```

复制基线时只复制运行必需的源码、config、词表/JSON 索引和测试；不要复制全部历史日志和几十个 epoch checkpoint。HDF5 继续用绝对路径读取。

独立性检查：

```bash
cd /data/chenyuan/videogrounding/w1m2/owlcap
rg -n "import cpl_lrev|from cpl_lrev|\.\./cpl_lrev|sys\.path.*cpl_lrev" .
```

结果必须为空；同时检查 `models.__file__`、`datasets.__file__`、`runners.__file__` 均解析到 `owlcap/`。

## 8. 文件级详细修改方案

### 8.1 `owlcap/datasets/base.py`——必须修改

#### 当前职责

对应基线 `BaseDataset.__getitem__()` 完成 NLTK token/POS、vocab 过滤、GloVe 构造和视频采样；`build_collate_data()` 负责 padding 和 tensor 化。

#### 新增内容

1. 新增纯函数 `build_query_units(tagged_kept_tokens, max_units)`；
2. 在 `__getitem__()` 的同一次 POS 遍历中保存 kept token 的原始 POS；
3. 返回 `unit_token_mask/unit_type_ids/unit_valid_mask/unit_weights`；
4. collate 为 `[B,Umax,Wbatch]` 等固定 tensor；
5. 截断 words 后同步截断 unit，并对空 unit 重新计算 valid；
6. OWLCAP 关闭时不构造 unit，保持基线数据开销。

建议接口：

```python
def build_query_units(
    words: list[str],
    pos_tags: list[str],
    max_units: int,
    include_whole_query: bool = True,
) -> dict[str, np.ndarray]:
    ...
```

返回的 mask token 维应对应 `words_id`，不包含 start placeholder。

#### 配置读取

从 `args.get('query_units', {})` 读取 `enabled/max_units/include_whole_query/type_weights`。

#### 兼容性

`CPL.forward(..., **kwargs)` 可忽略额外 tensor，但为避免不必要的 GPU 拷贝，baseline config 中 `query_units.enabled=false`。

### 8.2 `owlcap/datasets/activitynet.py`、`charades_sta.py`——推荐小改

两者当前只创建 collate closure 和读取 HDF5。若 `build_collate_data()` 需要 query-unit config，应把整个 `args` 或相关参数传入，而不是不断扩展位置参数。

推荐把接口从：

```python
build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim)
```

改为兼容可选参数：

```python
build_collate_data(
    max_num_frames,
    max_num_words,
    frame_dim,
    word_dim,
    query_unit_config=None,
)
```

旧调用不传最后一项时行为不变。

### 8.3 `owlcap/models/modules/bpse.py`——新增，必须

包含：

- `QueryUnitPooler`；
- `SalientEventTokenizer`；
- `build_soft_box_masks()`；
- `build_proposal_groups()`；
- `BPSEScorer`；
- shape/finite/value-range 的内部检查。

不要把 loss 或 runner 的 NumPy 排序写入该文件。该模块应能用纯随机小 tensor 在 CPU 上单测。

### 8.4 `owlcap/models/modules/__init__.py`——必须修改

新增：

```python
from .bpse import BPSEScorer
```

不要改变其他 export 名称。

### 8.5 `owlcap/models/cpl.py`——必须修改

#### `CPL.__init__()`

读取：

```python
bpse_config = config.get("bpse", {})
self.use_bpse = bpse_config.get("enabled", False)
```

仅在 enabled 时实例化 `BPSEScorer`。保存 routing start/ramp、perturbation 和 eval masking 配置。所有默认值必须使旧 config 不报错。

#### `CPL.forward()` 输入

保留现有显式参数，query unit 通过可选 kwarg 获取：

```python
unit_token_mask = kwargs.get("unit_token_mask")
unit_valid_mask = kwargs.get("unit_valid_mask")
unit_weights = kwargs.get("unit_weights")
```

enabled 时若缺失，抛出包含 shape 期望的 `ValueError`；disabled 时完全忽略。

#### 保存高分辨率状态

在 `frame_fc` 后、变量被 200→50 覆盖前保存视觉状态。在第一次 `self.trans()` 后保存 grounded/query states。不能改变原变量参与 baseline 的数值。

#### 调用位置

mixture 分支必须在 `combine()` 得到最终 `gauss_center/gauss_width` 后调用；single Gaussian 在 `generate_gauss_weight()` 前后均可，推荐统一在两分支汇合后、`pos_weight` 计算前调用。

伪代码：

```python
bpse_output = {}
if self.use_bpse:
    center_bn = gauss_center.view(bsz, self.num_props)
    width_bn = gauss_width.view(bsz, self.num_props)
    base_spans = torch.stack([
        (center_bn - width_bn / 2).clamp(0, 1),
        (center_bn + width_bn / 2).clamp(0, 1),
    ], dim=-1)
    bpse_output = self.bpse_scorer(
        visual_states=bpse_visual_states,
        grounded_states=bpse_grounded_states,
        frame_mask=bpse_frame_mask.bool(),
        query_states=bpse_query_states,
        query_mask=bpse_query_mask.bool(),
        unit_token_mask=unit_token_mask,
        unit_valid_mask=unit_valid_mask,
        unit_weights=unit_weights,
        base_spans=base_spans,
        build_perturbations=self.training,
        perturb_offsets=self.bpse_perturb_offsets,
    )
```

#### Output dict

新增键统一加 `bpse_` 前缀。至少返回：

```text
bpse_quality_logits
bpse_quality_probs
bpse_completeness
bpse_purity
bpse_equivalence
bpse_inside_shell_contrast
bpse_event_mass
bpse_group_quality_logits
bpse_group_equivalence
bpse_group_valid_mask
bpse_group_spans
bpse_routing_weights
bpse_schedule
```

`bpse_routing_weights [B,N]` 必须 detach。eval 时 group 字段为 `None`。

#### Masking

给 `_mask_words()` 增加可选 mode，但 baseline disabled 时保持当前 `np.random.choice` 行为。不要顺便重写所有 `.cuda()`；device portability 是独立清理任务。

### 8.6 `owlcap/models/loss.py`——必须修改

1. 新增 `bpse_loss()`；
2. `rec_loss()` 增加 `proposal_weights=None`；
3. `ivc_loss()` 增加 `proposal_weights=None`；
4. 无有效 pair 时返回 graph-connected zero；
5. 所有 mask 使用 bool，统计量 `.item()` 前 detach；
6. 每项 loss 和关键诊断单独放入 dict。

建议日志键：

```text
bpse_loss
bpse_rank_loss
bpse_abs_loss
bpse_cov_loss
bpse_equiv_loss
bpse_valid_pairs
bpse_pair_coverage
bpse_pair_accuracy
bpse_target_std
bpse_mean_completeness
bpse_mean_purity
bpse_mean_equivalence
bpse_mean_quality
bpse_mean_route_entropy
```

### 8.7 `owlcap/runners/main_runner.py`——必须修改

#### Import 和训练

从 `models.loss` import `bpse_loss`。在四个现有 loss 后计算第五项。根据 epoch 和 `output['bpse_schedule']` 决定是否把 `bpse_routing_weights` 传给 `rec_loss/ivc_loss`。

推荐先得到 routing：

```python
proposal_weights = None
if output.get("bpse_schedule", 0.0) > 0:
    proposal_weights = output.get("bpse_routing_weights")
```

再调用 rec/ivc。不要把 group perturbation 当成新的 reconstruction proposal；它们只训练 BPSE head。

#### Eval

增加 `bpse` 选择策略：

```python
if self.selection_strategy == "bpse":
    if output.get("bpse_quality_logits") is None:
        raise ValueError("bpse selector requires model.bpse.enabled=true")
    proposal_score = -output["bpse_quality_logits"]
else:
    # 保留当前 NLL/event score 逻辑
```

排序后仍使用现有 `top_1_metric/top_n_metric`。同时计算但不参与选择的 NLL baseline 和 oracle，用于判断增益来自排序还是候选集合变化。

建议新增诊断：

- `bpse_R1_mIoU` 与同批 `nll_R1_mIoU`；
- `oracle_R1_mIoU`；
- `bpse_top1_width`；
- top-1 `C/P/H/q`；
- q 与真实 candidate IoU 的 pairwise ordering accuracy（仅 eval）；
- `R@1 - R@5` 排序差距；
- 按 GT short/medium/long 分组的 BPSE R@1。

GT 只在 `eval()` 的 NumPy 诊断中使用，不回传模型。

#### Dataset 配置一致性

在 `_build_dataset()` 后验证：model BPSE enabled 时，dataset `query_units.enabled` 也必须为 true，且 `max_units` 一致。尽早失败比 forward 中出现难懂 shape error 更好。

### 8.8 `owlcap/train.py`——必须修改

1. `--selection-strategy` choices 加 `bpse`；
2. 新增 `--init-from-cpl`，与 `--resume/--init-from-v3` 互斥；
3. 可选提供常用 loss override：
   - `--bpse-rank-weight`；
   - `--bpse-abs-weight`；
   - `--bpse-cov-weight`；
   - `--bpse-equiv-weight`；
4. 将 override 写入 `args['loss']`；
5. 若选择 `bpse` 但 config 未 enabled，在 runner 构造前报错。

不要为每个 BPSE 超参数都增加 CLI；结构参数放 JSON，避免入口膨胀。

### 8.9 `owlcap/config/activitynet/baseline.json`——必须新增

从当前实际 `cpl_lrev/config/activitynet/main.json` 复制，保持 `outer`、5 proposals、LREV inference weight 0，并显式加：

```json
"query_units": {"enabled": false},
"bpse": {"enabled": false}
```

用于验证独立工程的 baseline parity。

### 8.10 `owlcap/config/activitynet/bpse.json`——必须新增

在同一 baseline 上只开启 query units 和 BPSE。完整配置见第 9 节。

### 8.11 `owlcap/config/charades/*.json`——推荐新增

首个研究结论建议先在 ActivityNet 验证。Charades 当前 `val_data=test_data`，不适合未经说明的严格模型选择。实现接口仍须支持 `N=8`，但正式 Charades 实验前应建立独立 val split 或明确标为 legacy protocol。

### 8.12 `owlcap/runners/main_runner.py` 的 checkpoint 方法——必须/推荐

- 必须新增 `_load_cpl_baseline()`：加载所有同名同形状参数，包括 LREV buffer；允许缺少 `bpse.*`；不恢复旧优化器进度；
- strict `_load_model()` 只用于结构相同的 BPSE checkpoint；
- 推荐 `_save_model()` 增加 `epoch/optimizer_state`，并让旧 checkpoint 缺字段时仍可加载；
- 最终模型选择继续基于 validation，不使用 test。

### 8.13 `owlcap/tests/`——必须新增

详见第 13 节。复制原 baseline tests 后，import 必须解析到 `owlcap` 本地模块。

## 9. 配置 specification

### 9.1 Dataset 配置

| 参数 | 类型 | 默认值 | 读取位置 | 作用 |
|---|---|---:|---|---|
| `dataset.query_units.enabled` | bool | false | `BaseDataset` | 是否构造 query units |
| `dataset.query_units.max_units` | int | 12 | dataset/collate | 最大 unit 数 |
| `dataset.query_units.include_whole_query` | bool | true | unit builder | 是否加入整句回退 unit |
| `dataset.query_units.action_weight` | float | 1.5 | unit builder | action 完整性权重 |
| `dataset.query_units.entity_weight` | float | 1.0 | unit builder | entity-detail 权重 |
| `dataset.query_units.relation_weight` | float | 1.0 | unit builder | relation 权重 |
| `dataset.query_units.whole_weight` | float | 0.25 | unit builder | whole-query 权重 |

### 9.2 Model 配置

| 参数 | 类型 | 默认值 | 作用 |
|---|---|---:|---|
| `model.config.bpse.enabled` | bool | false | 总开关 |
| `max_query_units` | int | 12 | 与 dataset 一致 |
| `num_event_tokens` | int | 16 | 分箱事件 token 数 |
| `box_temperature` | float | 0.02 | soft-box 边界温度 |
| `coverage_pool_temperature` | float | 0.1 | 帧方向 softmax pool 温度 |
| `unit_match_temperature` | float | 0.1 | event 到 unit softmax 温度 |
| `motion_saliency_weight` | float | 1.0 | 视觉变化权重 |
| `detail_saliency_weight` | float | 0.5 | 相对全局差异权重 |
| `saliency_floor` | float | 0.1 | 静态细节最低权重 |
| `event_pool_temperature` | float | 0.1 | 箱内软峰值温度 |
| `shell_width` | float | 0.05 | 外壳宽度 |
| `quality_hidden_size` | int | 64 | MLP hidden |
| `quality_dropout` | float | 0.1 | MLP dropout |
| `quality_detach_inputs` | bool | true | 防止伪目标改写证据 |
| `perturb_offsets` | list[float] | `[0.05]` | 训练扰动幅度 |
| `min_perturb_width` | float | 0.02 | 最小有效扰动 span |
| `routing_start_epoch` | int | 3 | BPSE 接管 rec/ivc 的起始 epoch |
| `routing_ramp_epochs` | int | 2 | routing 线性 ramp |
| `route_temperature` | float | 0.2 | proposal routing softmax 温度 |
| `eval_word_mask_mode` | str | `deterministic_topk` | OWLCAP eval 遮词策略 |

### 9.3 Loss 配置

| 参数 | 类型 | 默认值 | 作用 |
|---|---|---:|---|
| `bpse_rank_weight` | float | 1.0 | 组内 pairwise ranking |
| `bpse_abs_weight` | float | 0.5 | BCE 绝对质量拟合 |
| `bpse_cov_weight` | float | 0.1 | soft winner 完整性 |
| `bpse_equiv_weight` | float | 0.0 | 可选双向边界优化 |
| `bpse_pair_margin` | float | 0.05 | 形成可靠 pair 的 H 差值 |
| `bpse_rank_temperature` | float | 0.2 | pair logistic 温度 |
| `bpse_soft_winner_temperature` | float | 0.1 | 基础候选 soft winner |
| `bpse_mass_min` | float | 0.01 | 过滤低 event mass pair |
| `bpse_span_loss_detach_evidence` | bool | true | span loss 只更新边界 |

### 9.4 推荐 JSON 片段

Dataset：

```json
"query_units": {
  "enabled": true,
  "max_units": 12,
  "include_whole_query": true,
  "action_weight": 1.5,
  "entity_weight": 1.0,
  "relation_weight": 1.0,
  "whole_weight": 0.25
}
```

Model：

```json
"bpse": {
  "enabled": true,
  "max_query_units": 12,
  "num_event_tokens": 16,
  "box_temperature": 0.02,
  "coverage_pool_temperature": 0.1,
  "unit_match_temperature": 0.1,
  "motion_saliency_weight": 1.0,
  "detail_saliency_weight": 0.5,
  "saliency_floor": 0.1,
  "event_pool_temperature": 0.1,
  "shell_width": 0.05,
  "quality_hidden_size": 64,
  "quality_dropout": 0.1,
  "quality_detach_inputs": true,
  "perturb_offsets": [0.05],
  "min_perturb_width": 0.02,
  "routing_start_epoch": 3,
  "routing_ramp_epochs": 2,
  "route_temperature": 0.2,
  "eval_word_mask_mode": "deterministic_topk"
}
```

Loss：

```json
"bpse_rank_weight": 1.0,
"bpse_abs_weight": 0.5,
"bpse_cov_weight": 0.1,
"bpse_equiv_weight": 0.0,
"bpse_pair_margin": 0.05,
"bpse_rank_temperature": 0.2,
"bpse_soft_winner_temperature": 0.1,
"bpse_mass_min": 0.01,
"bpse_span_loss_detach_evidence": true
```

所有参数在 BPSE disabled 时应被忽略，不能要求旧 config 补齐。

## 10. 推荐训练协议

### 10.1 Phase A：baseline parity

目标：证明独立工程复制正确，尚未引入 BPSE 数值变化。

- 使用 `baseline.json`；
- 从复制后的 baseline checkpoint eval；
- 固定原选择策略和 legacy mask；
- 结果应在浮点误差内一致；
- 所有旧单测通过。

### 10.2 Phase B：quality-head warm-up

建议 2–3 epoch：

- 加载 V4 baseline；
- 冻结 `frame_fc/word_fc/trans/proposal generator/reconstructor/LREV`；
- query unit、event tokenizer 若无参数则无需训练；身份投影保持冻结；
- 只训练 `bpse.quality_head`；
- 构造扰动组；
- 只启用 `L_rank + L_abs`；
- routing 关闭，不改变 rec/ivc winner；
- 每 epoch 在 validation 比较 BPSE 与 NLL 的固定 proposal 排序。

如果此阶段 proposal set 固定而 BPSE R@1 没有改善，说明 C/P 定义或质量头无效，不应急于联合训练。

### 10.3 Phase C：BPSE soft routing

验证 quality head 后：

- 解冻 proposal generator；
- 可继续冻结大部分 Transformer 1–2 epoch；
- 开启 `bpse_routing_weights`，替代 rec/ivc hard-min；
- 开启 `L_cov=0.1`；
- 质量头和 proposal head 使用同一 optimizer，但可通过 param group 给 quality head `1x lr`、backbone `0.1x lr`；当前 optimizer wrapper 不支持 param-group 配置时，首版统一 lr，不要为此大改优化器；
- 监控宽度、P/C、routing entropy 和有效 pair 数。

### 10.4 Phase D：可选联合微调

只有 Phase C 稳定后才：

- 解冻最后一层 DualTransformer 或 identity projection；
- 可启用 `bpse_equiv_weight=0.05`；
- 仍默认 detach pair target 和 routing weight；
- 降低总学习率；
- 以 validation checkpoint 选择。

### 10.5 不使用 GT 的原则

- train forward/loss 不读取 `raw.timestamps`；
- GT 仅在 validation/test 指标和离线诊断中出现；
- 不用 GT IoU 训练 quality head；
- 不用 test 指标调超参数；
- Charades 正式实验需处理当前 val=test 的协议问题。

## 11. 推理、评估与实验判据

### 11.1 主对比

保持同一 proposal generator/checkpoint 或同一初始化，至少比较：

1. raw NLL；
2. 当前 semantic vote；
3. direct harmonic score `H(C,P)`，无质量头；
4. only completeness；
5. only purity；
6. BPSE quality head；
7. BPSE + soft routing；
8. BPSE + soft routing + optional equivalence boundary loss。

### 11.2 主要指标

- R@1 mIoU；
- R@1 IoU@0.3/0.5/0.7；
- R@5 mIoU 与 IoU@0.5；
- oracle candidate mIoU；
- BPSE 与 NLL 的 top-1 差异；
- 平均预测宽度和宽度分布；
- short-event R@1/R@5；
- q 对 candidate IoU 的 pairwise ordering accuracy，仅评估使用；
- completeness/purity 的均值、方差和相关性；
- 有效 pair 比率、routing entropy。

### 11.3 支持假设的最低结果

最小验证中，proposal set 保持不变，只训练 quality head。以下结果支持排序假设：

- ActivityNet R@1 IoU@0.5 提升至少 2 个百分点；
- R@5 基本不变；
- oracle 不变，说明提升来自排序；
- 平均 top-1 宽度接近 GT 分布，而非所有区间机械缩短；
- only-C 和 only-P 均弱于完整 BPSE；
- q 对 candidate IoU 的 ordering accuracy 高于 NLL。

以下结果意味着应停止或重定义 C/P：

- H 组内标准差长期接近 0，几乎没有有效 pair；
- BPSE 只通过统一缩短区间提高部分阈值，却让 R@1 mIoU 或低阈值 recall 明显下降；
- pure quality-head 阶段 R@1 不升，联合训练后仅靠 proposal set 改变出现不稳定增益；
- purity 与 proposal 宽度几乎完全负相关，未体现语义解释性；
- 三个随机种子方向不一致。

## 12. Checkpoint、兼容性和保存协议

### 12.1 三种加载语义

1. **`--resume`**：只接受结构相同的 BPSE checkpoint，默认 strict load；
2. **`--init-from-cpl`**：V4 baseline -> BPSE，加载所有同名同形状参数，保留 LREV buffer，缺少的 `bpse.*` 合法；重置 optimization step；
3. **`--init-from-v3`**：保留当前特殊迁移语义，仍可跳过 V4 mixture/LREV，不与 BPSE warm start 混用。

### 12.2 BPSE 初始化

- evidence projection：identity；
- LayerNorm：默认初始化；
- quality head 第一层：标准 Xavier；
- quality head 最后一层：全 0；
- 无跨 batch memory 或 EMA buffer。

### 12.3 建议扩展保存内容

推荐新 checkpoint：

```python
{
    "epoch": epoch,
    "num_updates": self.num_updates,
    "config": self.args,
    "model_parameters": self.model.state_dict(),
    "optimizer_state": self.optimizer.state_dict(),
    "rng_state": {...},  # 可选
}
```

加载旧 checkpoint 时这些字段可缺省；加载新 checkpoint strict resume 时恢复 optimizer。scheduler 当前主要由 `num_updates` 驱动，可继续 `step_update(num_updates)`。

### 12.4 必须保持的 backward compatibility

- BPSE disabled 时没有 `bpse.*` 参数和额外 loss；
- `rec_loss/ivc_loss(proposal_weights=None)` 与原实现等价；
- `nll/geometric_vote/semantic_vote` 继续可选；
- 旧 config 无 `query_units/bpse` 字段仍可运行；
- baseline checkpoint 可通过 baseline config strict eval；
- BPSE checkpoint 必须保存 query-unit 和 BPSE 配置；
- 不改变现有 center/width 的归一化定义。

## 13. 测试与验证矩阵

### 13.1 Query unit 单测

文件：`owlcap/tests/test_query_units.py`。

覆盖：

- 一个动作句能生成 action 和 entity-detail unit；
- mask 与 vocab 过滤后的 token 对齐；
- 截断后空 unit 被移除；
- unit 去重；
- unit 超过 `Umax` 的优先级；
- 无 content token 时 whole-query fallback；
- padding mask/weight 正确，权重和为 1。

### 13.2 Soft-box 和 event tokenizer 单测

文件：`owlcap/tests/test_bpse_module.py`。

覆盖：

- `spans [2,5,2] -> masks [2,5,200]`；
- membership 位于 `[0,1]`；
- span 变宽时 mask mass 单调不减；
- padding frame 权重严格为 0；
- `event_tokens [B,16,D]`、pool weights `[B,16,T]`；
- 每个有效 event 的 frame weight 和约为 1；
- 全静态视频仍 finite；
- 极短有效长度、首尾边界和全 padding 异常输入有明确行为。

### 13.3 C/P 因果 sanity

人工构造 evidence：

- 候选覆盖全部 query unit 时 `C` 高；去掉一个 unit 支持区时 `C` 降；
- 在候选内部加入无法解释的 event 时 `P` 降；
- 仅向外扩到无关事件时 `C` 不明显降、`P` 降；
- 仅向内缩掉必要证据时 `P` 可保持、`C` 降；
- harmonic `H` 对两类错误都下降；
- shell 中仍有 query 证据时 contrast 下降。

这是 BPSE 最关键的机制单测，不能只检查 shape。

### 13.4 Perturbation 和 pair loss

文件：`owlcap/tests/test_bpse_loss.py`。

覆盖：

- 单 offset 得到 `R=9`；
- clamp 后重复候选不形成 pair；
- 宽度小于阈值 invalid；
- 高分候选 logit 高于低分候选时 rank loss 更小；
- 无有效 pair 返回 finite graph-connected zero；
- pair target、routing weights 均 `requires_grad=false`；
- BCE target 在 `[0,1]`；
- total loss backward 后 quality head 有 finite gradient。

### 13.5 Forward/shape/backward 集成

文件：`owlcap/tests/test_bpse_integration.py`。

参考现有 `test_v4_integration.py` 的小模型和 `.cuda()` patch，检查：

```text
words_logit              [B*N,W,V]
center/width             [B*N]
bpse_quality_logits      [B,N]
bpse_completeness        [B,N]
bpse_purity              [B,N]
bpse_group_logits        [B,N,9]，train only
bpse_group_valid_mask    [B,N,9]
bpse_routing_weights     [B,N]
```

并验证：

- forward 所有输出 finite；
- `rec + ivc + event + mixture + bpse` finite；
- backward 正常；
- quality head 有 gradient；
- routing active 后 proposal center/width head 有 gradient；
- evidence detach 策略符合配置；
- eval 时 group 字段为 None，显存路径缩短。

### 13.6 Selection 测试

文件：`owlcap/tests/test_bpse_selection.py`。

- `quality_logits=[0,2,1]` 时选择第二个候选；
- 排序后 center、width、C/P/H 使用同一 index；
- selector 为 BPSE 但输出不存在时明确报错；
- NLL/semantic vote 的旧测试继续通过；
- ActivityNet N=5 和 Charades N=8 都工作。

### 13.7 Smoke、overfit 和 inference

1. 单 batch forward/backward 10 step，loss finite；
2. 固定 8–16 条样本训练，quality rank loss 和 pair error 能下降；
3. 小 batch overfit 时 q 能拟合停止梯度的 H 排序；
4. 运行一个 validation batch 的 BPSE/NLL 双路评估；
5. 完整 eval 两次，deterministic mask 下结果一致；
6. BPSE disabled 的 baseline config 完整训练 1 epoch；
7. `--init-from-cpl` 后缺失 key 只允许 `bpse.*`；
8. BPSE checkpoint strict resume 正常。

推荐命令形式：

```bash
cd /data/chenyuan/videogrounding/w1m2/owlcap
python -m pytest tests/test_query_units.py tests/test_bpse_module.py -q
python -m pytest tests/test_bpse_loss.py tests/test_bpse_selection.py -q
python -m pytest tests/test_bpse_integration.py -q
python train.py --config-path config/activitynet/baseline.json --eval --resume <baseline_ckpt>
python train.py --config-path config/activitynet/bpse.json --selection-strategy bpse --init-from-cpl <baseline_ckpt>
```

实际运行前使用项目的 `cpl` conda 环境；当前代码依赖 `fairseq/nltk/h5py` 和 CUDA。

## 14. 代码修改清单

### 14.1 必须修改

```text
owlcap/datasets/base.py
- 新增规则 query-unit builder
- 在 vocab 过滤后生成 unit mask/type/weight
- collate unit tensor 并正确截断/padding

owlcap/models/modules/bpse.py
- 新增 QueryUnitPooler
- 新增 SalientEventTokenizer
- 新增 soft-box、perturbation group
- 新增 completeness/purity/shell/equivalence/quality head

owlcap/models/modules/__init__.py
- export BPSEScorer

owlcap/models/cpl.py
- 读取 bpse config
- 保存 200-step visual/grounded/query states
- 在最终 center/width 后调用 BPSE
- 返回 BPSE tensors 和 detached routing weights
- 增加兼容的 eval word-mask mode

owlcap/models/loss.py
- 新增 bpse_loss
- rec_loss 支持 proposal_weights
- ivc_loss 支持 proposal_weights
- disabled 路径保持原行为

owlcap/runners/main_runner.py
- 训练 loop 接入 bpse_loss 和 routing schedule
- eval 接入 bpse selector
- 增加 BPSE/NLL/oracle 诊断
- 增加 V4 baseline -> BPSE 兼容加载

owlcap/train.py
- selection choices 加 bpse
- 新增 --init-from-cpl
- 接入常用 BPSE loss override

owlcap/config/activitynet/baseline.json
- 锁定实际 outer baseline，BPSE disabled

owlcap/config/activitynet/bpse.json
- 开启 query units/BPSE，并写全默认参数

owlcap/tests/test_query_units.py
owlcap/tests/test_bpse_module.py
owlcap/tests/test_bpse_loss.py
owlcap/tests/test_bpse_selection.py
owlcap/tests/test_bpse_integration.py
- 新增上述测试
```

### 14.2 推荐修改

```text
owlcap/datasets/activitynet.py
owlcap/datasets/charades_sta.py
- collate 接收 query_unit_config

owlcap/runners/main_runner.py
- checkpoint 保存 epoch/optimizer state
- validation 同时报告 BPSE 与 NLL

owlcap/config/charades/baseline.json
owlcap/config/charades/bpse.json
- 验证 N=8 的通用性；正式实验前处理 val split

owlcap/README.md
- 写清独立工程、环境、baseline parity、warm start、train/eval 命令

owlcap/scripts/*.sh
- 固化 ActivityNet BPSE 训练和评估命令
```

### 14.3 可选修改

```text
owlcap/models/modules/bpse.py
- 可学习 evidence projection
- listwise group loss
- direct H selector
- multi-offset perturbation

owlcap/models/loss.py
- L_equiv
- optional cross-video MIL calibration

owlcap/runners/main_runner.py
- bpse_nll_hybrid / bpse_vote ablation
- 输出 candidate-level CSV 做相关性分析

owlcap/tools/analyze_bpse_candidates.py
- 分析 C/P/q/width/IoU 关系，仅离线评估
```

## 15. Implementation Order

### Step 0：复制并锁定独立 baseline

修改/创建：`owlcap/` 完整工程、`baseline.json`、README。

完成：复制 `cpl_lrev` 的必要源码和数据索引；不复制历史日志；确认本地 import；baseline checkpoint eval parity。

快速验证：旧 tests 通过，BPSE disabled 的指标与源 baseline 一致。

下一步依赖：只有 parity 成立才能判断后续差异来自 BPSE。

### Step 1：实现 query unit 数据链

修改：`datasets/base.py`、dataset wrapper、`test_query_units.py`。

完成：规则 unit、截断、collate、GPU move。

快速验证：打印一个 batch 的 unit mask，与 retained token 手工核对；权重和、valid mask 正确。

下一步依赖：BPSE scorer 的 query unit 输入。

### Step 2：实现纯 BPSE module

修改：`models/modules/bpse.py`、`models/modules/__init__.py`、module tests。

完成：soft-box、event tokenizer、C/P/H/shell、quality head、perturbation groups。

快速验证：随机 tensor shape/finite；人工 evidence 的宽/窄因果 sanity。

下一步依赖：无 CPL forward，先把核心机制单独验证。

### Step 3：只接 forward，不接 loss/selector

修改：`models/cpl.py`、integration test。

完成：保存高分辨率 states，在最终 center/width 后调用 BPSE，返回 tensors。

快速验证：train/eval forward；train 有 `[B,N,9]` group，eval 无 group；原 output 不变。

下一步依赖：loss 和 runner 所需字段稳定。

### Step 4：实现 quality loss

修改：`models/loss.py`、`test_bpse_loss.py`。

完成：pairwise、BCE、coverage；无 pair 安全路径；metrics。

快速验证：合成排序中正确 logit 的 loss 更小，backward finite，target detached。

下一步依赖：训练 loop。

### Step 5：接训练 loop，只训练质量头

修改：`runners/main_runner.py`、`train.py`、`bpse.json`。

完成：第五项 loss、Phase B freeze、日志。

快速验证：8–16 样本 overfit；quality loss 下降；baseline 参数不变。

下一步依赖：先证明 BPSE 能排序固定 proposal set。

### Step 6：接 BPSE eval selector

修改：runner selector、selection tests。

完成：`selection_cost=-q`，同批输出 NLL/BPSE/oracle。

快速验证：人工 q 选择正确候选；一个 validation batch 跑通；R@5 候选集合未被意外改变。

下一步依赖：质量头的最小研究结论。

### Step 7：替代 hard-min routing

修改：`rec_loss/ivc_loss`、runner schedule、集成测试。

完成：optional `proposal_weights`，epoch 3 后 ramp。

快速验证：`None` 与旧 loss 数值相等；soft weight 手算一致；proposal head gradient finite。

下一步依赖：Phase C 联合训练。

### Step 8：checkpoint 与 deterministic eval

修改：train/runner/cpl masking/checkpoint tests。

完成：`--init-from-cpl`、strict BPSE resume、确定性 eval mask。

快速验证：baseline -> BPSE missing keys 只含 `bpse.*`；同一 eval 连跑结果相同。

下一步依赖：可重复正式实验。

### Step 9：smoke、消融和三随机种子

完成：1 epoch smoke、小 batch overfit、固定 proposal 排序实验、联合训练、C/P 消融、至少 3 seeds。

快速验证：日志含所有 BPSE 诊断，checkpoint 由 val 选择，test 只做最终报告。

## 16. 潜在风险与处理方法

### 16.1 Query 原子化错误

风险：规则 POS 无法准确表达复杂主谓宾或多词动作，虚假 unit 会让完整性错误下降。

处理：保留 whole-query 回退；unit 可视化抽检；首版报告规则失败率；冻结 LLM 离线原子化只做后续替代，不在训练时调用。

### 16.2 Query-unit mask 与 token 错位

风险：在 OOV 过滤前构造 mask，或截断 words 后未同步截断 unit。

处理：unit 必须基于 kept tokens；单测覆盖 OOV、截断、padding；forward 检查最后一维等于 query state 的 W。

### 16.3 Query-conditioned frame 造成纯净性虚高

风险：所有 `h_t` 都已看过 query，背景帧也可能显得与 query 相似。

处理：完整性用 grounded state，纯净性 event token 默认用 cross-modal 前的 visual state；对比“二者都用 h”的 ablation。

### 16.4 纯视觉与文本空间未充分校准

风险：`frame_fc` 和 text state 的 cosine 未必天然可靠。

处理：identity 初始化；先检查 evidence heatmap；可选只解冻投影，并加入同视频配对的 MIL 校准。若做 batch negatives，必须用 video id mask 掉同一视频的不同 query，且明确属于可选增强。

### 16.5 完整性天然偏向宽候选

风险：区间越宽越容易覆盖所有 query unit。

处理：必须联合 purity；主融合用 harmonic mean；监控 C/P 与 width 相关性；不能单独用 C 作为最终排序。

### 16.6 纯净性退化为长度惩罚

风险：如果 event-query matching 普遍偏低，越宽平均越低，P 只是隐式惩罚宽度。

处理：报告固定 width 下 P 与 IoU 的关系；比较显式 width-only ranker；event saliency 加静态 floor；检查无法解释事件是否真的位于错误区间。

### 16.7 扰动伪标签来自模型自身

风险：H 自身有偏差，质量头只是复制错误排序。

处理：target detach、高 margin pair、过滤低 mass/重复 span；固定 proposal 阶段先验证；报告 H 与真实 IoU 的 eval 相关性但不用于训练。

### 16.8 质量头通过改写输入逃避 BCE

风险：若 q 输入 C/P 保留 gradient，而 target 是同一 C/P 的 detach，模型可能联合改变 evidence 和 q，产生移动目标。

处理：默认 `quality_detach_inputs=true`；identity evidence projection 在 Phase B 冻结；解冻必须单独 ablate。

### 16.9 Span loss 使所有 similarity 升高

风险：`1-C` 反向进入 evidence 网络可能使全部帧都与全部 unit 相似。

处理：`span_loss_detach_evidence=true`，仅让 soft-box 对 center/width 传梯度；需要训练 evidence 时使用独立校准目标。

### 16.10 梯度被意外全部 detach

风险：target、quality input、routing 都 detach 后，除 quality head 外 proposal 无新梯度。

处理：分别断言：quality head 从 `L_rank/L_abs` 得梯度；proposal head 从 BPSE weighted reconstruction 和 `L_cov/L_equiv` 得梯度；target 和 routing 不得得梯度。

### 16.11 FP16/softmax 数值不稳定

风险：`log(soft_box)`、小温度 softmax、空 shell 产生 NaN。

处理：聚合转 FP32；减最大值；`clamp_min(1e-6)`；空集合显式 valid mask；单测全静态/空 shell/极窄 span。

### 16.12 显存增加

风险：训练组为 `N*9`，若显式保存 `[B,G,T,U]` 会显著增加显存。

处理：共享计算 `[B,T,U]` evidence；按 candidate chunk 计算；不把大中间量放 output；推理不构造 perturbation；默认 ActivityNet `5*9=45` 个轻量候选。

### 16.13 Mixture mask 与最终 span 不一致

风险：reconstruction 使用 multi-peak Gaussian mask，BPSE 对 outer/weighted 连续区间评分；两者仍是不同对象。

处理：这是 BPSE 要暴露和惩罚的问题之一。主实验使用当前真实 `outer`；记录 context width 与 final width；可选比较用 mixture mask 计算 C 和用连续 span 计算 C，但最终排序必须评价实际输出区间。

### 16.14 Eval 随机 masking

风险：候选 importance 和边界随 `np.random.choice` 改变，掩盖排序器真实差异。

处理：OWLCAP 主实验固定 `deterministic_topk`；同时跑 legacy mask 作为兼容检查；日志记录 mask mode。

### 16.15 Checkpoint key 不兼容

风险：baseline checkpoint strict load 到 BPSE 模型会缺新 key；误用 `_load_pretrained_model()` 又会重置 LREV。

处理：新增专用 `_load_cpl_baseline()`；只允许缺 `bpse.*`；shape mismatch 和其他 missing/unexpected key 一律报错。

### 16.16 Distributed training

当前仓库没有 `DataParallel/DistributedDataParallel/torch.distributed`。首版 BPSE 没有跨 batch gather，因此单 GPU/单进程行为明确。若以后加入 MIL batch negatives：

- gather 必须保持或明确切断 gradient；
- 同 video query 不能互为负样本；
- pair loss 的分母需要 all-reduce；
- 不应在首版假装已经支持 DDP。

### 16.17 Charades validation 泄漏

当前 `config/charades/main.json` 的 `val_data` 与 `test_data` 相同。处理：ActivityNet 先做主实验；Charades 结果必须标注 legacy protocol，或先构造独立 val split。

## 17. 当前代码中不能静默假设的事项

### 17.1 `outer` 还是 `weighted`

实际 ActivityNet config 和 checkpoint 是 `outer`，README 描述 `weighted`。推荐锁定实际 checkpoint 的 `outer` 做首轮 BPSE。`weighted` 作为独立 boundary-mode ablation。

### 17.2 用哪个视觉状态计算 purity

原 proposal 只写 `h`。推荐工程实现：coverage 用 query-conditioned `h`，purity 用 cross-modal 前 projected frame。理由是 purity 应检查额外事件，而不是让 query 已经写入所有 frame。备选方案为二者都用 `h`，必须单独消融。

### 17.3 是否让 BPSE 直接更新 proposal

原 proposal 主要强调质量头和排序，又包含 `lambda_cov`。推荐先固定 proposal 验证排序，再开启 soft routing 和小权重 span loss。若一开始全联合，无法区分排序增益与候选生成变化。

### 17.4 是否使用质量头还是直接 H

主方案使用 learned q；direct harmonic H 是必要对照。如果 H 已稳定优于 q，说明质量 MLP 没必要，应简化模型；如果 q 优于 H，检查 inside-shell/width 是否提供了真实补充信息。

### 17.5 是否把 perturbation 加入最终候选集合

推荐不加入。扰动只产生局部相对监督。若推理也输出扰动，会改变 proposal set 和 R@5，无法单独检验排序假设。

### 17.6 HMD-270K 是否需要迁移

不需要。当前研究问题是 CPL 的候选排序与边界质量，首轮只迁移 CSER 的双向集合机制。引入 HMD 或额外 caption 会改变监督来源和实验问题。

### 17.7 当前环境的 device 风格

基线大量使用 `.cuda()` 和 ByteTensor mask。BPSE 新代码必须使用输入 `device/dtype`，但不要在同一变更中重构整个基线 device 系统。CPU integration test 可继续参考现有 monkeypatch。

## 18. 验收标准

实现完成必须同时满足：

1. `owlcap/` 是完整可运行工程，或在实施者明确得到用户许可后按一一对应路径落入指定目录；
2. `cpl_lrev/` 未被修改；
3. BPSE disabled 的 baseline tests 和 eval 可运行；
4. query unit 与 retained token 完全对齐；
5. C/P/H 的人工因果 sanity 全通过；
6. train forward 有 perturbation group，eval forward 没有；
7. loss finite、backward 正常、关键梯度方向符合 detach 设计；
8. `rec_loss/ivc_loss(None)` 与原数值一致；
9. BPSE selector 在质量越大越优的方向上没有符号错误；
10. baseline -> BPSE warm start 只缺 `bpse.*`；
11. 同一 deterministic eval 连跑结果一致；
12. validation 同时输出 NLL、BPSE 和 oracle，能区分生成与排序变化；
13. 最小固定 proposal 实验至少完成 only-C、only-P、direct-H、learned-BPSE；
14. 正式结果用 validation 选 checkpoint，test 不参与调参；
15. 日志记录实际 boundary mode、mask mode、routing schedule 和所有 BPSE loss 权重。

## 19. 给后续 Codex 的执行说明

负责实现的 Codex 应按第 15 节顺序执行，先复制并验证 baseline parity，再做 query-unit 数据链和独立 `BPSEScorer`，不要一开始就修改训练目标。核心文件是：

```text
owlcap/datasets/base.py
owlcap/models/modules/bpse.py
owlcap/models/cpl.py
owlcap/models/loss.py
owlcap/runners/main_runner.py
owlcap/train.py
owlcap/config/activitynet/bpse.json
```

必须保持：

- `bpse.enabled=false` 的旧模型行为；
- 原 `nll/geometric_vote/semantic_vote`；
- 原 center/width 和 Gaussian mixture 输出语义；
- 旧 config 可读取；
- V4/LREV baseline 权重可完整 warm start；
- 训练阶段不读取 GT timestamp；
- 推理不依赖 perturbation 或训练期伪标签。

实现完成后至少运行：

```text
query-unit unit tests
BPSE module causal sanity tests
BPSE loss/detach tests
full forward + all losses + backward
baseline selector regression tests
BPSE selector tests
small-batch overfit
baseline disabled eval
BPSE enabled eval
baseline -> BPSE load
BPSE strict resume
deterministic eval repeatability
```

没有必要时不要重构 Transformer、attention、Gaussian mixture 数学、optimizer 或整个 device 管理。不要把 OwlCap 的 Qwen judge、GRPO 或 HMD 数据流程搬入该工程。首个研究问题应保持单一：在原候选集合上，双向完整性—纯净性质量估计能否比重构 NLL 更可靠地选出高 IoU 区间；只有这个假设得到验证后，才进入 soft routing 和联合边界优化。
