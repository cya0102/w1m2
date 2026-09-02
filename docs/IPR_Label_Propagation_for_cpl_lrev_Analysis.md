# 事件边界与 IPR Label Propagation 在 `cpl_lrev` 中的适用性分析

## 0. 结论先行

**结论：可以把 IPR 的 Label Propagation（LP）框架接到 `cpl_lrev` 上，且 IPR 论文已经证明该模块在结构上可以插入原始 CPL；但不能原样照搬来解决当前的“过长提议”问题。原始 LP 的因果方向与当前问题相反，直接移植很可能进一步强化长区间。**

更准确地说：

1. IPR 的 LP 不是坐标优化算法。它在固定的候选区间之间传播置信度伪标签，只改变候选排序，不改变 start/end。
2. IPR 假设种子提议是“高精度但只覆盖局部”的短区间，再把正标签传播给高 IoU 邻居，从而扩大覆盖范围。论文种子的平均归一化长度只有 `0.09`，GT 平均长度约 `0.25`。
3. `cpl_lrev` 当前恰好相反：ActivityNet 的多个候选宽度大量位于 `0.59~0.71`，训练后期甚至达到 `0.63~0.79`；原样 LP 会在这些高度重叠的长提议簇中继续传播正标签。
4. 当前事件边界非常适合做**候选端点字典和收缩约束**，但不适合把一个原子事件直接当作最终时刻。全量统计表明，最近事件边界对 GT 端点有较高的 oracle 覆盖能力；然而单个原子事件通常远短于查询对应的时刻。
5. 推荐方案是“**事件边界约束的、带坐标收缩的 LP**”（下文简称 EB-LP）：
   - 先用事件边界生成原提议的 inward-trim / snap 变体；
   - 用查询重建损失判断删掉两侧事件后是否仍能解释文本，即“最小充分区间”；
   - 在事件兼容图中传播正、负、忽略三态伪标签；
   - 对已确认的过长包含关系使用带负项的置信度损失；
   - 增加坐标/边界监督，否则 LP 仍只会重排固定候选。

最稳妥的研发顺序是：**先做无需训练的事件边界收缩与重评分，验证边界确实能改善短时刻；再加训练期边界正则；最后才上多阶段 EB-LP。**

---

## 1. 分析范围与证据来源

本分析直接检查了以下材料：

- IPR 论文：`CVPR23-Iterative_Proposal_Refinement_for_Weakly-Supervised_Video_Grounding.pdf`
- `cpl_lrev` 模型、损失、数据加载、推理选择器、配置、测试和 ActivityNet 训练日志
- 两份事件边界全量输出：
  - `CPL_baseline-main/eventBoundary/outputs/activitynet_c3d_event_boundaries.json`
  - `CPL_baseline-main/eventBoundary/outputs/charades_i3d_event_boundaries.json`
- `cpl_lrev` 的 ActivityNet/Charades 全部标注，用于统计事件边界和 GT 的几何关系

这里的“可行”分为三个层次：

- **接口可行性**：现有张量和候选是否足以接入 LP；
- **因果可行性**：LP 的作用方向是否能缓解过长提议；
- **数据可行性**：事件边界是否真的覆盖可能的 GT 端点，而非只有形式上可用。

只有同时满足后两项，模块才值得进入完整训练。

---

## 2. 从第一性原理重建 IPR 的 Label Propagation

### 2.1 IPR 每一阶段真正预测什么

设一个视频有固定的 `N` 个提议：

$$
u_n=[s_n,e_n],\quad n=1,\ldots,N,
$$

对应提议特征为 $p_n$。第 $k$ 个 refinement stage 通过三个 MLP 输出：

$$
e_n^k=\sigma(p_nW_e^k),
$$

$$
s_n^k=\sigma(p_nW_s^k)\cdot e_n^k,
$$

$$
c_n^k=\sigma(p_nW_c^k)\cdot e_n^k.
$$

其中：

- $e_n^k$：proposal confidence；
- $s_n^k$：semantic score；
- $c_n^k$：对 `M` 个高频概念词的 conceptual score；
- semantic target 来自冻结的视频-文本预训练模型对裁剪视频和查询的相似度；
- concept target 是查询是否含有高频名词、动词、形容词的 multi-hot 向量。

论文分别使用 semantic L1 和 conceptual BCE/交叉熵形式计算蒸馏误差。第 $k$ 阶段的种子为：

$$
i^k=\arg\min_n
\left(\mathcal L_{sem}^{k,n}+\mathcal L_{cpt}^{k,n}\right).
$$

也就是说，**种子不是当前 confidence 最大的提议，而是最符合外部语义和查询概念先验的提议。**

### 2.2 LP 实际传播的标签

以种子 $u_{i^k}$ 为中心，论文的 Algorithm 1 定义：

$$
\hat e_n^{k+1}=\begin{cases}
1,&\operatorname{IoU}(u_n,u_{i^k})>\beta,\\
0,&\text{otherwise}.
\end{cases}
$$

并强制种子自身为正。论文设置：

- `N = 8`；
- `K = 4`；
- `beta = 0.6`；
- concept vocabulary `M = 30`。

下一阶段使用 confidence rectification：

$$
\mathcal L_{con}^{k,n}
=-\hat e_n^k\log e_n^k.
$$

一个容易忽略但对本项目非常重要的细节是：论文 Eq. (5) 虽在文字中称为 BCE，但公式只包含正项，没有
$-(1-\hat e)\log(1-e)$。因此 Algorithm 1 中标记为 `0` 的非邻居，在这个损失里实际上没有被压低，只是“不奖励”。工作区已有的 IRON 实现也忠实采用了这一正项形式：`/data/chenyuan/videogrounding/iron-main/models/iron.py:271-320`。

### 2.3 它为什么能解决 IPR 的问题

IPR 的出发点是 weak supervision 容易只找到最有辨识度的局部动作。论文 Sec. 4.5 的实证是：

- 最小蒸馏损失种子的 coverage ratio 约为 `0.93`：几乎都落在 GT 内；
- 该种子平均归一化长度只有 `0.09`；
- GT 平均长度约为 `0.25`。

因此它的逻辑是：

> 从一个“短但纯”的种子出发，把正置信度传播给与之高度重叠的邻居，使最终排序逐渐偏向覆盖更完整的区间。

这是一个**扩张/补全机制**，而不是边界收缩机制。

### 2.4 它是否在 CPL 上有先例

有。论文 Table 2(b) 明确把 refinement 插入 CPL，在 Charades-STA 上：

- CPL：`R1@0.3/0.5/0.7 = 66.40/49.24/22.39`
- CPL + refinement：`70.14/50.55/24.61`

所以“能否在 CPL 类模型上运行”的答案是肯定的。但这只证明模块接口和一般排序能力，不证明它能治疗方向相反的长区间偏置。

论文 Table 4 还给出重要警告：用最大 confidence 作为传播源时，`R1@0.3 = 67.12`，比 semantic+concept 联合种子的 `70.71` 低 `3.59` 个点。因而不能简单把 `cpl_lrev` 的 NLL top-1 或 semantic-vote top-1 当成 IPR 种子并期望同样收益。

---

## 3. `cpl_lrev` 当前提议链路审计

### 3.1 输入和时间分辨率

两个数据集都在配置中使用 `max_num_frames = 200`：

- ActivityNet：C3D 500D，`cpl_lrev/config/activitynet/main.json:3-14`
- Charades：I3D 1024D，`cpl_lrev/config/charades/main.json:3-14`

`BaseDataset._sample_frame_features` 将每个源视频统一池化/重复采样成 200 个 clip，见 `cpl_lrev/datasets/base.py:24-37`。

但模型生成 proposal mask 前又执行：

```python
props_len = n_frames // 4
keep_idx = torch.linspace(0, n_frames - 1, steps=props_len).long()
```

即从 200 降到 50 个位置，见 `cpl_lrev/models/cpl.py:119-125`。因此：

- center/width 虽是连续值；
- 真正参与重建和边界梯度的视觉掩码只有约 `1/50 = 0.02` 的归一化时间分辨率；
- 对一个 300 秒视频，一个 mask bin 已约为 6 秒。

这比“归一化乘回 duration”更接近真实的误差来源。

### 3.2 ActivityNet 与 Charades 不是同一种 proposal generator

ActivityNet 使用 Gaussian Mixture：5 个提议分别包含 `1,2,3,4,5` 个分量，见 `cpl_lrev/models/modules/gaussian_mixture.py:44-48` 和 ActivityNet 配置 `:38-45`。

Charades 配置没有 `proposal_generator` 字段，代码默认回退到 `single_gaussian`，生成 8 个独立的 `(center,width)`，见 `cpl_lrev/models/cpl.py:31-40,72-74`。

所以：

- Charades 与原始 CPL/IPR 的候选形式更接近；
- ActivityNet 必须额外处理 mixture component 与最终报告区间不一致的问题。

### 3.3 ActivityNet 的最外包络会天然产生长区间

每个 mixture proposal 的所有分量先预测独立中心和共享宽度。当前 `boundary_mode = outer`，最终区间为：

$$
s_n=\min_m(c_{nm}-w_{nm}/2),\quad
e_n=\max_m(c_{nm}+w_{nm}/2).
$$

对应代码为 `cpl_lrev/models/modules/gaussian_mixture.py:131-162`，当前配置为 `cpl_lrev/config/activitynet/main.json:38-45`。

因此，只要一个低重要度分量离主体较远，最终 start/end 就会覆盖中间全部空白。即使 reconstruction mask 是多个局部峰的加权和，**评测区间仍是所有分量的 convex hull**。这是一条直接产生过长提议的结构路径。

还有一处配置漂移需要先冻结：`cpl_lrev/README.md:11-20,111-121` 声称最终设置是 `boundary_mode = weighted`，但当前 `main.json`、现有 `cpl_lverFinal` checkpoint 内保存的配置都是 `outer`；README 提到的 `final_weighted_diagnostic.json` 当前也不存在。后续消融必须以 checkpoint 内配置为准，不能只按 README 命名判断。

### 3.4 当前目标没有直接惩罚“中等程度的过长”

训练的主要弱监督信号是：

- 每个查询选择 reconstruction NLL 最小的 proposal，`cpl_lrev/models/loss.py:24-35`；
- 与全文、左右负区间做 margin ranking，`loss.py:54-97`；
- Event 分支用 reconstruction softmax 作为 proposal 权重，`cpl_lrev/models/cpl.py:232-257`；
- Event context loss 要求 proposal 外至少留出 `0.15` 的上下文，`loss.py:270-288`；
- proposal pair IoU 超过 `0.7` 才产生 overlap penalty，`loss.py:290-321`。

若 proposal 完全位于 `[0,1]` 内，左右可用上下文之和近似为 `1-width`。因此 `event_min_context=0.15` 主要在宽度超过约 `0.85` 时才强烈触发；宽度 `0.5~0.8` 的长提议基本不受该项约束。

ActivityNet 的 `alpha_2 = 0`，即原 CPL 的 Gaussian diversity 项被关闭，见 `config/activitynet/main.json:68-86`。mixture 自身仍有 inter-push，但它约束 mask 相似性，不直接保证最终 outer hull 短。

### 3.5 推理投票会让“长提议簇”成为中心节点

推理先按 reconstruction NLL 排序，再用 proposal 两两 IoU 做 geometric/semantic vote，见 `cpl_lrev/runners/main_runner.py:233-283,537-580`。

semantic vote 的本质是：

$$
vote_i=\sum_j \operatorname{IoU}(u_i,u_j)\cdot
\operatorname{softmax}(-L_{cap,j}/T).
$$

长区间往往与更多候选相交；当多个长区间高度重叠时，它们形成高连接度簇，投票会偏向簇中心。这个机制和原始 IPR 的 IoU 邻域传播十分接近，所以原样加入 LP 很可能只是把现有偏置重复一遍。

另外，`inference_vote_event_weight` 虽在配置和 runner 初始化中出现，但没有进入 `select_proposal_by_strategy`；实际 vote support 只有 proposal NLL。ActivityNet 的 `inference_event_weight` 也设为 `0.0`。因此当前 ActivityNet 推理并没有利用 Event score 修正长区间投票。

### 3.6 日志已经证明“长候选 + 排序缺口”同时存在

ActivityNet 的 validation-selected composite checkpoint 在 test 上的候选平均宽度为：

| proposal | P1 | P2 | P3 | P4 | P5 |
|---|---:|---:|---:|---:|---:|
| normalized width | 0.2669 | 0.3871 | 0.5904 | 0.5928 | 0.7081 |

同一 checkpoint：

- NLL top-1 mIoU：`0.3698`
- 5 个候选 oracle mIoU：`0.5522`
- short GT 的 R5 mIoU：`0.2770`
- long GT 的 R5 mIoU：`0.7491`

见 `cpl_lrev/logs/activitynet/cpl_lverFinal_2026-07-15_14-58-19.log:905-908`。

训练第 30 轮附近的 validation 候选宽度进一步达到约 `0.478/0.632/0.667/0.777/0.788`，pairwise IoU 均值约 `0.664`，80% 的 pairwise IoU 大于 0.5。说明问题不只是 top-1 选错，也包括 generator 的长区间漂移和候选簇同质化。

---

## 4. 事件边界数据能提供什么

### 4.1 与 `cpl_lrev` 输入严格对齐的优点

两份输出使用的正是当前模型的特征：

- ActivityNet C3D；
- Charades I3D；
- 都复用 CPL 的 200-clip 采样。

输出 metadata 明确记录 `resample_num_clips = 200`。因此无需重新建立视频帧到特征的索引关系，`boundary_positions_normalized` 可直接落在模型 0~1 时间轴上。

### 4.2 全量边界与标注统计

以下统计读取了两个完整 boundary JSON 和 `cpl_lrev` 的全部 query 标注，而非抽样：

| 指标 | ActivityNet | Charades |
|---|---:|---:|
| 视频数 | 14,926 | 6,067 |
| query 数 | 71,957 | 13,760 |
| 源特征 clip 数中位数 | 389 | 92 |
| 每视频 raw boundary 数中位数 | 28 | 18 |
| 每视频 raw boundary 数均值 | 29.40 | 21.02 |
| 原子事件宽度中位数 | 0.025 | 0.030 |
| 原子事件宽度均值 | 0.0352 | 0.0499 |
| 宽度不超过 0.02 的事件比例 | 48.6% | 40.4% |
| GT 宽度中位数 | 0.245 | 0.252 |
| GT 宽度均值 | 0.319 | 0.270 |
| 最近边界的单端点距离中位数 | 0.00481 | 0.01182 |
| GT 与最近边界对的 oracle 平均 IoU | 0.874 | 0.827 |
| 上述 oracle IoU ≥ 0.5 | 94.5% | 95.1% |
| 最佳单个原子事件平均 IoU | 0.371 | 0.380 |
| 最佳单个原子事件 IoU ≥ 0.5 | 24.8% | 22.0% |

这组数字支持两个不同结论：

1. **边界覆盖结论**：事件边界足够密，绝大多数 GT 可以由一对边界较好地近似，所以适合做端点候选集合。
2. **事件粒度结论**：查询时刻通常跨越多个原子事件，不能用单个相邻边界段替代 proposal。

oracle 数字不能被当作实际模型性能。因为它使用 GT 选择边界，而且边界本身较密；它证明的是 proposal search space 有覆盖能力，不证明 query-conditioned selector 能找到正确的一对。

### 4.3 输出必须先去除 plateau/过密峰

当前局部最大策略会保留相等峰值，且特征统一重采样会制造重复 clip。全量结果中：

- ActivityNet 相邻、等分数 plateau pair 占内部相邻峰对约 `6.0%`；
- Charades 占约 `14.7%`；
- 1-clip 原子事件分别占约 `5.6%` 和 `13.2%`。

Charades 源特征中位数只有 92，却被上采样到 200，重复平台问题尤其明显。

在进入 LP 前应执行：

1. 将连续且分数相同的峰合并为一个峰；
2. 对内部峰做 minimum-distance NMS；
3. 端点 `0/199` 是强制 delimiter，不能把 zero-padding 导致的端点高分当作置信度；
4. 保留原始 boundary score 作为软权重，不应把所有边界视为等可信。

### 4.4 200-grid 边界与 50-grid proposal mask 的冲突

事件边界在 200-grid 上，而 proposal reconstruction 在 50-grid 上。把边界映射到 50-grid 后，每视频平均仍有约：

- ActivityNet：25.5 个不同位置；
- Charades：18.1 个不同位置。

但 40%~49% 的原子事件宽度不超过一个 50-grid bin。这意味着：

- 边界可以提高最终 start/end 的报告精度；
- 但两个相邻边界生成的 mask 可能在 50-grid 上完全相同，模型无法通过 reconstruction 区分；
- 若要真正改善短时刻，最好把 proposal scoring 分辨率从 50 提高到 100/200，或至少为边界候选使用基于 bin-overlap 的软 mask，而不是最近邻取整。

---

## 5. 为什么“归一化还原”不是唯一根因

设归一化区间为 $u=[s,e]$，视频有效时长为 $D>0$，秒数区间为 $Du=[Ds,De]$。对于同一视频：

$$
\operatorname{IoU}(Du,Dg)
=\frac{D\,|u\cap g|}{D\,|u\cup g|}
=\operatorname{IoU}(u,g).
$$

因此，如果预测和 GT 使用同一个正确 duration，单纯执行 `t_seconds = t_normalized * duration` 不会改变 IoU，也不会凭空把 proposal 变长。

但归一化误差会在秒数上被长视频线性放大：

$$
\Delta t=D\cdot\Delta p.
$$

当前真正敏感的环节包括：

1. 源特征先统一重采样为 200，再降到 50 个 proposal mask 位置；
2. 连续 sigmoid 坐标由很粗的 mask reconstruction 间接监督；
3. ActivityNet outer hull 把分散分量之间的空白也报告为正区间；
4. duration metadata 与特征实际覆盖时长若不一致，会产生系统映射误差；
5. boundary point 使用 `index/199`，而 half-open event cut 使用 `cut/200`，混用会引入 off-by-one 式偏差。

当前 runner 的评测并没有先把预测乘回秒数；它直接执行 `gt = gt / duration` 后在归一化坐标上计算 IoU，见 `cpl_lrev/runners/main_runner.py:219-283`。因此已有低 IoU 主要是在归一化域内就已经产生，而非评测阶段的秒数还原造成。

实际输出秒数时应严格区分：

- proposal 端点对齐：使用 `boundary_positions_normalized` / `boundary_times_seconds`；
- 事件 pooling：使用 `segmentation_cuts_indices`、`event_intervals_indices_half_open` / `event_intervals_seconds`；
- 不要把 `/199` 的 point convention 与 `/200` 的 half-open cut convention 混在同一条链路中。

---

## 6. 直接移植 IPR LP 的兼容性矩阵

| IPR 所需条件 | `cpl_lrev` 状态 | 判断 |
|---|---|---|
| 每视频固定 N 个 proposal | ActivityNet 5 个；Charades 8 个 | 可用 |
| 每 proposal 的特征 `p_n` | 有 proposal mask、decoder hidden、Event vector，但当前未整理成 refinement 输入 | 需暴露/池化 |
| semantic distillation target | 有自监督 query reconstruction NLL，但没有冻结 VLM target | 只能近似，不等价 |
| conceptual target | 当前没有 | 需新增，或承受种子质量下降 |
| K 个 confidence heads | 当前没有 | 需新增 |
| proposal IoU graph | runner 已计算 pairwise IoU | 可复用 |
| confidence 传播能修改 start/end | 不能 | 不满足用户核心目标 |
| 种子“短且高精度” | 当前候选普遍偏长，NLL 也可能偏爱长上下文 | 与 IPR 假设相反 |
| 事件边界与特征时间轴一致 | 都是同 backbone、同 200-clip 采样 | 满足 |

所以最终判断是：

- **结构可接**：是；
- **原样能解决过长**：否；
- **加入事件边界和坐标收缩后可解决**：有较强可行性，值得分阶段验证。

---

## 7. 推荐方案：事件边界约束的 Proposal Refinement

### 7.1 第一性原理目标：寻找“最小充分区间”

过长提议的本质是：区间包含了能解释查询的证据，也包含了对解释查询没有边际贡献的额外事件。

因此不应直接施加“所有 proposal 都要短”的全局先验；那会损害 ActivityNet 中真实的长查询。更合理的目标是：

> 在保持查询解释能力基本不下降的候选中，选择边界对齐且包含冗余事件最少的区间。

记当前 proposal 为 $u$，由事件边界在其内部生成收缩候选集合 $\mathcal C_B(u)$，则可以使用约束形式：

$$
u^*=\arg\min_{v\in\mathcal C_B(u)} |v|
$$

满足：

$$
L_{cap}(v)\le L_{cap}(u)+\epsilon,
$$

以及 Event/text score 不显著下降。

相比直接优化 `L_cap + lambda * width`，这个“语义优先、宽度次优”的形式不需要假设统一的 GT 长度分布。

### 7.2 事件边界预处理

对每个视频构造：

$$
B_v=\{(b_j,a_j)\}_{j=0}^{J-1},
$$

其中 $b_j\in[0,1]$ 是边界位置，$a_j$ 是边界置信度。预处理应：

1. 合并相邻等分 plateau；
2. 用 minimum-distance NMS 控制密度；
3. 分数在视频内标准化；
4. 将强制端点作为几何端点而不是高置信语义峰；
5. 保留 `boundary_indices` 到秒数的明确映射。

训练时不建议让每个 DataLoader worker 直接加载 210 MB/61 MB 的完整 JSON。应预先生成只含 video id、边界位置、分数、event cuts 的紧凑 sidecar，或在主进程缓存为稀疏数组/200 维 dense boundary map。

### 7.3 阶段 A：无需训练的 inward trimming（最高优先级）

对每个原始 proposal $u_n=[s_n,e_n]$：

1. 保留原 proposal；
2. 取左端点右侧最近的 1~2 个高置信边界作为 inward-left；
3. 取右端点左侧最近的 1~2 个高置信边界作为 inward-right；
4. 组合出 left-trim、right-trim、both-trim 候选；
5. 也可保留距离很近的 outward snap，但默认不主动扩张。

然后使用现有 query decoder 对每个新 mask 重新计算 reconstruction NLL。`DualTransformer` 的 `gauss_weight` 实际上可接收一般软时间掩码，所以候选可以使用按事件 atom 聚合的 soft mask，不必伪装成单个 Gaussian。

最简单的选择规则是：

```text
先筛掉 reconstruction NLL 比原 proposal 差超过 epsilon 的候选；
在剩余候选中选最短者；
若都不合格，保留原 proposal。
```

这一步不需要 LP，也不需要重训。它能首先回答最关键的实验问题：**外侧事件是否真的是无用上下文，事件边界能否在不损害文本解释的前提下删掉它们。**

### 7.4 阶段 B：边界感知的排序与弱监督

可定义 proposal 的软边界距离：

$$
d_B(x)=-\tau_B\log\sum_j
\bar a_j\exp\left(-\frac{|x-b_j|}{\tau_B}\right),
$$

其中 $\bar a_j=a_j/\sum_l a_l$，避免边界数量本身改变 soft-min 的尺度。

$$
L_{boundary}(u_n)=d_B(s_n)+d_B(e_n).
$$

但不能单独最小化它，因为密集伪边界会吸引所有端点。该项必须由 query-conditioned 选择权重控制：

$$
q_n=\operatorname{softmax}(-L_{cap,n}/T),
$$

$$
L_{boundary}=\sum_n q_n L_{boundary}(u_n).
$$

进一步，对原 proposal 和其 trimmed candidate 构造弱监督排序：如果 trimmed proposal 的 NLL 不差于原区间，则要求：

$$
score(v_{trim})\ge score(u)+m.
$$

这比固定的 width penalty 更贴合“删除无边际贡献事件”的目标。

### 7.5 阶段 C：Event-Boundary-aware Label Propagation（EB-LP）

节点集合不应只有当前 `N` 个原 proposal，而应包括：

$$
\mathcal U=\mathcal U_{original}\cup\mathcal U_{trim/snap}.
$$

#### 7.5.1 种子选择

最稳健的种子仍遵循 IPR：增加冻结 VLM semantic target 和 concept target。若先做轻量版本，可用：

$$
Q_n=L_{cap,n}
-\lambda_E score_{event,n}
+\lambda_B L_{boundary,n}
+\lambda_X L_{extra,n}.
$$

其中 $L_{extra}$ 衡量删掉左右最外侧事件后，query reconstruction 是否几乎不变。种子为 `argmin Q_n`。

仅用 NLL/confidence 选种子风险较高：它是模型自己的弱监督输出，可能正包含要修复的长区间 shortcut；IPR 的 max-confidence 消融也已经显示该种子策略次优。

#### 7.5.2 事件兼容图，而非纯 IoU 图

令 $E(u)$ 为 proposal 覆盖的 event atom 集合，构造：

$$
A_{nm}=
\mathbf 1[\operatorname{IoU}(u_n,u_m)>\beta]
\cdot\exp(-d_{evt}(E_n,E_m)/\tau_E)
\cdot G_{nm}.
$$

其中：

- $d_{evt}$ 可为 boundary-weighted atom symmetric difference；
- $G_{nm}$ 是方向门控：若 $u_n$ 严格包含种子并多出若干低边际贡献的完整事件，则不允许它仅凭 IoU 成为正邻居；
- 对边界对相同、仅有轻微抖动的 proposal 给予更高 affinity。

这一门控可以避免长 proposal 成为图中的 hub。

#### 7.5.3 三态伪标签

原 IPR 的 0 标签在 Eq. (5) 中并不会受到负向监督。针对已知的长区间问题，建议定义：

- `positive`：与种子高 IoU、事件集合兼容、无明显额外事件；
- `negative`：严格包含种子，且删除额外边缘事件不损害 query reconstruction；
- `ignore`：其余不确定节点。

使用加权完整 BCE：

$$
L_{con}=-\sum_n\left[
w_n^+y_n\log e_n+
w_n^-(1-y_n)\log(1-e_n)\right],
$$

其中只有高置信 negative 才使用第二项，避免把弱监督下可能正确的长查询误杀。

这是为解决当前问题而对 IPR 做的必要方向性修改，而非逐公式复现。

#### 7.5.4 必须增加坐标更新

如果每个 stage 只预测新 confidence，最终 start/end 仍是原 generator 的固定输出。要称为“优化视频提议”，至少应选择下列一种：

1. 将 boundary-trim/snap 变体显式加入候选集合，最终从新坐标中选择；
2. 每阶段预测 $\Delta c_n^k,\Delta w_n^k$，再向软事件边界投影；
3. 用种子的最小充分边界对作为 pseudo coordinate target：

$$
L_{coord}=1-\operatorname{IoU}(u_n,\tilde u_n)
+\lambda_{cw}\operatorname{SmoothL1}((c_n,w_n),(\tilde c_n,\tilde w_n)).
$$

只加 IPR confidence head、不加这一步，最多改善排序，无法修复候选集合整体偏长。

---

## 8. 两个数据集应采用不同的接入策略

### 8.1 ActivityNet

优先级建议：

1. **先修复/消融 outer hull**。比较：
   - outer；
   - importance-weighted endpoints；
   - 按 component importance 的分位数包络；
   - 事件边界收缩后的包络。
2. negative mining 可继续使用完整 outer envelope，但最终报告区间不必等于它。需要明确“reconstruction mask、negative exclusion hull、reported interval”三者的语义。
3. proposal 数只有 5，而 IPR 使用 8；加 boundary variants 后图节点数才足够形成有意义的邻域。
4. ActivityNet 有大量短 query（归一化宽度 `<0.15` 的比例约 34.2%），应重点观察 short/medium-short 分桶。
5. 原子事件很短，绝不能强制最终区间只包含一个 event atom；应在多个连续 atom 之间做 query-conditioned trim。

### 8.2 Charades

优先级建议：

1. 当前是 8 个 single Gaussian，与 IPR/CPL 插件设置更匹配；可先验证 vanilla IPR LP 作为对照。
2. Charades 源 I3D clip 比 200 少，upsampling plateau 较严重；边界 NMS/plateau merge 比 ActivityNet 更重要。
3. GT `<0.15` 的比例只有约 7.9%，不应过度收缩；使用最小充分区间约束比固定短宽度先验更安全。
4. 当前 `val_data` 与 `test_data` 都指向 `data/charades/test.json`。超参数和 checkpoint 若据此选择会泄漏 test；正式实验前应从 train 划独立 validation，最后只测试一次。

---

## 9. 具体代码接入位置

本报告不直接修改模型，但推荐未来实现按以下边界组织：

### 9.1 数据层

`cpl_lrev/datasets/base.py`

- 在 `__getitem__` 根据 `vid` 读取紧凑 boundary record；
- 返回 boundary positions、scores、event cuts；
- collate 时生成：
  - padded sparse boundary list；或
  - `[B,200]` boundary confidence map；
- 秒数转换仍保留 duration，但训练几何统一使用规范化坐标。

### 9.2 proposal 与 refinement

`cpl_lrev/models/cpl.py`

- 在 `gauss_center/gauss_width` 产生后生成 boundary trim/snap variants；
- 提高候选 mask 的时间分辨率，或用 200-to-50 fractional overlap soft mask；
- 暴露每 proposal 的视觉池化特征、Event vector、query feature、NLL；
- 增加 `K` 个 confidence/semantic/concept heads；
- 完整版本增加 coordinate delta/refinement head。

### 9.3 ActivityNet mixture

`cpl_lrev/models/modules/gaussian_mixture.py`

- 将 outer hull、weighted/quantile hull、event-refined reported interval 分开；
- 不让一个低 importance 的远端 component 无条件决定最终边界；
- 保留 outer hull 给 negative mining 时，应在输出字段名中明确其只用于 context。

### 9.4 损失

`cpl_lrev/models/loss.py`

- 增加 boundary softmin loss；
- 增加原区间 vs trimmed 区间的 minimal-sufficiency ranking；
- 增加 EB-LP tri-state confidence loss；
- 增加 coordinate loss；
- 对 pseudo target 全部 `detach`，避免模型通过改变伪标签生成路径走捷径。

### 9.5 推理

`cpl_lrev/runners/main_runner.py`

- 先报告 NLL top-1、原 semantic vote、boundary rerank 三个独立结果；
- event-gated vote 的 affinity 不再只用 IoU；
- 输出 normalized、clip-index、seconds 三套坐标及所用 convention；
- 增加 selected-index、selected-width、endpoint boundary distance 诊断。

---

## 10. 推荐实验路线与停止条件

### Phase 0：冻结并诊断现有 baseline

先固定：

- checkpoint 内到底是 outer 还是 weighted；
- selection strategy 是 NLL 还是 semantic vote；
- proposal mask 的 50-grid 分辨率；
- ActivityNet 和 Charades 各自 generator 类型。

新增诊断：

- predicted width、GT width、`pred_width-GT_width`；
- 按 duration 和 GT width 分桶；
- start/end 到最近 event boundary 的距离；
- selected proposal 覆盖的 event atom 数；
- R1 与 oracle R@N 差距；
- whole-video/near-whole-video 比例；
- 秒级 start/end absolute error。

### Phase 1：无训练 event trimming

对照：

1. 原 NLL top-1；
2. 原 semantic vote；
3. nearest-boundary hard snap；
4. minimal-sufficient inward trim；
5. event-gated semantic vote。

若 hard snap 下降、minimal-sufficient trim 上升，说明问题不是“边界无效”，而是必须让 query semantics 决定边界组合。

### Phase 2：训练期 boundary/minimal-sufficiency loss

只加：

- soft boundary loss；
- trim-vs-original ranking；
- coordinate target。

暂时不要加 K-stage LP，以便判断收益来自边界坐标还是多阶段 confidence。

### Phase 3：EB-LP

消融至少包括：

1. vanilla IPR IoU LP；
2. vanilla LP + event boundary nodes；
3. event-gated positive-only LP；
4. tri-state EB-LP；
5. tri-state EB-LP + coordinate refinement；
6. 是否使用 frozen VLM semantic + concept targets。

建议从 `K=2` 开始，再比较 `K=3/4`；当前目标是防止过扩张，多阶段过平滑未必有利。`beta=0.6` 只能作为 IPR 基线，应结合当前 pairwise-IoU 分布重新扫参。

### 成功判据

不能只看整体 `R1@0.3`。至少同时满足：

- R1@0.5、mIoU 提升；
- short/medium-short 分桶显著改善；
- long 分桶无明显退化；
- selected width bias 和 near-whole-video rate 下降；
- oracle R@N 不下降，说明没有破坏候选覆盖；
- duration 长视频分桶的秒级端点误差下降；
- 三个随机种子方向一致。

若 vanilla LP 使 selected width 上升、short 分桶下降，应立即停止原样移植，转向 EB-LP，而不是继续增加 stage 数。

---

## 11. 主要风险

1. **事件边界是 query-agnostic 的**：视觉变化不等于语言查询边界，只能做软候选和先验。
2. **边界过密**：oracle IoU 高的一部分原因是候选点很多，必须报告 selector 的实际性能，不能把 oracle 当收益。
3. **原子事件远短于 GT**：硬限定一个 event 必然伤害多动作/长描述查询。
4. **自举错误**：用当前 NLL 同时选种子、造伪标签、训练 confidence，可能放大已有 shortcut；冻结 VLM target 或高置信 trim 对照可缓解。
5. **positive-only LP 无法压制长区间**：必须有高置信 negative 或坐标收缩。
6. **50-grid 信息瓶颈**：边界输出再精确，若 scoring mask 不提高分辨率，模型也可能无法区分相邻候选。
7. **mixture mask 与报告区间不一致**：ActivityNet 中修改 reported boundary 时必须重新验证 reconstruction evidence 与输出区间的一致性。
8. **验证集协议**：Charades 当前 val=test，不适合用于大量边界超参数搜索。
9. **配置漂移**：README 的 weighted 设置、当前 outer 配置和 checkpoint 命名不一致，实验必须保存完整 resolved config。

---

## 12. 最终建议

综合论文机制、当前代码和边界统计，推荐做如下决策：

1. **不直接把 IPR Algorithm 1 原样加到当前 proposal 上。** 它只能重排，且扩张方向与过长问题冲突。
2. **第一优先级是 ActivityNet 的 outer-hull 和 50-grid 信息瓶颈。** 这两项比 LP 更直接地产生长区间和秒级误差。
3. **把事件边界当成端点词表，而不是 event=moment 标签。** 让 query reconstruction 从多个连续 event atoms 中选择最小充分区间。
4. **先做 inference-only minimal-sufficient trimming。** 这是成本最低、最能验证因果假设的一步。
5. **若验证有效，再实现 EB-LP：事件门控图、三态伪标签、完整加权 BCE、坐标 refinement。**
6. **保留 vanilla IPR LP 作为必要对照。** 若它按预期扩大 width 或伤害短 GT，将成为 EB-LP 设计合理性的直接证据。

一句话概括：**IPR LP 的“多阶段置信度校正”思想值得借用，但解决 `cpl_lrev` 的关键不是把正标签传播得更远，而是利用事件边界和查询边际证据，把无用的外侧事件从 proposal 中可验证地收回来。**
