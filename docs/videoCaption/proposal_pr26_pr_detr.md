# Proposal 7：EARS——证据锚定的关系型时序槽

**对应论文**：*PR-DETR: Injecting Position and Relation Prior for Dense Video Captioning*。

## A. 论文核心方法与有效机制

PR-DETR 针对 DETR 式 event queries 缺少位置含义、彼此关系也未显式建模的问题。它对 GT center/duration 聚类形成二维静态 anchors，把位置编码注入 slot；通过 iterative slot attention 聚合当前视频的 scene-specific feature并预测相对 anchor 的 offset；在 decoder 中持续保留静态先验。论文还把 proposal pair 的相对距离、尺度比等关系编码成 head-specific attention bias，使事件 queries 按时间关系交互。有效机制是同时提供“每个槽应该去哪里”的绝对先验和“各槽如何分工”的相对先验，从而减少无序查询的冗余。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 只有 5 个 proposals，proposal 高重叠/坍缩，几何 push/pull 不等于语义分工；宽度分布又偏向中长事件，短段覆盖不足。PR-DETR 的 anchor bank 可令有限槽覆盖不同时间/尺度，relation attention 可阻止所有槽追随相同 NLL basin。

但 PR-DETR 用 GT event boundaries 聚类和 GIoU 监督，CPL_LREV 是弱监督 grounding，不能使用测试任务边界统计冒充可用先验。需用无标签的变化证据、固定覆盖设计或仅训练集合法统计，并以跨增强一致性/语义差异训练 offsets。

## C. 独立创新方案

### C1. 方案名称

**EARS（Evidence-Anchored Relational Slots，证据锚定关系槽）**。

### C2. 目标问题

主要解决 5 proposals 覆盖不足、短尺度缺席、proposal overlap/collapse，以及几何多样性与语义多样性脱节。

### C3. 核心假设

若 5 个槽从可解释的时间—尺度锚点出发，并在自注意力中看到彼此的 overlap、相对位置及语义证据相似度，它们会形成稳定分工；至少一个短尺度槽不会被 raw NLL 的宽区间优势吞没，R@5 覆盖和 R@1 排序均可能改善。

### C4. 方法设计

1. **无标签 anchor bank**：默认固定 5 个归一化 `(center,width)` 原型：两短、两中、一长；center 由视频的 query-agnostic变化峰附近初始化，width 集合固定为 `{0.10,0.18,0.30,0.48,0.70}`。严格不读 GT boundary。可选方案只在训练 split 上统计允许使用的 duration prior。
2. **scene/query-specific 聚合**：每个 anchor 编码为 slot query，先在 anchor 覆盖窗口内 cross-attend `h_t`，再与 query token cross-attend，输出 offset `Δc,Δlogw`。offset 用 `tanh` 限幅，防止第一步完全抹掉先验。
3. **关系矩阵**：每对槽计算 `R_ij=[IoU, log(w_i/w_j), (c_i-c_j)/w_i, cos(z_i,z_j)]`；MLP 生成每个 attention head 的 bias。高 IoU 且语义相似时施加负 bias，高 IoU但语义证据互补时允许交互。
4. **迭代更新**：两轮 `anchor→feature aggregation→relation self-attention→offset`；每轮都拼回原静态 anchor encoding，避免 slot 漂移。
5. **语义分工损失**：对 query 原子单元的 slot coverage 使用 set coverage；若两个槽的时间 mask 和 unit support 都近似，惩罚；若时间重叠但支持不同 query unit，不惩罚。替代单纯 component cosine push 的一部分。
6. **一致性监督**：同一视频做 crop/scale 后，匹配槽的归一化 anchor identity 保持，预测边界变换等变；不使用自 NLL 选择 winner。
7. **接入方式**：以 EARS 输出的 5 个 `(c,w)` 替代当前 proposal-level 外层参数；内部 Gaussian components 可先保留并围绕 slot boundary 初始化。推理仍输出 5 个候选，关系层只运行两次。

### C5. 与原论文的区别

PR-DETR 从 GT boxes K-means 获得锚点并以匹配/GIoU 训练 dense caption proposals。EARS 使用无标签变化峰和固定尺度覆盖，增加 query-unit 语义关系以区分“重叠冗余”与“重叠但功能互补”，并以时序增强等变性训练。在 CPL 中它还需与内部 Gaussian mixture 共存，而不是 DETR box decoder 的直接移植。

### C6. 为什么可能有效

当前 5 个随机/同质 proposal 容易被同一低 NLL 宽区间吸引。静态尺度 identity 保证短槽不会在早期完全变宽，关系 bias 让冗余槽彼此可见，语义分工损失则直接对准“几何不同但语义相同”的失败。这样提升的是 proposal set coverage 与 functional diversity，而非只改变单个 score。

### C7. 潜在风险

- 固定尺度与数据真实 duration 分布不匹配。
- 强 anchor 会限制 offset，使目标位于两个变化峰之间时漏检。
- 关系 attention 在仅 5 个槽时容量可能过剩。
- 与现有 Gaussian mixture 的多组件结构叠加后优化复杂，需先用单 Gaussian 验证。

### C8. 最小验证实验

只把 5 个 proposal 的初始 width 设为固定分层尺度并添加 pairwise relation rank head，保留原 generator 后续 offset、decoder 和 loss。比较 baseline、scale anchors only、relation only、EARS-lite；报告 R@5 分尺度 recall、pairwise IoU、有效独立 proposal 数与 R@1。若短段 R@5@0.5 提升 ≥3 点且平均 pairwise IoU 下降、长段 recall 不明显损失，支持假设；若槽最终 width 仍全部收敛到原分布，则 anchor persistence 不足。

### C9. 进一步实验 / Ablation

- 固定均匀尺度、变化峰 anchor、训练集 duration prior。
- 静态 anchor 仅初始化 vs. 每层持续注入。
- 几何 relation、语义 relation、二者联合。
- offset 限幅、迭代轮数 1/2/3。
- 单 Gaussian 外层 vs. 原 mixture 内层。
- proposal 数 5/8/10，检验收益来自关系设计还是纯数量。

## 依据

- 原论文：`PR26-PR-DETR- Injecting position and relation prior for dense video captioning.pdf`，重点为 position anchors、iterative slot attention 与 relation prior。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

