# Proposal 12：SPPO——保正样本的协同提议偏好优化

**对应论文**：*SynPO: Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning*。

## A. 论文核心方法与有效机制

Syn 处理详细 caption 的 preference optimization 容易“只压低 rejected 样本、同时破坏原有描述能力”的问题。论文从同一视频采样多个 caption，以时序分段事实核验、指令/流畅度/客观性和多样本自一致性构造 preference pair。SynPO 的对比项使用 token log-probability 的几何平均突出困难、区分性 token，同时加入 preferred response 的算术平均正奖励，避免负样本主导与 objective drift；不依赖 reference model。

有效机制在于 preference pair 的质量来自多个相对独立维度，而优化目标同时包含“拉开正负”和“维持正样本绝对质量”，不是一味把 rejected score 推低。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的 hard-min NLL 只强化当前赢家，容易自我确认；训练 loss 下降但 localization 下降，与 preference 中的 objective drift相似。R@5 中已有好候选说明候选间偏好学习具有空间。SynPO 的正样本保护项可避免通过普遍降低/扭曲 reconstruction 来制造 margin。

差异是 caption pair 有词概率，grounding pair 是区间。需要把 token-level 几何/算术聚合改成 query-unit support，并由时序事实、增强一致和紧致性构造 proposal preference；不能使用 GT IoU 选 pair。

## C. 独立创新方案

### C1. 方案名称

**SPPO（Synergistic Proposal Preference Optimization，协同提议偏好优化）**。

### C2. 目标问题

主要解决 hard winner 自我确认、NLL/IoU 错位、ranking 失败和优化中语义能力漂移。

### C3. 核心假设

在同一视频-query 内，对 base proposal 及其边界扰动进行相对比较，比跨样本绝对伪标签更可靠；偏好 loss 若同时保持 preferred 区间的绝对 query-unit support，就不会靠把 rejected 全部压低来获得虚假 margin。

### C4. 方法设计

1. **候选池**：5 个基础 proposals 各产生 inward/outward/left/right shift 及短/中/长 width variants，合并 NMS 后保留 20 个。
2. **三维伪评价**：`F` 为将区间分成 4 个 temporal blocks 后的 action/entity fact support；`C` 为两种 feature dropout/速度增强下 score 与边界的一致性；`B` 为内部支持减左右 shell 支持及无证据宽度罚。三项均 stop-gradient。
3. **可靠 pair**：只有某候选在至少两维胜出、无一维显著更差且综合 margin 超阈值，才成为 preferred；避免单一自举分数决定标签。
4. **unit-level logits**：质量头为每个 query unit 输出支持 logit `z_{n,l}`，另输出 boundary logit。proposal 总 score 不直接由 raw NLL产生。
5. **SynPO 式对比项**：对 preferred/rejected 的单位概率用几何平均 `G_n=exp(mean_l log σ(z_{n,l}))`，优化 `-log σ(β(G_+-G_-))`，使任一关键单元缺失都会拉低 preferred。
6. **正样本保护**：加入 `L_pos=-mean_l σ(z_{+,l})-γ q_B^+` 的算术平均绝对奖励；同时对 preferred 保留原 query reconstruction loss，对 rejected 不施加反向语言破坏。
7. **总体损失**：`L=L_base+λ_prefL_pair+λ_posL_pos+λ_consL_aug`。不使用 hard-min proposal identity；每个样本可贡献多个高置信 pair。
8. **推理**：只保留原 5 proposals 或扩展池均可，以 learned quality score排名；最小部署不改变 proposal generator。

### C5. 与原论文的区别

SynPO 优化生成 caption 的 token 概率，pair 由 caption 事实/风格/一致性评价得到。SPPO 优化 proposal 的 query-unit 与 boundary quality，pair 由区间时序事实、增强一致性和 shell 紧致性产生；正样本保护改成绝对视觉支持与重建保持项，不涉及语言风格或 reference-free LM preference。

### C6. 为什么可能有效

几何平均使“只支持 person/object 但漏掉关键 action”的候选不能靠高均值获胜；boundary维度使包含大量背景的低 NLL 区间受罚。多维胜出规则减少单一模型偏差，正样本项阻止 quality head 通过所有 score 一起下移满足 pair loss。它改变的是 training signal 与 ranking objective，直接对应 loss 降而定位退化的机制。

### C7. 潜在风险

- 三个伪评价仍共享同一视觉 encoder，错误可能相关而非独立。
- 高阈值导致可用 pairs 过少，低阈值则噪声大。
- 几何平均对一个错误 unit 极敏感，query 解析错误会放大伤害。
- 扩展候选池增加训练计算，推理最好仍用 5 个。

### C8. 最小验证实验

冻结 CPL_LREV，离线缓存每个样本 20 个候选 feature，只训练 quality head；以简单 action/entity CLIP support、两次 dropout 一致性、shell contrast 选 pair。比较 BCE 伪回归、普通 pairwise、SPPO 无正项、完整 SPPO。若完整 SPPO 在相同 proposal set 上提升 R@1@0.5 ≥2 点，且 preferred 的平均绝对 support 不随训练下降，支持假设；若 pair 数低于样本数 0.5 倍或不同伪评分高度冲突，应停止。

### C9. 进一步实验 / Ablation

- `F/C/B` 三种 pair 依据逐项去除。
- 算术平均、几何平均、soft-min 的 unit 聚合。
- 有/无正样本保护及其权重；保留 reconstruction 的作用。
- pair margin 与每样本 pair 数。
- 原 5候选 vs. 扰动扩展候选训练/推理。
- 与 hard-min、soft responsibility、listwise softmax 的直接比较。

## 依据

- 原论文：`TIP26-SynPO- Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning.pdf`，重点为 preference data 构造与正负协同目标。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

