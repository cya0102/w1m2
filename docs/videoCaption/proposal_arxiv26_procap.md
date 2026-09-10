# Proposal 15：PCMI——显著性完备的最小区间整流

**对应论文**：*ProCap: Prominence-guided Object Rectification for Faithful and Comprehensive Video Captioning*。

## A. 论文核心方法与有效机制

ProCap 是训练无关的 caption 后处理方法。它检测并跟踪视频对象，以空间面积 `A`、时间持续性 `P` 和关系运动/距离变化 `D` 计算 prominence，核心组合为 `A·(P+D-PD)`；再识别原 caption 漏掉的高显著对象，迭代补入描述，并尽量保留原有正确事实，直到显著性缺口闭合。有效机制是把“是否应该被描述”定义为持续可见或关系动态显著，而不是只看单帧置信；迭代式最小修改避免整句重写带来的新幻觉。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的 proposal 可能过窄而漏掉 query关键单元，也可能因 outer hull过宽；ProCap 的 prominence coverage 可判断关键视觉证据是否完整，最小修改可用于边界 expand/trim，而无需重训 generator。

差异是原 prominence 面向空间对象且 rectification 只会给 caption 加对象，直接照搬会鼓励区间越扩越宽。grounding 必须把 `A/P/D` 重释为 query evidence strength、连续持续性、关系/状态变化，并同时执行“补齐缺失 + 删除无关”，最终选择满足完备性的最短连续区间。

## C. 独立创新方案

### C1. 方案名称

**PCMI（Prominence-Complete Minimal Interval，显著性完备最小区间）**。

### C2. 目标问题

主要解决过宽区间、outer hull空隙、query单元遗漏和边界缺少可解释后处理。

### C3. 核心假设

目标区间应覆盖 query 中所有高显著证据轨迹，但无需覆盖无法增加这些轨迹完备性的上下文。以“满足 prominence coverage 的最小连续区间”为目标，可在 completeness 与 compactness 间得到比 NLL 更合理的平衡。

### C4. 方法设计

1. **query evidence tracks**：为 action、entity、relation 单元生成 `E_l(t)`。每个局部峰/track `i` 计算 `A_i`=query语义强度，`P_i`=在邻域内的连续占比，`D_i`=动作变化或实体关系距离变化。
2. **时序 prominence**：`π_i=A_i[P_i+D_i-P_iD_i]`。对静态实体，P 可补偿低 D；对短动作，D 可补偿低 P；低语义的剧烈镜头切换因 A 小不会入选。
3. **必要证据选择**：每个 query unit 保留 top prominence track，若 top1/top2差距小则保留多假设；用 unit type阈值避免停用词。
4. **迭代 expand**：若当前 proposal 未覆盖某必要 track，选择使新增 prominence/新增长度比最大的左或右扩展，一步只扩到该 track边缘。
5. **迭代 trim**：当所有必要 tracks 已覆盖，从两侧逐帧/逐簇删除；只要 coverage不降且被删帧 prominence低于阈值就接受。对内部低 prominence gap，若其两侧绑定不同事件假设则拆成两个 candidates，不盲目 outer hull。
6. **停止与分数**：最多 4步；最终 `q=coverage-λ length-μ outside_prominence`。只有 query evidence充足才允许 expand，避免场景上下文膨胀。
7. **接入**：作为 `GaussianMixtureProposalGenerator` 后的可选无参数 rectifier；输入原 boundary、frame/query feature，输出修正 boundary 与可解释的 covered units。最小版训练/推理流程均不变。

### C5. 与原论文的区别

ProCap 为 caption 补充遗漏的高显著空间对象，主要做单向 add。PCMI 把显著性对象换成时间证据轨迹，将关系动态用于发现动作边界，并设计 expand+trim 双向最小编辑及最短连续区间目标；它不生成/修改文字，处理的是 proposal boundaries。

### C6. 为什么可能有效

当前宽区间优势来自更多上下文可帮助重建，PCMI 不奖励一般上下文，只奖励对必要 query unit 有 prominence 的轨迹。漏关键动作的短段会定向扩到最近证据，而已覆盖所有单元的宽段会从两侧删除低贡献帧；P/D互补又可兼顾持续状态与瞬时动作，直接改善 foreground/background discrimination 和 boundary compactness。

### C7. 潜在风险

- 弱 feature 下 prominence track 可能把共现背景当必要证据。
- 多次相同动作时 top prominence 可能选择错误实例。
- 内部 gap 拆分会增加候选数或破坏复合事件。
- 训练无关后处理的阈值可能跨数据集不稳定。

### C8. 最小验证实验

仅用当前 frame-query cosine 和帧差构造 `A/P/D`，对 top-5 proposals 做最多一次 expand与一次 trim；不训练任何参数。对比原区间、仅 expand、仅 trim、PCMI，报告 R@1/R@5、平均宽度、unit coverage proxy及短事件 mIoU。若 PCMI 提升 IoU 0.5/0.7且平均宽度下降但 R@0.3不降，支持假设；若只 trim 导致 recall显著下降，prominence completeness 不可靠。

### C9. 进一步实验 / Ablation

- `A`、`A·P`、`A·D`、完整 prominence。
- expand-first、trim-first、联合优化。
- 连续最小区间 vs. 允许拆分候选。
- action/entity/relation 不同阈值。
- 规则无参、可学习 prominence head。
- 多实例 query、复合动作和静态状态分组失败分析。

## 依据

- 原论文：`arXiv26-ProCap- Prominence-guided Object Rectification for Faithful and Comprehensive Video Captioning.pdf`，重点为 prominence定义、缺失检测和迭代 rectification。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

