# Proposal 3：DFCQ——双层事实校准的区间质量头

**对应论文**：*DualFact+: A Multimodal Fact Verification Framework for Procedural Video Captioning*。

## A. 论文核心方法与有效机制

DualFact+ 解决 procedural video caption 的事实错误难以被单一文本相似度发现的问题。它把事实核验拆成两层：

- **概念层**：ACTION、OBJECT、TOOL、LOCATION 等语义角色是否正确、是否遗漏；必要时补全隐含论元。
- **上下文层**：谓词—论元关系是否与具体视频片段一致，避免各个词都出现但组合关系错误。
- **困难负事实**：通过合理的角色替换、关系交换构造“语言上通顺、视觉上错误”的事实，再由多模态核验器区分。

其有效机制是把笼统的相似度分解为可诊断的 role support 与 relation support，并以 plausible counterfactual negatives 迫使模型真正看视频。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的 decoder 在 teacher forcing 下可依赖语言前缀；固定远端 outside negative 太容易，活跃率不足 0.2%；raw NLL 又无法判断“人、物、动作都出现但发生在不同时间”的关系错误。DualFact+ 的角色层可检查元素是否齐全，上下文层可检查这些元素是否在同一候选区间形成正确关系，角色替换负例则比固定左右背景难得多。

差异在于 DualFact+ 是 caption factuality evaluator，通常有待核验 caption；grounding 只有 query 和候选区间。改造后应把 query 自身视为待验证事实，把区间视为证据，而不是再生成一条 caption。

## C. 独立创新方案

### C1. 方案名称

**DFCQ（Dual-level Fact-Calibrated Quality head，双层事实校准质量头）**。

### C2. 目标问题

主要针对语言捷径、容易负样本、语义关系判别弱和缺少独立 proposal quality estimator；重点改善 R@1 排序。

### C3. 核心假设

真正匹配 query 的区间不仅应支持独立的 action/object/tool/location，还应在同一时间上下文中支持正确的谓词—论元组合。两层核验加上角色级困难反事实，可产生与 IoU 更相关、且不由 reconstruction NLL 自举的质量信号。

### C4. 方法设计

1. **事实图解析**：将 query 解析为角色节点 `R={r_i}` 和关系三元组 `F={(predicate,arg_i,arg_j)}`；无法解析时退化为动词—名词边。解析结果缓存，不参与梯度。
2. **概念核验器**：每个 proposal 用 soft mask 对 `h_t` 做 role-conditioned attention，输出各角色支持概率 `p_i^n` 和角色缺失概率。不是平均池化：每个角色拥有独立 query vector。
3. **上下文核验器**：取被各角色关注的时间分布 `a_i^n(t)`，构造 pair feature `[z_i,z_j,z_i⊙z_j,overlap(a_i,a_j)]`，预测三元组成立概率 `p_f^n`。若 object 和 action 分别在区间两端、时间注意力不重叠，关系分会降低。
4. **困难反事实**：同 batch 或同视频中做三类替换：保持 object 换 action、保持 action 换 object、交换两个角色的时间证据。再加入相邻 query 的事实图作为 hard negative；负样本需与原 query 文本相似度高于阈值。
5. **质量分数**：`q_n=MLP([mean_i p_i^n,min_i p_i^n,mean_f p_f^n,width,inside-shell])`。训练用正事实对比负事实，另要求原 query 的 score 高于腐化事实；只在 top proposals 中做局部排序。
6. **边界伪比较**：对同一 proposal 的 trim/expand 版本，若概念支持不降而关系支持与 shell contrast 上升，则作为 preferred；反之不产生标签。这样伪监督由事实一致性而非 NLL 产生。
7. **接入点**：新增 `FactQualityHead`，读取现有 `h`、query token、`all_props` mask；不修改 `GaussianMixtureProposalGenerator` 的第一阶段。训练后推理以 `q_n` 排名，可与 NLL 做温度校准但默认不混合。

### C5. 与原论文的区别

原论文评估或核验生成的 procedural caption，事实是 caption—video 一致性。DFCQ 不生成文本，把 query 解析后的事实直接作为 grounding 条件；概念/上下文核验变成候选区间质量函数，并加入“时间注意力是否共现”这一 grounding 特有约束。负例也从文字事实替换扩展为角色时间证据交换。

### C6. 为什么可能有效

teacher forcing 可让模型凭“person is ...”预测后续词，但不能凭语言前缀证明 action 与 object 在候选时间内共现。DFCQ 的每个角色必须从区间视觉 feature 中取证，关系核验还要求证据时间重叠；宽区间把不同时刻的元素拼在一起时将被上下文层识别。可混淆的角色替换使负样本持续活跃，因而增强 semantic discrimination 与 foreground/background discrimination。

### C7. 潜在风险

- 自动语义角色解析在简短、非标准 query 上可能不稳定。
- C3D feature 对 tool/attribute 等细粒度角色的可识别性有限。
- 同视频相邻 query 可能实际重叠，错误当负例会引入噪声。
- 关系核验器参数较多，小数据上会过拟合语言角色模式。

### C8. 最小验证实验

只抽取 action 和 object 两类角色，冻结 CPL_LREV，训练概念核验器与一个二元 action-object 共现头；负例仅用 batch 内同 object/异 action 和同 action/异 object。保持 proposal set 不变，与 NLL/semantic vote 做 rerank 对比。观察 R@1@0.3/0.5/0.7、fact-negative AUC、负 margin 活跃率及 R@5。若活跃率明显高于原 outside loss 且 R@1@0.5 提升而 R@5 不变，支持方向；若核验 AUC 接近随机或 gains 只来自 query 文本先验，应停止。

### C9. 进一步实验 / Ablation

- 仅概念层、仅关系层、双层；有无隐含论元补全。
- 随机负例、固定 outside、角色替换、时间证据交换。
- role attention 的 softmax 温度和时间共现定义。
- quality head 是否加入 width/shell，是否与 NLL 融合。
- 冻结/联合训练 CPL encoder，阻断 query-only shortcut 的 video-shuffle 检验。
- 按 compositional query 与单动作 query 分组评估。

## 依据

- 原论文：`ACL26-DualFact+- A Multimodal Fact Verification Framework for Procedural Video Captioning.pdf`，重点为概念/上下文双层事实核验与负事实构造。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

