# Proposal 2：BPSE——双向语义集合等价的提议排序

**对应论文**：*OwlCap: Harmonizing Motion-Detail for Video Captioning via HMD-270K and Caption Set Equivalence Reward*。

## A. 论文核心方法与有效机制

OwlCap 关注详细视频描述中 motion 与 detail 不平衡，以及单一字符串相似度无法同时评价“说出的是否正确”和“该说的是否说全”。论文先通过 Motion-Detail Fusion、Fine-Grained Examination 和单元核验构造 HMD-270K，再提出 Caption Set Equivalence Reward（CSER）。

CSER 把预测与参考 caption 拆成原子语义单元，计算两个方向：预测单元能否在参考集合中找到对应项（correctness），参考单元能否在预测集合中被覆盖（completeness）。二者共同用于 group-relative preference optimization。有效机制是把一个容易被语言流畅度主导的整句得分，改成“无额外错误 + 无关键遗漏”的双向集合约束；论文消融表明只保留一向都会退化。

## B. 与 CPL_LREV 问题的对应关系

当前 CPL_LREV 的 raw NLL 更偏好包含上下文的宽 proposal；hard-min 只奖励当前最会重建句子的 proposal，且这个 proposal 不一定在时间上既充分又紧致。CSER 的两向标准恰好对应：completeness 要求目标语义均能在区间内找到证据，correctness 要求区间中的显著内容不要超出 query。

但 caption 的集合元素是文字事实，grounding 的输出是时间区间，不能把 CSER 文本匹配直接作为 reward。必须把 query 原子单元映射到帧证据，并把“多说”改造为“区间包含与 query 无关的显著事件”。

## C. 独立创新方案

### C1. 方案名称

**BPSE（Bidirectional Proposal-Set Equivalence，双向提议集合等价）**。

### C2. 目标问题

主要解决 NLL 与 IoU 不一致、宽 proposal 偏好、hard winner 自我确认和 R@5 到 R@1 的排序损失。

### C3. 核心假设

高 IoU 区间应同时满足：query 的所有必要动作/实体/关系都有视觉证据（完整性），且区间内的主导事件都可由 query 解释（正确性/排他性）。用这两个方向训练独立质量头，比“能否在 teacher forcing 下重建整句”更接近区间质量。

### C4. 方法设计

1. **原子化 query**：使用规则词性/依存分析或冻结文本模型，把 query 拆成 `U={u_l}`，类型为 action、entity、attribute、relation；保留整句 token 防止过度切分。
2. **帧—单元证据**：在 `DualTransformer` 输出 `h∈R^{B×T×D}` 上增加共享投影，得到 `E_{t,l}=cos(W_vh_t,W_uu_l)`；对每个 proposal 的软 mask `m_n` 聚合内部证据与外壳证据。
3. **完整性分数**：`C_n=mean_l softmaxpool_t(m_{n,t}E_{t,l})`，每个 query 单元都必须被区间内至少一段支持。
4. **正确性分数**：先以视频内局部峰值形成显著事件 token `z_j`；计算区间内每个 `z_j` 对任一 query 单元的最大匹配，`P_n=mean_{j∈n}max_l sim(z_j,u_l)`。高运动但无法被 query 解释的背景会降低它。
5. **质量头**：输入 `[C_n,P_n,width_n,inside-shell contrast]`，输出 `q_n`；推理排名用 `q_n`，而非 raw NLL 或 semantic vote。可用调和均值 `2C_nP_n/(C_n+P_n)`作为不可学习基线。
6. **组内偏好训练**：对每个基础 proposal 生成 inward/outward/left/right perturbations，共一组 10–20 个候选。用停止梯度的双向分数排序，采取组内相对 advantage；只比较分数差超过阈值的 pair，避免噪声伪标签。
7. **损失**：`L_BPSE=L_rank(q^+,q^-)+λ_abs BCE(q^+, stopgrad(H(C,P)))+λ_cov(1-C^+)`。原 reconstruction loss 保留为语义可识别性正则，但不再决定 hard-min winner。
8. **推理**：生成方式不变，计算一次 `E` 与质量头后排序；不需要生成 caption，也不需要 RL sampling。

### C5. 与原论文的区别

OwlCap 对生成文本与参考文本做原子事实集合等价，并通过 GRPO 优化 captioner。BPSE 对 query 单元、区间内部事件及外壳背景做跨模态集合等价；“正确性”从文字事实一致改为 foreground exclusivity，“完整性”从参考 caption 覆盖改为 query-unit visual coverage；采用可微 listwise/pairwise 训练而非大模型 RL。

### C6. 为什么可能有效

宽 proposal 虽然能降低 NLL，却往往在 `P_n` 上受罚，因为额外事件无法被 query 解释；过窄 proposal 会漏掉动作或参与者，在 `C_n` 上受罚。二者联合形成紧致区间的可识别条件。质量头在训练和推理一致使用，切断“训练 pseudo weight 来自 NLL、推理又不用 LREV”的循环，并把 R@5 中已有的好候选提到 R@1。

### C7. 潜在风险

- query 原子化错误会把同义短语拆坏，导致虚假缺失。
- 静态背景中的显著事件 token 未必是语义事件，正确性估计可能误罚。
- 由自身 feature 产生伪排序仍有偏差，因此必须停止梯度并使用高 margin pair。
- 多单元矩阵增加 `O(TL_q)` 开销，但 `L_q` 通常很小。

### C8. 最小验证实验

冻结 CPL_LREV，只缓存 5 个 proposal 及 frame/query feature；训练一个两层质量头。候选加入每侧缩/扩 `0.05` 的扰动，使用规则抽取动词和名词，双向分数只产生 pair 标签。对比 raw NLL、现有 semantic vote、仅 completeness、仅 correctness、BPSE。若 BPSE 在 proposal set 不变时使 R@1@0.5 提升 ≥2 个百分点、R@5 基本不变且平均预测宽度接近 GT 分布，假设成立；若 R@1 不增或只靠把所有区间缩短导致 recall 下降，应放弃当前打分定义。

### C9. 进一步实验 / Ablation

- 原子单元：规则、冻结 LLM、整句单 token。
- 集合聚合：mean、min、soft-min；两向调和、乘积、learned fusion。
- 是否加入区间内显著事件正确性、shell exclusivity、绝对质量项。
- 随机扰动 vs. 邻近高相似 hard candidates；margin 阈值敏感性。
- 质量头独立 feature vs. 与 reconstruction decoder 共享 feature。
- 按 query 单元数、GT 长度和动作/实体类型分层报告。

## 依据

- 原论文：`AAAI26-OwlCap- Harmonizing Motion-Detail for Video Captioning via HMD-270K and Caption Set Equivalence Reward.pdf`，重点为 HMD 构造和 CSER。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

