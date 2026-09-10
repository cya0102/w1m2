# Proposal 9：TTBP——双粒度奖励的测试时边界抛光

**对应论文**：*RefZVC: Refinable Zero-Shot Video Captioning by Test-Time Reinforcement Polishing*。

## A. 论文核心方法与有效机制

RefZVC 在没有目标域训练的条件下，测试时迭代修正 caption。AdaSkip policy 自适应选择帧密度；Gaussian RBF cache 保存与当前状态相似的历史视觉信息并门控更新；sentence-level CLIP reward 评价整体语义，entity/noun-level reward 保证关键细节，beam 内均值作为 baseline 产生正负反馈。只更新轻量策略/投影/cache，冻结语言模型和 reward model。

有效机制是把一次性生成改成小步、可回退的 test-time search；整体与实体两级 reward 防止只追求笼统语义；group baseline 只比较同一样本候选，降低跨样本尺度偏差；cache 抑制迭代震荡。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的 R@5 明显高于 R@1，说明候选集中常已有更好区间但推理排序失败；同时缺少直接边界 refinement，LREV 又未在推理使用。RefZVC 的候选内相对奖励和 test-time polishing 适合在不重训主模型的情况下修正 top proposals。

但 caption policy 选择帧并生成词，grounding action 应改成边界 shift/trim/expand；reward 不能仍用 CPL reconstruction NLL，否则继续自我确认，必须采用冻结且相对独立的视觉语言证据和边界排他性。

## C. 独立创新方案

### C1. 方案名称

**TTBP（Test-Time Boundary Polishing，测试时边界抛光）**。

### C2. 目标问题

主要解决 R@5→R@1 排序落差、wide proposal、边界精修不足以及训练/推理信号不一致。

### C3. 核心假设

若已有 5 个候选覆盖正确事件，冻结主模型后围绕它们做少量局部边界动作，并用独立的整句+关键实体/action reward 比较同组候选，就能找到更紧致区间；group baseline 与记忆可避免绝对 reward 偏置和来回震荡。

### C4. 方法设计

1. **状态与动作**：状态为 top-5 `(s,e)`、区间 pooled feature、左右 shell feature、query phrase tokens；动作集为左/右边界 `{-δ,0,+δ}` 的组合、对称 trim/expand 和 stop，`δ=0.02`。
2. **两级冻结 reward**：`R_sent` 为冻结 video-text encoder 的整句相似度；`R_unit` 为动词/实体单元在区间内的 soft-min support；`R_shell` 惩罚左右 shell 仍与 query高度相似，`R_len` 只惩罚超过数据先验的无证据宽度。总 reward 不读取 NLL。
3. **组内 advantage**：每轮从每个边界采样 6–10 个动作，`A_i=R_i-mean_jR_j`；可训练轻量 action policy，最小版可直接 beam/coordinate search，无需在线反传。
4. **Gaussian RBF memory**：缓存历史 `(boundary,evidence,reward)`；新状态与旧状态相似时融合历史 reward，若相同区间反复出现则提高 stop 概率，防止震荡。
5. **接受规则**：只有 `R_sent` 不下降且 `R_unit` 不下降、总 reward 提升超过 ε 才更新；因此不能用缩成极短片段投机。
6. **迭代与输出**：最多 3 轮，从 5 个原 proposals 共同搜索；最终以 polished reward 排名并输出。主 CPL 参数完全冻结，inference latency 约为冻结 encoder 的若干局部重算。
7. **可选学习版**：在训练集离线运行 search，用 reward improvement 作为 advantage 训练 boundary policy；推理只执行贪心 2 步。

### C5. 与原论文的区别

RefZVC 优化帧跳采样和 caption token 生成，奖励是 caption—video 语义。TTBP 把动作空间重定义为连续边界编辑，加入 shell exclusivity 和完整性接受规则，搜索对象是 5 个 grounding proposals；RBF cache 存的是边界轨迹而非 caption history，且最小实现不需要 RL。

### C6. 为什么可能有效

当前问题不是所有好候选都不存在，而是 good proposal 常排在 2–5 位。TTBP 对每个样本做局部相对比较，绕开跨样本 NLL 校准；entity/action reward防止只靠场景相似度，shell reward 对准过宽边界。由于 reward 推理时实际使用，也消除 LREV“训练有、推理权重为零”的脱节。

### C7. 潜在风险

- 冻结 video-text encoder 的相似度也可能偏好宽上下文。
- 多次重编码区间增加推理延迟；需要缓存帧 feature，仅重做池化。
- 搜索在初始 5 个 proposals 都未覆盖目标时无能为力。
- 局部 reward 噪声可能导致错误 trim；接受规则需严格。

### C8. 最小验证实验

实现无训练 coordinate search：每个原 proposal 只比较原区间、左右各 trim 0.03、左右 shift 0.03，共 5 个变体；reward 用冻结 feature 的整句相似度 + query unit soft-min − shell 相似度。记录 oracle upper bound、实际 R@1、每样本额外耗时。若 proposal set 扩展后的 oracle 与实际均提升、R@1@0.5 ≥2 点且 IoU 0.7不降，值得训练 policy；若 oracle 不提升，局部动作范围/初始覆盖是瓶颈，应停止。

### C9. 进一步实验 / Ablation

- 整句、unit、shell、length reward 的逐项消融。
- 固定 search、beam、learned policy；1/2/3/5 轮。
- 有无 group baseline与 RBF memory。
- δ 固定 vs. 与当前 width 成比例。
- 只从 top1 搜索 vs. 5 proposals 联合搜索。
- 按初始 oracle coverage、短/长事件报告成功率和失败类型。

## 依据

- 原论文：`TIP26-RefZVC- Refinable Zero-Shot Video Captioning by Test-Time Reinforcement Polishing.pdf`，重点为 AdaSkip、RBF cache、双粒度奖励和测试时更新。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

