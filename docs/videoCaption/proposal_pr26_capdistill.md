# Proposal 6：HSBD——高分辨率语义到边界的分层蒸馏

**对应论文**：*Dual-Hierarchical Knowledge Distillation for Video Captioning*（CapDistill）。

## A. 论文核心方法与有效机制

CapDistill 面对轻量 captioner 难以同时保留 object 与 action 知识的问题。它先依据 TF-IDF 语义相关性和多参考 caption 的 CIDEr 一致性评估监督质量，再以高质量 caption 分别引导 object/noun 与 action/verb teacher。高容量 teacher 具有 object self-attention、motion self-attention、object-motion cross-attention，以及 action-frame、frame-frame 两级层次建模；student 通过 object、action feature 与 word distribution 多层 KL 接收知识。论文结果显示不同层级的蒸馏互补，但盲目加入所有 feature 会产生冲突。

有效机制是：高容量、任务分解的 teacher 在训练期形成更细的结构，再把选择过的中间关系和输出分布传给轻 student，而不是要求 student 自己同时学会全部目标。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 将 200 帧降到 50，短事件边界丢失；同一 `DualTransformer` 又服务 proposal 与语言重建，存在目标冲突。CapDistill 的分层 teacher/student 思路可以让 200-step、动作专用 teacher 保留高分辨率变化，再蒸馏给当前 50-step student。

差异是 captioning teacher 有 GT caption 质量和输出词分布，grounding 没有边界标签。不能声称 teacher 知道真实边界；必须用 query-frame 对齐、增强一致性和独立变化信号训练 teacher，并只蒸馏高置信关系。

## C. 独立创新方案

### C1. 方案名称

**HSBD（Hierarchical Semantic-to-Boundary Distillation，高分辨率语义到边界蒸馏）**。

### C2. 目标问题

主要解决 200→50 降采样造成的短事件信息损失、共享表示冲突和缺乏直接 start/end 感知。

### C3. 核心假设

训练期的 200-step 专用 teacher 能分开建模 query 名词对应的状态证据与动词对应的变化证据，并在高时间分辨率上产生比 CPL proposal 更可靠的软边界。若只蒸馏置信度高、跨增强一致的关系，50-step student 可获得边界结构而不增加推理成本。

### C4. 方法设计

1. **Teacher 输入**：使用 `frame_fc` 后完整 `F^200`；文本拆为 noun/entity token 与 verb/action token。Object teacher 用帧 self-attention，Action teacher 输入 `[F_t,F_t-F_{t-1}]` 并做 object-motion cross-attention。
2. **两级层次**：先产生 token-to-frame 分布 `A_obj,A_act∈R^{L×200}`；再以局部窗口 attention 预测 start/end 分布 `p_s^T,p_e^T` 和 foreground `p_f^T`。start/end 由 action attention 的上升/下降与 object 持续性共同产生。
3. **无边界 teacher 训练**：正 video-query 对高于 batch 内负 query；时间 crop、速度扰动后对齐回原坐标应保持分布一致；打乱帧时边界置信度应下降。teacher 不读取 CPL proposal/NLL，避免闭环。
4. **高质量门控**：仅当两种增强下边界 Wasserstein 距离低、正负 query margin 高时，使用样本蒸馏；权重为停止梯度的 `w_T`。
5. **Student 接入**：在现有 50-step `h` 上增加 object/action attention 和 boundary heads。将 teacher 分布按 4 帧一组质量守恒地降采样，蒸馏三层：`KL(A^T||A^S)`、`KL(p_f^T||p_f^S)`、`KL(p_{s/e}^T||p_{s/e}^S)`。
6. **影响 proposal**：student start/end logits作为额外通道输入 `GaussianMixtureProposalGenerator`；推理时可用其对 outer boundary 做小范围 offset，teacher 完全删除。
7. **训练顺序**：先训练 teacher 并冻结；再在 baseline checkpoint 上只训练 student heads 5 epochs；最后以小学习率联合训练。原 NLL、mixture regularizer 不变。

### C5. 与原论文的区别

CapDistill 蒸馏 object/action/word prediction 来压缩 captioner，并以多参考 caption 评价监督质量。HSBD 将层级重定义为 semantic role→frame alignment→start/end distribution；质量由时序增强一致性而非 CIDEr 产生；目标不是压缩模型容量，而是训练期高分辨率到推理期低分辨率的 temporal knowledge transfer。

### C6. 为什么可能有效

短事件在 50 个采样点上可能不足两个点，主模型很难恢复已丢失的变化。teacher 在 200 steps 上先形成边界概率，再把概率质量而非单帧 feature 蒸馏给 student，可保留亚采样级边界信息。object/action 分开还减少语言重建与运动定位争抢共享表示；推理新增边界头直接服务区间而非 NLL，理论上尤其改善 IoU 0.7。

### C7. 潜在风险

- 无 GT 的 teacher 可能只是一个更复杂的错误伪标签器。
- 静态事件的 action attention 不可靠；增强一致也不代表正确。
- 训练 teacher 成本明显增加，且 200-step attention 显存较大。
- 强蒸馏会限制 student 超越 teacher，应使用置信门控和温度。

### C8. 最小验证实验

先不训练完整 teacher：以冻结视觉文本相似度和帧差生成 200-step软 foreground/start/end，跨两种 crop 一致时才蒸馏一个 50-step boundary head；仅用该 head rerank/微调 top proposals。保持原生成与 loss 不变。比较无蒸馏、foreground-only、foreground+boundary。若短段 R@5 mIoU 提升 ≥0.03、IoU 0.7 有提升且长段不降超过 1 点，说明高分辨率蒸馏值得投入；若 teacher confidence 与真实 IoU 无相关性，应停止。

### C9. 进一步实验 / Ablation

- 100/200-step teacher；规则 teacher vs. learned hierarchical teacher。
- object、action、foreground、boundary 四类蒸馏的组合。
- 无门控、margin 门控、增强一致门控。
- KL、EMD、soft cross-entropy；蒸馏温度和权重。
- teacher/student 是否共享 `frame_fc`。
- 仅训练期 teacher vs. 推理也使用 teacher，量化成本收益。

## 依据

- 原论文：`PR26-Dual-hierarchical knowledge distillation for video captioning.pdf`，重点为 caption grading、object/action teacher 与多层蒸馏。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

