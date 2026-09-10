# Proposal 5：EACA——证据式动作—上下文自适应分配

**对应论文**：*Ask and Focus More: Question-Prompt Uncertainty Allocation for Dual-Controllable Video Captioning*。

## A. 论文核心方法与有效机制

论文针对 caption 生成时“动作细节”和“全局句意”难以同时控制，采用 question prompt 指定关注粒度，并分别建模 object、action、sentence 信息。其关键模块 SFM 先融合多粒度特征，再用 Dirichlet evidential modeling 为不同信息源输出证据 `α=softplus(·)+1`；期望权重按证据自适应分配，总不确定性与证据总量反比。有效机制不是普通 feature fusion，而是：模型在每个样本上显式表示“哪一路可靠、哪里不确定”，避免固定比例融合把错误模态强行写入 caption。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 使用共享 Transformer 同时服务 proposal/reconstruction，动作变化与静态场景上下文容易混在一起；固定 loss 权重不适合不同 query；宽 proposal 因携带更多场景上下文而 NLL 更低。论文的 action/sentence 分解可对应“定位动作核心”和“识别场景语义”，evidential allocation 可对应候选级可靠性与 loss/pseudo-label 权重。

但 caption 中 question prompt 是用户控制生成内容；grounding 的 query 已给定，不需再生成问题。需要把 prompt 改成由 query 动词/实体自动形成的路由 token，并确保 context 分支不能单独把边界向外拉宽。

## C. 独立创新方案

### C1. 方案名称

**EACA（Evidential Action–Context Allocator，证据式动作—上下文分配器）**。

### C2. 目标问题

主要解决宽 proposal 偏好、共享表示目标冲突、固定损失权重和不可靠 pseudo weight；同时改善边界定位与排序置信度。

### C3. 核心假设

query 对场景身份和动作变化的依赖不同。若模型能分别估计 action branch 与 context branch 对每个候选的证据量，并在高不确定性时减少伪监督，那么静态上下文只能帮助“是什么”，动作变化负责“发生在哪”，可降低凭场景扩大区间的倾向。

### C4. 方法设计

1. **双分支位置**：在 `frame_fc` 后建立 `ActionEncoder` 与 `ContextEncoder`。ActionEncoder 输入一阶差分、局部 3-frame 卷积和原 feature；ContextEncoder 输入低通/窗口平均 feature。两者各 1 层 temporal Transformer，不共享最后层。
2. **query prompt 路由**：动词短语 token `q_a` 查询 action feature，实体/场景 token `q_c` 查询 context feature；整句 token提供全局校准。无显式动词时使用 learned null-action token。
3. **proposal 级证据**：对每个 proposal mask 聚合两支特征，两个 head 输出非负证据 `e_a,e_c`，令 `α_k=e_k+1`、权重 `w_k=α_k/(α_a+α_c)`、不确定性 `u=2/(α_a+α_c)`。
4. **不对称融合**：proposal 中心/边界 offset 仅由 action feature和 `w_a` 主导；context 只为 proposal semantic quality 与 reconstruction 提供条件。最终 rank feature 为 `[w_a z_a,w_c z_c,u,inside-shell_a]`，防止 context 直接驱动扩宽。
5. **证据监督**：视频时间打乱时 action evidence 应下降；背景模糊/颜色抖动时 context evidence 应下降但 action 相对保持；两种增强的一致性形成 evidential regularizer。错误高置信受到 evidential classification loss 惩罚。
6. **自适应训练权重**：proposal 的 reconstruction/pseudo-ranking 权重乘 `(1-u)`；高不确定候选不参与 hard-min。boundary compactness 权重乘 `w_c/(w_a+ε)`，即只靠 context 的候选承受更强宽度惩罚。
7. **推理**：用证据融合质量头排名；输出 `u_n` 供失败分析。生成 5 个 proposal 的流程不变，新增两支编码会增加少量计算。

### C5. 与原论文的区别

原论文通过 question prompt 控制 caption 偏动作或偏全局，并用 Dirichlet 不确定性融合 object/action/sentence。EACA 将控制信号变成 query 自动路由，将信息源重定义为“边界动作变化”和“语义上下文”，且采用不对称权限：context 可确认语义但不能直接产生边界；证据还用于伪标签和 loss gating，而不只是 decoder feature 融合。

### C6. 为什么可能有效

当前 raw NLL 喜欢宽区间，根因之一是场景/物体上下文让语言重建更容易。EACA 把这种线索隔离到 context branch，并要求边界由局部变化证据决定；若候选只有场景证据、没有动作证据，它会得到高不确定性和更强 compactness penalty。这样同时改变 temporal representation、boundary perception 与 training signal，并提供推理阶段真正使用的置信度。

### C7. 潜在风险

- 静态状态类 query 没有明显动作变化，硬限制 action 可能伤害性能。
- evidential head 可能通过无限增大证据逃避不确定性，需要 KL/证据正则。
- 双编码器增加约 1 层 Transformer 的计算量。
- 动词/实体自动路由错误时，两支信息被错误隔离。

### C8. 最小验证实验

不复制完整 Transformer，仅从现有 `h` 构造差分池化 `z_a` 和低通池化 `z_c`，训练 evidential rank head；proposal generator 与原 loss 冻结。对比固定 0.5/0.5 融合、softmax gate、Dirichlet EACA。报告 R@1、ECE/置信—IoU Spearman、预测宽度和短/长 query 分组。若 EACA 的 IoU—置信相关性提高、R@1@0.5 提升且仅 context 高权重候选的平均宽度下降，则值得联合训练；若证据饱和或动作弱 query 显著退化，需放弃硬分支权限。

### C9. 进一步实验 / Ablation

- 单共享表示、双 feature view、双独立 encoder。
- softmax gate vs. Dirichlet evidence；有无证据 KL 正则。
- context 是否允许预测 center、width、均不允许。
- 不确定性用于 ranking、pseudo-label gating、loss weighting 的独立贡献。
- 时间打乱、背景扰动、两者混合的一致性监督。
- 按动态动作/静态状态 query、GT 时长分层。

## 依据

- 原论文：`PR26-Ask and focus more- Question-prompt uncertainty allocation for dual-controllable video captioning.pdf`，重点为 Q-Prompt、SFM 与 evidential uncertainty allocation。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

