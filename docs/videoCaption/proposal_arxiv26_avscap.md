# Proposal 14：SBSB——语义—边界协同的局部事件绑定

**对应论文**：*AVSCap: Orchestrating Audio-Visual Synergy for Omni-modal Video Captioning*。

## A. 论文核心方法与有效机制

AVSCap 指出 audio 与 visual 直接早期融合会掩盖各自贡献，也难以表达真正的跨模态事件。它采用 decoupled-then-fused 流程：先分别形成视觉、音频 anchors/标签并保持标签一致，再通过局部事件绑定形成 audio-visual synergy；数据构造还核验 tag preservation 与 tag-event consistency。训练除 SFT 外，用 GRPO 分别奖励长度、speech recall 和 audio-visual synergy event recall。有效机制是先保证单模态证据不丢，再只对时间/语义上相互支持的局部事件融合，避免“两个模态都出现”被误当成协同。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的共享表示同时承担 query语义重建与时间边界，context 线索可能淹没局部变化；outer hull 又把相离 components 直接包成一个区间。AVSCap 的 decouple-then-bind 可重新解释为 semantic evidence 与 boundary-transition evidence 先分开，再只在局部绑定后形成有效 event proposal。

这里没有 audio，不能把 AVSCap 变成“加入音频”。需要将两种信息源定义为 query-conditioned semantic track 与 query-agnostic/弱条件 temporal transition track，并把 synergy 定义为语义峰与开始/结束变化在同一局部事件内闭合。

## C. 独立创新方案

### C1. 方案名称

**SBSB（Semantic–Boundary Synergy Binding，语义—边界协同绑定）**。

### C2. 目标问题

主要解决共享表示冲突、缺少边界信号、outer hull 跨背景和宽区间偏好。

### C3. 核心假设

语义相似只能说明“像目标”，变化证据只能说明“这里发生转移”；正确区间要求一个语义持续段被成对 start/end transition 局部包围。先解耦再验证这种 synergy，可排除仅有场景语义的宽段和仅有运动但语义错误的变化点。

### C4. 方法设计

1. **语义 anchor 分支**：query token 对 frame feature cross-attend，输出每个 query unit 的 support track `S_l(t)` 及整句 track `S(t)`。
2. **边界 anchor 分支**：不输入完整 query，只输入动作类型 gate（动态/静态）和 `[F_t-F_{t-1}, local variance]`，输出 start/end transition `B_s(t),B_e(t)`。与语义分支不共享最后两层。
3. **局部绑定矩阵**：对每个语义峰/持续段 `j` 与转移 pair `(s_k,e_m)` 计算 `Y_{jkm}`，包含顺序合法性 `s<j<e`、距离、内部语义覆盖、外部语义泄漏和边界变化方向。只允许局部半径内绑定。
4. **proposal 生成**：每个现有 Gaussian proposal提供一个候选语义段；绑定器选择 start/end anchors 并输出 `(s,e)`，而非直接使用 component outer min/max。若找不到成对转移，保留原边界但降低置信度。
5. **synergy score**：`q_n=semantic_coverage×boundary_pair_confidence×binding_consistency`，乘法保证任一证据缺失都不能被另一支补偿。
6. **训练信号**：时间 shift 后 semantic 与 boundary anchors 应同步移动；把另一视频的 transition track 与当前 semantic track拼接为 hard mismatch，synergy必须低；对同视频远端转移也做负绑定。无需 GT boundary。
7. **损失/推理**：加入 anchor contrastive、binding BCE/InfoNCE 与等变 loss；推理用绑定后的区间和 synergy score，训练/推理一致。

### C5. 与原论文的区别

AVSCap 的两路是 audio 与 visual，目标是完整 caption 的跨模态事件描述。SBSB 的两路是 query语义与时序转移，目标是 start/end interval；它借鉴“独立 anchoring 后只做局部一致绑定”，但新设计了成对边界拓扑、外部泄漏和错配 transition 负例，不使用音频或 caption GRPO。

### C6. 为什么可能有效

宽 proposal 的语义 coverage 可能高，但其左右边界未必有匹配转移，乘法 score 会压低；短而只覆盖动作峰的 proposal 若缺少必要实体持续证据也不能获高分。成对局部绑定让 final interval 必须是连续事件闭包，避免 outer hull 跨过中间背景，且分支解耦减轻语义重建对 boundary feature 的干扰。

### C7. 潜在风险

- 静态事件或渐变事件没有明显 transition pair。
- 查询相关边界并非总与视觉差分峰一致。
- 乘法 score 对一支未校准过于敏感，可能普遍压低置信度。
- anchor pairing 为 `O(T²)`；实际需 top-k start/end 限制到 k≤10。

### C8. 最小验证实验

用冻结 frame feature 计算 query相似 track 和帧差 track，仅对原 5 proposals 在边界附近 top-3变化点做吸附，并以乘法 synergy rerank。保持所有训练参数不变。比较语义分、变化分、早期相加、局部乘法绑定。若 R@1@0.5/0.7 提升、跨内部低语义 gap 的区间减少，且静态 query不明显退化，支持完整模型；若变化点吸附低于原边界，则边界分支信号不足。

### C9. 进一步实验 / Ablation

- 共享、部分共享、完全解耦 encoder。
- 相加、门控、乘法 synergy。
- 单边界峰 vs. start/end成对绑定。
- 同视频远端、跨视频、时间错位三种 mismatch negatives。
- 动态/静态 query gate；局部半径和 top-k。
- outer hull、最大连续 component、SBSB绑定边界对比。

## 依据

- 原论文：`arXiv26-AVSCap- Orchestrating Audio-Visual Synergy for Omni-modal Video Captioning.pdf`，重点为 decoupled-then-fused anchors、局部 synergy与奖励设计。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

