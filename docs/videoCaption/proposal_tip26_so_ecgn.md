# Proposal 11：QD-DMBR——查询主导的方向—幅度解耦边界修正

**对应论文**：*Subjective-Objective Emotion-Correlated Generation Network for Subjective Video Captioning*（SO-ECGN）。

## A. 论文核心方法与有效机制

SO-ECGN 处理“用户主观情绪意图”与“视频客观情绪证据”不一致的问题。它以 subjective emotion 作为主导基底，再由 video-conditioned objective emotion 做增量修正；通过视频条件的情绪词典域迁移、随生成历史变化的动态 mask 和时间重加权得到客观证据。论文还把 emotion 的 polarity 与 intensity 通过正交池化解耦，最后由 decoder 自适应融合事实与情绪。

有效机制是“主条件不被视觉噪声覆盖，但允许证据驱动的小步修正”，以及把容易纠缠的离散方向与连续强度拆开预测。该机制比直接拼接主客观 feature 更稳定、更可解释。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的 query 应决定“找什么”，video 则决定“具体在哪里”；当前 query 无关 cPCA 和无直接 start/end 信号使两者职责混乱。SO-ECGN 的主观—客观关系可重新解释为 query prior—video boundary evidence，polarity/intensity 可重新解释为边界移动方向—移动幅度。

但 emotion captioning 的标签与 grounding 边界无关，不能照搬情绪词典或情感损失。需要构建时序状态原型，预测 left/right/both/no-change 与 offset magnitude，并以时序等变性、局部证据变化训练。

## C. 独立创新方案

### C1. 方案名称

**QD-DMBR（Query-Dominant Direction–Magnitude Boundary Refiner，查询主导的方向—幅度边界修正器）**。

### C2. 目标问题

主要解决没有直接 start/end refinement、宽 proposal、IoU 0.7差和 query 无关背景/边界建模。

### C3. 核心假设

当前 coarse proposal 已提供一个 query 主导的事件假设；视频局部变化更适合回答“左边界向哪移动、多远”和“右边界向哪移动、多远”。将方向分类与幅度回归解耦，并限制为增量修正，比从混合 feature 一步重预测完整区间更稳定。

### C4. 方法设计

1. **时序状态字典**：学习五个 prototype：before、start-transition、sustain、end-transition、after。prototype 由 query token 初始化，再通过当前 proposal附近的 video feature 做低秩域迁移，保持 query 为主导。
2. **局部观察**：对每个 proposal 在左右边界各取内外 `K=4` 个 50-step feature，并从 200-step pre-downsample feature 取对应高分辨率窗口。query 动词/实体对这些窗口做动态 temporal mask。
3. **方向头**：左、右边界各输出 `{outward,inward,stay}`；输入为局部窗口与五状态 prototype 的相关序列。该分支只做分类，确定修正符号。
4. **幅度头**：在与方向 feature 正交投影后的子空间回归 `|Δs|,|Δe|∈[0,0.1]`，避免幅度受方向 logit大小支配。最终 `s'=s+sign_s·|Δs|`，`e'=e+sign_e·|Δe|`。
5. **迭代**：最多两次；第二次以第一步区间为状态。若状态序列置信度低则 stop，保持原 proposal。
6. **弱监督**：时间 crop/shift 后，方向标签按几何变换等变；对人工生成的扩宽 proposal，若外壳 query evidence 低，目标方向为 inward；对人工裁掉高证据帧的 proposal，目标为 outward。只使用高 margin 样本。
7. **损失与推理**：`L=L_base+λ_dir CE+λ_mag SmoothL1+λ_orth L_orth+λ_cycle L_cycle`。训练 first stage 冻结 coarse generator；推理 refinement 后以独立 evidence score 排名。

### C5. 与原论文的区别

原论文用 subjective emotion 控制 caption 基调、objective emotion 逐词修正，并解耦 polarity/intensity。QD-DMBR 将主客体重释为 query/coarse interval 与局部视频变化，把 polarity/intensity 映射为边界移动方向/距离；新增左右边界双头、时序状态 prototype、几何等变和 proposal corruption 监督，不涉及情感生成。

### C6. 为什么可能有效

高斯 width 同时改变两侧，且 outer hull 不能精细判断哪一侧多了背景。QD-DMBR 对左右边界独立做小步修正，方向分类先决定应 trim 还是 expand，幅度再决定距离，可减少大回归误差；query-conditioned状态字典只把与目标动作相关的变化视为 start/end，直接提升 boundary perception 和高 IoU recall。

### C7. 潜在风险

- 人工 proposal corruption 的方向伪标签可能错误，特别是重复动作视频。
- 正交约束不一定真正实现方向/幅度语义解耦。
- 两步修正可能累积误差或震荡。
- 50-step局部窗口仍太粗，需要高分辨率 feature 支持。

### C8. 最小验证实验

冻结 baseline，对其 5 proposals 生成人工 expand/trim 变体，仅训练一层左右方向头；幅度固定为 0.02，不做迭代。比较原边界、oracle direction、learned direction。报告 IoU 0.5/0.7、左/右边界平均绝对误差代理（若评估标签可用）和保持不动比例。若 learned direction 能实现 oracle gain 的至少 40%、IoU 0.7 提升且 IoU 0.3不降，则再训练幅度头；若方向准确率与启发式相当，应停止。

### C9. 进一步实验 / Ablation

- 无状态字典、固定字典、query-conditioned字典。
- 一步/两步；共享左右头 vs. 独立头。
- 联合 offset 回归 vs. 方向—幅度解耦；有无正交 loss。
- 50-step与200-step局部 feature。
- corruption 类型：expand、trim、shift、内含空隙。
- 按左右边界误差类型与短/长事件分组。

## 依据

- 原论文：`TIP26-Subjective-Objective Emotion-Correlated Generation Network for Subjective Video Captioning.pdf`，重点为主客观增量融合、动态 mask、polarity/intensity 解耦。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

