# Proposal 17：NTCCD——叙事转移校准的连续区间解码

**对应论文**：*Seeing Before Synthesizing: VLM-Guided Transition Event Discovery for Weakly-Supervised Dense Video Captioning*（SBS）。

## A. 论文核心方法与有效机制

SBS 与本任务最接近：在弱监督 dense video captioning 中，先用冻结 VLM 为帧生成描述，形成 narrative flow；用相邻 frame-caption embedding差异检测语义转移。对两个事件间隙，它以 `mean+β·std` 建自适应阈值，由 gap内最大差异产生 sigmoid gate；高置信时将事件中心从简单中点拉向 argmax change point，并在候选 width集合中以视觉—文本对齐选择跨度。训练使用 gated attraction loss，只让高置信、高对齐的转移伪标签产生梯度；推理无需 VLM。

有效机制是用“看见后的叙事变化”提供独立于主模型 loss 的边界信号，并通过自适应 gate 控制伪标签噪声，而非把所有视觉差分峰都当真。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 无直接 start/end监督，200→50后边界弱，outer hull可能跨过内部语义 gap；其 pseudo weight 又来自自身 NLL。SBS 的离线 narrative transition可作为外部、语义化的变化先验，直接校准边界且避免闭环。

差异是 SBS 借相邻多个 event captions 的顺序关系定位事件间 transition；CPL 每个样本只有一个 query，不能依赖前后 GT caption或假设所有 change point都属于目标。必须先用 query确认哪一侧是 foreground，再用转移定义其 entry/exit，并允许强内部转移阻止 components合并。

## C. 独立创新方案

### C1. 方案名称

**NTCCD（Narrative-Transition Calibrated Contiguous Decoding，叙事转移校准连续解码）**。

### C2. 目标问题

主要解决弱边界、outer hull跨背景、短事件和高 IoU recall不足；并用独立信号替代部分 NLL伪标签。

### C3. 核心假设

冻结 VLM 的逐帧叙事变化比 C3D feature差分更接近事件语义边界；若只在 query foreground evidence与转移方向一致、且多视图稳定时使用，就能为单 query proposal提供可靠 start/end锚点，并识别 mixture内部不应跨越的语义断点。

### C4. 方法设计

1. **离线 narrative flow**：每 2–4 原始帧用冻结 VLM生成短 action-state描述；冻结 sentence encoder得到 `n_t`，缓存相邻差异 `d_t=1-cos(n_t,n_{t-1})`，训练/推理均不在线调用 VLM。
2. **自适应 transition gate**：在每个 proposal左右邻域和内部计算局部 `μ+βσ`；`g_t=σ((d_t-threshold)/τ)`。仅保留跨两种帧采样/描述 prompt都稳定的峰。
3. **query方向验证**：计算 narrative与 query的相似 track `r_t`。合法 start需满足 `r`从低到高、合法 end从高到低；镜头变化但两侧都与 query无关时忽略。
4. **边界校准**：对 current `(s,e)`，在 `±0.1` 范围寻找最高合法 start/end；新中心在原值和转移点之间按 `g`加权，width从小型集合中选择，使内部 query alignment高、shell低。
5. **连续解码**：若 outer hull内部存在高置信 transition且 `r`显著下降，则在该点切开 component chain；选择 query coverage最高的连续子链，不允许跨 gap。
6. **训练 loss**：对高置信边界加 gated attraction `g·|s-s*|+g·|e-e*|`；低置信样本不产生边界 loss。另用 transition map蒸馏一个轻量 head，使 inference可选择是否只用主模型。
7. **推理**：方案 A 使用缓存 narrative map做校准；方案 B 使用蒸馏 head完全去掉 VLM缓存。排序加入 calibrated inside-shell contrast。

### C5. 与原论文的区别

SBS 在多个已排序事件之间发现 transition，并以其修正 dense caption proposals。NTCCD 对单个自然语言 query分别验证 transition方向，形成 start/end pair；新增内部断点切链和 shell exclusivity，专门解决 CPL mixture outer hull；监督对象也是现有 Gaussian boundary而非 caption event query。

### C6. 为什么可能有效

视觉差分会响应相机运动，而 narrative差异强调“发生的动作/状态是否改变”。query方向验证进一步去掉无关转移；高置信门控防止所有伪边界进入训练。内部断点规则直接消除 outer hull包含无关 gap，边界吸附则利用200帧之外的原始采样精度，有望显著提升短段和 IoU 0.7。

### C7. 潜在风险

- VLM frame caption会幻觉，连续帧描述措辞变化也可产生假峰。
- 离线生成和缓存成本高，且不同 prompt稳定性有限。
- 缓慢事件无明显 narrative transition。
- 数据若不可在 inference预处理，必须依赖蒸馏 head，收益可能缩水。

### C8. 最小验证实验

随机选 10%训练/验证视频生成低频 narrative；不训练模型，只以 transition map后处理原 top-5 boundaries。与 raw visual差分、无方向验证、NTCCD-lite比较，报告 R@1/R@5 IoU阈值、短段 mIoU与 transition距GT边界分布。若 query验证后的 transition比视觉差分更接近GT、IoU 0.7提升且校准覆盖率足够（>30%样本），再做全量缓存；若峰与GT无相关性则放弃。

### C9. 进一步实验 / Ablation

- frame feature变化、VLM narrative变化、二者融合。
- 固定阈值、自适应阈值、跨prompt稳定 gate。
- 只校准中心、只选width、内部切链、完整方案。
- 缓存先验 vs. 蒸馏 head；不同VLM/描述频率。
- query方向验证与 shell exclusivity。
- 按转场/无转场、短/长、静态/动态事件分组。

## 依据

- 原论文：`arXiv26-Seeing Before Synthesizing- VLM-Guided Transition Event Discovery for Weakly-Supervised Dense Video Captioning.pdf`，重点为 narrative flow、adaptive gate、center/width calibration 与 gated attraction。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

