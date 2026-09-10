# Proposal 19：SVEBS——合成视觉事件的显式边界监督

**对应论文**：*Watching Synthetic Videos: Aligning Cross-modal Representations with Visual Synthesis for Zero-shot Video Captioning*（WSV）。

## A. 论文核心方法与有效机制

WSV 解决 text-only训练、real-video推理之间的模态鸿沟。它不用文本 feature经简单线性映射伪装视觉，而是以 text-to-video模型直接生成4D时空 latent；轻量3D residual CNN polisher在保持时空结构的同时修正合成—真实分布差异，并由冻结 video-text encoder的对称对比 loss监督。第二阶段，3D CNN把 polished latent转成序列，learnable queries经 cross/self-attention形成 soft prompts供GPT-2生成 caption；推理时真实视频经同一T2V VAE encoder进入prompter。论文消融显示 synthetic visual、polisher和prompter均有独立贡献，3D CNN polisher优于破坏局部结构的Transformer版本。

有效机制是让训练样本真正具有视觉时空结构，并用轻量结构保持的 polisher校准合成域，而不是只在文本空间做对齐。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV最根本的弱点是无直接 start/end/IoU监督，所有 proposal质量由 reconstruction自举；WSV提示可以从文本生成视觉事件。若把 query生成的合成事件嵌入已知背景位置，就能人为构造拥有精确边界的训练序列，为 boundary/quality head提供不依赖NLL的监督。

但原论文的目标是零样本caption，整段synthetic video本身即正样本；grounding需要长视频中的 event-vs-context结构。不能直接用生成视频替换真实数据，也不能假设合成latent与C3D feature一致，必须有 feature-level polisher、背景拼接与real-data保持训练。

## C. 独立创新方案

### C1. 方案名称

**SVEBS（Synthetic Visual Event Boundary Supervision，合成视觉事件边界监督）**。

### C2. 目标问题

主要解决缺乏直接边界/IoU训练信号、hard-min自举、短尺度proposal缺失和quality head无监督。

### C3. 核心假设

尽管合成视频不等同真实视频，把query生成的事件latent放入随机上下文后，其插入位置提供无歧义的 start/end标签。经结构保持的polisher缩小合成—真实feature差异，并只用这些样本预训练边界感知/质量排序，可向弱监督真实数据注入“区间应该如何结束”的归纳偏置。

### C4. 方法设计

1. **离线事件生成**：从训练query抽样5k–20k条，规范化为action-object prompt；T2V只保存latent或低分辨率clip，不必全质量解码。生成2–6秒事件并记录有效帧长度。
2. **合成长序列**：从其他query生成或真实训练视频无关片段取左右背景，构成 `[bg_L,event,bg_R]`；随机插入位置、速度、持续时间，精确记录 `(s*,e*)`。另构造同场景异动作与同动作异实体困难背景。
3. **feature接口**：用与CPL兼容的视频encoder得到synthetic feature；`SyntheticPolisher`为2层 temporal/3D residual conv，输出维度500，与 `frame_fc`输入一致。冻结video-text encoder做对称contrastive，另用MMD/CORAL对齐真实训练feature的均值协方差；不需要真实边界。
4. **显式 heads**：在CPL `h`上增加 start/end distribution head和 proposal IoU-quality head。合成样本用CE监督 `p_s,p_e`，用真实 `(s*,e*)`计算每个候选IoU回归/排序；强制包含短duration桶。
5. **proposal预训练**：先仅在synthetic composite上训练 Gaussian centers/widths、boundary和quality head，loss包含 `L_s+L_e+L_IoU+L_diversity`；reconstruction仅作辅助。
6. **真实数据联合**：随后混合80%真实弱监督+20%合成强监督；真实样本沿用原loss并加增强一致性，合成样本保持直接边界loss。用domain-specific LayerNorm避免synthetic统计污染。
7. **推理**：完全不生成视频；只保留已训练的 boundary/quality heads，以quality而非NLL排序，额外计算很小。
8. **泄漏控制**：只使用训练query生成，生成seed与背景拆分固定；验证/测试query不进入T2V。记录生成失败并以独立CLIP阈值过滤。

### C5. 与原论文的区别

WSV用text-to-video latent让captioner在text-only条件下接触视觉分布，polisher连接synthetic训练与real inference。SVEBS进一步把合成事件嵌入背景，利用“插入位置”创造grounding专属的start/end和IoU监督；polisher对齐到CPL feature而非GPT soft prompt，训练对象是proposal generator/quality head，推理不做caption生成。

### C6. 为什么可能有效

当前NLL无法区分能重建query的宽/窄区间，且pseudo label来自自身。合成composite首次提供独立、精确的边界和proposal IoU排序，能教会width、start/end与quality head的基本几何—语义关系；专门过采样短事件可纠正现有width分布。polisher和真实数据联合训练减轻domain gap，使这些归纳偏置有机会迁移，而不要求合成像素完全逼真。

### C7. 潜在风险

- T2V昂贵，生成事件可能不忠实query或边界有淡入淡出伪线索。
- 模型可能识别synthetic拼接接缝，而非真实语义边界。
- feature-level domain alignment不保证条件分布对齐，负迁移风险高。
- 混合强/弱监督梯度尺度差大，可能覆盖真实数据学习。
- 方案工程量是19个方案中较大的，应严格先做小样本验证。

### C8. 最小验证实验

生成约1k个训练query事件，每个拼接3种背景，只训练新增start/end与quality heads，冻结CPL encoder和原proposal generator；与随机初始化heads、无polisher、仅合成预训练比较。真实验证集报告R@1/R@5 IoU、短段指标、quality-IoU Spearman；同时训练一个domain classifier检查feature gap。若真实R@1@0.5提升、短段提升且domain classifier准确率因polisher明显下降，扩大数据；若只在synthetic validation有效或模型依赖接缝，立即放弃像素拼接，改做latent平滑。

### C9. 进一步实验 / Ablation

- text feature伪视觉、原synthetic latent、polished latent。
- 真实/合成背景，随机接缝与cross-fade，检测shortcut。
- 仅boundary、仅IoU quality、两者联合。
- 合成数据1k/5k/20k与混合比例5/20/50%。
- duration均匀采样 vs.短事件过采样。
- polisher的contrastive、MMD/CORAL及domain-specific normalization。
- synthetic预训练后冻结 vs.真实数据联合微调。

## 依据

- 原论文：`arXiv26-Watching Synthetic Videos- Aligning Cross-modal Representations with Visual Synthesis for Zero-shot Video Captioning.pdf`，重点为synthetic visual latent、3D CNN polisher、prompter与消融。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

