# Proposal 10：QSTG——查询子图驱动的细粒度时序绑定

**对应论文**：*Scene Graph-Guided SegCaptioning Transformer With Fine-Grained Alignment for Controllable Video Segmentation and Captioning*（SG-FSCFormer）。

## A. 论文核心方法与有效机制

SG-FSCFormer 在 bounding-box prompt 控制下同时生成视频对象分割与 caption。它构造 prompt-centric temporal scene graph，Prompt-guided Temporal Graph Former 过滤与目标无关的节点/边并在相邻帧传播；Graph-guided Iterative Query Former 让 graph、visual memory、language query 反复交互。训练包含词—mask细粒度绑定 BCE 和对称 multi-entity contrastive，使多个实体、多个表达形成 many-to-many 对齐。

核心有效机制是：先用 prompt 从大场景图中裁出相关子图，再用结构关系约束视觉—语言对齐；监督不是仅在全局 pooled feature 上，而是落实到短语—视觉实体的绑定。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 主要以整句重建评价区间，缺少细词—帧对齐；cPCA query 无关，multi-Gaussian components 也没有被赋予不同语义角色。SG-FSCFormer 的 prompt-centric filtering 与 many-to-many binding 可让 query action/entity 只选择相关时序节点，并明确哪些组件/帧支持哪些短语。

但原论文有空间 box prompt 和 mask supervision；grounding 只有文本、输出是时间区间。需要把 scene graph 简化为 temporal evidence graph，用 query phrase 代替 box prompt，用弱对比和时序一致性代替像素 mask 标签。

## C. 独立创新方案

### C1. 方案名称

**QSTG（Query-Subgraph Temporal Grounder，查询子图时序定位器）**。

### C2. 目标问题

主要解决 query-video interaction 粗糙、细粒度语义判别不足、foreground/background 混淆和组件缺少语义分工。

### C3. 核心假设

目标事件不是一段全局相似视频，而是 query 的 action/entity/relation 在一组相邻时间节点上共同成立。先由 query 剪出相关 temporal subgraph，再从短语—节点绑定导出区间，可避免场景相似背景和单一实体共现造成误定位。

### C4. 方法设计

1. **temporal graph**：每 2–4 帧形成一个节点，节点 feature 为视觉 token；相邻边编码变化量，跨时边只连接 feature 相似且距离小于阈值的节点。不依赖目标检测器的最小版使用 learned latent nodes。
2. **query graph**：从 query 得到 action、entity、attribute、relation phrase nodes；保留依存边。无法解析时至少有 verb phrase 与 noun phrase。
3. **prompt adaptor**：query phrase 对 video nodes cross-attend，产生相关性 gate；仅 top-p 或 soft-gated nodes 进入子图，未选节点作为背景证据。gate 用 query/video shuffle contrastive 训练。
4. **时序图传播**：对选中节点做 2 层 message passing，相邻边传递状态持续信息，change edge 保留边界信息；query relation edge 调制 video edge attention。
5. **many-to-many binding**：预测矩阵 `B_{l,t}`。正方向要求每个必要 phrase 至少绑定一个节点；反方向要求每个前景节点可被某个 phrase 解释。加入对称 contrastive：同 batch 错 query、同视频错误时段为负。
6. **proposal 使用**：对每个 Gaussian component 计算其覆盖节点与 query phrase 的 binding；把 component summary 替换为图节点聚合，并将 `[coverage,exclusivity,relation satisfaction]` 输入 proposal quality head。
7. **区间解码**：选取同时满足 query graph 必要节点且在时间上连通的最小子图跨度；若两簇被高变化、低关系边隔开，则不能用 outer hull 合并，拆成不同 candidate。
8. **训练/推理**：先只训练 graph/binding与 quality reranker，后续再微调 mask；推理增加一次稀疏图传播，不生成 caption。

### C5. 与原论文的区别

原论文以用户 box 为 prompt，构建空间—时间 scene graph，并同时输出像素 mask/caption。QSTG 以文本短语图为 prompt，节点是时间 evidence，不需要空间 mask GT；细粒度 binding 用于决定时序 foreground、组件归属和区间连通性，而不是词—分割 mask 对齐。

### C6. 为什么可能有效

仅出现“person”或场景背景可以降低 NLL，却不能满足 action-object relation。QSTG 要求必要短语在相邻子图中共同绑定，并让所有前景节点可解释，因此同时压制过窄遗漏与过宽背景。组件得到不同 phrase binding 后，mixture diversity 由语义功能定义，不再只是 cosine/几何分离。

### C7. 潜在风险

- C3D 粒度可能不足以形成可靠 entity node。
- 自动 query graph 对口语 query 或隐含主语不稳。
- 弱监督的 binding 可能塌缩到全帧或单一峰值。
- 图模块与现有 Transformer 功能重叠，增加复杂度。

### C8. 最小验证实验

不构建显式 scene graph，只在现有 50-step `h` 上训练 phrase-frame binding matrix，并以双向 coverage/exclusivity rerank 5 proposals。固定所有 CPL 参数，对比整句相似、单向 phrase coverage、双向 binding。若 R@1@0.5 提升、视频 shuffle 后 binding score 显著下降，且背景帧平均 gate 降低，则图方向成立；若 binding 只对 noun 有效、action 无区分，应暂不增加图传播。

### C9. 进一步实验 / Ablation

- latent time nodes vs. object detector nodes；仅相邻边 vs. 相似度跨时边。
- 无 query graph、phrase nodes、完整 relation graph。
- 单向/双向 contrastive，最小连通跨度 vs. outer hull。
- graph layer 0/1/2/4，节点步长 2/4/8 帧。
- component-level binding 与 proposal-level binding。
- 对 relation-heavy、single-action、short-event query 分组。

## 依据

- 原论文：`TIP26-Scene Graph-Guided SegCaptioning Transformer With Fine-Grained Alignment for Controllable Video Segmentation and Captioning.pdf`，重点为 prompt-centric temporal graph、iterative query former 与细粒度 alignment。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

