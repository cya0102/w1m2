# Proposal 13：DTRPS——先描述后检索的提议选择器

**对应论文**：*A Reverse Sign Language Dictionary: Open-Vocabulary Sign Recognition from Continuous Signing via Video Captioning and Description Retrieval*。

## A. 论文核心方法与有效机制

论文把连续手语中的开放词汇识别拆成两步：LVLM 先把 sign clip 转成细粒度 articulation/procedural description，再用冻结 multilingual sentence encoder 在文字描述词典中检索目标 sign。视觉 tower 仅做轻量 LoRA 适配。实验指出 matcher 的上界接近饱和，主要瓶颈在描述是否忠实；模块化设计能处理未在训练分类头中出现的新词，但也暴露生成描述塌缩的风险。

核心 insight 是把难以直接对齐的视觉与标签，通过一个可解释的视觉事件描述作为中间接口，再由稳定、冻结的语义检索器完成判别；检索与生成职责分离。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 使用 teacher-forced NLL 判 proposal，语言捷径强；但若要求候选仅凭视觉先自由生成一个短事件描述，再与 query 检索匹配，错误区间无法利用 query 前缀“抄答案”。冻结 matcher 也可成为独立 ranking signal。

差异是原任务检索有限但开放扩展的 sign dictionary，grounding 每个样本只有一个 query 和多个时段候选。需要构造 query 描述/改写和同视频反事实描述库，并控制自由生成的计算与幻觉。

## C. 独立创新方案

### C1. 方案名称

**DTRPS（Describe-Then-Retrieve Proposal Selector，先描述后检索提议选择器）**。

### C2. 目标问题

主要解决 teacher forcing shortcut、raw NLL 排序偏差和缺少独立语义质量头。

### C3. 核心假设

正确区间仅从视觉生成的短描述应能在冻结语义空间中检索到 query；错误区间即使在 teacher forcing 下 NLL低，也会生成其真实事件或空描述，从而与 query 拉开。描述中间层还能暴露候选为何得分高。

### C4. 方法设计

1. **视觉描述器**：复用现有 reconstruction decoder，但训练/推理时不给 query token 前缀；输入仅为 proposal visual summary与通用 prompt token，生成 3–12词的 action-object phrase。第一版用 greedy，避免 beam成本。
2. **目标描述库**：为 query 建立 `D^+`：原句、动词—宾语规范化、1–3个离线 paraphrase；`D^-` 来自同视频其他 query、交换动词/实体的反事实和 `no relevant event`。
3. **冻结检索器**：使用冻结 sentence encoder 编码生成描述与 `D`，分数为对正库最大相似度减对 hard-negative 库最大相似度。matcher 绝不接收 proposal NLL。
4. **空证据机制**：描述器额外预测 `<unsupported>`；视频 shuffle、纯 outside mask、极低视觉依赖的候选以该 token 监督，避免每个区间都强行说一个常见动作。
5. **训练**：阶段一冻结 CPL，为现有 soft proposals 训练描述器，正样本责任由多视图一致且视觉依赖高的 proposals 提供，不用 raw min NLL单独选择；阶段二仅训练视觉 adapter，使描述 retrieval 对正确 query 高于反事实。
6. **ranking**：`q_n=sim(desc_n,D^+)-sim(desc_n,D^-)-λP(<unsupported>)`；推理以 `q_n` 排 5 candidates，可与边界 compactness 轻量融合。
7. **防止语言模板塌缩**：batch 内对比、描述多样性熵下限和 video-shuffle consistency；冻结 matcher，描述器 LoRA/小 decoder而非全模型微调。

### C5. 与原论文的区别

原论文把 sign clip 描述后检索固定词典，实现开放词汇识别。DTRPS 把每个 temporal proposal 描述后检索当前 query 的 paraphrase set；候选类别不是词典词，而是“哪个时间段与 query等价”。新增 `<unsupported>`、同视频反事实和 proposal ranking，不直接复现 sign 特征或多语 matcher 设定。

### C6. 为什么可能有效

query 不作为生成前缀，切断 teacher forcing 的语言信息泄漏。只有区间视觉能驱动描述，而冻结 matcher 把自然语言表述差异吸收掉；因此好候选的 score 来自视觉可描述性，坏候选无法靠包含更宽上下文自动获得低 NLL。该方案直接改变 query-video interaction 的方向：先 `video→description`，再 `description→query`。

### C7. 潜在风险

- 当前 decoder 可能无法无 query 条件生成可靠描述，形成常见动作塌缩。
- 描述错误会成为新瓶颈，matcher 再强也无济于事。
- 每 proposal 自回归生成增加推理延迟。
- paraphrase/反事实库质量会影响检索边界。

### C8. 最小验证实验

无需重新训练：用现有 decoder 对 top-5 proposal 做 greedy free-running（只给 BOS 与 visual summary），以冻结 text encoder 对生成文本与 query 相似度 rerank；同时记录生成多样性与 video-shuffle变化。若 R@1 提升、不同 proposals 生成文本有区分且 shuffle 后 similarity下降，才训练专用描述器；若 >70%描述相同或空泛，则这一中间接口不可行。

### C9. 进一步实验 / Ablation

- 自由文本、结构化 action-object slots、两者联合。
- 原 query 单参考 vs. paraphrase set；有无反事实库。
- `<unsupported>` 与强制生成的比较。
- 冻结/LoRA/full fine-tune 描述器；冻结 matcher选择。
- greedy、一次性 non-autoregressive phrase head、beam。
- 生成忠实度与最终 grounding 指标的误差传导分析。

## 依据

- 原论文：`arXiv26-A Reverse Sign Language Dictionary- Open-Vocabulary Sign Recognition from Continuous Signing via Video Captioning and Description Retrieval.pdf`，重点为 description generation→frozen retrieval 的模块化流程。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

