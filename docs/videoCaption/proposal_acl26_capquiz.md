# Proposal 4：TCQS——反语言捷径的时序问答式排序器

**对应论文**：*Putting Captions to the Test: Evaluating Video Caption Quality through Multiple-Choice Question Answering*（CapQuiz）。

## A. 论文核心方法与有效机制

CapQuiz 指出传统 caption metric 很难同时测 factual precision 与 semantic coverage。其做法是从视频生成经人工核验的多类型细粒度选择题，覆盖描述性与推理性信息；通过 blind-solvability 检查过滤“只看问题就能答”的题，构造困难干扰项并提供 unknown 选项。以 caption 为唯一证据回答：答对为 TP，选择 unknown 为 FN，答错为 FP，由此得到类似 precision、recall 和 F1 的质量指标。

真正有效的机制有三点：把整体相似度拆成可验证 probes；用盲答测试排除语言先验；明确区分“不知道”与“自信答错”，因此不会用冗长但错误的信息换取更高覆盖率。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的 query decoder 可能只靠 teacher-forced 前缀重建 query；easy outside negative 又无法检验细粒度视频依赖。CapQuiz 的 blind-solvability 对应“这个判断是否不看视频也能完成”，hard distractor 对应邻近且语义相似的错误区间，unknown 对应区间证据不足而非强迫模型匹配。

原论文用问题评价生成 caption，并有人工核验答案；grounding 无额外问答标签。必须从 query 自动产生 probes，把候选区间/帧证据作为回答依据，并以视频内反事实构造选项，而非调用 QA metric 直接评分。

## C. 独立创新方案

### C1. 方案名称

**TCQS（Temporal CapQuiz Selector，时序问答式候选选择器）**。

### C2. 目标问题

主要解决 teacher-forcing 语言捷径、负样本过易、语义细粒度不足和 proposal 排序失败。

### C3. 核心假设

如果候选区间真的对应 query，它应能支持一组围绕 who/what/action/relation/order 的小问题；相邻错误区间会在至少一个 probe 上选择错误答案或 unknown。要求 visual branch 通过 probes，而 query-only branch 不能通过，可强迫排序器利用视频证据。

### C4. 方法设计

1. **Probe 构造**：从 query 规则化生成 2–5 个 cloze probe，例如“谁执行动作？”“执行了什么动作？”“作用于什么物体？”“先后关系是什么？”。正确选项来自 query 对应槽位。
2. **干扰项**：从同视频其他 query、同 batch 近邻语义以及当前区间外高激活帧检索 2–3 个同类型答案；另加 `insufficient evidence`。干扰项必须与正确项词性一致并具有较高文本相似度。
3. **区间 QA 头**：对每个 proposal 和每个 probe，将 probe token 作为 query，对 mask 内 `h_t` cross-attend，输出选项分布 `p(a|segment,probe)`。正确项概率形成 coverage；错误非 unknown 概率形成 factual-error cost。
4. **盲答对抗**：复制一个只输入 query/probe、不输入 video 的 blind head。对可被 blind head 高置信答对的 probe 降低训练权重；同时对共享表示做 gradient reversal，使 segment QA 的判别信息不能只来自问题文字。
5. **质量定义**：`TP_n=Σ p(correct)`，`FP_n=Σ p(wrong non-unknown)`，`FN_n=Σ p(unknown)`；计算平滑 `Q_n=F1(TP/(TP+FP),TP/L)`。再由校准层结合 shell 上相同 probes 的答案差得到最终 `q_n`。
6. **监督**：正视频-query 对的 query 槽位为答案；视频 shuffle、时间 shift 及同视频其他 query 构成负证据，此时目标是 unknown，而不是任意错误选项。对 base proposals 使用软责任，不用 hard-min NLL 选赢家。
7. **推理**：只对 5 个 proposals × 少量 probes 计算 QA；以 `q_n` 排序。原 query reconstruction 可保留为辅助 loss，但 QA 分不从 decoder logits读取。

### C5. 与原论文的区别

CapQuiz 是 reference-free caption evaluation，问题由视频产生并经过人工过滤，caption 是答题证据。TCQS 的问题由 query 产生，候选视频区间是答题证据；blind-solvability 从数据筛选标准改造成训练期 query-only 对抗；unknown 从评价标签改成“候选区间缺乏证据”的显式状态。

### C6. 为什么可能有效

语言模型可以根据前缀猜下一个词，却无法在视频 shuffle 后稳定区分正确选项与 unknown。TCQS 将每个关键语义槽位都转成局部视觉核验，避免整句 NLL 被常见词主导。相邻 hard distractor 比固定远端 background 更容易触发 margin，且 FP/FN 分离使宽区间包含错误事件与窄区间遗漏关键实体受到不同惩罚，从而改善 semantic discrimination 和 ranking。

### C7. 潜在风险

- 自动 probes 可能高度相关，实际只重复检查同一实体。
- 弱视觉 feature 无法识别细粒度选项，unknown 可能塌缩为默认答案。
- 同视频 query 作为干扰项可能语义等价，产生错误负例。
- 5×5×4 选项会增加训练显存；需共享 probe 编码并限制数量。

### C8. 最小验证实验

仅构造 action 与 object 两类 cloze，各 1 个 probe；冻结主模型并缓存 proposal features，训练轻量 QA/ranker。比较：NLL、无 blind filtering 的 QA、TCQS；额外做 video-shuffle test。成功标准是 R@1@0.5 提升 ≥2 点、shuffle 后正确选项概率显著下降、unknown 在错误区间上的 AUROC >0.65。若 blind 与 video 条件准确率接近或 unknown 全局占比 >80%，说明 probe 信号不可用。

### C9. 进一步实验 / Ablation

- 描述性 probes 与顺序/因果 probes；probe 数量 1/3/5。
- 盲题降权、gradient reversal、两者同时。
- random distractor、文本近邻、区间外视觉近邻。
- 有无 unknown；F1 式组合 vs. learned scalar。
- QA 头用 pooled segment、token-frame cross-attention、组件级特征。
- 按语言先验强弱、query 长度及短事件分层分析。

## 依据

- 原论文：`ACL26-Putting Captions to the Test- Evaluating Video Caption Quality through Multiple-Choice Question Answering.pdf`，重点为题目类型、blind-solvability、unknown 与 CapQuiz 计分。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

