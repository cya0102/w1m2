# Proposal 18：CFIS——关注核心充分性的条件信息增益排序

**对应论文**：*VEGAS: Human-Aligned Video Caption Evaluation via Gaze*。

## A. 论文核心方法与有效机制

VEGAS 认为同一视频可有多种合理 caption，人的 gaze提供个体关注意图。它在测试时把视频分为 gaze-attended区域 `G` 和非关注补集 `Ḡ`，用冻结 VLM计算 caption在完整视频与仅关注区域下的逐 token概率比：`log P(t|G,Ḡ)/P(t|G)`。该值是条件点互信息的近似，越低表示非关注区域对 caption几乎没有额外贡献，即 caption主要由 gaze区域充分解释。论文在候选 caption中做 rejection sampling，无需训练；token分解还可指出哪些概念依赖 gaze外内容。正确 gaze腐化消融和检索实验支持它捕获了个体关注信息，但论文也指出抽象描述、gaze缺失与中心偏差是限制。

有效机制不是普通 saliency overlap，而是**反事实充分性**：比较“关注核心”与“核心+其余区域”对同一文字的条件预测增益，直接测多余视觉上下文是否必要。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 的宽区间因包含更多上下文而 NLL更低，outer背景又由当前 proposal自己定义。VEGAS的 full-vs-attended条件比可把候选内部再分成 query关注核心与其余上下文，测后者是否真正为 query提供额外信息；理想紧致区间的核心应已充分。

grounding数据没有人眼 gaze，不能声称存在真实注意轨迹。必须用独立、query-conditioned frame evidence构造“隐式关注核心”，并防止它与当前 proposal/NLL同源形成闭环；同时低信息增益可能来自模型完全不看视频，因此还要加核心本身的绝对支持下限。

## C. 独立创新方案

### C1. 方案名称

**CFIS（Counterfactual Focus-Information Sufficiency，反事实关注信息充分性）**。

### C2. 目标问题

主要解决宽 proposal偏好、背景污染、raw NLL排序失败和当前 proposal定义自身背景的循环。

### C3. 核心假设

正确且紧致的候选中，query相关核心帧应足以解释关键 token，候选其余帧带来的条件信息增益很小；错误或过宽候选要么核心本身支持弱，要么依赖额外上下文才能重建。联合“核心绝对支持”与“补集条件增益”可区分这两种情况。

### C4. 方法设计

1. **独立 focus map**：用冻结 vision-language encoder或独立轻量 token-frame matcher，从原始 frame feature与 query content tokens产生 `a_t`。它不读取 Gaussian mask、reconstruction logits或 LREV权重。
2. **关注核心 `G_n`**：在每个 proposal内部，选择累计 attention mass达到 `ρ=0.7` 的最小连续窗口；完整候选为 `V_n`，内部补集为 `V_n\G_n`。连续约束避免零散 top-k伪造“核心”。
3. **三次条件重建**：共享冻结/校准 decoder计算 `logP(q|V_n)`、`logP(q|G_n)`、`logP(q|∅)`。定义 token级 `IG_{n,l}=logP(q_l|V_n)-logP(q_l|G_n)`；越小表示补集贡献少。
4. **充分性与必要性**：`Suff_n=mean_content logP(q_l|G_n)-logP(q_l|∅)`保证核心确有视觉支持；`Excess_n=mean relu(IG_{n,l})`测多余上下文贡献。最终 `q_n=Suff_n-λExcess_n-μ|V_n\G_n|`。
5. **边界修正**：若去除某一侧帧不降低 `Suff` 且降低 `Excess`，向内 trim；若核心贴边且相邻 shell有高独立 focus，则允许小幅 expand。最多一次，防止迭代投机。
6. **训练选择**：第一版纯推理 rerank；训练版让 quality head拟合停止梯度的 `Suff/Excess`，并用随机 focus、中心 bias focus、跨视频focus做腐化负例，要求真实 query focus更优。
7. **效率**：语言-only概率每个 query只算一次；`G_n`与`V_n`共10次短 decoder forward，可离线缓存或蒸馏到一层 head。

### C5. 与原论文的区别

VEGAS 使用真实个体 gaze分割空间—时间区域，评价/选择候选 caption。CFIS 用独立 query-frame matcher构造候选内部的连续时间 focus，评价/选择候选 interval；加入 `P(q|∅)`必要性基线和边界 trim/expand，防止“模型不看视频也得到低VEGAS”这一 grounding退化。输出不是个性化 caption，而是紧致边界。

### C6. 为什么可能有效

raw NLL只奖励完整候选能否重建 query，天然偏爱更多上下文。CFIS问的是更强的问题：核心是否已足够、补集是否仍提供大量关键 token信息。过宽区间通常有较大可删除补集；错误区间核心相对语言-only无增益，不能靠低 `Excess`冒充好候选。token分解还能定位是 action还是entity在依赖背景，改善 semantic discrimination 与 foreground/background分离。

### C7. 潜在风险

- 隐式 focus不是人眼 gaze，若 matcher与当前模型偏差相同，独立性不足。
- decoder忽略视觉时 `Suff`和`Excess`均小，分数退化。
- 多次 decoder前向推理较慢。
- 对需要长程上下文的复合 query，补集不是冗余，trim可能有害。

### C8. 最小验证实验

在现有 checkpoint 上，用 frozen CLIP式 frame-query相似度构造连续核心，计算 full/core/language-only NLL，不训练。比较 raw NLL、仅 `Excess`、`Suff-Excess` rerank；同时做 random/center focus腐化。若真实 focus分数优于腐化、`Suff-Excess`与IoU相关性高于NLL且 R@1@0.5提升 ≥1.5点，则蒸馏；若语言-only与core概率几乎相同，说明 decoder视觉依赖不足，应先解决 Proposal 8类问题。

### C9. 进一步实验 / Ablation

- 独立 encoder、当前 h、随机/中心 focus。
- `ρ∈{0.5,0.7,0.9}`；连续最小窗 vs.零散top-k。
- full/core、core/empty两项分别与联合。
- token平均、content token、按action/entity分型。
- 仅rerank、边界trim、distilled quality head。
- 按query复杂度和GT时长分析补集是否真正冗余。

## 依据

- 原论文：`arXiv26-VEGAS- Human-Aligned Video Caption Evaluation via Gaze.pdf`，重点为条件点互信息定义、token分解、rejection sampling与 gaze corruption。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

