# Proposal 8：VDRR——视觉依赖诊断的重建分数校正

**对应文件**：`TCSVT26-Text-Conditional Visual-Language Alignment for Video Captioning.pdf`。**注意**：PDF 正文标题实际为 *Visual Evidence-Aware for Object Hallucinations Rectification in LLM-Based Video Captioning*；本方案以正文方法为准。

## A. 论文核心方法与有效机制

该论文处理 LLM video captioning 中 object/action hallucination。它选择与文本相关的关键帧，构造 global、object、action 及 scene-graph 视觉知识；对每个生成 token，诊断模型结合视频、前缀和 top-k候选词，分类为无幻觉、object hallucination 或 action hallucination。训练负例通过截断 noun/verb 前缀、替换成视觉上相似或共现上合理的错误词构造，并以视觉相似度做 label smoothing。推理时在候选 token 中选择幻觉概率最低者。

有效机制在于把生成概率与“该 token 是否真正依赖视觉证据”分离：语言上很合理的词不再自动可信；object/action 分型和结构证据使校正具有针对性。

## B. 与 CPL_LREV 问题的对应关系

CPL_LREV 用 teacher-forced reconstruction NLL 给 proposal 排名，但常见 query 的后续 token 可由语言前缀预测，导致视频区间错误仍有低 NLL。论文的 token-level hallucination diagnosis 正好可识别这种语言支持强、视觉支持弱的重建。

captioning 原方法修改生成 token；grounding 不应篡改用户 query，也不需要 scene graph 全套系统。需要把诊断结果改造成候选区间的视觉依赖度与校准分数，并用 corrupted query 训练，而不是生成纠错文本。

## C. 独立创新方案

### C1. 方案名称

**VDRR（Visual-Dependency Reconstruction Rectifier，视觉依赖重建校正器）**。

### C2. 目标问题

主要解决 teacher forcing 语言捷径、NLL/IoU 错位和 raw NLL 排序偏差；次要增强 action/object 语义判别。

### C3. 核心假设

正确 proposal 对关键 noun/verb token 的预测应显著依赖该区间视觉信息，而错误 proposal 的低 NLL 更多来自语言上下文。比较完整视觉条件与语言-only/视觉打乱条件的 token 概率，并由独立 hallucination head 校准，可滤除“会说但没看见”的候选。

### C4. 方法设计

1. **双路前向**：对每个 proposal 计算正常 token logits `ℓ^V_{n,l}`；再用相同 decoder、零化或 batch-shuffle proposal visual summary 得到 `ℓ^L_{n,l}`。两路均 teacher forcing，但参数共享。
2. **视觉依赖量**：对 GT query token 定义 `d_{n,l}=log p(t_l|prefix,V_n)-log p(t_l|prefix,∅)`；只对 noun、verb、relation token 聚合，stop-word 权重低。
3. **诊断头**：输入 `[decoder state, proposal visual token, top-k token embeddings,d]`，输出 supported/object-unsupported/action-unsupported。正例来自原 query；负例将 noun/verb替换为同 batch 高频、语义相近但与视频不匹配的词，并用视觉文本相似度软化标签。
4. **候选校准分数**：`S_n=-NLL_n+λ_d mean_l d_{n,l}-λ_h P(unsupported)`；另要求该分数在 video shuffle 后下降。推理不改变 query，只用该分数重排。
5. **防止退化**：语言-only 分支完全停止视觉梯度；依赖差不回传到语言-only logits，避免模型故意降低基线概率。诊断头不参与 Gaussian parameter 梯度的首阶段。
6. **结构证据轻量化**：不引入检测器，只为 noun 与 verb 分别从现有 frame feature 取 top-k token attention，作为 object/action evidence。
7. **训练顺序**：先冻结 CPL 训练诊断头；验证有效后再让 `λ_d` 的排序 loss 微调 proposal generator。现有 LREV 可关闭，避免两个自举权重叠加。

### C5. 与原论文的区别

原论文诊断并替换 caption 生成过程中的幻觉 token，依赖多种结构视觉知识。VDRR 将同一 insight 用于“固定 query 对候选区间是否有视觉依赖”的诊断：不改用户文本、不做 token replacement，输出的是 proposal calibration；结构证据由 CPL feature 的 noun/verb attention 提供，并加入 full-vs-language-only 的显式因果对照。

### C6. 为什么可能有效

错误区间也可借 teacher forcing 预测常见短语，所以 raw NLL 低；但拿掉它的视觉 summary 后概率变化很小，即 `d≈0`。正确区间中关键动作/实体确有视觉贡献，依赖差更大。将 token-level 差异聚合成候选分数，会减少 language shortcut 对 ranking 的影响，并使重建 loss 更关注真正可视的 discriminative tokens。

### C7. 潜在风险

- 当前 decoder 本就可能忽略视觉，所有候选 `d` 都接近零。
- 零化视觉会造成分布外输入；batch shuffle 是更稳健但更昂贵的对照。
- noun/verb替换可能仍与视频匹配，负例噪声高。
- 双路 decoder 训练成本接近翻倍；推理可缓存语言-only 一次而非每 proposal 重算。

### C8. 最小验证实验

无需训练新模块：在现有 checkpoint 上测 full、zero-visual、batch-shuffle 三种 NLL，计算关键 token 依赖差并与 proposal IoU 的 Spearman 相关；再网格搜索 `λ_d` 重排 5 candidates。若 `d` 与 IoU 显著正相关且 R@1@0.5 提升 ≥1.5 点，进入诊断头训练；若不同候选的 `d` 方差极小或仅句长相关，则该方向应放弃。

### C9. 进一步实验 / Ablation

- zero、shuffle、区间外 visual 三种反事实。
- 全 token、content token、noun/verb 分型聚合。
- 仅依赖差、仅诊断头、二者联合。
- top-k corruption 的频率、语义近邻和同视频实体。
- 推理分数与 NLL 相加、乘法、温度校准。
- 检查视觉打乱敏感度与 IoU，不只看最终 recall。

## 依据

- 原论文：`TCSVT26-Text-Conditional Visual-Language Alignment for Video Captioning.pdf`（正文题名与文件名不一致），重点为 token 幻觉诊断、合成负 token 与纠错。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

