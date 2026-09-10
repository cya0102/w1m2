# Video Caption 论文到 CPL_LREV 改进方案总览

## 完整性结论

- `docs/videoCaption` 中识别到 **19 篇 PDF 论文**。
- 已生成 **19 个独立 proposal 文件**；每篇论文只对应一个主方案，没有合并论文。
- 方案均以 [`cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md) 记录的实际故障为目标：NLL/IoU错位、teacher-forcing语言捷径、hard-min自确认、mask/outer边界不一致、短事件与宽度偏置、easy negatives、R@5→R@1排序失败、弱边界与共享表示冲突。

## 一一映射

| Paper | Proposal | Target CPL_LREV Issue | Core Idea | Expected Benefit |
|---|---|---|---|---|
| AAAI26 — Explicit Temporal-Semantic Modeling / CACMI | [Proposal 1：QCEC](proposal_aaai26_cacmi.md) | 200→50短事件丢失；query无关时序结构；outer gap | 高分辨率连续事件聚类、query短语条件化、变化点吸附与跨点惩罚 | 保留短事件与语义边界，减少跨无关事件的宽区间 |
| AAAI26 — OwlCap | [Proposal 2：BPSE](proposal_aaai26_owlcap.md) | NLL/IoU错位；宽proposal；hard winner；排序 | query-unit完整性与区间事件正确性双向集合等价，组内偏好训练 | 同时惩罚漏语义和多背景，把好候选提到R@1 |
| ACL26 — DualFact+ | [Proposal 3：DFCQ](proposal_acl26_dualfact_plus.md) | 语言捷径；easy negative；关系判别与质量头缺失 | action/object/tool/location概念核验 + 谓词论元时间共现核验 + 角色反事实 | 激活困难负例，拒绝“词都出现但时序关系错误”的区间 |
| ACL26 — Putting Captions to the Test / CapQuiz | [Proposal 4：TCQS](proposal_acl26_capquiz.md) | teacher forcing；盲语言先验；细粒度排序 | 从query生成视觉probes，hard options与unknown，query-only盲答对抗 | 强迫候选回答细粒度视觉问题，区分缺证据与错误证据 |
| PR26 — Ask and Focus More / QPDC | [Proposal 5：EACA](proposal_pr26_qpdc.md) | 宽区间；共享表示冲突；固定loss与伪权重 | action/context双分支，Dirichlet证据分配，不确定性门控 | 让动作决定边界、上下文只确认语义，提供可用置信度 |
| PR26 — Dual-Hierarchical Knowledge Distillation | [Proposal 6：HSBD](proposal_pr26_capdistill.md) | 高分辨率边界丢失；短事件；无start/end | 200-step object/action teacher向50-step student蒸馏frame与boundary分布 | 推理不增成本地恢复短时序和直接边界感知 |
| PR26 — PR-DETR | [Proposal 7：EARS](proposal_pr26_pr_detr.md) | 5 proposal覆盖不足、尺度偏宽、高重叠/坍缩 | 无标签尺度锚点、持续anchor注入、几何+语义relation slots | 保证短尺度槽并形成有功能的proposal分工 |
| TCSVT26 文件（正文：Visual Evidence-Aware Hallucination Rectification） | [Proposal 8：VDRR](proposal_tcsvt26_visual_evidence_rectification.md) | teacher-forcing语言捷径；NLL排序 | full visual与language-only token概率差 + action/object unsupported诊断 | 过滤“会说但没看见”的低NLL错误候选 |
| TIP26 — RefZVC | [Proposal 9：TTBP](proposal_tip26_refzvc.md) | R@5→R@1落差；边界无精修；推理不用LREV | 冻结主模型，shift/trim/expand测试时搜索，整句+unit+shell奖励和RBF记忆 | 从已有候选局部找更紧区间，训练/推理质量信号一致 |
| TIP26 — SG-FSCFormer | [Proposal 10：QSTG](proposal_tip26_sg_fscformer.md) | 粗粒度query-video交互；组件无语义分工；背景污染 | query phrase graph剪裁temporal subgraph，many-to-many phrase-frame绑定 | 要求动作、实体、关系在连续子图中共同成立 |
| TIP26 — SO-ECGN | [Proposal 11：QD-DMBR](proposal_tip26_so_ecgn.md) | 无左右边界修正；IoU 0.7差；query无关边界 | query主导的时序状态原型；left/right方向与幅度解耦的小步修正 | 独立修正两侧边界，避免共享width的一刀切 |
| TIP26 — SynPO | [Proposal 12：SPPO](proposal_tip26_synpo.md) | hard-min自确认；优化漂移；排序失败 | 多维可靠proposal pair，几何平均困难单元 + preferred绝对质量保护 | 学会候选偏好而不通过普遍压低分数制造margin |
| arXiv26 — Reverse Sign Language Dictionary | [Proposal 13：DTRPS](proposal_arxiv26_reverse_sign_dictionary.md) | teacher forcing；缺独立语义ranker | 候选仅凭视觉生成短事件描述，再由冻结语义模型检索query | 切断query前缀泄漏，提供可解释、开放语义的排序 |
| arXiv26 — AVSCap | [Proposal 14：SBSB](proposal_arxiv26_avscap.md) | 语义/边界共享冲突；outer hull；宽背景 | 语义anchor与transition anchor先解耦，再做局部start-event-end绑定 | 只有语义与成对边界协同的连续事件可获高分 |
| arXiv26 — ProCap | [Proposal 15：PCMI](proposal_arxiv26_procap.md) | 过宽/过窄；关键单元遗漏；无参边界校正 | query语义×持续×动态显著性，迭代expand+trim求完备最小区间 | 在不重训下兼顾semantic completeness与compactness |
| arXiv26 — RefCaptioner | [Proposal 16：PCB-GM](proposal_arxiv26_refcaptioner.md) | Gaussian组件无语义；mask/outer边界不一致；gap | phrase-component绑定、distractor gate与最大可合并连续链解码 | 无关远端组件不再把outer boundary拉宽 |
| arXiv26 — Seeing Before Synthesizing | [Proposal 17：NTCCD](proposal_arxiv26_sbs.md) | 弱边界；内部语义gap；短事件 | 冻结VLM帧叙事变化、自适应高置信gate、query方向验证 | 以独立语义转移校准start/end并阻止跨gap合并 |
| arXiv26 — VEGAS | [Proposal 18：CFIS](proposal_arxiv26_vegas.md) | 宽proposal偏好；背景污染；NLL排序 | 比较完整候选、连续关注核心与language-only的条件信息增益 | 衡量核心充分性和多余上下文依赖，支持trim与rerank |
| arXiv26 — Watching Synthetic Videos / WSV | [Proposal 19：SVEBS](proposal_arxiv26_watching_synthetic_videos.md) | 无直接边界/IoU监督；短尺度不足；自举 | T2V合成事件嵌入背景，polisher对齐真实feature，已知插入边界强监督 | 给boundary与quality heads提供独立显式监督并纠正尺度偏置 |

## 建议验证顺序

这不是合并方案，而是按成本/诊断价值给出的执行优先级：

1. **低成本推理/冻结 rerank**：Proposal 18 CFIS、15 PCMI、9 TTBP、8 VDRR。它们可先检验“现有5个候选是否主要是排序与边界校准问题”。
2. **中成本独立质量头**：Proposal 2 BPSE、3 DFCQ、12 SPPO、16 PCB-GM。它们直接替换 NLL 自举排序，且可以先冻结 backbone。
3. **结构性 proposal 改造**：Proposal 1 QCEC、6 HSBD、7 EARS、10 QSTG、11 QD-DMBR、14 SBSB、17 NTCCD。
4. **高成本新数据/生成路线**：Proposal 13 DTRPS、19 SVEBS；前者受视觉描述器瓶颈约束，后者需T2V离线生成与域对齐。

## 文件计数规则

`proposal_summary.md` 是索引，不计入方案数；除它以外，以 `proposal_*.md` 命名的独立方案文件应恰为 **19**。

