# Proposal 1：QCEC——查询条件的连贯事件簇先验

**对应论文**：*Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction*（下称 CACMI）。

## A. 论文核心方法与有效机制

CACMI 处理 dense video captioning 中“事件提议被切碎、跨模态语义不足”的问题。其关键 insight 不是简单增加注意力，而是先把连续帧组织成具有时间连贯性的显式事件单元，再让视频与检索文本在事件上下文中交互。

- **Contextual Feature Aggregation（CFA）**：使用带时间约束的凝聚聚类合并相邻且视觉相似的帧，以 Ward 距离控制簇内方差；对每个簇采用边界增强的加权池化，避免普通均值把变化位置抹平。
- **Contextual Feature Enhancement（CFE）**：从句子库检索相关事件文本，通过 query-guided 双向交互把视觉簇、原始帧和文本上下文融合，再输入 deformable Transformer。
- **有效机制**：CFA 给模型一个显式、连续、低噪声的事件结构，CFE 则用文本语义选择哪些结构与待描述事件相关。论文消融显示二者互补，说明收益来自“结构化聚合 + 条件语义增强”，而非单一更强编码器。

## B. 与 CPL_LREV 问题的对应关系

| CPL_LREV 的实际问题 | CACMI 中有关的机制 | 可能有效的原因 | 不能直接照搬及所需改造 |
|---|---|---|---|
| 200 帧被下采样到 50，短事件边界信息丢失 | 时间约束聚类与边界增强池化 | 在降采样前保存变化点，并把连续相似帧压成事件单元 | Captioning 可使用完整视频和通用句子库；grounding 必须由当前 query 决定保留哪些簇，不能把检索句当答案 |
| cPCA/背景建模与 query 无关 | 文本条件的跨模态增强 | query 可决定某一视觉变化是否为目标事件而非普通镜头变化 | 只需 query 的短语/改写作为语义上下文，不依赖外部 caption corpus |
| 多高斯软 mask 与 outer hull 区间不一致 | 显式连续事件簇 | 连续簇可作为合法区间的结构先验，避免跨过中间无关段 | 不直接预测 caption event proposal，而是约束 CPL_LREV 的中心、宽度和最终边界 |

## C. 独立创新方案

### C1. 方案名称

**QCEC（Query-conditioned Coherent Event Clusters，查询条件连贯事件簇）**。

### C2. 目标问题

主要解决短事件召回差、边界感知弱、query 无关的时序表示以及 mixture mask/outer boundary 不一致；次要改善 5 个 proposal 高重叠。

### C3. 核心假设

如果在 200 帧高分辨率特征上先提取连续事件簇，并让 query 决定簇的相关性，那么 proposal generator 接收到的将是“边界被显式保留的事件结构”，而非均匀下采样后的平滑序列。这样短动作不会因 4 倍降采样消失，跨越强变化点的宽区间也会受到结构性惩罚。

### C4. 方法设计

1. **插入位置**：在 `cpl_lrev/models/cpl.py` 的 `frame_fc` 之后、`n_frames // 4` 下采样之前使用高分辨率特征 `F^200∈R^{B×200×D}`；现有 50-step reconstruction 分支保持不变。
2. **连续聚类**：离线或无梯度计算相邻帧距离 `d_t=||F_t-F_{t-1}||`，仅允许相邻簇合并；以固定簇数 `M=32` 或阈值停止。每簇输出中心、跨度及一个边界增强池化 token，池化权重在簇首尾较大。
3. **查询条件化**：把 query 分为动词中心短语、实体短语和整句三个 token；用小型 cross-attention 得到簇—短语矩阵 `A∈R^{B×M×L_q}`。输出簇相关性 `r_m`、左右转移置信度 `b_m^L,b_m^R`。
4. **输入 proposal generator**：将 50-step `h` 与按时间映射回去的 `[r,b^L,b^R]` 三通道拼接，经线性层仍投影到原 hidden size；另将 top-related 簇中心/跨度编码为 5 个 proposal slot 的初始化偏置。原高斯参数仍由 `GaussianMixtureProposalGenerator` 预测。
5. **连贯性约束**：对最终 proposal 区间 `[s_n,e_n]` 定义 `L_cross`，若区间跨越一个高置信内部变化点且变化点两侧 query 相关性差异大，则惩罚；若边界靠近高置信转移则奖励。该损失只用无梯度簇先验，不让当前 proposal 自己产生标签。
6. **解码修改**：推理时先得到原 outer hull，再在其附近 `±2` 个高分辨率帧内吸附到最近的相关簇边界；不改变 proposal 数量和 reconstruction decoder。
7. **训练**：第一阶段冻结主干，仅训练 query-cluster 交互和融合层；第二阶段联合微调，`L=L_base+λ_bL_cross`。不以 reconstruction NLL 监督簇边界。

### C5. 与原论文的区别

原论文以聚类事件和检索文本增强 dense captioning 的视觉特征，并拥有显式 event-caption 监督。QCEC 只借鉴“连续聚类保存事件结构”和“文本条件选择上下文”两种机制：把外部句库替换为当前 query 的多粒度短语，把输出从 caption feature 改成 proposal 初始化、跨变化点约束和边界吸附，并适配弱监督、单 query 的 grounding。

### C6. 为什么可能有效

当前 CPL_LREV 的 50-step 表示对短段先天不利，outer hull 又可能把相隔的高斯分量之间的背景包含进来。QCEC 在降采样前保留变化点，令短动作至少对应一个完整簇；query 相关性使相同镜头变化只在与语义有关时影响边界；`L_cross` 则直接降低跨无关簇的宽区间分数。因而改变的是 temporal representation、query-video interaction 和 boundary perception，而不是仅增强语言重建。

### C7. 潜在风险

- 视觉距离的变化点不一定是语义边界，快速运动会导致过分割，静态动作会欠分割。
- 在线凝聚聚类增加预处理成本；簇数过小仍会损伤短事件。
- 边界吸附若置信度未校准，可能降低原本正确的粗定位。
- 与高斯 pull/push loss 可能产生相反梯度，应先冻结原 generator 验证。

### C8. 最小验证实验

只实现无梯度相邻聚类、三个簇先验通道和推理边界吸附，保留 backbone、5 个 proposal、全部原 loss 与训练轮数。比较 CPL_LREV baseline 与 QCEC-lite，报告 R@1/R@5 在 IoU 0.3/0.5/0.7、mIoU，并单列 GT 宽度 `<0.15` 和 proposal 跨强变化点比例。若短段 R@5 mIoU 提升至少 0.03、R@1@0.5 不下降且跨点比例下降，支持假设；若只提高 NLL 或长段指标、短段无改善，则不继续联合训练。

### C9. 进一步实验 / Ablation

- 时间约束聚类 vs. 均匀窗口 vs. 无聚类；边界增强池化 vs. 平均池化。
- query 条件簇相关性 vs. query 无关变化点。
- 只做 feature fusion、只做 boundary snapping、二者同时。
- `M∈{16,32,64}`，高分辨率 100/200 帧，`λ_b∈{0.05,0.1,0.2}`。
- 用 query 整句、动词/实体短语、二者组合。
- 检查 proposal overlap、宽度分布和高斯内部空隙，确认收益确由结构先验造成。

## 依据

- 原论文：`AAAI26-Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction.pdf`，重点为 CFA、CFE 与消融部分。
- 模型问题：[`../cpl_lrev_model_issues.md`](../cpl_lrev_model_issues.md)。

