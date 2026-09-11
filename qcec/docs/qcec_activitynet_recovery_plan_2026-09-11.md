# QCEC ActivityNet：退化排查与下一阶段实验方案

日期：2026-09-11  
状态：实施与实验计划；尚未执行本文提出的配置修改、诊断或训练。  
目标：先恢复可信基线并隔离退化原因，再判断 QCEC 是否提供可重复的定位增益。

## 1. 依据与结论边界

本方案依据以下材料及当前实际实现：

- 失败运行：`qcec/logs/activitynet_scratch/qcec_activitynet_scratch_2026-09-10_20-32-25.log`。
- 设计文档：`docs/videoCaption/proposal_aaai26_cacmi_detailed.md`，特别是第 7 节训练策略。
- 原论文：`docs/videoCaption/AAAI26-Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction.pdf`。
- 基线记录：`cpl_lrev/logs/activitynet/cpl_lrrv_final_outer_2026-07-08_18-02-24.log`。
- 实现：`qcec/models/cpl.py`、`qcec/models/modules/qcec.py`、`qcec/models/modules/gaussian_mixture.py`、`qcec/models/loss.py`、`qcec/runners/main_runner.py`。

以上路径相对于 `/data/chenyuan/videogrounding/w1m2`。文献和旧方案是分析材料，本文的实施顺序以本次诊断为依据。

### 1.1 已确认事实

1. scratch 运行使用 `freeze_backbone_epochs=3`；实际冻结主干，仅训练 QCEC 与 adapter。原设计明确要求 scratch 设为 0。
2. 冻结期间全局学习率计步继续。每轮 1170 步，主干解冻时已越过 400 步 warm-up。
3. epoch 3→4，末段训练 NLL 从 9.0616 降到 5.5855，平均候选宽度从 0.2746 升到 0.8197，Validation R5@0.5 从 57.81% 降到 34.29%。
4. best-r1 的 Test 候选两两平均 IoU 为 0.7317，97.41% 的候选对 IoU 超过 0.5；短事件 R5@0.5 为 0。
5. 本次启用 global adapter 与 crossing loss，但关闭 slot prior、snapping；barrier 使用 detach。
6. 历史基线使用 semantic_vote，本次使用 nll，不能把 R1 差值全部归因于 QCEC。

### 1.2 需要实验验证的解释

- 错误冻结及调度错位是否是主要退化来源。
- adapter 是否放大长区间重构捷径，或主要只是被错误训练起点影响。
- outer envelope 是否由远端低 importance 分量驱动，从而使训练掩码和定位边界脱节。
- relevance 是否反映视频对 query 的支持，还是主要反映已融合的 query 信息。
- crossing 的问题来自伪边界质量、梯度规模、梯度冲突，还是缺少正向覆盖约束。

当前不能宣称“QCEC 已被证明无效”，也不能宣称“冻结改成 0 就一定恢复”。

## 2. 总体路线与优先级

按以下顺序推进，每阶段形成可检查结果后再扩展：

1. P0：锁定评测与初始化协议，补齐最小诊断。
2. P1：同工程运行三个核心 scratch 对照：baseline、adapter-only、adapter+crossing。
3. P2：根据 P1 的退化位置，验证候选几何、relevance 和 crossing 梯度。
4. P3：每次只修复一个被证据支持的问题；必要时开展独立 warm-start 实验。
5. P4：多种子复现，最后评测 Test；确认有效后才迁移到 Charades。

第一轮不同时修改簇数、损失权重、高斯表示、学习率与排序。不直接以追平 LNP/SOTA 作为排查阶段验收条件。

## 3. P0：建立可信的实验协议

### 3.1 保存与配置约束

- 保留旧日志、checkpoint 和原始配置；新建独立配置与输出目录。
- 新实验在 `qcec/` 内完成，基线也使用该工程的 QCEC-disabled 路径。
- 记录代码版本或源码摘要、完整解析后配置、seed、数据与 cluster index 摘要、初始化来源、实际训练参数量。
- scratch 必须显式设 `model.config.qcec.freeze_backbone_epochs=0`。
- 在训练入口增加校验：未加载有效初始化权重却启用 QCEC-only 冻结时，直接报配置错误。对 resume，应依据恢复的 epoch 和初始化历史判断，避免误拒正常恢复。
- 每次阶段切换记录 trainable 参数、LR、全局 step、主干实际更新次数、event schedule。

### 3.2 初始化一致性

仅固定相同 seed 不保证不同模型结构的公共参数初值一致：新增模块会消耗随机数。

实施方法：每个 seed 先构建 disabled baseline 的初始 state_dict，再按名称与形状把公共参数复制到 A/C 模型；QCEC 参数另用固定 seed 初始化。逐 tensor 比较公共参数，禁止静默漏载。训练 shuffle 和数据随机状态在模型构建结束后重置。

在固定 batch、`eval()`、固定 word-mask 的条件下，验证零初始化 adapter 的 A 与 B 初始输出一致。比较 component 参数、mixture mask、最终区间和 NLL；float32 初始容差可设 `atol=1e-6, rtol=1e-5`，差异需追踪到具体算子。

### 3.3 固定评测随机性与 checkpoint 标准

当前 forward 中 `_mask_words()` 使用 NumPy 随机采样，`eval()` 不自动取消这类随机性。需要实现独立评测 RNG 上下文，评测后恢复训练 RNG；最好按样本 ID 固定 mask，避免依赖 batch 划分。

- 同 checkpoint、同数据、同 mask 重评两次，验证可重复性。
- 一次 forward 缓存 proposal 和 NLL，再计算 nll/semantic_vote 两种 R1，避免排序比较混入新 mask。
- 主排序统一为 `nll`；semantic_vote 只作为同一候选集的诊断结果。
- 主 checkpoint 固定按 Validation `R@1,mIoU` 选取；主表报告该 checkpoint 的全部 R1/R5。
- 同时保留 best-r5 和 best-composite 诊断表，但禁止跨 checkpoint 拼出一行“最优结果”。记录 composite 的实际公式。
- 开发阶段只看 Validation；当前 runner 每轮训练完成会自动评测 Test，应增加关闭最终 Test 的显式选项。最终方法、权重、seed 与选择规则锁定后，再对各预先指定运行评测 Test。
- 不使用 `--legacy-select-on-test`，不依 Test 选择方法或超参数。

### 3.4 最小新增日志

| 类别 | 必须记录的内容 | 用途 |
|---|---|---|
| 损失 | 实际 total_loss、各加权项及 raw 项 | 当前 final_loss 只来自 rec_loss，不代表总目标 |
| 几何 | 每个 slot 宽度分位数、预测整体宽度分布、两两 IoU | 区分少数异常值与系统坍缩 |
| 覆盖 | R5、分长度 R5、每组样本数 | 判断候选集是否仍有正确答案 |
| 排序 | 同候选的 nll 与 vote R1、oracle R5 | 区分生成和排序 |
| 约束 | 负 margin 激活率、barrier 分布与跨越率 | 判断损失是否仍提供约束 |

长度分组沿用现有代码：short `[0,0.15)`，medium_short `[0.15,0.35)`，medium_long `[0.35,0.60)`，long `[0.60,+∞)`，均为 GT 时长/视频时长。

## 4. P1：最小 scratch 对照矩阵

### 4.1 公共配置

从 `qcec/config/activitynet/qcec.json` 派生新文件，显式设定：

```json
{
  "model.config.qcec.freeze_backbone_epochs": 0,
  "model.config.qcec.use_slot_prior": false,
  "model.config.qcec.snap_enabled": false,
  "loss.qcec_component_cross_weight": 0.0,
  "loss.qcec_boundary_align_weight": 0.0
}
```

上面是配置键路径说明，不是可直接替换原 JSON 的完整文件。其余保留原配置：C3D、200 steps、5 proposals、outer、32 clusters、batch 32、LR 4e-4、warm-up 400、30 epochs，以及原 event/mixture 权重。

### 4.2 核心实验

| ID | 建议配置文件名 | QCEC | cross weight | 初始化 | 要回答的问题 |
|---|---|---:|---:|---|---|
| B | `recovery_b_baseline.json` | false | 0 | 同 seed 公共随机权重 | 当前工程与协议下的真实基线 |
| A | `recovery_a_adapter.json` | true | 0 | 与 B 公共权重相同 | adapter 本身带来什么变化 |
| C | `recovery_c_cross.json` | true | 0.1 | 与 B 公共权重相同 | crossing 在 adapter 之上的边际贡献 |

先做 seed=8 的三个完整实验；前三至五轮观察退化，但不因短期指标低就任意早停。通过后再补 seed=18、28。只在 NaN、数据错误、冻结校验失败等执行故障时直接终止并修复。

首轮共 3×30 epochs，后续多种子最多补 6×30 epochs；实际 GPU 时长以新环境测量为准。梯度诊断不在每个训练 batch 开启。

建议先运行 B，再 A，再 C。共享 GPU 时避免未经资源测量同时启动多个完整训练。

### 4.3 命令模板与前置条件

以下命令仅在新配置、初始化一致性与 P0 评测修改完成后执行。当前 CLI 已支持这里展示的参数；配置文件尚待创建。

```bash
cd /data/chenyuan/videogrounding/w1m2/qcec
python train.py \
  --config-path config/activitynet/recovery_b_baseline.json \
  --qcec-disabled --qcec-cross-weight 0 \
  --selection-strategy nll --select-on-val --seed 8 \
  --tag recovery_b_s8 --log_dir logs/activitynet_recovery/b_s8

python train.py \
  --config-path config/activitynet/recovery_a_adapter.json \
  --qcec-enabled --qcec-cross-weight 0 \
  --selection-strategy nll --select-on-val --seed 8 \
  --tag recovery_a_s8 --log_dir logs/activitynet_recovery/a_s8

python train.py \
  --config-path config/activitynet/recovery_c_cross.json \
  --qcec-enabled --qcec-cross-weight 0.1 \
  --selection-strategy nll --select-on-val --seed 8 \
  --tag recovery_c_s8 --log_dir logs/activitynet_recovery/c_s8
```

公共权重复制与关闭自动 Test 需要实施后接入命令或配置，不能把上述模板当作已包含这些能力。建议通过配置接入，避免虚构现有 CLI 参数。

### 4.4 结果分流

| 观察 | 下一步 |
|---|---|
| B 自身严重长区间坍缩 | 先查基线复现、初始化和 outer 几何，暂停 QCEC 扩展 |
| B 正常、A 退化 | 优先查 adapter 表示、relevance 捷径与全局压缩 |
| A 正常、C 退化 | 优先查 barrier 准确性和 crossing 梯度冲突 |
| A/C 的 R5 正常、R1 低 | 优先做同候选排序诊断 |
| A/C 的 R5 与短事件召回同时低 | 优先解决候选生成，不先改排序 |
| A/C 均恢复且胜过 B | 多种子复现，再考虑其他机制 |

若需要量化错误冻结的因果贡献，再添加 C-F3：除 `freeze_backbone_epochs=3` 外与 C 完全相同，作为有意复现的诊断实验并显式标记。历史失败日志因初始化、协议不同，只能提供旁证，不能替代这一控制实验。

## 5. P2：定位具体失效机制

### 5.1 候选几何与重构一致性

对 B/A/C 的早期、主 checkpoint 和最终 checkpoint，缓存同一批 Validation 输出，记录：

- 每个 component 的 center、width、importance。
- 最左/最右边界由哪个 component 决定，其 importance 分布。
- mixture 有效质量区间与 outer 区间的宽度差。
- proposal NLL 与 GT IoU 的相关性，并按 GT 长度分组。
- GT 外掩码质量比例，仅作离线诊断，不反传。

有效质量区间可用归一化 mixture 累计质量的 5%–95% 分位区间作为固定诊断定义。该区间不是已验证的新预测方法。

对同一 checkpoint 离线比较 outer、importance-weighted 和质量分位区间。它们只用于判断几何风险；如果某规则在 Validation 上有效，仍需单独训练/评测确认，不把后处理改进直接称为 QCEC 增益。

### 5.2 relevance 是否真正依赖视频

固定样本子集与随机种子，比较原始配对、同视频不同 query、跨视频打乱 query、固定 query 替换视频四种情况。

记录 relevance 的空间方差、饱和比例、原/扰动差异；在 Validation 上计算 relevance 与每簇 GT 覆盖比例的关系。需要同时有敏感性与正确性，不能仅凭“换 query 后数值变了”判定语义有效。

如果 relevance 主要依赖 query，则后续单独比较“用融合前视觉簇计算匹配分数”与当前“融合后计算分数”。保持其他参数预算、损失及训练设置尽可能一致。

### 5.3 聚类与 barrier 质量

只在 Validation 上做离线诊断：

- 每个 GT 起止点到最近 cluster 边界的距离，按视频时长归一化，报告分位数。
- 分别用 1/200、2/200、5/200 容差统计边界命中，并报告伪边界数量；更多簇天然提高命中，不能只比 recall。
- 检查 GT 内部 barrier 的数量、强度，以及 GT 两端附近 barrier 的数量、方向和强度。
- 统计同一 query 中高 relevance 区域是否多峰；检查 global soft_start/soft_end 是否落在两峰之间。

GT 仅用于诊断与模型选择，不进入训练输入、伪标签或 loss，保持弱监督设定。

注意：当前 transition score 基于 projected cluster token，并使用 detach；detach 只切断该次反向传播，不代表分数随训练完全固定。应记录跨 epoch 的分数漂移。

### 5.4 梯度诊断

在固定的少量训练 batch 上，于 epoch 1、解冻对应阶段、epoch 5/10 抽样分析。对每个加权 loss 使用 `autograd.grad`，不调用额外 optimizer step，不污染原参数梯度。

测量对象：center/width head 参数，以及建议新增暴露的 sigmoid 前 center/width logits。不要把现有 sigmoid 后 center 误标为 logit。

报告：各 loss 的梯度 L2 范数、cross 与 rec 的范数比、梯度余弦相似度、饱和 logit 比例；同时记录 clipping 前总范数。若使用训练态诊断，应恢复 RNG 并避免额外更新 event 运行统计。

不要通过 `qcec_loss / rec_loss` 直接推断梯度贡献。只有 barrier 有合理准确性且梯度确实不足时，才小范围调整 crossing 权重。

## 6. P3：按证据修复，逐项消融

### 6.1 优先级一：训练路径

scratch 固定无冻结；不在此基础上同时修改学习率。若 scratch 的 B 正常而 A 难以优化，可开展独立 warm-start 分支：

- 从 P1 中预先按 Validation 选定的 baseline checkpoint 初始化。
- W-B：baseline 在同样额外预算下继续训练，重新建立优化器与调度。
- W-A：同 checkpoint 加零初始化 adapter，只训练 adapter 1 轮，再联合训练。
- W-C：只有 W-A 有效后加入 crossing。
- 公共额外预算初定 10 轮、LR 4e-5；这些是待验证设置，不是已证明最优值。
- W-A/W-C 的冻结长度、event 语义 schedule 与已有 subspace buffer 必须显式记录；不因 fresh epoch 从 1 开始就静默重置已有有效训练阶段。
- 现有 `--init-from-baseline` 只支持 QCEC-enabled，W-B 需要受控的权重初始化路径，不能把 `--resume` 当作相同的优化器重置方案。

结论单独标记为“基线预训练后微调”，不与 scratch 等训练预算混报。

### 6.2 优先级二：几何一致性

如果远端低 importance 分量决定外包络的现象明显，单独试验质量区间或 importance-aware 边界。重点检查：重构证据、预测区间、crossing 约束和负样本范围是否仍有清晰一致的含义。

边界定义的改变属于基础生成器修改，应先在 B 上做对应对照，再判断是否与 QCEC 互补。不要将简单缩短全部区间作为正式修复；它可能提高短事件而损伤长事件。

### 6.3 优先级三：语义与时间结构

当 relevance 或全局压缩被证实存在问题时，再依次尝试：

1. 融合前视觉簇与 query 的匹配分支，避免用已注入 query 的表示自证相关。
2. 保留簇位置与序列结构，为不同 proposal 提供独立局部提示；首先使用已有 slot prior 接口做单因素消融。
3. M=16 或 M=10 的簇数对照；每个配置重新生成匹配的 cluster index 并验证元数据。不因原论文使用 10 就直接认定其适用于 C3D 弱监督任务。

如果增加跨视频对比监督，需单独设计同义/重复动作的假负样本处理，并作为新方法阶段；不与 P1 基础排查混在一起。

### 6.4 暂缓事项

- snapping：仅在 raw proposal 已有可用短事件召回、边界提示有准确性后开启。
- component crossing / boundary alignment：当前配置键存在不等于实际损失路径已实现；启用前验证函数、调用与梯度，不仅修改 JSON 权重。
- 调大 event 或 diversity：只有明确观察到相应约束失效且诊断支持时再做，不同时扫描多组损失。

## 7. 验收、停止扩展与报告标准

### 7.1 执行正确性

- scratch 无随机主干冻结；公共权重一致。
- 所有预期参数在对应阶段产生梯度并被 optimizer 管理。
- 固定评测可重复；排序比较使用同一候选缓存。
- 日志有 total_loss、配置来源、checkpoint epoch 与选择规则。

### 7.2 有效性

以 B 为配对基线，报告 seed=8/18/28 的均值、标准差和逐 seed 差值。主要看同一 Validation-selected checkpoint 的 R1 mIoU、R1@0.5、R5@0.5，同时提供分长度结果。

将“值得扩展”的工程门槛预设为：至少 2/3 seeds 的 R1 mIoU 改善，平均 R1 mIoU 至少提高 0.5 个百分点，R5@0.5 平均下降不超过 1 个百分点，且短事件 R5@0.5 不出现系统性退化。这是资源决策门槛，不是统计显著性声明。

只有生成能力改善而 R1 未改善时，转入排序专项；只有 R1 改善但短事件和 R5 明显下降时，不宣布全面成功。

如果 A 持续不如 B，且语义/几何诊断不能显示额外有效信息，停止增加 QCEC 复杂度，保留负结果并重设计假设。如果 A 有效而 C 持续无效，正式保留 adapter-only，不强行保留 crossing。

### 7.3 最终交付物

- 三组及后续必要对照的完整配置、运行清单与初始化摘要。
- 逐 epoch CSV/JSON 指标；固定样本诊断缓存。
- width、R1/R5、short R5、负 margin 激活率的训练曲线。
- 至少 20 个固定规则抽样的视频可视化，覆盖四个长度组与成功/失败案例；禁止只挑改进样本。
- 单 checkpoint 主表、多 seed 表、消融表及明确的 Test 最终结果。
- 每个假设的“支持／不支持／证据不足”结论，区分基线几何改进与 QCEC 额外收益。

## 8. 下一次实际执行的清单

1. 实施 P0 的 scratch 校验、公共初始化、固定评测和禁止开发阶段自动 Test。
2. 添加 total_loss 与训练阶段日志，使用小 batch 验证初始化 identity、参数更新和评测复现。
3. 创建 B/A/C 三个配置，保持已有默认配置与历史输出不变。
4. 依次完成 seed=8 的 B/A/C，生成统一 Validation 表和候选几何曲线。
5. 按第 4.4 节分流，只开展有证据需求的 P2 诊断。
6. 验证有希望的变体，补充另两个 seeds，锁定方案后最终评测 Test。

当前文档交付不包含启动训练、修改模型代码或生成上述配置。最优先工作是恢复实验的可解释性，而不是立即把更多模块打开。
