# QSTG 后续修复、验证与重训练方案

> 日期：2026-09-16。状态：实施方案，尚未执行代码修复或重训练。
>
> 工程根目录：`/data/chenyuan/videogrounding/w1m2/qstg`。
> 固定 Python：`/home/chenyuan/miniconda3/envs/cpl/bin/python`。
> 本文承接对 2026-09-15 六次 QSTG 实验的只读审计。本文中的新增配置键、工具和测试均为**待实现接口**，不能直接当作当前代码已支持的命令使用。
>
> 本次交付仅新增本方案。后续实现应另行授权；不得覆盖已有配置、日志、数据划分、checkpoint 或只读 baseline。

## 1. 目标与决策原则

本轮不以“堆叠模块后刷新某一列 test 指标”为目标，而是依次建立四个可验证的事实：

1. 数据协议和初始化来源可信，同协议 baseline 可以稳定复评。
2. Stage A 能学习匹配/质量，而不改变 baseline 候选。
3. proposal quality 的训练与推理输入一致，排序收益可以独立测量。
4. Stage B/C 在候选召回不退化的前提下，改善 Rank-1 或高 IoU 定位。

首轮保持 features、词表、query 长度、proposal 数量、200 帧采样和 50 节点不变。先修正确性、接口和梯度路径，再决定是否改变模型容量。

### 1.1 两条路线必须分开

| 路线 | 使用数据/权重 | 用途 | 可以对外宣称什么 |
|---|---|---|---|
| 历史诊断线 | 现有 bootstrap、现有六次运行 | 快速定位退化机制、评估打分贡献 | 历史结果诊断，不是严格无泄漏 benchmark |
| 正式认证线 | 清晰协议、干净 bootstrap、来源可追踪的新运行 | 最终模型比较和多 seed 验证 | 经协议说明的同条件增益 |

不要因为历史诊断线效果改善，就跳过正式认证线。尤其是 Charades，换一个新 validation 文件不会消除旧权重的污染。

### 1.2 首轮通过标准

- 确定性正确性测试全部通过，不存在有效输入导致的 NaN/Inf。
- Stage A 在 `eval()`、相同输入和 mask 下，raw center/width/membership 与冻结 baseline 一致。
- quality 的 pair 对角特征与推理特征逐维一致。
- 每个最终实验具有数据 hash、源码标识、完整配置、初始化 hash、scope、运行时开关和预定选模规则。
- 改善首先体现在 validation；不能观察 test 后选择 loss 权重、阶段、seed 或 checkpoint 目标。
- 不预承诺达到 SOTA。是否继续某个模块，由单变量实验决定。

## 2. 已知事实、假设与证据锚点

以下行号对应审计时源码；实施后应同时保留修复前后的源码版本标识。

| 发现 | 证据位置 | 性质 | 本轮处理 |
|---|---|---|---|
| A 阶段主干参数冻结，但 global adapter 仍改变 proposal | `models/modules/qstg.py:677–682`；`models/cpl.py:206–227`；AN-A 日志 167–168 行 | 已确认实现及退化现象；具体贡献仍需对照 | 首先切断 A 的 proposal 改写路径 |
| AN-A 五候选平均宽度 .9568–1，pairwise IoU .9815 | `logs/activitynet/qstg_stage_a_2026-09-15_14-53-09.log:168` | 实测 | 建立宽度、oracle recall 硬诊断 |
| 顶层 detach 配置没有转发给模型 | `models/cpl.py:80–82`；`runners/main_runner.py:441–445` | 已确认 | 规范配置解析并断言实际值 |
| quality 的 pair/eval 特征定义不一致 | `models/modules/qstg.py:796–859,861–930` | 已确认 | 共用一个特征构造器 |
| before/after 核方向反向 | `models/modules/qstg_ops.py:246–251` | 已通过两节点 CPU 验证 | 修方向，并独立测试 |
| 关系距离使用归一化时间，方案使用节点距离 | `models/modules/qstg_ops.py:238–242`；原方案第 5.3 节 | 已确认规格差异 | 显式单位；与方向修复分开评估 |
| refinement coverage 跨 batch 归一化，空支持最终 gather 有风险 | `models/modules/qstg_ops.py:528–536,596–601` | 已确认代码风险；当前未开启 | 开启前修复，不能解释现有 raw 掉点 |
| final_loss 不是实际反传总损失 | `models/loss.py:31–58`；`runners/main_runner.py:221–245` | 已确认 | 新增 backward_total_loss，保留旧字段兼容 |
| nll_topk 实为综合分数排序结果 | `runners/main_runner.py:343–346` | 已确认 | 重命名，并独立报告纯 NLL |
| Charades bootstrap 曾训练当前新 val，并在 test 上选模 | bootstrap 保存配置；`../cpl_lrev/logs/charades/lrevFinal_2026-09-04_16-01-36.log:1` | 已确认协议污染 | 正式线重训 bootstrap |
| B/C 精确初始化路径未记录 | `train.py:173–177` | 已确认可追溯性缺口 | 加 source_checkpoint manifest |
| fa 可能通过扩宽/饱和降低 loss | AN-B epoch1→2 R5@.5 从 50.13 降至 42.15，后续 fa 很低 | 待单变量验证 | 首轮 B 关闭 fa，再单独加入 |
| 50 节点分辨率可能限制高阈值 | 200 帧、stride4、长视频 | 待验证；不是首要已证根因 | 后期再做 stride ablation |

审计未发现六次日志中的 NaN/OOM；18 个最终选中 checkpoint 和两个 bootstrap 的张量检查均有限。**这不等于历史梯度全过程有限**，不能把数值稳定性检查省略。

## 3. P0：建立可复现、无混杂的协议

### 3.1 不变的历史材料

保留 `logs/`、`checkpoints/bootstrap/`、现有 A/B/C checkpoint、现有 config 和 `../cpl_lrev/`。新实验使用独立 `repair_v1` 名称和目录，禁止重用历史 run tag。

建议新增目录，仅在实施阶段创建：

```text
config/{activitynet,charades}/repair_v1/
logs/repair_v1/{activitynet,charades}/
checkpoints/repair_v1/{activitynet,charades}/
reports/repair_v1/
```

历史 checkpoint 评估时，只能称为 `historical_diagnostic`，不得把修复后的结构与旧权重拼接后称为完整重训练结果。

### 3.2 Split manifest

新增只读审计入口 `tools/audit_protocol.py`。默认只输出到 stdout；需要保存报告时显式传输出路径并禁止覆盖。

每个 split 记录：文件 SHA-256、query 数、视频数、重复 query 数、视频交集、完整样本交集、时长与 GT 宽度分布、时间戳越界情况、词表 hash、特征文件标识。

当前数据事实作为回归基准：

| 数据集 | train query/video | val query/video | test query/video | 备注 |
|---|---|---|---|---|
| ActivityNet | 37421/10009 | 17505/4917 | 17031/4885 | train 与评估视频无交集；val/test 共享4885视频，完整样本交集1条 |
| Charades QSTG | 9616/4378 | 986/464 | 3158/1225 | 三者视频互斥 |

ActivityNet 必须追溯这三份 JSON 与原始标注版本/标注集合的映射。共享评估视频不能直接等同训练泄漏，也不能忽略。未经认证，报告写明“本地协议，论文可比性待核实”。不要自行改动其 split 后继续与旧论文数字直接比较。

Charades 保持当前固定 video-disjoint 划分，正式 bootstrap 只能使用 9616 条训练 query，并在 986 条 validation 上选模。test 不参与 bootstrap、QSTG、权重搜索和停止决策。不要从已污染的 grounding 权重继续微调后声称已经去污染；可以使用来源合法、未在这些定位标注上训练的通用特征/词向量。

### 3.3 两种 baseline 明确命名

- `local_control`：QSTG 关闭，但保留本地 mixture/LREV 等现有设计；它是 QSTG 归因的主要对照。
- `paper_cpl_reference`：用户给出的论文 CPL 数字；只用于协议说明后的参考比较，不能替代 `local_control`。

首轮先复评历史初始化模型，隔离 mask 和 selector 差异；正式线再建立干净 `local_control`。如果进一步复现纯 CPL，应作为独立工作项，不混入本轮单变量试验。

### 3.4 固定选择目标

正式线预先指定唯一主目标：validation R1 mIoU。R5 mIoU 和 composite 可作诊断保存，但不能观察 test 后选择其中最好的方案。

同时报告所有匹配阈值；不能把不同 checkpoint 的最优列拼接为一个模型。跨方法比较只使用相同 dataset、Rank 和 IoU threshold，空缺 mIoU 保持空缺。

需要兼顾候选召回时，可预注册“validation oracle recall 不显著退化”的候选筛选条件；条件、容差及选择顺序必须在运行前固定，不能观察结果后修改。

### 3.5 Checkpoint provenance

建议扩展格式到 `format_version=3`，保留旧格式只读加载能力。新增：

```text
source_checkpoint: {absolute_path, sha256, source_run, source_epoch, source_stage}
initialization_mode: baseline_warmstart | stage_weights_only | exact_resume
source_code: {commit_if_available, dirty_flag, core_file_sha256}
data_manifest_sha256
resolved_config
runtime_flags
selection: {split, objective, evaluation_protocol_id}
completed_epoch, stage_local_epoch, global_update
stage_parent_run
best_selection_state
rng_states
```

本地审计的 bootstrap hash 可作为历史标识：

- AN：`20692b92a9f7d1f290a51914d0e8c214fc8e02264a71103b5bc2e3cdfc4dddd8`，对应原 run epoch7 的 best-r5/composite。
- CH：`f751280ceeb3cbd0517a936bbd9d86881fe02036ea99f2b10f0b9b906afb4397`，对应原 run epoch13 的 best-composite。

`stage_weights_only` 重建 optimizer/scheduler，但必须明确 event schedule 使用阶段局部 epoch 还是延续进度；初始低秩统计是否保留也要记录。不同策略不能藏在同一个“resume”名称下。

`exact_resume` 必须恢复 epoch、optimizer、scheduler、RNG 和 best selection 状态。若只支持 epoch 边界恢复，应明确限制；不能恢复权重后仍从 epoch1 开始并称为精确续训。

## 4. P1：修复基础接口与数值安全

### 4.1 配置归一化

在 `train.py` 加单一 `resolve_runtime_config()`，在日志输出、model 构造和 checkpoint 保存之前运行。

- 模型最终只从 `model.config` 读取 `detach_base_states`。
- 兼容旧顶层字段：存在旧字段时显式迁移；两处同时存在且值冲突则报错，不能静默覆盖。
- 日志打印解析后的实际值，而非只打印原 JSON。
- 新增枚举/范围校验；对已知关键字段拼写错误或未消费字段报错。
- `num_query_graph_layers` 当前并非真正可调的实现入口，不允许未来把修改该字段当作已完成多层实验；应支持它，或明确只允许1。

验收：顶层 true 能使实际模型属性为 true；冲突输入失败；A/B 冻结参数列表与预期一致；C 解冻列表符合预期。

### 4.2 有限性防护

训练循环顺序调整为：

```text
forward → 检查实际启用分支的必要输出
→ 计算 loss → 检查 backward_total_loss
→ backward → 检查有效梯度
→ 记录 clip 前 norm → clip → 检查 clip 后 norm
→ optimizer.step → 定期检查参数
→ 完整 epoch 校验 → 原子保存 checkpoint → validation选模
```

任何非有限值立即中止本 run，记录 batch/sample_uid、分支名、张量范围、阶段和来源；不得继续评估并产生正常 best checkpoint。只在新 run 目录保存隔离故障报告，不覆盖已有有效 checkpoint。

零权重 loss 应支持跳过计算，不以 `0 * NaN` 充当安全零；如需观测该项，用独立诊断模式且明确其不进入 backward。日志保留 enabled/configured/effective 权重。

对单高斯权重采用稳定的 log-weight 平移后指数化，再做归一化，避免极窄宽度和网格错位时全下溢。不要只添加分母 epsilon 后把全零 Gaussian 当作正常候选。保持正常宽度区间与原计算的数值兼容测试。

首轮使用 FP32。不将 FP16/BF16 加入同一轮性能实验。后续单独测试 attention mask、全padding、softmax/logsumexp、极端temperature和Gaussian宽度。

## 5. P2：重定义安全的 Stage A

### 5.1 核心约束：A 的候选必须固定

新增明确开关，而非只依赖零初始化：

```json
{
  "model": {"config": {
    "detach_base_states": true,
    "qstg": {
      "global_adapter_enabled": false,
      "proposal_prior_scale": 0.0,
      "importance_bias_scale": 0.0,
      "component_residual_enabled": false,
      "connected_refine": false
    }
  }},
  "loss": {"qstg_detach_proposal_geometry": true}
}
```

`global_adapter_enabled=false` 必须前向旁路，并使对应参数不可训练；不能仅将其输出乘0却在其他支路间接使用。proposal-facing 的 center/width/membership/component importance 在 contrastive quality 路径中 detach。

A 使用已有 baseline proposal 和 reconstruction NLL，训练匹配与质量。原 reconstruction/IVC/event/mixture 可以记录，但不进入 A 的反传目标。

### 5.2 明确真正能训练的参数

修复版 A 的最小 scope 命名为 `qstg_match_quality`，避免继续用“QSTG全部参数都在学”的模糊描述。包括：

- phrase encoder；
- pre-query/pre-visual binding projection；
- quality head；
- 共用质量特征需要的匹配参数。

在 quality 改用 clean/pre binding、候选固定之后，final grounded graph、global adapter、component adapter **可能没有任务梯度**。因此最小 A 不宣称预训练了整个 temporal graph；应冻结并记录这些模块，等 B 接入 reconstruction 路径时训练。

不要为了让每个参数“有梯度”而添加未验证辅助 loss。若未来需要 A 预训练 final graph，必须另设明确匹配目标和独立消融。

### 5.3 冻结模式与 dropout

`requires_grad=false` 不会自动关闭 dropout，也不会阻止 buffer 更新。

最小 A 明确执行：冻结 baseline 模块置 eval，匹配/quality 可训练模块置 train；每次全局 `model.train()` 后重新应用此策略。基线状态与冻结 buffer 在 A 前后应一致。

为了分离原因，先做历史行为复现实验：仅关闭 global adapter，其他训练模式不变；随后再单独测试冻结主干 eval 模式。正式 A 才采用经验证的冻结行为，不能把两项变化都称为单变量。

### 5.4 A 的验收门槛

1. 同一 batch、`eval()`、相同 deterministic mask，A 的 raw proposal 与 source baseline `allclose`，FP32初始建议 `atol=1e-6, rtol=1e-5`。
2. 更新 A 若干步后仍满足候选不变；原参数/冻结 buffer 不变。
3. scope 内至少主要投影和 quality head 有有限非零梯度；旁路模块无有效更新。
4. AN oracle R5 应保持不变；CH oracle R8 应保持不变。CH selector R5 可以因排序变化而变化。
5. 记录 correct-vs-wrong-video margin、检索准确率和 quality 方差；不能只用 mc/pq loss 下降判定成功。
6. 若候选发生变化，直接判实现不合格，不通过调低学习率掩盖问题。

## 6. P3：统一 quality 训练与推理

### 6.1 最小实现选择

第一版统一使用 **clean/pre pair binding** 构造 quality 和 analytic；final grounded binding保留用于图传播、adapter及后续独立诊断。这样无需为每个错误 query-video 对运行完整 grounded backbone，并消除当前 pre/final 分布错位。

这属于修订方案，不是声称原实现已经如此。若后续选择 final binding，则必须为所有正负 query-video 对建立对等的条件化计算，不能正对使用final、负对使用pre。

### 6.2 共用特征函数

新增 `build_pair_quality_features()`，输入语义固定：

```text
pair_binding_logits    [Q,V,P,M]  clean/pre，已除 binding_temperature
phrase_required/valid  [Q,P]
query_edge_type        [Q,P,P]
node_mask/centers      [V,M]
proposal_membership   [V,N,M]
proposal_center/width [V,N]
proposal_soft_box     [V,N,M]
transition_score      [V,M]
```

输出 `features [Q,V,N,10]`，每一维定义如下：

| 维度 | 统一定义 |
|---:|---|
| 0 | required phrase 的 membership 加权 evidence coverage |
| 1 | 截去 membership<.1 尾部后的 exclusivity |
| 2 | 当前 query-video pair 的 relation satisfaction；无关系时采用统一neutral值 |
| 3 | relation_valid，显式区分无关系与关系满足 |
| 4 | 当前pair barrier与proposal soft box得到的connectivity |
| 5 | membership加权的内部relevance均值，不用全视频均值替代 |
| 6 | proposal左边界邻域relevance，不用视频第一个node替代 |
| 7 | proposal右边界邻域relevance，不用视频最后一个node替代 |
| 8 | clipped start/end之差得到的几何宽度，不用membership积分替代 |
| 9 | membership与soft box的平均绝对差 |

pair barrier 可由该pair的pre relevance、关系affinity及visual transition构造，默认detach。不能将正确query生成的diagonal barrier直接广播给所有错误query，再称其完全query-specific。

同一个函数同时服务训练和推理：推理取 `features[b,b]`，再调用同一个 quality head。analytic也由同一份 C/X/R/H 计算，无关系时的权重重分配保持一致。

为控制开销，cross-query以固定chunk处理，例如Q维每次8条。归约必须在完整batch范围完成；不能把chunk误当成InfoNCE负样本集合。chunk大小改变不得改变结果。

### 6.3 温度契约

当前 binding logits 已含 `1/binding_temperature`，后续evidence softmax又有温度。第一版不同时调温度数值，而是给每个温度独立命名并记录实际组合：

```text
cosine → /binding_temperature → binding_logits
binding_logits → /evidence_temperature → node evidence
query-video score → /contrastive_temperature → InfoNCE
```

所有train/eval路径使用同一个evidence温度。原loss诊断coverage不得继续采用另一套softmax口径。若认为组合温度过尖，应作为后续单变量试验，不与接口修复混在一起。

### 6.4 损失维度与梯度约束

维持 `pair_quality_logits [query,video,proposal]`；reconstruction selector 属于候选来源video的proposal轴：

```python
pi = softmax(-proposal_nll / reconstruction_temperature, dim=-1).detach()
score_qv = (pair_quality_logits * pi.unsqueeze(0)).sum(-1)
loss_pq = symmetric_infonce(score_qv / contrastive_temperature, valid_pairs)
```

现有 `unsqueeze(0)` 的轴语义正确，不应为修复quality特征而随意改成query轴权重。

首轮 A/B 均对 `L_pq` 的 proposal geometry detach，使其主要训练匹配/quality，避免再次用扩宽候选降低对比损失。是否允许 pq 更新geometry，必须独立消融。

需要承认：候选本身由video样本对应的原query产生，跨query重用这些候选仍可能形成结构捷径；统一特征不是完全消除该问题。记录same-video query-shuffle与跨video shuffle作为诊断，不把同视频所有其他query强行当成负例，因为同视频可能包含多个真实相关事件。

### 6.5 quality 验收

- `eval()`关闭dropout时，pair对角features与逐样本推理features逐维allclose。
- 同一候选的quality logit不因batch组成、顺序、padding或chunk大小变化。
- query不同但video相同的pair，有机会产生不同relevance/边界特征；不能仅复制diagonal。
- 记录 per-query候选quality标准差、与NLL相关性、validation上与IoU的Spearman相关性。
- 单独报告纯NLL、NLL+quality、NLL+analytic、完整打分；不把oracle候选用于实际预测。
- GT仅允许进入独立验证统计，不进入forward、quality训练target、negative mask或decoder。

## 7. P4：关系、gate、connectivity 的修复与取舍

### 7.1 before/after 方向

使用显式变量避免广播语义混淆：

```python
t_i = node_centers.unsqueeze(-1)  # [B,M,1]
t_j = node_centers.unsqueeze(-2)  # [B,1,M]
before = t_j >= t_i
after = t_j <= t_i
```

最低测试：节点`.1,.9`，before[0,1]>0且before[1,0]=0；after相反；交换phrase方向后满足相应对称关系。

再测真实 parser 生成的 “A before B” 和 “B after A” 对应边，不能只测孤立kernel。相等时间是否允许保留原规格的非严格不等号，不在本次修复中另改语义。

### 7.2 距离尺度

新增 `relation_distance_unit = node_index | normalized_time`。正式方案默认node_index，使`relation_decay=4`表示4个节点，而不是4倍视频时长。

节点索引模式下：距离1的ASSOC核应约.7788，距离4约.3679；simultaneous采用其自身更短衰减。新增stride变化测试，明确单位选择的含义。

历史legacy模式保留用于复评，不能用修复kernel评估旧权重后将差值全部归因“训练变好”。方向修复、单位修复独立做ablation。

### 7.3 Gate 不先强行硬稀疏

保留现有soft gate作为第一轮对照，新增gate的p10/p50/p90、std、`>0.1/>0.5/>0.9`比例以及pre relevance分布。active ratio≈1不是gate值全1。

若确认gate近常数且不贡献性能，优先做恒定gate对照，判断模块是否必要；之后再考虑稀疏gate。不要直接加大budget loss或改top-k，以免把表征问题隐藏为选点策略问题。

### 7.4 Connectivity

记录barrier均值、分位数、非零比例及transition/relevance差/关系affinity三个因子，解释H≈1来自哪一项。

正式最小 B 先令 `qstg_connect_weight=0`。只有barrier能对人工/合成不连续事件产生响应，且加入connect改善validation，才启用该loss。

不得把“让connectivity loss下降”作为验收条件；更重要的是固定候选上，连续/不连续案例的分数有区分能力。

## 8. P5：重新组织 Stage B/C

### 8.1 Stage B0：仅恢复安全的 proposal 适配

从已验收 A 初始化，主干冻结。允许global adapter与原proposal head训练，保持显式prior、importance bias、component residual关闭。

第一版损失：

```text
L = L_rec + L_ivc + L_event + L_mixture + .05 L_mc + .05 L_pq
fa/explain/gate/connect/role = 0
L_pq 对proposal geometry detach
```

这样 proposal 的初始优化信号来自原 reconstruction/legacy objectives，而不是同时加入所有弱几何约束。graph可经global adapter得到reconstruction梯度；如果global adapter仍零初始化，最初一步上游梯度可能为零，测试应允许合理的启动过程并检查后续步骤。

保留数据集原有差异：AN mixture、CH single Gaussian。CH不启用component residual/role；配置解析阶段直接拒绝无意义组合或输出明确inactive原因。

### 8.2 B 的增量模块逐项加入

建议顺序：

1. B0：上述最小组合。
2. B-fa：只加入fa，初始沿用既有.1权重和ramp作机制对照；若确认有害，不继续盲目增权重。
3. B-component：仅AN打开component residual。
4. B-prior：仅打开center/width prior .25。
5. B-importance：仅AN打开importance bias .25。
6. explain/gate/connect/role分别验证；不能一次全部打开后无法归因。

每个候选都从同一指定source初始化，以相同步数/seed比较；若采用逐步组合，需要明确这是“在已接受模块基础上的增量”，不是完全独立单因素主效应。

### 8.3 防退化监控

每epoch记录与B初始化的差：

- raw proposal宽度分布、center分布、pairwise IoU和重复率；
- oracle R5（AN）/R8（CH）、selector R5、R1；
- GT宽度分组召回；
- mixture component中心跨度、importance熵、assignment熵；
- 各loss对proposal head/global adapter/pre projections的梯度norm。

可将“宽度中位数快速增长且oracle R@.5下降超过2pp”设为预注册调查触发器，保存故障快照后暂停该候选；2pp只是工程警报，不是统计显著性阈值。正式接受/拒绝要结合视频级bootstrap区间和多seed。

### 8.4 Stage C 不再同时改变推理规则

从同一个合格B checkpoint比较：

- C-frozen-control：保持B scope；
- C-unfreeze：仅将scope改为all。

其他配置、训练步数、LR、初始化、event权重、selector、mask、prior、loss和schedule全部一致，先测解冻本身。需要降低C LR时，另做LR对照，不能同时把scope和LR变化称作“仅解冻”。

正式候选可在该结果基础上采用较小LR，但要保存其独立试验记录。CH的event推理权重必须在B/C固定，默认诊断线均为0；`.1`是否有益单独评估。

若C未在validation稳定优于B，则选择B结束本轮，不因方案写了三阶段就强制采用C。

## 9. 候选边界与 connected refinement

### 9.1 AN：先做无训练的几何诊断

固定同一个checkpoint和component，分别评估outer与weighted边界，报告它们对短/中/长GT的影响。该实验仅说明边界表示的作用，不证明weighted就是最终正确选择。

不能同时修改negative context的outer-envelope语义。若推理边界与训练reconstruction证据出现不一致，需要显式报告，并在后续训练实验处理。

增加oracle诊断：在现有候选中心/宽度限制下，最高能达到多少召回；高阈值失败是中心偏差、宽度偏差，还是候选根本缺失。

### 9.2 CH：补 oracle R8

CH的selector R5只是8取5，不是全部候选上限。输出：

```text
oracle_R8 - selector_R5   → top5筛选损失
selector_R5 - selector_R1 → top1选择损失
1 - oracle_R8            → 当前候选几何上限不足
```

这些差值按同一个IoU threshold计算；不要将不同阈值的差值相减。

### 9.3 Refinement 实施前安全修复

逐样本归一化：

```python
req_total = required.sum(-1).clamp_min(eps)  # [B]
c_seg = numerator_seg / req_total[:, None, None]
c_raw = numerator_raw / req_total[:, None]
```

最终gather前，best_start/end也必须安全clamp；没有候选时直接返回raw，并令refine_mask=false，不能先非法gather再用where回退。

统一refinement与raw quality的关系分数定义。增加batch不变性测试：相同样本单独评估、与不同phrase数样本拼batch，结果应一致。

开启后raw/refined同时记录、分别命名，默认选模仍使用raw，直到预先决定将refined纳入主协议。禁止静默替换raw结果。

## 10. 配置矩阵与单变量实验台账

以下是待新增配置的职责，而非当前可直接运行的文件。

| 配置名 | 关键差异 | 输入checkpoint | 主要问题 |
|---|---|---|---|
| `control_clean.json` | qstg=false，干净split，固定selector/mask | 干净初始化 | 同协议本地baseline |
| `a_fixed_proposals.json` | global adapter旁路、geometry detach、match_quality scope | clean control | A是否不再破坏候选 |
| `b_minimal.json` | 开global adapter+proposal，保留legacy，新增几何loss全0 | 合格A | 最小适配是否有净收益 |
| `b_add_fa.json` | 仅fa由0改.1 | 与B0同source | fa是否诱发退化 |
| `b_add_component.json` | 仅AN component residual=true | 指定B基础 | component残差是否有效 |
| `b_add_prior.json` | 仅proposal prior= .25 | 指定B基础 | 显式几何bias是否有效 |
| `c_scope_control.json` | B scope不变 | 同一个B checkpoint | 联合微调对照 |
| `c_unfreeze_only.json` | 仅scope=all | 同上 | 解冻本身是否有效 |

统一新增键建议：

```text
protocol_id
initialization_policy
checkpoint_selection_objective
model.config.detach_base_states
model.config.qstg.global_adapter_enabled
model.config.qstg.quality_binding_source = clean_pre
model.config.qstg.evidence_temperature
model.config.qstg.relation_distance_unit
loss.qstg_detach_proposal_geometry
train.frozen_backbone_eval
train.finite_check_enabled
diagnostics.oracle_all_proposals
diagnostics.gradient_probe_interval
```

初始LR可沿用A=2e-4、B=1e-4作为历史对照，batch32、warmup400，不在正确性修复时同时搜索LR。随后若要统一AN/CH的warmup epoch比例，应另列单变量试验。

每个实验台账至少包括：假设、唯一改变量、固定项、source hash、配置diff、预定步数、预定评价目标、接受/否定标准、实测成本、是否使用test。机器检查配置diff，发现多余变化则拒绝“单变量”标签。

## 11. 诊断、统计与报告规范

### 11.1 必须新增的日志

| 类别 | 字段 |
|---|---|
| Loss | rec、IVC、event、mixture、QSTG原始/加权分项、backward_total_loss、ramp、实际启用项 |
| Grad | 关键模块grad None比例、零比例、有限性、clip前后norm、间隔性分loss梯度探针 |
| Matching | 正/负score均值与margin、video retrieval R1、same-video排除比例 |
| Quality | pair正对/负对分布、候选内std、与NLL/analytic相关性、validation IoU相关性 |
| Geometry | width/center分位数、pairwise IoU、短GT召回、oracle全部proposal召回 |
| Structure | gate分位数、barrier分布、relation_valid比例、按关系类型分组结果 |
| Selector | NLL/quality/analytic/event各项尺度、argmin翻转率、combined entropy与NLL entropy分别命名 |
| Runtime | GPU型号、allocated/reserved峰值显存、train/eval时间、真实LR、global update |

分loss梯度探针只在固定小批次/固定间隔运行，避免每一步多次backward造成不可控成本；主训练梯度不能被探针污染。

### 11.2 聚合定义

- 同时给出epoch逐优化步loss均值和样本加权loss均值，明确分母。
- validation/test按query聚合指标；诊断分组给出样本数，空组返回NA而不是0。
- relation指标同时报告all-query和relation-valid子集，不能让无关系样本的neutral=1抬高总体结论。
- coverage/exclusivity的训练和推理诊断使用统一函数及温度，旧字段如保留则加legacy前缀。
- 正式不确定性用至少3个训练seed；标准差使用样本标准差。训练seed之外可做按video重采样的配对bootstrap，避免同视频query相关性被忽略。
- 固定eval mask seed与训练seed分开记录；否则多seed差异会混合训练随机性和评估mask随机性。
- 当前sample_uid依赖JSON行号；正式保留data hash和顺序。若改为内容hash UID，应单独认证mask协议变化，不能静默替换。

### 11.3 工程警报与研究结论区分

oracle recall掉2pp、宽度异常、gate接近常量等可触发调查，但不是预先证明某模块无效。最终结论至少同时检查：R1、全部候选oracle、阈值曲线、时长分组、多个seed及训练成本。

## 12. 测试计划与代码落点

优先扩展已有测试，不先建立另一套互不一致的测试框架。

| 文件 | 需要新增/调整的测试 |
|---|---|
| `tests/test_qstg_baseline_compat.py` | QSTG关闭parity；A候选固定；adapter旁路后多步更新仍不改raw候选 |
| `tests/test_qstg_integration.py` | runtime config转发；各scope参数与buffer更新白名单；B/C启动梯度 |
| `tests/test_qstg_quality.py` | pair/eval对角特征一致；10维语义；batch/chunk不变性；geometry detach |
| `tests/test_qstg_binding.py` | clean/pre来源；query shuffle敏感性；不同query不复用diagonal特征 |
| `tests/test_qstg_loss.py` | selector在video/proposal轴；same-video mask；B=1/all-same-video；零权重跳过；温度口径 |
| `tests/test_temporal_graph.py` | before/after方向；距离单位；padding、单节点、无效node |
| `tests/test_qstg_connected_decode.py` | 全空support回退；不同required数量的batch不变性；raw/refined明确区分 |
| `tests/test_charades_pipeline.py` | bootstrap来源检查；val/test不混用；污染来源不能标记clean |
| `tests/test_qstg_runner_smoke.py` | 非有限loss/grad中止；不生成正常best；来源manifest；精确resume边界 |
| `tests/test_proposal_selection.py` | 纯NLL和综合score分别报告；CH R8 oracle；所有threshold定义不变 |

测试分四层：CPU纯算子 → 小型CPU模块 → 小batch GPU前后向 → 一个短epoch集成。不要直接用长训练发现接口错误。

建议新增工具：

```text
tools/audit_protocol.py          # split、来源、版本与hash检查
tools/evaluate_selectors.py      # 固定输出比较打分；明确eval-only
tools/inspect_training_contract.py # scope、梯度、proposal固定性
tools/summarize_repair_runs.py    # 去重日志、聚合、配置diff、统一结果表
```

工具默认不改历史文件，任何写输出必须显式指定新目录。selector复评可在内存复用模型输出；若缓存到磁盘，必须显式许可并附model/data/mask hash。

## 13. 执行顺序、成本与停止条件

| 里程碑 | 工作 | 预计成本级别 | 退出条件 |
|---|---|---|---|
| M0 | 来源/split/协议manifest；历史同协议复评 | CPU审计＋少量eval | baseline定义、污染标签和主选模规则明确 |
| M1 | 配置、数值、方向、quality接口、refinement安全修复 | 开发＋CPU/小GPU测试 | 所有确定性测试通过 |
| M2 | A adapter单变量诊断，随后正式固定候选A | 每数据集1–2epoch先筛查 | raw候选严格保持；匹配与quality有有效学习 |
| M3 | 干净Charades control训练；AN协议确认 | 主要训练成本 | 可作为正式source的baseline就绪 |
| M4 | B0与fa等逐项消融 | 每个候选先3–4epoch | 不以召回明显退化换少量R1提升 |
| M5 | C scope对照、边界/排序诊断 | 少量eval＋短训 | C有稳定增益才保留 |
| M6 | 固定方案，至少3seed，最后一次正式test | 最终主要成本 | 完整均值/std及可复现材料 |

M2历史诊断可早于M3执行，但不能作为正式结果。M1的正确性修复与性能实验必须保留版本区分，便于解释旧结果。

历史参考：AN一次评估约2分钟、训练每epoch数分钟；CH评估为十几秒到数十秒、训练约2分钟/epoch，受数据集split和GPU影响。不能承诺固定GPU小时数，先测新诊断开销再预算。bootstrap完整重训练单独计费/计时。

停止规则：

- 发现非有限值、候选固定性失败、split冲突、来源缺失：停止该运行，先修正确性。
- A匹配loss下降但shuffle margin无改善：不进入复杂B组合，先查匹配捷径。
- B只恢复到baseline：如实报告“恢复”，不称为新增模块增益。
- C没有稳定收益：最终采用B，不强制三阶段。
- quality去掉后指标更好：暂时不用于主selector，先修训练目标/输入，不盲目加权。
- geometry不合格时，不用增加N或更强投票掩盖；先记录oracle上限与短GT失败。

## 14. 最终交付清单

- [ ] 本方案中的新接口已落地并有配置校验，文档不再含未标注的虚构开关。
- [ ] 历史材料完整保留，新run使用独立目录。
- [ ] 数据与初始化协议报告，Charades正式source无已知污染。
- [ ] 同协议local_control结果，而不只是论文CPL参考数。
- [ ] Stage A候选不变测试及实际训练检查。
- [ ] quality pair/eval一致性测试和独立排序收益报告。
- [ ] relation方向与距离单位测试，修复贡献分开报告。
- [ ] B/C的来源、scope、schedule和实际selector可追溯。
- [ ] raw/refined、selector/oracle、train/val/test严格分开命名。
- [ ] 每epoch损失、诊断、数值检查、真实LR、显存与耗时记录。
- [ ] 单变量消融台账和至少3seed统计。
- [ ] 最终test按预定规则执行，不拼接不同checkpoint的最优列。

**最短可行路线：先把A改成真正固定候选的匹配/质量学习，再统一quality输入；用干净baseline验证B0，最后只保留被单变量实验支持的图约束与联合微调。**
