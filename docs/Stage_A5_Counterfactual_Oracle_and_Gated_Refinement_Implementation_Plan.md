# Stage A.5：候选 Oracle、反事实语义检验与门控式边界收缩实施与进展

> 最后更新：2026-09-02  
> 当前总决策：`STAGE_B_GO=false`。P0 协议修正与回归已完成；P1 全量 validation 因当前环境无可用 GPU 且 CPU full-model 前向耗时过长而阻塞，未用 smoke 结果替代正式证据。

## 1. 文档目的

本文现在同时承担两项职责：记录 `/data/chenyuan/videogrounding/w1m2/stageA` 的 Stage A/Stage A.5 已完成工作，以及约束后续完整实验和 Stage B Go/No-Go 决策。原始技术设计继续保留，已经实现的部分用当前代码和实验产物校准，不再把本文视为纯“待实现方案”。

Stage A.5 不是新的训练阶段，也不更新模型参数。它位于现有 Stage A 和文档中的 Stage B 之间，目标是回答两个问题：

1. 事件边界生成的 inward-trim 候选中，是否真的包含比原 proposal 更接近 GT 的区间？
2. 如果包含，能否用不依赖 GT、且比单向 reconstruction NLL 更可靠的规则选出它们？

只有这两个问题都得到肯定答案，才允许进入 Stage B。相关文档：

- `docs/Stage_A_Event_Boundary_Minimal_Sufficient_Trim_Implementation_Plan.md`
- `docs/IPR_Label_Propagation_for_cpl_lrev_Analysis.md`

### 1.1 整个 Stage A 的当前进度

| 环节 | 当前状态 | 已得到的结论 |
|---|---|---|
| 事件边界提取与索引 | 已完成 | ActivityNet-C3D 与 Charades-I3D 的事件边界已可供 Stage A 读取 |
| 基础 CPL 训练 | 已完成 | 两个数据集的 baseline checkpoint 已存在，包含 best-r1、best-composite 和 best-r5 |
| Stage A：NLL 最小充分收缩 | 已完成完整推理 | 能显著缩短 proposal，但两个数据集总体定位指标均下降，不能直接进入 Stage B |
| Stage A.5 工程实现 | 核心链路已完成 | 已实现确定性多 mask、候选/标签分离导出、Oracle/Regret、shell 对照、GT-free selector、扫描与冻结评估 |
| Stage A.5 测试 | 已完成当前测试集 | `stageA/tests` 共 `36 passed`；包含 P0 协议、扫描覆盖、partial/Charades 诊断和 GT 泄漏保护测试 |
| Stage A.5 smoke | 已完成 | 两个数据集各只运行了 1 个 query/1 个 video，仅能证明链路可执行 |
| Stage A.5 正式 validation | P1 阻塞 | ActivityNet 17,505 条 validation query、三个 checkpoint 尚未完整导出与评估；当前环境 `torch.cuda.is_available()=False`、GPU 数为 0，`nvidia-smi` 无法连接 NVIDIA driver；CPU full-model 单 batch 已因耗时过长中止 |
| Stage B / Label Propagation | 未开始，且当前禁止开始 | 尚无可靠的高置信正伪标签来源，当前 `STAGE_B_GO=false` |

### 1.2 当前决策的含义

- 当前结论不是“Stage A.5 已失败”，而是“Stage A.5 尚未完成科学验证”。
- `stageA/stageA5_tmp/stagea5_report.json` 中的 `STAGE_B_GO=true` 来自 6 条人工构造的 synthetic fixture，checkpoint 为 `/tmp/c.pt`，只能证明评估工具能产生正例，不能作为真实数据集结论。
- ActivityNet/Charades smoke 的正式字段均为 `STAGE_B_GO=false`；这些 partial 产物现在被 P0 协议保护，默认不能进入正式分析、扫描或冻结。
- 下一步不是实现 Stage B，而是先修正正式扫描前的三个协议问题，再完成 ActivityNet 全量 validation。

---

## 2. 当前证据与设计出发点

现有 Stage A 已证明：

- ActivityNet 的平均 proposal 宽度从 `0.6578` 降到 `0.5520`，但 R@1 mIoU 从 `0.3764` 降到 `0.3492`；
- Charades 的平均 proposal 宽度从 `0.3197` 降到 `0.2318`，R@1 mIoU 从 `0.4392` 降到 `0.3766`；
- 即使 `epsilon=0`，即只接受 NLL 不高于原 proposal 的候选，两个数据集仍然退化；
- 被选候选的平均 NLL 反而更低，但定位指标更差；
- ActivityNet 的短 GT 有局部收益，长 GT 明显受损；
- Charades 原本没有 near-whole-video 问题，却有 `84.19%` 的 proposal 被收缩。

因此不能再使用下面的蕴含作为伪标签依据：

\[
L_{cap}(v_{trim})\le L_{cap}(u)
\quad\Rightarrow\quad
v_{trim}\text{ 比 }u\text{ 更正确}.
\]

Stage A.5 必须把“候选是否有潜力”和“选择器是否能识别潜力”分开验证。任何新的选择规则都必须保留原 proposal，并以高精度、低覆盖的方式逐步开放收缩。

### 2.1 Stage A 完整推理结果

以下结果来自现有完整 test 推理日志，默认对比 `epsilon=0.02`：

| 数据集 | 指标 | Baseline | Stage A | 差值 |
|---|---:|---:|---:|---:|
| ActivityNet | R@1 mIoU | 0.3764 | 0.3492 | -0.0272 |
| ActivityNet | R@5 mIoU | 0.5235 | 0.5107 | -0.0128 |
| ActivityNet | mean width | 0.6578 | 0.5520 | -0.1058 |
| ActivityNet | changed fraction | — | 0.6495 | — |
| Charades | R@1 mIoU | 0.4392 | 0.3766 | -0.0626 |
| Charades | R@5 mIoU | 0.6798 | 0.6161 | -0.0637 |
| Charades | mean width | 0.3197 | 0.2318 | -0.0879 |
| Charades | changed fraction | — | 0.8419 | — |

即使把阈值收紧为 `epsilon=0`，ActivityNet 的 R@1 mIoU 仍降至 `0.3541`，Charades 仍降至 `0.3916`。这排除了“仅仅是 epsilon 过大”的解释。ActivityNet 的 long-GT R@5 mIoU 从 `0.8757` 降至 `0.7766`，说明对长事件的错误收缩是核心风险；Charades 原本 near-whole-video 比例为 `0`，却仍有 `84.19%` proposal 被修改，说明旧选择器没有把收缩限制在真正的长区间问题上。

### 2.2 Stage A.5 已完成的工程证据

- 已加入稳定 `sample_id` 和按 `sample_id + seed` 生成的确定性文本 mask；默认种子为 `[8, 18, 28]`。
- 已实现 `original + 6 trim` 的候选导出，features 与 GT labels 分离保存。
- 已实现 raw trim/removed-shell 互补 mask、trim/shell NLL、contrast 均值和多 mask 标准差。
- 已实现 parent/query oracle、legacy selector regret、NLL 判别力、GT-width 分桶和 video-cluster bootstrap。
- 已实现不接收 GT 的保守 selector、reason code、分层参数扫描和冻结配置评估。
- 使用 `cpl` 环境运行 `/home/chenyuan/miniconda3/envs/cpl/bin/python -m pytest -q stageA/tests`，结果为 `36 passed`。当前测试已覆盖 partial/Charades 诊断和 GT 洩漏保护，但仍缺少全量数据集集成验证。

### 2.3 当前 smoke 结果及其边界

| 数据集 | 实际样本 | Candidate-set oracle | 被选配置 | 结论 |
|---|---:|---:|---:|---|
| ActivityNet | 1 query / 1 video / 5 parents | mIoU gain `0.0000` | changed `0` | `STAGE_B_GO=false` |
| Charades | 1 query / 1 video / 8 parents | mIoU gain `+0.0874` | R@1 mIoU `+0.0235`，R@5 不变 | `STAGE_B_GO=false` |

ActivityNet validation 实际有 17,505 条 query，当前 smoke 覆盖约 `0.006%`；Charades test 有 3,158 条 query，且现有配置中 `val_data == test_data`。因此：

- 单视频 bootstrap 是退化的，不能解释为置信区间证据；
- 在一个 query 上扫描 414 个配置会严重过拟合；
- smoke 中的 NLL Spearman/AUC（ActivityNet `-0.462/0.667`，Charades `-0.374/0.576`）样本量过小，不能判断语义判别力；
- Charades smoke 不能被称为 validation，更不能据此冻结参数。

---

## 3. 总体决策流程

```text
[已完成] 冻结 checkpoint、实现确定性三 mask
                         │
                         ▼
[已完成] original + 6 trim + shell 的导出/分析/选择工具
                         │
                         ▼
[已完成] synthetic 与单 batch GPU smoke
                         │
                         ▼
[当前位置] 修正正式扫描协议，运行 ActivityNet 全量 validation
                         │
                         ▼
          三 checkpoint candidate oracle / regret / selector
                         │
                 ┌───────┴────────┐
                 │ 无稳定增益      │ 有稳定增益且 selector 过 gate
                 ▼                ▼
       停止收缩与 Stage B       冻结唯一配置
       改候选生成/基础模型          │
                                  ▼
                         仅一次 ActivityNet test
                                  │
                           ┌──────┴──────┐
                           │ test 不稳定 │ test 保持正向
                           ▼             ▼
                       不进入 Stage B  再评估 Stage B 伪标签
```

Stage A.5 分为两个必须串行执行的子阶段：

- **A.5-1：候选导出与 Oracle/Regret 诊断**。不改变候选生成，不增加新语义分数。
- **A.5-2：反事实语义检验与保守门控选择**。仅在 A.5-1 证明候选集合有效后实现。

这里的串行约束现在作用于“实验决策”而不是代码编写：A.5-2 代码已提前完成，但正式实验仍必须先看 A.5-1 oracle；oracle No-Go 时不运行大规模 selector scan，也不以已写好的 A.5-2 代码为理由推进 Stage B。

---

## 4. 实现边界与完成状态

### 4.1 已实现

- 确定性的评估文本 mask；
- 候选级 NPZ 导出；
- parent-level 和 query-level oracle；
- 当前 Stage-A selector regret；
- `delta_nll` 与 `delta_iou` 的相关性、分类能力和分桶统计；
- candidate interior 与 removed shell 的双向反事实 NLL；
- 不使用 GT 的保守门控选择器；
- validation 参数扫描、配置冻结和 test 一次性评估；
- video-cluster paired bootstrap 置信区间；
- 单元测试与集成测试。

上述核心能力均已落地。这里的“已实现”只表示代码路径存在且当前测试/smoke 可运行，不表示已经在完整 validation 上得到有效结论。

### 4.2 尚待完成的实验与防误用约束

- 已修复正式分层扫描的覆盖与配置选择问题；
- 已禁止 `partial=true` 的导出产物给出正式 `STAGE_B_GO=true`；
- 完成 ActivityNet 三类 checkpoint 的全量 validation；
- 对冻结配置补做 validation cluster bootstrap，并逐条核验 Stage B gate；
- 仅在 validation gate 通过后运行一次 test；
- 若要在 Charades 上给正式结论，先构造 video-disjoint validation 并重新训练 baseline；
- 增加 no-GT-leakage/partial-artifact/Charades-test-as-val guard 测试。

### 4.3 本阶段明确不实现

- 参数训练或微调；
- Stage B 的 boundary loss、ranking loss 或 coordinate loss；
- Label Propagation；
- 使用 test GT 调参；
- 删除原 proposal fallback；
- 默认 outward expansion；
- 把 oracle 选择结果作为推理输出；
- 把 GT、GT width 或由 GT 计算的特征传入在线选择器。

---

## 5. 已落地代码与输出目录

当前已经落地的 Stage A.5 主要文件如下：

```text
stageA/
├── datasets/base.py                       # raw 中加入稳定 sample_id
├── models/cpl.py                          # 确定性 mask override、trim/shell 解码
├── models/modules/
│   └── event_boundary_refinement.py       # raw trim/shell mask 与边界置信度
├── runners/
│   ├── main_runner.py                     # Stage A.5 在线评估接入
│   └── stage_a5.py                        # GT-free selector 与 reason code
├── tools/
│   ├── dump_stage_a5_candidates.py        # GPU 前向与候选导出
│   ├── analyze_stage_a5_oracle.py          # 纯离线 oracle/regret 分析
│   ├── scan_stage_a5_selectors.py          # 纯离线 selector 扫描
│   ├── evaluate_stage_a5.py                # 使用冻结配置评估
│   ├── make_charades_val_split.py          # 可选，正式 Charades 协议
│   └── stage_a5_utils.py                   # schema、metrics、bootstrap 公共函数
├── config/
│   ├── activitynet/stage_a5.json
│   └── charades/stage_a5.json
└── tests/
    ├── test_deterministic_eval_masks.py
    ├── test_stage_a5_candidate_dump.py
    ├── test_stage_a5_oracle.py
    ├── test_stage_a5_shell_masks.py
    └── test_stage_a5_selector.py
```

正式实验产物仍必须统一保存到：

```text
stageA/outputs/stage_a5/
├── activitynet/
│   └── <checkpoint_label>/
│       ├── val_candidates_features.npz
│       ├── val_candidates_labels.npz
│       ├── val_oracle_summary.json
│       ├── val_oracle_rows.csv
│       ├── selector_scan.csv
│       ├── selected_stage_a5_config.json
│       ├── val_frozen_report.json
│       ├── test_candidates_features.npz
│       ├── test_candidates_labels.npz
│       └── test_report.json
└── charades/
    └── <checkpoint_label>/...
```

不要覆盖 `stageA/logs/`、现有 checkpoint 或现有 Stage A 输出。

当前已有产物位于 `stageA/stageA5_tmp/`，性质如下：

- `stagea5_activitynet_smoke/`：ActivityNet best-r1，`partial=true`，1 query；
- `stagea5_charades_smoke/`：Charades best-r1，`partial=true`，1 query；
- 根目录下的 `stagea5_*`：synthetic fixture；
- `stageA/outputs/stage_a5/`：尚无完整 validation/test 正式产物。

`stageA5_tmp` 只能用于调试和回归，正式报告不得从这里读取 Go/No-Go 结论。

---

## 6. 前置修正：确定性评估协议

**实现状态：已完成并通过测试。** 正式实验仍必须固定使用同一组 mask seeds，不能退回单次随机 mask。

### 6.1 问题

修正前的 `CPL._mask_words` 在 `eval()` 下仍调用 `np.random.choice`。虽然一次 forward 内 baseline 和 Stage A 共享同一个 masked query，二者是配对的，但当时存在以下问题：

- 不同运行之间不能严格复现；
- 不同 checkpoint 可能看到不同 mask；
- 单次 NLL 对候选的判断方差未知；
- selector scan 可能把 mask 噪声当作真实增益。

### 6.2 数据侧稳定 sample id

Dataset 的每个样本增加稳定标识，但保持现有 `raw` 前四项兼容：

```python
sample_id = "{}:{}".format(video_id, dataset_index)

raw = [video_id, duration, gt_interval, sentence, sample_id]
```

禁止使用 Python 内置 `hash()`，因为它可能随进程变化。需要随机种子时使用：

```python
digest = hashlib.sha256(
    "{}:{}".format(sample_id, eval_mask_seed).encode("utf8")
).digest()
sample_seed = int.from_bytes(digest[:8], "little")
```

### 6.3 显式 mask override

为 `CPL.forward` 增加可选参数：

```python
eval_word_mask: Optional[torch.BoolTensor]  # [B, max_words + 1]
```

修改 `_mask_words`：

```python
def _mask_words(self, words_feat, words_len, weights=None,
                mask_override=None):
```

要求：

- 训练时 `mask_override` 必须为 `None`，完全保留现有随机逻辑；
- 推理若提供 override，不能再次调用 RNG；
- 每个样本 mask 数量仍为 `max(word_len // 3, 1)`；
- mask 只能落在 `[1, word_len]`；
- candidate original/trim/shell 必须共享同一份 mask。

### 6.4 多 mask 聚合

默认评估三个固定 mask seed：

```json
"eval_mask_seeds": [8, 18, 28]
```

对同一候选记录：

\[
\bar L(c)=\frac{1}{M}\sum_{m=1}^{M}L^{(m)}(c),
\qquad
\sigma_L(c)=Std_m(L^{(m)}(c)).
\]

选择器使用 `nll_mean`，`nll_std` 用作不确定性门控。调试可只用一个 seed，正式结果必须使用三个或以上固定 seed。

### 6.5 Charades 验证协议

当前 Charades 配置中 `val_data == test_data`。Stage A.5 不允许在这个 split 上扫描参数后再汇报同一 split。

允许两种模式：

1. **现有 checkpoint 的诊断模式**：参数只在 ActivityNet validation 或预先登记的固定规则上确定；Charades test 只运行一次，不能根据结果改参数。
2. **正式 Charades 模式**：按 video id 从原训练集划出固定 validation，随后重新训练 baseline。旧 checkpoint 已见过完整 train，不能把事后划出的子集宣称为真正 validation。

正式划分要求：

- video-disjoint；
- 固定 seed `20260902`；
- 按视频划分约 10%，不是按 query 随机划分；
- 输出 split manifest，记录 video id、源文件 SHA256、seed 和样本数量；
- `test.json` 在配置冻结前不得读取 GT 指标。

---

## 7. A.5-1：候选级数据导出

**实现状态：已完成并通过 synthetic/GPU 单 batch smoke。** 当前只缺全量 validation 导出；正式运行不得传 `--max-batches`。

### 7.1 导出粒度

对每个 query、每个 parent proposal、每个候选导出一行逻辑记录。仍保持张量化 NPZ，不使用 object array 或 pickle。

推荐 NPZ schema：

```python
{
    "schema_version": int32[1],
    "dataset": unicode[1],
    "split": unicode[1],
    "checkpoint_path": unicode[1],
    "checkpoint_sha256": unicode[1],
    "config_sha256": unicode[1],
    "mask_seeds": int64[M],

    "sample_ids": unicode[Q],
    "video_ids": unicode[Q],
    "durations": float32[Q],
    "gt_normalized": float32[Q, 2],       # 仅分析工具读取

    "parent_start": float32[Q, N],
    "parent_end": float32[Q, N],
    "parent_event_score": float32[Q, N],

    "candidate_start": float32[Q, N, 7],
    "candidate_end": float32[Q, N, 7],
    "candidate_valid": bool[Q, N, 7],
    "candidate_type": int8[7],
    "candidate_nll_mean": float32[Q, N, 7],
    "candidate_nll_std": float32[Q, N, 7],
    "candidate_left_boundary_score": float32[Q, N, 7],
    "candidate_right_boundary_score": float32[Q, N, 7],
    "candidate_boundary_confidence": float32[Q, N, 7],

    "legacy_selected_index": int8[Q, N],
    "metadata_json": unicode[1],
}
```

`gt_normalized` 只允许出现在导出文件的分析分区。在线选择器 API 不接收包含 GT 的对象。若需要更强的防泄漏，可同时生成：

- `*_features.npz`：无 GT，可用于 selector；
- `*_labels.npz`：只有 `sample_id` 和 GT，仅 oracle 工具加载。

正式实现推荐采用双文件形式。

### 7.2 边界置信度

绝对 boundary score 在不同视频间可能不可比。应在每个视频内部计算 percentile rank：

\[
r_j=\frac{rank(a_j)-1}{\max(J-1,1)}.
\]

候选的边界置信度定义为：

```text
original: 0
left trim: left rank
right trim: right rank
both trim: min(left rank, right rank)
```

使用 `min` 是为了防止 both-trim 只靠一侧强边界通过门控。

### 7.3 一致性断言

导出前必须检查：

- candidate 0 与 original 的坐标逐元素相等；
- candidate 0 的 NLL 与 baseline parent NLL 在 `1e-6` 内相等；
- 所有 valid trim 都满足 inward-only；
- `end > start`；
- invalid candidate 的 NLL 为 `inf`，且不会进入 selector；
- 同一个 sample 的三个 mask seed 对应完全相同的几何候选；
- sample id 在同一 split 中唯一；
- 导出行数与 Dataset 长度一致。

### 7.4 CLI

```bash
cd /data/chenyuan/videogrounding/w1m2/stageA

PYTHONPATH=. python tools/dump_stage_a5_candidates.py \
  --config-path config/activitynet/stage_a5.json \
  --checkpoint checkpoints/activitynet/<run>/model-best-r1.pt \
  --checkpoint-label activitynet_best_r1 \
  --split val \
  --mask-seeds 8,18,28 \
  --output outputs/stage_a5/activitynet/activitynet_best_r1/val_candidates
```

`--output` 是前缀，工具应生成分离的 `*_features.npz` 和 `*_labels.npz`。

---

## 8. A.5-1：Oracle 与选择器 Regret

**实现状态：分析工具已完成。** P0 已增加 partial 和 Charades test-as-validation 保护；由于 P1 全量 validation 被环境阻塞，本节的 A.5-1 Go/No-Go 尚未在真实 validation 上执行完成。

### 8.1 IoU 定义

对 query `q`、parent `n`、candidate `k`：

\[
I_{qnk}=IoU(c_{qnk},GT_q).
\]

candidate 0 是原 proposal，所以所有 oracle 都不会比对应原集合差。

### 8.2 Parent-level oracle

\[
k^*_{qn}=\arg\max_k I_{qnk},
\]

\[
G^{parent}_{qn}=I_{qn k^*}-I_{qn0}.
\]

输出：

- mean/median parent oracle gain；
- `gain > 0.01`、`gain < -0.01` 和 neutral 的比例；
- 最优候选类型分布；
- 按 parent width、GT width、boundary confidence 分桶。

理论上 parent oracle 不应为负；若出现负值说明 candidate 0、IoU 或 validity 实现有错误。

### 8.3 Query-level candidate-set oracle

原始 proposal 集合上限：

\[
I_q^{orig}=\max_n I_{qn0}.
\]

加入 trim 后的集合上限：

\[
I_q^{cand}=\max_{n,k} I_{qnk}.
\]

候选集合增益：

\[
G_q^{set}=I_q^{cand}-I_q^{orig}.
\]

报告 candidate-set oracle 的 mIoU 和 IoU@`0.3/0.5/0.7`，但必须明确标为上限，不能与可部署模型混为一谈。

### 8.4 Legacy selector regret

现有 Stage A 对每个 parent 的选择为 `hat{k}`：

\[
R_{qn}=I_{qnk^*}-I_{qn\hat k}.
\]

同时统计实际变化：

\[
\Delta I^{legacy}_{qn}=I_{qn\hat k}-I_{qn0}.
\]

至少输出：

- helpful：`delta_iou > 0.01`；
- harmful：`delta_iou < -0.01`；
- neutral；
- `precision_trim = helpful / (helpful + harmful)`；
- mean regret；
- changed fraction；
- 按数据集、GT width、parent width、candidate type 分桶。

### 8.5 NLL 是否有判别力

定义：

\[
\Delta L_{qnk}=L(c_{qnk})-L(c_{qn0}),
\qquad
\Delta I_{qnk}=I_{qnk}-I_{qn0}.
\]

计算：

- Spearman `rho(delta_nll, delta_iou)`；
- 使用 `-delta_nll` 判断 `delta_iou > 0.01` 的 ROC-AUC；
- `delta_nll` 十分位中的平均 `delta_iou`；
- NLL 方差 `candidate_nll_std` 与错误选择率的关系；
- `epsilon=0/0.01/0.02/0.05` 下 helpful/harmful 比例。

如果 AUC 接近 `0.5` 或相关性不稳定，则禁止在 Stage B 中继续用单向 NLL 生成 ranking label。

### 8.6 A.5-1 Go/No-Go

仅当下面条件同时满足，才实施 A.5-2：

1. candidate-set oracle 在 validation 上至少有一个主要指标获得明确正增益；
2. 增益不是只来自极小样本桶；
3. 至少存在可观比例的 helpful trim；
4. 原 proposal fallback 和 oracle 计算通过所有一致性检查。

如果 candidate-set oracle 几乎没有增益，立即停止 Stage A/Stage B 的边界收缩路线，优先改候选生成或原 proposal generator。

---

## 9. A.5-2：Removed-shell 反事实语义检验

**实现状态：已完成并通过 shell mask 测试。** 当前已有 `candidate_shell_nll_*` 与 `candidate_contrast_*` 导出字段。

### 9.1 第一性原理

单向 sufficiency 只问：

```text
保留区间能否重建文本？
```

语言模型可能只凭未 mask 的词完成重建。Stage A.5 增加一个对照问题：

```text
被删除的外侧内容能否同样重建文本？
```

只有保留区间显著优于 removed shell 时，才说明收缩方向具有相对语义证据。

### 9.2 Shell mask

设原 mask 为 `m_orig(t)`，由候选坐标产生的未归一化软窗口为 `w_c(t)`：

\[
m_{trim}^{raw}(t)=m_{orig}(t)w_c(t),
\]

\[
m_{shell}^{raw}(t)=m_{orig}(t)(1-w_c(t)).
\]

trim 与 shell 分别除以自身最大值；最大值小于 `1e-6` 时标为 invalid。不要用“已归一化 trim mask”直接从 original 相减，否则重新归一化可能导致负值和错误补集。

original candidate 不需要 shell NLL，其 shell validity 为 false。

### 9.3 三个反事实量

对每个 trim candidate 计算：

充分性代价：

\[
D_{suff}=L_{trim}-L_{orig}.
\]

保留区间相对 shell 的优势：

\[
D_{contrast}=L_{shell}-L_{trim}.
\]

多 mask 不确定性：

\[
U=Std_m(D_{contrast}^{(m)}).
\]

理想候选应满足：

```text
D_suff 小：收缩后仍可解释文本
D_contrast 大：保留区间比被删除内容更相关
U 小：结论不依赖某一次词 mask
```

需要在候选导出中增加：

```python
candidate_shell_nll_mean: float32[Q, N, 7]
candidate_shell_nll_std: float32[Q, N, 7]
candidate_contrast_mean: float32[Q, N, 7]
candidate_contrast_std: float32[Q, N, 7]
```

### 9.4 分块解码

继续复用当前 Stage A 的 chunked decoder。trim 和 shell 分开打包，仅解码 valid rows，并在一个 mask seed 完成后立即丢弃 logits。

不得一次物化 `[B, N, 7, M, words, vocab]`。

---

## 10. A.5-2：保守门控选择器

**实现状态：GT-free selector 与扫描器已完成；第 16.1 节的协议修正已完成。正式扫描仍需等待 P1 validation。**

### 10.1 选择原则

选择器仍逐 parent 工作。默认返回 original，只在所有硬门槛通过时允许 trim。

选择器函数签名建议为：

```python
def select_stage_a5_candidates(
    candidate_start,
    candidate_end,
    candidate_valid,
    candidate_nll_mean,
    candidate_nll_std,
    shell_nll_mean,
    shell_nll_std,
    boundary_confidence,
    config,
):
    """GT-free selector; returns refined props, scores, indices, reasons."""
```

函数参数中不允许出现 `gt`、`iou`、`duration_bucket` 或任何由 GT 派生的量。

### 10.2 硬门控

候选必须同时满足：

```text
candidate_valid
parent_width >= min_parent_width
retained_ratio >= min_retained_ratio
endpoint_shift / parent_width <= max_relative_shift
D_suff <= max_nll_increase
D_contrast >= min_contrast_margin
contrast_std <= max_contrast_std
boundary_confidence >= min_boundary_percentile
candidate type 在 allowed_candidate_types 中
```

第一轮只开放单侧 trim：

```json
"allowed_candidate_types": ["left_near", "left_strong",
                              "right_near", "right_strong"]
```

`both_near` 和 `both_strong` 只有在单侧方案通过验收后才能单独做消融，不能直接默认启用。

### 10.3 候选评分

在通过硬门控的候选中最小化：

\[
S(c)=
D_{suff}
-\lambda_C D_{contrast}
+\lambda_E r_{edit}
+\lambda_U U
-\lambda_B a_{boundary},
\]

其中：

\[
r_{edit}=1-\frac{width(c)}{width(u)}.
\]

注意 `lambda_E` 是收缩惩罚，防止重新退化为“越短越好”。原 proposal 的分数固定为 `0`。trim 除了通过硬门控，还必须满足：

\[
S(c)<-m_{accept}
\]

否则回退 original。

### 10.4 推荐初始扫描空间

以下只用于 validation 网格，禁止直接在 test 上扫描：

```json
{
  "min_parent_width": [0.35, 0.50, 0.65],
  "min_retained_ratio": [0.60, 0.75, 0.85],
  "max_relative_shift": [0.10, 0.20, 0.30],
  "max_nll_increase": [0.00, 0.01],
  "min_contrast_margin": [0.00, 0.01, 0.02, 0.05],
  "max_contrast_std": [0.01, 0.03, 0.05],
  "min_boundary_percentile": [0.50, 0.70, 0.85],
  "lambda_contrast": [0.5, 1.0, 2.0],
  "lambda_edit": [0.1, 0.25, 0.5],
  "lambda_uncertainty": [0.25, 0.5, 1.0],
  "lambda_boundary": [0.0, 0.1, 0.25],
  "accept_margin": [0.00, 0.01, 0.02]
}
```

完整笛卡尔积过大。工具应采用逐层扫描：

1. 固定默认 score 权重，先扫描几何硬门控；
2. 保留 validation 表现和 trim precision 最好的前 10 个配置；
3. 对前 10 个配置扫描语义阈值；
4. 最后只对前 5 个配置扫描 score 权重；
5. 使用预先定义的 composite objective 选一个配置并冻结。

### 10.5 Validation 选择目标

不能只优化单一 R@1 mIoU，否则容易牺牲长 GT。建议：

\[
J=
\Delta R1\_mIoU
+0.5\Delta R5\_mIoU
+0.5\Delta R1@0.5
-2P_{harm}
-P_{long},
\]

其中：

- `P_harm` 是 harmful trim 比例；
- `P_long=max(0, -delta_long_R5_mIoU)`；
- 如果没有 long 样本，该项跳过而不是置零参与跨数据集比较。

同时设置硬约束：任何主要总体指标或主要 GT bucket 超过允许退化量，配置直接淘汰。

---

## 11. 统计检验与报告

**实现状态：统计函数和报告字段已实现；完整 validation 的有效 bootstrap 报告尚未产生。** 单 query smoke 的 cluster bootstrap 不具备统计意义。

### 11.4 P0 协议修正记录

2026-09-02 已完成并通过 CPL 环境回归：

- selector 的 weight grid 改为可复用完整笛卡尔积；每个保留的 semantic parent 都扫描全部权重组合，默认扫描行数按理论网格计数；
- 配置选择改为 gate-first；存在 gate-passing 配置时不再被更高 objective 的 no-change 配置覆盖；selected JSON 记录扫描总数、gate-passing 数量、选择原因和 `STAGE_B_GO`；
- features/labels metadata 增加数据文件路径、`validation_is_test`、`partial` 和 `query_count`；Oracle、scan、evaluate 默认拒绝 partial/test-as-validation 产物；显式诊断模式强制 `diagnostic_only=true`、`STAGE_B_GO=false`；
- Charades `val_data == test_data` 的诊断 scan 仅运行固定 shipped selector 规则，不进行参数扫描，也不输出 validation-frozen 语义；
- `/home/chenyuan/miniconda3/envs/cpl/bin/python -m pytest -q stageA/tests`：`36 passed, 3 warnings`。

### 11.1 配对统计

所有 Stage A.5 与 baseline 指标必须基于同一 checkpoint、同一 sample、同一组 mask seeds。报告绝对值和差值。

### 11.2 Video-cluster bootstrap

同一视频可能有多条 query，不能把它们当完全独立样本。按 video id 有放回抽样，重复至少 2000 次，计算：

- `delta R@1 mIoU`；
- `delta R@1 IoU@0.5`；
- `delta R@5 mIoU`；
- 各 GT-width bucket 的差值；
- helpful/harmful trim 比例。

输出 95% percentile confidence interval。

### 11.3 必需报告字段

每次实验至少报告：

- checkpoint 和 SHA256；
- config 和 SHA256；
- split、mask seeds、query 数、video 数；
- baseline 与 Stage A.5 全部 R@1/R@5 指标；
- changed fraction；
- mean width before/after；
- helpful/harmful/neutral；
- trim precision；
- candidate-set oracle；
- selector regret；
- 按 GT width 的指标；
- 按 candidate type 的选择率和收益；
- paired bootstrap CI；
- 是否通过 Stage B gate。

---

## 12. 配置建议

ActivityNet 与 Charades 的 `stage_a5.json` 已按本节建立。文件中的 selector 参数仍是扫描起点，不是冻结后的最终参数。

在 `event_boundary_refinement` 下新增独立 `stage_a5` 节点，不能改变旧 Stage A 配置的含义：

```json
{
  "event_boundary_refinement": {
    "enabled": true,
    "report_only": true,
    "stage_a5": {
      "enabled": true,
      "eval_mask_seeds": [8, 18, 28],
      "export_candidates": true,
      "score_shell": true,
      "selector": "counterfactual_gated",
      "allowed_candidate_types": [
        "left_near", "left_strong", "right_near", "right_strong"
      ],
      "min_parent_width": 0.50,
      "min_retained_ratio": 0.75,
      "max_relative_shift": 0.20,
      "max_nll_increase": 0.00,
      "min_contrast_margin": 0.02,
      "max_contrast_std": 0.03,
      "min_boundary_percentile": 0.70,
      "lambda_contrast": 1.0,
      "lambda_edit": 0.25,
      "lambda_uncertainty": 0.5,
      "lambda_boundary": 0.1,
      "accept_margin": 0.01
    }
  }
}
```

以上数值是扫描起点，不是最终参数。ActivityNet 和 Charades 最终配置必须分开保存。Charades 诊断模式不能根据 test 结果反复修改。

---

## 13. 实验矩阵

### 13.1 Checkpoint

每个数据集至少评估：

- `model-best-r1.pt`；
- `model-best-composite.pt`；
- `model-best-r5.pt`。

三个 checkpoint 文件在两个数据集上均已存在。当前 Stage A.5 只对 `model-best-r1.pt` 做了单 query smoke；best-r1、best-composite、best-r5 的完整 validation 均未完成，不能据此判断跨 checkpoint 稳定性。

### 13.2 必做消融

当前仅完成 legacy Stage A 的 A0–A2 完整对比；A3–A8 只有功能级 smoke，尚未形成完整 validation 消融结果。

按顺序运行：

```text
A0  baseline
A1  legacy Stage A, epsilon=0
A2  legacy Stage A, epsilon=0.02
A3  geometry gate only, one-sided
A4  geometry + boundary-confidence gate
A5  geometry + D_suff
A6  geometry + D_suff + D_contrast
A7  A6 + uncertainty gate
A8  A7 + both-trim（仅消融，不默认）
```

必须先完成 A3–A7，不能只比较 baseline 和最终组合，否则无法知道收益来自哪一个条件。

### 13.3 数据集策略

- ActivityNet：主要目标是减少异常长和高重叠 proposal，同时保护 long GT。
- Charades：默认采取更高 `min_parent_width` 和 `min_retained_ratio`；若 validation 不支持，允许最终配置为“Stage A.5 disabled”。关闭无效模块是合法结论。

---

## 14. Stage B 的硬进入门槛

Stage A.5 只有同时满足以下条件，才能生成 Stage B 伪标签：

**截至 2026-09-02 的判定：`STAGE_B_GO=false`。** 原因不是某一条 smoke 指标为负，而是候选集合、可部署选择器和稳定性三组门槛都尚未在完整独立 validation 上得到证据。任何 synthetic 或 `partial=true` 报告都不能改变此结论。

### 14.1 候选集合门槛

- candidate-set oracle 的 R@1 mIoU 或 IoU@0.5 有稳定正增益；
- oracle 增益在多个 checkpoint 上方向一致；
- 增益不只来自占比很小的 GT bucket。

### 14.2 可部署选择器门槛

- validation 的 R@1 mIoU 不低于 baseline；
- validation 的 R@5 mIoU 不低于 baseline；
- 至少一个主要高 IoU 指标获得提升；
- 主要 GT-width bucket 的 mIoU 下降不超过 `0.01`；
- helpful trim 数量多于 harmful trim；
- `trim_precision >= 0.60`，更推荐 `>=0.70`；
- changed fraction 不追求高，优先控制在高置信范围；
- paired bootstrap 的主要指标差值置信区间不能显示显著负收益。

### 14.3 稳定性门槛

- 至少三个固定 mask seed 结论一致；
- best-r1 与 best-composite checkpoint 方向一致；
- 配置只由 validation 决定；
- test 只在配置冻结后运行一次；
- Charades 若无真正 validation，不得给出“通过 Stage B gate”的正式结论。

当前扫描器只计算点估计，未把 validation cluster-bootstrap CI 写入 `stage_b_go` 布尔表达式。因此正式决策必须在冻结候选配置后，额外运行一次 validation `evaluate_stage_a5.py`，人工/汇总脚本核验 CI 不显示显著负收益；不能只读取 `selected_stage_a5_config.json` 的布尔字段。

### 14.4 Stage B 伪标签范围

即使通过门槛，也只把满足全部硬门控且 `S(c)<-m_accept` 的 trim 作为正伪标签。其他 proposal：

- original 与 trim 均不产生 ranking label；
- 不施加 coordinate target；
- 不施加边界吸附损失。

Stage B 必须使用 confidence weight：

\[
w_{pseudo}=clip\left(
\sigma(D_{contrast}/T_C)
\cdot a_{boundary}
\cdot e^{-U/T_U},
0,1\right).
\]

低置信样本权重为零，而不是被迫选择 original 或 trim 作为硬标签。

---

## 15. 测试要求

当前测试状态：在 `cpl` Conda 环境中运行全部 `stageA/tests`，结果为 `36 passed, 3 warnings`。warning 来自 fairseq 的 `np.float` 弃用提示与当前环境无法初始化 NVML，不影响 CPU 测试结论。

在开始全量实验前仍应补充三类回归：

- `partial=true` 产物默认禁止正式扫描/冻结，只有显式 smoke 参数才能读取；
- selector/scan 的在线输入不包含 GT 派生字段；
- 默认分层扫描确实遍历全部 top semantic parents，而不是只遍历第一个父配置。

### 15.1 确定性 mask

- 相同 sample id 和 seed 必须产生相同 mask；
- 改变 batch 顺序不改变某个 sample 的 mask；
- 不同 DataLoader worker 数不改变 mask；
- mask 数量和合法位置正确；
- train 模式仍使用原随机逻辑。

### 15.2 Shell mask

- `trim_raw + shell_raw == original_raw`，容差 `1e-6`；
- left/right/both trim 的 shell 位于被删除一侧；
- original 的 shell invalid；
- 极窄 shell 被安全标 invalid；
- 所有有效归一化 mask 有限且最大值为 1。

### 15.3 Oracle

- parent oracle gain 永不为负；
- candidate-set oracle 不低于 original-set oracle；
- synthetic case 能准确识别已知最优候选；
- invalid candidate 永不参与 oracle 或 selector；
- sample id 错位时分析工具立即失败。

### 15.4 选择器

- 无候选通过门控时返回 original；
- original 永远可用；
- 加大 edit penalty 不能导致更激进收缩；
- 提高 min contrast 不能增加 eligible 候选数；
- `both` 未开放时永不被选；
- selector API 不接受 GT；
- 所有 reason code 可复现。

建议 reason code：

```text
0 original_default
1 invalid_geometry
2 parent_not_long
3 retained_ratio_too_small
4 shift_too_large
5 insufficient_by_nll
6 weak_shell_contrast
7 high_mask_uncertainty
8 weak_boundary
9 candidate_type_disabled
10 score_margin_not_met
11 trim_selected
```

### 15.5 集成测试

- Stage A.5 disabled 时输出逐元素等于现有 baseline；
- `score_shell=false` 时不触发 shell decoder；
- dump 小批次后可被 oracle 和 selector 工具完整读取；
- report-only 模式不修改 checkpoint selection；
- 不保存 logits 或隐藏状态到 NPZ；
- CPU synthetic smoke test 和 GPU 单 batch smoke test 均通过。

---

## 16. 从当前位置开始的执行顺序

### 16.1 P0：正式全量运行前先修正扫描协议

**状态：已完成（2026-09-02）。**

已修改并测试 `stageA/tools/scan_stage_a5_selectors.py`，完成以下三项：

1. `weight_grid` 改为可复用 tuple，默认 `top_semantic=5` 时每个 semantic parent 都获得完整权重扫描。
2. 最终结果采用 gate-first：有 `stage_b_go=true` 时只在通过配置中按 objective 选择；否则保存全体 objective 最优者但强制 `STAGE_B_GO=false`，并记录选择原因与数量。
3. candidate metadata 已加入数据协议字段；analyze/scan/evaluate 默认拒绝 `partial=true` 和 test-as-validation，显式诊断参数强制 `STAGE_B_GO=false`；Charades test-as-validation 诊断 scan 只运行固定规则。

同时补充相应测试。修正后再次运行：

```bash
cd /data/chenyuan/videogrounding/w1m2
/home/chenyuan/miniconda3/envs/cpl/bin/python -m pytest -q stageA/tests
```

结果：`36 passed, 3 warnings`。

### 16.2 P1：ActivityNet 三 checkpoint 全量 validation

**状态：环境阻塞，未执行正式导出。**

预检结果：CPL 环境中 `torch.cuda.is_available()` 为 `False`、
`torch.cuda.device_count()` 为 `0`，`nvidia-smi` 无法与 NVIDIA driver 通信。
此前 CPU full-model 单 batch checkpoint 加载正常，但前向长时间未产生输出而
中止。不能用该 smoke 或 partial 产物替代下列全量命令。恢复 GPU/驱动后直接执行：

P0 通过后，进入 `stageA`，对 17,505 条 ActivityNet validation query 完整导出。正式命令不得带 `--max-batches`，输出不得写入 `stageA5_tmp`。

```bash
cd /data/chenyuan/videogrounding/w1m2/stageA

for label in best-r1 best-composite best-r5; do
  PYTHONPATH=. python tools/dump_stage_a5_candidates.py \
    --config-path config/activitynet/stage_a5.json \
    --checkpoint checkpoints/activitynet/baseline_activitynet_2026-09-01_22-49-42/model-${label}.pt \
    --checkpoint-label activitynet_${label} \
    --split val \
    --mask-seeds 8,18,28 \
    --output outputs/stage_a5/activitynet/${label}/val_candidates
done
```

当前没有生成 P1 正式 validation NPZ；P2 Oracle、selector scan 和冻结配置评估因此未启动。

每个导出必须核验：

- metadata 中 `partial=false`、`query_count=17505`；
- features/labels 的 sample id 完全一致；
- checkpoint/config SHA256 非空且与运行对象一致；
- 三个 seed 的候选几何完全一致；
- 输出目录磁盘空间充足，文件未被 smoke 覆盖。

### 16.3 P2：先看 Oracle，再决定是否扫描 selector

对三个 checkpoint 先运行 oracle/regret 分析：

```bash
for label in best-r1 best-composite best-r5; do
  PYTHONPATH=. python tools/analyze_stage_a5_oracle.py \
    --features outputs/stage_a5/activitynet/${label}/val_candidates_features.npz \
    --labels outputs/stage_a5/activitynet/${label}/val_candidates_labels.npz \
    --rows-output outputs/stage_a5/activitynet/${label}/val_oracle_rows.csv \
    --summary-output outputs/stage_a5/activitynet/${label}/val_oracle_summary.json
done
```

此处先做候选集合 Go/No-Go：

- 若 candidate-set oracle 在 best-r1 与 best-composite 上都没有明确增益，或增益只存在于极小桶，立即停止 Stage A.5/Stage B，不再扫描 selector；下一研究对象应改为候选生成方式、事件边界与 proposal 的对应关系，或基础 proposal generator。
- 若 oracle 在多个 checkpoint 和非极小 GT-width 桶上稳定为正，再运行 selector scan。oracle 有增益只说明“答案在候选中”，不说明可部署选择器能找到它。

```bash
for label in best-r1 best-composite best-r5; do
  PYTHONPATH=. python tools/scan_stage_a5_selectors.py \
    --features outputs/stage_a5/activitynet/${label}/val_candidates_features.npz \
    --labels outputs/stage_a5/activitynet/${label}/val_candidates_labels.npz \
    --output outputs/stage_a5/activitynet/${label}/selector_scan.csv \
    --selected-config-output outputs/stage_a5/activitynet/${label}/selected_stage_a5_config.json
done
```

对每个冻结配置再在同一 validation 导出上运行 `evaluate_stage_a5.py`，生成有效的 2,000 次 video-cluster bootstrap 报告。只有第 14 节全部门槛通过，且 best-r1/best-composite 方向一致，ActivityNet validation 才判为 Go。

### 16.4 P3：只有 validation Go 才运行一次 test

若 P2 为 Go，从 validation 预先指定唯一 checkpoint 和唯一冻结配置，然后只导出/评估一次 ActivityNet test；不得看到 test 指标后回到 validation 改阈值。test 只用于外部确认，不能挽救 validation No-Go。

如果 test 也保持非负总体指标、长 GT 无明显退化且 precision 达标，才开始设计 Stage B 的高置信伪标签生成。即使进入 Stage B，也只使用门控通过的 proposal，其他 proposal 的伪标签权重为 0。

### 16.5 P4：Charades 的处理顺序

当前 Charades `val_data == test_data`，因此不要继续在这 3,158 条 test query 上扫描参数。建议先等 ActivityNet 给出候选/选择器有效信号：

- 若 ActivityNet No-Go，Charades 不再投入正式 Stage A.5 训练与扫描；
- 若 ActivityNet Go，使用 `make_charades_val_split.py` 从 train 按 video id 构造固定 validation，并从新的 train 子集重新训练 baseline；
- 旧 Charades checkpoint 只能做固定配置的跨数据集诊断，不能产生正式 `STAGE_B_GO=true`。

---

## 17. 可直接交给下一位 Agent 的任务描述

> 不要实现 Stage B。先修正 `/data/chenyuan/videogrounding/w1m2/stageA/tools/scan_stage_a5_selectors.py` 的一次性 `weight_grid`、硬门槛优先选配置和 partial/Charades-test 防误用问题，并补齐回归测试。全部 `stageA/tests` 通过后，对 ActivityNet validation 的 best-r1、best-composite、best-r5 三个 checkpoint 运行完整三-mask候选导出，禁止使用 `--max-batches`，正式产物写入 `stageA/outputs/stage_a5/activitynet/<label>/`。先汇总 candidate-set oracle、parent oracle、legacy regret、NLL/contrast 判别力和 GT-width buckets；只有 oracle 在多个 checkpoint 和非极小桶上稳定正向时才运行 selector scan。冻结配置后在 validation 上生成 video-cluster bootstrap 报告，逐条核验第 14 节门槛。最终输出证据表和 `STAGE_B_GO=true/false`；validation 未通过时停止，不运行 test、不实现 Label Propagation。

---

## 18. 当前完成定义与检查表

- [x] 事件边界提取结果可用，并生成 Stage A 索引；
- [x] ActivityNet/Charades baseline 训练与 checkpoint 保存完成；
- [x] 两个数据集的 legacy Stage A 完整推理完成，确认总体退化；
- [x] 确定性 eval mask、稳定 sample id、三 mask 聚合完成；
- [x] 候选/标签双 NPZ、oracle/regret、shell contrast、GT-free selector、扫描和评估工具完成；
- [x] 当前 `stageA/tests` 为 `36 passed`；
- [x] synthetic 与两个数据集单 query smoke 完成；
- [x] 修复正式扫描覆盖、gate-first 选配置和 partial/Charades test-as-val 防误用；
- [ ] ActivityNet 三 checkpoint 的全量 validation 候选与 oracle 报告齐全；
- [ ] 若 oracle Go，完成 selector scan、validation bootstrap 和唯一配置冻结；
- [ ] 若 validation Go，只运行一次 ActivityNet test；
- [ ] Charades 明确采用正式重训 validation，或仅报告跨数据集诊断；
- [ ] 最终以完整证据输出 `STAGE_B_GO=true/false`。

在最后六项完成前，Stage A.5 的工程实现可视为完成，但实验阶段仍未完成；Stage B 保持禁止启动。
