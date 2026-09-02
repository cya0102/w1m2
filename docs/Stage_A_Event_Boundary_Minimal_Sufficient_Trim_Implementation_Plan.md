# 阶段 A：事件边界最小充分区间收缩实施方案

## 1. 目标

在 `/data/chenyuan/videogrounding/w1m2/stageA` 中实现一个仅在推理阶段运行、无需重新训练、不会改变 checkpoint 参数的事件边界 proposal refinement 模块。

给定 `cpl_lrev` 生成的原始提议：

\[
u_n=[s_n,e_n]
\]

以及该视频的事件边界集合：

\[
B_v=\{(b_j,a_j)\},
\]

为每个原始提议生成若干 inward-trim 候选，并使用同一个 query decoder 重新计算 reconstruction NLL。在满足：

\[
L_{\mathrm{cap}}(u_{\mathrm{trim}})
\le
L_{\mathrm{cap}}(u_{\mathrm{original}})+\epsilon
\]

的候选中选择最短者。

阶段 A 必须同时输出：

- 原始 baseline 结果；
- Stage-A 修正结果；
- proposal 宽度、NLL 变化和候选选择诊断。

初期不得用 Stage-A 结果选择训练 checkpoint，也不得修改训练损失。

相关背景分析见：

- `docs/IPR_Label_Propagation_for_cpl_lrev_Analysis.md`
- `docs/CVPR23-Iterative_Proposal_Refinement_for_Weakly-Supervised_Video_Grounding.pdf`

---

## 2. 实现范围

### 2.1 需要实现

- 事件边界紧凑索引；
- Dataset 加载和 batch padding；
- inward-trim 候选生成；
- 候选软时间 mask；
- 使用现有 decoder 分块重评分；
- 最小充分候选选择；
- baseline/Stage-A 双轨评测；
- 单元测试和集成测试。

### 2.2 不实现

- Label Propagation；
- 新 confidence head；
- semantic/concept distillation；
- 坐标训练损失；
- 参数更新；
- outward expansion；
- 修改现有 ActivityNet outer/weighted 设置；
- 修改现有 checkpoint。

---

## 3. 总体数据流

```text
大型事件边界 JSON
        │
        ▼
一次性构建紧凑 CSR NPZ
        │
        ▼
Dataset 按 video id 加载边界
        │
        ▼
原始 N 个 proposal + 原始时间 mask
        │
        ▼
每个 proposal 生成 7 个 inward candidates
        │
        ▼
原 mask × 边界软窗口
        │
        ▼
复用同一次 masked query，分块计算 candidate NLL
        │
        ▼
NLL 约束下选择最短候选
        │
        ▼
仍得到 N 个 refined proposals
        │
        ▼
复用现有 NLL / geometric_vote / semantic_vote
        │
        ├── Baseline 指标
        └── Stage-A 指标与诊断
```

---

## 4. 前置代码：紧凑事件边界索引

原始 JSON 分别约为 210 MB 和 61 MB，不应由 train/val/test Dataset 分别完整加载。

### 4.1 新增文件

```text
cpl_lrev/tools/build_event_boundary_index.py
cpl_lrev/datasets/event_boundaries.py
cpl_lrev/tests/test_event_boundary_index.py
```

### 4.2 输入与输出

输入：

```text
CPL_baseline-main/eventBoundary/outputs/
├── activitynet_c3d_event_boundaries.json
└── charades_i3d_event_boundaries.json
```

输出：

```text
cpl_lrev/data/event_boundaries/
├── activitynet_c3d_stage_a.npz
└── charades_i3d_stage_a.npz
```

### 4.3 NPZ 格式

采用 CSR/ragged 格式，避免 `[video, 200]` padding：

```python
{
    "video_ids": np.ndarray[V],        # Unicode string
    "offsets": np.ndarray[V + 1],      # int64
    "indices": np.ndarray[K],          # int16，200-grid 边界索引
    "positions": np.ndarray[K],        # float32，index / 199
    "scores": np.ndarray[K],           # float32
    "metadata": np.ndarray[1],         # JSON string
}
```

其中 `K` 是所有视频内部边界数量之和。

`metadata` 至少包含：

```json
{
  "schema_version": 1,
  "source_json": "...",
  "dataset": "activitynet 或 charades",
  "num_clips": 200,
  "position_mapping": "index / (num_clips - 1)",
  "plateau_tolerance": 1e-7,
  "min_gap_clips": 2,
  "endpoint_policy": "internal detected boundaries only"
}
```

### 4.4 边界预处理

只处理 `detected_boundary_indices`，强制端点 `0/199` 不进入内部候选。

步骤：

1. 读取内部边界和对应的 `boundary_scores[1:-1]`；
2. 合并连续且分数相等的 plateau；
3. 执行 minimum-distance temporal NMS；
4. 按时间顺序保存；
5. 校验索引、位置和数值合法性。

Plateau 合并规则：

- 连续条件：`idx[j] == idx[j-1] + 1`；
- 分数条件：`abs(score[j] - score[j-1]) <= 1e-7`；
- 合并位置：组内 index 均值四舍五入；
- 合并分数：组内最大值。

Temporal NMS：

- 默认 `min_gap_clips=2`；
- 按 score 从高到低保留；
- 已保留边界距离小于 `min_gap_clips` 的候选被抑制；
- 最后重新按 index 升序排列。

必须校验：

- `num_clips == 200`；
- index 严格递增；
- index 位于 `(0,199)`；
- position 等于 `index/199`；
- score 全部有限；
- offsets 单调且最后一个值等于 `K`。

建议函数：

```python
def merge_equal_score_plateaus(indices, scores, tolerance=1e-7):
    ...


def temporal_nms(indices, scores, min_gap_clips=2):
    ...


def build_boundary_index(input_path, output_path, min_gap_clips=2):
    ...
```

输出应先写入目标目录下的临时文件，校验成功后再原子替换正式文件。

### 4.5 索引生成命令

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev

python tools/build_event_boundary_index.py \
  --input ../CPL_baseline-main/eventBoundary/outputs/activitynet_c3d_event_boundaries.json \
  --output data/event_boundaries/activitynet_c3d_stage_a.npz \
  --min-gap-clips 2

python tools/build_event_boundary_index.py \
  --input ../CPL_baseline-main/eventBoundary/outputs/charades_i3d_event_boundaries.json \
  --output data/event_boundaries/charades_i3d_stage_a.npz \
  --min-gap-clips 2
```

### 4.6 索引加载器

在 `cpl_lrev/datasets/event_boundaries.py` 实现：

```python
class EventBoundaryIndex:
    def __init__(self, path):
        ...

    def get(self, video_id):
        """
        Returns:
            positions: float32[J]
            scores: float32[J]
        """
```

增加模块级缓存，防止 train/val/test 重复加载：

```python
from functools import lru_cache


@lru_cache(maxsize=None)
def load_event_boundary_index(path):
    return EventBoundaryIndex(path)
```

路径必须相对 `cpl_lrev` 项目根目录解析，不能依赖运行时当前目录。

异常策略：

- 缺少整个 video id：strict 模式下报错；
- 视频存在但无内部边界：返回两个空数组；
- schema、dataset 或 `num_clips` 不匹配：加载时立即报错。

---

## 5. Dataset 接入

修改：

```text
cpl_lrev/datasets/base.py
```

配置增加：

```json
"dataset": {
  "event_boundary_path": "data/event_boundaries/activitynet_c3d_stage_a.npz"
}
```

`BaseDataset.__init__`：

```python
self.event_boundary_index = None
boundary_path = args.get("event_boundary_path")
if boundary_path:
    self.event_boundary_index = load_event_boundary_index(boundary_path)
```

`__getitem__` 增加：

```python
boundary_positions, boundary_scores = self.event_boundary_index.get(vid)

return {
    ...,
    "event_boundary_positions": boundary_positions,
    "event_boundary_scores": boundary_scores,
}
```

`collate_data` 对 batch 内边界动态 padding：

```text
event_boundary_positions: [B, J_max] float32
event_boundary_scores:    [B, J_max] float32
event_boundary_mask:      [B, J_max] bool
```

如果 batch 中所有视频都没有内部边界，令 `J_max=1`，mask 全 false，避免零维张量。

这三个张量加入 `batch["net_input"]`，现有 `move_to_cuda` 会自动移动它们。

---

## 6. 候选生成模块

### 6.1 新增文件

```text
cpl_lrev/models/modules/event_boundary_refinement.py
```

并在：

```text
cpl_lrev/models/modules/__init__.py
```

中导出。

该模块不包含任何可学习参数，以保证旧 checkpoint 能 strict load。

### 6.2 接口

```python
class EventBoundaryRefiner:
    CANDIDATE_NAMES = (
        "original",
        "left_near",
        "left_strong",
        "right_near",
        "right_strong",
        "both_near",
        "both_strong",
    )

    def build_candidates(
        self,
        center,
        width,
        boundary_positions,
        boundary_scores,
        boundary_mask,
    ):
        """
        Args:
            center: [B, N]
            width: [B, N]
            boundary_positions: [B, J]
            boundary_scores: [B, J]
            boundary_mask: [B, J]

        Returns:
            starts: [B, N, 7]
            ends: [B, N, 7]
            valid: [B, N, 7]
            candidate_type: [7]
        """
```

### 6.3 固定七类候选

对于原提议：

\[
s=\max(0,c-w/2),\qquad e=\min(1,c+w/2)
\]

以当前区间中心：

\[
m=(s+e)/2
\]

将内部边界分为左右两组。

左边界池：

\[
s+\delta_b < b_j \le m
\]

右边界池：

\[
m \le b_j < e-\delta_b
\]

其中：

\[
\delta_b =
\frac{\text{min_boundary_margin_clips}}{199}.
\]

默认 `min_boundary_margin_clips=1`。

边界选择：

- `left_near`：左池中最靠近原 start 的边界，即位置最小者；
- `left_strong`：左池中 score 最大者；
- `right_near`：右池中最靠近原 end 的边界，即位置最大者；
- `right_strong`：右池中 score 最大者。

生成七个候选：

| 类型 | start | end |
|---|---|---|
| original | s | e |
| left_near | left_near | e |
| left_strong | left_strong | e |
| right_near | s | right_near |
| right_strong | s | right_strong |
| both_near | left_near | right_near |
| both_strong | left_strong | right_strong |

候选有效条件：

```python
candidate_width >= max(
    min_candidate_width,
    min_retained_ratio * original_width,
)
```

推荐默认值：

```json
"min_candidate_width": 0.02,
"min_retained_ratio": 0.25
```

必须满足：

- 原始候选永远 valid；
- 不存在对应边界时，相关候选 invalid；
- 坐标重复的候选只保留优先级更高者；
- 所有 valid 候选位于原区间内部；
- Stage A 只允许收缩，不能移动到原区间外；
- 坐标必须满足 `0 <= start < end <= 1`。

---

## 7. 候选时间 mask

当前 attention 支持任意非负 `gauss_weight`，见：

```text
cpl_lrev/models/modules/mutihead_attention.py:189-194
```

因此不需要新增 decoder。

对于候选区间 \([s',e']\)，在 `props_len=50` 的时间位置：

\[
t_l=\frac{l}{L-1}
\]

构造软窗口：

\[
W_l=
\sigma\left(\frac{t_l-s'}{\tau}\right)
\sigma\left(\frac{e'-t_l}{\tau}\right).
\]

候选 mask：

\[
M_{\mathrm{candidate}}
=
M_{\mathrm{original}}\odot W.
\]

按最大值归一化：

```python
candidate_mask = candidate_mask / (
    candidate_mask.amax(dim=-1, keepdim=True).clamp_min(1e-6)
)
```

推荐：

```json
"soft_window_temperature": 0.01
```

重要约束：

- `original` 必须直接使用原始 `pos_weight`，不能再乘窗口；
- original candidate 的 NLL 必须与现有 baseline 完全一致；
- 对 Gaussian mixture，软窗口用于删除边界外的远端 component evidence；
- 不改成纯矩形 mask，否则会同时替换原 mixture/Gaussian 的语义表示；
- mask 最大值过小或非有限时，将候选标为 invalid。

---

## 8. 在 `CPL.forward` 中重评分

修改：

```text
cpl_lrev/models/cpl.py
```

### 8.1 配置解析

增加：

```json
"event_boundary_refinement": {
  "enabled": true,
  "num_clips": 200,
  "candidate_policy": "near_and_strong",
  "min_boundary_margin_clips": 1,
  "min_candidate_width": 0.02,
  "min_retained_ratio": 0.25,
  "soft_window_temperature": 0.01,
  "decode_chunk_size": 64,
  "max_nll_increase": 0.02,
  "report_only": true
}
```

该配置不得创建新的 `nn.Parameter`。

### 8.2 显式运行开关

训练 forward 不运行 Stage A。

runner 在 eval 时传入：

```python
net_input["run_stage_a"] = self.stage_a_enabled
```

模型读取：

```python
run_stage_a = bool(kwargs.get("run_stage_a", False))
```

当 `run_stage_a=True` 时：

- 必须处于 `model.eval()`；
- 必须提供 boundary tensors；
- 否则抛出明确异常。

### 8.3 复用同一次 masked query

当前 `_mask_words` 即使在 eval 中也会选择 masked words。候选重评分不能再次调用 `_mask_words`，否则原始和候选看到的查询不同。

将当前代码整理为：

```python
masked_query_base, masked_words = self._mask_words(...)
masked_query_base = masked_query_base + words_pos
masked_query_base = masked_query_base[:, :-1]
query_mask_base = words_mask[:, :-1]
```

原 proposal 和所有 Stage-A 候选都从同一个 `masked_query_base` 扩展。

不要在 Stage A 中改变 `_mask_words` 的既有行为，否则 baseline 也会发生变化。

### 8.4 提取通用 NLL 函数

把当前重复的 reconstruction NLL 逻辑提取为：

```python
@staticmethod
def reconstruction_nll_from_logits(
    words_logit,
    words_id,
    words_mask,
):
    """
    words_logit: [Q, W, V]
    words_id:    [Q, W]
    words_mask:  [Q, W]
    returns:     [Q]
    """
```

必须与 `cal_nll_loss` 保持一致：

- label smoothing 为 `0.1`；
- 按有效 word mask 求平均；
- 不改变 baseline 数值。

### 8.5 分块解码

不能一次解码 `B × N × 7` 个候选。ActivityNet 的完整候选 logits 可能达到：

```text
32 × 5 × 7 × 20 × 8000
```

会占用数百 MB 显存。

建议接口：

```python
def score_stage_a_candidates(
    frames_feat,             # downsample 后 [B,L,H]
    frames_mask,             # [B,L]
    masked_query_base,       # [B,W,H]
    query_mask_base,         # [B,W]
    words_id,                # [B,W]
    original_masks,          # [B,N,L]
    candidate_masks,         # [B,N,7,L]
    candidate_valid,         # [B,N,7]
    original_nll,            # [B,N]
    chunk_size,
):
    ...
```

实现要求：

1. candidate 0 直接复用 `original_nll`，不重新解码；
2. 只解码 valid 的 candidate 1~6；
3. 每次最多解码 `decode_chunk_size=64` 个 flat candidates；
4. 每个 chunk 计算完 NLL 后立即释放 logits；
5. invalid candidate NLL 设为 `+inf`；
6. 返回 `[B,N,7]`，不返回 candidate logits 或 candidate masks。

模型输出新增：

```python
{
    "stage_a_candidate_start": ...,  # [B,N,7]
    "stage_a_candidate_end": ...,    # [B,N,7]
    "stage_a_candidate_valid": ...,  # [B,N,7]
    "stage_a_candidate_nll": ...,    # [B,N,7]
}
```

---

## 9. 最小充分候选选择

在：

```text
cpl_lrev/runners/main_runner.py
```

增加纯函数：

```python
def select_minimal_sufficient_candidates(
    candidate_start,
    candidate_end,
    candidate_nll,
    candidate_valid,
    max_nll_increase,
    width_tolerance=1e-6,
):
    """
    Returns:
        refined_props: [B,N,2]
        refined_nll: [B,N]
        selected_candidate_index: [B,N]
    """
```

对每个原 proposal 单独选择。

设 candidate 0 是原 proposal，候选合格条件：

```python
eligible = candidate_valid & (
    candidate_nll <= candidate_nll[..., :1] + max_nll_increase
)
```

选择顺序必须是严格词典序：

1. 最小宽度；
2. 宽度相同时选 NLL 最小；
3. 仍相同时选 candidate index 更小；
4. 原候选永远是回退项。

不要使用：

```python
nll + lambda_width * width
```

因为这会引入固定的全局短区间偏置。

建议至少扫描：

```text
max_nll_increase ∈ {0.00, 0.01, 0.02, 0.05}
```

不要只报告 `0.02`。

---

## 10. 与现有 proposal 排序的关系

Stage A 先对每个 parent proposal 选一个 refined variant，最终仍得到 `N` 个 proposals。

之后：

1. 使用 refined NLL 排序；
2. parent 的 Event score 暂时原样继承；
3. 最终分数为：

\[
score_n^{A}
=
L_{\mathrm{cap},n}^{A}
-\lambda_E score_{\mathrm{event},n}^{\mathrm{parent}};
\]

4. 再调用现有 `select_proposal_by_strategy(...)`。

这样可以保持三种现有策略兼容：

- `nll`；
- `geometric_vote`；
- `semantic_vote`。

Stage A MVP 不重新计算 Event vector：

- ActivityNet 当前 `inference_event_weight=0`，没有影响；
- Charades 应在日志中明确“trim candidate 继承 parent Event score”。

---

## 11. 双轨评测

Stage A 初期必须设置：

```json
"report_only": true
```

同一次 eval 输出两组结果：

```text
Baseline(Test): ...
StageA(Test, epsilon=0.02): ...
```

原有未加前缀的 metrics 保持 baseline，避免改变 checkpoint selection。

Stage-A 额外记录：

```text
stage_a_changed_fraction
stage_a_mean_width_before
stage_a_mean_width_after
stage_a_mean_width_reduction
stage_a_mean_nll_delta
stage_a_valid_candidates_per_proposal
stage_a_original_fraction
stage_a_left_trim_fraction
stage_a_right_trim_fraction
stage_a_both_trim_fraction
stage_a_endpoint_boundary_distance_before
stage_a_endpoint_boundary_distance_after
```

还应分别报告：

- GT short；
- medium-short；
- medium-long；
- long；
- duration 分桶；
- R1 mIoU；
- R1@0.3/0.5/0.7；
- near-whole-video proposal 比例。

Stage A 不能只报告总体 `R1@0.3`。

---

## 12. 配置文件

不要直接修改当前 `main.json`，新增诊断配置：

```text
cpl_lrev/config/activitynet/stage_a.json
cpl_lrev/config/charades/stage_a.json
```

配置应从当前实际 baseline 复制，只增加：

- `dataset.event_boundary_path`；
- `model.config.event_boundary_refinement`。

参考配置：

```json
{
  "dataset": {
    "event_boundary_path": "data/event_boundaries/activitynet_c3d_stage_a.npz"
  },
  "model": {
    "config": {
      "event_boundary_refinement": {
        "enabled": true,
        "num_clips": 200,
        "candidate_policy": "near_and_strong",
        "min_boundary_margin_clips": 1,
        "min_candidate_width": 0.02,
        "min_retained_ratio": 0.25,
        "soft_window_temperature": 0.01,
        "decode_chunk_size": 64,
        "max_nll_increase": 0.02,
        "report_only": true
      }
    }
  }
}
```

注意：当前 ActivityNet 的 README/测试声称使用 weighted boundary，但实际 `main.json` 和现有 checkpoint 是 outer。Stage A 任务不得顺手修改该漂移，应以 checkpoint 保存的 resolved config 为 baseline。

---

## 13. 测试要求

### 13.1 边界索引测试

`test_event_boundary_index.py`：

- 连续等分 plateau 正确合并；
- 不同分数相邻峰不会错误合并；
- NMS 保留更高分峰；
- index/position 映射正确；
- CSR offsets 正确；
- video id 查找正确；
- 无内部边界返回空数组；
- 缺少 video id 在 strict 模式下报错。

### 13.2 候选几何测试

`test_stage_a_refinement.py`：

- 所有 valid 候选位于原 proposal 内；
- original 永远 valid；
- left candidate 只移动 start；
- right candidate 只移动 end；
- both 同时移动；
- 不小于 `min_candidate_width`；
- 不小于 `min_retained_ratio × original_width`；
- 边界不存在时只保留 original；
- 重复 near/strong 候选被去重。

### 13.3 Mask 测试

- original mask 与原 `pos_weight` 完全相同；
- trimmed mask 的边界外权重降低；
- mask 最大值为 1；
- mask 不产生 NaN；
- 极短/无效候选不会进入 decoder。

### 13.4 NLL 测试

- candidate 0 NLL 与现有 proposal NLL 误差不超过 `1e-6`；
- `chunk_size=1` 与大 chunk 的结果一致；
- invalid candidate NLL 为 inf；
- 候选评分不再次执行 `_mask_words`。

### 13.5 选择器测试

至少覆盖：

- 短候选 NLL 在 epsilon 内：选择短候选；
- 短候选超过 epsilon：回退 original；
- 两候选同宽：选择 NLL 更低者；
- 全部 invalid：选择 original；
- epsilon 为 0：只接受不劣于原始 NLL 的候选。

### 13.6 集成测试

使用现有 CPU patched-CUDA 测试风格验证：

- Stage A disabled 时 forward 输出和 baseline 一致；
- Stage A enabled 时输出形状正确；
- 旧 checkpoint 能 strict load；
- Stage A 不增加模型参数；
- single Gaussian 可运行；
- Gaussian mixture 可运行。

运行：

```bash
cd /data/chenyuan/videogrounding/w1m2/cpl_lrev
pytest -q
```

当前已有测试对 ActivityNet `boundary_mode` 的预期与实际配置可能不一致。实现 agent 应将其记录为既有 baseline 漂移，不要误判为 Stage-A 引入的失败。

---

## 14. 验收标准

### 14.1 工程验收

- Stage A disabled 时 baseline 指标不变；
- 不修改 checkpoint state dict；
- 不进行训练；
- 所有候选坐标有限且满足 `0 <= start < end <= 1`；
- 任意异常都能回退原 proposal；
- candidate logits 不保留在 output；
- GPU 不出现明显峰值内存爆炸；
- 两份数据集边界索引覆盖全部标注 video id。

### 14.2 实验验收

Stage A 是否值得进入阶段 B，应依据：

1. `changed_fraction` 不是接近 0；
2. 平均 selected width 明显下降；
3. short/medium-short 的 mIoU 或 R1@0.5 提升；
4. long GT 分桶没有明显下降；
5. NLL 增量受 epsilon 控制；
6. 提升在多个 epsilon 下方向一致；
7. 不是仅靠异常端点候选获得收益。

如果 Stage A 只能降低宽度但 IoU 全面下降，应停止进入阶段 B，优先检查：

- query reconstruction 对视觉区间是否不敏感；
- 50-grid 是否无法区分相邻边界；
- boundary plateau/NMS 是否不足；
- mixture mask 和 reported interval 是否不一致。

---

## 15. 建议实施顺序

1. 实现并测试 JSON → CSR NPZ 转换；
2. 接入 Dataset，但保持模型行为不变；
3. 实现纯候选几何模块和单元测试；
4. 实现软窗口 mask；
5. 重构通用 NLL，验证 baseline 数值不变；
6. 实现分块 candidate decode；
7. 实现纯最小充分选择器；
8. 接入 runner 双轨评测；
9. 构建两个数据集的边界索引；
10. 在固定 checkpoint 上扫描 epsilon；
11. 汇总宽度、分桶指标和运行成本；
12. 根据验收条件决定是否进入阶段 B。

---

## 16. 可直接交给实现 Agent 的任务描述

> 在 `/data/chenyuan/videogrounding/w1m2/cpl_lrev` 中实现 inference-only Stage-A event-boundary minimal-sufficient trimming。先把两个大型事件边界 JSON 转为带 plateau merge 和 temporal NMS 的 CSR NPZ 索引；在 Dataset 中按 video id 加载并 padding；对每个原 proposal 生成 original、left-near、left-strong、right-near、right-strong、both-near、both-strong 七类 inward candidates；以原 proposal mask 乘事件区间 soft window 得到候选 mask；必须复用同一次 forward 的 masked query，并按 chunk 重用现有 DualTransformer decoder 计算 candidate reconstruction NLL；在 `NLL <= original NLL + epsilon` 的候选中按“最小宽度、最低 NLL、最小 index”选择；每个 parent proposal 修正后再进入现有 NLL/geometric/semantic selector。Stage A 默认只报告诊断结果，不改变 checkpoint selection，不修改训练损失和 checkpoint 参数。补齐边界索引、候选几何、mask、NLL chunking、选择器和双 generator 集成测试，并同时报告 Baseline 与 StageA 指标。
