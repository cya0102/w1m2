# CPL-LRRV Final

本目录是最终版 `cpl_lrrv`，整理自：

```text
version: V4_stage01
experiment: stage1_old_v4_weighted_diagnostic
source code: /data/chenyuan/videogrounding/cpl_lrrv/cpl_lrrv_V4_stage01
```

该版本的核心设置是：

- 使用 V4/Stage01 的 Gaussian Mixture proposal generator；
- 定位边界采用 `boundary_mode = weighted`；
- negative exclusion 仍使用完整 component outer envelope；
- 使用低秩 Event Disentanglement；
- 推理时使用 `--vote`，即 semantic weighted voting；
- 默认随机种子为 8。

需要注意：`stage1_old_v4_weighted_diagnostic` 的最好结果来自一个诊断式评估：它没有重新训练，而是使用 weighted boundary 推理配置，加载旧 V4 的 `lrrv_v4_outer_semantic` checkpoint 进行测试。为了让最终目录可独立复现，我已经把该 checkpoint 复制到了本目录下：

```text
checkpoints/activitynet/stage1_old_v4_weighted_diagnostic/model-best.pt
```

## 环境

```bash
cd /data/chenyuan/videogrounding/cpl_lrrv/cpl_lrrv
conda activate cpl
```

## 复现最终诊断结果

如果目标是复现 `stage1_old_v4_weighted_diagnostic_2026-07-07_00-52-17.log`
中的点数，必须使用本节命令。这个结果不是从头训练得到的，而是：

```text
weighted boundary 代码/配置 + 旧 V4 outer_semantic checkpoint + --eval
```

对应日志中的结果：

```text
R1@0.1 82.50 | R1@0.3 59.57 | R1@0.5 33.16
R5@0.1 92.44 | R5@0.3 78.85 | R5@0.5 59.71
```

运行：

```bash
python train.py \
  --config-path config/activitynet/final_weighted_diagnostic.json \
  --resume checkpoints/activitynet/stage1_old_v4_weighted_diagnostic/model-best.pt \
  --eval \
  --log_dir logs/activitynet \
  --tag final_weighted_diagnostic \
  --vote
```

`config/activitynet/main.json` 与 `config/activitynet/final_weighted_diagnostic.json` 内容一致；后者只是为了让最终实验配置名称更明确。

也可以直接运行脚本：

```bash
bash scripts/run_final_diagnostic.sh
```

## 从头训练 weighted 配置

如果你想用当前最终代码和 weighted 配置重新训练，可以运行：

```bash
python train.py \
  --config-path config/activitynet/main.json \
  --log_dir logs/activitynet \
  --tag cpl_lrrv_final \
  --vote
```

注意：这条命令会从随机初始化开始训练，得到的是
`stage1_a1_weighted_current_s8` 类型的重新训练结果，不会复现
`stage1_old_v4_weighted_diagnostic` 的点数。后者必须使用上一节的
`--resume ... --eval` 命令。

训练结束后会保存：

```text
checkpoints/activitynet/cpl_lrrv_final_<日期>/
```

并在日志里输出：

```text
Test(best-r1-validation)
Test(best-r5-validation)
Test(best-composite-validation)
```

正式比较时建议优先使用 `Test(best-composite-validation)`；如果目标是最大化 Rank5，则可查看 `Test(best-r5-validation)`。

## 关键配置

最终默认配置位于：

```text
config/activitynet/main.json
config/activitynet/final_weighted_diagnostic.json
```

其中最重要的字段为：

```json
"proposal_generator": {
  "type": "gaussian_mixture",
  "max_components": 5,
  "component_sigma": 4.0,
  "importance_temperature": 1.0,
  "boundary_mode": "weighted",
  "boundary_shrink": 0.0
}
```

低秩事件解耦设置为：

```json
"event_disentanglement": {
  "enabled": true,
  "rank": 8,
  "cpca_alpha": 1.0,
  "covariance_ema": 0.95,
  "normalize_covariance": true,
  "selection_temperature": 0.1,
  "warmup_epochs": 5,
  "ramp_epochs": 5,
  "score_separation_weight": 0.5,
  "inference_event_weight": 0.0,
  "inference_vote_event_weight": 0.0
}
```

## 目录说明

```text
models/      模型与 LRRV/Event/Gaussian Mixture proposal 代码
runners/     训练、验证、测试和 checkpoint 选择逻辑
datasets/    ActivityNet / Charades 数据加载
config/      最终配置
data/        数据索引与词向量
checkpoints/ 仅包含最终诊断 checkpoint
logs/        新运行日志输出目录
```
