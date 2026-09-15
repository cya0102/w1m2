# QSTG 独立工程

QSTG（Query-Subgraph Temporal Grounder）在 CPL 的 query reconstruction
候选上增加固定 POS phrase graph、50-step temporal evidence graph、双向
phrase/node binding、proposal quality 以及可选 connected refinement。

本目录是独立源码副本，不导入或软链接 `cpl_lrev/`。视频 HDF5 和 GloVe
路径仍由配置中的共享绝对路径提供。`qstg.enabled=false` 时使用原 CPL
路径；Gaussian bias 和 residual adapter 均为 identity/zero 初始化。

## 验证

```bash
cd /data/chenyuan/videogrounding/w1m2/qstg
~/miniconda3/envs/cpl/bin/python -m pytest tests/ -q
~/miniconda3/envs/cpl/bin/python tools/inspect_phrase_graph.py \
  --config config/activitynet/qstg_stage_a.json --limit 100
~/miniconda3/envs/cpl/bin/python tools/compare_baseline_parity.py \
  --config config/activitynet/qstg_stage_a.json \
  --checkpoint checkpoints/bootstrap/activitynet-model-best.pt
```

## 分阶段训练

Charades 正式 QSTG 实验先生成固定、按视频 disjoint 的 validation split：

```bash
./tools/build_charades_val_split.py
```

然后将对应数据集的 baseline checkpoint 放入 `checkpoints/bootstrap/`，再按顺序运行
三个阶段。ActivityNet 使用 `run_qstg_stage_a.sh`、`run_qstg_stage_b.sh` 和
`run_qstg_full.sh`；Charades 使用 `run_charades_stage_a.sh`、
`run_charades_stage_b.sh` 和 `run_charades_full.sh`。Stage B/C 使用环境变量
指定上一阶段的 weights-only checkpoint：

```bash
QSTG_STAGE_A_CHECKPOINT=checkpoints/activitynet/stage_a/<run>/model-best.pt \
  scripts/run_qstg_stage_b.sh
QSTG_STAGE_B_CHECKPOINT=checkpoints/activitynet/stage_b/<run>/model-best.pt \
  scripts/run_qstg_full.sh
```

Charades 的完整 pipeline 使用独立的 video-disjoint validation split 和
Charades baseline bootstrap：

```bash
cp /data/chenyuan/videogrounding/w1m2/cpl_lrev/checkpoints/charades/lrevFinal_2026-09-04_16-01-36/model-best.pt \
  checkpoints/bootstrap/charades-model-best.pt
./scripts/run_charades_stage_a.sh
QSTG_CHARADES_STAGE_A_CHECKPOINT=checkpoints/charades/stage_a/<run>/model-best.pt \
  ./scripts/run_charades_stage_b.sh
QSTG_CHARADES_STAGE_B_CHECKPOINT=checkpoints/charades/stage_b/<run>/model-best.pt \
  ./scripts/run_charades_full.sh
```

ActivityNet 和 Charades 的 Stage A/B/C 分别为 5/10/10 epoch，训练 scope
均为 `qstg_only` → `qstg_and_proposal` → `all`。

`--init-from-baseline` 只加载 shape-compatible baseline 参数并允许缺少
`qstg.*`；`--resume` 才会恢复 optimizer、scheduler、epoch 和 update 状态。
