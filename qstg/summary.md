# QSTG 实施进度总结

> 日期：2026-09-14  
> 方案：`/data/chenyuan/videogrounding/w1m2/docs/videoCaption/proposal_tip26_sg_fscformer_detailed.md`  
> 工程：`/data/chenyuan/videogrounding/w1m2/qstg`

## 1. 约束与基线

- `qstg/` 是独立源码副本，不 import、不软链 `cpl_lrev/`。
- `cpl_lrev/` 与 `docs/` 未修改。
- HDF5/GloVe 仍由配置中的共享绝对路径提供。
- `qstg.enabled=false` 不创建 QSTG、不执行 QSTG bias/residual 路径，旧 CPL 路径保持不变。
- 所有 residual/bias 输出层零初始化；`proposal_prior_scale`、`importance_bias_scale` 默认 0。
- `connected_refine` 仅 eval 且在 `no_grad` 中运行；训练 forward 不接收 GT timestamp。
- 测试/训练环境固定使用 `/home/chenyuan/miniconda3/envs/cpl/bin/python`。

## 2. 已完成实现

### Step 0–1：工程复制与 phrase graph 数据链

- 复制了 baseline 运行所需源码、配置、数据 JSON 和 GloVe 到 `qstg/`。
- `datasets/base.py` 实现确定性的 POS phrase graph：GLOBAL/ACTION/ENTITY/ATTRIBUTE/RELATION、容量控制、方向关系边、required/global fallback。
- OOV 过滤、同步截断、全 OOV `<unk>` 保护、`sample_uid`、稳定 `video_group_id` 和 `query_fallback` 已接入 dataset/collate。
- phrase graph 关闭时旧输入数值路径不变。

### Step 2：纯 tensor QSTG ops

`models/modules/qstg_ops.py` 已完成并通过 CPU 冒烟：

- temporal pooling、transition score、固定度数 temporal adjacency；
- normalized smooth max、proposal/node resampling、analytic soft box；
- typed relation kernel、diagonal/cross-batch relation satisfaction；
- coverage、connectivity、detached barrier、same-video negative mask；
- eval-only connected refinement 及回退规则。

实现没有 NumPy 或 `.cuda()`，支持可变长度、padding 和 FP16 reduction 边界。

### Step 3–5：QSTG 三阶段模块

`models/modules/qstg.py` 已实现：

- `QueryPhraseGraphEncoder`：phrase pooling、type embedding、query graph attention；
- `TemporalEvidenceGraphEncoder`：clean/grounded node fusion、稀疏 temporal relation propagation、node gate；
- `QuerySubgraphTemporalGrounder`：
  `forward_pre_proposal()`、`enhance_components()`、`score_proposals()`；
- many-to-many phrase/node binding、global summary、proposal C/X/R/H、quality head、pair quality、component assignment、global/component residual 和 proposal bias channels；
- 输出包含方案 10.4 的 pre/component/proposal 字段；无效位置使用 mask 或零值隔离。

### Step 6：弱监督 QSTG loss

`models/loss.py:qstg_loss()` 已加入：

`L_mc/L_pq/L_fa/L_ex/L_gate/L_conn/L_role`、same-video false-negative ignore、reconstruction selector detach/confidence downweight、barrier detach 和完整诊断日志。QSTG 关闭或分支缺失时返回与 `words_logit` 相连的零。

### Step 7–10：CPL、mixture、runner、CLI

- `models/cpl.py`：clean `frame_fc` 双路径；三阶段 QSTG 调用；single/mixture 共用 proposal scoring；component semantic fusion；deterministic `_mask_words`；trainable scopes。
- `gaussian_mixture.py`：`center_logit_bias`、`width_logit_bias`、`importance_logit_bias` 全部 optional；None 调用与旧实现逐 tensor `torch.equal`。
- `runners/main_runner.py`：第五 loss、Stage A 仅 QSTG 总目标、Stage B/C weak-loss ramp、`qstg` combined selector、raw/refined 指标、diagnostics、v2 checkpoint、baseline warm-start allowlist。
- `train.py`：`--init-from-baseline`、`--init-weights`、`--selection-strategy qstg`、QSTG score/refine/mask/scope CLI；显式 CLI 覆盖优先，否则沿用配置。
- runner 顺序固定为 `_build_model()` → `set_trainable_scope()` → `_build_optimizer()`。

### Step 11–12：配置、脚本、工具与测试

- 配置：`config/activitynet/{baseline,qstg_stage_a,qstg_stage_b,qstg_full}.json`、`config/charades/{baseline,qstg_stage_a,qstg_stage_b,qstg_full}.json`；Charades 三阶段均使用独立 validation split。
- 脚本：ActivityNet 使用 `scripts/run_qstg_stage_a.sh`、`run_qstg_stage_b.sh`、`run_qstg_full.sh`；Charades 使用 `run_charades_stage_a.sh`、`run_charades_stage_b.sh`、`run_charades_full.sh`。
- 工具：phrase graph 100 条抽查、binding/quality inspect、baseline parity compare、Charades 固定 video-disjoint split 构建。
- 测试矩阵已覆盖 phrase/temporal/binding/quality/loss/refinement/integration/baseline/runner smoke 与 Stage B ramp。

## 3. 已完成验证

执行：

```bash
cd /data/chenyuan/videogrounding/w1m2/qstg
/home/chenyuan/miniconda3/envs/cpl/bin/python -m pytest tests/ -q
```

结果：`43 passed, 3 warnings`（包含 Charades split 与三阶段 pipeline 配置测试）。

已完成的额外验证：

- ActivityNet 实际尺寸：`B=2,T=200,Dv=500,W=20,N=5,M=50,P=8`，finite；
- Charades 实际尺寸：`B=2,T=200,Dv=1024,W=20,N=8,M=50,P=8`，finite；
- single/mixture 各一次 CPL forward/backward；
- CPU runner：2 batch train + 1 batch eval；
- eval connected refine：输出形状、边界和有限性；
- FP16 ops smoke：空 mask 与 adjacency 均 finite；
- ActivityNet phrase graph 随机抽查 100 条：`no_verb=2,no_noun=0,truncated=8,global_only=0,fallback=0`；
- ActivityNet baseline parity（legacy 与 deterministic masking）：
  两种模式的 `center/width/gauss_weight/words_logit` 均为 `0.0`，`max_abs_err=0.0`；
- baseline warm-start：加载 132 个 baseline tensor，缺失 102 个且全部为 `qstg.*`，optimizer/update 从 0 开始。
- Charades baseline warm-start：实际 vocab size `1112`，加载无 unexpected key，缺失参数全部为 `qstg.*`。

## 4. Bootstrap checkpoint

已复制（真实复制，不是软链）：

`/data/chenyuan/videogrounding/w1m2/qstg/checkpoints/bootstrap/activitynet-model-best.pt`

`/data/chenyuan/videogrounding/w1m2/qstg/checkpoints/bootstrap/charades-model-best.pt`

源 checkpoint 位于只读 baseline 工程对应数据集的 `model-best.pt`。

## 5. 阶段运行顺序

1. Stage 0：baseline parity，`qstg.enabled=false` 或零 residual/bias。
2. Stage A：`qstg_only`，LR `2e-4`，只优化 `L_mc/L_pq`，proposal prior/residual/bias 关闭；ActivityNet/Charades 均为 5 epoch。
3. Stage B：`qstg_and_proposal`，LR `1e-4`，接入 component residual 与完整弱监督项；ActivityNet/Charades 均为 10 epoch。
4. Stage C：`all`，LR `5e-5`，联合微调；ActivityNet/Charades 均为 10 epoch。

Stage B/C 使用上一阶段的 weights-only checkpoint；同阶段恢复使用 `--resume`。ActivityNet 与 Charades 均有独立三阶段 pipeline。Charades QSTG 使用 `data/charades/train_qstg.json` 与 `val_qstg.json`，固定规则为 `qstg-charades-val-v1`，当前统计 train/val 为 9616/986 queries、4378/464 videos，且与 test videos 互斥。

## 6. 仍需做的实验工作

- 在可用 GPU 上实际运行 ActivityNet/Charades Stage A/B/C，多 seed 训练与 validation 选模；
- 在 validation 上网格选择 `qstg_quality_weight/qstg_analytic_weight`；
- 记录 peak GPU memory、raw/refined 指标及 query 分组诊断；
- 固定 8–16 样本做 100–300 step overfit，确认 mc/pq margin、gate、quality variance；
- split 已重建；正式三 seed 结果归档前仍需在 GPU 上运行训练。
