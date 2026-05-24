# Smoke 验证通过记录 — 2026-05-18 18:56

> 这是 5/17 smoke 暴露 launcher 传参断点后的复验，验证范围是传参链、梯度护栏配置和 500 步 warmup 阶段行为。事件原点见 [5/15 原始失败 summary](../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md)，直接前序见 [5/17 smoke 失败 summary](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md)，后续 A 判定见 [5/18 A summary](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md)。

## 一句话总结

500 步金丝雀 ✅ 全部按预期工作：传参链通了、护栏配置生效、梯度护栏行为符合设计、loss 在 warmup 阶段平稳下降。**只是 lr 还在 warmup（500/2000），不足以判定 GEP/zero_prob 这两个上次失败的 per-position head 是否真的修好** — 该判定已由后续 A 实验完成：护栏正常，但 primary-task 仍失败。

---

## 1. 修复的 bug（自上次 5/17 smoke 失败以来）

| Bug | 修复位置 | 修复后效果 |
|---|---|---|
| launcher 没传 adaptive 参数 | commit `56a24e5`（5/18 14:57） | sys.argv 现在带 `--grad_skip_*` / `--max_train_steps=` 等 |
| `GRAD_CLIP_MAX=1000` 默认过大 | 5 个文件全部对齐到 `0.5` | clip threshold 顶到 0.5 而不是 1000 |
| 各层 wrapper 默认值不一致 | `pretrain_pipeline.sh:113,114,116,120` + `launch_multinode_torchrun.sh:153,154,156,160` 改成安全默认 | 任何一层缺 export 都不再回退到危险值 |
| 健康检查只看梯度 | `check_stability_health.py` 新增 `[5/5] primary-task health` | 能捕获 "梯度稳但 GEP/zero_prob 不学" 这种隐蔽失败 |

详见 [`../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md`](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md) 的"问题点"章节。

## 2. 启动配置（意图 vs 实际生效）

| 参数 | 意图 | log 第 1 行实际 | 状态 |
|---|---|---|---|
| `max_train_steps` | 500 | **500** | ✅ |
| `grad_skip_ratio` | 100.0 | **100.0** | ✅ |
| `grad_skip_max` | 1.0e+11 | **1.0e+11** | ✅ |
| `grad_clip_hard_raw_norm_limit` | 1.0e+11 | **1.0e+11** | ✅ |
| `grad_clip_max` | 0.5 | **0.5** | ✅ |
| `adaptive_grad_clip` | true | true | ✅ |
| `grad_clip_ema_beta` | 0.98 | 0.98 | ✅ |
| `grad_clip_ratio` | 3.0 | 3.0 | ✅ |
| `grad_clip_min` | 0.5 | 0.5 | ✅ |
| `grad_clip_warmup_steps` | 200 | 200 | ✅ |
| `grad_clip_max_consecutive_skips` | 50 | 50 | ✅ |
| `learning_rate` / `min_lr` | 1e-6 / 1e-7 | 1e-6 / 1e-7 | ✅ |
| `warmup_iters` | 2000 | 2000 | ✅ |
| `batch_size` | 256 | 256 | ✅ |

**全部 14 项核心参数都对得上**。传参链 5 层 wrapper（`stability_experiments → step3 → smoke_500m_3node → pretrain_pipeline → launch_multinode_torchrun`）完整工作。

## 3. 运行情况

- 启动：2026-05-18 18:56:15
- 结束：自动停止（hit `max_train_steps=500`），未手动干预
- 完成 update_step：**500 / 500** ✅
- metrics.0-0.jsonl 行数：500

## 4. 健康检查输出

```
=== Stability health for .../metrics.0-0.jsonl ===
rows: 500, step range: 1 → 500
observed clip_threshold ceiling: 5.000e-01
configured skip_max: 1.000e+11
configured hard_raw_norm_limit: 1.000e+11

[1/5] grad_action distribution:
        pass :        0  (  0.0%)
        clip :      500  (100.0%)
   skip_norm :        0  (  0.0%)
    skip_nan :        0  (  0.0%)
     no_step :        0  (  0.0%)

[2/5] raw/EMA vs fuses:
      EMA peak:    9.469e+07
      raw peak:    9.962e+07
      skip_thr:    3.328e+08 → 1.000e+11
      clip_thr max:5.000e-01
  ✅ raw peak / skip_max = 0.00
  ✅ raw peak / hard_raw_norm_limit = 0.00

[3/5] consecutive_skips max seen: 0
      consecutive_clips max seen: 500
  ✅ well below safety threshold (50)

[4/5] loss trajectory:
  step 1 → 491
  loss_total: 629.23  →  453.93  (Δ -175.30)
              loss_gep:  187.505  →   183.068
        loss_zero_prob:    0.720  →     0.718
             loss_gepc:  439.037  →   268.170
   loss_gepc_zero_prob:    1.964  →     1.974

[5/5] primary-task health:
  skipped: observed 500 steps < primary_min_steps=4000

=== VERDICT: ✅ HEALTHY ===
```

## 5. 关键观察

### 5.1 梯度控制器行为 — 符合设计

- **clip threshold 全程 0.5** ✅ — `GRAD_CLIP_MAX=0.5` 生效，等价于 100M 实验验证过的静态裁剪
- **500/500 步全部 clip，0 个 skip** ✅ — 早期所有梯度都被压回 0.5，跟设计意图一致
- **EMA peak 9.47e+07，raw peak 9.96e+07** — 500M 的自然梯度尺度（与上次失败实验一致），但**这次没有撕扯到 1e+10+**（因为 update 被 cap 在 0.5，模型权重没被推到死区）
- **`raw peak / skip_max = 0.00`** — 离 1e+11 这道物理保险还差 3 个数量级，护栏配置上限合理
- **consecutive_clips = 500** — 100% 都在 clip，是预期行为（自然 grad >> 0.5 时 clip 一直触发）

### 5.2 loss 状态 — 不足以判定，但无异常迹象

| Head | step 1 | step 491 | min | min @ step |
|---|---|---|---|---|
| `loss_gep` | 187.51 | 183.07 | 156.24 | 161 |
| `loss_zero_prob` | 0.7201 | 0.7177 | 0.7173 | 431 |
| `loss_gepc` | 439.04 | 268.17 | 255.37 | 401 |
| `loss_gepc_zero_prob` | 1.9638 | 1.9738 | 1.9393 | 41 |

- `loss_total` 下降 28% (629→454) — warmup 期的正常 GEPC 主导下降
- `loss_gep` 微动（187→183），`loss_zero_prob` 几乎不动（0.720→0.718）— **但这不能下结论**

**为什么不能下结论**：step 491 时 lr ≈ 6.56e-08，仅是峰值 1e-6 的 **6.5%**（warmup 才走了 25%）。在这种 lr 下任何 head 都不该有显著变化。

**横向对比 100M 同期数据（来自 `training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839`）**：

| @step ~500 | 100M（健康） | 这次 500M smoke |
|---|---|---|
| loss_gep | 189.79 | **183.07** |
| loss_zero_prob | 0.6881 | 0.7177 |
| loss_gepc | 198.50 | 268.17 |

GEP 跟 100M 接近，zero_prob 略高（差 0.03，预期是 warmup 不同节奏），GEPC 略高（500M 起点 439 vs 100M 起点 ~280）。整体没有"显著恶化"的信号。

### 5.3 primary-task health 自动跳过 — 这是设计正确

`[5/5]` 显示 `skipped: observed 500 steps < primary_min_steps=4000`。**这是预期且正确的行为** — `--primary-min-steps=4000` 就是为了避免在 warmup 阶段错误判死刑（100M 也是要到 ~3000 步才能看出 zero_prob 真正在学）。后续 A 实验已经完成该判定。

## 6. 问题点

**本次 smoke 范围内无未解决问题**。所有上次 smoke 暴露的传参链 / 护栏配置 bug 都已修复且验证生效。

后续 [A 实验](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md) 已经回答 **Issue #3（500M 自然梯度尺度异常 → 怀疑深层初始化）**：梯度护栏正常、0 skip，但 GEP/zero_prob 仍不学，强烈支持 scaled init 方向。

## 7. 下一步

`stability_experiments.sh A` 已完成，目录：

```text
training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646
```

健康检查：

```bash
python3 work_record/check_stability_health.py \
  training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/metrics.0-0.jsonl
```

结果：`VERDICT: ❌ UNHEALTHY`。A 的 `[5/5] primary-task health` 判定：

- `loss_gep` window avg: 191.699 → 177.095，只降 7.6%（要求 50%）
- `loss_zero_prob` window avg: 0.689 → 0.685，末段仍高于 0.65
- `loss_gepc` window avg: 404.835 → 34.777，正常下降 91.4%

下一步：改 500M 配置的 `initializer_range` 从 0.02 → `0.02 / sqrt(2*24) ≈ 0.0029` 后复跑同等 13,000 步 canary。

## 8. 启动命令（供归档）

```
cd /data/disk1/SpeciesLLM
bash work_record/stability_experiments.sh smoke
```

启动版本：包含 commit `56a24e5`（5/18 14:57 launcher 修复）+ 本地未提交修改（`pretrain_pipeline.sh` / `launch_multinode_torchrun.sh` 默认值对齐 + `stability_experiments.sh` COMMON_ARGS `GRAD_CLIP_MAX=0.5`）。

## 9. 相关文件

- `metrics.0-0.jsonl`（500 行）— 主数据
- `log.0-0.txt` 第 1 段 args dict — 14 项参数全部 ✅
- 事件原点：[`../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md`](../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md)
- 失败对照实验：[`../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md`](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md)
- A 实验判定：[`../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md`](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md)
