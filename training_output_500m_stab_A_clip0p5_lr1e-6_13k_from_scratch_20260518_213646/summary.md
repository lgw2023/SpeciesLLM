# A 实验 primary-task 失败记录 — 2026-05-18 21:36

> 这是 5/18 smoke 验证通过后的 13,000 步 A 实验，目标是回答一个具体问题：在 launcher 传参链已修复、`GRAD_CLIP_MAX=0.5` 已生效、不会再进入 skip 死循环的前提下，500M 的 GEP / zero_prob primary head 是否能恢复学习。

---

## 一句话总结

A 实验完整跑到 `max_train_steps=13000`，梯度护栏行为正常：13,000/13,000 步全部 clip、0 个 skip、raw peak `5.98e8`，距离 `1e11` 保险还有两个数量级。**但 primary-task health 明确失败**：`loss_gep` 只下降约 8%，`loss_zero_prob` 仍卡在 0.685 附近；与此同时 `loss_gepc` / `loss_gepc_zero_prob` 下降约 90%+。这说明 5/15 原始失败里至少有两个问题叠加：梯度护栏 / skip 死循环已修复，但 500M per-position head 不学的问题仍然存在，强烈支持深层初始化 / 残差流方差问题。

---

## 1. 启动配置

| 参数 | 实际生效值 | 备注 |
|---|---:|---|
| 启动时间 | 2026-05-18 21:37 | log 第 1 行 |
| 结束时间 | 2026-05-19 20:11 | `Complete pretraining` |
| `max_train_steps` | 13000 | 正常触达 |
| `learning_rate` / `min_lr` | `1e-6` / `1e-7` | 原始 LR |
| `warmup_iters` | 2000 | 与 5/15 一致 |
| `grad_clip_max` | 0.5 | 已修复值 |
| `grad_skip_ratio` | 100.0 | 已生效 |
| `grad_skip_max` | `1e11` | 已生效 |
| `grad_clip_hard_raw_norm_limit` | `1e11` | 已生效 |
| `grad_clip_max_consecutive_skips` | 50 | 已生效 |
| `initializer_range` | 0.02 | 仍是旧 500M 配置，正是本次要验证的问题 |

启动命令：

```bash
cd /data/disk1/SpeciesLLM
bash work_record/stability_experiments.sh A
```

---

## 2. 健康检查结论

运行：

```bash
python3 work_record/check_stability_health.py \
  training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/metrics.0-0.jsonl
```

核心输出：

```text
rows: 13000, step range: 1 → 13000

[1/5] grad_action distribution:
        clip :    13000  (100.0%)
   skip_norm :        0  (  0.0%)
    skip_nan :        0  (  0.0%)

[2/5] raw/EMA vs fuses:
      EMA peak:    3.933e+08
      raw peak:    5.984e+08
  ✅ raw peak / skip_max = 0.01
  ✅ raw peak / hard_raw_norm_limit = 0.01

[5/5] primary-task health:
  ❌ loss_gep window avg: 191.699 → 177.095 (drop 7.6%, required 50.0%)
  ❌ loss_zero_prob window avg: 0.689 → 0.685 (final max 0.650)
  info: loss_gepc window avg: 404.835 → 34.777 (drop 91.4%)
  ❌ GEPC is learning while GEP/zero-prob is stuck; this is task-unhealthy

=== VERDICT: ❌ UNHEALTHY — see flags above ===
```

---

## 3. 关键轨迹

### 3.1 梯度控制器：修复成功

| 区间 | raw_norm 中位数 | raw_norm 最大值 | EMA 最大值 | grad_action |
|---:|---:|---:|---:|---|
| 1-500 | `8.45e7` | `1.70e8` | `1.60e8` | 500 clip / 0 skip |
| 501-2000 | `1.55e6` | `7.04e6` | `5.13e6` | 1500 clip / 0 skip |
| 2001-4000 | `1.05e5` | `1.33e6` | `5.82e5` | 2000 clip / 0 skip |
| 4001-6000 | `7.26e4` | `6.00e5` | `2.31e5` | 2000 clip / 0 skip |
| 6001-8000 | `4.73e5` | `2.35e7` | `7.67e6` | 2000 clip / 0 skip |
| 8001-10000 | `2.92e6` | `2.29e8` | `1.06e8` | 2000 clip / 0 skip |
| 10001-12000 | `9.41e6` | `3.13e8` | `2.03e8` | 2000 clip / 0 skip |
| 12001-13000 | `2.32e7` | `5.98e8` | `3.93e8` | 1000 clip / 0 skip |

与 5/15 最大区别：这次 raw_norm 虽然后期重新升高，但没有进入 `1e10+`，也没有触发 skip 死循环。`grad_clip_max=0.5` 和 decoupled skip fuse 的修复有效。

### 3.2 Loss：GEPC 学，GEP / zero_prob 仍不学

按完整 loss 记录窗口统计：

| 窗口 | `loss_gep` avg | `loss_zero_prob` avg | `loss_gepc` avg | `loss_gepc_zero_prob` avg |
|---:|---:|---:|---:|---:|
| 1-1000 | 192.0 | 0.6882 | 318.5 | 2.624 |
| 1001-2000 | 180.3 | 0.6865 | 192.8 | 1.987 |
| 2001-4000 | 175.2 | 0.6848 | 73.8 | 1.553 |
| 4001-6000 | 174.4 | 0.6845 | 32.0 | 0.496 |
| 6001-8000 | 172.8 | 0.6849 | 18.1 | 0.246 |
| 8001-10000 | 175.2 | 0.6850 | 39.3 | 0.218 |
| 10001-12000 | 178.2 | 0.6851 | 46.9 | 0.265 |
| 12001-13000 | 175.7 | 0.6848 | 41.0 | 0.278 |

逐点对照：

| step | `loss_gep` | `loss_zero_prob` | `loss_gepc` | `loss_gepc_zero_prob` |
|---:|---:|---:|---:|---:|
| 1 | 208.47 | 0.6883 | 447.52 | 2.6737 |
| 1001 | 187.44 | 0.6883 | 252.76 | 2.2730 |
| 3001 | 161.97 | 0.6851 | 62.66 | 1.5760 |
| 5001 | 181.61 | 0.6844 | 29.01 | 0.4016 |
| 7001 | 190.06 | 0.6851 | 15.09 | 0.2657 |
| 9001 | 184.30 | 0.6879 | 162.53 | 0.6550 |
| 11001 | 197.90 | 0.6829 | 18.40 | 0.2371 |
| 12991 | 181.68 | 0.6854 | 26.44 | 0.1198 |

解释：

- `loss_gepc` 和 `loss_gepc_zero_prob` 大幅下降，说明数据流、optimizer、反向传播、macrogene_emb anchor 路径不是整体坏掉。
- `loss_zero_prob` 从头到尾贴在 `-log(0.5) ≈ 0.693` 附近，仍是 uninformative logit 状态。
- `loss_gep` 有局部低点（min 125.27 @ step 7801），但没有形成健康的持续下降；末段窗口均值仍在 175-177。

---

## 4. 与历史实验对照

### A vs 100M 健康实验

| step | 100M `gep` | A `gep` | 100M `zero` | A `zero` |
|---:|---:|---:|---:|---:|
| ~3000 | 172.77 | 182.90 | 0.6585 | 0.6859 |
| ~5000 | 122.82 | 174.98 | 0.4382 | 0.6856 |
| ~7000 | 73.85 | 176.36 | 0.4165 | 0.6831 |
| ~11000 | 31.06 | 175.83 | 0.3873 | 0.6847 |
| ~12991 | 35.89 | 181.68 | 0.4137 | 0.6854 |

100M 在 step 5000-7000 已经明显学会 zero_prob；A 到 step 13000 仍没有启动。

### A vs 5/15 原始 500M 失败

| step | 5/15 `gep` | A `gep` | 5/15 `zero` | A `zero` |
|---:|---:|---:|---:|---:|
| ~500 | 205.69 | 202.63 | 0.6861 | 0.6883 |
| ~3000 | 158.46 | 182.90 | 0.6841 | 0.6859 |
| ~7000 | 181.41 | 176.36 | 0.6827 | 0.6831 |
| ~11000 | 180.85 | 175.83 | 0.6827 | 0.6847 |
| ~12991 | 176.39 | 181.68 | 0.6832 | 0.6854 |

A 几乎复现了 5/15 的 primary head 死亡轨迹，但没有复现 skip 死循环。这个对照把两个问题拆开了：

1. `GRAD_CLIP_MAX=1000` + 缺少 skip 保险导致 5/15 后半程烧机时。
2. 500M GEP / zero_prob 从早期就不学，是独立于 skip 死循环的模型健康问题。

---

## 5. 结论

### 已修复

- launcher 传参链：A 的 log 第 1 行确认 `--grad_skip_*` / `--grad_clip_max=0.5` / `--max_train_steps=13000` 全部生效。
- 梯度护栏：0 skip，未触发 hard raw-norm fuse，未进入 5/15 的永久 skip 死循环。
- `grad_clip_max=0.5`：成功防止 raw_norm 被更新推到 `1e10+` 死区。

### 未修复

- `loss_gep` 未达到 50% 下降要求。
- `loss_zero_prob` 末段仍 > 0.65，实际约 0.685。
- GEPC 两个 cell-level head 正常学习，而 GEP / zero_prob 两个 per-position head 不学，结构性二分仍存在。

### 目前最强解释

`args_2nd_run_500m.json` 仍使用 `initializer_range=0.02`。对 24 层 × hidden 1280 的 500M 模型，这个初始化尺度很可能让深层 transformer 残差流方差累积，导致 per-position 输出路径从一开始就难以学习；GEPC 因为走 `CLS × macrogene_emb` anchor 路径，所以能继续下降。

这不是最终数学证明，但 A 实验已经把“只是梯度护栏没修好 / 只是 skip 死循环导致 primary 不学”的解释基本排除。

---

## 6. 下一步

优先做 scaled init 复验，而不是继续把重点放在 B/static：

1. 新建 500M scaled-init 配置，例如 `args_2nd_run_500m_init0p0029.json`，只把 `initializer_range` 从 `0.02` 改为 `0.02 / sqrt(2*24) ≈ 0.0029`。
2. 用该配置复跑 13,000 步 canary，其他 A 条件保持不变：LR `1e-6`、`GRAD_CLIP_MAX=0.5`、`MAX_TRAIN_STEPS=13000`。
3. 判定标准沿用 `[5/5] primary-task health`：
   - `loss_gep` 下降 ≥ 50%
   - `loss_zero_prob` 末段 < 0.65

如果 scaled init 后 `loss_zero_prob` 在 step 3000-7000 明显下降，说明 Issue #3 基本成立，可以再推全量训练；如果仍卡住，再回头检查 per-position head 架构、loss 权重、label 分布或 zero_prob target 构造。

---

## 7. 相关文件

- `metrics.0-0.jsonl`（13,000 行）— 主分析依据
- `log.0-0.txt` — 启动参数、训练结束和 adaptive grad clip summary
- 事件原点：[`../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md`](../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md)
- 前序 smoke：[`../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md`](../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md)
- 5/17 传参链失败 smoke：[`../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md`](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md)
