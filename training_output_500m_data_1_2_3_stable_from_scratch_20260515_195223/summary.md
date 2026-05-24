# 500M data_1_2_3 全量预训练失败 — 2026-05-15 19:53

> **这是 500M 稳定性事件的原始失败实验**，整个 `work_record/stability_experiments.sh` 工具链都是为了诊断和修复它而生。后续的 [5/17 失败 smoke](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md) 和 [5/18 验证通过 smoke](../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md) 都是这次事件的后续。

---

## 一句话总结

500M data_1_2_3 全量预训练在 `--adaptive_grad_clip` **默认启用后的首个长时实验**中崩盘 — 跑了 ~42 小时 / 24,916 步，前 5,000 步出现典型「GEP/zero_prob 死亡、GEPC 正常」的 head 二分模式，第 11,895 步触发首次 `skip_norm`，最终从 step 12919 起**连续 11,998 步全部 skip**（约 47% 的运行时间在死循环烧 NPU），到运行结束权重再未更新。

---

## 1. 启动配置

| 参数 | 值 | 备注 |
|---|---|---|
| 启动方式 | 直接 `step3_model_500M.sh` | 当时还没有 `stability_experiments.sh` |
| 数据 | `all_flatten_data_full_no_1st_human_mouse_20260506_165244_external` | data_1_2_3 全量 |
| 模型配置 | `args_2nd_run_500m.json` | 24 层 × hidden 1280，`initializer_range=0.02` |
| `batch_size` | 256 | |
| `learning_rate` / `min_lr` | 1e-6 / 1e-7 | |
| `warmup_iters` | 2000 | |
| `beta1` / `beta2` | 0.9 / 0.98 | |
| `grad_clip` | 0.5 | warmup 期静态 clip |
| **`adaptive_grad_clip`** | **true** | ⚠️ 5/15 19:32 commit `43918dd` 默认开启，21 分钟后启动此次实验 |
| `grad_clip_ema_beta` | 0.98 | |
| `grad_clip_ratio` | 3.0 | clip threshold = ratio × ema |
| `grad_skip_ratio` | 10.0 | skip threshold = ratio × ema |
| `grad_clip_min` | 0.5 | |
| **`grad_clip_max`** | **1000.0** | ⚠️ 这次失败的关键值 |
| `grad_clip_warmup_steps` | 200 | |
| `max_train_steps` | 0 | 无限步 |

**没有 `grad_skip_max` / `grad_clip_hard_raw_norm_limit` / `grad_clip_max_consecutive_skips`** — 这些安全网在 5/17 dc1a291 才加入。也就是说**这次实验完全没有"上限保险"**，进入死循环后只能靠人工 kill。

## 2. 失败时序（按阶段）

### 阶段 A：warmup 表面正常（step 1 ~ 2000）

```
step    1: gep=189.7  zero=0.6867  gepc=747.9   raw=2.76e+08  ema=2.76e+08  → clipped to 0.5
step  501: gep=183.5  zero=0.6865  gepc=270.8   raw=2.62e+06  ema=1.38e+07  → threshold=1000
step 1001: gep=162.4  zero=0.6856  gepc=209.4   raw=1.88e+06  ema=2.51e+06
step 2001: gep=187.9  zero=0.6837  gepc=130.5   raw=4.99e+05  ema=6.73e+05
```

- warmup 期梯度被静态 clip 压到 norm=0.5，看起来正常
- warmup 结束后 threshold 从 0.5 跳到 1000（`min(3*ema, 1000)`），**clip 实质失效** — 健康梯度也就 1-10 量级
- **关键观察**：从 step 1 起 `loss_zero_prob ≈ 0.687`，已经卡在 `-log(0.5) = 0.693` 附近，**zero_prob head 从未真正开始学**

### 阶段 B：GEPC 在学，GEP/zero_prob 已死（step 2000 ~ 11000）

```
step 3001: gep=168.6  zero=0.6831  gepc= 52.2  gz=1.03    ← GEPC 快速下降
step 5001: gep=150.4  zero=0.6832  gepc= 25.5  gz=0.18
step 5221: gep=125.5 ★             ← loss_gep 全程最低点
step 7001: gep=156.1  zero=0.6832  gepc= 20.5  gz=0.15    ← gep 已开始回涨
step 9001: gep=157.5  zero=0.6841  gepc=  6.7  gz=0.06
step 10401:                        gepc=  5.19 ★          ← loss_gepc 全程最低点
step 11001: gep=157.9  zero=0.6814  gepc= 30.2  gz=0.058
step 11011:                                    gz=0.0297★ ← loss_gepc_zero_prob 最低点
```

精确的 head 二分死亡：

| Head | 路径 | 行为 |
|---|---|---|
| **loss_gep** | 每 token 输出 → MSE | 189 → 125 (@step 5221) → 回涨 → 死锁后徘徊 170±20 |
| **loss_zero_prob** | 每 token 输出 → sigmoid BCE | **全程 0.68 ± 0.005**，最低 0.6796 @ step 19361 (skip_norm 期间的随机抖动) |
| `loss_gepc` | CLS × macrogene_emb 内积 | 748 → 5.19 (@step 10401) ✅ **正常下降 99%** |
| `loss_gepc_zero_prob` | CLS × macrogene_emb 内积 | 2.15 → 0.030 (@step 11011) ✅ **正常下降 99%** |

> 所有用 macrogene 预训练 embedding 做内积解码的 head 都正常学习；所有从 raw transformer 每 token 输出投影的 head 都死了。**完全按 head 结构二分**。

### 阶段 C：梯度爆炸 + skip_norm 首次触发（step 11500 ~ 11942）

```
step 11501: raw=5.98e+07  ema=4.61e+07         act=clip
step 11801: raw=7.22e+08  ema=7.43e+08         act=clip  ← 在 90 步内涨了 10×
step 11891: raw=1.52e+10  ema=2.12e+09         act=clip  ← 又一个 20×
step 11895: raw=2.59e+10  ema=2.39e+09         act=skip_norm ★ ← 首次 skip
            (raw/ema = 10.83 > grad_skip_ratio=10)
```

- step 11895 起出现零星 skip
- step 11942 ~ 12917：**976 步连续 skip**（EMA 暂时无法跟上 raw 飙升）
- step 12919 起：**11,998 步连续 skip 直到运行结束**

### 阶段 D：永久 skip 死循环（step 12919 ~ 24916）

```
step 13001: raw=3.57e+10  ema=3.02e+09  ratio=11.8  skip_norm
step 16001: raw=3.89e+10  ema=3.02e+09  ratio=12.9  skip_norm
step 20001: raw=3.53e+10  ema=3.02e+09  ratio=11.7  skip_norm
step 24001: raw=3.70e+10  ema=3.02e+09  ratio=12.2  skip_norm
step 24916: raw=3.93e+10  ema=3.02e+09             skip_norm (final, manual kill)
```

- **EMA 被冻结在 3.02e+09**（因为 raw 一直高于 skip 阈值，没有 valid grad 来更新 EMA — 当时还没有 `skip_max` 这个 fuse 来打破死锁）
- **raw_norm 稳定在 3.5e+10 ± 0.5e+10**：模型权重停止更新后，每个 batch 的前向产生稳定的"死区"梯度
- 整个阶段权重**完全不变**，但 LR scheduler 还在按计划 decay，optimizer state 持续衰减，~12,000 步 = 约 20 小时机时浪费

### 整体 grad_action 统计

```
clip       : 11,900  (47.8%)
skip_norm  : 13,016  (52.2%)
skip_nan   :      0
```

**`raw_norm` 全程分位数**：
```
p50: 3.35e+10   ← 超过半数的 step 都在死循环中
p90: 3.85e+10
max: 4.31e+10
```

中位数已经是 3.35e+10 — 说明**这个数据点本身就标示着病态**：健康训练的 grad_norm 中位数应该接近 EMA 量级，这里却被死循环段拉到 30 亿量级。

## 3. 失败机理（链式因果）

### 表层：skip_norm 死循环锁住权重

step 12919 起 skip_threshold = `min(10×ema, ∞) = 3e+10`，而 raw 一直在 3.5e+10 — **永远 skip**。当时没有 `max_consecutive_skips` 安全网，所以训练既不前进也不退出。

### 中层：是什么让 raw_norm 从 1e+5 飙到 1e+10？

step 11500 ~ 11900 期间 raw 在 90 步内涨了 1000×。回看 step 7001~11500 的过程，grad 从 1e+6 量级断断续续爬升到 7e+7，每次"看似正常的 clip"实际上写回的是 **norm=1000 的更新方向**：

```
raw = 7e+7,  clip_threshold = 1000  →  scale factor = 1.4e-5
actual update written = (grad_direction) × norm 1000  ← 不是 norm 0.5！
```

在 lr=1e-6 下，**单步 ΔW ≈ 1e-3 量级**。500M 模型经不起这种尺度的"半失控"更新累积 4000 步 — 权重最终被推进激活/数值死区，输出失稳，进而每个新 batch 都产生 1e+10 量级的梯度。

### 深层：为什么 `grad_clip_max=1000`？为什么 GEP/zero_prob 一开始就没在学？

#### `GRAD_CLIP_MAX=1000.0` 是死亡条款

`grad_clip_max` 本意是 adaptive clip 的"安全上限"，但 1000 相对 100M 实验验证过的 `grad_clip=0.5` **大了 2000 倍**，完全失去保护意义。warmup 后 `threshold = min(3*ema, 1000)`，由于 500M 的 ema 长期在 1e+5 ~ 1e+8，threshold 长期顶到上限 1000 — adaptive clip 退化为"想 clip 也 clip 不到 0.5"的失效状态。

#### 但 GEP/zero_prob 死亡更早（疑似初始化）

值得注意的是 `loss_zero_prob` **从 step 1 就在 0.687**（uninformative logit≈0 状态），`loss_gep` 在 step 1001 就达到了一个准平台。这两个 head 似乎从初始化时就处于"难以学习"的状态。

对比同条件的 100M 实验（[`../training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839/`](../training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839/)），第 7001 步时：
- 100M: `gep=74.6, zero=0.41`（明显在学）
- 此次 500M: `gep=156.1, zero=0.683`（zero_prob 仍未启动）

**怀疑：500M = 24 层 × hidden 1280 + `initializer_range=0.02`** 导致深层 transformer 残差流方差线性叠加，per-position 输出尺度被深度噪声淹没；而 GEPC 通过 macrogene_emb 这个预训练 anchor 不受影响。这也精确解释了 head 按结构二分死亡的现象。

修复 `grad_clip_max` 只能让训练「不爆炸」，不能自动修复 GEP/zero_prob。5/18 启动的 [A 实验](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md) 已经验证：梯度护栏正常、0 skip，但 GEP/zero_prob 仍不学。下一步应改 `args_2nd_run_500m.json` 的 `initializer_range` 从 0.02 → `0.02/sqrt(2×24) ≈ 0.0029` 后复验。

## 4. 这次事件直接催生的修复

| 时间 | commit / 变更 | 内容 |
|---|---|---|
| 5/15 22:42 | `cad9378` | 把 adaptive grad clip 正式化为可配置功能 |
| 5/17 16:21 | `dc1a291` | 加入 `grad_clip_max_consecutive_skips`、`grad_clip_ema_runaway_factor`、`grad_clip_hard_raw_norm_limit` 三道安全网 |
| 5/17 17:14 | `695a1b4` | 引入 `grad_skip_max` 独立于 `grad_clip_max` 的 raw-norm 保险 |
| 5/17 17:20 | smoke #1 启动 | 验证修复 — **但因 launcher 还没更新，传参链断裂，配置全部静默回落到危险默认值**（见 [对应 summary](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md)） |
| 5/18 14:57 | `56a24e5` | 给 `launch_multinode_torchrun.sh` 补上 adaptive 参数的 CLI 传递 |
| 5/18 (uncommitted) | 5 个文件同步对齐 | `GRAD_CLIP_MAX` 默认 1000 → 0.5；`GRAD_SKIP_MAX` / `GRAD_CLIP_HARD_RAW_NORM_LIMIT` 默认对齐 1e+11 |
| 5/18 (uncommitted) | `check_stability_health.py` | 新增 `[5/5] primary-task health` 检查，捕获"梯度稳但 GEP/zero_prob 不学"的隐蔽失败 |
| 5/18 18:56 | smoke #2 验证通过 | 14 项核心参数全部 ✅（见 [对应 summary](../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md)） |
| 5/18 21:36 | A 实验启动 | 13,000 步验证 primary-task health；梯度护栏正常但 GEP/zero_prob 仍不学（见 [对应 summary](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md)） |

## 5. 经验教训

1. **新功能默认开启前必须先在目标尺度上做长跑验证**：`43918dd` 在 5/15 19:32 把 `adaptive_grad_clip` 设为默认 true，21 分钟后此次实验启动，没有任何先验 smoke 验证 — 直接撞上 `grad_clip_max=1000` 不适合 500M 的天坑。
2. **adaptive clip 的"安全上限"必须基于实测梯度尺度选择，不能拍脑袋**：`grad_clip_max=1000` 是按"反正不会到的极大值"取的，但对 500M 这个上限**比 100M 验证过的安全裁剪值大了 2000 倍**，等于没设。修复后改成与 100M 的 `grad_clip=0.5` 一致。
3. **必须给死循环安装"会自杀"的保险丝**：当时缺 `grad_clip_max_consecutive_skips` 这个机制，导致 step 12919 起的 11,998 步是肉眼可见的浪费（每步 ~6s，约 20 小时算力被烧），完全可以自动 kill。
4. **健康检查只看梯度控制器会被骗**：本次实验在 step 5000 时，`grad_action=clip 100%`、grad 距离 skip 阈值还很远，看起来非常"健康"，但 `loss_gep` 已经在 125 触底准备回涨、`loss_zero_prob` 已经卡死在 0.683 — primary-task 早就病了。必须加 head-level 健康检查。
5. **"按 head 结构二分死亡"是定位深层 transformer 初始化问题的强信号**：cell-level 内积 head 活、per-position head 死 — 强烈指向残差流方差累积 / scaled init 问题，而不像单纯优化器 / loss / 数据问题；13,000 步 A 实验已在 0 skip 条件下复现该二分，下一步需要 scaled init 复验。

## 6. 相关文件

- `metrics.0-{0..7}.jsonl`（每个 ~24,916 行 × 8 rank）— 完整训练 metrics
- `log.0-0.txt` 第 1 段 args dict — 实际生效的 CLI 参数（adaptive 系列已经存在，但 `grad_skip_max` / `grad_clip_hard_raw_norm_limit` 完全未配置）
- 后续修复实验链：
  - [5/17 smoke（自身失败，但揭露第二个 bug）](../training_output_500m_stab_smoke_500_from_scratch_20260517_172006/summary.md)
  - [5/18 smoke（验证完整修复链路）](../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md)
  - [5/18 A 实验（护栏正常，但 primary-task 仍失败）](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md)
- 控制对照：
  - 100M 同条件成功实验：`../training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839/`
