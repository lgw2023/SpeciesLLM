# Smoke 失败实验记录 — 2026-05-17 17:20

> 这是 5/15 原始失败后的第一轮 500 步 smoke 验证，目标是验证梯度护栏修复；但它自身因为 launcher 传参链断裂而失败，并暴露出第二个独立 bug。事件原点见 [5/15 原始失败 summary](../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md)，后续修复验证见 [5/18 smoke summary](../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md)。

## 一句话总结

意图跑 500 步金丝雀，**实际跑了 13,051 步**且 adaptive 安全网全部失效；500M 模型出现「**GEP / zero_prob 两个 per-position head 完全不学，GEPC 两个 cell-level head 正常**」的精确二分死亡，**根本原因是两个独立 bug 叠加**：launcher 没传新参数 + `GRAD_CLIP_MAX=1000` 默认值过大。

---

## 1. 启动意图 vs 实际生效

| 参数 | `stability_experiments.sh smoke` 意图 | log 第 1 行实际生效 | 状态 |
|---|---|---|---|
| `max_train_steps` | **500** | **0**（无限步） | ❌ |
| `grad_skip_ratio` | 100.0 | **10.0** | ❌ |
| `grad_skip_max` | 1.0e+11 | **0.0**（禁用） | ❌ |
| `grad_clip_hard_raw_norm_limit` | 1.0e+11 | **0.0**（禁用） | ❌ |
| `grad_clip_max` | 1000.0 | 1000.0 | ⚠ 是当时的默认 |
| `adaptive_grad_clip` | true | true | ✅ |
| `grad_clip_ema_beta` | 0.98 | 0.98 | ✅ |
| `grad_clip_ratio` | 3.0 | 3.0 | ✅ |
| `learning_rate` / `min_lr` | 1e-6 / 1e-7 | 1e-6 / 1e-7 | ✅ |
| `warmup_iters` | 2000 | 2000 | ✅ |
| `batch_size` | 256 | 256 | ✅ |

**所有新加的护栏参数都丢了**。train 脚本走的全是 argparse 默认值 — 跟 `stability_experiments.sh` 想要的安全配置毫无关系。

## 2. 实际跑了多少 / 怎么停的

- 启动：2026-05-17 17:20:46
- 结束：手动 kill 于 2026-05-18 16:22（任务跑了 ~23 小时）
- 完成 update_step：**13,051**（设计 500 步）
- metrics.0-0.jsonl 行数：13,054

## 3. 关键日志数据

### 3.1 loss 轨迹（按 head 二分死亡）

```
step       gep      zero      gepc      gepc_zero
   1    187.51    0.7357    276.66    2.220   ← 起始
 161    156.24   ...        ...       ...     ← gep 短暂触底（warmup 中）
 501    181.00    0.7128    247.54    1.986
1001    175.39    0.6853    200.33    1.950
2271    124.55    ...       ...       ...     ← loss_gep 全程最低点
4001    188.01    0.6827     37.99    0.436   ← gep 已经回涨
7501    177.22    0.6840     18.85    0.093   ← gep 完全不动
12431    ...      0.6801    ...       ...     ← zero_prob 最低点（仍≈log(2)）
13051   207.26    0.6824      6.13    0.039   ← 结束（gep 比起始还高 11%）
```

- **`loss_gep`** 从 187 → 124（@step 2271，warmup 后短暂学到一点）→ 207（结束，回升到起点之上） — **典型「学过又忘」**
- **`loss_zero_prob`** 全程在 0.68 上下徘徊，最低 0.6801 ≈ -log(0.5) = 0.6931 — 模型输出 sigmoid logit ≈ 0，**uninformative 状态**
- **`loss_gepc`** 277 → 6.1（**正常下降 98%**） ✅
- **`loss_gepc_zero_prob`** 2.2 → 0.04（**正常下降 98%**） ✅

> 用 macrogene_emb 内积解码的两个 cell-level head 学得很好；从 transformer 每 token 输出直接投影的两个 per-position head 完全死了。**完全按 head 结构二分**。

### 3.2 梯度分布（自然范数病态）

13,054 个 step 的 `grad_norm_raw` 分布：

```
p1   : 9.16e+03
p10  : 2.56e+04
p25  : 2.55e+05
p50  : 1.97e+06     ← 中位数已经是百万量级
p75  : 1.35e+07
p90  : 7.31e+07
p99  : 2.55e+08
p99.9: 3.17e+08
max  : 3.76e+08
mean : 2.25e+07
```

- **81% 的 step 满足 `raw_norm > 100 × clip_threshold(1000)`** — 即 clip 一直在以 100× 以上倍率压缩梯度
- 健康 500M 训练自然梯度范数应该在 1~10 量级，这里整体高出 **5-6 个数量级**

### 3.3 grad_action 分布

```
clip       :  11,900  ( 47.6% )   ← 大部分时间在硬 clip
skip_norm  :  13,016  ( 52.0% )   ← 进入死区后大量 skip
```

- 全程 `clip_threshold` 顶到 ceiling=**1000.0**
- step 1~11900：raw_norm 1e+4 ~ 1e+8 范围内被压到 norm=1000 写回
- step 12000~：raw_norm 飙到 **3.5e+10**，触发 `grad_skip_ratio=10 × ema=2.9e+9` 的 skip 阈值，连续 skip 12,601 次直到任务被 kill

## 4. 问题点（按重要性排序）

### Issue #1: launcher 传参链断点（已修复）

**现象**：`stability_experiments.sh smoke` 设的 `GRAD_SKIP_RATIO=100 / GRAD_SKIP_MAX=1e+11 / GRAD_CLIP_HARD_RAW_NORM_LIMIT=1e+11 / MAX_TRAIN_STEPS=500` 全部未生效。

**根因**：
- 5/17 17:14 commit `695a1b4` 更新了 `stability_experiments.sh` COMMON_ARGS
- 5/17 17:20 启动此次 smoke
- **但 `scripts/launch_multinode_torchrun.sh` 直到 5/18 14:57 commit `56a24e5` 才加上 `--adaptive_grad_clip=` / `--grad_skip_*` / `--max_train_steps=` 这一串 CLI 行**
- 启动时刻的 launcher 不知道这些参数 → train 脚本只收到旧 CLI → 走 argparse 默认值

**修复**：commit `56a24e5`（5/18 14:57）已补上 launcher 这一行；又在后续修改中把 `scripts/pretrain_pipeline.sh` 和 `scripts/launch_multinode_torchrun.sh` 的旧默认值从 `10.0 / 0.0 / 1000.0 / 0.0` 改成 `100.0 / 1e+11 / 0.5 / 1e+11` 做 defense-in-depth。

### Issue #2: `GRAD_CLIP_MAX=1000` 过大（已修复）

**现象**：adaptive clip 在 warmup 后期 `threshold = min(ratio*ema, grad_clip_max)` 长期顶在上限 1000；当 raw_norm=7e+7 时，被裁的 update norm = 1000，方向是 raw 梯度方向。在 lr=1e-6 下，单步 ΔW ≈ 1e-3，足以把 500M 模型推入激活/数值死区。

**根因**：`grad_clip_max=1000` 是按"安全上限"逻辑设的，但相对 100M 实验验证过的 `grad_clip=0.5` 大了 **2000 倍**，本质失去了 clip 的保护意义。

**修复**：把 `GRAD_CLIP_MAX` 默认值改为 **0.5**（5 个文件全部同步：`stability_experiments.sh` COMMON_ARGS / `step3_model_500M.sh` / `smoke_500m_3node.sh` / `pretrain_pipeline.sh` / `launch_multinode_torchrun.sh`）。

### Issue #3: 500M 自然梯度尺度异常（A 实验已复现，待 scaled init 复验）

**现象**：梯度范数中位数 1.97e+6，远超 100M 模型同条件下应有的 1~10 量级。

**疑似根因**：500M 是 24 层 × hidden 1280，`initializer_range=0.02` 与 100M（12 层 × 640）相同。深层 transformer 在 std=0.02 初始化下残差流方差线性叠加，per-position 输出端梯度尺度被深度放大。**这也解释了为什么 GEP / zero_prob（per-position head）死了，而 GEPC（CLS × macrogene_emb 内积）活着** — 后者输出经过 macrogene_emb 这个预训练 anchor，不受残差流方差爆炸影响。

**已验证**：GRAD_CLIP_MAX=0.5 修复后的 [A 实验](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md) 完整跑到 13,000 步且 0 skip，但 GEP/zero_prob 仍卡住。这高度支持初始化问题 → 下一步把 `args_2nd_run_500m.json` 的 `initializer_range` 从 0.02 → `0.02 / sqrt(2*24) ≈ 0.0029`（GPT-2 scaled init）后复验。

## 5. 经验教训

1. **任何加了新参数的 commit，必须沿着完整启动链路 verify 一遍 sys.argv**：5 层 wrapper（`stability_experiments → step3 → smoke_500m_3node → pretrain_pipeline → launch_multinode_torchrun`）任何一层漏掉 export 都会让上游配置静默掉到默认。
2. **每层 wrapper 都应该把"安全默认"对齐**：不能依赖"上游肯定会设"。这次的 fallback 就是因为 `pretrain_pipeline.sh` 和 `launch_multinode_torchrun.sh` 的旧默认是有害值（10.0 / 0.0 / 1000.0）。
3. **健康检查脚本不能只看梯度控制器**：`grad_action=clip 100%` 看起来 "HEALTHY"，但同时 `loss_gep` 不降反升、`loss_zero_prob` 卡在 -log(0.5) ≈ 0.68 — 这种「梯度稳了但任务没在学」的失败模式必须由 primary-task 检查捕获（已在 `check_stability_health.py [5/5]` 中加上）。
4. **"按 head 结构二分"是定位深层 transformer 初始化问题的强信号**：cell-level 内积 head 活、per-position head 死，优先指向残差流方差累积 / 初始化问题，而不是单纯优化器 / loss / 数据问题；A 实验已在 0 skip 条件下复现该二分，下一步需要 scaled init 复验。

## 6. 启动命令（供归档）

```
cd /data/disk1/SpeciesLLM
bash work_record/stability_experiments.sh smoke
```

启动版本：commits up to `695a1b4`（5/17 17:14），但 launcher 还是旧版（缺 `56a24e5` 5/18 14:57）。

## 7. 相关文件

- `metrics.0-0.jsonl`（13,054 行）— 主分析依据
- `log.0-0.txt` 第 1 段 args dict — 用于核对实际生效的 CLI
- 事件原点：[`../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md`](../training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/summary.md)
- 下次实验对比：[`../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md`](../training_output_500m_stab_smoke_500_from_scratch_20260518_185615/summary.md)
- A 实验判定：[`../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md`](../training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646/summary.md)
