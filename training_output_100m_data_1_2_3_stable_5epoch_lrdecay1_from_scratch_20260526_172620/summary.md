# 100M data_1_2_3 lr_decay_epochs=1 续训训练报告 — 2026-05-26 启动 / 05-28 续训

> 这是 100M `data_1_2_3_stable` 第一个**多 epoch（EPOCH=5）+ 快速 LR 衰减（lr_decay_epochs=1）**的实验，且经历了一次**续训**：首训在 epoch1→2 边界因 RAM 打爆被杀，改代码后从最近 checkpoint 接着跑。
> 分析依据 `metrics.0-0.jsonl`（rank0）。本目录续训段 `cluster_*` 多为 null，故主指标用 rank0 `loss_total`（每 `log_interval=10` 记一次）+ `grad_norm_raw` / `grad_action`。统计前已按 `update_step` 去重（keep-last），消除续训重放段。

---

## 一句话总结

续训机制本身完全健康（loss 边界连续、无 NaN、RAM 修复生效并跨过了害死首训的 epoch 边界），但 **`lr_decay_epochs=1` 让 LR 在 epoch1 末就砸到地板 1e-7，模型从 epoch1 中段（~step 30k）起基本停止学习**，epoch2 全程持平在 29–30，零进步。

**判定**：续训成功但调度无效 —— epoch2–5 是纯算力浪费，不应作为最终模型。最佳 loss 21.82@30081 出现在 epoch1 中段，稳态 floor ≈ 29–30。

---

## 1. 启动配置与运行状态

| 参数 | 实际生效值 |
| --- | --- |
| 数据 | `…/all_flatten_data_full_no_1st_human_mouse_20260506_165244_external` |
| 模型配置 | `args_2nd_run_100m.json`（`intermediate_size=5120`） |
| batch_size / epoch | `512` / `5` |
| **lr_decay_epochs** | **`1`**（衰减只覆盖 epoch1，之后 LR 恒在地板） |
| learning_rate / min_lr | `1e-6` / `1e-7` |
| warmup_iters / ratio | `2000` / `0.10`（实际 warmup 至 step≈3740 到峰值） |
| beta2 | `0.98` |
| grad_clip / clip_min / clip_max | `0.5` / `0.5` / `0.5`（**钉死，非自适应**） |
| grad_skip_ratio / **grad_skip_max** | `100` / **`1e11`（skip 保险丝被禁用）** |
| hard_raw_norm_limit | `1e11`（形同虚设） |
| amp_dtype | **`bfloat16`**（由 `auto` 经 torch_npu 的 cuda shim 误选；该 NPU 实际无可用 bf16） |
| num_batches / epoch | `37398` |

**两段运行（已从时间戳还原）：**

| 段 | 步数范围 | epoch | 起止 | 结局 |
| --- | --- | --- | --- | --- |
| 首训 (pid 945330) | 1 → 37395 | 仅 epoch1 | 05-26 17:40 → 05-28 08:20 | **OOM 杀进程于 step 37395/37398（epoch1 倒数第 3 步，正卡在 epoch1→2 内存峰值）**；rank0 日志无 error 行，符合 OOM 强杀 |
| *停机 3.4h* | | | | 改代码（reduce epoch memory pressure）/ git pull / rsync ckpt / 重启 |
| 续训 (pid 363306) | 37025 → 57344 | epoch1 尾 + epoch2 | 05-28 11:45 → 05-29 10:06（最后写入） | 从 `step-37024-loss-29.18` ckpt 恢复，跑到 epoch2 batch 19946/37398（~53%）。`NUM_WORKERS=4 / persistent_workers=false` 跨过了首训死掉的边界 |

去重：原始 57715 行 → 57344 个 update_step（续训重放了 37025–37395 段）。adaptive 统计：observed=57344，**clipped=57344，skipped=0**。

## 2. 与 LR sweep / 调度的横向对照

| 实验 | 峰值LR | 衰减 | 结构/代码 | 最低 loss | 末值 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| **本 run（lrdecay1）** | 1e-6 | **decay1（早砸地板）** | 5120 / clip0.5 | **21.82@30081** | ~30（持平） | **不发散但 epoch2 零学习** |
| lr1em6_epoch5（decay5） | 1e-6 | decay5（hold 峰值） | 5120 / clip0.5 | 15.90@24641 | **161.8** | 发散 |
| stable `_0514`（旧） | 1e-6 | ~1 epoch | **4864 / 无 clip** | **14.07@35381** | 15.07 | ✅ 稳定收敛 |
| lr5em6 / lr5em5 / lr1em5 / 5e-4 / 1e-4 | 5e-6~1e-4 | — | 5120 / clip0.5 | 15–30 | 90–200 | 全部发散 |

要点：同样峰值 LR 1e-6，**decay1（本 run）靠把 LR 早早砍到地板"躲过"发散，代价是停在 ~30 学不动；decay5 hold 住峰值则直接发散到 162**——同一个病的两种症状。唯一稳定到 loss 14 的是 5/15 之前的旧结构（4864、无 adaptive clip）。

## 3. Loss 轨迹（rank0 loss_total）

### 3.1 关键点

| 点位 | step | epoch | loss | gep | gepc | lr |
| --- | --- | --- | --- | --- | --- | --- |
| 起始 | 1 | 1 | 731.83 | 195.33 | 533.95 | 2.67e-10 |
| 最低 | 30081 | 1 | **21.82** | 17.45 | 3.98 | 2.01e-07 |
| 最高 | 31 | 1 | 755.89 | 206.40 | 546.97 | 8.29e-09 |
| 末次 | 57335 | 2 | 32.71 | 28.11 | 4.16 | 1.00e-07 |

### 3.2 阶段窗口

| 窗口(step) | n | mean | median | min | max |
| --- | --- | --- | --- | --- | --- |
| 1–2000 | 204 | 412.4 | 374.3 | 280.2 | 755.9 |
| 2001–5000 | 308 | 206.0 | 204.4 | 112.7 | 341.6 |
| 5001–10000 | 512 | 82.3 | 79.4 | 37.3 | 163.5 |
| 10001–20000 | 1024 | 32.9 | 30.6 | 22.9 | 54.2 |
| **20001–30000** | 1024 | **55.9** | 42.7 | 22.2 | **385.6** ← epoch1 中段不稳定尖峰 |
| 30001–37398 | 759 | 27.5 | 27.4 | 21.8 | 33.8 |
| 37399–45000 (ep2) | 778 | 30.3 | 30.3 | 26.8 | 35.5 |
| 45001–52000 (ep2) | 717 | 29.3 | 29.2 | 24.4 | 33.7 |
| 52001–57344 (ep2) | 546 | 29.3 | 29.3 | 24.1 | 34.3 |

### 3.3 阈值穿越

首次 <100: step 5661；<50: 8861；<40: 9611；<30: 14081；<25: 18031；<22: 30081（即全程最低）。此后**不再创新低**，反弹回 ~29–30 持平至今。

### 3.4 决定性证据：同一批 batch 跨 epoch 对比

控制数据顺序后，看 epoch1 **已收敛之后**的区段（前面 batch 因 epoch1 仍在 warmup 下降，不可比）：

| batch_index | epoch1 (LR) | epoch2 (LR 1e-7) | Δ |
| --- | --- | --- | --- |
| 12500–14999 | 33.32 (8.2e-7) | 28.92 | −4.40 |
| 15000–17499 | 28.64 (7.3e-7) | 28.93 | **+0.29** |
| 17500–19999 | 27.87 (6.3e-7) | 29.74 | **+1.87** |

**同样的数据，epoch2 不降反略升 → 模型已冻结，epoch2 零学习。**

## 4. 梯度行为

raw_norm（去重后 n=57344）：median `9683`，p95 `1.25e5`，p99 `3.34e5`，**最大 `1.11e6`**；动作统计：**clip=57344（每步都裁），skip=0（全程从未触发）**。

| 窗口(step) | raw median | raw p95 | raw max | clip / skip |
| --- | --- | --- | --- | --- |
| 1–2000 | 5792 | 1.07e6 | 1.11e6 | 2000 / 0 |
| 2001–5000 | 1553 | 1.31e5 | 6.00e5 | 3000 / 0 |
| 5001–10000 | 8574 | 8.22e4 | 7.41e5 | 5000 / 0 |
| 10001–20000 | 1.71e4 | 1.31e5 | 5.11e5 | 10000 / 0 |
| 20001–30000 | 3778 | 1.02e5 | 4.04e5 | 10000 / 0 |
| 30001–37398 | 2170 | 6668 | 1.91e4 | 7398 / 0 |
| 37399–45000 (ep2) | 8893 | 3.96e4 | 1.05e5 | 7602 / 0 |
| 45001–52000 (ep2) | 3.82e4 | 1.24e5 | 2.71e5 | 7000 / 0 |
| 52001–57344 (ep2) | 5.98e4 | 1.73e5 | 4.08e5 | 5344 / 0 |

稳态梯度几千、尖刺常冲到 1e5–1e6，但 **skip 阈值 = `ema×100`（≈7.7e6）随尖刺水涨船高，加上 `skip_max=1e11` 禁用了绝对保险丝，导致没有任何一步被 skip**。clip 到 0.5 在 AdamW 下被坐标级归一化抵消，未起实质保护。

## 5. 续训健康度核验（结论：完全正常）

- **参数正确**：`resume_update_step=37024`、`resume_skip_batches=37024`、`init_model/optimizer_path` 指向 step-37024 ckpt、`append_output_logs=true`。
- **loss 边界连续**：续训点前 (36500–37024) 均值 **30.15**，续训点后 (37025–37600) 均值 **29.83**，无跳变、无 NaN。这是判断续训健康的最强证据。
- **RAM 修复生效**：`num_workers 8→4`、`persistent_workers true→false`（+ commit f43789a），续训成功跨过了首训死掉的 epoch1→2 边界。
- metrics 中 37025–37395 的重复 update_step 属续训重放，画图/统计已去重，数据无损。

## 6. 解释

- `lr_decay_epochs=1` + cosine：LR warmup 至峰值 1e-6（step≈3740），在 epoch1 内单调衰减，**~step 35k（epoch1 末）触地板 1e-7，此后 epoch2–5 恒为 1e-7**。
- 模型在 epoch1 中段已逼近能力上限（step 18k 时 <25，step 30k 触底 21.82）；step 20–27k 期间出现不稳定尖峰（窗口 max 385），随 LR 衰减恢复，但稳态停在 ~27–30。
- epoch2 在 1e-7 地板上，对同一批数据复现 epoch1 的 floor（~29）且不再下降 → 学习实质停止。
- 21.82 的单步最低含数据顺序/恢复区噪声成分（`data_1_2_3` 三拼接数据集若未充分打散，epoch 内 loss 波动部分反映数据难度而非学习进度），不代表存在更优权重。

## 7. 结论与后续

- 实验目录：`training_output_100m_data_1_2_3_stable_5epoch_lrdecay1_from_scratch_20260526_172620`
- 最低 loss：`21.82@30081`（epoch1）；epoch2 末段均值：`29.3`
- 续训是否健康：**是**；是否建议作为最终模型：**否**（epoch2–5 不学习）

**根因（结合 sweep）**：当前代码（5120 结构 + clip 钉死 0.5 + skip 禁用 + bf16）下整条 LR sweep 要么发散、要么靠砍 LR 不学习；而模型本身被证明能到 loss 14（旧 `_0514`）。发散由 **GEP（masked MSE）在低 loss 区被离群基因的平方误差主导 → 梯度尖刺 → 无 skip 放行 → 污染 Adam** 驱动。

**已落地的修复（见 git 改动）→ 下一步跑 E1/E2（均 from scratch、fp32）：**
- **E1**：启用真 skip（`grad_skip_max=1e5`）+ 解除 clip 钉死 + 全程 cosine（`lr_decay_epochs=5`）+ `amp_dtype=float32`（弃 bf16）。
- **E2**：E1 + GEP/GEPC 改 Huber（`gep_loss=huber`，`huber_delta≈5`）从源头消尖刺。

## 8. 相关文件

- `metrics.0-0.jsonl` — 主分析依据（含续训重放，统计需按 update_step 去重）
- `loss_to_log.0-0.txt` — CSV 训练记录
- `log.0-0.txt` — 两段启动参数（pid 945330 / 363306）与状态
- `*_bak/` — 05-28 的图（grad_clip / loss_detail / training_curves / step_timing），缺最近一天 epoch2 数据
- 脚本参数来源：`work_record/step3_model_100M.sh`
