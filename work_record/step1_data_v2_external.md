# Stage2 v2 训练数据（external 全局打乱）

本文档对应 `work_record/step1_data_1_2_3_v2_external.sh`，用于在服务器上从 **v2 上游预处理目录** 生成新一轮三路合并打平数据。

与 v1（`step1_data_1_2_3.sh` + 默认 `SHUFFLE_MODE=batch`）相比，本批数据的两个关键变化：

1. **上游输入**：`*_preprocessed_step4_v2`（而非 `*_step4`）
2. **打乱模式**：`SHUFFLE_MODE=external`（每行随机 key → 分桶 → 排序输出，全局行级随机）

---

## 默认路径与产物

| 项目 | 默认值 |
|------|--------|
| `RUN_ID` | `v2_604_external_20260708` |
| `INPUT_1ST` | `${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4_v2` |
| `INPUT_2ND` | `${STAGE2_ROOT}/2nd_pretrain_data_preprocessed_step4_v2` |
| `INPUT_3SC` | `${STAGE2_ROOT}/3scbasecount_pretrain_data_preprocessed_step4_v2` |
| `MERGED_DIR` | `${STAGE2_ROOT}/all_merged_full_no_1st_human_mouse_${RUN_ID}` |
| `FLAT_DIR` | `${STAGE2_ROOT}/all_flatten_data_full_no_1st_human_mouse_${RUN_ID}` |

训练时 `--data_path` 指向 **`FLAT_DIR`**。

仍排除 **1st** 批次中的 `Homo_sapiens` / `Mus_musculus`；**2nd / 3sc** 中的人鼠目录会正常进入合并。

---

## 服务器已生成的 v2 external 数据集（2026-07-13）

统一根目录：

```text
/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData
```

按最终可用于训练的 shuffle + flatten 目录计，目前共生成 **3 份** v2 数据集。每份数据同时保留一个 merged 中间目录；merged 目录不另计为训练数据集。

| 数据组成 | merged 中间目录 | 最终 flatten 目录 | Parquet 分片数 | 已确认结果 |
|----------|-----------------|------------------|----------------|------------|
| 1st 去人鼠 + 完整 2nd + 完整 3sc | `all_merged_full_no_1st_human_mouse_v2_604_external_20260708` | `all_flatten_data_full_no_1st_human_mouse_v2_604_external_20260708` | **28057** | 459680768 行；external shuffle 验证通过 |
| 完整 1st-only | `data_1_only_merged_full_v2_604_1st_external_20260708` | `data_1_only_flatten_data_full_v2_604_1st_external_20260708` | **4344** | 已确认服务器目录与分片数；总行数及 shuffle 统计尚未记录 |
| 1st 去人鼠 + 完整 3sc | `data_1_3_merged_full_no_1st_human_mouse_v2_604_1_3_external_20260712` | `data_1_3_flatten_data_full_no_1st_human_mouse_v2_604_1_3_external_20260712` | **13630** | 223303680 行；80 文件抽样 external shuffle 验证通过 |

上述分片数来自服务器侧 2026-07-13 实测：

```bash
find /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_only_flatten_data_full_v2_604_1st_external_20260708 | grep parquet | wc -l
find /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_3_flatten_data_full_no_1st_human_mouse_v2_604_1_3_external_20260712 | grep parquet | wc -l
find /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_full_no_1st_human_mouse_v2_604_external_20260708 | grep parquet | wc -l
```

依次输出：

```text
4344
13630
28057
```

后续复核训练分片时，建议使用更严格的顶层文件名口径，避免把 `_shuffle_tmp` 中的临时 Parquet 计入：

```bash
find "$FLAT_DIR" -maxdepth 1 -type f -name 'all_flatten_part_*.parquet' | wc -l
```

最初的三个 `*_preprocessed_step4_v2` 目录是上游输入，不计入上述 3 份下游训练数据；测试目录和训练命令记录目录也不计入。

---

## 待生成策略：1st 去人鼠 + 完整 2nd（2026-07-16）

该策略使用 `work_record/step1_data_1_2.sh`，只合并：

- `1st_pretrain_data_preprocessed_step4_v2`，排除 `Homo_sapiens` 和 `Mus_musculus`
- 完整 `2nd_pretrain_data_preprocessed_step4_v2`

不读取 `3scbasecount_pretrain_data_preprocessed_step4_v2`。建议固定：

```text
RUN_ID=v2_604_1_2_external_20260716
MERGED_DIR=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_2_merged_full_no_1st_human_mouse_v2_604_1_2_external_20260716
FLAT_DIR=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_2_flatten_data_full_no_1st_human_mouse_v2_604_1_2_external_20260716
```

### 1）生成 merged + external flatten

```bash
cd /data/disk1/SpeciesLLM

RUN_ID=v2_604_1_2_external_20260716
STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData
LOG=work_record/step1_data_1_2_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log

df -h "$STAGE2_ROOT"

nohup bash work_record/step1_data_1_2.sh \
  STAGE2_ROOT="$STAGE2_ROOT" \
  INPUT_1ST="$STAGE2_ROOT/1st_pretrain_data_preprocessed_step4_v2" \
  INPUT_2ND="$STAGE2_ROOT/2nd_pretrain_data_preprocessed_step4_v2" \
  RUN_ID="$RUN_ID" \
  SHUFFLE_MODE=external \
  SHUFFLE_BUCKETS=512 \
  WORKERS=32 \
  ROWS_PER_FILE=16384 \
  SHUFFLE_SEED=42 \
  SKIP_SYNC=1 \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
tail -f "$LOG"
```

所有覆盖项均放在脚本路径后面，以适配脚本的 clean-env 重启逻辑。生成完成前，不把该策略计入“服务器已生成数据集”。

### 2）核对 manifest 并验证 shuffle

```bash
cd /data/disk1/SpeciesLLM

FLAT_DIR=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_2_flatten_data_full_no_1st_human_mouse_v2_604_1_2_external_20260716

test -f "$FLAT_DIR/shuffle_manifest.csv"
find "$FLAT_DIR" -maxdepth 1 -type f -name 'all_flatten_part_*.parquet' | wc -l
awk -F, 'NR>1 {rows += $4; files++} END {printf "manifest_files=%d rows=%.0f\n", files, rows}' \
  "$FLAT_DIR/shuffle_manifest.csv"

/data/miniconda3/bin/python \
  work_record/step1_verify_shuffle.py \
  "$FLAT_DIR" 80 \
  | tee "$FLAT_DIR/shuffle_verify_80.txt"
```

以最终输出中的 `Shuffle looks correct.` 作为 shuffle 质量通过信号。

### 3）验证通过后同步训练节点

```bash
cd /data/disk1/SpeciesLLM

RUN_ID=v2_604_1_2_external_20260716
STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData
MERGED_DIR=$STAGE2_ROOT/data_1_2_merged_full_no_1st_human_mouse_$RUN_ID
FLAT_DIR=$STAGE2_ROOT/data_1_2_flatten_data_full_no_1st_human_mouse_$RUN_ID

bash work_record/step1_data_1_2.sh \
  STAGE2_ROOT="$STAGE2_ROOT" \
  INPUT_1ST="$STAGE2_ROOT/1st_pretrain_data_preprocessed_step4_v2" \
  INPUT_2ND="$STAGE2_ROOT/2nd_pretrain_data_preprocessed_step4_v2" \
  RUN_ID="$RUN_ID" \
  MERGED_DIR="$MERGED_DIR" \
  FLAT_DIR="$FLAT_DIR" \
  SKIP_MERGE=1 \
  SKIP_FLATTEN=1 \
  SKIP_SYNC=0
```

---

## 默认流水线参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `SHUFFLE_MODE` | `external` | 写死在 wrapper 内 |
| `SHUFFLE_BUCKETS` | `512` | external 分桶数 |
| `WORKERS` | `32` | merge + flatten 并行度 |
| `ROWS_PER_FILE` | `16384` | 每个输出 parquet 行数上限 |
| `SHUFFLE_SEED` | `42` | 可复现随机种子 |
| `SKIP_SYNC` | `1`（generate 阶段） | 先本地生成，验证后再 sync |

**空间提示**：`external` 模式在 `${FLAT_DIR}/_shuffle_tmp` 下需要约 **1× 打平输出体量** 的临时空间；若同时新建 merged copy，峰值磁盘压力更大。generate 前会执行 `df -h` 预检（可用 `SKIP_DISK_CHECK=1` 跳过）。

---

## 三阶段运行方式

所有覆盖项必须写在脚本路径**后面**：

```bash
cd /data/disk1/SpeciesLLM
```

### 1）合并 + external 打平

```bash
bash work_record/step1_data_1_2_3_v2_external.sh
```

或显式指定 `RUN_ID`：

```bash
bash work_record/step1_data_1_2_3_v2_external.sh \
  RUN_ID=v2_604_external_20260708
```

### 2）验证 shuffle 质量

```bash
bash work_record/step1_data_1_2_3_v2_external.sh ACTION=verify
```

默认抽样 `80` 个输出文件，日志写入 `${FLAT_DIR}/shuffle_verify_80.txt`。

### 3）验证通过后同步到训练节点

```bash
bash work_record/step1_data_1_2_3_v2_external.sh ACTION=sync
```

---

## 与底层脚本的关系

wrapper 内部调用现有 `work_record/step1_data_1_2_3.sh`，不修改 merge / shuffle Python 逻辑：

- **generate**：完整 merge + flatten，`SKIP_SYNC=1`
- **verify**：检查 manifest、统计行数、运行 `step1_verify_shuffle.py`
- **sync**：`SKIP_MERGE=1 SKIP_FLATTEN=1 SKIP_SYNC=0`，仅 rsync `FLAT_DIR`

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `work_record/step1_data_1_2_3_v2_external.sh` | v2 三阶段入口脚本 |
| `work_record/step1_data_1_2_3.sh` | 底层 merge + flatten + sync |
| `work_record/step1_data_1_2.sh` | 1st 去人鼠 + 2nd 的 merge + flatten + sync |
| `work_record/step1_data.md` | v1 流水线总览 |
| `work_record/step1_verify_shuffle.py` | 打平结果统计验证 |
| `merge_macrogene_rounds.py` | 多批次物种合并 |
| `shuffle_flatten_macrogene.py` | external / batch 打乱打平 |
