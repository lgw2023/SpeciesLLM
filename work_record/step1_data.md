# Stage2 Step1 训练数据说明（`step1_data_*.sh`）

本文档对应仓库内三条「合并 → 打乱打平 →（可选）多机同步」流水线脚本，便于在**没有挂载真实数据盘**的开发机上理解：数据从哪来、长什么样、训练入口读什么路径。

- `work_record/step1_data_1_2_3.sh`：第一轮 **1st + 2nd + 3scbasecount** 三路合并后再打平。
- `work_record/step1_data_1_2.sh`：第一轮 **1st + 2nd** 两路合并后再打平（**不含 3scbasecount**）。
- `work_record/step1_data_1_3.sh`：第一轮 **1st + 3scbasecount** 两路合并后再打平（**不含 2nd**）。

三者除合并批次不同外，其余逻辑（过滤物种、打平参数、同步方式）一致。

---

## 环境与前置假设

- 脚本开头 `cd /data/disk1/SpeciesLLM`，并 `source .env`（部署相关变量以服务器 `.env` 为准）。
- Python：`PYTHON_BIN=/data/miniconda3/bin/python`。
- 仓库根目录下调用：`merge_macrogene_rounds.py`、`shuffle_flatten_macrogene.py`。

**Stage2 根目录**（脚本内写死，也可被环境覆盖习惯用法）：

`/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData`

与 `AGENTS.md` 一致的三份**上游已预处理输入**（step4，按物种分子目录）：

| 目录名 | 含义 |
|--------|------|
| `1st_pretrain_data_preprocessed_step4` | 第一轮预训练 macrogene parquet |
| `2nd_pretrain_data_preprocessed_step4` | 第二轮预训练 macrogene parquet |
| `3scbasecount_pretrain_data_preprocessed_step4` | scbasecount 相关批次 |

输入布局约定（每个物种一个子目录），与 `merge_macrogene_rounds.py` 文档字符串一致；**实测目录深度为两层**（`batch_root/<Species>/`，物种目录下不再分子文件夹）。

```text
batch_root/
├── Species_A/
│   ├── macrogene_0.parquet
│   ├── macrogene_1.parquet
│   ├── ...
│   └── Species_A_macrogene_lookup.parquet   # 每物种通常 1 个，见下「文件类型」
└── Species_B/
    └── ...
```

### 物种目录下的文件类型

| 类型 | 文件名模式 | 是否参与 merge / shuffle_flatten |
|------|------------|-----------------------------------|
| 训练分片 | `macrogene_<非负整数>.parquet` | **是**（`FILE_PATTERN` / `--pattern macrogene_*.parquet`） |
| lookup | `<Species>_macrogene_lookup.parquet` | **否**（名称不匹配 `macrogene_<idx>.parquet`，留在上游物种目录内） |
| 合并清单 | `merge_manifest.csv` | 仅出现在**合并输出根目录**，不在各物种子目录 |
| 打平清单 | `shuffle_manifest.csv` | 仅出现在**打平输出根目录** |
| 打平分片 | `all_flatten_part_<序号>.parquet` | 打平输出；训练 `glob` 常匹配目录下所有 `.parquet`（若目录内只有此类则无歧义） |
| 打乱中间桶（可选残留） | `_shuffle_tmp/bucket_*.parquet` | `shuffle_flatten` 的中间产物；默认桶数与 `SHUFFLE_BUCKETS` 一致 |

---

## 物种列表与各物种下 macrogene 分片数量（服务器实测快照）

以下统计来自 **`/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData`** 当前磁盘内容：每个物种目录下 **`macrogene_*.parquet` 文件个数**（即训练分片数）；`*_macrogene_lookup.parquet` 在上游为 **每物种 1 个**（1st：11、2nd：18、3sc：20）。若上游数据更新，行数与物种集合会变；可用 `find <batch_root> -mindepth 2 -maxdepth 2 -name 'macrogene_*.parquet' | ...` 自行复核。

### `1st_pretrain_data_preprocessed_step4`（11 物种）

| 物种 | `macrogene_*.parquet` 个数 |
|------|---------------------------|
| Callithrix_jacchus | 147 |
| Chlorocebus_sabaeus | 8 |
| Gallus_gallus | 57 |
| Gorilla_gorilla | 10 |
| Homo_sapiens | 2671 |
| Macaca_fascicularis | 86 |
| Macaca_mulatta | 195 |
| Mus_musculus | 998 |
| Pan_paniscus | 4 |
| Pan_troglodytes | 14 |
| Rattus_norvegicus | 174 |

**批次合计**：`macrogene_*.parquet` **4364**；`*_macrogene_lookup.parquet` **11**。

### `2nd_pretrain_data_preprocessed_step4`（18 物种）

| 物种 | `macrogene_*.parquet` 个数 |
|------|---------------------------|
| Anas_platyrhynchos | 47 |
| Bos_taurus | 70 |
| Canis_lupus_familiaris | 5 |
| Capra_hircus | 41 |
| Danio_rerio | 59 |
| Drosophila_melanogaster | 6 |
| Equus_caballus | 2 |
| Gallus_gallus | 24 |
| Homo_sapiens | 8009 |
| Macaca_fascicularis | 13 |
| Macaca_mulatta | 10 |
| Mesocricetus_auratus | 2 |
| Mus_musculus | 5278 |
| Oreochromis_niloticus | 7 |
| Oryctolagus_cuniculus | 21 |
| Ovis_aries | 19 |
| Rattus_norvegicus | 396 |
| Sus_scrofa | 426 |

**批次合计**：`macrogene_*.parquet` **14435**；`*_macrogene_lookup.parquet` **18**。

### `3scbasecount_pretrain_data_preprocessed_step4`（20 物种）

| 物种 | `macrogene_*.parquet` 个数 |
|------|---------------------------|
| Arabidopsis_thaliana | 98 |
| Bos_taurus | 25 |
| Caenorhabditis_elegans | 17 |
| Callithrix_jacchus | 10 |
| Danio_rerio | 220 |
| Drosophila_melanogaster | 102 |
| Equus_caballus | 7 |
| Gallus_gallus | 41 |
| Gorilla_gorilla | 1 |
| Heterocephalus_glaber | 25 |
| Homo_sapiens | 6349 |
| Macaca_mulatta | 218 |
| Mus_musculus | 5659 |
| Oryctolagus_cuniculus | 23 |
| Oryza_sativa | 16 |
| Ovis_aries | 12 |
| Pan_troglodytes | 19 |
| Solanum_lycopersicum | 1 |
| Sus_scrofa | 108 |
| Zea_mays | 7 |

**批次合计**：`macrogene_*.parquet` **12958**；`*_macrogene_lookup.parquet` **20**。

### 合并输出示例：`all_merged_full_no_1st_human_mouse_20260506_165244`

`step1_data_1_2_3.sh` 在该次运行下的合并目录：**29** 个物种子目录；仅包含重编号后的 `macrogene_*.parquet`（**无**各物种的 `*_macrogene_lookup.parquet`）；根目录另有 **`merge_manifest.csv`**（**1** 个文件）。

按物种的 **`macrogene_*.parquet` 个数**（合并后、打平前），按字母序：

| 物种 | 分片数 |
|------|--------|
| Anas_platyrhynchos | 47 |
| Arabidopsis_thaliana | 98 |
| Bos_taurus | 95 |
| Caenorhabditis_elegans | 17 |
| Callithrix_jacchus | 157 |
| Canis_lupus_familiaris | 5 |
| Capra_hircus | 41 |
| Chlorocebus_sabaeus | 8 |
| Danio_rerio | 279 |
| Drosophila_melanogaster | 108 |
| Equus_caballus | 9 |
| Gallus_gallus | 122 |
| Gorilla_gorilla | 11 |
| Heterocephalus_glaber | 25 |
| Homo_sapiens | 14358 |
| Macaca_fascicularis | 99 |
| Macaca_mulatta | 423 |
| Mesocricetus_auratus | 2 |
| Mus_musculus | 10937 |
| Oreochromis_niloticus | 7 |
| Oryctolagus_cuniculus | 44 |
| Oryza_sativa | 16 |
| Ovis_aries | 31 |
| Pan_paniscus | 4 |
| Pan_troglodytes | 33 |
| Rattus_norvegicus | 570 |
| Solanum_lycopersicum | 1 |
| Sus_scrofa | 534 |
| Zea_mays | 7 |

**合计**：`macrogene_*.parquet` **28088**；其余文件仅为根目录 **`merge_manifest.csv`**。

### 打平输出示例：`all_flatten_data_full_no_1st_human_mouse_20260506_165244`

| 位置 | 文件类型 | 数量（约） |
|------|----------|------------|
| 目录顶层 | `all_flatten_part_*.parquet` | **28057** |
| 目录顶层 | `shuffle_manifest.csv` | **1** |
| `_shuffle_tmp/` | `bucket_*.parquet`（中间桶） | **512**（与默认 `SHUFFLE_BUCKETS=512` 一致；可视情况清理） |

训练脚本若将整个目录当作数据源，应注意：**同一目录下若仍有 `_shuffle_tmp` 内的 `bucket_*.parquet`，可能被一并读入**。当前常用做法是训练路径指向仅含 `all_flatten_part_*.parquet` 的目录，或保证目录内无其它干扰 parquet。

---

## 三条脚本差异一览

| 项目 | `step1_data_1_2_3.sh` | `step1_data_1_2.sh` | `step1_data_1_3.sh` |
|------|------------------------|---------------------|---------------------|
| 合并批次 | `FIRST_VIEW`（筛过的 1st）+ **2nd** + **3sc** | `FIRST_VIEW`（筛过的 1st）+ **2nd** | `FIRST_VIEW`（筛过的 1st）+ **3sc** |
| `merge_macrogene_rounds.py` 的 `--batch-names` | `1st`, `2nd`, `3scbasecount` | `1st`, `2nd` | `1st`, `3scbasecount` |
| 中间视图目录 | `views/1st_no_human_mouse_${RUN_ID}` | `views_1_2/1st_no_human_mouse_${RUN_ID}` | `views_1_3/1st_no_human_mouse_${RUN_ID}` |
| 合并输出目录 | `all_merged_full_no_1st_human_mouse_${RUN_ID}` | `data_1_2_merged_full_no_1st_human_mouse_${RUN_ID}` | `data_1_3_merged_full_no_1st_human_mouse_${RUN_ID}` |
| 打平输出目录 | `all_flatten_data_full_no_1st_human_mouse_${RUN_ID}` | `data_1_2_flatten_data_full_no_1st_human_mouse_${RUN_ID}` | `data_1_3_flatten_data_full_no_1st_human_mouse_${RUN_ID}` |

`RUN_ID` 默认为执行时刻 `YYYYMMDD_HHMMSS`，因此每次完整跑通会生成**新的一整套**合并目录与打平目录；可通过显式设置 `RUN_ID` 固定命名以便复现。

---

## 对 1st 批次的特殊处理（排除人 / 小鼠， 这部分数据暂不加入训练）

三个脚本在构建「视作假的第一批目录」`FIRST_VIEW` 时，对 **`1st_pretrain_data_preprocessed_step4` 下每个物种子目录**做符号链接；但会 **跳过**：

- `Homo_sapiens`
- `Mus_musculus`

因此命名里的 `no_1st_human_mouse` 表示：**在合并使用的 1st 数据中不包含这两个人类与小鼠物种目录**（2nd / 3sc 侧仍按各自目录原样参与合并；若其中也有人鼠目录则会进入合并结果）。

`merge_macrogene_rounds.py` 里还有一层逻辑：当批次路径以约定后缀结尾时，会在 **原始** `1st_pretrain_data_preprocessed_step4` 目录上对人鼠做 auto-skip（与脚本手工建 `FIRST_VIEW` 互为补充，含义都是训练管线里常用人鼠与其它物种分开策略）。

---

## 流水线步骤（两段 Python）

### 1）合并：`merge_macrogene_rounds.py`

- **作用**：按物种对齐，把多批次的 `macrogene_*.parquet` 按顺序接到一起，输出目录仍为「按物种分子目录」，文件重命名为连续的 `macrogene_0.parquet`, `macrogene_1.parquet`, …。
- **模式**：脚本使用 `--mode copy`（拷贝，不破坏上游）。
- **并行**：`WORKERS`（默认 16）。
- **清单**：`merge_manifest.csv`，列包括：`species`, `new_index`, `new_filename`, `batch_order`, `batch_name`, `original_index`, `source_path`, `target_path`, `status`, `size_bytes`。

可选环境变量：`SKIP_MERGE=1`（跳过合并复用已有 `MERGED_DIR`）、`SKIP_EXISTING=1`（合并时跳过已存在目标文件）。

### 2）打乱并打平：`shuffle_flatten_macrogene.py`

- **输入**：合并目录 `MERGED_DIR`，递归匹配 `macrogene_*.parquet`。
- **输出**：**扁平目录** `FLAT_DIR`，文件名前缀默认为 `all_flatten_part_<序号>.parquet`，与训练脚本 `train_MNodes_torchrun_mfu_preindexparquet.py` 期望的「目录下一堆 parquet」用法一致。
- **默认参数（脚本里导出）**：
  - `ROWS_PER_FILE`：16384（每个输出文件最多行数；最后一文件可能更少，`--keep-remainder` 保留尾块）
  - `SHUFFLE_SEED`：42
  - `SHUFFLE_MODE`：`batch`（大模型数据常用的分批打乱实现）
  - `BATCH_FILES`：2048，`SHUFFLE_BUCKETS`：512
  - `--compression snappy`
  - `--validate-all-schemas`：写前校验 schema
  - 临时目录：`${FLAT_DIR}/_shuffle_tmp`
- **清单**：`shuffle_manifest.csv`，列为：`output_file`, `start_row`, `end_row_exclusive`, `num_rows`。
- **`SHUFFLE_MODE=batch` 的局限**：仅在批内（每批 `BATCH_FILES` 个文件，对当前数据约占 7%）做完整 shuffle，**批间只靠 input file 全局 permutation 做粗粒度混合**。input 文件数 < `BATCH_FILES` 的稀有物种（如本数据集中的 Solanum_lycopersicum 1 个 input、Mesocricetus_auratus 2 个、Pan_paniscus 4 个）会在 output 文件号轴上呈**段状聚集**（行级仍是均匀打散在那段区间内）。多 epoch 训练靠训练侧文件级 shuffle 抹平；**单 epoch + 大 world_size 训练**或对稀有物种 embedding 稳定性敏感时，请切 `SHUFFLE_MODE=external`（脚本已支持，需 ≈1× 输出体量临时空间，总 IO 约 2×）。

可选：`SKIP_FLATTEN=1` 跳过打平。

---

## 训练可读 Parquet Schema（打平后）

打平脚本 `shuffle_flatten_macrogene.py` 内嵌固定 **PyArrow Schema**（开发机无数据也可据此写代码 / mock）：

| 列名 | 类型 | 说明 |
|------|------|------|
| `X` | `list<float64>` | 宏基因表达（长度由上游 macrogene 定义；训练侧会有对应 `seq_len` / 模型配置） |
| `soma_joinid` | `int64` | 细胞 / 记录标识 |
| `dataset_id` | `int64` | 数据集编号 |
| `assay` | `int64` | Assay 类别 ID |
| `cell_type` | `int64` | 细胞类型 ID |
| `development_stage` | `int64` | 发育阶段 ID |
| `disease` | `int64` | 疾病 ID |
| `tissue` | `int64` | 组织 ID |
| `sex` | `int64` | 性别 ID |
| `tech_sample` | `int64` | 技术重复 / sample ID |
| `species` | `int64` | 物种 ID |
| `idx` | `int64` | 索引字段 |

训练时通过 `train_MNodes_torchrun_mfu_preindexparquet.py` 的 **`--data_path`** 指向**打平后的目录**（或其符号链接），由脚本自行 `glob` 该目录下的 `*.parquet`。

> **训练侧每 epoch 文件级 shuffle（关键依赖）**：`DistributedFileSampler` 每 epoch 用 `seed+epoch` 重做 `randperm(num_files)` 后再切给各 rank。这是 `SHUFFLE_MODE=batch` 模式下「磁盘上 batch-block 结构对训练无害」的唯一依据；**改训练 loader 时务必保留这一行为**，否则磁盘上的批块顺序会直接暴露给训练，且单 epoch 下无法被自然抹平。

---

## 多机同步

脚本默认：

- `HOSTS`：逗号分隔主机列表（默认脚本内写了三台内网 IP）；在列表**第一台**上执行本脚本时，会把本地 `FLAT_DIR` 用 `rsync` 同步到其余主机同名路径。
- `LOCAL_HOST`：默认为 `HOSTS` 中第一台；与本机 IP 匹配的主机会跳过 rsync。
- `SKIP_SYNC=1`：不做远端同步。

训练侧请直接在配置里写出本次的 `FLAT_DIR`（`..._${RUN_ID}` 打平目录的绝对路径），不要依赖任何统一软链接。

---

## 服务器当前快照（仅供参考，会随重跑变化）

以下数据在 **本仓库所在服务器** 上于文档编写时探测得到，用于量级直觉；**远程开发机若没有挂载 `SpeciesLLM_obs` 则不具备这些体积与文件**。

| 路径 | 体量（约） |
|------|------------|
| `Stage2_SpeciesLLMData/` 整棵目录树 | ~4.2 TB |
| `1st_pretrain_data_preprocessed_step4/` | ~174 GB |
| `2nd_pretrain_data_preprocessed_step4/` | ~577 GB |
| `3scbasecount_pretrain_data_preprocessed_step4/` | ~520 GB |
| 第一次三路全量打平目录 `all_flatten_data_full_no_1st_human_mouse_20260506_165244/` | ~1.2 TB |

各批次物种清单及每个物种目录下的 **`macrogene_*.parquet` 个数**见上文「**物种列表与各物种下 macrogene 分片数量**」；合并 / 打平目录的文件类型与数量见该节表格。

**该次跑出来的打平数据**（同上 RUN_ID）：

- `shuffle_manifest.csv` 中 `num_rows` 合计约 **459,680,768**（约 **4.60×10⁸** 条样本；与 `ROWS_PER_FILE=16384` 及 `--keep-remainder` 一致）

---

## 常用运行方式

```bash
# 三路合并 + 打平（默认 SHUFFLE_MODE=batch；批内充分 shuffle，批间粗粒度混合）
bash work_record/step1_data_1_2_3.sh BATCH_FILES=2048 WORKERS=32

# 严格全局打乱（external 模式：稀有物种均匀分布到所有 output 文件；总 IO 约 2×、临时空间 ≈1× 输出体量）
bash work_record/step1_data_1_2_3.sh SHUFFLE_MODE=external WORKERS=32

# 仅 1st（过滤后）+ 3sc
bash work_record/step1_data_1_3.sh BATCH_FILES=2048 WORKERS=32
```

仅调试管线时可组合：`SKIP_MERGE=1`、`SKIP_FLATTEN=1`、`SKIP_SYNC=1` 等；这些覆盖项必须写在脚本路径后面，或者写入 `.env`。

> **`--num_of_used_data` 警告**：训练脚本 `train_MNodes_torchrun_mfu_preindexparquet.py --num_of_used_data N` 取的是 `sorted(glob)[:N]`（**顺序前 N 个，不是随机抽样**）。小规模实验请生成独立的打平目录（如现有的 `all_flatten_data_test_100m`），**不要对全集套小 N**：会按 batch 边界切，稀有物种可能被完全丢失。

---

## 相关代码文件

| 文件 | 作用 |
|------|------|
| `merge_macrogene_rounds.py` | 多批次按物种合并、`merge_manifest.csv` |
| `shuffle_flatten_macrogene.py` | 全局打乱（多种模式）、输出 `all_flatten_part_*.parquet`、`shuffle_manifest.csv` |
| `train_MNodes_torchrun_mfu_preindexparquet.py` | 读取 `--data_path` 下 parquet 做预训练 |
| `work_record/step1_verify_shuffle.py` | 可对打平结果做抽样校验（如 `species` 列） |

---

## 给远程开发者的结论性提示

1. **真实数据不在仓库里**：大体量在 `/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/` 下；本地只需遵守 **目录约定 + Parquet schema** 即可开发与单测。
2. **物种目录内常有 lookup parquet**：`*_macrogene_lookup.parquet` **不参与** merge / shuffle 的 `macrogene_*.parquet` 匹配；不要认为物种目录下只有 macrogene 分片。
3. **训练消费的最终形态**：单层目录 + 大量 `all_flatten_part_*.parquet`，snappy 压缩，schema 固定。
4. **三条 Step1 脚本的本质区别**：分别合并 1st+2nd+3sc、1st+2nd、1st+3sc；产物目录前缀不同，勿混用。
5. **体量与行数**随上游与 RUN_ID 变化；精确数以对应目录下 `shuffle_manifest.csv` 汇总为准。
