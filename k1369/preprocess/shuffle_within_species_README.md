# 按物种混合多来源单细胞数据

`shuffle_within_species.py` 默认读取本目录下的
`1st_pretrain_data_preprocessed_step4_K1500_map_then_log1p/`，自动发现全部物种。
依赖同目录的 `shuffle_species.py`（复用分桶及输出工具），以及 `numpy`、`pyarrow`。

## 运行

本机已验证可用的 Python：

```bash
/home/wangj0g/Documents/tools/ENTER/envs/celleval/bin/python shuffle_within_species.py --dry-run
/home/wangj0g/Documents/tools/ENTER/envs/celleval/bin/python shuffle_within_species.py
```

默认输出至新的同级目录：

```text
1st_pretrain_data_preprocessed_step4_K1500_map_then_log1p_shuffle_within_species/
  Homo_sapiens/
    macrogene_000000.parquet
    macrogene_000001.parquet
    ...
    Homo_sapiens_macrogene_lookup.parquet
    expm1_mapping_log1p.version
    shuffle_metadata.json
  Mus_musculus/
    ...
```

每个物种独立输出；每个细胞文件包含 16384 行，最后一个文件可以不足 16384 行。
lookup 文件是基因轴信息，不计入细胞分片。
不会修改输入文件，也不会覆盖已经存在的物种输出目录。

指定物种和位置：

```bash
/home/wangj0g/Documents/tools/ENTER/envs/celleval/bin/python shuffle_within_species.py \
  --species Pan_paniscus Chlorocebus_sabaeus \
  --output-dir ./within_species_trial \
  --seed 42
```

## 混合含义与完整性

以每个细胞为单位，在物种内跨全部输入文件进行全局随机置换。
`dataset_id`、`tissue`、`tech_sample` 连同 `X` 和其他字段整行移动，保持对应关系；
来源组合相同的细胞也会被打散。原有 `idx`、`soma_joinid` 均不重编号，表达值不重新计算。

这是保持来源原始数量和比例的随机混合，不是来源等量抽样。
每个分片的来源比例在统计意义上接近该物种总体比例，不保证每片含所有来源或精确比例。
稀有来源可能只出现在部分分片中。

流程先随机将细胞分配至磁盘桶，再对每个桶独立随机置换，并跨桶拼接到固定大小输出。
所有桶接收来自全部文件的细胞，避免只在单个原始文件中洗牌。
依次处理物种，不一次性载入整个物种。

脚本校验所有输入文件 schema、输入/输出细胞总数、每个来源三元组的细胞数，
并检查输出 parquet 的实际行数；全部成功后才将隐藏临时输出目录改名为最终物种目录。
`shuffle_metadata.json` 记录随机种子、参数、版本、计数和验证结果。
相同输入、参数、NumPy/PyArrow 版本及种子可复现行顺序；
单独选择某物种不会改变该物种的随机种子。

## 内存、磁盘和失败处理

- 默认最多 256 个桶，小物种自动减少桶数。第二阶段需同时容纳一个解压后的桶、其置换副本，以及输出缓冲；并非严格固定内存上限。
- 此数据的 X 为 1500 个 double，单细胞仅 X 就约 12 KB。大型物种处理可能需要数 GB 内存。
- 内存不足时增大 `--num-buckets`（例如 512），可减小单桶；但会增加文件句柄和第一阶段缓冲。每个桶可能打开一个文件，需留意 `ulimit -n`。
- 第一阶段缓冲默认每桶 64 行，读取批次为 4096 行，可通过 `--bucket-buffer-rows` 和 `--read-batch-rows` 调小。
- 需要额外磁盘保存最终输出和当前物种的临时桶。建议至少预留全部输入压缩体积两倍的可用空间，实际压缩率会随混洗改变，这不是硬性上限。
- 用 `--temp-dir /path/to/scratch` 指定临时桶目录；默认在输出根目录下创建。临时桶在成功使用后删除。
- 失败时保留隐藏的 `.物种名.partial-*` 和 `物种名.buckets-*` 目录用于排查；它们不是完整输出。确认对应进程已停止后可手动删除，再运行该物种。
- 已完成的物种不会被覆盖；中断后使用 `--species` 仅选择尚未完成的物种。没有从临时桶中间恢复的功能。
- `--dry-run` 仅读取全部输入文件元数据并检查 schema，不执行混洗、不创建输出。

## 测试

```bash
/home/wangj0g/Documents/tools/ENTER/envs/celleval/bin/python -m unittest -v test_shuffle_within_species.py
```

覆盖多个物种、多桶混洗、完整行与 schema/metadata 保留、种子复现、固定分片和尾片、
空数据、含空值的来源字段、lookup 复制、已有输出保护，以及 dry-run 不产生输出。
