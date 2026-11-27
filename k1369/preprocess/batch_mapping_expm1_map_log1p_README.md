# expm1 → gene-to-macrogene mapping → log1p

输入为自然对数 `log1p` 后的非负 gene 表达。若归一化表达为 N，原方案为
`log1p(N) @ W`，新方案为 `log1p(N @ W)`（公式省略基因对齐和重复列合并）。
两者通常不同，实验应固定输入细胞、基因集合和 mapping 权重。
`expm1` 仅撤销 log1p，不撤销 normalization，不能恢复原始 counts。
若 log1p 后经过 scale、截断、量化或其他不可逆操作，不能准确恢复；
本脚本不支持其他底数。float32 输入还原存在浮点误差。

## 运行

在安装了 numpy、pandas、scipy、anndata、dask、pyarrow 的 Python 环境中：

```bash
python batch_mapping_expm1_map_log1p.py \
  --csv task_config_1st_expm1_map_log1p.csv \
  --num-workers 2 --chunk-size 1024 \
  --log expm1_map_log1p_qc.log
```

配置来自 task_config_1st.csv，覆盖全部 11 个物种：

- 输入：`1st_pretrain_data_preprocessed_step3/<species>/obs/part.*.parquet`。
- 基因顺序：对应物种的 `var.parquet`，优先使用 `feature_name`。
- 权重：`SpeciesLLM_3rdrun_K1500_mapping_weights`，与现有第一轮配置一致。
- 输出：`1st_pretrain_data_preprocessed_step4_K1500_map_then_log1p/<species>/`。

对照应使用相同 K1500 权重产生的 `1st_pretrain_data_preprocessed_step4_K1500`。
如果研究的是另一套权重，请同步修改两种方案的配置；不要混用不同 K 的输出。
原有脚本、配置和数据均不修改。本次只创建脚本和配置，不执行全量转换。

## 行为与 QC

保留原脚本大小写不敏感匹配、重复 weight 行取平均、重复输入 gene 列求和、
未匹配基因丢弃的策略。关键顺序是**先逐列 expm1，再按 X_groups 合并重复基因**。
不会对权重重新归一化，也不会在 macrogene 层面再做 normalize_total。

保留 obs 其他列及行顺序，X 改为 macrogene log1p 表达，另写 macrogene lookup。
原有 obs QC 列保持输入含义，不代表新 macrogene 的统计量。
日志标注 `totals_scale=normalized_linear`：input_total、matched_total 和
macro_total 都在 log1p 前的线性尺度上统计。只有匹配权重的每行和为 1 时，
macro_total 才应约等于 matched_total；未匹配基因仍会造成损失。
负值、NaN/Inf、溢出、输入维度不匹配将报错，不静默裁剪。

矩阵运算按 chunk 执行，逆变换只分配当前 chunk；每个 parquet 分区仍整体读入内存。
最终结果为 float32。内存不足时先减少 num-workers，再减少 chunk-size。
支持 threads 和 synchronous；线程模式沿用权重缓存和日志锁。

默认跳过已存在且非空的输出，`--overwrite` 强制重算，`--fresh-log` 清空日志。
写入采用临时文件加原子替换，避免中断产生半个正式文件。
专用输出目录带有流程标记，拒绝直接复用无标记的旧 mapping 结果目录。
该标记不校验输入文件/权重版本；修改输入或权重后须使用新目录或 --overwrite。
