# Shuffle 质量验证

## 背景

`step1_data_1_2_3.sh` 生成 flatten 数据后，需要确认数据是否真正跨物种、批次、文件打乱，而不是同物种数据聚集在一起。

## 使用方法

```bash
python work_record/verify_shuffle.py <flatten_data_dir> [sample_files]
```

示例：
```bash
python work_record/verify_shuffle.py /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_full_no_1st_human_mouse_20260506_165244
```

## 验证方法

脚本从 flatten 目录均匀抽样 N 个 `all_flatten_part_*.parquet` 文件，对每个文件做：

| 检验项 | 方法 | 说明 |
|--------|------|------|
| 行级物种混合 | 对比 observed vs expected change_ratio | expected 根据文件内实际物种比例计算，若两者接近则说明行顺序随机 |
| 统计检验 | Wald-Wolfowitz runs test | 对 Top 2 物种检验是否存在显著聚集 (p<0.01 为异常) |
| 稀有物种分布 | 跨文件分布检查 | 验证稀有物种不会全部集中在某个输出文件里 |

## 运行结果 (2026-05-06)

数据：`all_flatten_data_full_no_1st_human_mouse_20260506_165244`，共 28057 个输出文件，抽样 10 个。

物种 5+8（人和小鼠）占 90.2%，稀有物种（≤50 行/文件）有 6 个。

### 行级混合

| 指标 | 值 | 结论 |
|------|-----|------|
| Mean delta (observed - expected) change_ratio | **+0.00134** | 与完全随机打乱几乎无差别 |
| 最大绝对偏差 | 0.0034 (≈0.6%) | 在正常统计波动范围内 |

### 统计检验

- 10/10 文件全部通过 runs test (p 值 0.14~0.80)
- **无任何文件存在显著物种聚集**

### 稀有物种分布

| 物种 ID | 总行数 | 分布文件数 | 结论 |
|---------|--------|-----------|------|
| 28 | 40 | 4/10 | 正常分散 |
| 13 | 29 | 5/10 | 正常分散 |
| 4 | 29 | 2/10 | 正常分散 |
| 19 | 20 | 3/10 | 正常分散 |
| 0 | 14 | 2/10 | 正常分散 |
| 18 | 13 | 2/10 | 正常分散 |

无稀有物种集中在单一文件。

## 结论

**Shuffle 质量良好。** `SHUFFLE_MODE=batch, BATCH_FILES=2048` 配置在当前数据集上有效实现了跨物种随机打乱。
