# K1369 预训练（Ascend）

本目录对应合作者任务 396057 的 351M 训练流程。旧 SpeciesLLM 基线在提交 `644db0b`。本分支的 `cb98cf0` 保存了收到的预处理代码、任务表和原始 H100 代码包，其中包括启动器、Pixi 锁文件及校验清单。可用 `git show cb98cf0:k1369/README.txt` 查阅合作者的续训说明。

## 数据和物种采样

从第一、第二批的 Step 3 开始。每个物种提供 `obs/part.*.parquet` 和描述列顺序的 `var.parquet`。`X` 是经过归一化并取自然对数 log1p 的 gene 表达。转换顺序为逐基因 expm1、对齐并合并重复基因、乘物种映射权重、log1p。物种内洗牌保留整行及各来源的细胞数，默认每个完整分片 16,384 行。

两批任务表分别含 11 和 18 条任务。实际物种范围以各批 CSV 为准。将 `2ndrun_K1369.zip` 解压至服务器数据根目录，保留 `2ndrun_K1369/mapping_weights/` 和三份 NPY。31 份映射的列轴都是 `MacroGene-0` 至 `MacroGene-1368`。训练词表将 `gene_1` 至 `gene_1369` 映射到位置 0 至 1368，再单独加入 CLS。三份 NPY 的行数都是 1369，列宽依次为 ESM2 1280、描述 1280、DNA 2560。NPY 自身没有行标签，部署时使用同一发布包的完整资源。

权重包 SHA-256 为 `9a47749247c43b2789e423eef08f511d4132a9a4ebba2896371828a23ea1400f`。数据、静态 embedding 和 checkpoint 在服务器上管理。

从仓库根目录运行下列命令。CSV 内使用服务器布局，可按部署位置编辑副本并用 `--csv` 传入。更换输入或映射权重时使用新的输出目录，转换脚本默认跳过已存在的分片。

```bash
set -e
stage2_root=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData
for batch in 1st 2nd; do
  python k1369/preprocess/batch_mapping_expm1_map_log1p.py \
    --csv "k1369/preprocess/task_config_${batch}_expm1_map_log1p.csv" \
    --log "$stage2_root/${batch}_k1369_mapping.log"
  python k1369/preprocess/shuffle_within_species.py \
    --input-path "$stage2_root/${batch}_pretrain_data_preprocessed_step4_K1369_map_then_log1p"
done
```

训练读取每批的 `*_shuffle_within_species/<物种>/macrogene_*.parquet`。第一批按物种留尾部分片验证，第二批默认留最后三个分片，小物种按 `--small_species_validation` 处理。任务 396057 采用第二批占 75%、各批内部按物种训练细胞数的 0.6 次幂分配配额，支持重复抽样。第三批需自行提供映射后的同布局分片、统一的 metadata ID 及相应容量配置。

## 在服务器上验证和启动

沿用仓库目标节点的 torch、torch_npu 和 CANN 8.2.1.RC1 环境，数据处理需要 pandas、pyarrow、anndata、scipy 和 dask。训练入口使用 FP32、AdamW 和 HCCL，选卡依据 torchrun 的 LOCAL_RANK。DataLoader 在主机端预取，张量在当前设备流上复制。表达值用 Huber，非零标签在掩码位置用 FP32 logits BCE。吞吐量记录为 cells/s。

下面的合成测试可以在服务器验证单卡训练、严格续训和双卡 HCCL。它只创建临时文件。默认在 CPU 执行，设置环境变量后使用 NPU。

```bash
K1369_TEST_DEVICE=npu python -m pytest -q k1369/tests/test_pipeline.py
```

以下是单机八卡、351M、两批数据的从头启动示例。初始每卡 micro-batch 取 1，真实节点的显存和吞吐需要实测。示例的全局每次更新为 80 个细胞；合作者 4 卡实验的对应值是 2560。调整每卡 batch 和累积次数时同时核对每次更新细胞数与每轮预算。

```bash
torchrun --standalone --nproc-per-node=8 k1369/scripts/train_ascend.py \
  --train_mode scratch --model_config k1369/configuration/350m_batch1_batch2.json \
  --batch_size 1 --global_accumulation_steps 80 \
  --out_path /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/k1369_runs/351m_scratch_01
```

`--global_accumulation_steps` 是**全部 rank 合计**的 micro-batch 数，应能整除 world size。每次 optimizer 更新的细胞数等于此值乘以每卡 batch_size。每次启动使用新的输出目录。做预算对照时显式设置 `--epoch_cells`。

`warm_start` 加载匹配的模型权重，从头建立优化器和计划。`resume` 校验词表、模型、分片清单、全局更新批量及采样设置。清单指纹包含绝对路径、文件大小和修改时间；跨机器复制后的数据可能无法通过原 checkpoint 的严格续训校验。未完成 epoch 的剩余预算会重新抽样。合作者的真实 checkpoint 需另行提供。
