SpeciesLLM 预训练代码包 v2
===========================

以任务 396057 的稳定 logits 训练版本为基线。使用原始 logits 的 FP32 masked
BCE、两路 Huber 和逐卡轻量诊断。v1 压缩包和原始工程仍可用于复现原版本。

从本目录运行 sha256sum -c SHA256SUMS 可校验打包文件。Slurm 默认使用原 Orix
Pixi 环境。可设置 SPECIESLLM_PIXI_MANIFEST，也可使用包内 pixi.toml 和 pixi.lock。
checkpoint、Parquet 数据和静态 embedding 从外部路径读取。

选择模型配置
------------
用 --model_config PATH，或提交时设置 SPECIESLLM_MODEL_CONFIG。
configuration/ 下提供以下四个 JSON。default_sources 控制本配置默认的数据批数，
model 保存模型参数。第三批使用任意联合配置，按其 metadata ID 范围调整类别数。

  100m_batch1.json          640 hidden / 12 层 / 10 heads / FFN 5120，第一批
  100m_batch1_batch2.json   640 hidden / 12 层 / 10 heads / FFN 5120，两批
  350m_batch1_batch2.json  1024 hidden / 16 层 / 16 heads / FFN 8192，两批
  500m_batch1_batch2.json  1280 hidden / 24 层 / 20 heads / FFN 5120，两批

训练模式
--------
1. resume 严格续训，同一模型和数据 manifest，沿用原更新计数和学习率计划。
   默认从任务 396057 的第 80,000 次更新恢复 350M 模型和 AdamW。

       sbatch launch_resume_4h100.slurm

2. scratch 随机初始化，从第 0 次更新建立计划。可用脚本名后面的参数
   覆盖启动器中的训练预算、学习率或数据路径。

       SPECIESLLM_TRAIN_MODE=scratch \
       SPECIESLLM_MODEL_CONFIG=configuration/100m_batch1.json \
       sbatch launch_resume_4h100.slurm --batch1_path /data/batch1

3. warm_start 加载兼容 checkpoint 的模型权重，重新初始化 AdamW，从第 0 次
   更新建立新计划。用旧模型接入第三批、修改采样比例或改变总预算时，可用
   此模式。要求相同的基因词表顺序、CLS ID 和模型参数形状。

       SPECIESLLM_TRAIN_MODE=warm_start \
       SPECIESLLM_CHECKPOINT=/weights/checkpoint_latest.pth \
       sbatch launch_resume_4h100.slurm \
         --batch1_path /data/batch1 --batch2_path /data/batch2 \
         --batch3_path /data/batch3 --batch2_fraction 0.60 --batch3_fraction 0.20 \
         --embedding_dir /data/macrogene_embeddings

直接运行 Python 训练入口时，使用 --train_mode scratch|resume|warm_start
和 --checkpoint_path PATH。scratch 不传 checkpoint。Slurm 每个任务使用新输出
目录，也可设置 SPECIESLLM_OUT_DIR 指向新目录。500M 配置的显存与吞吐需实测；
如调小 batch_size，需相应核对实际 global batch 和训练预算。

数据及 embedding 路径
---------------------
--batch1_path、--batch2_path、--batch3_path 分别指向各批数据根目录，
均使用 root/species/*.parquet 布局。可单独用 --validation_data_path 指定
第一批单源模式的验证集。默认第一批沿用约 0.5% 尾部分片验证规则。
第二、第三批每个物种默认留最后 3 个分片，分片不足时应用 keep_one_train。
每个启用来源必须有训练分片，且验证集至少能组成一个全局 micro-batch。

第二、第三批比例均针对每个 epoch 的总采样量。上述例子的来源比例为
20% / 60% / 20%。未启用第三批时保持原两批策略。各批内部按 N^alpha 分配
物种配额，各 rank/worker 再跨批与物种混合。验证结果按来源比例加权。
scratch 和 warm_start 的 Slurm 默认使用 --epoch_cells 0。
--epoch_cells 0 在联合模式下仍以第二批训练细胞数除以第二批占比估算总预算。
为使不同数据版本的对照预算一致，请显式传入 --epoch_cells。

--embedding_dir 设置三份 embedding 的共同根目录。若分开存放，可使用
--esm_embedding_dir、--desc_embedding_dir、--dna_embedding_dir 分别覆盖。
文件名对应如下，行顺序须匹配 1369 基因词表。

  3rdrun_K1500_Eukaryota_macrogene_esm2_weighted_average.npy
  3rdrun_K1500_Eukaryota_macrogene_gene_desc_weighted_average.npy
  3rdrun_K1500_Eukaryota_macrogene_DNAseq_weighted_average.npy

如果第三批含新 metadata 类别，先统一三批类别 ID，再按配置设置容量。
新增类别导致 embedding 形状变化时，旧模型权重需要单独迁移；当前权重加载
按形状严格检查。configuration/job_396057_sampling_config.json 是历史记录，
其中第 35,000 次更新是该任务的启动点，当前默认 checkpoint 位于第 80,000 次。
