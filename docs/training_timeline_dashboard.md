# SpeciesLLM Training Timeline Dashboard

这个本地应用用于把现有 `training_output*` 训练目录整理成时间线、单次训练详情、对比视图和阶段报告。它只做离线回顾，不运行训练、不加载 checkpoint、不扫描大数据集，也不连接远程服务器。

## 安装依赖

后端依赖：

```bash
python -m pip install -r requirements-training-timeline.txt
```

前端依赖：

```bash
cd training_timeline_ui
npm install
```

## 重建索引

默认扫描当前仓库根目录：

```bash
python -m training_timeline.cli rebuild \
  --db .training_timeline/timeline.sqlite \
  --source /Volumes/SSD1/SpeciesLLM
```

可以追加额外训练输出目录：

```bash
python -m training_timeline.cli rebuild \
  --db .training_timeline/timeline.sqlite \
  --source /Volumes/SSD1/SpeciesLLM \
  --source /path/to/copied/training/outputs
```

索引只读取配置的 source root 内文件。第一版候选目录为 `training_output*`，会跳过 `*_text_split` 镜像目录。

## 启动应用

后端脚本会先重建一次本地索引，再启动服务，避免首次打开页面时误显示为空：

```bash
scripts/run_training_timeline_backend.sh
```

前端：

```bash
scripts/run_training_timeline_frontend.sh
```

如果后端没有使用默认 `8766` 端口，可以指定代理地址：

```bash
cd training_timeline_ui
TRAINING_TIMELINE_BACKEND_URL=http://127.0.0.1:8766 npm run dev
```

脚本默认使用 `8766`，也可以统一指定端口：

```bash
TRAINING_TIMELINE_BACKEND_PORT=8770 scripts/run_training_timeline_backend.sh
TRAINING_TIMELINE_BACKEND_PORT=8770 scripts/run_training_timeline_frontend.sh
```

浏览器打开：

```text
http://127.0.0.1:5173
```

## 页面

- Timeline：按推断开始时间从上到下展示训练关系图，节点代表实验，边优先使用索引出的关系证据，标注相对父实验改变的训练数据、模型、配方和训练参数。
- Sources：展示 source root、run 数量，并触发重建索引。
- Run Detail：展示摘要、曲线、自动诊断、人工深度复核和 artifact。
- Compare：对比多个 run 的配置差异、摘要指标和诊断事件。
- Report：按训练阶段组织 run，形成训练过程叙事。

## 诊断边界

自动诊断只做初筛，标签包括 `converged`、`bad_plateau`、`clip_storm`、`skip_loop`、`primary_head_failure`、`lr_floor_freeze` 和 `resume_boundary`。这些结论在 UI 中标记为 `Preliminary`。

复杂训练任务需要逐次阅读日志、配置、曲线和 summary。Run Detail 的 deep review 区域用于补充人工阅读结论，必要时可覆盖或解释自动诊断。

关系边和 `auto-context` 笔记是自动推断的第二层证据，不覆盖原始日志事实。索引会组合目录名、`run_record.json`、`summary/metrics`、`work_record/`、`scripts/`、相关 Git 记录、本地记忆记录、相关对话记录和 Run Detail 里的人工分析笔记；后续人工补出的结论应写入 deep review / curated note，并优先于自动笔记展示。

## 本地安全约束

- 不修改 `training_output*` 原始目录。
- 不运行训练、评估、checkpoint 加载或大数据集处理。
- 不假设本机存在服务器数据、parquet shards、checkpoints 或真实 NPU 环境。
- 真正训练环境仍按 Ascend NPU、CANN `8.2.1.RC1` 和 `torch_npu` 约束处理。
