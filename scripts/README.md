# Local Command Scripts

These scripts resolve the project root from their own location before running
anything, so they can be called from any current working directory while still
using the project-root Python imports, `.env`, data paths, and embedding paths.

## Generate test data

```bash
bash scripts/generate_test_data.sh
```

This runs two steps:

1. Merge a small per-species test dataset into
   `Stage2_SpeciesLLMData/all_shuffled_test`.
2. Flatten and globally shuffle that dataset into
   `Stage2_SpeciesLLMData/all_flatten_data_test`, which is the directory read
   by `TRAIN_DATASET=test`.

Useful overrides:

```bash
DRY_RUN=1 bash scripts/generate_test_data.sh
SKIP_EXISTING=1 bash scripts/generate_test_data.sh
OUTPUT_DIR=/path/to/output bash scripts/generate_test_data.sh
FLATTEN_OUTPUT_DIR=/path/to/flat/output bash scripts/generate_test_data.sh
SKIP_FLATTEN=1 bash scripts/generate_test_data.sh
```

The intermediate merged dataset defaults to:

```text
Stage2_SpeciesLLMData/all_shuffled_test
```

The final training-ready dataset defaults to:

```text
Stage2_SpeciesLLMData/all_flatten_data_test
```

## Pretraining checks

Reusable pretraining checks live in:

```text
scripts/pretrain_config.py
scripts/pretrain_checks.py
```

They provide strict model JSON loading, seq_len derivation, source parquet
preflight, flattened parquet validation, embedding checks, distributed file-plan
generation, output path resolution, and post-training log/artifact checks. The
test pipeline defaults to the 500M model JSON:

```text
Stage2_macrogene_embeddings/args_2nd_run.json
```

Missing model structure fields, label switches, label counts, or inconsistent
`vocab_size` / `max_position_embeddings` fail fast.

## Three-node pretraining smoke test

```bash
bash scripts/pretrain_pipeline.sh all
```

This server-oriented script uses
`merge_macrogene_rounds.py --test-mode`, flattens the small merged
sample, validates parquet schema / label ranges / macrogene embedding shapes,
creates a 24-rank file distribution plan, and generates three-node training
commands under `Stage2_SpeciesLLMData/pretrain_500m_test_commands`.

Model structure and label parameters are read strictly from
`Stage2_macrogene_embeddings/args_2nd_run.json`; if a required JSON field is
missing, the script exits instead of falling back to shell defaults.

The shell wrapper only orchestrates commands. It calls
`scripts/pretrain_checks.py` for reusable checks and passes
`--config_json` to the training entry instead of expanding model structure or
label parameters in shell.

Typical server invocation:

```bash
STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData \
WORKDIR=/path/to/SpeciesLLM \
HOSTS=host0,host1,host2 MASTER_ADDR=host0 \
bash scripts/pretrain_pipeline.sh all
```

For end-to-end three-node orchestration, including remote sync, path checks,
dry-run, and optional launch, use:

```bash
bash scripts/pretrain_3node.sh
```

After the distributed job finishes:

```bash
bash scripts/pretrain_pipeline.sh check-training
```

## Train

Multi-node launcher:

```bash
bash scripts/launch_multinode_torchrun.sh
```

The launcher reads model structure and label configuration from
`MODEL_CONFIG_JSON`, defaulting to `Stage2_macrogene_embeddings/args_2nd_run.json`
under `EMB_PATH`. It passes that file to
`train_MNodes_torchrun_mfu_preindexparquet.py --config_json`; shell defaults are
kept only for cluster paths and training hyperparameters, not model structure or
label settings.

Run multi-node training on the test dataset:

```bash
TRAIN_DATASET=test \
DATA_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData \
EMB_ROOT=/data/disk1/SpeciesLLM \
bash scripts/launch_multinode_torchrun.sh
```

For server runs, copy the root env template and edit only server/cluster
settings:

```bash
cp .env.example .env
```

Do not put data paths or training hyperparameters in `.env`. Pass them in the
launch command instead. `DATA_ROOT` and `EMB_ROOT` are independent; for example,
test data can live under `/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData`
while embeddings live under `/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings`.

Single-node command:

```bash
bash scripts/launch_singlenode_torchrun.sh
```

ModelArts command:

```bash
bash scripts/train_modelarts.sh
```

ModelArts environment setup:

```bash
bash scripts/create_env_modelarts.sh
```

The training entrypoint remains in the project root:

```text
train_MNodes_torchrun_mfu_preindexparquet.py
```

## Compress training output text logs

Use the stdlib-only archive helper when `training_output` text logs are too
large to copy directly from the server. It only includes files whose basenames
match `log*txt`, `loss_to_log*txt`, or `metrics*jsonl`; checkpoints, figures,
and other artifacts are skipped. Run it after the training process has stopped
writing these files.

Use `split-pack` when every transferred file must stay below 80 KB. It first
tries a structure-aware lossless encoding for `metrics*.jsonl`,
`loss_to_log*.txt`, and `log*.txt`, verifies that the encoded form can recreate
the original bytes exactly, then xz-compresses and splits the encoded stream
into many small `.xzpart` files plus split manifest files. If a file does not
match the expected stable format, it automatically falls back to raw-byte
compression for that file.

With the normal 8-rank layout the 24 `log`/`loss_to_log`/`metrics` files can be
compressed by up to 24 worker processes.

On the server:

```bash
python scripts/archive_training_output_text.py split-pack \
  /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_xxx \
  --output-dir /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_xxx_text_split \
  --chunk-size 80KB \
  --jobs 24
```

Add `--raw-bytes` if you want to disable the structure-aware step and only do
plain byte compression/splitting.

Copy the generated split directory to the workstation, then merge/decompress:

```bash
python scripts/archive_training_output_text.py split-unpack \
  training_output_xxx_text_split \
  --output-dir .
```

The split output is lossless: merge/decompress verifies every restored file
against the manifest SHA-256 checksum. To inspect or verify without extracting:

```bash
python scripts/archive_training_output_text.py split-list training_output_xxx_text_split
python scripts/archive_training_output_text.py split-verify training_output_xxx_text_split
```

If you prefer one archive file instead of many small files, use `pack`:

```bash
python scripts/archive_training_output_text.py pack \
  /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_xxx \
  --output /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_xxx_text.tar \
  --jobs 24
```

For the old single-stream `.tar.xz` format, which can be slightly smaller but is
serial and slower on many large files, add `--single-stream`:

```bash
python scripts/archive_training_output_text.py pack \
  /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_xxx \
  --output /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_xxx_text.tar.xz \
  --single-stream
```
