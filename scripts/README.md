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

## Stage 2 training checks

Reusable Stage 2 checks live in:

```text
scripts/stage2_training_config.py
scripts/stage2_training_checks.py
```

They provide strict model JSON loading, seq_len derivation, source parquet
preflight, flattened parquet validation, embedding checks, distributed file-plan
generation, output path resolution, and post-training log/artifact checks. Both
the smoke-test wrapper and the multi-node launcher use the same fixed 500M model
JSON:

```text
Stage2_macrogene_embeddings/args_2nd_run.json
```

Missing model structure fields, label switches, label counts, or inconsistent
`vocab_size` / `max_position_embeddings` fail fast.

## Stage 2 500M three-node smoke test

```bash
bash scripts/test_stage2_500m_multinode.sh all
```

This server-oriented script uses
`merge_macrogene_rounds_parallel.py --test-mode`, flattens the small merged
sample, validates parquet schema / label ranges / macrogene embedding shapes,
writes a 24-rank file distribution plan, and generates 500M three-node training
commands under `Stage2_SpeciesLLMData/stage2_500m_test_commands`.

Model structure and label parameters are read strictly from
`Stage2_macrogene_embeddings/args_2nd_run.json`; if a required JSON field is
missing, the script exits instead of falling back to shell defaults.

The shell wrapper only orchestrates commands. It calls
`scripts/stage2_training_checks.py` for reusable checks and passes
`--config_json` to the training entry instead of expanding model structure or
label parameters in shell.

Typical server invocation:

```bash
STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData \
WORKDIR=/path/to/SpeciesLLM \
HOSTS=host0,host1,host2 MASTER_ADDR=host0 \
bash scripts/test_stage2_500m_multinode.sh all
```

After the distributed job finishes:

```bash
bash scripts/test_stage2_500m_multinode.sh check-training
```

## Train

Multi-node launcher:

```bash
bash scripts/train_multinode.sh
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
bash scripts/train_multinode.sh
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
bash scripts/train_singlenode.sh
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
