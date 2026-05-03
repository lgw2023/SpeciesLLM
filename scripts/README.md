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

## Train

Multi-node launcher:

```bash
bash scripts/train_multinode.sh
```

Run multi-node training on the test dataset:

```bash
TRAIN_DATASET=test \
DATA_ROOT=/data1/.../Stage2_SpeciesLLMData \
EMB_ROOT=/data2/... \
bash scripts/train_multinode.sh
```

For server runs, copy the root env template and edit only server/cluster
settings:

```bash
cp .env.example .env
```

Do not put data paths or training hyperparameters in `.env`. Pass them in the
launch command instead. `DATA_ROOT` and `EMB_ROOT` are independent; for example,
test data can live under `/data1/.../Stage2_SpeciesLLMData` while embeddings
live under `/data2/.../Stage2_macrogene_embeddings`.

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
