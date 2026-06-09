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

The launcher now also exposes `ADAPTIVE_GRAD_CLIP`, `LR_DECAY_EPOCHS`, and the
full adaptive grad-clip family (`GRAD_CLIP_RATIO`, `GRAD_SKIP_RATIO`,
`GRAD_SKIP_MAX`, `GRAD_CLIP_MIN/MAX`, `GRAD_CLIP_WARMUP_STEPS`,
`GRAD_CLIP_MAX_CONSECUTIVE_SKIPS`, `GRAD_CLIP_EMA_RUNAWAY_FACTOR`,
`GRAD_CLIP_HARD_RAW_NORM_LIMIT`) as env passthroughs. Their defaults match the
training-script argparse defaults, so leaving them unset keeps prior behavior.

### First-run inputs with the current training recipe

`scripts/launch_singlenode_1st_inputs_current_recipe.sh` runs one node x 8 NPUs
with the **early/first-version inputs** but the **current repo code and current
training recipe**, to regression-check whether code changes altered model
behavior. It drives the existing `scripts/launch_multinode_torchrun.sh` in
single-node mode (`HOSTS=127.0.0.1`, `NNODES=1`, `SYNC_SELF=0`, so node_rank 0
runs locally with no ssh/scp) and only sets env/parameters — no launcher or
training-entry code is modified.

Pinned to the first version (the "necessary file configs"):

- model config `Stage2_macrogene_embeddings/args_1st_run_<size>.json` —
  `seq_len=862`, `use_batch_labels=true`, label dims `12028/11/154/28/143/5/3`;
- 1st-run macrogene embeddings (862 rows, `Stage1_macrogene_embeddings`);
- early shuffled training data in the old 862-macrogene layout. By default the
  wrapper reads `/data/disk1/SpeciesLLM/all_shuffled_data`, whose expected files
  are `shuffled_part_*.parquet`.

Everything else follows the current recipe, matching
the second resumed run in
`training_output_100m_data_1_2_3_E2_huber_fp32_from_scratch_20260529_160058`:
`gep_loss=huber` (`huber_delta=5.0`), `amp_dtype=float32`, `lr 1e-6 -> 1e-7`,
`warmup_iters=2000`, `warmup_ratio=0.10`, `beta2=0.98`, `epoch=5`,
`adaptive_grad_clip=true` with norm-skip + aborts off (`grad_skip_ratio=0`,
`grad_skip_max=0`, `max_consecutive_skips=0`, `ema_runaway_factor=0`) and the
`1e8` hard raw-norm fuse kept, `compile=false`, `batch 512 x grad_accum 1`.
Resume/init-only values from that run (`INIT_MODEL_PATH`, `RESUME_*`,
`APPEND_OUTPUT_LOGS=true`) are intentionally not defaults here because this
wrapper starts a fresh early-data run.

`MODEL_SIZE` selects the model structure: `100m` (default, matches the cited
reference) or `500m` (matches the early first-version config). The training
script reads the `2nd_run_macrogene_features_sum_*.npy` files selected by
`GENE_EMBEDDING_MODALITIES` (default `esm2,gene_desc,dnaseq`), so the 1st-run
arrays are exposed under those names: by default the wrapper symlinks
`Stage1_macrogene_embeddings/1st_run_*.npy` into a sibling `*_as_2nd_run` dir.
Override `SRC_EMB_PATH` (dir with `1st_run_*.npy`) or `EMB_PATH` (dir already
using `2nd_run_*` names) as needed.

`DATA_PATH` defaults to `/data/disk1/SpeciesLLM/all_shuffled_data` and may be
overridden with any flattened, training-ready early parquet dir whose samples
carry 862 macrogene features (collate raises if features != `seq_len`).

```bash
bash scripts/launch_singlenode_1st_inputs_current_recipe.sh
# 500M structure instead of 100M:
DATA_PATH=... MODEL_SIZE=500m bash scripts/launch_singlenode_1st_inputs_current_recipe.sh
```

The grad-control family is set explicitly because the multi-node launcher's own
defaults differ from this recipe (e.g. `GRAD_CLIP_MAX=0.5`, `GRAD_SKIP_RATIO=100`,
`GRAD_CLIP_HARD_RAW_NORM_LIMIT=1e11`, `COMPILE=true`).

Training runs **detached** (the launcher backgrounds node_rank 0 via `nohup`);
follow it at `torchrun_logs/node_rank0.log`. Preview the assembled command
without launching by adding `DRY_RUN=1`. The cited run used 3 nodes (global batch
`3*8*512*1=12288`); this wrapper keeps `GRADIENT_ACCUMULATION_STEPS=1` to match
the recorded argument value. Set it to `3` only if matching effective global
batch is more important than matching the reference argument list. If `batch 512`
OOMs at `seq_len=862`, lower `BATCH_SIZE` and raise
`GRADIENT_ACCUMULATION_STEPS` proportionally.

## DDP gradient synchronization probe

Use this before real training when you need to verify that Ascend
NPU/CANN/torch_npu DDP really synchronizes gradients during `loss.backward()`.
The probe uses a one-parameter model and rank-specific targets, so expected
gradients are deterministic and do not require real datasets.

Single-node 8-card test:

```bash
bash scripts/run_ddp_grad_sync_probe.sh single
```

`single` mode does not load `.env` by default, so three-node variables such as
`MASTER_ADDR` or `WORKDIR` do not accidentally override the local 8-card probe.
Pass `LOAD_ENV_FILE=1` only if you intentionally want to read `.env`.

Expected key line for 8 ranks:

```text
PASS ddp_backward: observed=[-9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0] expected=global average -9.0
```

Three-node 24-card test, run from the master node:

```bash
HOSTS=host0,host1,host2 \
MASTER_ADDR=host0 \
WORKDIR=/data/disk1/SpeciesLLM \
bash scripts/run_ddp_grad_sync_probe.sh multinode
```

Use the same host order and SSH settings as `scripts/launch_multinode_torchrun.sh`.
The local node runs in the foreground; remote node logs are written under:

```text
ddp_grad_probe_logs/node_rank1.log
ddp_grad_probe_logs/node_rank2.log
```

Expected key line for 24 ranks:

```text
PASS ddp_backward: observed=[-25.0, ...] expected=global average -25.0
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

## Publish training outputs to GitHub

Local `training_output/` directories stay in `.gitignore`, including checkpoints
and other large artifacts. To share logs/metrics (and optional plot snapshots)
with collaborators via GitHub, build git-friendly mirror directories:

```bash
bash scripts/publish_training_outputs_to_git.sh --overwrite --git-add
git commit -m "Share training output logs/metrics mirrors for collaborators."
git push
```

This writes `training_output_<run>_text_split/` next to each local run. Those
mirror directories are explicitly un-ignored in `.gitignore`. Weights such as
`*.pt` remain excluded. Collaborators restore the original text files with:

```bash
python scripts/archive_training_output_text.py split-unpack \
  training_output_<run>_text_split \
  --output-dir .
```
