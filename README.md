# SpeciesLLM

A multi-species single-cell foundation model that learns universal cell embeddings across diverse species through a novel **Macrogene** mechanism.

## Overview

SpeciesLLM addresses a fundamental challenge in single-cell biology: different species have different gene vocabularies, making it impossible to directly train a shared foundation model. Instead of forcing a unified gene space, SpeciesLLM introduces a **Macrogene Construction Layer** that maps species-specific gene spaces into a shared biological/functional space before training the cell embedding backbone.

```
Protein Sequence (ESM2) ─┐
Gene Description Text ───┤──→ Cross-Species Gene Representation ──→ Soft Gene-to-Macrogene ──→ Cell × Macrogene Matrix ──→ BERT Backbone ──→ Cell Embedding
DNA Sequence ─────────────┘
```

This design decouples two responsibilities:
- **Macrogene Layer**: aligns gene spaces across species
- **Embedding Backbone**: learns cell state representations

### Macrogene count / `seq_len` (single source of truth)

The number of macrogenes **N** is the model's sequence length and the length of
every cell's expression vector `X` (the GEP regression target). It is not a free
constant — it is fixed by the macrogene feature matrices and must agree everywhere:

- **Source of truth**: rows of `Stage2_macrogene_embeddings/2nd_run_macrogene_features_sum_*.npy`.
- Encoded in the model config as `vocab_size = N + 1` (the `+1` is the CLS token).
  Training derives `seq_len = vocab_size − 1` (`scripts/pretrain_config.py`);
  preflight asserts the `.npy` row count `== seq_len`; the collator raises if any
  cell's `X` length `!= seq_len`.
- **Current (2nd) run: N = 640.** The earlier **1st run used N = 862**
  (`Stage1_macrogene_embeddings`, `args_1st_run_*.json`), so scripts pinned to
  1st-run inputs (e.g. `scripts/launch_singlenode_1st_inputs_current_recipe.sh`)
  legitimately say 862. The two are not interchangeable.
- **Do not hardcode the number.** Pass `--config_json` (preferred, used by
  `scripts/launch_*_torchrun.sh`) or read it from the `.npy` shape.

## Model Architecture

The model (`BERTForPreTraining`) is built on BERT with several key extensions:

### Input Encoding
- **GeneEncoder**: maps macrogene IDs to embeddings
- **ContinuousValueEncoder**: encodes continuous expression values via `Linear → ReLU → Linear → LayerNorm`
- **EnhancedFusion**: fuses selected gene pre-trained embeddings with weighted concatenation + linear projection; default is ESM2 protein, gene description text, and DNA sequence

### Cell Metadata Injection
Seven cell-level metadata types are independently encoded and summed, then concatenated to each gene position's hidden state:

| Metadata | Default | Categories |
|----------|---------|------------|
| Species | ✓ | 11–29 |
| Tissue | ✓ | 154–336 |
| Assay | ✓ | 28–30 |
| Disease | ✓ | 143–1921 |
| Age | ✓ | 5 |
| Sex | ✓ | 3 |
| Batch | ✗ | 12k–62k |

### Pre-training Objectives
- **GEP** (Gene Expression Prediction): masked MSE loss on masked macrogene expression values
- **NZLP** (Non-Zero Log-Likelihood Probability): Bernoulli distribution for explicit zero-expression modeling
- **MVC** (Masked Value Prediction for Cell): predicts masked expression from cell-level embedding
- **Adversarial Batch Correction** (optional): gradient reversal layer for domain adaptation

### Default Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 1280 |
| num_hidden_layers | 24 |
| num_attention_heads | 20 |
| intermediate_size | 5120 |
| cell_hidden_size | 128 |
| seq_len (macrogene count) | 640 |

## Data Pipeline

```
scBaseCount raw h5ad data
    ↓ Step 0: Filter & standardize (0_scbasecount_filter.py)
    ↓ Step 2: Split obs/var/X (02_scbasecount.py)
    ↓ Step 3: Normalize + INT encode + Parquet output (03_scbasecount.py)
    ↓ Stage 2: Gene-to-Macrogene conversion (batch_mapping.py)
    ↓ Shuffle & chunk (shuffle_species.py)
    ↓ Training data (Parquet)
```

### Gene-to-Macrogene Conversion

The core preprocessing step that transforms `cell × gene` matrices into `cell × macrogene` matrices:
- Computes cross-species gene similarity from three gene embeddings
- Uses soft assignment with top-r sparse softmax (each gene can belong to multiple functional modules)
- Aggregates expression: `X_macro = X_gene × W`

## Training

### Requirements

Ascend NPU environment:
```
pytorch_2.1.0-cann_8.0.rc2-py_3.9-euler_2.10.7-aarch64-snt9b
```

Key dependencies: `torch_npu`, `apex`, `transformers==4.40.2`, `dask==2023.12.1`, `pyarrow==16.0.0`, `scanpy==1.10.1`, `anndata==0.10.9`

See [pyproject.toml](pyproject.toml) for full dependency list.

### Quick Start

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  train_MNodes_torchrun_mfu_preindexparquet.py \
  --data_path=./Stage2_SpeciesLLMData/all_flatten_data \
  --emb_path=./Stage2_macrogene_embeddings \
  --seq_len=640 \
  --batch_size=16 \
  --epoch=10 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-5 \
  --min_lr=1e-6 \
  --warmup_ratio=0.05 \
  --hidden_size=1280 \
  --num_hidden_layers=24 \
  --num_attention_heads=20 \
  --intermediate_size=5120 \
  --use_species_labels=true \
  --use_tissue_labels=true \
  --use_seqmethod_labels=true \
  --use_disease_labels=true \
  --use_age_labels=true \
  --use_sex_labels=true \
  --explicit_zero_prob=true
```

### Key Training Features
- **Distributed training**: DDP with HCCL backend on Ascend NPUs
- **Mixed precision**: bfloat16 / float16 with `torch_npu.npu.amp`
- **File-level distributed sampling**: `DistributedFileSampler` shards parquet files across ranks
- **Cosine decay LR schedule** with linear warmup
- **Gradient checkpointing** for memory efficiency
- **MFU estimation** built into training loop

## Embedding Extraction

```bash
python get_embedding.py \
  --model_path <checkpoint.pt> \
  --data_path <input_data> \
  --emb_path <gene_embeddings>
```

Extracts cell embeddings (CLS token) and gene embeddings from pre-trained checkpoints, output as AnnData with `adata.obsm["X_SpeciesLLM"]`.

## Project Structure

```
SpeciesLLM/
├── train_MNodes_torchrun_mfu_preindexparquet.py   # Main training script
├── scripts/
│   ├── generate_test_data.sh                       # Test-data generation command
│   ├── pretrain_pipeline.sh                        # Reusable pretraining data/check/command pipeline
│   ├── pretrain_3node.sh                           # Three-node pretraining orchestrator
│   ├── launch_multinode_torchrun.sh                # Multi-node torchrun launcher
│   ├── launch_singlenode_torchrun.sh               # Single-node torchrun launcher
│   ├── train_modelarts.sh                          # ModelArts training wrapper
│   └── create_env_modelarts.sh                     # ModelArts environment setup
├── get_embedding.py                                # Embedding extraction
├── shuffle_species.py                              # Legacy data flattening/shuffling
├── shuffle_flatten_macrogene.py                    # Flatten species folders into training chunks
├── merge_macrogene_rounds.py                       # Multi-round macrogene data merging
├── nanoBERT/
│   ├── model/
│   │   ├── nanoBERTmodel_cellmeta2_plusEncode_adbc.py  # Core model
│   │   └── util.py                                     # GradReverse, DSBN utilities
│   └── utils/
│       ├── MultispeciesDataset.py     # Dataset classes (Parquet/Lazy/Preindexed)
│       ├── data_collator_3GeneEmb.py  # Tri-modal gene embedding collator
│       ├── gene_tokenizer.py          # GeneVocab
│       ├── losses.py                  # Loss functions
│       └── torch_vocab.py             # lightweight local Vocab implementation
├── Stage2_Convert_gene_to_macrogene/
│   ├── batch_mapping.py              # Gene-to-Macrogene conversion
│   └── check_var_weight_gene_sets.py # Macrogene quality check
├── scbasecount_demo/
│   └── code/
│       ├── 0_scbasecount_filter.py   # Raw data filtering
│       ├── 02_scbasecount.py         # Data splitting
│       └── 03_scbasecount.py         # Normalization & Parquet output
└── presentation/
    └── speciesllm_macrogene_revised_proposal.md  # Macrogene design document
```

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.
