#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

cd "$PROJECT_ROOT"

exec torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  train_MNodes_torchrun_mfu_preindexparquet.py \
  --data_path=./Stage2_SpeciesLLMData/all_flatten_data \
  --num_of_used_data=0 \
  --emb_path=./Stage2_macrogene_embeddings \
  --seq_len=640 \
  --out_path=training_output \
  --batch_size=32 \
  --epoch=10 \
  --gradient_accumulation_steps=8 \
  --learning_rate=0.00001 \
  --min_lr=0.000001 \
  --decay_lr=true \
  --warmup_iters=2000 \
  --warmup_ratio=0.05 \
  --weight_decay=0.1 \
  --save_data_interval=5120000 \
  --beta1=0.9 \
  --beta2=0.95 \
  --grad_clip=1.0 \
  --compile=true \
  --backend=hccl \
  --device=npu \
  --device_type=npu \
  --hidden_size=1280 \
  --num_hidden_layers=24 \
  --num_attention_heads=20 \
  --intermediate_size=5120 \
  --hidden_act=gelu \
  --hidden_dropout_prob=0.1 \
  --cell_hidden_size=128 \
  --attention_probs_dropout_prob=0.1 \
  --type_vocab_size=2 \
  --initializer_range=0.02 \
  --layer_norm_eps=1e-12 \
  --_attn_implementation=eager \
  --use_batch_labels=false \
  --num_batch_labels=62223 \
  --use_species_labels=true \
  --num_species_labels=29 \
  --use_tissue_labels=true \
  --num_tissue_labels=336 \
  --use_seqmethod_labels=true \
  --num_seqmethod_labels=30 \
  --use_disease_labels=true \
  --num_disease_labels=1921 \
  --use_age_labels=true \
  --num_age_labels=5 \
  --use_sex_labels=true \
  --num_sex_labels=3 \
  --cell_emb_style=cls \
  --chunk_size_feed_forward=0 \
  --explicit_zero_prob=true
