# 500M NPU Memory Diagnosis Record

This file is the local analysis template for the single-node memory diagnosis.
The runnable script also writes a timestamped copy under:

`training_output/memory_diagnosis_500m_*/analysis_record.md`

## Experiment Setup

- Training entry: `train_MNodes_torchrun_mfu_preindexparquet.py`
- Launcher: `scripts/launch_singlenode_torchrun.sh`
- Experiment runner: `scripts/run_500m_memory_diagnosis_singlenode.sh`
- Data subset: first 10 sorted parquet files from `all_flatten_data_test_500m`
- Default per-rank batch size: 16
- Default local steps per rank: 3
- Final checkpoint saving: disabled for diagnosis runs

## Memory Tags

Use `memory_events.log` in each case directory and compare these tags:

| Tag | Meaning |
|---|---|
| `after_model_to` | Full model parameters on NPU |
| `after_static_gene_inputs` | Static macrogene embedding buffers |
| `after_optimizer_init` | AdamW object before first-step state allocation |
| `after_ddp_wrap` | DDP buckets/communication buffers |
| `after_to_device` | Batch tensors on NPU |
| `after_forward` | Forward activations and attention buffers |
| `after_loss` | Loss-side tensors |
| `after_backward` | Gradients after backward |
| `after_optimizer_step` | AdamW m/v states materialized |

## Case Matrix

| Case | Expected signal |
|---|---|
| `00_baseline_current` | Reproduce current memory shape |
| `01_attention_sdpa` | Large drop after forward/backward means eager attention is the main factor |
| `02_gradient_checkpointing` | Drop after forward/backward means saved activations are the main factor |
| `03_no_mvc_forward` | Drop after forward/loss means MVC branch activation/loss is relevant |
| `04_no_mvc_module` | Drop at setup means MVC parameters/optimizer states are relevant |
| `05_no_zero_prob` | Drop at setup/loss means explicit zero-prob heads/losses are relevant |
| `06_static_gene_amp_dtype` | Drop after static inputs means static embedding dtype matters |
| `07_sdpa_checkpoint` | Best-case combined activation-memory probe |
| `08_batch8_baseline` | Approximate activation scaling with batch size |
| `09_ffn_chunk128` | Drop after forward/backward means FFN intermediate activations matter |
| `10_amp_bfloat16` | Optional bf16 autocast comparison when enabled |

## Summary To Fill After Run

| Case | after_ddp_wrap GiB | after_forward GiB | after_backward GiB | after_optimizer_step GiB | Notes |
|---|---:|---:|---:|---:|---|
| `00_baseline_current` | | | | | |
| `01_attention_sdpa` | | | | | |
| `02_gradient_checkpointing` | | | | | |
| `03_no_mvc_forward` | | | | | |
| `04_no_mvc_module` | | | | | |
| `05_no_zero_prob` | | | | | |
| `06_static_gene_amp_dtype` | | | | | |
| `07_sdpa_checkpoint` | | | | | |
| `08_batch8_baseline` | | | | | |
| `09_ffn_chunk128` | | | | | |
| `10_amp_bfloat16` | | | | | |
