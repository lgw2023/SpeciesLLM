# 500M NPU Memory Diagnosis Record

This file records the local analysis for the single-node 500M NPU memory
diagnosis. The runnable script also writes a timestamped copy under:

`training_output/memory_diagnosis_500m_*/analysis_record.md`

Analysis date: 2026-05-07

Local copied result root:

`/Users/liguowei/Downloads/memory_diagnosis_500m`

Result runs inspected:

- `memory_diagnosis_500m_20260506_170759`: cases 00-09; case 09 failed.
- `memory_diagnosis_500m_20260506_175946`: fixed rerun of case 09.
- `memory_diagnosis_500m_20260506_180631`: case 10.

Files inspected for each run:

- `analysis_record.md`
- `case_status.tsv`
- `runner.log`
- `*/memory_events.log`
- `*/console.log`
- `*/npu_smi_after.txt`

## Experiment Setup

- Training entry: `train_MNodes_torchrun_mfu_preindexparquet.py`
- Launcher: `scripts/launch_singlenode_torchrun.sh`
- Experiment runner: `scripts/run_500m_memory_diagnosis_singlenode.sh`
- Data subset: first 16 sorted parquet files from `all_flatten_data_test_500m`
- Default per-rank batch size: 16
- Default local steps per rank: 100
- Final checkpoint saving: disabled for diagnosis runs
- NPU count: 8 Ascend 910B1 devices
- Model shape: hidden size 1280, 24 layers, 20 heads, FFN intermediate 5120
- Runtime sequence length: 641 tokens, because the training sequence includes
  `<cls>` on top of `seq_len=640`

Important measurement note:

- `allocated_gib` is closer to live tensor memory.
- `reserved_gib` is closer to the torch-npu allocator footprint.
- The observed `npu-smi` training HBM can be higher than torch reserved. In these
  runs, `npu_smi_after.txt` still showed about 3385 MB HBM per card after the
  process exited, so about 3.3 GiB of the earlier 44.6 GB/card observation is
  consistent with NPU runtime, HCCL, or driver-side overhead outside torch tensor
  allocation.

## Memory Tags

Use `memory_events.log` in each case directory and compare these tags:

| Tag | Meaning |
|---|---|
| `after_model_to` | Full model parameters on NPU |
| `after_static_gene_inputs` | Static macrogene embedding buffers |
| `after_optimizer_init` | AdamW object before first-step state allocation |
| `after_ddp_wrap` | DDP buckets/communication buffers |
| `after_to_device` | Batch tensors on NPU |
| `after_forward` | Forward activations and attention buffers; this is the main peak in the baseline |
| `after_loss` | Loss-side tensors |
| `after_backward` | Gradients after backward |
| `after_optimizer_step` | AdamW m/v states materialized after the first step |

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

## Analysis Method

The comparison below uses the latest successful run for each case. Case 09 from
`20260506_170759` is excluded because it failed before forward; the fixed case 09
from `20260506_175946` is used instead.

For every case, `memory_events.log` was parsed for all ranks. The critical tags
were checked across ranks and were effectively identical, so the tables below use
rank 0 as the per-card representative. For repeated batch-level tags, the table
uses the maximum current `allocated_gib` or `reserved_gib` observed at that tag
over the 100 local steps. `peak alloc/res` uses the maximum reported
`max_allocated_gib` and `max_reserved_gib`.

## Rank 0 Case Summary

All values are GiB.

| Case | Model | Static / opt init | DDP | Forward alloc / res | Backward alloc | Step alloc | Peak alloc / res |
|---|---:|---:|---:|---:|---:|---:|---:|
| `00_baseline_current` | 1.889 | 1.902 | 3.742 | 38.862 / 41.102 | 9.320 | 7.475 | 39.048 / 41.102 |
| `01_attention_sdpa` | 1.889 | 1.902 | 3.742 | 21.743 / 23.313 | 9.331 | 7.487 | 21.817 / 23.313 |
| `02_gradient_checkpointing` | 1.889 | 1.902 | 3.742 | 10.257 / 13.197 | 9.384 | 7.539 | 11.408 / 13.201 |
| `03_no_mvc_forward` | 1.889 | 1.902 | 3.784 | 38.751 / 39.982 | 9.304 | 7.479 | 39.047 / 39.982 |
| `04_no_mvc_module` | 1.869 | 1.881 | 3.702 | 38.670 / 40.227 | 9.221 | 7.397 | 38.965 / 40.227 |
| `05_no_zero_prob` | 1.869 | 1.881 | 3.702 | 38.619 / 40.717 | 9.218 | 7.395 | 38.963 / 40.717 |
| `06_static_gene_amp_dtype` | 1.889 | 1.895 | 3.735 | 39.049 / 40.705 | 9.313 | 7.467 | 39.236 / 40.705 |
| `07_sdpa_checkpoint` | 1.889 | 1.902 | 3.742 | 10.295 / 11.561 | 9.420 | 7.576 | 11.261 / 11.561 |
| `08_batch8_baseline` | 1.889 | 1.902 | 3.742 | 23.625 / 24.461 | 9.321 | 7.476 | 23.743 / 24.461 |
| `09_ffn_chunk128` | 1.889 | 1.902 | 3.742 | 38.875 / 40.691 | 9.326 | 7.478 | 39.059 / 40.691 |
| `10_amp_bfloat16` | 1.889 | 1.895 | 3.735 | 39.049 / 40.705 | 9.313 | 7.467 | 39.236 / 40.705 |

## Savings Versus Baseline

Positive values mean saved memory versus `00_baseline_current`.

| Case | Forward allocated saved | Forward reserved saved | Peak allocated saved | Peak reserved saved | Interpretation |
|---|---:|---:|---:|---:|---|
| `01_attention_sdpa` | 17.119 | 17.789 | 17.231 | 17.789 | Eager attention is a major source. |
| `02_gradient_checkpointing` | 28.605 | 27.904 | 27.640 | 27.900 | Saved transformer activations are the largest single source. |
| `03_no_mvc_forward` | 0.111 | 1.119 | 0.001 | 1.119 | MVC forward/loss is not a primary tensor-memory source. |
| `04_no_mvc_module` | 0.192 | 0.875 | 0.083 | 0.875 | MVC module params and states are small. |
| `05_no_zero_prob` | 0.243 | 0.385 | 0.085 | 0.385 | Explicit zero-prob heads/losses have small memory impact. |
| `06_static_gene_amp_dtype` | -0.187 | 0.397 | -0.188 | 0.397 | Static buffer dtype saves only about 0.007 GiB at setup; reserved differences are allocator noise. |
| `07_sdpa_checkpoint` | 28.567 | 29.541 | 27.787 | 29.541 | Best observed memory configuration. |
| `08_batch8_baseline` | 15.237 | 16.641 | 15.305 | 16.641 | Activation memory scales strongly with batch size. |
| `09_ffn_chunk128` | -0.013 | 0.410 | -0.011 | 0.410 | FFN chunking did not reduce live allocated memory in this probe. |
| `10_amp_bfloat16` | -0.187 | 0.397 | -0.188 | 0.397 | Not a clean dtype comparison because baseline `amp_dtype=auto` already resolved to bf16. |

## Memory Source Breakdown

Baseline rank 0:

- Parameters after `model.to`: 1.889 GiB.
- Static gene buffers: +0.014 GiB, reaching 1.902 GiB.
- AdamW init before first step: no visible NPU state allocation, still 1.902 GiB.
- DDP wrap: +1.840 GiB allocated, reaching 3.742 GiB.
- First optimizer step materializes AdamW state: steady post-step allocation is
  7.475 GiB, so optimizer-state-related growth from DDP steady state is about
  3.733 GiB.
- Forward peak: `after_forward` is 38.862 GiB allocated and 41.102 GiB reserved.
  Compared with post-step steady allocation, forward saved activations and
  attention-side tensors account for about 31.387 GiB allocated.
- Backward leaves 9.320 GiB allocated, about 1.845 GiB above post-step steady
  state, consistent with gradients and backward-side state.

## Main Findings

1. The 44 GB/card observation is dominated by forward activations, not by model
   parameters or AdamW alone. Baseline forward live allocation is 38.862 GiB and
   torch reserved reaches 41.102 GiB.
2. Eager attention is a major component. Switching only to SDPA drops peak
   reserved from 41.102 GiB to 23.313 GiB, saving 17.789 GiB.
3. Activation checkpointing is the largest single switch. It drops peak reserved
   to 13.201 GiB, saving 27.900 GiB.
4. SDPA plus checkpointing is the best observed configuration. It drops peak
   reserved to 11.561 GiB, saving 29.541 GiB versus baseline.
5. Batch size matters almost linearly for the activation-heavy baseline. Batch 8
   drops peak reserved to 24.461 GiB, saving 16.641 GiB.
6. AdamW state is material but secondary. It contributes about 3.733 GiB after
   the first optimizer step.
7. DDP adds about 1.840 GiB after wrapping.
8. MVC, explicit zero-prob, static gene dtype, and FFN chunking are not the main
   causes of the observed memory peak.

## Runtime Notes

Rank 0 total runtime for 100 local steps:

| Case | Runtime seconds | Note |
|---|---:|---|
| `00_baseline_current` | 106.70 | Baseline. |
| `01_attention_sdpa` | 87.99 | Lower memory and faster in this run. |
| `02_gradient_checkpointing` | 129.42 | Lower memory, slower because of recomputation. |
| `07_sdpa_checkpoint` | 123.79 | Best memory result, moderate recomputation cost. |
| `08_batch8_baseline` | 83.16 | Smaller batch, not comparable for throughput per sample. |
| `09_ffn_chunk128` | 111.26 | Did not reduce allocated memory and was slower. |
| `10_amp_bfloat16` | 110.81 | Comparable to baseline; not a true dtype contrast. |

The explicit zero-prob branch has small memory impact but large loss-time impact:
baseline accumulated `loss_gep_zero_prob` time was 13.16 s, while
`05_no_zero_prob` reduced it to 0.03 s.

## Recommendations

Preferred memory configuration:

```bash
--runtime_attn_implementation=sdpa --gradient_checkpointing=true
```

If throughput is more important and the full checkpointing overhead is too high,
use SDPA first:

```bash
--runtime_attn_implementation=sdpa
```

Operational priority:

1. Make SDPA the default attention implementation for the 500M run if numerical
   checks pass.
2. Enable gradient checkpointing for memory-constrained runs or when increasing
   batch size.
3. Use batch size as the next direct activation-memory lever.
4. Consider ZeRO or optimizer sharding only after activation memory is addressed;
   AdamW state is about 3.7 GiB/card here, so it cannot explain the 44 GB/card
   peak by itself.
5. Do not prioritize MVC removal, zero-prob removal, static gene dtype changes,
   or FFN chunking as primary HBM fixes. Zero-prob removal may still be useful
   for speed.

## Failed Or Non-comparable Cases

- The first `09_ffn_chunk128` run in `memory_diagnosis_500m_20260506_170759`
  failed because the old `apply_chunking_to_forward` path required dimension 641
  to be divisible by chunk size 128. The fixed rerun in
  `memory_diagnosis_500m_20260506_175946` is the comparable result used above.
- `10_amp_bfloat16` is not a clean comparison against a float32 AMP baseline.
  The baseline `amp_dtype=auto` already resolved to bf16, and case 10 also used
  `static_gene_dtype=amp`, making it effectively a repeat of the static dtype
  probe plus explicit bf16 selection.
