#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --partition=pi-gaox
#SBATCH --qos=pi-gaox
#SBATCH --job-name=SpeciesLLM_stable_351M
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


set -eo pipefail

# Submit with the release directory as Slurm's working directory.
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted release directory}"
BUNDLE_DIR=$PWD
TRAIN_ENTRY=scripts/continue_train_MNodes_torchrun_mfu_parquet_speciesSampling_logits_diagnostics_dataPrefetch.py
test -f "$TRAIN_ENTRY"
export PYTHONPATH="$BUNDLE_DIR"

TRAIN_MODE="${SPECIESLLM_TRAIN_MODE:-resume}"
MODEL_CONFIG="${SPECIESLLM_MODEL_CONFIG:-configuration/350m_batch1_batch2.json}"
CHECKPOINT_PATH="${SPECIESLLM_CHECKPOINT:-/mnt/data/u/wangj0g/projects/SpeciesLLM/model_weights/1st2nd_data_pretrain_prefetch_1369macrogenes_350M_SpeciesSampling_HuberLoss_logits_lr1e-5_resume35000_396057/checkpoint_latest.pth}"
OUT_PATH="${SPECIESLLM_OUT_DIR:-/mnt/data/u/wangj0g/projects/SpeciesLLM/model_weights/SpeciesLLM_stable_351M_${SLURM_JOB_ID}}"
MODE_ARGS=(--train_mode "$TRAIN_MODE" --model_config "$MODEL_CONFIG")
case "$TRAIN_MODE" in
    resume) MODE_ARGS+=(--checkpoint_path "$CHECKPOINT_PATH" --optimizer_state keep --epoch_cells 200000000) ;;
    warm_start) MODE_ARGS+=(--checkpoint_path "$CHECKPOINT_PATH" --epoch_cells 0) ;;
    scratch) MODE_ARGS+=(--epoch_cells 0); CHECKPOINT_PATH= ;;
    *) echo "Unknown SPECIESLLM_TRAIN_MODE: $TRAIN_MODE" >&2; exit 1 ;;
esac

# Use the existing validated environment on Orix, or the bundle's locked environment.
ENV_MANIFEST="${SPECIESLLM_PIXI_MANIFEST:-/home/wangj0g/projects/SpeciesLLM/mmGeneration_edit_V3_h200/pixi.toml}"
if [[ ! -f "$ENV_MANIFEST" && -z "${SPECIESLLM_PIXI_MANIFEST:-}" ]]; then
    ENV_MANIFEST="$BUNDLE_DIR/pixi.toml"
fi
PIXI_HOOK="$(pixi shell-hook --shell bash --manifest-path "$ENV_MANIFEST" --locked)"
eval "$PIXI_HOOK"
cd "$BUNDLE_DIR"
export PYTHONPATH="$BUNDLE_DIR"

echo "Bundle: $BUNDLE_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUT_PATH"
echo "Python: $(command -v python)"

export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2000
export OMP_NUM_THREADS=1

srun -n 1 -N 1 -c "${SLURM_CPUS_PER_TASK}" --gpus="${SLURM_GPUS_PER_NODE}" \
    torchrun --standalone --nproc_per_node="${SLURM_GPUS_PER_NODE}" "$TRAIN_ENTRY" \
    --seq_len 1369 \
    --epoch 3 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --new_data_fraction 0.75 \
    --sampling_seed 20261115 \
    --warmup_steps 300 \
    --warmup_start_lr 1e-6 \
    "${MODE_ARGS[@]}" \
    --diagnostics \
    --diagnostic_stats_interval 50 \
    --diagnostic_max_snapshots 3 \
    --diagnostic_cooldown_updates 1000 \
    --batch_size 32 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --log_interval 50 \
    --out_path "$OUT_PATH" \
    --wanlog True \
    --compile_flag True "$@"
