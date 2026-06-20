#!/usr/bin/env bash
# =============================================================================
# run.sh — one-shot launcher for SFT_training.py (distributed train + eval)
#
#   Usage:
#       bash run.sh                # train + eval with the defaults below
#       NUM_GPUS=4 bash run.sh     # override any setting via env var
#       MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct bash run.sh
#       DO_TRAIN=0 bash run.sh     # evaluation only
#
# Every argument of SFT_training.py is instantiated below; tweak the variables
# in the CONFIG section (or pass them as environment variables).
# =============================================================================
set -euo pipefail

# Directory containing this script (so SFT_training.py is found from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# CONFIG (override any of these via environment variables)
# ----------------------------------------------------------------------------
# --- environment ------------------------------------------------------------
# CONDA_ENV="${CONDA_ENV:-3dgraphllm}"            # conda env that has torch/transformers
# PROJECT_DIR="${PROJECT_DIR:-/home/yuhanghe/HPC}"
NUM_GPUS="${NUM_GPUS:-8}"                         # processes for torchrun
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"  # GPUs to expose
MASTER_PORT="${MASTER_PORT:-29500}"

# --- model (ModelArguments) -------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"   # or Qwen/Qwen2.5-Coder-32B-Instruct
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"                   # sdpa | flash_attention_2 | eager
USE_LORA="${USE_LORA:-0}"                         # 1 = LoRA (fits 32B); 0 = full FT
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# --- full fine-tuning (used only when USE_LORA=0) ---------------------------
# These feed the explicit FSDP wrapper in SFT_training.run_training():
#   FSDP           -> sharding strategy + auto_wrap (full_shard | shard_grad_op |
#                     hybrid_shard | no_shard, optionally with "auto_wrap").
#   FSDP_WRAP_CLS  -> transformer decoder layer class to shard per-block.
# Full FT of a 32B model needs full_shard so params/grads/optimizer states are
# split across the 8 GPUs (DDP would replicate everything and OOM).
FSDP="${FSDP:-full_shard auto_wrap}"
FSDP_WRAP_CLS="${FSDP_WRAP_CLS:-Qwen3DecoderLayer}"   # Qwen3-32B; use Qwen2DecoderLayer for Qwen2.5
# FSDP_WRAP_CLS="${FSDP_WRAP_CLS:-Qwen2DecoderLayer}"

# --- data (DataArguments) ---------------------------------------------------
TRAIN_FILE="${TRAIN_FILE:-/mnt/blob-data-sigmasystem/xuehui/sft_training_format_concat_full_thread_materialized_evidence_v1_plus_diagnostic/stage1_full_85_15_diag72_full/stage1_train.jsonl}"
DEV_FILE="${DEV_FILE:-/mnt/blob-data-sigmasystem/xuehui/sft_training_format_concat_full_thread_materialized_evidence_v1_plus_diagnostic/stage1_full_85_15_diag72_full/stage1_dev.jsonl}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

# --- training (transformers.TrainingArguments) ------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/blob-data-sigmasystem-out/yuhang/SFT/Qwen3-32B}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
GRAD_CKPT="${GRAD_CKPT:-True}"
BF16="${BF16:-True}"
FP16="${FP16:-False}"
SEED="${SEED:-20260617}"
REPORT_TO="${REPORT_TO:-none}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# --- evaluation (EvalArguments) ---------------------------------------------
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
CLASS_MAX_NEW_TOKENS="${CLASS_MAX_NEW_TOKENS:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"                    # empty = self-judge with policy model
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"             # set to use an OpenAI-compatible judge
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_API_MODEL="${JUDGE_API_MODEL:-gpt-4o-mini}"

# ----------------------------------------------------------------------------
# Activate conda environment
# ----------------------------------------------------------------------------
# if command -v conda >/dev/null 2>&1; then
#     # shellcheck disable=SC1091
#     source "$(conda info --base)/etc/profile.d/conda.sh"
#     conda activate "${CONDA_ENV}"
# fi

# cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TOKENIZERS_PARALLELISM=false

# Create the output directory if it does not already exist.
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------------
# Assemble arguments
# ----------------------------------------------------------------------------
ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --trust_remote_code "${TRUST_REMOTE_CODE}"
    --attn_implementation "${ATTN_IMPL}"
    --train_file "${TRAIN_FILE}"
    --dev_file "${DEV_FILE}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --logging_steps "${LOGGING_STEPS}"
    --save_strategy "${SAVE_STRATEGY}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --gradient_checkpointing "${GRAD_CKPT}"
    --bf16 "${BF16}"
    --fp16 "${FP16}"
    --seed "${SEED}"
    --report_to "${REPORT_TO}"
    --dataloader_num_workers "${DATALOADER_WORKERS}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
    --class_max_new_tokens "${CLASS_MAX_NEW_TOKENS}"
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}"
    --judge_api_model "${JUDGE_API_MODEL}"
)

[[ "${DO_TRAIN}" == "1" ]] && ARGS+=(--do_train)
[[ "${DO_EVAL}"  == "1" ]] && ARGS+=(--do_eval)

# LoRA vs. full fine-tuning (FSDP)
if [[ "${USE_LORA}" == "1" ]]; then
    ARGS+=(
        --use_lora
        --lora_r "${LORA_R}"
        --lora_alpha "${LORA_ALPHA}"
        --lora_dropout "${LORA_DROPOUT}"
        --lora_target_modules "${LORA_TARGET_MODULES}"
    )
else
    # Full fine-tuning via FSDP. The custom training loop reads:
    #   --fsdp         -> sharding strategy / auto_wrap flags
    #   --fsdp_config  -> JSON with "transformer_layer_cls_to_wrap" (the decoder
    #                     block class to shard per-layer).
    FSDP_CONFIG_FILE="$(mktemp -t fsdp_config.XXXXXX.json)"
    cat > "${FSDP_CONFIG_FILE}" <<EOF
{
    "transformer_layer_cls_to_wrap": ["${FSDP_WRAP_CLS}"]
}
EOF
    ARGS+=(
        --fsdp "${FSDP}"
        --fsdp_config "${FSDP_CONFIG_FILE}"
    )
fi

# Optional judge overrides
[[ -n "${JUDGE_MODEL}"    ]] && ARGS+=(--judge_model "${JUDGE_MODEL}")
[[ -n "${JUDGE_API_BASE}" ]] && ARGS+=(--judge_api_base "${JUDGE_API_BASE}")
[[ -n "${JUDGE_API_KEY}"  ]] && ARGS+=(--judge_api_key "${JUDGE_API_KEY}")

# ----------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------
echo "================ SFT launch ================"
# echo "  env         : ${CONDA_ENV}"
echo "  model       : ${MODEL_NAME}"
echo "  GPUs        : ${CUDA_DEVICES}  (nproc=${NUM_GPUS})"
echo "  mode        : $( [[ ${USE_LORA} == 1 ]] && echo LoRA || echo 'full FT (FSDP)' )"
echo "  output_dir  : ${OUTPUT_DIR}"
echo "  do_train=${DO_TRAIN}  do_eval=${DO_EVAL}"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/SFT_training.py" "${ARGS[@]}"
