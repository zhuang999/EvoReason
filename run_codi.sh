#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=$(find /root/anaconda3/envs/think/lib -name 'libstdc++.so.6' 2>/dev/null | head -1)
if [[ -z "${LD_PRELOAD}" ]]; then
    echo "WARNING: libstdc++.so.6 not found"
else
    echo "LD_PRELOAD=${LD_PRELOAD}"
fi
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TIMEOUT=3600000

REC_MODEL="/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/checkpoints/beauty_RA/epoch_2"
DATA_PATH="/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_RA_train.parquet"
OUTPUT_DIR="./checkpoints/latent_codi_stage1"
MEMORY_PATH="/home/weizhipeng/zhuang/distill/reasonrec/R3/MemRec/results/instructrec-beauty_memory_20260712_043242/memory.jsonl"
COT_PATH=""
NUM_GPUS=8

if [[ $# -ge 1 ]]; then REC_MODEL="$1"; fi
if [[ $# -ge 2 ]]; then DATA_PATH="$2"; fi
if [[ $# -ge 3 ]]; then OUTPUT_DIR="$3"; fi

if [[ ! -d "${REC_MODEL}" ]]; then echo "ERROR: model not found: ${REC_MODEL}"; exit 1; fi
if [[ ! -f "${DATA_PATH}" ]]; then echo "ERROR: data not found: ${DATA_PATH}"; exit 1; fi
if [[ -n "${MEMORY_PATH}" && ! -f "${MEMORY_PATH}" ]]; then
    echo "WARNING: memory file not found: ${MEMORY_PATH}, training without memory"
    MEMORY_PATH=""
fi

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%H:%M:%S')] ========== Stage 1: SFT (CE + hidden distill) =========="
echo "  Model:  ${REC_MODEL}"
echo "  Data:   ${DATA_PATH}"
echo "  Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29560 \
    train_codi.py \
    --model_name_or_path "${REC_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --stage 1 \
    --max_length 2048 \
    --num_latent 8 \
    --num_jacobi_steps 3 \
    --use_prj True \
    --prj_dim 1024 \
    --distill_loss_factor 0.1 \
    --distill_loss_type smooth_l1 \
    --distill_loss_div_std False \
    --kv_distill_factor 0.0 \
    --mix_lambda 0.07 \
    --kv_window_size 4 \
    --kv_loss_p 1 \
    --opd_kl_factor 0.0 \
    --opd_gate_beta 5.0 \
    --ref_loss_factor 1.0 \
    --div_factor 0.1 \
    --use_primitive_distill True \
    --primitive_distill_factor 1.0 \
    --use_lora False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    ${MEMORY_PATH:+--memory_path "${MEMORY_PATH}"} \
    ${COT_PATH:+--cot_path "${COT_PATH}"}

echo "[$(date '+%H:%M:%S')] Stage 1 SFT complete! Model saved to: ${OUTPUT_DIR}"
