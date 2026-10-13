#!/bin/bash
# Phase C: On-Policy OPD Training with epoch-level vLLM switching
# 8 A800 GPUs: training uses all 8, vLLM uses all 8 (mutual exclusion)
#
# Usage:
#   bash run_opd_loop.sh \
#       /path/to/beauty_RA_cot/epoch_3 \
#       /path/to/training_RA_train.parquet \
#       checkpoints/opd_loop \
#       3

set -e

MODEL_PATH=${1:?"Usage: run_opd_loop.sh MODEL_PATH DATA_PATH OUTPUT_DIR [NUM_EPOCHS]"}
DATA_PATH=${2:?"DATA_PATH required"}
OUTPUT_DIR=${3:-"checkpoints/opd_loop"}
NUM_EPOCHS=${4:-3}

VLLM_MODEL="/code/QwQ-32B"
REPORT_PATH="reasoning_primitives/benchmarks/rec/induction_out/original/rec/report.json"
DATA_JSON="reasoning_primitives/benchmarks/rec/data/rec_train.json"

GPUS=8
MASTER_PORT=29501

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Phase C: On-Policy OPD Training"
echo "  MODEL_PATH:  $MODEL_PATH"
echo "  DATA_PATH:   $DATA_PATH"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  NUM_EPOCHS:  $NUM_EPOCHS"
echo "  VLLM_MODEL:  $VLLM_MODEL"
echo "============================================"

# ── Helper: start vLLM ──
start_vllm() {
    echo "[vLLM] Starting QwQ-32B (4 instances, TP=2)..."
    mkdir -p /tmp/vllm_logs
    for i in 0 1 2 3; do
        port=$((12350 + i))
        gpu_start=$((i * 2))
        CUDA_VISIBLE_DEVICES=${gpu_start},$((gpu_start + 1)) \
            nohup python -m vllm.entrypoints.openai.api_server \
            --model "$VLLM_MODEL" \
            --served-model-name QwQ-32B \
            --port ${port} \
            --tensor-parallel-size 2 \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.95 \
            > /tmp/vllm_logs/vllm_${port}.log 2>&1 &
        sleep 2
    done
    echo "[vLLM] Waiting for services to be ready..."
    for i in 0 1 2 3; do
        port=$((12350 + i))
        for attempt in $(seq 1 60); do
            if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
                echo "[vLLM] Instance :${port} ready (${attempt}s)"
                break
            fi
            sleep 1
        done
    done
    echo "[vLLM] All instances ready"
}

# ── Helper: stop vLLM ──
stop_vllm() {
    echo "[vLLM] Stopping all instances..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
    echo "[vLLM] All instances stopped"
}

# ── Helper: update ptools ──
update_ptools() {
    local epoch=$1
    local yr_path="${OUTPUT_DIR}/epoch_${epoch}_yr.parquet"
    local ptool_dir="${OUTPUT_DIR}/epoch_${epoch}_ptool_update"

    if [ ! -f "$yr_path" ]; then
        echo "[PtoolUpdate] No yr parquet found at ${yr_path}, skipping"
        return
    fi

    echo "[PtoolUpdate] Updating ptool library (epoch ${epoch})..."
    python update_ptools.py \
        --yr_path "$yr_path" \
        --report_path "$REPORT_PATH" \
        --data_json "$DATA_JSON" \
        --train_parquet "$DATA_PATH" \
        --output_dir "$ptool_dir" \
        --model openai/QwQ-32B \
        --api-bases http://localhost:12350/v1 http://localhost:12351/v1 \
                    http://localhost:12352/v1 http://localhost:12353/v1 \
        --epoch "$epoch" \
        --skip-cot
    echo "[PtoolUpdate] Done"
}

# ── Main loop ──
for EPOCH in $(seq 0 $((NUM_EPOCHS - 1))); do
    echo ""
    echo "========== Epoch ${EPOCH} / $((NUM_EPOCHS - 1)) =========="

    # ── Step 1: Train one epoch (all 8 GPUs) ──
    echo "[Train] Starting epoch ${EPOCH}..."
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    WANDB_MODE=disabled \
    torchrun --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
        train_opd_loop.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --start_epoch ${EPOCH} \
        --single_epoch True \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 8e-4 \
        --ppo_coef 1.0 \
        --opd_coef 1.0 \
        --clip_ratio 0.2 \
        --use_primitive_distill True \
        --kv_compress_mode rkv_cyclic \
        --tokens_per_primitive 4 \
        --div_factor 0.02 \
        --max_cot_length 1024 \
        --use_kl_penalty False \
        --kl_coef 0.01 \
        --update_ptools False

    # Extract HF model from DeepSpeed checkpoint for next epoch
    CKPT_DIR="${OUTPUT_DIR}/epoch_${EPOCH}"
    HF_DIR="${OUTPUT_DIR}/epoch_${EPOCH}_hf"
    if [ -d "${CKPT_DIR}" ] && [ ! -d "${HF_DIR}" ]; then
        echo "[Extract] Converting DeepSpeed checkpoint to HF format..."
        python -c "
import torch, sys, os
from model_codi import LatentRecCoDi
ckpt_dir = '${CKPT_DIR}'
hf_dir = '${HF_DIR}'
os.makedirs(hf_dir, exist_ok=True)
# Find the DeepSpeed checkpoint subdirectory
for root, dirs, files in os.walk(ckpt_dir):
    for f in files:
        if f == 'model_states.pt':
            ds_ckpt = os.path.join(root, f)
            break
    else:
        continue
    break
else:
    print('No model_states.pt found, skipping extraction')
    sys.exit(0)
state_dict = torch.load(ds_ckpt, map_location='cpu')
model = LatentRecCoDi(model_name_or_path='${MODEL_PATH}', use_lora=True)
model.codi.load_state_dict(state_dict, strict=False)
model.codi.save_pretrained(hf_dir)
model.tokenizer.save_pretrained(hf_dir)
print(f'Saved HF model to {hf_dir}')
" || echo "[WARN] HF extraction failed, will try loading DS checkpoint directly"
    fi
    if [ -d "${HF_DIR}" ]; then
        MODEL_PATH="${HF_DIR}"
    else
        MODEL_PATH="${CKPT_DIR}"
    fi
    echo "[Train] Epoch ${EPOCH} done. Next model: ${MODEL_PATH}"

    # ── Step 2: Update ptool library (if not last epoch) ──
    if [ ${EPOCH} -lt $((NUM_EPOCHS - 1)) ]; then
        echo ""
        echo "--- Ptool Library Update ---"
        start_vllm
        update_ptools ${EPOCH}
        stop_vllm
        echo "--- Ptool Library Update Done ---"
    fi

    echo "========== Epoch ${EPOCH} complete =========="
done

echo ""
echo "All ${NUM_EPOCHS} epochs completed!"
