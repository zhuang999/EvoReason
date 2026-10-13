#!/bin/bash
# TRD-style Latent OPD: 4-step iterative training loop
# Phase C-1: Student generate y_o (SID via latent_generate)
# Phase C-2: Teacher refine generate y_r (based on student latent + refine prompt)
# Phase C-3: OPD training (forward KL on SID tokens)
# Phase C-4: PtoolUpdater — evolve primitive library P*

set -e

MODEL_PATH=${1:-"checkpoints/teacher_sft"}
DATA_PATH=${2:-"data/beauty_eval.parquet"}
OUTPUT_DIR=${3:-"checkpoints/latent_opd"}
NUM_EPOCHS=${4:-3}
GPUS=${5:-4}
REPORT_PATH=${6:-"reasoning_primitives/benchmarks/rec/induction_out/original/rec/report.json"}
DATA_JSON=${7:-"reasoning_primitives/benchmarks/rec/data/rec_train.json"}
TRAIN_PARQUET=${8:-"/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_RA_train.parquet"}
VLLM_MODEL=${9:-"openai/QwQ-32B"}
VLLM_API_BASES=${10:-"http://localhost:12350/v1 http://localhost:12351/v1 http://localhost:12352/v1 http://localhost:12353/v1"}
SKIP_COT=${SKIP_COT:-""}

echo "============================================"
echo "TRD-style Latent OPD Training"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  DATA_PATH:  $DATA_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  NUM_EPOCHS: $NUM_EPOCHS"
echo "  GPUS:       $GPUS"
echo "  REPORT_PATH: $REPORT_PATH"
echo "  VLLM_MODEL: $VLLM_MODEL"
echo "============================================"

for EPOCH in $(seq 1 $NUM_EPOCHS); do
    echo ""
    echo "========== OPD Iteration $EPOCH / $NUM_EPOCHS =========="

    # C-1: Student generate y_o
    YO_PATH="${OUTPUT_DIR}/epoch_${EPOCH}/yo_responses.parquet"
    echo "[C-1] Generating y_o with student latent_generate..."
    mkdir -p "${OUTPUT_DIR}/epoch_${EPOCH}"
    python generate_yo.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_path "$YO_PATH" \
        --batch_size 2

    # C-2: Teacher refine generate y_r based on student latent
    YR_PATH="${OUTPUT_DIR}/epoch_${EPOCH}/yr_responses.parquet"
    echo "[C-2] Generating y_r with teacher latent_refine_generate..."
    python generate_yr.py \
        --model_path "$MODEL_PATH" \
        --data_path "$YO_PATH" \
        --output_path "$YR_PATH" \
        --batch_size 2 \
        --temperature 0.7

    # C-3: OPD training with forward KL
    EPOCH_DIR="${OUTPUT_DIR}/epoch_${EPOCH}"
    echo "[C-3] Training with forward KL (MC)..."
    torchrun --nproc_per_node=$GPUS train_codi.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_path "$YR_PATH" \
        --output_dir "$EPOCH_DIR" \
        --opd_kl_factor 1.0 \
        --div_factor 0.02 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 8e-4

    MODEL_PATH="$EPOCH_DIR"

    # C-4: PtoolUpdater — evolve primitive library P*
    PTOOL_DIR="${EPOCH_DIR}/ptool_update"
    echo "[C-4] Evolving ptool library..."
    python update_ptools.py \
        --yr_path "$YR_PATH" \
        --report_path "$REPORT_PATH" \
        --data_json "$DATA_JSON" \
        --train_parquet "$TRAIN_PARQUET" \
        --output_dir "$PTOOL_DIR" \
        --model "$VLLM_MODEL" \
        --api-bases $VLLM_API_BASES \
        --epoch $EPOCH \
        ${SKIP_COT:+--skip-cot}

    echo "========== Iteration $EPOCH done. Next model: $MODEL_PATH =========="
done

echo "All $NUM_EPOCHS iterations completed!"
