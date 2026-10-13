#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=$(find /root/anaconda3/envs/think/lib -name 'libstdc++.so.6' 2>/dev/null | head -1)
export WANDB_MODE=disabled

MODEL_PATH="${1:?Usage: run_eval_codi.sh MODEL_PATH [DATA_PATH] [OUTPUT_FILE]}"
DATA_PATH="${2:-/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_RA_test.parquet}"
OUTPUT_FILE="${3:-eval_codi_results.json}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model not found: ${MODEL_PATH}"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] ========== LatentRec CoDi Evaluation =========="
echo "  Model: ${MODEL_PATH}"
echo "  Data:  ${DATA_PATH}"
echo "  Output: ${OUTPUT_FILE}"

CUDA_VISIBLE_DEVICES=0 python eval_codi.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --num_latent 8 \
    --use_prj True \
    --prj_dim 1024 \
    --num_beams 10 \
    --num_return_sequences 10 \
    --max_new_tokens 30 \
    --batch_size 4

echo "[$(date '+%H:%M:%S')] Evaluation complete!"
