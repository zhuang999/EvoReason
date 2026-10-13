#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=$(find /root/anaconda3/envs/think/lib -name 'libstdc++.so.6' 2>/dev/null | head -1)
export NCCL_DEBUG=WARN

MODEL_PATH="/code/checkpoints/latent_codi_v8"
DATA_PATH="/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_prediction_sid_data_test.parquet"
NUM_GPUS=8
NUM_BEAMS=10
BATCH_SIZE=1
MAX_NEW_TOKENS=6
NUM_LATENT=4
LOG_DIR="eval_logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR}/eval_${TS}"
mkdir -p "${LOG_DIR}"

TOTAL_SAMPLES=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${DATA_PATH}')))")
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))

echo "[$(date '+%H:%M:%S')] ========== 8-GPU Parallel Evaluation =========="
echo "  Model: ${MODEL_PATH}"
echo "  Data:  ${DATA_PATH}"
echo "  Total samples: ${TOTAL_SAMPLES}"
echo "  Samples per GPU: ${SAMPLES_PER_GPU}"
echo "  Beams: ${NUM_BEAMS}"
echo "  Log dir: ${LOG_DIR}"

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    offset=$((gpu_id * SAMPLES_PER_GPU))
    log_file="${LOG_DIR}/gpu_${gpu_id}.log"

    echo "  Starting GPU ${gpu_id}: offset=${offset}, samples=${SAMPLES_PER_GPU}"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 -u eval_codi.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --num_latent ${NUM_LATENT} \
        --num_beams ${NUM_BEAMS} \
        --num_return_sequences ${NUM_BEAMS} \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_samples ${SAMPLES_PER_GPU} \
        --sample_offset ${offset} \
        --gpu_id ${gpu_id} \
        --output_file "${LOG_DIR}/result_gpu${gpu_id}.json" \
        > "${log_file}" 2>&1 &

    pids+=($!)
    sleep 2
done

echo ""
echo "All ${NUM_GPUS} processes started:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU ${i}: PID ${pids[$i]} -> ${LOG_DIR}/gpu_${i}.log"
done

echo ""
echo "Waiting for all processes to complete..."
wait "${pids[@]}"

echo ""
echo "All processes completed. Merging results..."
python3 -c "
import json, os, numpy as np

log_dir = '${LOG_DIR}'
all_preds = []
all_gts = []
total = 0

for gpu_id in range(${NUM_GPUS}):
    f = os.path.join(log_dir, f'result_gpu{gpu_id}.json')
    if not os.path.exists(f):
        print(f'  WARNING: {f} not found')
        continue
    with open(f) as fh:
        data = json.load(fh)
    n = data.get('n_test', 0)
    total += n
    m = data.get('metrics', {})
    print(f'  GPU {gpu_id}: {n} samples, hit@5={m.get(\"hit@5\",0):.4f}, ndcg@5={m.get(\"ndcg@5\",0):.4f}, hit@10={m.get(\"hit@10\",0):.4f}, ndcg@10={m.get(\"ndcg@10\",0):.4f}')

# Merge per-sample results from logs for final aggregate
# Each GPU result file already has its own metrics; average them
metrics_keys = ['hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']
all_metrics = {k: [] for k in metrics_keys}
weighted_metrics = {k: 0.0 for k in metrics_keys}
total_n = 0

for gpu_id in range(${NUM_GPUS}):
    f = os.path.join(log_dir, f'result_gpu{gpu_id}.json')
    if not os.path.exists(f):
        continue
    with open(f) as fh:
        data = json.load(fh)
    n = data.get('n_test', 0)
    total_n += n
    for k in metrics_keys:
        v = data.get('metrics', {}).get(k, 0)
        all_metrics[k].append(v)
        weighted_metrics[k] += v * n

final = {}
for k in metrics_keys:
    if total_n > 0:
        final[k] = weighted_metrics[k] / total_n
    else:
        final[k] = 0

print()
print('===== Final Results =====')
print(f'  Total samples: {total_n}')
for k, v in final.items():
    print(f'  {k}: {v:.4f}')

with open(os.path.join(log_dir, 'final_results.json'), 'w') as fh:
    json.dump({'total_samples': total_n, 'metrics': final}, fh, indent=2)
print(f'Saved to {log_dir}/final_results.json')
"

echo "[$(date '+%H:%M:%S')] Evaluation complete!"
