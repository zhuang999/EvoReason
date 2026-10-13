set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: MODEL=/checkpoint bash run_onerec_eval.sh <Beauty|Sports|Toys> [extra args]" >&2
  exit 2
fi
source "$(dirname "$0")/onerec_dataset_env.sh" "$1"
shift
: "${MODEL:?Set MODEL to the LatentR3 OneRec SFT or RL checkpoint}"
PYTHON="${PYTHON:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${#GPUS[@]}"
NUM_BEAMS="${NUM_BEAMS:-50}"
SPLIT="${SPLIT:-test}"
MODE="${MODE:-latent}"
PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-onerec}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/eval-${SPLIT}-${MODE}}"
case "${SPLIT}" in val|test) ;; *) echo "SPLIT must be val or test" >&2; exit 2 ;; esac
case "${MODE}" in direct|latent|latent-off) ;; *) echo "Invalid MODE: ${MODE}" >&2; exit 2 ;; esac
if [[ -f "${EVAL_DIR}/summary.json" ]]; then
  echo "Refusing to overwrite completed evaluation: ${EVAL_DIR}/summary.json" >&2
  exit 1
fi
mkdir -p "${EVAL_DIR}"

"${PYTHON}" "${ROOT_DIR}/onerec/preflight.py" --dataset "${DATASET_SLUG}" \
  --data-dir "${PARQUET_DIR}" --metadata "${METADATA}" --model "${MODEL}"
pids=()
for shard in "${!GPUS[@]}"; do
  shard_dir="${EVAL_DIR}/shard_${shard}"
  mkdir -p "${shard_dir}"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${PYTHON}" "${ROOT_DIR}/onerec/evaluate.py" \
    --model "${MODEL}" --data "${PARQUET_DIR}/training_prediction_sid_data_${SPLIT}.parquet" \
    --metadata "${METADATA}" --output "${shard_dir}/result.json" \
    --records-output "${shard_dir}/records.jsonl" \
    --num-shards "${NUM_SHARDS}" --shard-index "${shard}" --num-beams "${NUM_BEAMS}" \
    --max-sequence-length 512 --mode "${MODE}" --prompt-protocol "${PROMPT_PROTOCOL}" \
    "$@" >"${shard_dir}/eval.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
"${PYTHON}" "${ROOT_DIR}/onerec/aggregate_eval.py" \
  --inputs "${EVAL_DIR}"/shard_*/result.json --output "${EVAL_DIR}/summary.json"
"${PYTHON}" "${ROOT_DIR}/onerec/aggregate_records.py" \
  --inputs "${EVAL_DIR}"/shard_*/records.jsonl --output "${EVAL_DIR}/records.jsonl"
echo "Evaluation written to ${EVAL_DIR}"
