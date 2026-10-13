set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: MODEL=/expanded/base bash run_onerec_sft.sh <Beauty|Sports|Toys> [extra args]" >&2
  exit 2
fi
source "$(dirname "$0")/onerec_dataset_env.sh" "$1"
shift
: "${MODEL:?Set MODEL to the shared expanded Qwen base model}"
PYTHON="${PYTHON:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_GPUS="${#GPUS[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "GPU_LIST must contain at least one GPU id" >&2
  exit 2
fi
SEED="${SEED:-42}"
LR="${LR:-3e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/seed-${SEED}/sft-lr-${LR}}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 10000 + 20000))}"

"${PYTHON}" "${ROOT_DIR}/onerec/preflight.py" --dataset "${DATASET_SLUG}" \
  --data-dir "${PARQUET_DIR}" --metadata "${METADATA}" --model "${MODEL}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" -m torch.distributed.run --master_port "${MASTER_PORT}" \
  --nproc_per_node "${NUM_GPUS}" "${ROOT_DIR}/onerec/train_sft.py" \
  --base-model "${MODEL}" \
  --train-data "${PARQUET_DIR}/training_prediction_sid_data_train.parquet" \
  --eval-data "${PARQUET_DIR}/training_prediction_sid_data_val.parquet" \
  --output-dir "${OUTPUT_DIR}" --batch-size 128 --micro-batch-size 4 \
  --epochs 3 --learning-rate "${LR}" --max-sequence-length 512 --seed "${SEED}" --bf16 "$@"
