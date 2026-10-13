set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: MODEL=/sft/checkpoint bash run_onerec_rl.sh <Beauty|Sports|Toys> [extra args]" >&2
  exit 2
fi
source "$(dirname "$0")/onerec_dataset_env.sh" "$1"
shift
: "${MODEL:?Set MODEL to the LatentR3 OneRec SFT checkpoint}"
PYTHON="${PYTHON:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
if [[ ! "${GPU_LIST}" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "GPU_LIST must contain exactly two GPU ids for official modified GRPO" >&2
  exit 2
fi
SEED="${SEED:-42}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/seed-${SEED}/rl-lr-${LR}}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 10000 + 20000))}"

"${PYTHON}" "${ROOT_DIR}/onerec/preflight.py" --dataset "${DATASET_SLUG}" \
  --data-dir "${PARQUET_DIR}" --metadata "${METADATA}" --model "${MODEL}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" -m torch.distributed.run \
  --master_port "${MASTER_PORT}" --nproc_per_node 2 "${ROOT_DIR}/onerec/train_rl.py" \
  --model "${MODEL}" \
  --train-data "${PARQUET_DIR}/training_prediction_sid_data_train.parquet" \
  --eval-data "${PARQUET_DIR}/training_prediction_sid_data_val.parquet" \
  --metadata "${METADATA}" --output-dir "${OUTPUT_DIR}" \
  --learning-rate "${LR}" --epochs 1 --batch-size 256 --micro-batch-size 8 \
  --num-generations 8 --beta 0.0 --num-iterations 1 --epsilon 0.2 \
  --epsilon-high 0.28 --noise-std 1.5 --max-sequence-length 512 --seed "${SEED}" "$@"
