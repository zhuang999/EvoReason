set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
: "${MODEL:?}"
: "${DATA_ROOT:?}"
DATASET="${1:?}"
shift
case "${DATASET}" in
  Beauty) SLUG=beauty; STEPS=2 ;;
  Sports) SLUG=sports; STEPS=1 ;;
  *) exit 2 ;;
esac
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${SLUG}/best-eval}"
mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" scripts/evaluate.py --model "${MODEL}" --data "${DATA_ROOT}/${SLUG}/training_prediction_sid_data_${SPLIT:-test}.parquet" --metadata "${DATA_ROOT}/${DATASET}.pretrain.json" --output "${OUTPUT_DIR}/summary.json" --records-output "${OUTPUT_DIR}/records.jsonl" --num-beams 10 --fixed-latent-steps "${STEPS}" --hybrid-alpha 0.1 --hybrid-mode log_linear --max-sequence-length 512 "$@"
