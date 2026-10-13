set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
: "${MODEL_STAGE1:?}"
: "${DATA_ROOT:?}"
DATASET="${1:?}"
shift
PYTHON="${PYTHON:-python3}"
SLUG="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${SLUG}/sft-lr-1e-4}"
MODEL="${MODEL_STAGE1}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" GPU_LIST="${GPU_LIST:-0,1,2,3}" SEED=42 LR=1e-4 OUTPUT_DIR="${OUTPUT_DIR}" bash scripts/run_onerec_sft.sh "${DATASET}" --freeze-backbone --prompt-protocol onerec "$@"
MODEL="${OUTPUT_DIR}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" GPU_LIST="${GPU_LIST:-0,1,2,3}" NUM_BEAMS=10 SPLIT=test MODE=latent PROMPT_PROTOCOL=onerec EVAL_DIR="${OUTPUT_DIR}/test-latent" bash scripts/run_onerec_eval.sh "${DATASET}" --end-k -1 --latent-component full
