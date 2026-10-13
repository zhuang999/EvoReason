#!/usr/bin/env bash
# End-to-end: prepare → Phase-1 ReAct traces → induction → induced configs → optional simulate CoT
#
# Usage (from this directory: benchmarks/rec/):
#   bash scripts/run_pipeline.sh \
#       /path/to/RecThinker-main/src/dataset/yelp \
#       /path/to/MemRec/results/.../memory.jsonl \
#       openai/QwQ-32B \
#       http://localhost:12350/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REC_DIR}"

DATA_DIR="${1:?RecThinker dataset dir (step2 output)}"
MEMORY="${2:?MemRec memory.jsonl}"
MODEL="${3:-openai/QwQ-32B}"
API_BASE="${4:-http://localhost:12350/v1}"
MAX_USERS="${5:-0}"
SEED="${6:-42}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONPATH="${REC_DIR}/../../src:${REC_DIR}:${PYTHONPATH:-}"

echo "=== [0] prepare_data ==="
PREP_ARGS=(
  --data_dir "${DATA_DIR}"
  --memory "${MEMORY}"
  --split valid
  --output data/rec_valid.json
  --seed "${SEED}"
)
if [[ "${MAX_USERS}" != "0" ]]; then
  PREP_ARGS+=(--max_users "${MAX_USERS}")
fi
python prepare_data.py "${PREP_ARGS[@]}"

echo "=== [1] Phase-1 ReAct with evidence tools → results.jsonl ==="
python expt.py run --config-file conf/rec_react.yaml \
  "llm.model=${MODEL}" \
  "llm.api_base=${API_BASE}" \
  "dataset.shuffle_seed=${SEED}"

# Find newest results.jsonl under results/
TRACE="$(ls -td results/*/results.jsonl 2>/dev/null | head -1 || true)"
if [[ -z "${TRACE}" ]]; then
  echo "[ERROR] No results.jsonl found under results/"
  exit 1
fi
echo "Using trace: ${TRACE}"

echo "=== [2] Induction (thoughts → ptool specs) ==="
python induction.py \
  --task rec \
  --trace "${TRACE}" \
  --model "${MODEL}" \
  --api-base "${API_BASE}" \
  --only-correct \
  --variant original \
  --max-ptools 5

REPORT="induction_out/original/rec/report.json"
if [[ ! -f "${REPORT}" ]]; then
  echo "[ERROR] Missing ${REPORT}"
  exit 1
fi

echo "=== [3] Generate induced ptools + YAML configs ==="
python generate_induced_configs.py \
  --seed "${SEED}" \
  --report "${REPORT}" \
  --suffix correct

PTools="$(ls induced/induced_ptools_seed${SEED}_correct_rec.py | head -1)"
echo "Induced module: ${PTools}"

echo "=== [4] (optional) Simulate ptools → think_content jsonl ==="
python simulate_cot.py \
  --ptools "${PTools}" \
  --data data/rec_valid.json \
  --output out/think_contents_valid.jsonl \
  --model "${MODEL}" \
  --api-base "${API_BASE}" \
  --workers 4

echo "=== DONE ==="
echo "  report:   ${REPORT}"
echo "  ptools:   ${PTools}"
echo "  thoughts: out/think_contents_valid.jsonl"
echo "  eval cfg: conf/rec_induced_b_seed${SEED}_correct.yaml"
echo ""
echo "To evaluate with induced library (variant B):"
echo "  python expt.py run --config-file conf/rec_induced_b_seed${SEED}_correct.yaml"
