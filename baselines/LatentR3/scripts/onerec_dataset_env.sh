set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: source onerec_dataset_env.sh <Beauty|Sports|Toys>" >&2
  return 2 2>/dev/null || exit 2
fi
DATASET_LOWER="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
case "${DATASET_LOWER}" in
  beauty) DATASET_NAME="Beauty"; DATASET_SLUG="beauty" ;;
  sports) DATASET_NAME="Sports"; DATASET_SLUG="sports" ;;
  toys) DATASET_NAME="Toys"; DATASET_SLUG="toys" ;;
  *) echo "Unsupported dataset: $1" >&2; return 2 2>/dev/null || exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data}"
PARQUET_DIR="${PARQUET_DIR:-${DATA_ROOT}/${DATASET_SLUG}}"
METADATA="${METADATA:-${DATA_ROOT}/${DATASET_NAME}.pretrain.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/onerec-${DATASET_SLUG}}"
export DATASET_NAME DATASET_SLUG ROOT_DIR REPO_DIR DATA_ROOT PARQUET_DIR METADATA OUTPUT_ROOT
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
