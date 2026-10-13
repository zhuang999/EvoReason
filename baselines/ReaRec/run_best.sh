set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON="${PYTHON:-python3}"
"${PYTHON}" run_sports_prefix.py --execute --lr 0.0005 --dropout 0.6 --temperature 2.0 --temp-scale 1.0 --reason-step 2 --emb-size 256 --batch-size 2048 --eval-batch-size 512 --early-stop 5 --epoch 300 --filter-history --output-dir "${OUTPUT_DIR:-outputs/sports-prefix/best}" "$@"
