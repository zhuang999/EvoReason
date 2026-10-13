set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
: "${MODEL_STAGE1:?}"
: "${DATA_ROOT:?}"
: "${ANCHOR_DIR:?}"
PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m torch.distributed.run --nproc_per_node "${NUM_GPUS:-8}" --master_port "${MASTER_PORT:-29720}" scripts/train_sft.py --model "${MODEL_STAGE1}" --train-data "${DATA_ROOT}/beauty/training_prediction_sid_data_train.parquet" --eval-data "${DATA_ROOT}/beauty/training_prediction_sid_data_val.parquet" --train-anchors "${ANCHOR_DIR}/train.pt" --eval-anchors "${ANCHOR_DIR}/val.pt" --output-dir "${OUTPUT_DIR:-outputs/beauty/sft}" --eval-limit 2000 --epochs 3 --learning-rate 5e-5 --batch-size 8 --gradient-accumulation-steps 4 --align-weight 0.0001 --policy-weight 0 --fixed-train-steps 1 --latent-adapter --latent-adapter-scale 1e-3 --trainable-markers --init-latent-specials --freeze-backbone --deepspeed configs/ds_zero2.json --max-sequence-length 512 --seed 42 --bf16 "$@"
