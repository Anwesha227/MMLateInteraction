#!/usr/bin/env bash
set -euo pipefail

REPO=/u/abasu2/VLM2Vec
PY=/projects/bfln/abasu2/conda_envs/omni_vlm2vec/bin/python

MODEL_NAME="VLM2Vec/VLM2Vec-V2.0"
MODEL_BACKBONE="qwen2_vl"

# Downloaded MMEB-V2 root:
DATA_BASEDIR="/work/nvme/bfln/abasu2/MMEB-V2"
OUT_BASE="/work/nvme/bfln/abasu2/vlm2vec_outputs/VLM2Vec-V2.0"
BATCH_SIZE=16

cd "$REPO"

# for MODALITY in image visdoc; do
for MODALITY in visdoc; do
  CFG="experiments/public/eval/${MODALITY}.yaml"
  OUT="${OUT_BASE}/${MODALITY}"
  mkdir -p "$OUT"

  echo "=== Running ${MODALITY} ==="
  $PY -u eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --model_backbone "${MODEL_BACKBONE}" \
    --model_name "${MODEL_NAME}" \
    --dataset_config "${CFG}" \
    --encode_output_path "${OUT}" \
    --data_basedir "${DATA_BASEDIR}"
done

echo "Done"