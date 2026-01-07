#!/usr/bin/env bash
set -euo pipefail

REPO=/u/abasu2/VLM2Vec
PY=/projects/bfln/abasu2/conda_envs/omni_vlm2vec/bin/python

MODEL_NAME="nvidia/omni-embed-nemotron-3b"
MODEL_BACKBONE="omni_embed"

# Downloaded MMEB-V2 root:
DATA_BASEDIR="/work/nvme/bfln/abasu2/MMEB-V2"
OUT_BASE="/work/nvme/bfln/abasu2/omni_outputs/omni-embed-nemotron-3b/ultra_hres"
# OUT_BASE="omni_outputs/omni-embed-nemotron-3b/ultra_hres"
BATCH_SIZE=2

cd "$REPO"

# for MODALITY in image visdoc; do
for MODALITY in image; do
  CFG="experiments/public/eval/${MODALITY}.yaml"
  OUT="${OUT_BASE}/${MODALITY}"
  mkdir -p "$OUT"

  echo "=== Running ${MODALITY} ==="
  DEBUG_OMNI=1 $PY -u eval.py \
    --pooling mean \
    --normalize true \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --model_backbone "${MODEL_BACKBONE}" \
    --model_name "${MODEL_NAME}" \
    --dataset_config "${CFG}" \
    --encode_output_path "${OUT}" \
    --data_basedir "${DATA_BASEDIR}"
done

echo "Done"
