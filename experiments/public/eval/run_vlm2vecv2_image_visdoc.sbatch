#!/bin/bash
#SBATCH -A bfld-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH -t 01:00:00
#SBATCH -J vlm2vecv2_visdoc
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out

set -euo pipefail

export LD_PRELOAD=""
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Keep HF downloads/cache off $HOME
export HF_HOME=/projects/bfln/abasu2/cache/hf
export HF_HUB_ENABLE_HF_TRANSFER=0

bash /u/abasu2/VLM2Vec/experiments/public/eval/run_vlm2vecv2_image_visdoc.sh
