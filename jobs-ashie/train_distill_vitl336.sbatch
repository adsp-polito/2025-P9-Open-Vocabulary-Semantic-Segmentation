#!/bin/bash
#SBATCH --job-name=distill_vitl336
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/distill_vitl336_%j.out
#SBATCH --error=logs/distill_vitl336_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
OUTPUT_DIR="output/distill_vitl_336"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT_DIR}"

echo "============================================================"
echo "GS-Distill — Phase 3: Student distillation (ViT-L/14@336px)"
echo "  Cache dir:   cache/vitl_336/"
echo "  Image dir:   gs_net/data/datasets/LandDiscover_50K/TR_Image"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-N/A}"
echo "Node:    ${SLURM_NODELIST:-N/A}"
echo "Start:   $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_distill.py \
    --image-dir       gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --cache-dir       cache/vitl_336/ \
    --clip-pretrained 'ViT-L/14@336px' \
    --output-dir      "${OUTPUT_DIR}" \
    --epochs          30 \
    --batch-size      16 \
    --hidden-dim      128 \
    --d-dino          768 \
    --num-classes     40 \
    --clip-layers     4 8 10 12 \
    --amp

echo ""
echo "============================================================"
echo "Done. Output: ${OUTPUT_DIR}"
echo "End: $(date)"
echo "============================================================"