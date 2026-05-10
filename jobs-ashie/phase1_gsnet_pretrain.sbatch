#!/bin/bash
#SBATCH --job-name=gsnet_pretrain
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs-ashie/logs/gsnet_pretrain_%j.out
#SBATCH --error=jobs-ashie/logs/gsnet_pretrain_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/gsnet_pretrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'

echo "============================================================"
echo "Phase 1 — GSNet pre-training (CLIP attention, DINOv3 frozen)"
echo "  Config:  configs/vitl_336_dinov3.yaml"
echo "  Output:  output/gsnet_pretrain"
echo "  GPUs:    ${SLURM_GPUS_ON_NODE:-N/A}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

sh scripts/train.sh \
    configs/vitl_336_dinov3.yaml \
    1 \
    output/gsnet_pretrain \
    MODEL.SEM_SEG_HEAD.DINO_FINETUNE frozen

echo ""
echo "============================================================"
echo "Phase 1 done. Checkpoint: output/gsnet_pretrain/model_final.pth"
echo "End: $(date)"
echo "============================================================"
