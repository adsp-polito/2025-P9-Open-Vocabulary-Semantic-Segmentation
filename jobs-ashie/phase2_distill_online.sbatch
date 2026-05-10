#!/bin/bash
#SBATCH --job-name=distill_online
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=jobs-ashie/logs/distill_online_%j.out
#SBATCH --error=jobs-ashie/logs/distill_online_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/distill

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'

GSNET_WEIGHTS="output/gsnet_pretrain/model_final.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"
    echo "        Run phase1_gsnet_pretrain.sbatch first."
    exit 1
fi

echo "============================================================"
echo "Phase 2 — Online student distillation (ViT-L/14@336px)"
echo "  Teacher: ${GSNET_WEIGHTS}"
echo "  Images:  gs_net/data/datasets/LandDiscover_50K/TR_Image"
echo "  Output:  output/distill"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_distill_online.py \
    --gsnet-config  configs/vitl_336_dinov3.yaml \
    --gsnet-weights "${GSNET_WEIGHTS}" \
    --image-dir     gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --output-dir    output/distill \
    --epochs        30 \
    --batch-size    8 \
    --lr            1e-4 \
    --num-workers   4 \
    --amp \
    --wandb-project gs-distill \
    --wandb-run     vitl336-run1

echo ""
echo "============================================================"
echo "Phase 2 done. Best checkpoint: output/distill/student_best.pth"
echo "End: $(date)"
echo "============================================================"
