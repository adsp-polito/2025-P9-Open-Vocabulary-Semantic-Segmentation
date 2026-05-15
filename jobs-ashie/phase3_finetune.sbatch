#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs-ashie/logs/finetune_%j.out
#SBATCH --error=jobs-ashie/logs/finetune_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/finetune

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'

# Load secrets (WANDB_API_KEY etc.) — .env is gitignored
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

GSNET_WEIGHTS="output/gsnet_pretrain/model_final.pth"
STUDENT_CKPT="output/distill/student_best.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"
    echo "        Run phase1_gsnet_pretrain.sbatch first."
    exit 1
fi

if [[ ! -f "${STUDENT_CKPT}" ]]; then
    echo "[ERROR] Student checkpoint not found: ${STUDENT_CKPT}"
    echo "        Run phase2_distill_online.sbatch first."
    exit 1
fi

echo "============================================================"
echo "Phase 3 — GS-Distill fine-tuning on LD50K (ViT-L/14@336px)"
echo "  GSNet:    ${GSNET_WEIGHTS}"
echo "  Student:  ${STUDENT_CKPT}"
echo "  Images:   gs_net/data/datasets/LandDiscover_50K/TR_Image"
echo "  Labels:   gs_net/data/datasets/LandDiscover_50K/GT_ID"
echo "  Output:   output/ashie/finetune"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_finetune.py \
    --gsnet-config          configs/vitl_336_dinov3.yaml \
    --gsnet-weights         "${GSNET_WEIGHTS}" \
    --student-ckpt          "${STUDENT_CKPT}" \
    --image-dir             gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --label-dir             gs_net/data/datasets/LandDiscover_50K/GT_ID \
    --output-dir            output/ashie/finetune \
    --epochs                15 \
    --batch-size            2 \
    --lr                    1e-5 \
    --warmup-decoder-epochs 3 \
    --num-workers           2 \
    --amp \
    --wandb-project         gs-distill \
    --wandb-run             finetune-vitl336-run1

echo ""
echo "============================================================"
echo "Phase 3 done. Best checkpoint: output/ashie/finetune/finetune_best.pth"
echo "End: $(date)"
echo "============================================================"
