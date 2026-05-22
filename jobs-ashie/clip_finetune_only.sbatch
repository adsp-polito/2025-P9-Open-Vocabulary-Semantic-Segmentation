#!/bin/bash
#SBATCH --job-name=clip_finetune
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs-ashie/logs/clip_finetune_%j.out
#SBATCH --error=jobs-ashie/logs/clip_finetune_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/clip

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

GSNET_WEIGHTS="output/gsnet_pretrain/model_final.pth"
DISTILL_CKPT="output/ashie/clip/student_distill_best.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"; exit 1
fi
if [[ ! -f "${DISTILL_CKPT}" ]]; then
    echo "[ERROR] Distill checkpoint not found: ${DISTILL_CKPT}"; exit 1
fi

echo "============================================================"
echo "CLIP — Phase 3 only (finetune on LD50K from distill ckpt)"
echo "  Distill ckpt: ${DISTILL_CKPT}"
echo "  GSNet:        ${GSNET_WEIGHTS}"
echo "  Output:       output/ashie/clip"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_clip.py \
    --gsnet-config          configs/vitl_336_dinov3.yaml \
    --gsnet-weights         "${GSNET_WEIGHTS}" \
    --image-dir             gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --label-dir             gs_net/data/datasets/LandDiscover_50K/GT_ID \
    --output-dir            output/ashie/clip \
    --skip-distill \
    --distill-ckpt          "${DISTILL_CKPT}" \
    --finetune-epochs       15 \
    --batch-size            1 \
    --grad-accum            2 \
    --clip-layers           8 16 20 23 \
    --lr                    1e-5 \
    --num-workers           2 \
    --amp \
    --wandb-project         gs-distill \
    --wandb-run             clip-vitl336-finetune-only

echo ""
echo "============================================================"
echo "Finetune done. Best checkpoint: output/ashie/clip/finetune_best.pth"
echo "End: $(date)"
echo "============================================================"
