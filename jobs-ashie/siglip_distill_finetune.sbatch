#!/bin/bash
#SBATCH --job-name=siglip_distill_ft
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=jobs-ashie/logs/siglip_distill_ft_%j.out
#SBATCH --error=jobs-ashie/logs/siglip_distill_ft_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/siglip

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

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"
    echo "        Run phase1_gsnet_pretrain.sbatch first."
    exit 1
fi

echo "============================================================"
echo "SigLIP — Phase 2 (distill) + Phase 3 (finetune) on LD50K"
echo "  Backbone:  google/siglip-base-patch16-384"
echo "  GSNet:     ${GSNET_WEIGHTS}"
echo "  Images:    gs_net/data/datasets/LandDiscover_50K/TR_Image"
echo "  Labels:    gs_net/data/datasets/LandDiscover_50K/GT_ID"
echo "  Output:    output/ashie/siglip"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_siglip.py \
    --gsnet-config          configs/vitl_336_dinov3.yaml \
    --gsnet-weights         "${GSNET_WEIGHTS}" \
    --image-dir             gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --label-dir             gs_net/data/datasets/LandDiscover_50K/GT_ID \
    --output-dir            output/ashie/siglip \
    --distill-epochs        30 \
    --finetune-epochs       15 \
    --batch-size            1 \
    --grad-accum            2 \
    --lr                    1e-5 \
    --num-workers           2 \
    --amp \
    --wandb-project         gs-distill \
    --wandb-run             siglip-vitb384-run1

echo ""
echo "============================================================"
echo "SigLIP job done."
echo "  Distill best:  output/ashie/siglip/student_distill_best.pth"
echo "  Finetune best: output/ashie/siglip/finetune_best.pth"
echo "End: $(date)"
echo "============================================================"
