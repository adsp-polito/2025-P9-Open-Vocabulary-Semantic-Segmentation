#!/bin/bash
#SBATCH --job-name=eval_finetune
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=jobs-ashie/logs/eval_finetune_%j.out
#SBATCH --error=jobs-ashie/logs/eval_finetune_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/finetune/eval

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
FINETUNE_CKPT="output/ashie/finetune/finetune_best.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"
    echo "        Run phase1_gsnet_pretrain.sbatch first."
    exit 1
fi

if [[ ! -f "${FINETUNE_CKPT}" ]]; then
    echo "[ERROR] Finetune checkpoint not found: ${FINETUNE_CKPT}"
    echo "        Run train_finetune.py first."
    exit 1
fi

echo "============================================================"
echo "Phase 3 — GS-Distill finetune evaluation (ViT-L/14@336px)"
echo "  GSNet:     ${GSNET_WEIGHTS}"
echo "  Finetune:  ${FINETUNE_CKPT}"
echo "  Images:    gs_net/data/datasets/LandDiscover_50K/TE_Image"
echo "  Labels:    gs_net/data/datasets/LandDiscover_50K/TE_Label"
echo "  Output:    output/ashie/finetune/eval"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/eval_finetune.py \
    --gsnet-config  configs/vitl_336_dinov3.yaml \
    --gsnet-weights "${GSNET_WEIGHTS}" \
    --finetune-ckpt "${FINETUNE_CKPT}" \
    --image-dir     gs_net/data/datasets/LandDiscover_50K/TE_Image \
    --label-dir     gs_net/data/datasets/LandDiscover_50K/TE_Label \
    --class-json    datasets/landdiscover.json \
    --output-dir    output/ashie/finetune/eval \
    --batch-size    4 \
    --num-workers   4 \
    --amp

echo ""
echo "============================================================"
echo "Phase 3 done."
echo "  mIoU and per-class results: output/ashie/finetune/eval/report.txt"
echo "  CSV:                        output/ashie/finetune/eval/per_class_iou.csv"
echo "End: $(date)"
echo "============================================================"
