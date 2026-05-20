#!/bin/bash
#SBATCH --job-name=eval_clip
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=jobs-ashie/logs/eval_clip_%j.out
#SBATCH --error=jobs-ashie/logs/eval_clip_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/clip/eval

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
FINETUNE_CKPT="output/ashie/clip/finetune_best.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"; exit 1
fi
if [[ ! -f "${FINETUNE_CKPT}" ]]; then
    echo "[ERROR] Finetune checkpoint not found: ${FINETUNE_CKPT}"; exit 1
fi

echo "============================================================"
echo "Eval — CLIP GS-Distill on Potsdam / FloodNet / FLAIR / FAST"
echo "  Checkpoint: ${FINETUNE_CKPT}"
echo "  Output:     output/ashie/clip/eval"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/eval_clip.py \
    --gsnet-config  configs/vitl_336_dinov3.yaml \
    --gsnet-weights "${GSNET_WEIGHTS}" \
    --finetune-ckpt "${FINETUNE_CKPT}" \
    --output-dir    output/ashie/clip/eval \
    --amp

echo ""
echo "============================================================"
echo "Eval done. Results in output/ashie/clip/eval/"
echo "End: $(date)"
echo "============================================================"
