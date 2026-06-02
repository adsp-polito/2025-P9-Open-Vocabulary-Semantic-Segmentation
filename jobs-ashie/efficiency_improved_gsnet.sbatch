#!/bin/bash
#SBATCH --job-name=efficiency_improved_gsnet
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=jobs-ashie/logs/efficiency_improved_gsnet_%j.out
#SBATCH --error=jobs-ashie/logs/efficiency_improved_gsnet_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/efficiency

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_17.pth'

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

GSNET_WEIGHTS="output/training_clip_attention_freeze_17/model_final.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] Improved GSNet checkpoint not found: ${GSNET_WEIGHTS}"; exit 1
fi
if [[ ! -f "${RSIB_CKPT}" ]]; then
    echo "[ERROR] RSIB checkpoint not found: ${RSIB_CKPT}"; exit 1
fi

echo "============================================================"
echo "Efficiency profiling — Exp 2: improved_gsnet"
echo "  Checkpoint: ${GSNET_WEIGHTS}"
echo "  RSIB:       ${RSIB_CKPT}"
echo "  Output:     output/efficiency"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments   improved_gsnet \
    --gsnet-config  configs/vitl_336_dinov3.yaml \
    --gsnet-weights "${GSNET_WEIGHTS}" \
    --output-dir    output/efficiency

echo ""
echo "============================================================"
echo "Profiling done. Results in output/efficiency/"
echo "End: $(date)"
echo "============================================================"
