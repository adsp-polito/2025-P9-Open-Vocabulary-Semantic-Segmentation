#!/bin/bash
#SBATCH --job-name=efficiency_old_gsnet
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=jobs-ashie/logs/efficiency_old_gsnet_%j.out
#SBATCH --error=jobs-ashie/logs/efficiency_old_gsnet_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/efficiency

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

OLD_GSNET_WEIGHTS="dinov3/vitl16-sat493m/RSIB.pth"

if [[ ! -f "${OLD_GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] Old GSNet checkpoint not found: ${OLD_GSNET_WEIGHTS}"; exit 1
fi

echo "============================================================"
echo "Efficiency profiling — Exp 1: old_gsnet"
echo "  Checkpoint: ${OLD_GSNET_WEIGHTS}"
echo "  Output:     output/efficiency"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments      old_gsnet \
    --old-gsnet-weights "${OLD_GSNET_WEIGHTS}" \
    --gsnet-config      configs/vitb_384.yaml \
    --output-dir        output/efficiency

echo ""
echo "============================================================"
echo "Profiling done. Results in output/efficiency/"
echo "End: $(date)"
echo "============================================================"
