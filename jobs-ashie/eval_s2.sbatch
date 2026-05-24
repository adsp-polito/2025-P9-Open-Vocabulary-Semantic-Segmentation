#!/bin/bash
#SBATCH --job-name=eval_s2
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=jobs-ashie/logs/eval_s2_%j.out
#SBATCH --error=jobs-ashie/logs/eval_s2_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/s2/eval

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
S1_SOURCE="${S1_SOURCE:-distill}"  # distill | baseline
S2_CKPT="output/ashie/s2/${S1_SOURCE}/s2_best.pth"
S2_EVAL_DIR="output/ashie/s2/${S1_SOURCE}/eval"

if [[ ! -f "${S2_CKPT}" ]]; then
    echo "[ERROR] Student-2 checkpoint not found: ${S2_CKPT}"
    exit 1
fi

echo "============================================================"
echo "Eval - Student-2 GS-Distill on Potsdam / FloodNet / FLAIR / FAST"
echo "  S1 source:  ${S1_SOURCE}"
echo "  Checkpoint: ${S2_CKPT}"
echo "  Output:     ${S2_EVAL_DIR}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/eval_s2.py \
    --gsnet-config  configs/vitl_336_dinov3.yaml \
    --gsnet-weights "${GSNET_WEIGHTS}" \
    --s2-ckpt       "${S2_CKPT}" \
    --output-dir    "${S2_EVAL_DIR}" \
    --amp

echo ""
echo "============================================================"
echo "Eval done. Results in ${S2_EVAL_DIR}/"
echo "End: $(date)"
echo "============================================================"
