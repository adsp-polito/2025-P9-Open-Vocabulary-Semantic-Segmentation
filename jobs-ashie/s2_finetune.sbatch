#!/bin/bash
#SBATCH --job-name=s2_finetune
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=jobs-ashie/logs/s2_finetune_%j.out
#SBATCH --error=jobs-ashie/logs/s2_finetune_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/s2

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
if [[ "${S1_SOURCE}" == "distill" ]]; then
    S1_CKPT="output/ashie/clip/finetune_best.pth"
elif [[ "${S1_SOURCE}" == "baseline" ]]; then
    S1_CKPT="output/ashie/baseline/baseline_best.pth"
else
    echo "[ERROR] S1_SOURCE must be 'distill' or 'baseline'"; exit 1
fi
S2_OUTPUT_DIR="output/ashie/s2/${S1_SOURCE}"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"
    exit 1
fi
if [[ ! -f "${S1_CKPT}" ]]; then
    echo "[ERROR] Student-1 checkpoint not found: ${S1_CKPT}"
    exit 1
fi

echo "============================================================"
echo "Student-2 two-stage training on LD50K"
echo "  Teacher:   ${GSNET_WEIGHTS}"
echo "  S1 source: ${S1_SOURCE}"
echo "  Student-1: ${S1_CKPT}"
echo "  Output:    ${S2_OUTPUT_DIR}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_s2.py \
    --gsnet-config          configs/vitl_336_dinov3.yaml \
    --gsnet-weights         "${GSNET_WEIGHTS}" \
    --s1-source             "${S1_SOURCE}" \
    --s1-ckpt               "${S1_CKPT}" \
    --image-dir             gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --label-dir             gs_net/data/datasets/LandDiscover_50K/GT_ID \
    --output-dir            "${S2_OUTPUT_DIR}" \
    --distill-epochs        5 \
    --distill-lr            5e-5 \
    --distill-lambda-seg    0.5 \
    --distill-lambda-logit  1.0 \
    --distill-lambda-feat   0.1 \
    --finetune-epochs       15 \
    --batch-size            1 \
    --grad-accum            2 \
    --clip-layers           8 16 20 23 \
    --lr                    1e-5 \
    --lambda-logit          0.25 \
    --lambda-feat           0.0 \
    --head-mode             aggregator \
    --head-hidden-dim       64 \
    --head-num-layers       1 \
    --num-workers           2 \
    --amp \
    --wandb-project         gs-distill \
    --wandb-run             s2-two-stage-run1

echo ""
echo "============================================================"
echo "Student-2 two-stage training done."
echo "  Best: ${S2_OUTPUT_DIR}/s2_best.pth"
echo "End: $(date)"
echo "============================================================"
