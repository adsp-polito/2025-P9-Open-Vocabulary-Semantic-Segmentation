#!/bin/bash
#SBATCH --job-name=s2_scratch
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs-ashie/logs/s2_scratch_%j.out
#SBATCH --error=jobs-ashie/logs/s2_scratch_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/s2/scratch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

GSNET_WEIGHTS="output/gsnet_pretrain/model_final.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"
    exit 1
fi

echo "============================================================"
echo "Student-2 Scratch: random init, CE-only (ablation baseline)"
echo "  GSNet weights:  ${GSNET_WEIGHTS}"
echo "  Decoder config: hidden_dim=64  num_layers=2  dec_dims=32 8"
echo "  Epochs:         20"
echo "  Loss:           CE only (no distillation, no cached logits)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

"${PYTHON}" scripts/train_s2_scratch.py \
    --gsnet-config      configs/vitl_336_dinov3.yaml \
    --gsnet-weights     "${GSNET_WEIGHTS}" \
    --image-dir         gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --label-dir         gs_net/data/datasets/LandDiscover_50K/GT_ID \
    --class-json        datasets/landdiscover.json \
    --output-dir        output/ashie/s2/scratch \
    --epochs            20 \
    --batch-size        1 \
    --grad-accum        4 \
    --lr-decoder        1e-4 \
    --lr-branches       1e-5 \
    --weight-decay      1e-4 \
    --head-hidden-dim   64 \
    --head-num-layers   2 \
    --head-window-size  4 \
    --head-dec-dims     32 8 \
    --clip-layers       8 16 20 23 \
    --clip-skip-indices 7 15 \
    --num-workers       2 \
    --amp

echo ""
echo "============================================================"
echo "Scratch done."
echo "  Best checkpoint: output/ashie/s2/scratch/s2_scratch_best.pth"
echo "End: $(date)"
echo "============================================================"
