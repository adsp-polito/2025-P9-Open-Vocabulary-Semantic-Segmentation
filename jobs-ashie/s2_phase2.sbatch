#!/bin/bash
#SBATCH --job-name=s2_phase2
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=jobs-ashie/logs/s2_phase2_%j.out
#SBATCH --error=jobs-ashie/logs/s2_phase2_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/ashie/s2/phase2

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
PHASE1_CKPT="output/ashie/s2/phase1/s2_phase1_best.pth"

if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] GSNet checkpoint not found: ${GSNET_WEIGHTS}"; exit 1
fi
if [[ ! -f "${PHASE1_CKPT}" ]]; then
    echo "[ERROR] Phase 1 checkpoint not found: ${PHASE1_CKPT}"; exit 1
fi

echo "============================================================"
echo "Student-2 Phase 2: joint fine-tuning (decoder + branches)"
echo "  Phase 1 ckpt: ${PHASE1_CKPT}"
echo "  Loss:         0.5 x KL-distill + 0.5 x CE"
echo "  LR decoder:   1e-4   LR branches: 1e-5"
echo "  Epochs:       20"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi || true

"${PYTHON}" scripts/train_s2_phase2.py \
    --gsnet-config        configs/vitl_336_dinov3.yaml \
    --gsnet-weights       "${GSNET_WEIGHTS}" \
    --phase1-ckpt         "${PHASE1_CKPT}" \
    --image-dir           gs_net/data/datasets/LandDiscover_50K/TR_Image \
    --cache-dir           cache_logits \
    --label-dir           gs_net/data/datasets/LandDiscover_50K/GT_ID \
    --class-json          datasets/landdiscover.json \
    --output-dir          output/ashie/s2/phase2 \
    --epochs              20 \
    --batch-size          1 \
    --grad-accum          4 \
    --lr-decoder          1e-4 \
    --lr-branches         1e-5 \
    --weight-decay        1e-4 \
    --tau                 4.0 \
    --lam-distill         0.5 \
    --lam-ce              0.5 \
    --head-hidden-dim     64 \
    --head-num-layers     2 \
    --head-window-size    4 \
    --head-dec-dims       32 8 \
    --clip-layers         8 16 20 23 \
    --clip-skip-indices   7 15 \
    --num-workers         2 \
    --amp

echo ""
echo "============================================================"
echo "Phase 2 done."
echo "  Checkpoint: output/ashie/s2/phase2/s2_phase2_best.pth"
echo "End: $(date)"
echo "============================================================"
