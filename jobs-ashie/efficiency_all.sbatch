#!/bin/bash
#SBATCH --job-name=efficiency_all
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=jobs-ashie/logs/efficiency_all_%j.out
#SBATCH --error=jobs-ashie/logs/efficiency_all_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
cd "${PROJECT_ROOT}"
mkdir -p jobs-ashie/logs output/efficiency

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

export DETECTRON2_DATASETS='gs_net/data/datasets'

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

OLD_GSNET_WEIGHTS="gs_net_base/GSNet_base.pth"
OLD_RSIB_CKPT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation/DinoV1/RSIB.pth"
GSNET_WEIGHTS="output/gsnet_pretrain/model_final.pth"
RSIB_CKPT_PATH='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_07.pth'
CLIP_FINETUNE_CKPT="output/ashie/clip/finetune_best.pth"

if [[ ! -f "${OLD_GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] Old GSNet checkpoint not found: ${OLD_GSNET_WEIGHTS}"; exit 1
fi
if [[ ! -f "${OLD_RSIB_CKPT}" ]]; then
    echo "[ERROR] DinoV1 RSIB checkpoint not found: ${OLD_RSIB_CKPT}"; exit 1
fi
if [[ ! -f "${GSNET_WEIGHTS}" ]]; then
    echo "[ERROR] Improved GSNet checkpoint not found: ${GSNET_WEIGHTS}"; exit 1
fi
if [[ ! -f "${RSIB_CKPT_PATH}" ]]; then
    echo "[ERROR] RSIB checkpoint not found: ${RSIB_CKPT_PATH}"; exit 1
fi
if [[ ! -f "${CLIP_FINETUNE_CKPT}" ]]; then
    echo "[ERROR] GS-Distill finetune checkpoint not found: ${CLIP_FINETUNE_CKPT}"; exit 1
fi

echo "============================================================"
echo "Efficiency profiling — ALL experiments (old_gsnet, improved_gsnet, gs_distill)"
echo "  Old GSNet weights:    ${OLD_GSNET_WEIGHTS}"
echo "  Improved GSNet:       ${GSNET_WEIGHTS}"
echo "  RSIB backbone:        ${RSIB_CKPT_PATH}"
echo "  GS-Distill student:   ${CLIP_FINETUNE_CKPT}"
echo "  Output:               output/efficiency"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date)"
echo "============================================================"

nvidia-smi

# ── Run 1 — old_gsnet (uses a different GSNet config: vitb_384.yaml) ─────────
echo ""
echo "------------------------------------------------------------"
echo "[Run 1/3] old_gsnet  (config: configs/vitb_384.yaml)"
echo "------------------------------------------------------------"
"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments       old_gsnet \
    --old-gsnet-weights "${OLD_GSNET_WEIGHTS}" \
    --old-rsib-ckpt     "${OLD_RSIB_CKPT}" \
    --gsnet-config      configs/vitb_384.yaml \
    --output-dir        output/efficiency

# ── Run 2 — improved_gsnet (config: vitl_336_dinov3.yaml) ────────────────────
export RSIB_CKPT="${RSIB_CKPT_PATH}"
echo ""
echo "------------------------------------------------------------"
echo "[Run 2/3] improved_gsnet  (config: configs/vitl_336_dinov3.yaml)"
echo "------------------------------------------------------------"
"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments         improved_gsnet \
    --gsnet-config        configs/vitl_336_dinov3.yaml \
    --gsnet-weights       "${GSNET_WEIGHTS}" \
    --output-dir          output/efficiency

# ── Run 3 — gs_distill (config: vitl_336_dinov3.yaml, no --gsnet-weights) ───
echo ""
echo "------------------------------------------------------------"
echo "[Run 3/3] gs_distill  (config: configs/vitl_336_dinov3.yaml)"
echo "------------------------------------------------------------"
"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments         gs_distill \
    --gsnet-config        configs/vitl_336_dinov3.yaml \
    --clip-finetune-ckpt  "${CLIP_FINETUNE_CKPT}" \
    --output-dir          output/efficiency

echo ""
echo "============================================================"
echo "float32 profiling done. Combined CSV: output/efficiency/efficiency_combined.csv"
echo "============================================================"

# ── fp16 (AMP) pass — same experiments, separate output dir ──────────────────
mkdir -p output/efficiency_fp16

echo ""
echo "------------------------------------------------------------"
echo "[fp16 pass 1/3] old_gsnet  (config: configs/vitb_384.yaml)"
echo "------------------------------------------------------------"
"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments       old_gsnet \
    --old-gsnet-weights "${OLD_GSNET_WEIGHTS}" \
    --old-rsib-ckpt     "${OLD_RSIB_CKPT}" \
    --gsnet-config      configs/vitb_384.yaml \
    --amp \
    --output-dir        output/efficiency_fp16

echo ""
echo "------------------------------------------------------------"
echo "[fp16 pass 2/3] improved_gsnet  (config: configs/vitl_336_dinov3.yaml)"
echo "------------------------------------------------------------"
"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments         improved_gsnet \
    --gsnet-config        configs/vitl_336_dinov3.yaml \
    --gsnet-weights       "${GSNET_WEIGHTS}" \
    --amp \
    --output-dir          output/efficiency_fp16

echo ""
echo "------------------------------------------------------------"
echo "[fp16 pass 3/3] gs_distill  (config: configs/vitl_336_dinov3.yaml)"
echo "------------------------------------------------------------"
"${PYTHON}" scripts/efficiency_metrics.py \
    --experiments         gs_distill \
    --gsnet-config        configs/vitl_336_dinov3.yaml \
    --clip-finetune-ckpt  "${CLIP_FINETUNE_CKPT}" \
    --amp \
    --output-dir          output/efficiency_fp16

echo ""
echo "============================================================"
echo "All profiling done."
echo "  float32: output/efficiency/efficiency_combined.csv"
echo "  fp16:    output/efficiency_fp16/efficiency_combined.csv"
echo "End: $(date)"
echo "============================================================"
