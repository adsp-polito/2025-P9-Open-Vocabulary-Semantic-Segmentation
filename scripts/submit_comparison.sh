#!/bin/bash
#SBATCH --job-name=gsnet_cmp
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/comparison_%j.out
#SBATCH --error=logs/comparison_%j.err

# ============================================================
# GSNet Comparison: Self-Supervised DINOv3 vs Baseline
# Replicates Table 2 experiment:
#   - GSNet trained 5K iterations, Attention mode for CLIP & DINOv3
#   - FloodNet evaluation with sliding window
#   - Varying DINOv3 fine-tuning epochs: 0, 5, 10, 15
# ============================================================

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

cd /nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation

export DETECTRON2_DATASETS='gs_net/data/datasets'

echo "============================================"
echo "  SLURM JOB INFO"
echo "============================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURM_NODELIST"
echo "  Partition:   $SLURM_JOB_PARTITION"
echo "  GPUs:        $SLURM_GPUS_ON_NODE"
echo "  Start time:  $(date)"
echo "============================================"
nvidia-smi

# ---- helper: run one training + eval ----------------------------
run_one() {
  local run_label="$1"
  local rsib_ckpt="$2"
  local output_dir="$3"

  echo ""
  echo "========================================================"
  echo "  ${run_label}"
  echo "========================================================"
  echo "  Start: $(date)"

  export RSIB_CKPT="${rsib_ckpt}"

  # Training (5K iterations, attention mode for CLIP & DINOv3)
  python3 train_net.py --config configs/vitb_384.yaml \
    --num-gpus 1 \
    --dist-url "auto" \
    OUTPUT_DIR "${output_dir}" \
    MODEL.SEM_SEG_HEAD.DINO_FINETUNE "attention" \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.MAX_ITER 5000 \
    TEST.EVAL_PERIOD 5000

  # Eval (sliding window on FloodNet)
  python3 train_net.py --config configs/vitb_384.yaml \
    --num-gpus 1 \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "${output_dir}/eval/FloodNet" \
    MODEL.WEIGHTS "${output_dir}/model_final.pth" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]"

  echo "  ${run_label} done: $(date)"
}

# ============================================================
# Run 1: Baseline (0 epochs, original pretrained DINOv3)
# ============================================================
run_one "RUN 1/4: Baseline (0 self-sup epochs)" \
  "dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \
  "output/comparison/selfsup_epoch0"

# ============================================================
# Run 2: Self-supervised 5 epochs
# ============================================================
run_one "RUN 2/4: Self-supervised DINOv3 (5 epochs)" \
  "checkpoints/dinov3_selfsup_epoch_5.pth" \
  "output/comparison/selfsup_epoch5"

# ============================================================
# Run 3: Self-supervised 10 epochs
# ============================================================
run_one "RUN 3/4: Self-supervised DINOv3 (10 epochs)" \
  "checkpoints/dinov3_selfsup_epoch_10.pth" \
  "output/comparison/selfsup_epoch10"

# ============================================================
# Run 4: Self-supervised 15 epochs
# ============================================================
run_one "RUN 4/4: Self-supervised DINOv3 (15 epochs)" \
  "checkpoints/dinov3_selfsup_epoch_15.pth" \
  "output/comparison/selfsup_epoch15"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================"
echo "  ALL 4 RUNS COMPLETE"
echo "========================================================"
echo "  Finished at: $(date)"
echo ""
echo "  Results (check mIoU in each eval log):"
echo "    Epoch 0:  output/comparison/selfsup_epoch0/eval/FloodNet/"
echo "    Epoch 5:  output/comparison/selfsup_epoch5/eval/FloodNet/"
echo "    Epoch 10: output/comparison/selfsup_epoch10/eval/FloodNet/"
echo "    Epoch 15: output/comparison/selfsup_epoch15/eval/FloodNet/"
echo "========================================================"
