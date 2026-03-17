#!/bin/bash
#SBATCH --job-name=dinov3_selfsup
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/selfsup_train_%j.out
#SBATCH --error=logs/selfsup_train_%j.err

# Create logs directory
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

# Print job info
echo "============================================"
echo "  SLURM JOB INFO"
echo "============================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURM_NODELIST"
echo "  Partition:   $SLURM_JOB_PARTITION"
echo "  GPUs:        $SLURM_GPUS_ON_NODE"
echo "  Start time:  $(date)"
echo "============================================"

# Verify GPU
nvidia-smi

# Run training
cd /nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation
python scripts/self_supervised_train.py

echo ""
echo "Job finished at: $(date)"
