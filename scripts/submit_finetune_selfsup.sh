#!/bin/bash
#SBATCH --job-name=gsnet_tr
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=output/finetune_selfsup/slurm_%j.out
#SBATCH --error=output/finetune_selfsup/slurm_%j.err

set -euo pipefail

cd /nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation
mkdir -p output/finetune_selfsup

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

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

python gs_net/third_party/finetune_selfsup_dino.py

echo ""
echo "Job finished at: $(date)"
