#!/bin/bash
# Submit the baseline fine-tune job.
# Requires phase1_gsnet_pretrain to have already run (output/gsnet_pretrain/model_final.pth).
#
# Usage:
#   sh scripts/submit_baseline.sh

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

JOB=$(sbatch --parsable jobs-ashie/baseline_finetune.sbatch)
echo "Baseline submitted: job ${JOB}"
echo "  Logs: jobs-ashie/logs/baseline_finetune_${JOB}.out"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel:        scancel ${JOB}"
