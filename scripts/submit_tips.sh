#!/bin/bash
# Submit the TIPSv2 distill + finetune job.
# Requires phase1_gsnet_pretrain to have already run (output/gsnet_pretrain/model_final.pth).
#
# Usage:
#   sh scripts/submit_tips.sh

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

JOB=$(sbatch --parsable jobs-ashie/tips_distill_finetune.sbatch)
echo "TIPSv2 submitted: job ${JOB}"
echo "  Logs: jobs-ashie/logs/tips_distill_ft_${JOB}.out"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel:        scancel ${JOB}"
