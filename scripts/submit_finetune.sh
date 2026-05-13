#!/bin/bash
# Submit Phase 3 — GS-Distill segmentation fine-tune (CLIP backbone).
# Requires both Phase 1 and Phase 2 to have already completed:
#   output/gsnet_pretrain/model_final.pth
#   output/distill/student_best.pth
#
# Usage:
#   sh scripts/submit_finetune.sh                          # run Phase 3 alone
#   sh scripts/submit_finetune.sh --after <phase2_job_id>  # chain after Phase 2

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

if [ "${1:-}" = "--after" ] && [ -n "${2:-}" ]; then
    JOB=$(sbatch --parsable --dependency=afterok:${2} jobs-ashie/phase3_finetune.sbatch)
    echo "Phase 3 submitted: job ${JOB}  (starts after job ${2} succeeds)"
else
    JOB=$(sbatch --parsable jobs-ashie/phase3_finetune.sbatch)
    echo "Phase 3 submitted: job ${JOB}"
fi

echo "  Logs: jobs-ashie/logs/finetune_${JOB}.out"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel:        scancel ${JOB}"
