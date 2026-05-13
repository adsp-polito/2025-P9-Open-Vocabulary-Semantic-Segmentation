#!/bin/bash
# Submit Phase 3 — CLIP GS-Distill segmentation fine-tune, then auto-eval on
# Potsdam / FloodNet / FLAIR / FAST immediately after training completes.
#
# Usage:
#   sh scripts/submit_finetune.sh                          # run Phase 3 + eval
#   sh scripts/submit_finetune.sh --after <phase2_job_id>  # chain after Phase 2

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

if [ "${1:-}" = "--after" ] && [ -n "${2:-}" ]; then
    TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${2} jobs-ashie/phase3_finetune.sbatch)
    echo "Phase 3 (finetune) submitted: job ${TRAIN_JOB}  (starts after job ${2} succeeds)"
else
    TRAIN_JOB=$(sbatch --parsable jobs-ashie/phase3_finetune.sbatch)
    echo "Phase 3 (finetune) submitted: job ${TRAIN_JOB}"
fi
echo "  Logs: jobs-ashie/logs/finetune_${TRAIN_JOB}.out"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} jobs-ashie/eval_clip.sbatch)
echo "Eval submitted:     job ${EVAL_JOB}  (starts after job ${TRAIN_JOB} succeeds)"
echo "  Logs: jobs-ashie/logs/eval_clip_${EVAL_JOB}.out"
echo ""
echo "Datasets: Potsdam / FloodNet / FLAIR / FAST"
echo "Results:  output/ashie/finetune/eval/"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel both:   scancel ${TRAIN_JOB} ${EVAL_JOB}"
