#!/bin/bash
# Submit baseline fine-tune, then auto-eval on Potsdam / FloodNet / FLAIR / FAST.
# Requires phase1_gsnet_pretrain to have already run (output/gsnet_pretrain/model_final.pth).
#
# Usage:
#   sh scripts/submit_baseline.sh

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

TRAIN_JOB=$(sbatch --parsable jobs-ashie/baseline_finetune.sbatch)
echo "Baseline (train) submitted: job ${TRAIN_JOB}"
echo "  Logs: jobs-ashie/logs/baseline_finetune_${TRAIN_JOB}.out"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} jobs-ashie/eval_baseline.sbatch)
echo "Eval submitted:             job ${EVAL_JOB}  (starts after job ${TRAIN_JOB} succeeds)"
echo "  Logs: jobs-ashie/logs/eval_baseline_${EVAL_JOB}.out"
echo ""
echo "Datasets: Potsdam / FloodNet / FLAIR / FAST"
echo "Results:  output/ashie/baseline/eval/"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel both:   scancel ${TRAIN_JOB} ${EVAL_JOB}"
