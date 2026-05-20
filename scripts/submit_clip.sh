#!/bin/bash
# Submit CLIP distill+finetune, then auto-eval on Potsdam / FloodNet / FLAIR / FAST.
#
# Usage:
#   sh scripts/submit_clip.sh               # Phase 1 (GSNet pretrain) → Phase 2+3 (distill+finetune) → eval
#   sh scripts/submit_clip.sh --skip-phase1  # skip Phase 1 (checkpoint already exists) → Phase 2+3 → eval

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

if [ "${1:-}" = "--skip-phase1" ]; then
    echo "Skipping Phase 1 — submitting CLIP distill+finetune directly."
    TRAIN_JOB=$(sbatch --parsable jobs-ashie/clip_distill_finetune.sbatch)
    echo "CLIP (distill+finetune) submitted: job ${TRAIN_JOB}"
    echo "  Logs: jobs-ashie/logs/clip_distill_ft_${TRAIN_JOB}.out"
else
    PHASE1_JOB=$(sbatch --parsable jobs-ashie/phase1_gsnet_pretrain.sbatch)
    echo "Phase 1 (GSNet pretrain) submitted: job ${PHASE1_JOB}"
    echo "  Logs: jobs-ashie/logs/gsnet_pretrain_${PHASE1_JOB}.out"

    TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PHASE1_JOB} jobs-ashie/clip_distill_finetune.sbatch)
    echo "CLIP (distill+finetune) submitted:  job ${TRAIN_JOB}  (starts after job ${PHASE1_JOB} succeeds)"
    echo "  Logs: jobs-ashie/logs/clip_distill_ft_${TRAIN_JOB}.out"
fi

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} jobs-ashie/eval_clip.sbatch)
echo "Eval submitted:                     job ${EVAL_JOB}  (starts after job ${TRAIN_JOB} succeeds)"
echo "  Logs: jobs-ashie/logs/eval_clip_${EVAL_JOB}.out"
echo ""
echo "Datasets: Potsdam / FloodNet / FLAIR / FAST"
echo "Results:  output/ashie/clip/eval/"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel ${TRAIN_JOB} ${EVAL_JOB}"
