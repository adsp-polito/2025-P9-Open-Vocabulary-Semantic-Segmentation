#!/bin/bash
# Submit both distillation phases with automatic dependency chaining.
# Phase 2 starts only after Phase 1 completes successfully.
#
# Usage:
#   sh scripts/submit_distill.sh               # run both phases
#   sh scripts/submit_distill.sh --phase2-only  # skip phase 1 (checkpoint already exists)

cd "$(dirname "$0")/.."

mkdir -p jobs-ashie/logs

if [ "${1:-}" = "--phase2-only" ]; then
    echo "Skipping Phase 1 — submitting Phase 2 directly."
    JOB2=$(sbatch --parsable jobs-ashie/phase2_distill_online.sbatch)
    echo "Phase 2 submitted: job ${JOB2}"
    echo "  Logs: jobs-ashie/logs/distill_online_${JOB2}.out"
else
    JOB1=$(sbatch --parsable jobs-ashie/phase1_gsnet_pretrain.sbatch)
    echo "Phase 1 submitted: job ${JOB1}"
    echo "  Logs: jobs-ashie/logs/gsnet_pretrain_${JOB1}.out"

    JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} jobs-ashie/phase2_distill_online.sbatch)
    echo "Phase 2 submitted: job ${JOB2}  (starts after job ${JOB1} succeeds)"
    echo "  Logs: jobs-ashie/logs/distill_online_${JOB2}.out"

    echo ""
    echo "Monitor with:  squeue -u \$USER"
    echo "Cancel both:   scancel ${JOB1} ${JOB2}"
fi
