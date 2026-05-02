# Cluster Run Guide (Project + Copilot Reference)

This document is a practical guide for running scripts on the cluster with Slurm.
It is written for both:
- people new to cluster usage
- Copilot/agents that need a reliable run workflow

## 1) Quick Start (5 commands)

```bash
cd /nfs/home/maatouk/multimodal-rag-cir
source .venv/bin/activate
sinfo -s
sacctmgr -n show assoc where user=$USER format=account,partition,qos%30 -P
squeue -u $USER
```

## 2) Understand The Cluster Basics

- `partition`: queue or hardware pool (for example `RTX`, `A100`)
- `account` + `qos`: permissions and scheduling policy tied to your user
- `sbatch`: submit a batch script
- `squeue`: see queued/running jobs
- Slurm log path is controlled by `#SBATCH --output=...`

## 3) Check Available GPU Resources

```bash
sinfo -s
sinfo -o '%P %a %l %D %G %N'
```

Useful interpretation:
- `RTX` partition: RTX cards
- `A100` partition: A100 and possible MIG profiles

## 4) Check Your Access Rights (Important)

```bash
sacctmgr -n show assoc where user=$USER format=account,partition,qos%30 -P
```

If your output does not include a partition, you cannot submit there.

## 5) Safe Pre-Check Before Real Submit

Use test-only mode first.

```bash
# Example RTX check
sbatch --test-only --partition=RTX --gres=gpu:1 --time=00:05:00 --wrap='echo ok'

# Example A100 check
sbatch --test-only --partition=A100 --gres=gpu:1 --time=00:05:00 --wrap='echo ok'
```

If you see:

```text
allocation failure: Invalid account or account/partition combination specified
```

you need admin permission for that account/partition pair.

## 6) Recommended Folder Layout

```text
jobs/                 # sbatch scripts
results/slurm/        # stdout/stderr logs from jobs
```

Create once:

```bash
mkdir -p jobs results/slurm
```

## 7) Standard Slurm Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --partition=RTX
#SBATCH --account=ads
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/home/maatouk/multimodal-rag-cir/results/slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /nfs/home/maatouk/multimodal-rag-cir
source .venv/bin/activate

nvidia-smi || true
python your_script.py
```

## 8) Submit, Monitor, Cancel

Submit:

```bash
sbatch jobs/your_job.sbatch
```

Monitor your jobs:

```bash
squeue -u $USER
```

Tail log:

```bash
tail -f results/slurm/<job-name>-<jobid>.out
```

Cancel a job:

```bash
scancel <jobid>
```

## 9) Copilot Agent Checklist (Runbook)

When an agent runs jobs, follow this exact order:

1. Verify partition availability (`sinfo -s`).
2. Verify user association (`sacctmgr ... assoc ...`).
3. Dry-run with `sbatch --test-only`.
4. Write/update `.sbatch` in `jobs/`.
5. Submit with explicit `--account` and `--qos` when needed.
6. Print job id and log path immediately.
7. Poll `squeue -j <jobid>` and log tail.
8. Report final state as one of: queued, running, completed, failed, blocked-by-permission.

## 10) Project-Specific Notes (LamRA)

- This project uses large multimodal checkpoints.
- A single forward may require high VRAM.
- RTX 11GB can fail with CUDA OOM even with quantization attempts.
- For stable full standalone reranker runs, prefer A100.
- Current full standalone reranker script: jobs/lamra_full_standalone_a100.sbatch

## 11) Known Local Status (Updated)

Based on recent checks:
- Visible partitions include `RTX` and `A100`.
- Current effective user association observed: `ads||interactive,normal`.
- RTX submissions work.
- A100 `--test-only` succeeds with `--account=ads --qos=normal`.
- A100 requests can be blocked by scheduler memory policy when `--mem` is too high; reduce RAM request if pending reason shows `AssocGrpMemLimit`.
- A100 full standalone reranker job expects a MIG profile request (`--gpus=1g.20gb:1`) in current scripts.

## 12) If You Need A100 Access

Ask admins to provide:

1. Allowed `account` for A100
2. Required `qos` for A100
3. Whether A100 should be requested as full GPU or MIG profile

After they update permissions, rerun:

```bash
sbatch --test-only --partition=A100 --time=00:05:00 --wrap='echo ok'
```

If successful, proceed with the real job.

## 13) LamRA Full Standalone Reranker Runs

Use these commands for full validation-set standalone reranker runs:

```bash
# CIRR full standalone reranker
sbatch jobs/lamra_full_standalone_a100.sbatch cirr

# FashionIQ full standalone reranker
sbatch jobs/lamra_full_standalone_a100.sbatch fashioniq
```

Expected output root:

```text
results/lamra_full_a100/
```

Each run folder includes:
- metrics.csv
- metrics.json
- metrics_raw.json
- metrics_structured.csv
- metrics_structured.json
- evaluation_runtime.json
- evaluation_protocol.json
- run_config.json

Latency fields now include:
- latency_seconds
- latency_seconds_per_query
- latency_seconds_per_scored_pair
- FashionIQ per-class latency fields (dress/shirt/toptee)

## 14) Fair Comparison Rules (Speed/Quality)

For latency comparisons across models/pipelines:
- use same GPU type (A100 vs A100)
- keep same candidate pool policy
- keep same seed, dtype, batch settings, and workers
- compare both quality metrics and normalized latency metrics