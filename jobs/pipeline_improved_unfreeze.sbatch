#!/bin/bash
#SBATCH --job-name=pipeline_v2
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pipeline_v2_%j.out
#SBATCH --error=logs/pipeline_v2_%j.err

# Pipeline: Backbone fine-tune (v2: lower LR + L2 reg) → GSNet train → Eval all 4 datasets
#
# Steps:
#   1. finetune_supervised_unfreeze_v2.py (2 warmup + 5 train epochs, LR_BB=1e-5, L2=1e-2)
#   2. GSNet training with fine-tuned backbone (epoch_05), 30K iters
#   3. Eval FloodNet  (sliding window, [1,1] pooling, floodnet.json)
#   4. Eval FAST      (sliding window, [1,1] pooling, fast.json)
#   5. Eval Potsdam   (no SW, 1536/4096, [2,2] pooling, potsdam_improved.json)
#   6. Eval FLAIR     (no SW, 1536/4096, [2,2] pooling, flair_improved.json)
#   7. Summary CSV

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
CONFIG="configs/vitl_336_dinov3.yaml"

EXP_DATE=$(date +%Y%m%d_%H%M%S)
EXP_TAG="sup_unfreeze_v2_${EXP_DATE}"
EXP_DIR="gs_net/third_party/experiments/${EXP_TAG}"
GSNET_OUT="output/${EXP_TAG}"
SUMMARY_CSV="${GSNET_OUT}/eval_summary.csv"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/${EXP_DIR}"
mkdir -p "${PROJECT_ROOT}/${GSNET_OUT}"

cd "${PROJECT_ROOT}"
export DETECTRON2_DATASETS='gs_net/data/datasets'

echo "================================================================"
echo "PIPELINE: sup_unfreeze_v2 (LR_BB=1e-5, L2=1e-2)"
echo "Job ID:   ${SLURM_JOB_ID:-N/A}"
echo "Node:     ${SLURM_NODELIST:-N/A}"
echo "Tag:      ${EXP_TAG}"
echo "Start:    $(date)"
echo "================================================================"

# ================================================================
# STEP 1 — Backbone fine-tuning (v2)
# ================================================================
echo ""
echo "================================================================"
echo "STEP 1: Backbone fine-tuning (v2)"
echo "================================================================"

"${PYTHON}" gs_net/third_party/finetune_supervised_unfreeze_v2.py \
  --output-dir "${PROJECT_ROOT}/${EXP_DIR}"

BACKBONE_CKPT="${EXP_DIR}/backbone_for_gsnet/epoch_17.pth"
if [[ ! -f "${PROJECT_ROOT}/${BACKBONE_CKPT}" ]]; then
  echo "[ERROR] Backbone checkpoint not found: ${BACKBONE_CKPT}"
  exit 1
fi
echo "Backbone checkpoint ready: ${BACKBONE_CKPT}"

# ================================================================
# STEP 2 — GSNet training
# ================================================================
echo ""
echo "================================================================"
echo "STEP 2: GSNet training (30K iters, batch=2)"
echo "================================================================"

export RSIB_CKPT="${BACKBONE_CKPT}"

"${PYTHON}" train_net.py \
  --config "${CONFIG}" \
  --num-gpus 1 \
  --dist-url "auto" \
  OUTPUT_DIR "${GSNET_OUT}" \
  MODEL.SEM_SEG_HEAD.IGNORE_VALUE 0 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 40 \
  MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "datasets/landdiscover.json" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/floodnet.json" \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 30000 \
  TEST.EVAL_PERIOD 0 \
  DATASETS.TRAIN \(\"LandDiscover_50K\"\,\) \
  DATASETS.TEST \(\"FloodNet\"\,\)

MODEL_CKPT="${GSNET_OUT}/model_final.pth"
if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "[ERROR] GSNet model_final.pth not found: ${MODEL_CKPT}"
  exit 1
fi
echo "GSNet training complete: ${MODEL_CKPT}"

# ================================================================
# Helper: extract mIoU from a finished eval dir
# ================================================================
extract_miou() {
  local out_dir="$1"
  "${PYTHON}" - "${out_dir}" <<'PY'
import csv, json, os, sys, torch

eval_dir = sys.argv[1]
candidates = [
    os.path.join(eval_dir, "inference", "sem_seg_evaluation.pth"),
    os.path.join(eval_dir, "sem_seg_evaluation.pth"),
]
metrics_path = next((p for p in candidates if os.path.isfile(p)), None)
if metrics_path is None:
    print(f"[WARN] sem_seg_evaluation.pth not found under {eval_dir}", file=sys.stderr)
    print("nan")
    sys.exit(0)

raw = torch.load(metrics_path, map_location="cpu")
metrics = raw["sem_seg"] if isinstance(raw, dict) and "sem_seg" in raw else raw

with open(os.path.join(eval_dir, "summary_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

iou_rows = sorted(
    [(k.replace("IoU-", ""), float(v)) for k, v in metrics.items() if k.startswith("IoU-")],
    key=lambda x: x[0].lower()
)
with open(os.path.join(eval_dir, "per_class_iou.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class", "iou_percent"])
    for name, iou in iou_rows:
        writer.writerow([name, f"{iou:.4f}"])

miou = metrics.get('mIoU', 'nan')
print(f"mIoU={miou}")
PY
}

# ================================================================
# STEP 3 — Eval FloodNet (original settings: sliding window)
# ================================================================
echo ""
echo "================================================================"
echo "STEP 3: Eval FloodNet"
echo "================================================================"

EVAL_FLOODNET="${GSNET_OUT}/eval/FloodNet"
mkdir -p "${EVAL_FLOODNET}"

"${PYTHON}" train_net.py \
  --config "${CONFIG}" \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR "${EVAL_FLOODNET}" \
  MODEL.WEIGHTS "${MODEL_CKPT}" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/floodnet.json" \
  DATASETS.TEST \(\"FloodNet\"\,\) \
  TEST.SLIDING_WINDOW "True" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  | tee "${EVAL_FLOODNET}/eval_stdout.log"

FLOODNET_MIOU=$(extract_miou "${EVAL_FLOODNET}" | grep "mIoU=" | sed 's/mIoU=//')
echo "FloodNet mIoU: ${FLOODNET_MIOU}"

# ================================================================
# STEP 4 — Eval FAST (original settings: sliding window)
# ================================================================
echo ""
echo "================================================================"
echo "STEP 4: Eval FAST"
echo "================================================================"

EVAL_FAST="${GSNET_OUT}/eval/FAST"
mkdir -p "${EVAL_FAST}"

"${PYTHON}" train_net.py \
  --config "${CONFIG}" \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR "${EVAL_FAST}" \
  MODEL.WEIGHTS "${MODEL_CKPT}" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/fast.json" \
  DATASETS.TEST \(\"FAST\"\,\) \
  TEST.SLIDING_WINDOW "True" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  | tee "${EVAL_FAST}/eval_stdout.log"

FAST_MIOU=$(extract_miou "${EVAL_FAST}" | grep "mIoU=" | sed 's/mIoU=//')
echo "FAST mIoU: ${FAST_MIOU}"

# ================================================================
# STEP 5 — Eval Potsdam (improved: no SW, 1536/4096, [2,2])
# ================================================================
echo ""
echo "================================================================"
echo "STEP 5: Eval Potsdam (improved class names)"
echo "================================================================"

EVAL_POTSDAM="${GSNET_OUT}/eval/Potsdam"
mkdir -p "${EVAL_POTSDAM}"

"${PYTHON}" train_net.py \
  --config "${CONFIG}" \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR "${EVAL_POTSDAM}" \
  MODEL.WEIGHTS "${MODEL_CKPT}" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/potsdam_improved.json" \
  DATASETS.TEST \(\"PotsdamSplit\"\,\) \
  TEST.SLIDING_WINDOW "False" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  INPUT.MIN_SIZE_TEST 1536 \
  INPUT.MAX_SIZE_TEST 4096 \
  | tee "${EVAL_POTSDAM}/eval_stdout.log"

POTSDAM_MIOU=$(extract_miou "${EVAL_POTSDAM}" | grep "mIoU=" | sed 's/mIoU=//')
echo "Potsdam mIoU: ${POTSDAM_MIOU}"

# ================================================================
# STEP 6 — Eval FLAIR (improved: no SW, 1536/4096, [2,2])
# ================================================================
echo ""
echo "================================================================"
echo "STEP 6: Eval FLAIR (improved class names)"
echo "================================================================"

EVAL_FLAIR="${GSNET_OUT}/eval/FLAIR"
mkdir -p "${EVAL_FLAIR}"

"${PYTHON}" train_net.py \
  --config "${CONFIG}" \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR "${EVAL_FLAIR}" \
  MODEL.WEIGHTS "${MODEL_CKPT}" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/flair_improved.json" \
  DATASETS.TEST \(\"FLAIR_test\"\,\) \
  TEST.SLIDING_WINDOW "False" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  INPUT.MIN_SIZE_TEST 1536 \
  INPUT.MAX_SIZE_TEST 4096 \
  | tee "${EVAL_FLAIR}/eval_stdout.log"

FLAIR_MIOU=$(extract_miou "${EVAL_FLAIR}" | grep "mIoU=" | sed 's/mIoU=//')
echo "FLAIR mIoU: ${FLAIR_MIOU}"

# ================================================================
# STEP 7 — Summary
# ================================================================
echo ""
echo "================================================================"
echo "STEP 7: Summary"
echo "================================================================"

{
  echo "config,floodnet_miou,fast_miou,potsdam_miou,flair_miou"
  echo "sup_unfreeze_v2,${FLOODNET_MIOU},${FAST_MIOU},${POTSDAM_MIOU},${FLAIR_MIOU}"
} > "${SUMMARY_CSV}"

echo ""
echo "  FloodNet : ${FLOODNET_MIOU}"
echo "  FAST     : ${FAST_MIOU}"
echo "  Potsdam  : ${POTSDAM_MIOU}"
echo "  FLAIR    : ${FLAIR_MIOU}"
echo ""
echo "Summary written to: ${SUMMARY_CSV}"
echo "End: $(date)"
echo "================================================================"
