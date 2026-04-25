#!/bin/bash
#SBATCH --job-name=gsnet_potsdam_sup5
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/potsdam_sup5_%j.out
#SBATCH --error=logs/potsdam_sup5_%j.err

set -euo pipefail

# ---------------------------
# Paths and fixed inputs
# ---------------------------
PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
CONFIG_FILE="configs/potsdam_eval_sup5.yaml"
MODEL_CKPT="checkpoints/GSNet/GSNet_sup_partial_unfreeze_5epochs.pth"
RSIB_CKPT="dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
RUN_TAG="run_${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="output/potsdam_diagnosis/sup_partial_unfreeze_5epochs/${RUN_TAG}"

# Variant A: reproduce legacy settings used in previous evaluation flow
LEGACY_OUT="${OUTPUT_ROOT}/variant_a_legacy_sw640_promptsingle"

# Variant B: proposed high-detail settings for Potsdam diagnosis
# NOTE: Keep prompt mode as 'single' because this checkpoint is not compatible
# with multi-template prompt channels at inference time.
PROPOSED_OUT="${OUTPUT_ROOT}/variant_b_highres_nosw_promptsingle"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/${LEGACY_OUT}"
mkdir -p "${PROJECT_ROOT}/${PROPOSED_OUT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

cd "${PROJECT_ROOT}"
export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT="${RSIB_CKPT}"

echo "============================================"
echo "POTSDAM DIAGNOSIS JOB"
echo "============================================"
echo "Job ID:           ${SLURM_JOB_ID:-N/A}"
echo "Node:             ${SLURM_NODELIST:-N/A}"
echo "Checkpoint:       ${MODEL_CKPT}"
echo "Config:           ${CONFIG_FILE}"
echo "RSIB init ckpt:   ${RSIB_CKPT}"
echo "Start time:       $(date)"
echo "============================================"

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "[ERROR] Model checkpoint not found: ${MODEL_CKPT}"
  exit 1
fi

if [[ ! -f "${RSIB_CKPT}" ]]; then
  echo "[ERROR] RSIB checkpoint not found: ${RSIB_CKPT}"
  exit 1
fi

extract_per_class_iou() {
  local eval_dir="$1"
  python3 - "$eval_dir" <<'PY'
import csv
import json
import os
import sys
import torch

eval_dir = sys.argv[1]
candidates = [
    os.path.join(eval_dir, "inference", "sem_seg_evaluation.pth"),
    os.path.join(eval_dir, "sem_seg_evaluation.pth"),
]
metrics_path = next((p for p in candidates if os.path.isfile(p)), None)

if metrics_path is None:
    print(f"[WARN] sem_seg_evaluation.pth not found under {eval_dir}")
    sys.exit(0)

raw = torch.load(metrics_path, map_location="cpu")
if isinstance(raw, dict) and "sem_seg" in raw:
    metrics = raw["sem_seg"]
elif isinstance(raw, dict):
    metrics = raw
else:
    print(f"[WARN] Unexpected metrics format in {metrics_path}: {type(raw)}")
    sys.exit(0)

summary_json = os.path.join(eval_dir, "summary_metrics.json")
with open(summary_json, "w") as f:
    json.dump(metrics, f, indent=2)

iou_rows = []
for key, value in metrics.items():
    if key.startswith("IoU-"):
        iou_rows.append((key.replace("IoU-", ""), float(value)))

iou_rows.sort(key=lambda x: x[0].lower())
per_class_csv = os.path.join(eval_dir, "per_class_iou.csv")
with open(per_class_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class", "iou_percent"])
    for name, iou in iou_rows:
        writer.writerow([name, f"{iou:.4f}"])

report_txt = os.path.join(eval_dir, "report.txt")
with open(report_txt, "w") as f:
    f.write("Potsdam evaluation summary\n")
    f.write(f"metrics_file: {metrics_path}\n")
    f.write(f"mIoU: {metrics.get('mIoU', 'NA')}\n")
    f.write(f"fwIoU: {metrics.get('fwIoU', 'NA')}\n")
    f.write(f"mACC: {metrics.get('mACC', 'NA')}\n")
    f.write(f"pACC: {metrics.get('pACC', 'NA')}\n")
    f.write("\nPer-class IoU (percent):\n")
    for name, iou in iou_rows:
        f.write(f"{name}: {iou:.4f}\n")

print(f"[OK] Wrote {summary_json}")
print(f"[OK] Wrote {per_class_csv}")
print(f"[OK] Wrote {report_txt}")
print(f"[OK] mIoU={metrics.get('mIoU', 'NA')}")
PY
}

run_eval() {
  local tag="$1"
  local out_dir="$2"
  shift 2

  echo ""
  echo "========================================================"
  echo "Running ${tag}"
  echo "Output: ${out_dir}"
  echo "========================================================"

  python3 train_net.py --config "${CONFIG_FILE}" \
    --num-gpus 1 \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "${out_dir}" \
    MODEL.WEIGHTS "${MODEL_CKPT}" \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/potsdam.json" \
    DATASETS.TEST \(\"PotsdamSplit\"\,\) \
    "$@" | tee "${out_dir}/eval_stdout.log"

  extract_per_class_iou "${out_dir}"
}

# --------------------------------------------------------
# Variant A: Legacy settings (for direct comparison)
# --------------------------------------------------------
run_eval \
  "Variant A - legacy (sliding-window + single prompt)" \
  "${LEGACY_OUT}" \
  TEST.SLIDING_WINDOW "True" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  INPUT.MIN_SIZE_TEST 640 \
  INPUT.MAX_SIZE_TEST 2560

# --------------------------------------------------------
# Variant B: Proposed settings (high-detail diagnosis)
# --------------------------------------------------------
run_eval \
  "Variant B - proposed (no sliding + higher test scale + single prompt)" \
  "${PROPOSED_OUT}" \
  TEST.SLIDING_WINDOW "False" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
  INPUT.MIN_SIZE_TEST 1536 \
  INPUT.MAX_SIZE_TEST 4096

echo ""
echo "========================================================"
echo "Done. Key outputs:"
echo "  ${LEGACY_OUT}/summary_metrics.json"
echo "  ${LEGACY_OUT}/per_class_iou.csv"
echo "  ${PROPOSED_OUT}/summary_metrics.json"
echo "  ${PROPOSED_OUT}/per_class_iou.csv"
echo "End time: $(date)"
echo "========================================================"
