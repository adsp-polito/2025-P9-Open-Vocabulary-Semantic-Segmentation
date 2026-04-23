#!/bin/bash
#SBATCH --job-name=potsdam_varb
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/potsdam_varb_%j.out
#SBATCH --error=logs/potsdam_varb_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
CONFIG_FILE="configs/potsdam_eval_sup5.yaml"
MODEL_CKPT="checkpoints/GSNet/GSNet_sup_partial_unfreeze_5epochs.pth"
RSIB_CKPT="dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
RUN_TAG="run_${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="output/potsdam_diagnosis/variantb_highres_nosw/${RUN_TAG}"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsnet

cd "${PROJECT_ROOT}"
export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT="${RSIB_CKPT}"

echo "============================================"
echo "POTSDAM - VARIANT B ONLY"
echo "  High-res, no sliding window, single prompt"
echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      ${SLURM_NODELIST:-N/A}"
echo "Ckpt:      ${MODEL_CKPT}"
echo "Output:    ${OUTPUT_DIR}"
echo "Start:     $(date)"
echo "============================================"

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "[ERROR] Model checkpoint not found: ${MODEL_CKPT}"; exit 1
fi
if [[ ! -f "${RSIB_CKPT}" ]]; then
  echo "[ERROR] RSIB checkpoint not found: ${RSIB_CKPT}"; exit 1
fi

python3 train_net.py --config "${CONFIG_FILE}" \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.WEIGHTS "${MODEL_CKPT}" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/potsdam.json" \
  DATASETS.TEST \(\"PotsdamSplit\"\,\) \
  TEST.SLIDING_WINDOW "False" \
  MODEL.PROMPT_ENSEMBLE_TYPE "single" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
  INPUT.MIN_SIZE_TEST 1536 \
  INPUT.MAX_SIZE_TEST 4096 \
  | tee "${OUTPUT_DIR}/eval_stdout.log"

# Extract per-class IoU from evaluation results
python3 - "${OUTPUT_DIR}" <<'PY'
import csv, json, os, sys, torch

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

with open(os.path.join(eval_dir, "report.txt"), "w") as f:
    f.write("Potsdam Variant B evaluation summary\n")
    f.write(f"metrics_file: {metrics_path}\n")
    f.write(f"mIoU:  {metrics.get('mIoU', 'NA')}\n")
    f.write(f"fwIoU: {metrics.get('fwIoU', 'NA')}\n")
    f.write(f"mACC:  {metrics.get('mACC', 'NA')}\n")
    f.write(f"pACC:  {metrics.get('pACC', 'NA')}\n")
    f.write("\nPer-class IoU (percent):\n")
    for name, iou in iou_rows:
        f.write(f"  {name}: {iou:.4f}\n")

print(f"[OK] mIoU={metrics.get('mIoU', 'NA')}")
print(f"[OK] per_class_iou.csv written")
print(f"[OK] report.txt written")
PY

echo ""
echo "============================================"
echo "Done. Output: ${OUTPUT_DIR}"
echo "End: $(date)"
echo "============================================"
