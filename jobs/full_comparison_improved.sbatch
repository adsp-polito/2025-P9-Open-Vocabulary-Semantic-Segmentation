#!/bin/bash
#SBATCH --job-name=full_comparison
#SBATCH --account=intrn
#SBATCH --partition=RTX
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/full_comparison_%j.out
#SBATCH --error=logs/full_comparison_%j.err

# Re-evaluates all 9 GSNet configurations on Potsdam and FLAIR using the
# improved CLIP-friendly class names (datasets/potsdam_improved.json and
# datasets/flair_improved.json).  Results are written per-config and also
# aggregated into a single summary CSV at the end.
#
# Inference settings match what gave the best Potsdam result:
#   no sliding window, 1536/4096 resolution, POOLING_SIZES [2,2]
#
# Configurations tested (label → model_final.pth):
#   sup_unfreeze_5ep   output/training_clip_attention_freeze/model_final.pth
#   sup_unfreeze_10ep  output/training_clip_attention_freeze_12/model_final.pth
#   sup_unfreeze_15ep  output/training_clip_attention_freeze_17/model_final.pth
#   lora_5ep           output/training_clip_attention_freeze_lora_07/model_final.pth
#   lora_10ep          output/training_clip_attention_freeze_lora_12/model_final.pth
#   lora_15ep          output/training_clip_attention_freeze_lora_17/model_final.pth
#   selfsup_5ep        output/training_clip_attention_self_07/model_final.pth
#   selfsup_10ep       output/training_clip_attention_self_12/model_final.pth
#   selfsup_15ep       output/training_clip_attention_self_17/model_final.pth

set -euo pipefail

PROJECT_ROOT="/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation"
PYTHON="/nfs/home/hassan/miniconda3/envs/gsnet/bin/python3"
CONFIG="configs/vitl_336_dinov3.yaml"
RSIB_CKPT="dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
OUT_ROOT="output/full_comparison_improved/run_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/${OUT_ROOT}"

cd "${PROJECT_ROOT}"
export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT="${RSIB_CKPT}"

echo "================================================================"
echo "FULL COMPARISON — improved class names (Potsdam + FLAIR)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}   Node: ${SLURM_NODELIST:-N/A}"
echo "Output: ${OUT_ROOT}"
echo "Start:  $(date)"
echo "================================================================"

if [[ ! -f "${RSIB_CKPT}" ]]; then
  echo "[ERROR] RSIB checkpoint not found: ${RSIB_CKPT}"; exit 1
fi

# ------------------------------------------------------------------ #
# Helper: run one eval and extract mIoU                               #
# ------------------------------------------------------------------ #
run_eval() {
  local label="$1"        # e.g. sup_unfreeze_5ep
  local model_ckpt="$2"   # path to model_final.pth
  local dataset="$3"      # potsdam | flair
  local class_json="$4"   # datasets/potsdam_improved.json etc.
  local dataset_key="$5"  # PotsdamSplit | FLAIR_test

  local out_dir="${OUT_ROOT}/${label}/${dataset}"

  echo ""
  echo "--- ${label} / ${dataset} ---"

  if [[ ! -f "${model_ckpt}" ]]; then
    echo "[SKIP] Model not found: ${model_ckpt}"
    echo "nan"
    return
  fi

  mkdir -p "${out_dir}"

  "${PYTHON}" train_net.py --config "${CONFIG}" \
    --num-gpus 1 \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR "${out_dir}" \
    MODEL.WEIGHTS "${model_ckpt}" \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
    DATASETS.TEST \(\"${dataset_key}\"\,\) \
    TEST.SLIDING_WINDOW "False" \
    MODEL.PROMPT_ENSEMBLE_TYPE "single" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
    INPUT.MIN_SIZE_TEST 1536 \
    INPUT.MAX_SIZE_TEST 4096 \
    | tee "${out_dir}/eval_stdout.log"

  # Extract per-class IoU and mIoU
  "${PYTHON}" - "${out_dir}" <<'PY'
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

print(f"mIoU={metrics.get('mIoU', 'NA')}")
PY
}

# ------------------------------------------------------------------ #
# Collect mIoU from a finished eval dir                               #
# ------------------------------------------------------------------ #
get_miou() {
  local out_dir="${OUT_ROOT}/$1/$2"
  local json_path="${out_dir}/summary_metrics.json"
  if [[ -f "${json_path}" ]]; then
    "${PYTHON}" -c "import json; d=json.load(open('${json_path}')); print(f\"{d.get('mIoU','nan'):.4f}\")" 2>/dev/null || echo "nan"
  else
    echo "nan"
  fi
}

# ------------------------------------------------------------------ #
# Configuration table                                                  #
# ------------------------------------------------------------------ #
declare -A MODELS
MODELS["sup_unfreeze_5ep"]="output/training_clip_attention_freeze/model_final.pth"
MODELS["sup_unfreeze_10ep"]="output/training_clip_attention_freeze_12/model_final.pth"
MODELS["sup_unfreeze_15ep"]="output/training_clip_attention_freeze_17/model_final.pth"
MODELS["lora_5ep"]="output/training_clip_attention_freeze_lora_07/model_final.pth"
MODELS["lora_10ep"]="output/training_clip_attention_freeze_lora_12/model_final.pth"
MODELS["lora_15ep"]="output/training_clip_attention_freeze_lora_17/model_final.pth"
MODELS["selfsup_5ep"]="output/training_clip_attention_self_07/model_final.pth"
MODELS["selfsup_10ep"]="output/training_clip_attention_self_12/model_final.pth"
MODELS["selfsup_15ep"]="output/training_clip_attention_self_17/model_final.pth"

ORDERED_LABELS=(
  "sup_unfreeze_5ep"
  "sup_unfreeze_10ep"
  "sup_unfreeze_15ep"
  "lora_5ep"
  "lora_10ep"
  "lora_15ep"
  "selfsup_5ep"
  "selfsup_10ep"
  "selfsup_15ep"
)

# ------------------------------------------------------------------ #
# Run all evals                                                        #
# ------------------------------------------------------------------ #
for label in "${ORDERED_LABELS[@]}"; do
  ckpt="${MODELS[$label]}"
  run_eval "${label}" "${ckpt}" "potsdam" "datasets/potsdam_improved.json" "PotsdamSplit"
  run_eval "${label}" "${ckpt}" "flair"   "datasets/flair_improved.json"   "FLAIR_test"
done

# ------------------------------------------------------------------ #
# Build summary CSV                                                    #
# ------------------------------------------------------------------ #
echo ""
echo "================================================================"
echo "Building summary CSV..."
echo "================================================================"

echo "config,potsdam_miou,flair_miou" > "${SUMMARY_CSV}"
for label in "${ORDERED_LABELS[@]}"; do
  potsdam_miou=$(get_miou "${label}" "potsdam")
  flair_miou=$(get_miou "${label}" "flair")
  echo "${label},${potsdam_miou},${flair_miou}" >> "${SUMMARY_CSV}"
  echo "  ${label}: Potsdam=${potsdam_miou}  FLAIR=${flair_miou}"
done

echo ""
echo "Summary written to: ${SUMMARY_CSV}"
echo "End: $(date)"
echo "================================================================"
