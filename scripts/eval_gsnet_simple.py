"""
Evaluate the original GSNet model on 4 test datasets using the same simple
protocol as eval_decoder.py:
  - No sliding window (single forward pass)
  - GSNet internally resizes to clip_resolution=336×336 and dino_resolution=384×384
  - Output upsampled to original image resolution, then argmax → mIoU

This gives a fair, protocol-matched comparison against the student model.

Results saved to: results_v2/gsnet_simple/
  {dataset}.json   — per-class IoU + mIoU
  summary.json     — mIoU per dataset

Usage:
  python scripts/eval_gsnet_simple.py \\
      --checkpoint  output/gsnet_pretrain/model_final.pth \\
      --config      output/gsnet_pretrain/config.yaml \\
      --rsib-ckpt   dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \\
      --data-root   gs_net/data/datasets \\
      --results-dir results_v2/gsnet_simple
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./gs_net'))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

sys.path.insert(0, os.path.abspath('./gs_net'))
from config import add_cat_seg_config


# ── Dataset configs (same as eval_decoder.py) ────────────────────────────────

DATASETS = {
    "FloodNet": {
        "img_dir":      "FloodNet/val+test/img",
        "mask_dir":     "FloodNet/val+test/lbl",
        "img_ext":      ".jpg",
        "mask_ext":     ".png",
        "class_json":   "datasets/floodnet.json",
        "ignore_label": 0,
    },
    "FAST": {
        "img_dir":      "FAST/val/images",
        "mask_dir":     "FAST/val/semlabels/gray",
        "img_ext":      ".png",
        "mask_ext":     ".png",
        "class_json":   "datasets/fast.json",
        "ignore_label": 255,
    },
    "Potsdam": {
        "img_dir":      "PotsdamSplit/img_dir/val",
        "mask_dir":     "PotsdamSplit/ann_dir/val",
        "img_ext":      ".png",
        "mask_ext":     ".png",
        "class_json":   "datasets/potsdam.json",
        "ignore_label": 5,
    },
    "FLAIR": {
        "img_dir":      "FLAIR_test/image",
        "mask_dir":     "FLAIR_test/mask",
        "img_ext":      ".png",
        "mask_ext":     ".png",
        "class_json":   "datasets/flair.json",
        "ignore_label": 12,
    },
}


# ── mIoU ─────────────────────────────────────────────────────────────────────

def compute_miou(all_preds, all_gts, num_classes, ignore_label):
    intersection = np.zeros(num_classes, dtype=np.float64)
    union        = np.zeros(num_classes, dtype=np.float64)
    for pred, gt in zip(all_preds, all_gts):
        gt, pred = gt.astype(np.int64), pred.astype(np.int64)
        valid    = gt != ignore_label
        pv, gv   = pred[valid], gt[valid]
        for c in range(num_classes):
            tp = int(np.sum((pv == c) & (gv == c)))
            fp = int(np.sum((pv == c) & (gv != c)))
            fn = int(np.sum((pv != c) & (gv == c)))
            intersection[c] += tp
            union[c]        += tp + fp + fn
    iou = np.where(union > 0, intersection / union, np.nan)
    return float(np.nanmean(iou)), iou


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_pairs(data_root, cfg):
    img_dir  = Path(data_root) / cfg["img_dir"]
    mask_dir = Path(data_root) / cfg["mask_dir"]
    pairs = []
    for p in sorted(img_dir.glob(f"*{cfg['img_ext']}")):
        m = mask_dir / (p.stem + cfg["mask_ext"])
        if m.exists():
            pairs.append((p, m))
    return pairs


# ── GSNet model loader ────────────────────────────────────────────────────────

def load_gsnet(config_path, checkpoint_path, rsib_ckpt):
    os.environ["RSIB_CKPT"] = rsib_ckpt

    # Import gs_net modules so Detectron2 registries are populated
    import gs_net.GSNet            # noqa: registers GSNet meta-arch
    import gs_net.modeling         # noqa: registers heads/predictor

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list([
        "MODEL.WEIGHTS",        checkpoint_path,
        "TEST.SLIDING_WINDOW",  "False",
        # Force any dataset for initial build; we swap classes at eval time
        "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON", "datasets/floodnet.json",
        "MODEL.DEVICE",         "cuda" if torch.cuda.is_available() else "cpu",
    ])
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(checkpoint_path)
    model.eval()
    return model


# ── Per-dataset eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dataset(name, cfg_ds, model, data_root, device):
    class_json   = cfg_ds["class_json"]
    ignore_label = cfg_ds["ignore_label"]

    with open(class_json) as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Swap test classes in the predictor without rebuilding the model
    predictor = model.sem_seg_head.predictor
    predictor.test_class_texts = class_names
    predictor.cache  = None   # clear cached text embeddings
    predictor.tokens = None   # force re-tokenisation

    print(f"\n{'─'*60}")
    print(f"Dataset: {name}  ({num_classes} classes, ignore={ignore_label})")

    pairs = load_pairs(data_root, cfg_ds)
    print(f"Images:  {len(pairs)}")

    all_preds, all_gts = [], []

    for img_path, mask_path in tqdm(pairs, desc=name):
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_np  = np.array(img, dtype=np.float32)          # (H, W, 3)  [0,255]
        img_t   = torch.from_numpy(img_np).permute(2,0,1)  # (3, H, W)

        # GSNet normalises internally; feed at original resolution,
        # specify output size = original so sem_seg_postprocess upsamples correctly
        batched = [{"image": img_t.to(device), "height": orig_h, "width": orig_w}]
        result  = model(batched)                            # [{'sem_seg': (K, H, W)}]

        sem_seg = result[0]["sem_seg"]                      # (K, orig_H, orig_W)
        pred    = sem_seg.argmax(dim=0).cpu().numpy().astype(np.uint8)
        gt      = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        all_preds.append(pred)
        all_gts.append(gt)

    miou, iou_per_class = compute_miou(all_preds, all_gts, num_classes, ignore_label)

    result_dict = {
        "dataset":       name,
        "num_classes":   num_classes,
        "ignore_label":  ignore_label,
        "miou":          round(miou * 100, 2),
        "per_class_iou": {
            cls: (round(float(iou)*100, 2) if not np.isnan(iou) else None)
            for cls, iou in zip(class_names, iou_per_class)
        },
    }
    print(f"mIoU: {result_dict['miou']:.2f}%")
    for cls, iou in result_dict["per_class_iou"].items():
        print(f"  {cls:<35} {iou:.2f}%" if iou is not None else f"  {cls:<35} —")
    return result_dict


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="output/gsnet_pretrain/model_final.pth")
    p.add_argument("--config",      default="output/gsnet_pretrain/config.yaml")
    p.add_argument("--rsib-ckpt",   default="dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    p.add_argument("--data-root",   default="gs_net/data/datasets")
    p.add_argument("--results-dir", default="results_v2/gsnet_simple")
    p.add_argument("--datasets",    nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading GSNet from {args.checkpoint} ...")
    model = load_gsnet(args.config, args.checkpoint, args.rsib_ckpt)
    model.to(device)
    print("Model loaded.\n")

    summary = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"Unknown dataset '{name}', skipping.")
            continue
        res = eval_dataset(name, DATASETS[name], model, args.data_root, device)
        summary[name] = res["miou"]
        out = Path(args.results_dir) / f"{name.lower()}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved → {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    gsnet_orig = {"FloodNet": 42.63, "FAST": 16.61, "Potsdam": 45.75, "FLAIR": 20.00}
    student_d  = {"FloodNet":  9.03, "FAST": 12.44, "Potsdam": 21.32, "FLAIR": 14.70}
    student_s  = {"FloodNet":  9.57, "FAST": 13.45, "Potsdam": 18.95, "FLAIR": 17.50}

    summary_out = {
        "model":      "GSNet original (no sliding window, single-pass eval)",
        "checkpoint": args.checkpoint,
        "results":    summary,
    }
    with open(Path(args.results_dir) / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    avg_gsnet   = sum(gsnet_orig.values()) / 4
    avg_simple  = sum(summary.values()) / len(summary) if summary else 0
    avg_stud_d  = sum(student_d[d] for d in summary if d in student_d) / len(summary)
    avg_stud_s  = sum(student_s[d] for d in summary if d in student_s) / len(summary)

    print(f"\n{'='*72}")
    print(f"COMPARISON — same single-pass protocol")
    print(f"{'='*72}")
    print(f"{'Dataset':<12} {'GSNet-slide':>12} {'GSNet-simple':>13} {'Distilled':>10} {'Scratch':>9}")
    print(f"{'─'*60}")
    for ds in summary:
        print(f"{ds:<12} {gsnet_orig.get(ds,'—'):>11}%  {summary[ds]:>12.2f}%"
              f"  {student_d.get(ds,'—'):>9}%  {student_s.get(ds,'—'):>8}%")
    print(f"{'─'*60}")
    print(f"{'Average':<12} {avg_gsnet:>11.2f}%  {avg_simple:>12.2f}%"
          f"  {avg_stud_d:>9.2f}%  {avg_stud_s:>8.2f}%")
    print(f"\nFull results → {args.results_dir}/")


if __name__ == "__main__":
    main()
