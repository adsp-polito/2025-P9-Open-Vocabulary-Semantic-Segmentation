"""
Student model evaluation script — Phase 2 zero-shot and Phase 3 fine-tuned.

Two modes:
  --mode zero-shot  TIPS backbone + dataset-specific text embeddings + bilinear upsample.
                    No decoder used. Works with any Phase 2 checkpoint.
                    NOTE: since TIPS is frozen in Phase 2, ALL Phase 2 checkpoints produce
                    identical results in this mode. Use this to measure the TIPS baseline
                    before fine-tuning, not to compare Phase 2 checkpoints.

  --mode student    Full student pipeline with Phase 3 checkpoint (K-class decoder).
                    Use after Phase 3 fine-tuning for the final mIoU numbers.

Supported datasets:
  FloodNet  Potsdam  FLAIR  FAST

Usage (zero-shot, any Phase 2 checkpoint):
    python scripts/eval_student.py \\
        --mode       zero-shot \\
        --tips-dir   checkpoints/tipsv2-l14 \\
        --dataset    FloodNet \\
        --output-dir results/phase2_zeroshot/FloodNet

Usage (Phase 3 fine-tuned student):
    python scripts/eval_student.py \\
        --mode       student \\
        --checkpoint output/finetune/floodnet/student_best.pth \\
        --tips-dir   checkpoints/tipsv2-l14 \\
        --dataset    FloodNet \\
        --output-dir results/phase3/FloodNet
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path("gs_net/data/datasets")

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "FloodNet": {
        "image_dir": ROOT / "FloodNet/val+test/img",
        "mask_dir":  ROOT / "FloodNet/val+test/lbl",
        "class_json": "datasets/floodnet.json",
        "ignore_label": 0,       # Background is ignored in eval
        "img_ext": ".jpg",
        "mask_ext": ".png",
    },
    "Potsdam": {
        "image_dir": ROOT / "PotsdamSplit/img_dir/val",
        "mask_dir":  ROOT / "PotsdamSplit/ann_dir/val",
        "class_json": "datasets/potsdam.json",
        "ignore_label": 5,
        "img_ext": ".png",
        "mask_ext": ".png",
    },
    "FLAIR": {
        "image_dir": ROOT / "FLAIR_test/image",
        "mask_dir":  ROOT / "FLAIR_test/mask",
        "class_json": "datasets/flair.json",
        "ignore_label": 12,
        "img_ext": ".png",
        "mask_ext": ".png",
    },
    "FAST": {
        "image_dir": ROOT / "FAST/val/images",
        "mask_dir":  ROOT / "FAST/val/semlabels/gray",
        "class_json": "datasets/fast.json",
        "ignore_label": 255,
        "img_ext": ".png",
        "mask_ext": ".png",
    },
}


# ── mIoU computation ──────────────────────────────────────────────────────────

def compute_miou(preds, gts, num_classes, ignore_label):
    """
    Args:
        preds: list of (H, W) int32 numpy arrays (predicted class indices).
        gts:   list of (H, W) int32 numpy arrays (GT class indices).
        num_classes:  number of valid classes.
        ignore_label: GT value to exclude from evaluation.

    Returns:
        dict with 'mIoU', 'per_class_iou', 'mACC'.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds, gts):
        mask = gt != ignore_label
        p = pred[mask].astype(np.int64)
        g = gt[mask].astype(np.int64)
        valid = (g >= 0) & (g < num_classes) & (p >= 0) & (p < num_classes)
        np.add.at(confusion, (g[valid], p[valid]), 1)

    iou_per_class = []
    acc_per_class = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp
        fp = confusion[:, c].sum() - tp
        denom = tp + fn + fp
        iou_per_class.append(float(tp) / denom if denom > 0 else float("nan"))
        acc_per_class.append(float(tp) / confusion[c, :].sum()
                             if confusion[c, :].sum() > 0 else float("nan"))

    valid_iou = [v for v in iou_per_class if not np.isnan(v)]
    valid_acc = [v for v in acc_per_class if not np.isnan(v)]
    return {
        "mIoU":         float(np.mean(valid_iou)) * 100,
        "mACC":         float(np.mean(valid_acc)) * 100,
        "per_class_iou": [round(v * 100, 2) if not np.isnan(v) else None
                          for v in iou_per_class],
    }


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_pairs(cfg):
    """Return sorted list of (image_path, mask_path) pairs."""
    img_dir  = Path(cfg["image_dir"])
    mask_dir = Path(cfg["mask_dir"])
    ext_i    = cfg["img_ext"]
    ext_m    = cfg["mask_ext"]

    stems = sorted(p.stem for p in img_dir.glob(f"*{ext_i}"))
    pairs = []
    for stem in stems:
        ip = img_dir  / f"{stem}{ext_i}"
        mp = mask_dir / f"{stem}{ext_m}"
        if mp.exists():
            pairs.append((ip, mp))
    assert len(pairs) > 0, f"No matched image/mask pairs in {img_dir}"
    return pairs


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_image_tips(path, resolution):
    """Load image as [0,1] float tensor (TIPS input format)."""
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, (resolution, resolution))
    return TF.to_tensor(img)  # (3, H, W) in [0, 1]


def run_zeroshot(tips_model, text_embeds, image_tensor, out_h, out_w, device):
    """
    Zero-shot inference: TIPS features → correlation → bilinear upsample.
    No decoder used.

    Args:
        text_embeds: (K, D) L2-normalised text embeddings for K dataset classes.

    Returns:
        (H, W) int64 predicted class map.
    """
    img = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, patch_tokens = tips_model.vision_encoder(img)
        # patch_tokens: (1, N, D) = (1, 576, 1024)

    # Reshape to spatial feature map
    patch_feats = patch_tokens[0]                              # (N, D)
    H_p = W_p = int(patch_feats.shape[0] ** 0.5)              # 24
    patch_feats = patch_feats.view(H_p, W_p, -1)              # (24, 24, D)
    patch_feats = F.normalize(patch_feats, dim=-1)

    # Cosine similarity with class texts
    corr = torch.einsum("h w d, k d -> k h w", patch_feats, text_embeds)  # (K, 24, 24)

    # Bilinear upsample to original image resolution
    corr = corr.unsqueeze(0)                                   # (1, K, 24, 24)
    corr = F.interpolate(corr, size=(out_h, out_w), mode="bilinear", align_corners=False)
    pred = corr[0].argmax(dim=0).cpu().numpy().astype(np.int32)
    return pred


def run_student(student_model, image_tensor, out_h, out_w, device):
    """
    Full student inference (Phase 3 checkpoint, K-class decoder).

    Returns:
        (H, W) int64 predicted class map.
    """
    img = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = student_model(img)    # (1, K, 96, 96)
    logits = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy().astype(np.int32)
    return pred


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["zero-shot", "student"], required=True)
    p.add_argument("--dataset",     choices=list(DATASETS), required=True)
    p.add_argument("--tips-dir",    required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--checkpoint",  default=None,
                   help="Student checkpoint (.pth). Required for --mode student.")
    p.add_argument("--class-json",  default=None,
                   help="Override dataset class list JSON (optional).")
    p.add_argument("--resolution",  type=int, default=336,
                   help="Input resolution for TIPS (default 336).")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = DATASETS[args.dataset]
    class_json = args.class_json or cfg["class_json"]

    with open(class_json) as f:
        class_names = json.load(f)
    num_classes  = len(class_names)
    ignore_label = cfg["ignore_label"]

    print(f"Dataset:      {args.dataset}")
    print(f"Classes:      {num_classes}  (ignore_label={ignore_label})")
    print(f"Mode:         {args.mode}")
    print(f"Class list:   {class_names}")

    # ── Load TIPS ─────────────────────────────────────────────────────────────
    from gs_distill.student import _load_tips
    print(f"\nLoading TIPS from {args.tips_dir} ...")
    tips_model = _load_tips(args.tips_dir).to(device).eval()

    # ── Pre-compute text embeddings for this dataset ──────────────────────────
    with torch.no_grad():
        text_embeds = tips_model.encode_text(class_names).to(device)  # (K, D)
        text_embeds = F.normalize(text_embeds, dim=-1)
    print(f"Text embeds:  {text_embeds.shape}")

    # ── Optionally load full student (Phase 3) ────────────────────────────────
    student_model = None
    if args.mode == "student":
        assert args.checkpoint, "--checkpoint required for --mode student"
        from gs_distill.student import TIPSDistillStudent
        from gs_distill.decoder import LightweightDecoder

        ckpt = torch.load(args.checkpoint, map_location=device)
        saved_args = ckpt.get("args", {})
        hidden_dim = saved_args.get("hidden_dim", 128)

        student_model = TIPSDistillStudent(
            tips_dir=args.tips_dir,
            num_classes=num_classes,
            hidden=hidden_dim,
            class_texts=class_names,
        ).to(device)
        student_model.load_state_dict(ckpt["student"])
        student_model.eval()
        print(f"Loaded student checkpoint: {args.checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    pairs = load_pairs(cfg)
    print(f"\nEvaluating {len(pairs)} images ...")

    all_preds, all_gts = [], []

    for img_path, mask_path in tqdm(pairs):
        gt = np.array(Image.open(mask_path), dtype=np.int32)
        out_h, out_w = gt.shape[:2]

        img_tensor = load_image_tips(img_path, args.resolution)

        if args.mode == "zero-shot":
            pred = run_zeroshot(tips_model, text_embeds, img_tensor, out_h, out_w, device)
        else:
            pred = run_student(student_model, img_tensor, out_h, out_w, device)

        all_preds.append(pred)
        all_gts.append(gt)

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_miou(all_preds, all_gts, num_classes, ignore_label)

    print(f"\n{'='*50}")
    print(f"Dataset:   {args.dataset}")
    print(f"Mode:      {args.mode}")
    print(f"mIoU:      {metrics['mIoU']:.2f}%")
    print(f"mACC:      {metrics['mACC']:.2f}%")
    print(f"\nPer-class IoU:")
    for name, iou in zip(class_names, metrics["per_class_iou"]):
        val = f"{iou:.2f}%" if iou is not None else "N/A"
        print(f"  {name:<35} {val}")
    print(f"{'='*50}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "dataset":       args.dataset,
        "mode":          args.mode,
        "checkpoint":    args.checkpoint,
        "class_json":    class_json,
        "num_images":    len(pairs),
        "num_classes":   num_classes,
        "ignore_label":  ignore_label,
        "mIoU":          round(metrics["mIoU"], 4),
        "mACC":          round(metrics["mACC"], 4),
        "per_class_iou": {
            name: iou for name, iou in zip(class_names, metrics["per_class_iou"])
        },
    }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
