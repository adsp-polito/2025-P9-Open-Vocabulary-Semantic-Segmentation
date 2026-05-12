"""
GS-Distill Phase 4 evaluation — runs the finetuned student + RIPD on a
labelled split and reports per-class IoU, mIoU, mACC, and pACC.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/eval_finetune.py \\
        --gsnet-config  configs/vitl_336_dinov3.yaml \\
        --gsnet-weights output/gsnet_pretrain/model_final.pth \\
        --finetune-ckpt output/finetune/finetune_best.pth \\
        --image-dir     gs_net/data/datasets/LandDiscover_50K/TE_Image \\
        --label-dir     gs_net/data/datasets/LandDiscover_50K/TE_Label \\
        --class-json    datasets/landdiscover.json \\
        --output-dir    output/finetune/eval \\
        [--batch-size 4] [--amp] [--device cuda]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

import clip as openai_clip

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_distill.student import GSDistillStudent
from gs_distill.inference import gs_distill_inference


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384,
                 clip_mean=(0.48145466, 0.4578275, 0.40821073),
                 clip_std=(0.26862954, 0.26130258, 0.27577711)):
        self.resolution = resolution
        self.clip_mean  = clip_mean
        self.clip_std   = clip_std

        img_stems = {p.stem for p in Path(image_dir).rglob("*") if p.suffix.lower() in self.EXTENSIONS}
        lbl_stems = {p.stem for p in Path(label_dir).rglob("*") if p.suffix.lower() in self.EXTENSIONS}
        stems = sorted(img_stems & lbl_stems)

        self.samples = []
        for stem in stems:
            for ext in (".png", ".jpg", ".tif"):
                ip = Path(image_dir) / (stem + ext)
                lp = Path(label_dir) / (stem + ext)
                if ip.exists() and lp.exists():
                    self.samples.append((str(ip), str(lp)))
                    break

        if not self.samples:
            raise RuntimeError(f"No matching image/label pairs in {image_dir} / {label_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.clip_mean, std=self.clip_std)

        lbl = Image.open(lbl_path)
        lbl = TF.resize(lbl, (self.resolution, self.resolution),
                        interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(lbl.tobytes())).numpy()
        ).long()
        if lbl.dim() == 1:
            lbl = lbl.view(self.resolution, self.resolution)
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--finetune-ckpt",  required=True,
                   help="Path to finetune_best.pth produced by train_finetune.py")
    p.add_argument("--image-dir",      required=True)
    p.add_argument("--label-dir",      required=True)
    p.add_argument("--class-json",     default="datasets/landdiscover.json")
    p.add_argument("--output-dir",     default="output/finetune/eval")
    p.add_argument("--batch-size",     type=int, default=4)
    p.add_argument("--num-workers",    type=int, default=4)
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_gsnet(config_file, weights_file, device):
    from gs_net import add_cat_seg_config
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    from detectron2.modeling import build_model as d2_build
    model = d2_build(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()
    return model


def build_text_features(class_json, clip_model, device):
    with open(class_json) as f:
        class_names = json.load(f)
    templates = ["a photo of a {}."]
    all_feats = []
    with torch.no_grad():
        for name in class_names:
            texts = [t.format(name) for t in templates]
            tokens = openai_clip.tokenize(texts).to(device)
            feats = clip_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
    return torch.stack(all_feats, dim=0).unsqueeze(0)   # (1, T, P, C)


def compute_iou(confusion: np.ndarray, ignore_idx: int = 255) -> dict:
    """
    Given a (C, C) confusion matrix (rows=GT, cols=Pred), return per-class
    IoU, mIoU, mACC, and pACC.  Classes with no GT pixels are excluded from
    the mean.
    """
    n = confusion.shape[0]
    tp  = np.diag(confusion)
    fp  = confusion.sum(axis=0) - tp   # predicted as class c but isn't
    fn  = confusion.sum(axis=1) - tp   # is class c but predicted as something else

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
        acc = np.where(confusion.sum(axis=1) > 0,
                       tp / confusion.sum(axis=1), np.nan)

    present = ~np.isnan(iou)
    miou    = float(np.nanmean(iou[present]))
    macc    = float(np.nanmean(acc[present]))
    pacc    = float(tp.sum() / confusion.sum()) if confusion.sum() > 0 else 0.0

    return {
        "per_class_iou": iou.tolist(),
        "mIoU":  miou,
        "mACC":  macc,
        "pACC":  pacc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── GSNet → CLIP, RIPD, projection layers ────────────────────────────────
    print("Loading GSNet ...")
    gsnet = build_gsnet(args.gsnet_config, args.gsnet_weights, str(device))
    gsnet = gsnet.to(device)

    clip_model = gsnet.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_skip_indices = tuple(gsnet.layer_indexes)

    ripd             = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1   = gsnet.upsample1
    clip_upsample2   = gsnet.upsample2
    dino_decod_proj1 = gsnet.dino_decod_proj1
    dino_decod_proj2 = gsnet.dino_decod_proj2

    # ── Student ───────────────────────────────────────────────────────────────
    print(f"Loading finetuned student from {args.finetune_ckpt} ...")
    ckpt = torch.load(args.finetune_ckpt, map_location=device)

    # finetune_best.pth stores student weights directly; args come from the
    # distill checkpoint that was loaded during training — fall back to defaults.
    student_args = ckpt.get("args", {})
    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=student_args.get("hidden_dim", 128),
        d_dino=student_args.get("d_dino", 768),
        num_classes=student_args.get("num_classes", 40),
        clip_layers=student_args.get("clip_layers", [4, 8, 10, 12]),
    ).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()

    # If RIPD was unfrozen during finetuning, load those weights too.
    if ckpt.get("ripd") is not None:
        ripd.load_state_dict(ckpt["ripd"])
        print("  RIPD weights loaded from finetune checkpoint.")
    ripd.eval()

    # ── Class names + text features ───────────────────────────────────────────
    with open(args.class_json) as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    text_feats = build_text_features(args.class_json, clip_model, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SegDataset(args.image_dir, args.label_dir)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"  Eval samples: {len(dataset)}")

    # ── Inference + confusion matrix ─────────────────────────────────────────
    ignore_idx = 255
    confusion  = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.expand(B, -1, -1, -1)

            with autocast(enabled=args.amp):
                logit = gs_distill_inference(
                    image=images,
                    text_feats=tf,
                    student=student,
                    clip_model=clip_model,
                    ripd=ripd,
                    clip_upsample1=clip_upsample1,
                    clip_upsample2=clip_upsample2,
                    dino_decod_proj1=dino_decod_proj1,
                    dino_decod_proj2=dino_decod_proj2,
                    clip_skip_layer_indices=clip_skip_indices,
                )
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                         mode="bilinear", align_corners=False)

            pred = logit_up.argmax(dim=1).cpu().numpy()   # (B, H, W)
            gt   = labels.cpu().numpy()                    # (B, H, W)

            mask = (gt != ignore_idx)
            gt_valid   = gt[mask]
            pred_valid = pred[mask]

            np.add.at(confusion, (gt_valid, pred_valid), 1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_iou(confusion, ignore_idx)
    per_class = metrics["per_class_iou"]

    print("\n" + "=" * 60)
    print(f"mIoU : {metrics['mIoU'] * 100:.2f}%")
    print(f"mACC : {metrics['mACC'] * 100:.2f}%")
    print(f"pACC : {metrics['pACC'] * 100:.2f}%")
    print("-" * 60)
    for i, name in enumerate(class_names):
        iou_val = per_class[i]
        if not (iou_val != iou_val):   # not NaN
            print(f"  {name:<35s} {iou_val * 100:6.2f}%")
        else:
            print(f"  {name:<35s}    N/A  (no GT pixels)")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "mIoU":  metrics["mIoU"],
        "mACC":  metrics["mACC"],
        "pACC":  metrics["pACC"],
        "finetune_ckpt": args.finetune_ckpt,
        "image_dir":     args.image_dir,
        "label_dir":     args.label_dir,
        "class_json":    args.class_json,
    }
    with open(os.path.join(args.output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "per_class_iou.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou_percent"])
        for i, name in enumerate(class_names):
            iou_val = per_class[i]
            row_val = f"{iou_val * 100:.4f}" if not (iou_val != iou_val) else "NA"
            writer.writerow([name, row_val])

    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write("GS-Distill finetune evaluation\n")
        f.write(f"checkpoint:  {args.finetune_ckpt}\n")
        f.write(f"image_dir:   {args.image_dir}\n")
        f.write(f"label_dir:   {args.label_dir}\n")
        f.write(f"class_json:  {args.class_json}\n")
        f.write(f"mIoU:  {metrics['mIoU'] * 100:.2f}%\n")
        f.write(f"mACC:  {metrics['mACC'] * 100:.2f}%\n")
        f.write(f"pACC:  {metrics['pACC'] * 100:.2f}%\n")
        f.write("\nPer-class IoU:\n")
        for i, name in enumerate(class_names):
            iou_val = per_class[i]
            row_val = f"{iou_val * 100:.2f}%" if not (iou_val != iou_val) else "N/A"
            f.write(f"  {name:<35s} {row_val}\n")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
