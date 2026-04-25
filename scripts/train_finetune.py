"""
Phase 4 (optional) — End-to-end fine-tune of student + RIPD on LD50K segmentation labels.

Loads:
  - trained student conv head (from distillation checkpoint)
  - frozen RIPD from GSNet checkpoint (optionally unfrozen)

Trains end-to-end on LD50K with cross-entropy segmentation loss.
CLIP stays frozen throughout.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'   # still needed to load GSNet; RSIB stays frozen

    python scripts/train_finetune.py \\
        --gsnet-config    configs/vitb_384.yaml \\
        --gsnet-weights   path/to/gsnet_checkpoint.pth \\
        --student-ckpt    output/distill/student_best.pth \\
        --image-dir       path/to/LD50K/images \\
        --label-dir       path/to/LD50K/labels \\
        --class-json      datasets/landdiscover.json \\
        --output-dir      output/finetune/ \\
        [--epochs 10] \\
        [--batch-size 8] \\
        [--lr 1e-5] \\
        [--unfreeze-ripd] \\
        [--amp] \\
        [--device cuda]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

import clip as openai_clip

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_distill.student import GSDistillStudent
from gs_distill.inference import gs_distill_inference


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation dataset
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """
    Flat-folder RGB image + single-channel label dataset.
    Label pixels are class indices in [0, num_classes-1]; 255 = ignore.
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384,
                 clip_mean=(0.48145466, 0.4578275, 0.40821073),
                 clip_std=(0.26862954, 0.26130258, 0.27577711)):
        self.resolution = resolution
        self.clip_mean = clip_mean
        self.clip_std  = clip_std

        img_stems = {
            p.stem for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
        lbl_stems = {
            p.stem for p in Path(label_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
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
        lbl = TF.resize(lbl, (self.resolution, self.resolution), interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(lbl.tobytes())).numpy()
        ).long()
        if lbl.dim() == 1:
            lbl = lbl.view(self.resolution, self.resolution)
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--student-ckpt",   required=True)
    p.add_argument("--image-dir",      required=True)
    p.add_argument("--label-dir",      required=True)
    p.add_argument("--class-json",     default="datasets/landdiscover.json")
    p.add_argument("--output-dir",     default="output/finetune")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--unfreeze-ripd",  action="store_true")
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction",   type=float, default=0.05)
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
            feats = clip_model.encode_text(tokens).float()  # (P, C)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
    # (T, P, C) → (1, T, P, C) batch dim
    text_feats = torch.stack(all_feats, dim=0).unsqueeze(0)
    return text_feats   # (1, T, P, C)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load CLIP ────────────────────────────────────────────────────────────
    print("Loading CLIP ...")
    clip_model, _ = openai_clip.load("ViT-B/16", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ── Load GSNet (to extract RIPD + projection layers) ─────────────────────
    print("Loading GSNet ...")
    gsnet = build_gsnet(args.gsnet_config, args.gsnet_weights, str(device))
    gsnet = gsnet.to(device)

    ripd            = gsnet.sem_seg_head.predictor
    clip_upsample1  = gsnet.upsample1
    clip_upsample2  = gsnet.upsample2
    dino_decod_proj1 = gsnet.dino_decod_proj1
    dino_decod_proj2 = gsnet.dino_decod_proj2

    if not args.unfreeze_ripd:
        for p in ripd.parameters():
            p.requires_grad = False
    else:
        print("  RIPD unfrozen for fine-tuning.")

    # ── Load student ─────────────────────────────────────────────────────────
    print(f"Loading student from {args.student_ckpt} ...")
    ckpt = torch.load(args.student_ckpt, map_location=device)
    student_args = ckpt.get("args", {})
    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=student_args.get("hidden_dim", 128),
        d_dino=student_args.get("d_dino", 768),
        num_classes=student_args.get("num_classes", 40),
        clip_layers=student_args.get("clip_layers", [4, 8, 10, 12]),
    ).to(device)
    student.load_state_dict(ckpt["student"])

    # ── Text features (frozen) ────────────────────────────────────────────────
    text_feats = build_text_features(args.class_json, clip_model, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = SegDataset(args.image_dir, args.label_dir)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"  Train: {n_train}  Val: {n_val}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    params = list(student.trainable_parameters())
    if args.unfreeze_ripd:
        params += list(ripd.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler    = GradScaler(enabled=args.amp)
    ignore_idx = 255

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        student.train()
        if args.unfreeze_ripd:
            ripd.train()

        train_loss = 0.0
        n_train_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.expand(B, -1, -1, -1)   # (B, T, P, C)

            optimizer.zero_grad(set_to_none=True)

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
                )
                # logit: (B, T, H, W)  labels: (B, H, W)
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                         mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        # Validation
        student.eval()
        if args.unfreeze_ripd:
            ripd.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]
                tf = text_feats.expand(B, -1, -1, -1)
                with autocast(enabled=args.amp):
                    logit = gs_distill_inference(
                        image=images, text_feats=tf, student=student,
                        clip_model=clip_model, ripd=ripd,
                        clip_upsample1=clip_upsample1, clip_upsample2=clip_upsample2,
                        dino_decod_proj1=dino_decod_proj1, dino_decod_proj2=dino_decod_proj2,
                    )
                    logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                             mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)
                val_loss += loss.item()
                n_val_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val   = val_loss   / max(n_val_batches,   1)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "ripd": ripd.state_dict() if args.unfreeze_ripd else None,
            "val_loss": avg_val,
        }, os.path.join(args.output_dir, "finetune_latest.pth"))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.output_dir, "finetune_best.pth")
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "ripd": ripd.state_dict() if args.unfreeze_ripd else None,
                "val_loss": avg_val,
            }, best_path)
            print(f"  ✓ New best val loss: {avg_val:.4f} → {best_path}")

    print(f"\nFine-tuning complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
