"""
Baseline — same architecture as GS-Distill but NO distillation pre-training.

Builds a GSDistillStudent (identical shared trunk + 4 heads) with randomly
initialised weights, then fine-tunes directly on LD50K segmentation labels.

This is the apples-to-apples benchmark for GS-Distill: same model, same
fine-tuning procedure, only difference is the student starts from random
weights instead of a distilled checkpoint.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/train_baseline.py \\
        --gsnet-config    configs/vitl_336_dinov3.yaml \\
        --gsnet-weights   path/to/gsnet_checkpoint.pth \\
        --image-dir       path/to/LD50K/images \\
        --label-dir       path/to/LD50K/labels \\
        --output-dir      output/baseline/ \\
        [--epochs 15] \\
        [--batch-size 4] \\
        [--lr 1e-5] \\
        [--unfreeze-ripd] \\
        [--warmup-decoder-epochs 3] \\
        [--amp] \\
        [--device cuda]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

import clip as openai_clip
import wandb

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_distill.student import GSDistillStudent
from gs_distill.inference import gs_distill_inference


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation dataset  (identical to train_finetune.py)
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
        self.clip_mean  = clip_mean
        self.clip_std   = clip_std

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
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--image-dir",      required=True)
    p.add_argument("--label-dir",      required=True)
    p.add_argument("--class-json",     default="datasets/landdiscover.json")
    p.add_argument("--output-dir",     default="output/baseline")
    p.add_argument("--epochs",         type=int,   default=15)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--hidden-dim",     type=int,   default=128)
    p.add_argument("--d-dino",         type=int,   default=768)
    p.add_argument("--num-classes",    type=int,   default=40)
    p.add_argument("--clip-layers",    type=int,   nargs="+", default=[4, 8, 10, 12])
    p.add_argument("--unfreeze-ripd",          action="store_true")
    p.add_argument("--warmup-decoder-epochs",  type=int, default=0,
                   help="Train RIPD decoder alone for N epochs, then unfreeze student heads.")
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction",   type=float, default=0.05)
    p.add_argument("--wandb-project",  default="gs-distill")
    p.add_argument("--wandb-run",      default=None, help="W&B run name (auto if omitted)")
    p.add_argument("--no-wandb",       action="store_true", help="Disable W&B logging")
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
    text_feats = torch.stack(all_feats, dim=0).unsqueeze(0)  # (1, T, P, C)
    return text_feats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load GSNet (to extract CLIP, RIPD + projection layers) ──────────────
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

    warmup        = args.warmup_decoder_epochs
    ripd_unfrozen = args.unfreeze_ripd or warmup > 0

    for p in ripd.parameters():
        p.requires_grad = ripd_unfrozen

    decoder_proj_modules = [m for m in [clip_upsample1, clip_upsample2,
                                         dino_decod_proj1, dino_decod_proj2]
                             if m is not None]
    for m in decoder_proj_modules:
        for p in m.parameters():
            p.requires_grad = ripd_unfrozen

    # ── Build student from scratch (random init — no distillation) ───────────
    print("Building baseline student (random init, same arch as GS-Distill) ...")
    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=args.hidden_dim,
        d_dino=args.d_dino,
        num_classes=args.num_classes,
        clip_layers=args.clip_layers,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"  Trainable params: {n_params / 1e6:.2f}M")

    if warmup > 0:
        for p in student.parameters():
            p.requires_grad = False
        print(f"  Student frozen for {warmup}-epoch RIPD warmup.")
    elif ripd_unfrozen:
        print("  RIPD unfrozen for joint fine-tuning.")

    # ── Text features ─────────────────────────────────────────────────────────
    text_feats = build_text_features(args.class_json, clip_model, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = SegDataset(args.image_dir, args.label_dir)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  Train: {n_train}  Val: {n_val}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def _student_head_params():
        return (
            list(student.shared_trunk.parameters())
            + list(student.fusion_branch.parameters())
            + list(student.dino_l4_branch.parameters())
            + list(student.dino_l8_branch.parameters())
        )

    def _decoder_proj_params():
        return [p for m in decoder_proj_modules for p in m.parameters()]

    if warmup > 0:
        active_params = list(ripd.parameters()) + _decoder_proj_params()
    else:
        active_params = _student_head_params()
        if ripd_unfrozen:
            active_params += list(ripd.parameters()) + _decoder_proj_params()

    optimizer  = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler     = GradScaler(enabled=args.amp)
    ignore_idx = 255
    best_val_loss = float("inf")

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        if warmup > 0 and epoch == warmup:
            for p in _student_head_params():
                p.requires_grad = True
            active_params = _student_head_params() + list(ripd.parameters()) + _decoder_proj_params()
            optimizer = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
            print(f"  Epoch {epoch+1}: student heads unfrozen — joint fine-tuning begins.")

        student_training = (warmup == 0) or (epoch >= warmup)
        student.train() if student_training else student.eval()
        if ripd_unfrozen:
            ripd.train()

        train_loss = 0.0
        n_train_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.expand(B, -1, -1, -1)

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
                    clip_skip_layer_indices=clip_skip_indices,
                )
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                         mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(active_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(active_params, 1.0)
                optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        # Validation
        student.eval()
        if ripd_unfrozen:
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
                        clip_skip_layer_indices=clip_skip_indices,
                    )
                    logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                             mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)
                val_loss += loss.item()
                n_val_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val   = val_loss   / max(n_val_batches,   1)
        phase     = "warmup" if (warmup > 0 and epoch < warmup) else "joint"
        print(f"Epoch {epoch+1:3d}/{args.epochs}  [{phase}]  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            wandb.log({
                "epoch":         epoch + 1,
                "train/loss":    avg_train,
                "val/loss":      avg_val,
                "phase":         phase,
                "best_val_loss": best_val_loss,
            }, step=epoch + 1)

        torch.save({
            "epoch":   epoch,
            "student": student.state_dict(),
            "ripd":    ripd.state_dict() if ripd_unfrozen else None,
            "clip_upsample1":   clip_upsample1.state_dict()   if ripd_unfrozen and clip_upsample1   is not None else None,
            "clip_upsample2":   clip_upsample2.state_dict()   if ripd_unfrozen and clip_upsample2   is not None else None,
            "dino_decod_proj1": dino_decod_proj1.state_dict() if ripd_unfrozen and dino_decod_proj1 is not None else None,
            "dino_decod_proj2": dino_decod_proj2.state_dict() if ripd_unfrozen and dino_decod_proj2 is not None else None,
            "val_loss": avg_val,
        }, os.path.join(args.output_dir, "baseline_latest.pth"))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.output_dir, "baseline_best.pth")
            torch.save({
                "epoch":   epoch,
                "student": student.state_dict(),
                "ripd":    ripd.state_dict() if ripd_unfrozen else None,
                "clip_upsample1":   clip_upsample1.state_dict()   if ripd_unfrozen and clip_upsample1   is not None else None,
                "clip_upsample2":   clip_upsample2.state_dict()   if ripd_unfrozen and clip_upsample2   is not None else None,
                "dino_decod_proj1": dino_decod_proj1.state_dict() if ripd_unfrozen and dino_decod_proj1 is not None else None,
                "dino_decod_proj2": dino_decod_proj2.state_dict() if ripd_unfrozen and dino_decod_proj2 is not None else None,
                "val_loss": avg_val,
            }, best_path)
            print(f"  ✓ New best val loss: {avg_val:.4f} → {best_path}")

    if use_wandb:
        wandb.finish()
    print(f"\nBaseline training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
