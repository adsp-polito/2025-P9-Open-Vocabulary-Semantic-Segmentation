"""
Supervised Fine-Tuning v2 — Lower LR + L2 Regularization toward Init
======================================================================

Changes vs v1:
  - LR_BACKBONE: 1e-4 → 1e-5  (10x lower, reduces catastrophic forgetting)
  - L2_LAMBDA:   new           (penalises deviation from pretrained weights)
  - --output-dir: CLI arg so the calling sbatch knows the exact path

Hypothesis: v1's backbone LR was too aggressive, causing the model to forget
satellite pretraining by epoch 10. Lower LR + L2 penalty should allow more
epochs without performance degradation.

Usage:
    python finetune_supervised_unfreeze_v2.py --output-dir PATH
"""

import argparse
import os
import json
import csv
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import timm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_WEIGHTS = os.path.join(_SCRIPT_DIR, "../../dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")

DATASET_ROOT = os.path.join(_SCRIPT_DIR, "../data/datasets/LandDiscover_50K")
IMG_DIR      = os.path.join(DATASET_ROOT, "TR_Image")
GT_DIR       = os.path.join(DATASET_ROOT, "GT_ID")

NUM_CLASSES      = 40
IMG_SIZE         = 384
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 4
WARMUP_EPOCHS    = 2
TRAIN_EPOCHS     = 15          # 15 train epochs to match v1 comparison configs
TOTAL_EPOCHS     = WARMUP_EPOCHS + TRAIN_EPOCHS
SAVE_EVERY       = 5           # save backbone every 5 train epochs (ep 7, 12, 17)
DATA_FRACTION    = 1
DATA_SEED        = 42

NUM_UNFROZEN_BLOCKS = 2        # unfreeze last N transformer blocks

# --- Key hyperparameter changes vs v1 ---
LR_HEAD     = 1e-3             # unchanged
LR_BACKBONE = 1e-5             # was 1e-4 → 10x lower to reduce forgetting
WEIGHT_DECAY = 0.05
L2_LAMBDA    = 1e-2            # L2 penalty toward pretrained init (new in v2)

IGNORE_INDEX = 255
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# DATASET
# ==============================================================================

class LandDiscoverSegDataset(Dataset):
    def __init__(self, img_root, gt_root, img_size=384, augment=True):
        self.img_root   = img_root
        self.gt_root    = gt_root
        self.img_size   = img_size
        self.augment    = augment
        self.files      = sorted([
            f for f in os.listdir(img_root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        assert len(self.files) > 0, f"No images found in {img_root}"
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        self.normalize    = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name   = self.files[idx]
        img    = Image.open(os.path.join(self.img_root, name)).convert("RGB")
        gt_path = os.path.join(self.gt_root, name)
        if not os.path.exists(gt_path):
            base    = os.path.splitext(name)[0]
            gt_path = os.path.join(self.gt_root, base + ".png")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT mask not found for '{name}'")
        gt = Image.open(gt_path)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        gt  = gt.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, gt = TF.hflip(img), TF.hflip(gt)
            if random.random() > 0.5:
                img, gt = TF.vflip(img), TF.vflip(gt)
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = TF.rotate(img, angle=k * 90)
                gt  = TF.rotate(gt,  angle=k * 90)
            img = self.color_jitter(img)

        img = self.normalize(TF.to_tensor(img))
        gt  = torch.from_numpy(np.array(gt)).long()
        return img, gt


def create_subset(dataset, fraction, seed):
    rng         = np.random.RandomState(seed)
    num_samples = int(len(dataset) * fraction)
    indices     = rng.permutation(len(dataset))[:num_samples]
    print(f"Dataset: {len(indices)} / {len(dataset)} images ({fraction*100:.0f}%)")
    return Subset(dataset, indices)


# ==============================================================================
# SEGMENTATION HEAD
# ==============================================================================

class SegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, patch_size=16, img_size=384):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        patches = x[:, N - self.grid_size * self.grid_size:, :]
        patches = patches.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)
        logits  = self.head(patches)
        return F.interpolate(logits, size=(self.grid_size * 16, self.grid_size * 16),
                             mode="bilinear", align_corners=False)


# ==============================================================================
# TRAINING
# ==============================================================================

def train_one_epoch(backbone, seg_head, loader, optimizer, epoch, phase,
                    grad_accum_steps, backbone_init_state=None):
    backbone.train()
    seg_head.train()
    total_loss, num_batches = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    for i, (imgs, gts) in enumerate(pbar):
        imgs = imgs.to(DEVICE)
        gts  = gts.to(DEVICE)

        feats  = backbone.forward_features(imgs)
        logits = seg_head(feats)
        loss   = F.cross_entropy(logits, gts, ignore_index=IGNORE_INDEX)

        # L2 penalty toward pretrained init (only during unfreeze phase)
        if backbone_init_state is not None:
            l2 = sum(
                torch.sum((p - backbone_init_state[n]) ** 2)
                for n, p in backbone.named_parameters()
                if p.requires_grad and n in backbone_init_state
            )
            loss = loss + L2_LAMBDA * l2

        (loss / grad_accum_steps).backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


# ==============================================================================
# UTILITIES
# ==============================================================================

def save_checkpoint(exp_dir, epoch, backbone, seg_head, optimizer, loss, config, phase, scheduler=None):
    path = os.path.join(exp_dir, "checkpoints", f"epoch_{epoch:02d}.pth")
    ckpt = {"epoch": epoch, "phase": phase, "backbone": backbone.state_dict(),
            "seg_head": seg_head.state_dict(), "optimizer": optimizer.state_dict(),
            "train_loss": loss, "config": config}
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)
    print(f"  Saved checkpoint: {path}")


def save_backbone_for_gsnet(backbone, epoch, exp_dir):
    path = os.path.join(exp_dir, "backbone_for_gsnet", f"epoch_{epoch:02d}.pth")
    torch.save(backbone.state_dict(), path)
    print(f"  Saved backbone for GSNet: {path}")


def log_epoch(exp_dir, epoch, loss, phase):
    path       = os.path.join(exp_dir, "train_log.csv")
    write_hdr  = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(["epoch", "train_loss", "phase"])
        w.writerow([epoch, f"{loss:.6f}", phase])


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Experiment output directory (set by sbatch)")
    args = parser.parse_args()
    exp_dir = args.output_dir

    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "backbone_for_gsnet"), exist_ok=True)

    print("=" * 70)
    print("Supervised Fine-Tuning v2  (lower LR + L2 reg)")
    print(f"  LR_BACKBONE : {LR_BACKBONE}  (was 1e-4 in v1)")
    print(f"  L2_LAMBDA   : {L2_LAMBDA}")
    print(f"  Output      : {exp_dir}")
    print("=" * 70)

    config = {
        "version": "v2",
        "method": "supervised_partial_unfreeze",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "num_classes": NUM_CLASSES, "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
        "warmup_epochs": WARMUP_EPOCHS, "train_epochs": TRAIN_EPOCHS,
        "num_unfrozen_blocks": NUM_UNFROZEN_BLOCKS,
        "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY, "l2_lambda": L2_LAMBDA,
        "data_fraction": DATA_FRACTION, "data_seed": DATA_SEED,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Dataset ---
    print("\nLoading dataset...")
    dataset = create_subset(
        LandDiscoverSegDataset(IMG_DIR, GT_DIR, img_size=IMG_SIZE, augment=True),
        DATA_FRACTION, DATA_SEED
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    # --- Backbone ---
    print("\nLoading DINOv3 backbone...")
    backbone = timm.create_model(
        "vit_large_patch16_dinov3.sat493m", pretrained=False,
        num_classes=0, dynamic_img_size=True, img_size=IMG_SIZE,
    )
    pretrained = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
    if isinstance(pretrained, dict) and "teacher_state_dict" in pretrained:
        backbone.load_state_dict(pretrained["teacher_state_dict"], strict=False)
    elif isinstance(pretrained, dict) and "model" in pretrained:
        backbone.load_state_dict(pretrained["model"], strict=False)
    else:
        backbone.load_state_dict(pretrained, strict=False)
    print("Pretrained weights loaded.")

    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(DEVICE)

    # --- Segmentation head ---
    seg_head = SegHead(backbone.num_features, NUM_CLASSES).to(DEVICE)

    # ================================================================
    # PHASE 1: HEAD WARMUP
    # ================================================================
    print(f"\nPhase 1: Head warmup ({WARMUP_EPOCHS} epochs, backbone frozen)")
    opt_warmup = AdamW(seg_head.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    avg_loss = 0.0
    for epoch in range(1, WARMUP_EPOCHS + 1):
        avg_loss = train_one_epoch(backbone, seg_head, loader, opt_warmup,
                                   epoch, "warmup", GRAD_ACCUM_STEPS)
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f}")
        log_epoch(exp_dir, epoch, avg_loss, "warmup")
    save_checkpoint(exp_dir, WARMUP_EPOCHS, backbone, seg_head, opt_warmup,
                    avg_loss, config, "warmup")

    # ================================================================
    # PHASE 2: PARTIAL UNFREEZE + L2 REG
    # ================================================================
    print(f"\nPhase 2: Unfreeze last {NUM_UNFROZEN_BLOCKS} blocks + L2 reg "
          f"({TRAIN_EPOCHS} epochs, LR_BACKBONE={LR_BACKBONE})")

    for block in backbone.blocks[-NUM_UNFROZEN_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    # Snapshot init weights for L2 penalty
    backbone_init_state = {
        n: p.clone().detach()
        for n, p in backbone.named_parameters()
        if p.requires_grad
    }
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"  Unfrozen: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
    print(f"  L2 snapshot taken for {len(backbone_init_state)} parameter tensors")

    unfrozen_params = [p for b in backbone.blocks[-NUM_UNFROZEN_BLOCKS:]
                       for p in b.parameters()]
    optimizer = AdamW([
        {"params": seg_head.parameters(), "lr": LR_HEAD},
        {"params": unfrozen_params,       "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-6)

    for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
        avg_loss = train_one_epoch(backbone, seg_head, loader, optimizer,
                                   epoch, "unfreeze", GRAD_ACCUM_STEPS,
                                   backbone_init_state=backbone_init_state)
        scheduler.step()
        lr_h = optimizer.param_groups[0]["lr"]
        lr_b = optimizer.param_groups[1]["lr"]
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"LR head: {lr_h:.2e} | LR backbone: {lr_b:.2e}")
        log_epoch(exp_dir, epoch, avg_loss, "unfreeze")

        train_epoch_num = epoch - WARMUP_EPOCHS
        if train_epoch_num % SAVE_EVERY == 0 or epoch == TOTAL_EPOCHS:
            save_checkpoint(exp_dir, epoch, backbone, seg_head, optimizer,
                            avg_loss, config, "unfreeze", scheduler)
            save_backbone_for_gsnet(backbone, epoch, exp_dir)

    print("\nFine-tuning complete.")
    print(f"Backbone for GSNet: {exp_dir}/backbone_for_gsnet/epoch_{TOTAL_EPOCHS:02d}.pth")


if __name__ == "__main__":
    main()
