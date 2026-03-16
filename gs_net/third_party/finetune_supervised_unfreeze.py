"""
Supervised Fine-Tuning of DINOv3 ViT-L/16 Specialist Backbone with Partial Unfreezing
======================================================================================

This script fine-tunes the DINOv3 satellite-pretrained backbone on LandDiscover50K
by unfreezing the last 2 transformer blocks + a lightweight segmentation head.

Pipeline:
    1. Load DINOv3 ViT-L/16 satellite pretrained weights
    2. Freeze entire backbone
    3. Train segmentation head for 2 warmup epochs (backbone frozen)
    4. Unfreeze last 2 blocks, train jointly for remaining epochs
    5. Save backbone weights for use in GSNet

Output:
    experiments/sup_unfreeze_<timestamp>/
        config.json              - All hyperparameters
        train_log.csv            - Per-epoch loss
        checkpoints/epoch_XX.pth - Full checkpoints (resumable)
        backbone_for_gsnet/epoch_XX.pth - Backbone weights for GSNet

Usage:
    python finetune_supervised_unfreeze.py

Then test in GSNet:
    export RSIB_CKPT='experiments/sup_unfreeze_<timestamp>/backbone_for_gsnet/epoch_XX.pth'

Requirements:
    pip install timm torchvision tqdm numpy
"""

import os
import sys
import json
import csv
import time
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

PRETRAINED_WEIGHTS = "sftp://hassan@130.192.93.114/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation/dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

DATASET_ROOT = "sftp://hassan@130.192.93.114/nfs/home/hassan/2025-P9-Open-Vocabulary-Semantic-Segmentation/gs_net/data/datasets/LandDiscover_50K"
IMG_DIR = os.path.join(DATASET_ROOT, "TR_Image")   # RGB images
GT_DIR = os.path.join(DATASET_ROOT, "GT_ID")        # Segmentation masks

# Training config
NUM_CLASSES = 40
IMG_SIZE = 384
BATCH_SIZE = 1              # Physical batch size (limited by GPU memory)
GRAD_ACCUM_STEPS = 4        # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 4
WARMUP_EPOCHS = 2           # Head-only warmup
TRAIN_EPOCHS = 5            # Backbone + head training
TOTAL_EPOCHS = WARMUP_EPOCHS + TRAIN_EPOCHS  # 7 total
SAVE_EVERY = 1              # Save checkpoint every N epochs after warmup
DATA_FRACTION = 0.5         # Use 50% of dataset (~26K images)
DATA_SEED = 42              # Fixed seed for reproducible subset

# Unfreezing config
NUM_UNFROZEN_BLOCKS = 2     # Unfreeze last N transformer blocks

# Learning rates
LR_HEAD = 1e-3              # Segmentation head
LR_BACKBONE = 1e-4          # Unfrozen backbone blocks
WEIGHT_DECAY = 0.05

IGNORE_INDEX = 255
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# EXPERIMENT DIRECTORY
# ==============================================================================

def create_experiment_dir():
    """Create organized output directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/sup_unfreeze_{timestamp}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/backbone_for_gsnet", exist_ok=True)
    return exp_dir

# ==============================================================================
# DATASET
# ==============================================================================

class LandDiscoverSegDataset(Dataset):
    """LandDiscover50K dataset for supervised segmentation.
    
    Geometric augmentations (flip, rotation) are applied jointly to image and GT.
    Color augmentations and normalization are applied to image only.
    """

    def __init__(self, img_root, gt_root, img_size=384, augment=True):
        self.img_root = img_root
        self.gt_root = gt_root
        self.img_size = img_size
        self.augment = augment

        self.files = sorted([
            f for f in os.listdir(img_root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        assert len(self.files) > 0, f"No images found in {img_root}"

        # Color augmentation (image only)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        # Normalization (image only)
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        # Load image and GT
        img = Image.open(os.path.join(self.img_root, name)).convert("RGB")
        
        # GT file: try same name, then try swapping extension
        gt_path = os.path.join(self.gt_root, name)
        if not os.path.exists(gt_path):
            base = os.path.splitext(name)[0]
            gt_path = os.path.join(self.gt_root, base + ".png")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(
                f"GT mask not found for image '{name}'. "
                f"Tried: {os.path.join(self.gt_root, name)} and {gt_path}"
            )
        # Load GT raw - do NOT convert mode, pixel values are class IDs
        # (works correctly whether GT is mode 'L', 'P', or 'I')
        gt = Image.open(gt_path)

        # Resize both to same size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        gt = gt.resize((self.img_size, self.img_size), Image.NEAREST)

        # --- Joint geometric augmentations (applied to BOTH image and GT) ---
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                gt = TF.hflip(gt)

            if random.random() > 0.5:
                img = TF.vflip(img)
                gt = TF.vflip(gt)

            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = TF.rotate(img, angle=k * 90)
                gt = TF.rotate(gt, angle=k * 90)

            img = self.color_jitter(img)

        # Convert to tensors
        img = TF.to_tensor(img)
        img = self.normalize(img)
        gt = torch.from_numpy(np.array(gt)).long()

        return img, gt


def create_subset(dataset, fraction, seed):
    """
    Create a random subset of the dataset with fixed seed.
    Prints class coverage to verify all classes are present.
    """
    rng = np.random.RandomState(seed)
    num_samples = int(len(dataset) * fraction)
    indices = rng.permutation(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)

    print(f"Dataset subset: {len(subset)} / {len(dataset)} images ({fraction*100:.0f}%)")

    # Quick check: sample a few GTs to verify class coverage
    print("Verifying class coverage (sampling 500 images)...")
    check_indices = rng.choice(len(indices), min(500, len(indices)), replace=False)
    all_classes = set()
    for i in check_indices:
        _, gt = dataset[indices[i]]
        classes_in_img = set(gt.unique().numpy().tolist())
        classes_in_img.discard(IGNORE_INDEX)
        all_classes.update(classes_in_img)
    print(f"  Found {len(all_classes)} / {NUM_CLASSES} classes in sample")
    if len(all_classes) < NUM_CLASSES:
        missing = set(range(NUM_CLASSES)) - all_classes
        print(f"  Warning: classes {missing} not found in sample (may appear in full subset)")

    return subset


# ==============================================================================
# NOTE: Augmentations are handled inside LandDiscoverSegDataset.__getitem__()
# Geometric transforms (flip, rotation) are applied jointly to image AND GT.
# Color transforms (jitter) and normalization are applied to image only.
# ==============================================================================

# ==============================================================================
# SEGMENTATION HEAD
# ==============================================================================

class SegHead(nn.Module):
    """
    Lightweight segmentation head.
    3-layer conv decoder: embed_dim -> 256 -> num_classes.
    Uses GroupNorm (works with batch_size=1, unlike BatchNorm).
    """
    def __init__(self, embed_dim, num_classes, patch_size=16, img_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 384/16 = 24

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N = special_tokens + H*W patches
        Returns:
            logits: (B, num_classes, img_size, img_size)
        """
        B, N, C = x.shape
        expected_patches = self.grid_size * self.grid_size  # 576
        num_special = N - expected_patches  # CLS + registers

        # Skip special tokens, keep only patch tokens
        patches = x[:, num_special:, :]  # (B, 576, C)
        patches = patches.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)

        logits = self.head(patches)  # (B, num_classes, 24, 24)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=(self.grid_size * self.patch_size,
                                             self.grid_size * self.patch_size),
                               mode="bilinear", align_corners=False)
        return logits  # (B, num_classes, 384, 384)


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def save_config(exp_dir, config_dict):
    """Save experiment configuration to JSON."""
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {path}")


def save_checkpoint(exp_dir, epoch, backbone, seg_head, optimizer, train_loss, config):
    """Save full training checkpoint (resumable)."""
    path = os.path.join(exp_dir, "checkpoints", f"epoch_{epoch:02d}.pth")
    torch.save({
        "epoch": epoch,
        "backbone": backbone.state_dict(),
        "seg_head": seg_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "config": config,
    }, path)
    print(f"  Saved checkpoint: {path}")


def save_backbone_for_gsnet(backbone, epoch, exp_dir):
    """Save backbone-only weights in format compatible with DINOv3Wrapper."""
    path = os.path.join(exp_dir, "backbone_for_gsnet", f"epoch_{epoch:02d}.pth")
    torch.save(backbone.state_dict(), path)
    print(f"  Saved backbone for GSNet: {path}")


def log_epoch(exp_dir, epoch, train_loss, phase):
    """Append one row to the training log CSV."""
    path = os.path.join(exp_dir, "train_log.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "phase"])
        writer.writerow([epoch, f"{train_loss:.6f}", phase])


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_one_epoch(backbone, seg_head, loader, optimizer, epoch, phase, grad_accum_steps=4):
    """
    Train for one epoch with gradient accumulation.

    Args:
        backbone: ViT backbone
        seg_head: Segmentation head
        loader: DataLoader
        optimizer: Optimizer
        epoch: Current epoch number (for display)
        phase: "warmup" or "unfreeze" (for display)
        grad_accum_steps: Number of steps to accumulate gradients

    Returns:
        avg_loss: Average training loss for this epoch
    """
    backbone.train()
    seg_head.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    for i, (imgs, gts) in enumerate(pbar):
        imgs = imgs.to(DEVICE)
        gts = gts.to(DEVICE)

        # Forward through backbone
        feats = backbone.forward_features(imgs)  # (B, N, C)

        # Forward through segmentation head
        logits = seg_head(feats)  # (B, num_classes, 384, 384)

        # Loss
        loss = F.cross_entropy(logits, gts, ignore_index=IGNORE_INDEX)
        loss = loss / grad_accum_steps  # Scale for accumulation
        loss.backward()

        # Step optimizer every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps  # Undo scaling for logging
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("Supervised Fine-Tuning with Partial Unfreezing")
    print("=" * 80)

    # --- Create experiment directory ---
    exp_dir = create_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # --- Build config dict ---
    config = {
        "method": "supervised_partial_unfreeze",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "dataset_root": DATASET_ROOT,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "warmup_epochs": WARMUP_EPOCHS,
        "train_epochs": TRAIN_EPOCHS,
        "total_epochs": TOTAL_EPOCHS,
        "num_unfrozen_blocks": NUM_UNFROZEN_BLOCKS,
        "data_fraction": DATA_FRACTION,
        "data_seed": DATA_SEED,
        "lr_head": LR_HEAD,
        "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY,
        "augmentations": ["hflip", "vflip", "rot90", "colorjitter"],
    }
    save_config(exp_dir, config)

    # --- Dataset ---
    print("\nLoading dataset...")
    full_dataset = LandDiscoverSegDataset(IMG_DIR, GT_DIR, img_size=IMG_SIZE, augment=True)
    dataset = create_subset(full_dataset, DATA_FRACTION, DATA_SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    # --- Backbone ---
    print("\nLoading DINOv3 backbone...")
    backbone = timm.create_model(
        "vit_large_patch16_dinov3.sat493m",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        img_size=IMG_SIZE,
    )

    # Load pretrained weights
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
    if isinstance(ckpt, dict) and "teacher_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["teacher_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        backbone.load_state_dict(ckpt["model"], strict=False)
    else:
        backbone.load_state_dict(ckpt, strict=False)
    print("Pretrained weights loaded.")

    # Freeze entire backbone
    for p in backbone.parameters():
        p.requires_grad = False

    backbone = backbone.to(DEVICE)

    # Print backbone info
    total_blocks = len(backbone.blocks)
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone: ViT-L/16, {total_blocks} blocks, {total_params:,} params (all frozen)")

    # --- Segmentation head ---
    seg_head = SegHead(
        embed_dim=backbone.num_features,  # 1024 for ViT-L
        num_classes=NUM_CLASSES,
        patch_size=16,
        img_size=IMG_SIZE,
    ).to(DEVICE)

    # ==================================================================
    # PHASE 1: HEAD WARMUP (backbone fully frozen, only head trains)
    # ==================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 1: Head Warmup ({WARMUP_EPOCHS} epochs)")
    print("=" * 80)

    optimizer_warmup = AdamW(seg_head.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        avg_loss = train_one_epoch(backbone, seg_head, loader, optimizer_warmup,
                                   epoch=epoch, phase="warmup", grad_accum_steps=GRAD_ACCUM_STEPS)
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f}")
        log_epoch(exp_dir, epoch, avg_loss, "warmup")

    # Save warmup checkpoint
    save_checkpoint(exp_dir, WARMUP_EPOCHS, backbone, seg_head, optimizer_warmup, avg_loss, config)
    print("Head warmup complete.\n")

    # ==================================================================
    # PHASE 2: PARTIAL UNFREEZE + HEAD TRAINING
    # ==================================================================
    print("=" * 80)
    print(f"PHASE 2: Unfreeze last {NUM_UNFROZEN_BLOCKS} blocks + Head ({TRAIN_EPOCHS} epochs)")
    print("=" * 80)

    # Unfreeze last N blocks
    for block in backbone.blocks[-NUM_UNFROZEN_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Unfrozen: {trainable_params:,} / {total_params:,} backbone params ({100*trainable_params/total_params:.1f}%)")

    # Optimizer for unfrozen backbone blocks + head
    unfrozen_params = []
    for block in backbone.blocks[-NUM_UNFROZEN_BLOCKS:]:
        unfrozen_params.extend(block.parameters())

    optimizer = AdamW([
        {"params": seg_head.parameters(), "lr": LR_HEAD},
        {"params": unfrozen_params, "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine LR scheduler (per-epoch stepping)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-6)

    for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
        avg_loss = train_one_epoch(backbone, seg_head, loader, optimizer,
                                   epoch=epoch, phase="unfreeze", grad_accum_steps=GRAD_ACCUM_STEPS)

        # Step scheduler once per epoch
        scheduler.step()

        current_lr_head = optimizer.param_groups[0]["lr"]
        current_lr_backbone = optimizer.param_groups[1]["lr"]
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | LR head: {current_lr_head:.2e} | LR backbone: {current_lr_backbone:.2e}")
        log_epoch(exp_dir, epoch, avg_loss, "unfreeze")

        # Save checkpoint and backbone for GSNet
        if (epoch - WARMUP_EPOCHS) % SAVE_EVERY == 0 or epoch == TOTAL_EPOCHS:
            save_checkpoint(exp_dir, epoch, backbone, seg_head, optimizer, avg_loss, config)
            save_backbone_for_gsnet(backbone, epoch, exp_dir)

    # ==================================================================
    # FINAL SAVE
    # ==================================================================
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nExperiment directory: {exp_dir}")
    print(f"\nTo test in GSNet:")
    print(f"  export RSIB_CKPT='{exp_dir}/backbone_for_gsnet/epoch_{TOTAL_EPOCHS:02d}.pth'")
    print(f"\nTo compare different epochs, try each backbone_for_gsnet/epoch_XX.pth")


if __name__ == "__main__":
    main()
