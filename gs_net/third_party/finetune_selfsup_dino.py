"""
Self-Supervised Fine-Tuning of DINOv3 ViT-L/16 using DINO Self-Distillation
=============================================================================

This script fine-tunes the DINOv3 satellite-pretrained backbone on LandDiscover50K
using DINO-style teacher-student self-distillation (no labels needed).

Pipeline:
    1. Load DINOv3 ViT-L/16 satellite pretrained weights for both student and teacher
    2. Train projection heads for 2 warmup epochs (backbone frozen)
    3. Unfreeze last 2 backbone blocks, train jointly for remaining epochs
    4. Teacher updated via EMA of student
    5. Save backbone weights for use in GSNet

Output:
    output/selfsup_finetuning/selfsup_dino_<timestamp>/
        config.json              - All hyperparameters
        train_log.csv            - Per-epoch loss
        checkpoints/epoch_XX.pth - Full checkpoints (resumable)
        backbone_for_gsnet/epoch_XX.pth - Student backbone weights for GSNet

Usage:
    python finetune_selfsup_dino.py

Then test in GSNet:
    export RSIB_CKPT='output/selfsup_finetuning/selfsup_dino_<timestamp>/backbone_for_gsnet/epoch_XX.pth'

Requirements:
    pip install timm torchvision tqdm numpy

Changes from previous self-supervised script (self_supervised_tuning.py):
==========================================================================
The previous attempt produced WORSE results than no fine-tuning at all
(mIoU dropped from 19.82 to 14.31 after 5 epochs). Here is what changed and why:

- Data fraction: 25% (~13K) → 50% (~26K). More diversity, matches supervised scripts.
- Batch size: 2 with no accumulation → 1 + accumulation 4 (effective 4). Matches supervised scripts, smoother gradients.
- LR backbone: 5e-5 → 1e-5. 5x lower because self-supervised gradients are noisier than CE, and DINOv3 is already well-trained.
- EMA momentum: 0.996 → 0.999. Slower teacher updates = more stable target.
- Teacher temp: 0.04→0.07 warmup → 0.04 fixed. Warmup caused a loss spike at epoch 5 in the previous run.
- Head warmup: None → 2 epochs. Stabilizes heads before touching backbone.
- Gradient accumulation: None → 4 steps. Smoother gradients.
- Weight decay: 1e-4 → 0.05. Stronger regularization, matches supervised scripts.
- EMA teacher update: Always → Only when backbone is unfrozen. During warmup the student backbone doesnt change, so the teacher shouldnt drift.
- Total epochs: 20 → 7 (2 warmup + 5 train). Matches supervised scripts for fair comparison.
- Experiment output: Single save dir → Full experiment dir with config/logs/auto GSNet export. Matches supervised scripts.
"""

import os
import sys
import json
import csv
import copy
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
from PIL import Image
from tqdm import tqdm

import timm

# ==============================================================================
# CONFIGURATION - CHANGE THESE PATHS AS NEEDED
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output", "selfsup_finetuning")

PRETRAINED_WEIGHTS = os.path.join(
    PROJECT_DIR,
    "dinov3",
    "vitl16-sat493m",
    "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "gs_net",
    "data",
    "datasets",
    "LandDiscover_50K",
    "TR_Image",
)  # Only images, no labels needed

# Training config
IMG_SIZE = 384
BATCH_SIZE = 1              # Physical batch size (limited by GPU memory)
GRAD_ACCUM_STEPS = 4        # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 4
WARMUP_EPOCHS = 2           # Head-only warmup (backbone frozen)
TRAIN_EPOCHS = 15            # Backbone + head training
TOTAL_EPOCHS = WARMUP_EPOCHS + TRAIN_EPOCHS  # 17 total (matches supervised scripts)
SAVE_EVERY = 1              # Save checkpoint every N epochs after warmup
DATA_FRACTION = 0.5         # Use 50% of dataset (~26K images, matches supervised scripts)
DATA_SEED = 42              # Fixed seed for reproducible subset

# Backbone unfreezing
NUM_UNFROZEN_BLOCKS = 2     # Unfreeze last 2 transformer blocks

# Learning rates (lower than supervised — self-supervised gradients are noisier)
LR_BACKBONE = 1e-5          # Unfrozen backbone blocks
LR_HEADS = 5e-4             # Projection heads
WEIGHT_DECAY = 0.05         # Matches supervised scripts

# DINO-specific settings
OUT_DIM = 4096              # Projection head output dimension
HIDDEN_DIM = 2048           # Projection head hidden dimension
NUM_HEAD_LAYERS = 3         # Projection head depth
EMA_MOMENTUM = 0.999        # Teacher EMA momentum (higher = more stable teacher)
TEACHER_TEMP = 0.04         # Fixed teacher temperature (no warmup — avoids instability)
STUDENT_TEMP = 0.1          # Student temperature
CENTER_MOMENTUM = 0.9       # Centering momentum (anti-collapse)
GRAD_CLIP = 3.0             # Max gradient norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# EXPERIMENT DIRECTORY
# ==============================================================================

def create_experiment_dir():
    """Create organized output directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"selfsup_dino_{timestamp}"
    exp_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/backbone_for_gsnet", exist_ok=True)
    return exp_dir


# ==============================================================================
# DATASET (no labels needed)
# ==============================================================================

class LandDiscoverUnlabeledDataset(Dataset):
    """
    LandDiscover50K dataset for self-supervised learning.
    Returns two different augmented views of the same image (DINO-style).
    """

    def __init__(self, img_dir, img_size=384):
        self.img_dir = img_dir
        self.img_size = img_size
        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        assert len(self.files) > 0, f"No images found in {img_dir}"

        # Strong augmentation for both views (different random crops/transforms)
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.25),
            T.RandomApply([T.RandomRotation(degrees=(180, 180))], p=0.25),
            T.RandomApply([T.RandomRotation(degrees=(270, 270))], p=0.25),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        # Return two differently augmented views of the same image
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


def create_subset(dataset, fraction, seed):
    """Create a random subset of the dataset with fixed seed."""
    rng = np.random.RandomState(seed)
    num_samples = int(len(dataset) * fraction)
    indices = rng.permutation(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)
    print(f"Dataset subset: {len(subset)} / {len(dataset)} images ({fraction*100:.0f}%)")
    return subset


# ==============================================================================
# DINO COMPONENTS
# ==============================================================================

class DINOHead(nn.Module):
    """DINO projection head: MLP with L2 normalization on output."""

    def __init__(self, in_dim=1024, out_dim=4096, hidden_dim=2048, nlayers=3):
        super().__init__()
        layers = []
        for i in range(nlayers):
            dim_in = in_dim if i == 0 else hidden_dim
            dim_out = out_dim if i == nlayers - 1 else hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if i < nlayers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, dim=-1, p=2)


class DINOLoss(nn.Module):
    """
    DINO cross-entropy loss with teacher centering to prevent mode collapse.
    (Caron et al., 2021, Section 3.3)
    """

    def __init__(self, out_dim, center_momentum=0.9):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out, student_temp, teacher_temp):
        # Teacher: center then sharpen
        teacher_probs = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        # Student: sharpen
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
        # Cross-entropy
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        """Update center with EMA of teacher outputs."""
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


@torch.no_grad()
def update_ema(student, teacher, momentum):
    """Update teacher parameters as EMA of student."""
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data = p_t.data * momentum + p_s.data * (1 - momentum)


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def save_config(exp_dir, config_dict):
    """Save experiment configuration to JSON."""
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {path}")


def save_checkpoint(exp_dir, epoch, student, teacher, student_head, teacher_head,
                    optimizer, dino_criterion, train_loss, config):
    """Save full training checkpoint (resumable)."""
    path = os.path.join(exp_dir, "checkpoints", f"epoch_{epoch:02d}.pth")
    torch.save({
        "epoch": epoch,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "student_head": student_head.state_dict(),
        "teacher_head": teacher_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "center": dino_criterion.center,
        "train_loss": train_loss,
        "config": config,
    }, path)
    print(f"  Saved checkpoint: {path}")


def save_backbone_for_gsnet(student, epoch, exp_dir):
    """Save student backbone weights for GSNet (no heads, no teacher)."""
    path = os.path.join(exp_dir, "backbone_for_gsnet", f"epoch_{epoch:02d}.pth")
    torch.save(student.state_dict(), path)
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


def log_run_summary(exp_dir, config, final_loss, duration_minutes):
    """Append one row to global run summary CSV under output root."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, "run_summary.csv")
    run_id = os.path.basename(exp_dir)
    final_checkpoint = os.path.join(exp_dir, "checkpoints", f"epoch_{TOTAL_EPOCHS:02d}.pth")
    final_backbone = os.path.join(exp_dir, "backbone_for_gsnet", f"epoch_{TOTAL_EPOCHS:02d}.pth")

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp",
                "run_id",
                "exp_dir",
                "final_loss",
                "duration_minutes",
                "data_fraction",
                "warmup_epochs",
                "train_epochs",
                "total_epochs",
                "num_unfrozen_blocks",
                "lr_heads",
                "lr_backbone",
                "weight_decay",
                "ema_momentum",
                "teacher_temp",
                "student_temp",
                "final_checkpoint",
                "final_backbone",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            run_id,
            exp_dir,
            f"{final_loss:.6f}",
            f"{duration_minutes:.2f}",
            config["data_fraction"],
            config["warmup_epochs"],
            config["train_epochs"],
            config["total_epochs"],
            config["num_unfrozen_blocks"],
            config["lr_heads"],
            config["lr_backbone"],
            config["weight_decay"],
            config["ema_momentum"],
            config["teacher_temp"],
            config["student_temp"],
            final_checkpoint,
            final_backbone,
        ])


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_one_epoch(student, teacher, student_head, teacher_head,
                    dino_criterion, loader, optimizer, epoch, phase,
                    grad_accum_steps=4, unfreeze_backbone=False):
    """
    Train one epoch of DINO self-distillation with gradient accumulation.

    Args:
        student/teacher: ViT backbone (student is trained, teacher is EMA)
        student_head/teacher_head: Projection heads
        dino_criterion: DINOLoss with centering
        loader: DataLoader returning (view1, view2) pairs
        optimizer: Optimizer
        epoch: Current epoch (for display)
        phase: "warmup" or "dino" (for display)
        grad_accum_steps: Gradient accumulation steps
        unfreeze_backbone: Whether backbone blocks are being trained

    Returns:
        avg_loss: Average training loss
    """
    student.train()
    student_head.train()
    teacher.eval()
    teacher_head.eval()

    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    for i, (view1, view2) in enumerate(pbar):
        view1 = view1.to(DEVICE)
        view2 = view2.to(DEVICE)

        # Student forward on both views
        s_feat1 = student(view1)  # CLS token (B, 1024)
        s_feat2 = student(view2)
        s_out1 = student_head(s_feat1)
        s_out2 = student_head(s_feat2)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_feat1 = teacher(view1)
            t_feat2 = teacher(view2)
            t_out1 = teacher_head(t_feat1)
            t_out2 = teacher_head(t_feat2)

        # Cross-view DINO loss: student(view1) matches teacher(view2) and vice versa
        loss = 0.5 * (
            dino_criterion(s_out1, t_out2, STUDENT_TEMP, TEACHER_TEMP) +
            dino_criterion(s_out2, t_out1, STUDENT_TEMP, TEACHER_TEMP)
        )
        loss = loss / grad_accum_steps
        loss.backward()

        # Gradient clipping and optimizer step (every grad_accum_steps)
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            trainable_params = [p for p in student.parameters() if p.requires_grad] + \
                               list(student_head.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            # EMA update for teacher (sync with optimizer step, not every batch)
            if unfreeze_backbone:
                update_ema(student, teacher, EMA_MOMENTUM)
            update_ema(student_head, teacher_head, EMA_MOMENTUM)

        # Update teacher center with both views (every batch is fine for centering)
        with torch.no_grad():
            teacher_batch = torch.cat([t_out1, t_out2], dim=0)
            dino_criterion.update_center(teacher_batch)

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("Self-Supervised Fine-Tuning with DINO Self-Distillation")
    print("=" * 80)

    # --- Create experiment directory ---
    exp_dir = create_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # --- Build config dict ---
    run_start_time = time.time()
    config = {
        "method": "selfsup_dino",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "dataset_dir": DATASET_DIR,
        "output_root": OUTPUT_ROOT,
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
        "lr_backbone": LR_BACKBONE,
        "lr_heads": LR_HEADS,
        "weight_decay": WEIGHT_DECAY,
        "out_dim": OUT_DIM,
        "ema_momentum": EMA_MOMENTUM,
        "teacher_temp": TEACHER_TEMP,
        "student_temp": STUDENT_TEMP,
        "center_momentum": CENTER_MOMENTUM,
        "grad_clip": GRAD_CLIP,
    }
    save_config(exp_dir, config)

    # --- Dataset ---
    print("\nLoading dataset...")
    full_dataset = LandDiscoverUnlabeledDataset(DATASET_DIR, img_size=IMG_SIZE)
    dataset = create_subset(full_dataset, DATA_FRACTION, DATA_SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    # --- Student backbone ---
    print("\nLoading DINOv3 student backbone...")
    student = timm.create_model(
        "vit_large_patch16_dinov3.sat493m",
        pretrained=False,
        num_classes=0,
        img_size=IMG_SIZE,
    )

    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
    if isinstance(ckpt, dict) and "teacher_state_dict" in ckpt:
        student.load_state_dict(ckpt["teacher_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        student.load_state_dict(ckpt["model"], strict=False)
    else:
        student.load_state_dict(ckpt, strict=False)
    print(f"Student loaded: embed_dim={student.num_features}, blocks={len(student.blocks)}")

    # Freeze entire student backbone initially
    for p in student.parameters():
        p.requires_grad = False

    student = student.to(DEVICE)

    # --- Teacher backbone (EMA copy of student, always frozen) ---
    print("Creating teacher as EMA copy of student...")
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(DEVICE)

    # --- Projection heads ---
    embed_dim = student.num_features  # 1024 for ViT-L
    student_head = DINOHead(in_dim=embed_dim, out_dim=OUT_DIM,
                            hidden_dim=HIDDEN_DIM, nlayers=NUM_HEAD_LAYERS).to(DEVICE)
    teacher_head = DINOHead(in_dim=embed_dim, out_dim=OUT_DIM,
                            hidden_dim=HIDDEN_DIM, nlayers=NUM_HEAD_LAYERS).to(DEVICE)
    # Initialize teacher head from student head
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_head.parameters():
        p.requires_grad = False

    # --- DINO loss ---
    dino_criterion = DINOLoss(out_dim=OUT_DIM, center_momentum=CENTER_MOMENTUM).to(DEVICE)

    # ==================================================================
    # PHASE 1: HEAD WARMUP (backbone frozen, only projection heads train)
    # ==================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 1: Head Warmup ({WARMUP_EPOCHS} epochs)")
    print("=" * 80)

    optimizer_warmup = AdamW(student_head.parameters(), lr=LR_HEADS, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        avg_loss = train_one_epoch(
            student, teacher, student_head, teacher_head,
            dino_criterion, loader, optimizer_warmup,
            epoch=epoch, phase="warmup",
            grad_accum_steps=GRAD_ACCUM_STEPS, unfreeze_backbone=False,
        )
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f}")
        log_epoch(exp_dir, epoch, avg_loss, "warmup")

    # Save warmup checkpoint
    save_checkpoint(exp_dir, WARMUP_EPOCHS, student, teacher, student_head, teacher_head,
                    optimizer_warmup, dino_criterion, avg_loss, config)
    print("Head warmup complete.\n")

    # ==================================================================
    # PHASE 2: BACKBONE + HEAD TRAINING
    # ==================================================================
    print("=" * 80)
    print(f"PHASE 2: Unfreeze last {NUM_UNFROZEN_BLOCKS} blocks + Heads ({TRAIN_EPOCHS} epochs)")
    print("=" * 80)

    # Unfreeze last N blocks of student
    total_blocks = len(student.blocks)
    for block in student.blocks[-NUM_UNFROZEN_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    trainable_backbone = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_backbone = sum(p.numel() for p in student.parameters())
    print(f"Unfrozen: {trainable_backbone:,} / {total_backbone:,} backbone params "
          f"({100*trainable_backbone/total_backbone:.1f}%)")

    # Optimizer for unfrozen backbone blocks + projection heads
    unfrozen_params = []
    for block in student.blocks[-NUM_UNFROZEN_BLOCKS:]:
        unfrozen_params.extend(block.parameters())

    optimizer = AdamW([
        {"params": student_head.parameters(), "lr": LR_HEADS},
        {"params": unfrozen_params, "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine LR scheduler (per-epoch stepping)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-7)

    for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
        avg_loss = train_one_epoch(
            student, teacher, student_head, teacher_head,
            dino_criterion, loader, optimizer,
            epoch=epoch, phase="dino",
            grad_accum_steps=GRAD_ACCUM_STEPS, unfreeze_backbone=True,
        )

        current_lr_heads = optimizer.param_groups[0]["lr"]
        current_lr_backbone = optimizer.param_groups[1]["lr"]
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"LR heads: {current_lr_heads:.2e} | LR backbone: {current_lr_backbone:.2e}")
        log_epoch(exp_dir, epoch, avg_loss, "dino")

        scheduler.step()

        # Save checkpoint and backbone for GSNet
        if (epoch - WARMUP_EPOCHS) % SAVE_EVERY == 0 or epoch == TOTAL_EPOCHS:
            save_checkpoint(exp_dir, epoch, student, teacher, student_head, teacher_head,
                            optimizer, dino_criterion, avg_loss, config)
            save_backbone_for_gsnet(student, epoch, exp_dir)

    # ==================================================================
    # FINAL SAVE
    # ==================================================================
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    duration_minutes = (time.time() - run_start_time) / 60.0
    log_run_summary(exp_dir, config, final_loss=avg_loss, duration_minutes=duration_minutes)

    print(f"\nExperiment directory: {exp_dir}")
    print(f"Global run summary: {os.path.join(OUTPUT_ROOT, 'run_summary.csv')}")
    print(f"\nTo test in GSNet:")
    print(f"  export RSIB_CKPT='{exp_dir}/backbone_for_gsnet/epoch_{TOTAL_EPOCHS:02d}.pth'")
    print(f"\nTo compare different epochs, try each backbone_for_gsnet/epoch_XX.pth")


if __name__ == "__main__":
    main()
