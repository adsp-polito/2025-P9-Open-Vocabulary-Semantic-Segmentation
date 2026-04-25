#!/usr/bin/env python3
"""
Self-Supervised Fine-Tuning of DINOv3 on LandDiscover-50K
Converted from self_supervised_tuning.ipynb for SLURM batch execution.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
import timm
import copy
import time
from tqdm import tqdm

# ==============================================================
# 0. Configuration and Utilities
# ==============================================================

# Paths (use absolute paths for batch job reliability)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DINOV3_CKPT_PATH = os.path.join(PROJECT_DIR, "dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
DATASET_DIR = os.path.join(PROJECT_DIR, "gs_net/data/datasets/LandDiscover_50K/TR_Image")
SAVE_DIR = os.path.join(PROJECT_DIR, "checkpoints/")

# Fine-tuning strategy
DATA_FRACTION = 0.25       # Use 25% of the dataset (~13K images)
NUM_UNFROZEN_BLOCKS = 2    # Only fine-tune last 2 transformer blocks (blocks 22-23 of 24)
EPOCHS = 20
IMG_SIZE = 384
BATCH_SIZE = 2
NUM_WORKERS = 2
LR = 5e-5
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.996
CHECKPOINT_EVERY = 5

# DINO-specific anti-collapse settings
OUT_DIM = 4096              # Projection output dim (was 256 — too small, caused collapse)
TEACHER_TEMP_INIT = 0.04    # Teacher temperature start
TEACHER_TEMP_FINAL = 0.07   # Teacher temperature end (warmup over TEACHER_WARMUP_EPOCHS)
TEACHER_WARMUP_EPOCHS = 5   # Epochs to linearly warmup teacher temp
STUDENT_TEMP = 0.1          # Student temperature (fixed)
CENTER_MOMENTUM = 0.9       # Momentum for teacher output centering
GRAD_CLIP = 3.0             # Max gradient norm

def save_model_every_n_epochs(model, optimizer, epoch, n=CHECKPOINT_EVERY, save_dir=SAVE_DIR):
    if (epoch + 1) % n == 0 or epoch == 0:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"dinov3_selfsup_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Checkpoint saved at: {save_path}")

# ==============================================================
# 1. Data Preparation and Augmentation
# ==============================================================

class LandDiscoverUnlabeledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        assert len(self.files) > 0, "No images found in dataset directory."
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        return img, img

strong_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

full_dataset = LandDiscoverUnlabeledDataset(DATASET_DIR, transform=strong_transform)
num_samples = int(len(full_dataset) * DATA_FRACTION)
indices = np.random.RandomState(42).permutation(len(full_dataset))[:num_samples]
dataset = Subset(full_dataset, indices)
print(f"Using {len(dataset)} / {len(full_dataset)} images ({DATA_FRACTION*100:.0f}%)")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# ==============================================================
# 2. Model, Teacher, and Projection Head Setup
# ==============================================================

class DINOHead(nn.Module):
    """DINO projection head with L2 normalization on output."""
    def __init__(self, in_dim=1024, out_dim=4096, hidden_dim=2048, nlayers=3):
        super().__init__()
        layers = []
        for i in range(nlayers):
            dim1 = in_dim if i == 0 else hidden_dim
            dim2 = out_dim if i == nlayers - 1 else hidden_dim
            layers.append(nn.Linear(dim1, dim2))
            if i < nlayers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, dim=-1, p=2)  # L2 normalize before loss


class DINOLoss(nn.Module):
    """DINO loss with teacher centering to prevent mode collapse.
    
    The center is a running mean of teacher outputs, subtracted before
    the teacher softmax. This prevents the teacher from collapsing to
    a single dominant dimension (Caron et al., 2021, Section 3.3).
    """
    def __init__(self, out_dim, center_momentum=0.9):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_out, teacher_out, student_temp, teacher_temp):
        # Teacher: center then sharpen
        teacher_centered = (teacher_out - self.center) / teacher_temp
        teacher_probs = F.softmax(teacher_centered, dim=-1)
        # Student: sharpen
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
        # Cross-entropy loss
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_out):
        """Update center with EMA of teacher outputs."""
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# DINOv3 ViT-L/16 backbone (24 blocks, 1024-dim)
student = timm.create_model(
    'vit_large_patch16_dinov3.sat493m',
    pretrained=False,
    num_classes=0,
    img_size=IMG_SIZE
)
student.load_state_dict(torch.load(DINOV3_CKPT_PATH, map_location='cpu'), strict=False)

# Freeze entire backbone first
for p in student.parameters():
    p.requires_grad = False

# Unfreeze only the last NUM_UNFROZEN_BLOCKS blocks
total_blocks = len(student.blocks)  # 24
for block_idx in range(total_blocks - NUM_UNFROZEN_BLOCKS, total_blocks):
    for p in student.blocks[block_idx].parameters():
        p.requires_grad = True

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
print(f"Backbone: {trainable:,} / {total:,} params trainable ({trainable/total*100:.1f}%)")
print(f"Unfrozen blocks: {total_blocks - NUM_UNFROZEN_BLOCKS} to {total_blocks - 1}")

# Teacher is an EMA copy of the student (fully frozen)
teacher = copy.deepcopy(student)
for p in teacher.parameters():
    p.requires_grad = False

# Projection heads (out_dim=4096 to prevent collapse)
student_head = DINOHead(in_dim=1024, out_dim=OUT_DIM, hidden_dim=2048, nlayers=3)
teacher_head = DINOHead(in_dim=1024, out_dim=OUT_DIM, hidden_dim=2048, nlayers=3)
teacher_head.load_state_dict(student_head.state_dict())
for p in teacher_head.parameters():
    p.requires_grad = False

# ==============================================================
# 3. Loss Functions, EMA Update, and Optimizer
# ==============================================================

# DINO loss with centering (replaces the old dino_loss function)
dino_criterion = DINOLoss(out_dim=OUT_DIM, center_momentum=CENTER_MOMENTUM)

def get_teacher_temp(epoch, batch_idx, num_batches):
    """Linear warmup of teacher temperature over TEACHER_WARMUP_EPOCHS."""
    total_warmup_steps = TEACHER_WARMUP_EPOCHS * num_batches
    current_step = epoch * num_batches + batch_idx
    if current_step >= total_warmup_steps:
        return TEACHER_TEMP_FINAL
    progress = current_step / total_warmup_steps
    return TEACHER_TEMP_INIT + (TEACHER_TEMP_FINAL - TEACHER_TEMP_INIT) * progress

def update_teacher(student, teacher, m=EMA_MOMENTUM):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = param_t.data * m + param_s.data * (1. - m)

def update_teacher_head(student_head, teacher_head, m=EMA_MOMENTUM):
    for param_s, param_t in zip(student_head.parameters(), teacher_head.parameters()):
        param_t.data = param_t.data * m + param_s.data * (1. - m)

trainable_params = [p for p in student.parameters() if p.requires_grad] + list(student_head.parameters())
optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
print(f"Optimizer: {sum(p.numel() for p in trainable_params):,} trainable parameters")

# ==============================================================
# 4. Training Loop
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = student.to(device)
teacher = teacher.to(device)
student_head = student_head.to(device)
teacher_head = teacher_head.to(device)
dino_criterion = dino_criterion.to(device)

# Print training configuration summary
print("=" * 60)
print("       SELF-SUPERVISED FINE-TUNING CONFIGURATION")
print("=" * 60)
print(f"  Device:              {device}")
if device.type == 'cuda':
    print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_mem = getattr(gpu_props, 'total_memory', getattr(gpu_props, 'total_mem', 0))
    print(f"  GPU Memory:          {gpu_mem / 1024**3:.1f} GB")
print(f"  Dataset:             {DATASET_DIR}")
print(f"  Data fraction:       {DATA_FRACTION*100:.0f}% ({len(dataset):,} / {len(full_dataset):,} images)")
print(f"  Batch size:          {BATCH_SIZE}")
print(f"  Batches per epoch:   {len(loader):,}")
print(f"  Total epochs:        {EPOCHS}")
print(f"  Unfrozen blocks:     last {NUM_UNFROZEN_BLOCKS} of {total_blocks} (blocks {total_blocks - NUM_UNFROZEN_BLOCKS}-{total_blocks - 1})")
trainable_count = sum(p.numel() for p in trainable_params)
print(f"  Trainable params:    {trainable_count:,}")
print(f"  Learning rate:       {LR}")
print(f"  Optimizer:           AdamW (weight_decay={WEIGHT_DECAY})")
print(f"  Scheduler:           CosineAnnealingLR (T_max={EPOCHS})")
print(f"  EMA momentum:        {EMA_MOMENTUM}")
print(f"  Projection out_dim:  {OUT_DIM}")
print(f"  Teacher temp:        {TEACHER_TEMP_INIT} -> {TEACHER_TEMP_FINAL} (warmup {TEACHER_WARMUP_EPOCHS} epochs)")
print(f"  Student temp:        {STUDENT_TEMP}")
print(f"  Center momentum:     {CENTER_MOMENTUM}")
print(f"  Gradient clipping:   {GRAD_CLIP}")
print(f"  Checkpoint every:    {CHECKPOINT_EVERY} epochs (+ epoch 1)")
print(f"  Save directory:      {SAVE_DIR}")
print("=" * 60)
sys.stdout.flush()

start_time = time.time()

for epoch in range(EPOCHS):
    student.train()
    student_head.train()
    teacher.eval()
    teacher_head.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=True)
    num_batches = len(loader)
    for batch_idx, (img1, img2) in enumerate(pbar):
        img1, img2 = img1.to(device), img2.to(device)
        # Get current teacher temperature (with warmup)
        cur_teacher_temp = get_teacher_temp(epoch, batch_idx, num_batches)
        # Student forward on both views
        s_out1 = student_head(student(img1))
        s_out2 = student_head(student(img2))
        # Teacher forward (no grad)
        with torch.no_grad():
            t_out1 = teacher_head(teacher(img1))
            t_out2 = teacher_head(teacher(img2))
        # Cross-view DINO loss with centering
        loss = 0.5 * (
            dino_criterion(s_out1, t_out2, STUDENT_TEMP, cur_teacher_temp) +
            dino_criterion(s_out2, t_out1, STUDENT_TEMP, cur_teacher_temp)
        )
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()
        # Update teacher center with both views
        with torch.no_grad():
            teacher_batch = torch.cat([t_out1, t_out2], dim=0)
            dino_criterion.update_center(teacher_batch)
        # EMA update for teacher
        update_teacher(student, teacher, m=EMA_MOMENTUM)
        update_teacher_head(student_head, teacher_head, m=EMA_MOMENTUM)
        total_loss += loss.item()
        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            t_temp=f"{cur_teacher_temp:.4f}"
        )

    scheduler.step()
    avg_loss = total_loss / len(loader)
    elapsed = time.time() - start_time
    eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
    print(f"  -> Epoch {epoch+1}/{EPOCHS} done | Avg Loss: {avg_loss:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
    sys.stdout.flush()
    save_model_every_n_epochs(student, optimizer, epoch, n=CHECKPOINT_EVERY, save_dir=SAVE_DIR)

total_time = time.time() - start_time
print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")

# ==============================================================
# 5. Save Final Backbone
# ==============================================================

final_path = os.path.join(SAVE_DIR, "dinov3_selfsup_backbone_final.pth")
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(student.state_dict(), final_path)
print(f"Final backbone saved at: {final_path}")
