import os
import torch
import timm
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ======================
# CONFIG
# ======================
ROOT = "../data/datasets/LandDiscover_50K"
IMG_DIR = os.path.join(ROOT, "TR_Image")
GT_DIR = os.path.join(ROOT, "GT_ID")
# PRETRAINED_WEIGHTS = "../../dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
PRETRAINED_WEIGHTS = "./finetune_landdiscover_seg_epoch_15.pth"

NUM_CLASSES = 40          # adjust if needed
IMG_SIZE = 384
BATCH_SIZE = 1            # ViT-L + segmentation is heavy
EPOCHS = 5                # Stage 2: epochs 16-20
START_EPOCH = 15          # Starting from epoch 15
NUM_UNFROZEN_BLOCKS = 4   # Unfreeze last 4 blocks (stage 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = 255        # common for segmentation

# ======================
# DATASET
# ======================
class LandDiscoverSegDataset(Dataset):
    def __init__(self, img_root, gt_root, transform=None):
        self.img_root = img_root
        self.gt_root = gt_root
        self.transform = transform

        self.files = sorted([
            f for f in os.listdir(img_root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        assert len(self.files) > 0, "❌ No images found in TR_Image"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(self.img_root, name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_root, name))

        if self.transform:
            img = self.transform(img)

        gt = torch.from_numpy(np.array(gt)).long()
        return img, gt

# ======================
# TRANSFORMS
# ======================
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ======================
# DATALOADER
# ======================
dataset = LandDiscoverSegDataset(IMG_DIR, GT_DIR, train_transform)
print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ======================
# BACKBONE
# ======================
backbone = timm.create_model(
    "vit_large_patch16_dinov3.sat493m",
    pretrained=False,
    num_classes=0,
    dynamic_img_size=True,
    img_size=IMG_SIZE
).to(DEVICE)

ckpt = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
if "teacher_state_dict" in ckpt:
    backbone.load_state_dict(ckpt["teacher_state_dict"], strict=False)
elif "backbone" in ckpt:
    # Fine-tuned checkpoint format with separate backbone/seg_head
    backbone.load_state_dict(ckpt["backbone"], strict=False)
    print(f"✓ Loaded backbone from fine-tuned checkpoint (epoch {ckpt.get('epoch', '?')})")
else:
    backbone.load_state_dict(ckpt, strict=False)

# Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False

# Unfreeze last N ViT blocks (gradual unfreezing)
print(f"Unfreezing last {NUM_UNFROZEN_BLOCKS} blocks (blocks {24 - NUM_UNFROZEN_BLOCKS} to 23)...")
for block in backbone.blocks[-NUM_UNFROZEN_BLOCKS:]:
    for p in block.parameters():
        p.requires_grad = True

# Count trainable parameters
total_params = sum(p.numel() for p in backbone.parameters())
trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Backbone parameters: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.1f}%)")

# ======================
# SEGMENTATION HEAD
# ======================
class ViTSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # Handle both 2D [B, C] and 3D [B, N, C] outputs
        if len(x.shape) == 2:
            # [B, C] -> need to get spatial features differently
            # This shouldn't happen with num_classes=0, but handle it
            raise ValueError(f"Unexpected 2D output from backbone: {x.shape}")

        # x: [B, N, C] where N = 1 (CLS) + num_registers + H*W (patches)
        B, N, C = x.shape

        # Calculate expected number of spatial patches
        # For 384x384 with patch_size=16: (384/16)^2 = 576
        expected_patches = (384 // 16) ** 2

        # Determine number of special tokens (CLS + registers)
        # Total tokens = CLS + registers + spatial patches
        num_special_tokens = N - expected_patches

        # Extract only the spatial patch tokens (skip CLS and register tokens)
        x = x[:, num_special_tokens:, :]

        N_patches = x.shape[1]
        H = W = int(N_patches ** 0.5)

        # Verify we have a perfect square
        if H * W != N_patches:
            raise ValueError(f"Number of patches {N_patches} is not a perfect square. Got {H}x{W}={H*W}")

        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.conv(x)

seg_head = ViTSegHead(backbone.num_features, NUM_CLASSES).to(DEVICE)

# ======================
# OPTIMIZER & LOSS
# ======================
# Learning rate schedule for gradual unfreezing
# Stage 1 (2 blocks): 1e-4
# Stage 2 (4 blocks): 5e-5
# Stage 3 (8 blocks): 2e-5
# Stage 4 (12 blocks): 1e-5
# Stage 5 (24 blocks): 5e-6

if NUM_UNFROZEN_BLOCKS == 2:
    backbone_lr = 1e-4
    seg_head_lr = 1e-3
elif NUM_UNFROZEN_BLOCKS == 4:
    backbone_lr = 5e-5
    seg_head_lr = 5e-4  # Also reduce seg_head LR
elif NUM_UNFROZEN_BLOCKS == 8:
    backbone_lr = 2e-5
    seg_head_lr = 2e-4
elif NUM_UNFROZEN_BLOCKS == 12:
    backbone_lr = 1e-5
    seg_head_lr = 1e-4
else:  # Full unfreezing (24 blocks)
    backbone_lr = 5e-6
    seg_head_lr = 5e-5

print(f"Learning rates: backbone={backbone_lr}, seg_head={seg_head_lr}")

optimizer = AdamW(
    [
        {"params": seg_head.parameters(), "lr": seg_head_lr},
        {"params": backbone.blocks[-NUM_UNFROZEN_BLOCKS:].parameters(), "lr": backbone_lr},
    ],
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# ======================
# TRAINING LOOP
# ======================
print(f"🚀 Starting fine-tuning from epoch {START_EPOCH + 1}...")

for epoch in range(EPOCHS):
    backbone.train()
    seg_head.train()

    total_loss = 0.0

    # Progress bar for batches
    current_epoch = START_EPOCH + epoch + 1
    pbar = tqdm(loader, desc=f"Epoch {current_epoch}/{START_EPOCH + EPOCHS}")

    for i, (imgs, gts) in enumerate(pbar):
        imgs = imgs.to(DEVICE)
        gts = gts.to(DEVICE)

        feats = backbone.forward_features(imgs)  # [B, N, C] - get patch tokens

        # Debug print for first batch of first epoch
        if epoch == 0 and i == 0:
            print(f"\nDebug: feats shape = {feats.shape}")
            print(f"Debug: expected patches = {(IMG_SIZE // 16) ** 2} + 1 CLS = {(IMG_SIZE // 16) ** 2 + 1}")

        logits = seg_head(feats)             # [B, C, H, W]

        logits = nn.functional.interpolate(
            logits, size=gts.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = criterion(logits, gts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with current losss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    current_epoch = START_EPOCH + epoch + 1
    print(f"Epoch [{current_epoch}/{START_EPOCH + EPOCHS}] | Avg Loss: {avg_loss:.4f}")

    if current_epoch % 5 == 0:
        torch.save(
            {
                "epoch": current_epoch - 1,
                "backbone": backbone.state_dict(),
                "seg_head": seg_head.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"finetune_landdiscover_seg_epoch_{current_epoch}.pth"
        )

# Save final model (two versions)
print("💾 Saving final fine-tuned weights...")

final_epoch = START_EPOCH + EPOCHS

# 1. Save complete checkpoint (for resuming training)
torch.save(
    {
        "epoch": final_epoch - 1,
        "backbone": backbone.state_dict(),
        "seg_head": seg_head.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    f"finetune_landdiscover_seg_epoch_{final_epoch}.pth"
)

# 2. Save ONLY backbone in wrapper-compatible format (for replacing in dinov3 folder)
# This matches the format that DINOv3Wrapper._extract_state_dict() expects
torch.save(backbone.state_dict(), f"dinov3_finetuned_backbone_only_epoch_{final_epoch}.pth")

print("✅ Fine-tuning completed!")
print(f"   - Full checkpoint: 'finetune_landdiscover_seg_epoch_{final_epoch}.pth'")
print(f"   - Backbone only: 'dinov3_finetuned_backbone_only_epoch_{final_epoch}.pth'")
print(f"   → Use 'dinov3_finetuned_backbone_only_epoch_{final_epoch}.pth' as RSIB_CKPT")


# ------
# import os
# import torch
# import timm
# import torch.nn as nn
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from tqdm import tqdm

# # ======================
# # CONFIG
# # ======================
# ROOT = "../data/datasets/LandDiscover_50K"
# IMG_DIR = os.path.join(ROOT, "TR_Image")
# GT_DIR = os.path.join(ROOT, "GT_ID")
# # PRETRAINED_WEIGHTS = "../../dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
# PRETRAINED_WEIGHTS = "./finetune_landdiscover_seg_epoch_5.pth"

# NUM_CLASSES = 40          # adjust if needed
# IMG_SIZE = 384
# BATCH_SIZE = 1            # ViT-L + segmentation is heavy
# EPOCHS = 50
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IGNORE_INDEX = 255        # common for segmentation

# # ======================
# # DATASET
# # ======================
# class LandDiscoverSegDataset(Dataset):
#     def __init__(self, img_root, gt_root, transform=None):
#         self.img_root = img_root
#         self.gt_root = gt_root
#         self.transform = transform

#         self.files = sorted([
#             f for f in os.listdir(img_root)
#             if f.lower().endswith((".jpg", ".png", ".jpeg"))
#         ])

#         assert len(self.files) > 0, "❌ No images found in TR_Image"

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         name = self.files[idx]

#         img = Image.open(os.path.join(self.img_root, name)).convert("RGB")
#         gt = Image.open(os.path.join(self.gt_root, name))

#         if self.transform:
#             img = self.transform(img)

#         gt = torch.from_numpy(np.array(gt)).long()
#         return img, gt

# # ======================
# # TRANSFORMS
# # ======================
# train_transform = T.Compose([
#     T.Resize((IMG_SIZE, IMG_SIZE)),
#     T.RandomHorizontalFlip(),
#     T.RandomVerticalFlip(),
#     T.ToTensor(),
#     T.Normalize(mean=[0.5]*3, std=[0.5]*3),
# ])

# # ======================
# # DATALOADER
# # ======================
# dataset = LandDiscoverSegDataset(IMG_DIR, GT_DIR, train_transform)
# print("Dataset size:", len(dataset))

# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True
# )

# # ======================
# # BACKBONE
# # ======================
# backbone = timm.create_model(
#     "vit_large_patch16_dinov3.sat493m",
#     pretrained=False,
#     num_classes=0,
#     dynamic_img_size=True,
#     img_size=IMG_SIZE
# ).to(DEVICE)

# ckpt = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
# if "teacher_state_dict" in ckpt:
#     backbone.load_state_dict(ckpt["teacher_state_dict"], strict=False)
# else:
#     backbone.load_state_dict(ckpt, strict=False)

# # Freeze backbone
# for p in backbone.parameters():
#     p.requires_grad = False

# # Unfreeze last 2 ViT blocks (~8–10%)
# for block in backbone.blocks[-2:]:
#     for p in block.parameters():
#         p.requires_grad = True

# # ======================
# # SEGMENTATION HEAD
# # ======================
# class ViTSegHead(nn.Module):
#     def __init__(self, embed_dim, num_classes):
#         super().__init__()
#         self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

#     def forward(self, x):
#         # Handle both 2D [B, C] and 3D [B, N, C] outputs
#         if len(x.shape) == 2:
#             # [B, C] -> need to get spatial features differently
#             # This shouldn't happen with num_classes=0, but handle it
#             raise ValueError(f"Unexpected 2D output from backbone: {x.shape}")

#         # x: [B, N, C] where N = 1 (CLS) + num_registers + H*W (patches)
#         B, N, C = x.shape

#         # Calculate expected number of spatial patches
#         # For 384x384 with patch_size=16: (384/16)^2 = 576
#         expected_patches = (384 // 16) ** 2

#         # Determine number of special tokens (CLS + registers)
#         # Total tokens = CLS + registers + spatial patches
#         num_special_tokens = N - expected_patches

#         # Extract only the spatial patch tokens (skip CLS and register tokens)
#         x = x[:, num_special_tokens:, :]

#         N_patches = x.shape[1]
#         H = W = int(N_patches ** 0.5)

#         # Verify we have a perfect square
#         if H * W != N_patches:
#             raise ValueError(f"Number of patches {N_patches} is not a perfect square. Got {H}x{W}={H*W}")

#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         return self.conv(x)

# seg_head = ViTSegHead(backbone.num_features, NUM_CLASSES).to(DEVICE)

# # ======================
# # OPTIMIZER & LOSS
# # ======================
# optimizer = AdamW(
#     [
#         {"params": seg_head.parameters(), "lr": 1e-3},
#         {"params": backbone.blocks[-2:].parameters(), "lr": 1e-4},
#     ],
#     weight_decay=1e-4
# )

# criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# # ======================
# # TRAINING LOOP
# # ======================
# print("🚀 Starting fine-tuning...")

# for epoch in range(EPOCHS):
#     backbone.train()
#     seg_head.train()

#     total_loss = 0.0

#     # Progress bar for batches
#     pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

#     for i, (imgs, gts) in enumerate(pbar):
#         imgs = imgs.to(DEVICE)
#         gts = gts.to(DEVICE)

#         feats = backbone.forward_features(imgs)  # [B, N, C] - get patch tokens

#         # Debug print for first batch of first epoch
#         if epoch == 0 and i == 0:
#             print(f"\nDebug: feats shape = {feats.shape}")
#             print(f"Debug: expected patches = {(IMG_SIZE // 16) ** 2} + 1 CLS = {(IMG_SIZE // 16) ** 2 + 1}")

#         logits = seg_head(feats)             # [B, C, H, W]

#         logits = nn.functional.interpolate(
#             logits, size=gts.shape[-2:], mode="bilinear", align_corners=False
#         )

#         loss = criterion(logits, gts)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         # Update progress bar with current losss
#         pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#     avg_loss = total_loss / len(loader)
#     print(f"Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {avg_loss:.4f}")

#     if (epoch + 1) % 5 == 0:
#         torch.save(
#             {
#                 "epoch": epoch,
#                 "backbone": backbone.state_dict(),
#                 "seg_head": seg_head.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#             },
#             f"finetune_landdiscover_seg_epoch_{epoch+1}.pth"
#         )

# # Save final model (two versions)
# print("💾 Saving final fine-tuned weights...")

# # 1. Save complete checkpoint (for resuming training)
# torch.save(
#     {
#         "epoch": EPOCHS - 1,
#         "backbone": backbone.state_dict(),
#         "seg_head": seg_head.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     },
#     "finetune_landdiscover_seg_final.pth"
# )

# # 2. Save ONLY backbone in wrapper-compatible format (for replacing in dinov3 folder)
# # This matches the format that DINOv3Wrapper._extract_state_dict() expects
# torch.save(backbone.state_dict(), "dinov3_finetuned_backbone_only.pth")

# print("✅ Fine-tuning completed!")
# print("   - Full checkpoint: 'finetune_landdiscover_seg_final.pth'")
# print("   - Backbone only (wrapper-compatible): 'dinov3_finetuned_backbone_only.pth'")
# print("   → Replace 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth' with 'dinov3_finetuned_backbone_only.pth'")
