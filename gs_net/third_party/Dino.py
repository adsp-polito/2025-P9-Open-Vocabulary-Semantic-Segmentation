import os
import torch
import timm
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ======================
# CONFIG
# ======================
ROOT = "../data/datasets/LandDiscover_50K"
IMG_DIR = os.path.join(ROOT, "TR_Image")
GT_DIR = os.path.join(ROOT, "GT_ID")
PRETRAINED_WEIGHTS = "../../dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

NUM_CLASSES = 40          # adjust if needed
IMG_SIZE = 384
BATCH_SIZE = 2            # ViT-L + segmentation is heavy
EPOCHS = 50
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
    dynamic_img_size=False
).to(DEVICE)

ckpt = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
if "teacher_state_dict" in ckpt:
    backbone.load_state_dict(ckpt["teacher_state_dict"], strict=False)
else:
    backbone.load_state_dict(ckpt, strict=False)

# Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False

# Unfreeze last 2 ViT blocks (~8–10%)
for block in backbone.blocks[-2:]:
    for p in block.parameters():
        p.requires_grad = True

# ======================
# SEGMENTATION HEAD
# ======================
class ViTSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        x = x[:, 1:, :]  # remove CLS token
        H = W = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.conv(x)

seg_head = ViTSegHead(backbone.num_features, NUM_CLASSES).to(DEVICE)

# ======================
# OPTIMIZER & LOSS
# ======================
optimizer = AdamW(
    [
        {"params": seg_head.parameters(), "lr": 1e-3},
        {"params": backbone.blocks[-2:].parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# ======================
# TRAINING LOOP
# ======================
print("🚀 Starting fine-tuning...")

for epoch in range(EPOCHS):
    backbone.train()
    seg_head.train()

    total_loss = 0.0

    for imgs, gts in loader:
        imgs = imgs.to(DEVICE)
        gts = gts.to(DEVICE)

        feats = backbone(imgs)               # [B, N, C]
        logits = seg_head(feats)             # [B, C, H, W]

        logits = nn.functional.interpolate(
            logits, size=gts.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = criterion(logits, gts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "seg_head": seg_head.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"finetune_landdiscover_seg_epoch_{epoch+1}.pth"
        )

print("✅ Fine-tuning completed.")
