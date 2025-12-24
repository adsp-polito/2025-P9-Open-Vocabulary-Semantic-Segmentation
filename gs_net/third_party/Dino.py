# 1 - Satellite-aware DINO augmentations
import torch
import torchvision.transforms as T
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SatelliteDINOAugmentation:
    def __init__(self, global_size=384, local_size=192):
        self.global_transforms = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([
                T.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ], p=0.3),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        self.local_transforms = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.GaussianBlur(3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __call__(self, img):
        return {
            "global1": self.global_transforms(img),
            "global2": self.global_transforms(img),
            "locals": [self.local_transforms(img) for _ in range(8)]
        }

# 2 - Dataset class (no labels)
from torch.utils.data import Dataset
import os

class UnlabeledImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.files = sorted(os.listdir(root))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        return self.transform(img)

# 3 - Student & Teacher models (ViT-L/16 from scratch)
import timm
import torch.nn as nn
import copy

def build_vit():
    model = timm.create_model(
        "vit_large_patch16_dinov3.sat493m",
        pretrained=False,
        num_classes=0,  # This already removes the classifier
        dynamic_img_size=True
    )
    # Don't call reset_classifier again, num_classes=0 already handles it
    return model


student = build_vit().to(device)
teacher = copy.deepcopy(student).to(device)

for p in teacher.parameters():
    p.requires_grad = False

# Get the embedding dimension from the model
embed_dim = student.embed_dim if hasattr(student, 'embed_dim') else student.num_features
print(f"Model embedding dimension: {embed_dim}")

# 4 - DINO projection head
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.GELU(),
            nn.Linear(4096, out_dim)
        )

    def forward(self, x):
        # x is already the output features, not patch embeddings
        # Check if we need to extract CLS token or if it's already pooled
        if x.dim() == 3:  # Shape: [batch, num_tokens, embed_dim]
            x = x[:, 0]  # CLS token
        # If x.dim() == 2, it's already pooled
        x = self.mlp(x)
        return x

student_head = DINOHead(in_dim=embed_dim).to(device)
teacher_head = copy.deepcopy(student_head).to(device)

for p in teacher_head.parameters():
    p.requires_grad = False

# 5 - DINO loss (centering + sharpening)
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        student_out = student_out / self.student_temp
        teacher_out = (teacher_out - self.center) / self.teacher_temp

        teacher_probs = torch.softmax(teacher_out, dim=-1)
        loss = torch.sum(-teacher_probs * torch.log_softmax(student_out, dim=-1), dim=-1).mean()

        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# 6 - Create dataset and dataloader
from torch.utils.data import DataLoader

# TODO: Set your dataset path
dataset = UnlabeledImageDataset(
    root="../data/datasets/LandDiscover_50K/TR_Image",
    transform=SatelliteDINOAugmentation(global_size=384, local_size=192)
)

loader = DataLoader(
    dataset,
    batch_size=2,  # Reduced to 2 for memory constraints
    shuffle=True,
    num_workers=2,  # Reduced from 4 as per warning
    pin_memory=True
)

# 7 - Training loop
epochs = 300

optimizer = torch.optim.AdamW(
    list(student.parameters()) + list(student_head.parameters()),
    lr=1e-4,
    weight_decay=0.04
)

criterion = DINOLoss(out_dim=768).to(device)

@torch.no_grad()
def update_teacher(student, teacher, m=0.996):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data * (1 - m))

print(f"Starting DINO pretraining for {epochs} epochs...")
print(f"Dataset size: {len(dataset)} images")
print(f"Batch size: {loader.batch_size}")

for epoch in range(epochs):
    student.train()
    epoch_loss = 0.0

    for i, batch in enumerate(loader):
        g1 = batch["global1"].to(device)
        g2 = batch["global2"].to(device)

        # Debug: print shapes on first iteration
        if epoch == 0 and i == 0:
            with torch.no_grad():
                test_out = student(g1)
                print(f"Student output shape: {test_out.shape}")
                print(f"Input shape: {g1.shape}")

        s_out1 = student_head(student(g1))
        s_out2 = student_head(student(g2))

        with torch.no_grad():
            t_out1 = teacher_head(teacher(g1))
            t_out2 = teacher_head(teacher(g2))

        loss = (criterion(s_out1, t_out2) + criterion(s_out2, t_out1)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_teacher(student, teacher)
        update_teacher(student_head, teacher_head)

        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'student_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'student_head_state_dict': student_head.state_dict(),
            'teacher_head_state_dict': teacher_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"dinov3_vitl16_checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

# Save final model
torch.save(
    teacher.state_dict(),
    "dinov3_vitl16_landdiscovery50k_pretrain.pth"
)
print("Training complete! Final model saved.")
