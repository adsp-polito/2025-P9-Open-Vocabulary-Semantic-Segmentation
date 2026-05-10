"""
Dataset for Phase 2 distillation training.

LD50KDistillDataset pairs each LD50K image with its teacher-cached logit file.
Images are loaded as [0, 1] float tensors (no CLIP normalisation) to match TIPS input format.

Cache layout:
    {cache_dir}/{image_id}.pt  →  {'logits': fp16 tensor (40, 96, 96)}

Image directory layout:
    {image_dir}/**/{image_id}.png  (recursive search)
"""

import os
import glob
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF


class LD50KDistillDataset(Dataset):
    """
    Args:
        image_dir:       root folder containing LD50K images (recursive search).
        cache_dir:       root folder containing teacher cache .pt files.
        resolution:      spatial size to resize images to (336 for TIPS ViT-L/14).
        return_image_id: if True, include the image stem in each sample.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(
        self,
        image_dir: str,
        cache_dir: str,
        resolution: int = 336,
        return_image_id: bool = False,
    ):
        self.resolution = resolution
        self.return_image_id = return_image_id

        stem_to_img = {
            p.stem: str(p)
            for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }

        cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        self.samples = []
        for cache_path in cache_files:
            stem = os.path.splitext(os.path.basename(cache_path))[0]
            if stem in stem_to_img:
                self.samples.append((stem_to_img[stem], cache_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image/cache pairs found.\n"
                f"  image_dir: {image_dir}\n"
                f"  cache_dir: {cache_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cache_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = TF.resize(image, (self.resolution, self.resolution))
        image = TF.to_tensor(image)  # [0, 1] — TIPS native input range

        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        teacher_logits = cache["logits"].float()  # (40, 96, 96)

        if self.return_image_id:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            return image, teacher_logits, stem
        return image, teacher_logits


def build_dataloaders(
    image_dir,
    cache_dir,
    batch_size=16,
    val_fraction=0.05,
    num_workers=4,
    resolution=336,
    seed=42,
):
    """Split into train/val and return DataLoaders."""
    full_ds = LD50KDistillDataset(image_dir, cache_dir, resolution=resolution)
    n_val = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
