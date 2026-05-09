"""
Dataset for distillation training.

CachedLD50KDataset pairs each raw LD50K image with its corresponding teacher cache .pt file.

Cache directory layout (one file per image):
    {cache_dir}/{image_id}.pt  →  dict:
        'fused_corr_embed': fp16 tensor (hidden_dim, T, 24, 24)
        'dino_L4':          fp16 tensor (768, 48, 48)
        'dino_L8':          fp16 tensor (768, 48, 48)

Image directory layout mirrors the cache directory:
    {image_dir}/{image_id}.jpg  (or .png / .tif)
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Tuple


class CachedLD50KDataset(Dataset):
    """
    Args:
        image_dir:  root folder containing LD50K images.
        cache_dir:  root folder containing teacher cache .pt files.
        clip_mean:  per-channel mean for CLIP normalisation (default ImageNet).
        clip_std:   per-channel std  for CLIP normalisation.
        resolution: spatial size to resize images to before CLIP forward (384).
        return_image_id: if True, include the image stem in each sample.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        image_dir: str,
        cache_dir: str,
        clip_mean: Tuple[float, float, float] = None,
        clip_std:  Tuple[float, float, float] = None,
        resolution: int = 384,
        return_image_id: bool = False,
    ):
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.return_image_id = return_image_id
        self.clip_mean = clip_mean or self.CLIP_MEAN
        self.clip_std  = clip_std  or self.CLIP_STD

        # Build stem → path map from image_dir (recursive, mirrors cache_teacher.py)
        EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        from pathlib import Path
        stem_to_img = {
            p.stem: str(p)
            for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in EXTENSIONS
        }

        # Match against available cache files
        cache_files = glob.glob(os.path.join(cache_dir, "*.pt"))
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

        # Load and pre-process image
        image = Image.open(img_path).convert("RGB")
        image = TF.resize(image, (self.resolution, self.resolution))
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.clip_mean, std=self.clip_std)

        # Load teacher cache; cast to float32 for loss computation
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        target = {k: v.float() for k, v in cache.items()}

        if self.return_image_id:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            return image, target, stem
        return image, target


def build_dataloaders(image_dir, cache_dir, batch_size=16, val_fraction=0.05,
                      num_workers=4, resolution=384, seed=42):
    """
    Convenience function: split into train/val and return DataLoaders.
    """
    from torch.utils.data import DataLoader, random_split

    full_ds = CachedLD50KDataset(image_dir, cache_dir, resolution=resolution)
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
