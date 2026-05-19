"""
Dataset for Phase 2 distillation training.

LD50KDistillDataset pairs each LD50K image with its teacher-cached logit file,
and optionally with CLIP patch features cached by cache_teacher.py (v2).

Cache layouts:
    {cache_dir}/{image_id}.pt       →  {'logits':     fp16 (40,   96, 96)}
    {clip_feats_dir}/{image_id}.pt  →  {'clip_feats': fp16 (1024, 24, 24)}  (optional)

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
        cache_dir:       root folder containing teacher logit .pt files.
        resolution:      spatial size to resize images to (336 for TIPS ViT-L/14).
        clip_feats_dir:  optional folder containing CLIP feature .pt files from
                         cache_teacher.py v2. When provided, __getitem__ returns
                         (image, teacher_logits, clip_feats). When None, returns
                         (image, teacher_logits) — original behaviour preserved.
        return_image_id: if True, append the image stem as a final return value.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(
        self,
        image_dir: str,
        cache_dir: str,
        resolution: int = 336,
        clip_feats_dir: str = None,
        return_image_id: bool = False,
    ):
        self.resolution = resolution
        self.clip_feats_dir = clip_feats_dir
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
                feat_path = (
                    os.path.join(clip_feats_dir, f"{stem}.pt")
                    if clip_feats_dir else None
                )
                self.samples.append((stem_to_img[stem], cache_path, feat_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image/cache pairs found.\n"
                f"  image_dir: {image_dir}\n"
                f"  cache_dir: {cache_dir}"
            )

        if clip_feats_dir:
            n_feats = sum(1 for _, _, fp in self.samples if fp and os.path.isfile(fp))
            print(f"  LD50KDistillDataset: {len(self.samples)} samples, "
                  f"{n_feats} with CLIP features ({clip_feats_dir})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cache_path, feat_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = TF.resize(image, (self.resolution, self.resolution))
        image = TF.to_tensor(image)  # [0, 1] — TIPS native input range

        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        teacher_logits = cache["logits"].float()  # (40, 96, 96)

        result = [image, teacher_logits]

        if self.clip_feats_dir:
            if feat_path and os.path.isfile(feat_path):
                feat_cache = torch.load(feat_path, map_location="cpu", weights_only=True)
                clip_feats = feat_cache["clip_feats"].float()  # (1024, 24, 24)
            else:
                clip_feats = torch.zeros(1024, 24, 24)  # fallback — missing file
            result.append(clip_feats)

        if self.return_image_id:
            result.append(os.path.splitext(os.path.basename(img_path))[0])

        return tuple(result)


def build_dataloaders(
    image_dir,
    cache_dir,
    batch_size=16,
    val_fraction=0.05,
    num_workers=4,
    resolution=336,
    clip_feats_dir=None,
    seed=42,
):
    """Split into train/val and return DataLoaders."""
    full_ds = LD50KDistillDataset(
        image_dir, cache_dir,
        resolution=resolution,
        clip_feats_dir=clip_feats_dir,
    )
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
