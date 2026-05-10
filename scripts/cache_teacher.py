"""
Phase 1 — Cache teacher logits from pretrained GSNet.

For each image in LD50K (51,846 images) this script:
  1. Runs GSNet teacher in eval / no_grad mode with 40 LD50K classes (not 10 FloodNet).
  2. Captures the final logits (B, 40, 96, 96) from RIPD.Fusion_conv_decoer via monkey-patch.
  3. Saves {cache_dir}/{image_stem}.pt  →  {'logits': fp16 tensor (40, 96, 96)}.

Storage: ~720 KB/image → ~37 GB total for 51,846 images.

Usage:
    export RSIB_CKPT='dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

    python scripts/cache_teacher.py \\
        --config   configs/vitl_336_dinov3.yaml \\
        --weights  checkpoints/GSNet/GSNet_sup_partial_unfreeze_5epochs.pth \\
        --image-dir gs_net/data/datasets/LandDiscover_50K/TR_Image \\
        --cache-dir cache_logits/ \\
        [--batch-size 2] \\
        [--num-workers 4] \\
        [--device cuda]

Re-entrant: images whose .pt cache already exists are skipped automatically.
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_net import add_cat_seg_config
import gs_net  # noqa: side-effect registrations


# ── Capture state ─────────────────────────────────────────────────────────────
_captured = {"logits": None}


# ── Dataset ───────────────────────────────────────────────────────────────────

class LD50KImageDataset(Dataset):
    """Loads LD50K images resized to resolution×resolution as float [0, 255].

    GSNet normalises internally (subtracts CLIP pixel mean/std), so we pass
    raw pixel values in detectron2's expected format.
    """

    def __init__(self, image_dir: str, resolution: int = 336):
        self.paths = sorted(Path(image_dir).glob("*.png"))
        if not self.paths:
            self.paths = sorted(Path(image_dir).rglob("*.[jp][pn]g"))
        assert len(self.paths) > 0, f"No images found in {image_dir}"
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = TF.to_tensor(img) * 255.0   # (3, H, W), float [0, 255]
        return img, p.stem


# ── Teacher setup ─────────────────────────────────────────────────────────────

def build_teacher(config_file: str, weights_file: str, device: str):
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    from detectron2.modeling import build_model as d2_build_model
    model = d2_build_model(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def fix_class_texts(model):
    """Override eval-mode class texts from FloodNet (10) to LD50K (40).

    Two steps required:
      1. Replace test_class_texts with the 40-class training list.
      2. Clear both caches so get_text_embeds() re-encodes on the next forward.
    """
    predictor = model.sem_seg_head.predictor
    predictor.test_class_texts = predictor.class_texts   # 40 LD50K classes
    predictor.cache = None       # stale 10-class text embedding cache
    predictor.tokens = None      # stale 10-class tokenised inputs


def patch_ripd_capture(model):
    """Monkey-patch RIPD.Fusion_conv_decoer to capture logits before return.

    The method is replaced on the instance so the class definition is untouched.
    Returns an unpatch callable that restores the original.
    """
    ripd = model.sem_seg_head.predictor.transformer
    original_fcd = ripd.Fusion_conv_decoer   # bound method

    def patched_fcd(x, clip_guidance, dino_guidance):
        result = original_fcd(x, clip_guidance, dino_guidance)
        # result: (B, T, 96, 96)
        _captured["logits"] = result.detach().cpu().half()
        return result

    ripd.Fusion_conv_decoer = patched_fcd

    def unpatch():
        ripd.Fusion_conv_decoer = original_fcd

    return unpatch


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cache GSNet teacher logits for LD50K")
    p.add_argument("--config",      required=True, help="GSNet config yaml")
    p.add_argument("--weights",     required=True, help="GSNet checkpoint .pth")
    p.add_argument("--image-dir",   required=True, help="LD50K TR_Image directory")
    p.add_argument("--cache-dir",   required=True, help="Output directory for .pt files")
    p.add_argument("--batch-size",  type=int, default=2,
                   help="Keep low — teacher is ~860M params")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution",  type=int, default=336)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Loading GSNet teacher from {args.weights} ...")
    model, _ = build_teacher(args.config, args.weights, args.device)
    model = model.to(args.device)

    fix_class_texts(model)
    unpatch = patch_ripd_capture(model)

    n_classes = len(model.sem_seg_head.predictor.test_class_texts)
    print(f"Class fix applied: {n_classes} classes (expected 40)")
    assert n_classes == 40, f"Expected 40 classes, got {n_classes}. Check TRAIN_CLASS_JSON in config."

    dataset = LD50KImageDataset(args.image_dir, resolution=args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    already_cached = sum(
        1 for p in dataset.paths
        if os.path.isfile(os.path.join(args.cache_dir, f"{p.stem}.pt"))
    )
    print(f"Images: {len(dataset)}  |  Already cached: {already_cached}  |  To process: {len(dataset) - already_cached}")

    total_saved = 0
    total_skipped = 0

    for images, stems in tqdm(loader, desc="Caching teacher logits"):
        to_process = []
        for i, stem in enumerate(stems):
            if os.path.isfile(os.path.join(args.cache_dir, f"{stem}.pt")):
                total_skipped += 1
            else:
                to_process.append(i)

        if not to_process:
            continue

        sub_images = images[to_process].to(args.device)

        batched_inputs = [
            {
                "image":  sub_images[i],
                "height": args.resolution,
                "width":  args.resolution,
            }
            for i in range(len(to_process))
        ]

        with torch.no_grad():
            model(batched_inputs)

        logits = _captured["logits"]   # (B_sub, 40, 96, 96) fp16 CPU

        assert logits.shape[1] == 40, (
            f"Captured {logits.shape[1]} classes — class-text fix did not take effect."
        )

        for j, orig_idx in enumerate(to_process):
            stem = stems[orig_idx]
            torch.save({"logits": logits[j]}, os.path.join(args.cache_dir, f"{stem}.pt"))
            total_saved += 1

    unpatch()

    saved_gb = total_saved * 40 * 96 * 96 * 2 / 1e9
    print(f"\nDone.  Saved: {total_saved}  |  Skipped: {total_skipped}")
    print(f"Approximate new cache written: {saved_gb:.1f} GB")


if __name__ == "__main__":
    main()
