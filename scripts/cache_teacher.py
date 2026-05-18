"""
Phase 1 — Cache teacher logits and CLIP features from pretrained GSNet.

For each image in LD50K (51,846 images) this script:
  1. Runs GSNet teacher in eval / no_grad mode with 40 LD50K classes.
  2. Captures final logits (B, 40, 96, 96) via monkey-patch on Fusion_conv_decoer.
  3. Captures CLIP patch features (B, 1024, 24, 24) via monkey-patch on
     GSNetPredictor.forward — these are the raw CLIP ViT-L/14 features before
     DINOv3 fusion (QGFF). Same architecture/dim as TIPS, used for feature-level
     distillation in Phase 2.

Two separate output directories:
  --cache-dir      (existing)  {stem}.pt  →  {"logits": fp16 (40, 96, 96)}
  --clip-feats-dir (new)       {stem}.pt  →  {"clip_feats": fp16 (1024, 24, 24)}

Re-entrant: images already present in BOTH dirs are skipped. If --clip-feats-dir
is not provided, only logits are captured (original behaviour preserved).

Storage:
  Logits:     ~72 GB total for 51,846 images  (already cached in cache_logits/)
  CLIP feats: ~61 GB total for 51,846 images  (new, cache_clip_feats/)

Usage:
    export RSIB_CKPT='dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

    python scripts/cache_teacher.py \\
        --config        configs/vitl_336_dinov3.yaml \\
        --weights       checkpoints/GSNet/GSNet_sup_partial_unfreeze_5epochs.pth \\
        --image-dir     gs_net/data/datasets/LandDiscover_50K/TR_Image \\
        --cache-dir     cache_logits/ \\
        --clip-feats-dir cache_clip_feats/ \\
        [--batch-size 2] [--num-workers 4] [--device cuda]
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

_captured = {"logits": None, "clip_feats": None}


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
    """Override eval-mode class texts from FloodNet (10) to LD50K (40)."""
    predictor = model.sem_seg_head.predictor
    predictor.test_class_texts = predictor.class_texts
    predictor.cache = None
    predictor.tokens = None


def patch_ripd_capture(model):
    """Monkey-patch RIPD.Fusion_conv_decoer to capture final logits."""
    ripd = model.sem_seg_head.predictor.transformer
    original_fcd = ripd.Fusion_conv_decoer

    def patched_fcd(x, clip_guidance, dino_guidance):
        result = original_fcd(x, clip_guidance, dino_guidance)
        _captured["logits"] = result.detach().cpu().half()   # (B, 40, 96, 96)
        return result

    ripd.Fusion_conv_decoer = patched_fcd

    def unpatch():
        ripd.Fusion_conv_decoer = original_fcd
    return unpatch


def patch_clip_capture(model):
    """Monkey-patch GSNetPredictor.forward to capture raw CLIP features.

    GSNetPredictor.forward receives x=(B, 1024, 24, 24) as its first positional
    argument — the CLIP ViT-L/14 patch feature map before QGFF/DINOv3 fusion.
    Capturing here gives us a direct feature-level distillation target for TIPS,
    since both models are ViT-L/14 and produce 1024-dim patch features at 24×24.
    """
    predictor = model.sem_seg_head.predictor
    original_fwd = predictor.forward

    def patched_fwd(*args, **kwargs):
        # x is the first positional arg (CLIP features)
        x = args[0] if len(args) > 0 else kwargs.get("x", None)
        if x is not None:
            _captured["clip_feats"] = x.detach().cpu().half()  # (B, 1024, 24, 24)
        return original_fwd(*args, **kwargs)

    predictor.forward = patched_fwd

    def unpatch():
        predictor.forward = original_fwd
    return unpatch


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cache GSNet teacher logits and CLIP features")
    p.add_argument("--config",          required=True,  help="GSNet config yaml")
    p.add_argument("--weights",         required=True,  help="GSNet checkpoint .pth")
    p.add_argument("--image-dir",       required=True,  help="LD50K TR_Image directory")
    p.add_argument("--cache-dir",       required=True,  help="Output dir for logit .pt files")
    p.add_argument("--clip-feats-dir",  default=None,
                   help="Output dir for CLIP feature .pt files (new). "
                        "If omitted, only logits are cached (original behaviour).")
    p.add_argument("--batch-size",      type=int, default=2)
    p.add_argument("--num-workers",     type=int, default=4)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution",      type=int, default=336)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    if args.clip_feats_dir:
        os.makedirs(args.clip_feats_dir, exist_ok=True)

    capture_feats = args.clip_feats_dir is not None

    print(f"Loading GSNet teacher from {args.weights} ...")
    model, _ = build_teacher(args.config, args.weights, args.device)
    model = model.to(args.device)

    fix_class_texts(model)
    unpatch_logits = patch_ripd_capture(model)
    if capture_feats:
        unpatch_feats = patch_clip_capture(model)
        print("CLIP feature capture: enabled  →  " + args.clip_feats_dir)
    else:
        unpatch_feats = None
        print("CLIP feature capture: disabled (pass --clip-feats-dir to enable)")

    n_classes = len(model.sem_seg_head.predictor.test_class_texts)
    print(f"Class fix applied: {n_classes} classes (expected 40)")
    assert n_classes == 40, f"Expected 40 classes, got {n_classes}."

    dataset = LD50KImageDataset(args.image_dir, resolution=args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Count already-cached files for progress reporting
    already_logits = sum(
        1 for p in dataset.paths
        if os.path.isfile(os.path.join(args.cache_dir, f"{p.stem}.pt"))
    )
    already_feats = (
        sum(
            1 for p in dataset.paths
            if os.path.isfile(os.path.join(args.clip_feats_dir, f"{p.stem}.pt"))
        )
        if capture_feats else 0
    )
    print(f"\nImages: {len(dataset)}")
    print(f"  Logits  already cached : {already_logits}  (dir: {args.cache_dir})")
    if capture_feats:
        print(f"  Feats   already cached : {already_feats}  (dir: {args.clip_feats_dir})")
    print()

    total_saved_logits = 0
    total_saved_feats  = 0
    total_skipped      = 0

    for images, stems in tqdm(loader, desc="Caching teacher outputs"):
        # Determine which images need processing:
        # Run forward only when at least one output is missing for that image.
        to_process = []
        for i, stem in enumerate(stems):
            need_logit = not os.path.isfile(os.path.join(args.cache_dir, f"{stem}.pt"))
            need_feat  = (
                capture_feats and
                not os.path.isfile(os.path.join(args.clip_feats_dir, f"{stem}.pt"))
            )
            if need_logit or need_feat:
                to_process.append(i)
            else:
                total_skipped += 1

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

        logits     = _captured["logits"]      # (B_sub, 40,   96, 96) fp16 CPU
        clip_feats = _captured["clip_feats"]  # (B_sub, 1024, 24, 24) fp16 CPU or None

        assert logits is not None and logits.shape[1] == 40, (
            f"Captured {logits.shape[1] if logits is not None else 'None'} classes — "
            "class-text fix did not take effect."
        )
        if capture_feats:
            assert clip_feats is not None, (
                "CLIP feature capture returned None — check patch_clip_capture hook."
            )
            assert clip_feats.shape[1] == 1024 and clip_feats.shape[2] == 24, (
                f"Unexpected clip_feats shape: {clip_feats.shape} — expected (B, 1024, 24, 24)."
            )

        for j, orig_idx in enumerate(to_process):
            stem = stems[orig_idx]

            logit_path = os.path.join(args.cache_dir, f"{stem}.pt")
            if not os.path.isfile(logit_path):
                torch.save({"logits": logits[j]}, logit_path)
                total_saved_logits += 1

            if capture_feats:
                feat_path = os.path.join(args.clip_feats_dir, f"{stem}.pt")
                if not os.path.isfile(feat_path):
                    torch.save({"clip_feats": clip_feats[j]}, feat_path)
                    total_saved_feats += 1

    unpatch_logits()
    if unpatch_feats is not None:
        unpatch_feats()

    logits_gb = total_saved_logits * 40   * 96 * 96 * 2 / 1e9
    feats_gb  = total_saved_feats  * 1024 * 24 * 24 * 2 / 1e9
    print(f"\nDone.")
    print(f"  Logits  saved: {total_saved_logits:6d}   (~{logits_gb:.1f} GB raw tensors)")
    if capture_feats:
        print(f"  Feats   saved: {total_saved_feats:6d}   (~{feats_gb:.1f} GB raw tensors)")
    print(f"  Skipped (both existed): {total_skipped}")


if __name__ == "__main__":
    main()
