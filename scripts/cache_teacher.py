"""
Phase 1 — Generate teacher cache from pretrained GSNet.

For each image in the LD50K dataset this script:
  1. Runs pretrained GSNet teacher in eval / no_grad mode.
  2. Captures `fused_corr_embed` from inside RIPD.forward() via a forward hook,
     immediately after the QGFF step (line 250 in RIPD.py), before AggregatorLayers.
  3. Captures `dino_feat_L4` and `dino_feat_L8` from GSNet.forward() (BEFORE projection).
  4. Saves each image's tensors as {cache_dir}/{image_id}.pt in fp16.

Usage:
    export RSIB_CKPT='path/to/dinov3_checkpoint.pth'

    python scripts/cache_teacher.py \\
        --config    configs/vitb_384.yaml \\
        --weights   path/to/gsnet_checkpoint.pth \\
        --image-dir path/to/LD50K/images \\
        --cache-dir cache/ \\
        --batch-size 4 \\
        --num-workers 4 \\
        [--device cuda]

The script is re-entrant: images whose .pt cache already exists are skipped.
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

# ── detectron2 setup ─────────────────────────────────────────────────────────
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_net import add_cat_seg_config         # noqa: registers GSNet
import gs_net                                  # noqa: side-effect registrations


# ─────────────────────────────────────────────────────────────────────────────
# Hook state (module-level so closures can write into it)
# ─────────────────────────────────────────────────────────────────────────────
_hook_state = {
    "fused_corr_embed": None,
}


def _ripd_hook(module, input, output):
    """Attached to RIPD — intercepts fused_corr_embed after QGFF, before Aggregators."""
    # fused_corr_embed is NOT a module output; we capture it via a custom hook
    # by temporarily patching the forward call.  See _patch_ripd() below.
    pass


def _patch_ripd(ripd_module):
    """
    Monkey-patch RIPD.forward to capture fused_corr_embed right after line 250
    (i.e. after QGFF produces the fused tensor, before AggregatorLayers).

    Returns an 'unpatch' callable that restores the original forward.
    """
    original_forward = ripd_module.forward

    def patched_forward(img_feats, dino_feat, text_feats, appearance_guidance, dino_guidance):
        import torch.nn.functional as F_
        from einops import rearrange as rr

        # ── Replicate QGFF logic to compute fused_corr_embed ─────────────────
        if dino_feat is not None and img_feats is not None:
            if ripd_module.fusion_type == 'query_guided':
                corr      = ripd_module.correlation(img_feats, text_feats)
                dino_corr = ripd_module.correlation(dino_feat,  text_feats)
                fused_corr_embed, clip_embed_corr, _ = ripd_module.corr_fusion_embed_seperate(
                    clip_corr=corr, dino_corr=dino_corr
                )
                fused_corr_embed = fused_corr_embed + clip_embed_corr
            elif ripd_module.fusion_type == 'simple_concatenation':
                simple_corr = ripd_module.simple_concatenation_corr(img_feats, dino_feat, text_feats)
                T = simple_corr.shape[2]
                fused_corr_embed = ripd_module.simple_concatenation_corr_embed(
                    rr(simple_corr, "B C T H W -> (B T) C H W"))
                fused_corr_embed = rr(fused_corr_embed, "(B T) C H W -> B C T H W", T=T)
            elif ripd_module.fusion_type == 'fusion_query':
                fused_feat = ripd_module.fusion_feats(torch.cat([img_feats, dino_feat], dim=1))
                corr = ripd_module.correlation(fused_feat, text_feats)
                fused_corr_embed = ripd_module.corr_embed(corr)
        elif dino_feat is not None:
            corr = ripd_module.correlation(dino_feat, text_feats)
            fused_corr_embed = ripd_module.corr_embed(corr)
        elif img_feats is not None:
            corr = ripd_module.correlation(img_feats, text_feats)
            fused_corr_embed = ripd_module.corr_embed(corr)

        # ── Store capture ─────────────────────────────────────────────────────
        _hook_state["fused_corr_embed"] = fused_corr_embed.detach().cpu().half()

        # ── Continue with the original remainder of forward ───────────────────
        return original_forward(img_feats, dino_feat, text_feats, appearance_guidance, dino_guidance)

    ripd_module.forward = patched_forward

    def unpatch():
        ripd_module.forward = original_forward

    return unpatch


# ─────────────────────────────────────────────────────────────────────────────
# Image dataset
# ─────────────────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """Simple flat-folder image dataset; yields (tensor, stem)."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, clip_mean, clip_std, resolution=384):
        self.paths = sorted(
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        self.clip_mean  = clip_mean
        self.clip_std   = clip_std
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.clip_mean, std=self.clip_std)
        return img, p.stem


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config_file, weights_file, device):
    cfg = get_cfg()
    # Add GSNet / cat-seg config keys
    from gs_net import add_cat_seg_config
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    from detectron2.modeling import build_model as d2_build_model
    model = d2_build_model(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="GS-Distill Phase 1: cache teacher outputs")
    parser.add_argument("--config",      required=True, help="Path to GSNet config yaml")
    parser.add_argument("--weights",     required=True, help="Path to pretrained GSNet checkpoint")
    parser.add_argument("--image-dir",   required=True, help="Root folder of LD50K images")
    parser.add_argument("--cache-dir",   required=True, help="Output folder for .pt cache files")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",      action="store_true", help="Skip images already cached")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Build & load teacher ─────────────────────────────────────────────────
    print(f"Loading GSNet from {args.weights} ...")
    model, cfg = build_model(args.config, args.weights, args.device)
    model = model.to(args.device)

    # Locate RIPD module inside the model
    # GSNet.sem_seg_head.predictor → RIPD instance
    ripd = model.sem_seg_head.predictor.transformer
    unpatch = _patch_ripd(ripd)

    # ── Prepare hook to capture dino_feat_L4 / dino_feat_L8 ─────────────────
    # These are produced inside GSNet.forward at lines 297-298 BEFORE projection.
    # We capture them via a hook on the GSNet forward pass itself by monkey-patching.
    dino_l4_capture = [None]
    dino_l8_capture = [None]

    original_gsnet_forward = model.forward.__func__

    def patched_gsnet_forward(self_, batched_inputs):
        # We need access to dino_feat_L4/L8 that are local variables in GSNet.forward.
        # Strategy: run the original forward, but also run the DINOv3 model
        # independently to extract the same intermediate layers.
        # (Avoids duplicating 400+ lines of GSNet.forward)
        if self_.dino_model is not None:
            # Build the same dino_images_resized that GSNet.forward uses
            images = [x["image"].to(self_.device) for x in batched_inputs]
            import torch.nn.functional as F_
            from detectron2.structures import ImageList
            clip_imgs = [(x - self_.clip_pixel_mean) / self_.clip_pixel_std for x in images]
            clip_imgs_list = ImageList.from_tensors(clip_imgs, self_.size_divisibility)
            dino_images_resized = F_.interpolate(
                clip_imgs_list.tensor, size=self_.dino_resolution,
                mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                dino_feat = self_.dino_model.get_intermediate_layers(dino_images_resized, n=12)
            dino_l4_capture[0] = rearrange(
                dino_feat[3][:, 1:, :], "B (H W) C -> B C H W", H=48
            ).detach().cpu().half()
            dino_l8_capture[0] = rearrange(
                dino_feat[7][:, 1:, :], "B (H W) C -> B C H W", H=48
            ).detach().cpu().half()

        return original_gsnet_forward(self_, batched_inputs)

    import types
    model.forward = types.MethodType(patched_gsnet_forward, model)

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    clip_mean = tuple(v / 255.0 for v in cfg.MODEL.CLIP_PIXEL_MEAN) if hasattr(cfg.MODEL, "CLIP_PIXEL_MEAN") else (0.48145466, 0.4578275, 0.40821073)
    clip_std  = tuple(v / 255.0 for v in cfg.MODEL.CLIP_PIXEL_STD)  if hasattr(cfg.MODEL, "CLIP_PIXEL_STD")  else (0.26862954, 0.26130258, 0.27577711)

    dataset = ImageFolderDataset(args.image_dir, clip_mean, clip_std, resolution=384)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Found {len(dataset)} images. Caching to {args.cache_dir} ...")

    total_saved = 0
    total_skipped = 0

    for images, stems in tqdm(loader, desc="Caching"):
        batch_stems = list(stems)

        # Check which images already have a cache file
        to_process = []
        for i, stem in enumerate(batch_stems):
            out_path = os.path.join(args.cache_dir, f"{stem}.pt")
            if args.resume and os.path.isfile(out_path):
                total_skipped += 1
            else:
                to_process.append(i)

        if not to_process:
            continue

        # Run teacher forward on sub-batch that needs caching
        sub_images = images[to_process].to(args.device)

        # Build batched_inputs format expected by GSNet.forward
        batched_inputs = []
        for img in sub_images:
            # GSNet.forward normalises internally; we pass raw [0,1] tensor
            # but CachedLD50KDataset already normalised with CLIP stats.
            # For cache_teacher we load raw pixel images and let GSNet normalise.
            # Re-denormalise to raw pixels:
            img_raw = img * torch.tensor(clip_std, device=img.device).view(3,1,1) \
                         + torch.tensor(clip_mean, device=img.device).view(3,1,1)
            img_raw = (img_raw * 255.0).clamp(0, 255)
            batched_inputs.append({"image": img_raw})

        with torch.no_grad():
            model(batched_inputs)   # side effects: fills _hook_state and dino captures

        # Save per-image cache files
        B_sub = len(to_process)
        fused = _hook_state["fused_corr_embed"]   # (B_sub, hidden_dim, T, 24, 24) on CPU
        l4    = dino_l4_capture[0]                 # (B_sub, 768, 48, 48)
        l8    = dino_l8_capture[0]                 # (B_sub, 768, 48, 48)

        for j, orig_idx in enumerate(to_process):
            stem = batch_stems[orig_idx]
            out_path = os.path.join(args.cache_dir, f"{stem}.pt")
            torch.save({
                "fused_corr_embed": fused[j],
                "dino_L4":          l4[j],
                "dino_L8":          l8[j],
            }, out_path)
            total_saved += 1

    unpatch()

    print(f"\nDone. Saved {total_saved} cache files, skipped {total_skipped} existing.")


if __name__ == "__main__":
    main()
