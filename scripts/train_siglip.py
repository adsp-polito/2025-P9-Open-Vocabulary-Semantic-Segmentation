"""
SigLIP backbone — combined Phase 2 (online distillation) + Phase 3 (segmentation fine-tune).

Replaces the OpenAI CLIP image backbone with google/siglip-base-patch16-384
(HuggingFace transformers). The student predicts text-independent DINO
substitutes and normal RIPD computes dynamic text-conditioned correlations.

Key differences from the CLIP pipeline:
  - SigLIP backbone loaded via AutoModel (HuggingFace)
  - Intermediate layers hooked on vision_model.encoder.layers[l] (output: B x seq x C)
  - Image normalisation: mean=0.5, std=0.5  (SigLIP convention)
  - Text features: SigLIP text encoder via model.get_text_features()
  - GSNet teacher still uses original CLIP — loaded from the GSNet checkpoint as usual
  - No Phase 1 attention fine-tuning required for SigLIP

Phase 2 (distillation) completes first, saves student_distill_best.pth, then
Phase 3 (fine-tune) starts automatically from that checkpoint.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/train_siglip.py \\
        --gsnet-config  configs/vitb_384.yaml \\
        --gsnet-weights path/to/gsnet.pth \\
        --image-dir     path/to/LD50K/TR_Image \\
        --label-dir     path/to/LD50K/TR_Label \\
        --output-dir    output/ashie/siglip \\
        [--distill-epochs 30] [--finetune-epochs 15] \\
        [--batch-size 4] [--lr 1e-5] [--amp] [--wandb-project gs-distill]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

import wandb
from transformers import AutoModel

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_net import add_cat_seg_config
import gs_net  # noqa: side-effect registrations

from gs_distill.losses import distillation_loss_per_branch


# ─────────────────────────────────────────────────────────────────────────────
# LD50K vocabulary
# ─────────────────────────────────────────────────────────────────────────────
CLASSES_LandDiscover50K = [
    'background', 'bare land', 'grass', 'pavement', 'road', 'tree', 'water',
    'agriculture land', 'buildings', 'forest land', 'barren land', 'urban land',
    'large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
    'plane', 'ship', 'soccer-ball-field', 'basketball-court',
    'ground-track-field', 'small-vehicle', 'baseball-diamond',
    'tennis-court', 'roundabout', 'storage-tank', 'harbor',
    'container-crane', 'airport', 'helipad', 'chimney',
    'expressway service area', 'expresswalltoll station', 'dam',
    'golf field', 'overpass', 'stadium', 'train station',
    'vehicle', 'windmill',
]

# SigLIP normalisation (different from CLIP)
_SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5])
_SIGLIP_STD  = torch.tensor([0.5, 0.5, 0.5])

# SigLIP ViT-B/16 @ 384px has 12 layers (0–11).
# Mirror CLIP default [4, 8, 10, 12] → [4, 8, 10, 11] (max index = 11).
SIGLIP_LAYERS_DEFAULT = [4, 8, 10, 11]
TRUNK_OUT = 512


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _siglip_native_res(siglip_model) -> int:
    return siglip_model.config.vision_config.image_size


def extract_siglip_layers(siglip_model, image: torch.Tensor, layers: list) -> torch.Tensor:
    """
    Extract intermediate patch features from a SigLIP ViT for specified encoder layer indices.

    SiglipEncoderLayer returns a plain tensor (B, seq_len, C) — no CLS token, no tuple.
    All 576 tokens (24×24) are patch tokens for siglip-base-patch16-384.

    Args:
        siglip_model: frozen SiglipModel.
        image: (B, 3, H, W) SigLIP-normalised, any spatial size.
        layers: list of encoder layer indices to hook, e.g. [4, 8, 10, 11].

    Returns:
        (B, len(layers)*C, H_grid, W_grid) stacked spatial features.
    """
    native_res = _siglip_native_res(siglip_model)
    if image.shape[-2] != native_res or image.shape[-1] != native_res:
        image = F.interpolate(image, size=(native_res, native_res),
                              mode="bilinear", align_corners=False)

    captured = {}
    hooks = []
    for l in layers:
        def make_hook(idx):
            def hook(module, input, output):
                # SiglipEncoderLayer returns a plain tensor (B, seq_len, C)
                captured[idx] = output
            return hook
        h = siglip_model.vision_model.encoder.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        siglip_model.vision_model(pixel_values=image)

    for h in hooks:
        h.remove()

    feature_maps = []
    for l in layers:
        feat = captured[l]              # (B, seq_len, C) — all tokens are patches, no CLS
        B, n, C = feat.shape
        H = W = int(n ** 0.5)
        feat = rearrange(feat, "B (H W) C -> B C H W", H=H, W=W)
        feature_maps.append(feat)

    return torch.cat(feature_maps, dim=1)   # (B, len(layers)*C, H_grid, W_grid)


def get_siglip_skips(siglip_model, image: torch.Tensor, layer_indices: list):
    """
    Return patch feature maps at given encoder layer indices and CLS embedding.
    Mirrors get_clip_skips() from gs_distill/utils.py.

    Returns:
        clip_features: (B, C) CLS token from the final layer output
        skips: dict mapping layer_index -> (B, C, H_grid, W_grid)
    """
    native_res = _siglip_native_res(siglip_model)
    if image.shape[-2] != native_res or image.shape[-1] != native_res:
        image = F.interpolate(image, size=(native_res, native_res),
                              mode="bilinear", align_corners=False)

    captured = {}
    hooks = []
    for l in layer_indices:
        def make_hook(idx):
            def hook(module, input, output):
                # SiglipEncoderLayer returns a plain tensor (B, seq_len, C)
                captured[idx] = output
            return hook
        h = siglip_model.vision_model.encoder.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        vit_out = siglip_model.vision_model(pixel_values=image)

    for h in hooks:
        h.remove()

    # last_hidden_state: (B, seq_len, C) — all patch tokens, no CLS
    last_hidden_state = vit_out.last_hidden_state

    skips = {}
    for l in layer_indices:
        feat = captured[l]   # (B, seq_len, C) — all tokens are patches
        B, n, C = feat.shape
        H = W = int(n ** 0.5)
        skips[l] = rearrange(feat, "B (H W) C -> B C H W", H=H, W=W)

    return last_hidden_state, skips


def siglip_normalise(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [0, 1] → SigLIP-normalised."""
    mean = _SIGLIP_MEAN.to(images.device).view(1, 3, 1, 1)
    std  = _SIGLIP_STD.to(images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP student — identical architecture to GSDistillStudent but self-contained
# ─────────────────────────────────────────────────────────────────────────────

class SigLIPStudent(nn.Module):
    """
    GS-Distill student with a frozen SigLIP image backbone.

    Predicts text-independent DINO substitutes and optional decoder skip
    adapters. RIPD remains responsible for dynamic text-conditioned fusion.
      - Frozen SigLIP ViT (multi-layer features)
      - Shared conv trunk (4*C → 512 channels)
      - DINO-L4 head: predicts dino_L4           (B, d_dino, 2H, 2W)
      - DINO-L8 head: predicts dino_L8           (B, d_dino, 2H, 2W)
    """

    def __init__(self, siglip_model: nn.Module, d_dino: int = 768,
                 siglip_layers: list = None, clip_skip_dims=None):
        super().__init__()

        self.siglip_model = siglip_model
        for p in self.siglip_model.parameters():
            p.requires_grad = False

        self.siglip_layers = siglip_layers if siglip_layers is not None else SIGLIP_LAYERS_DEFAULT
        self.d_dino       = d_dino

        # Infer hidden dim from SigLIP config
        clip_dim  = siglip_model.config.vision_config.hidden_size   # 768 for ViT-B/16
        trunk_in  = len(self.siglip_layers) * clip_dim
        clip_skip_dims = clip_skip_dims or (clip_dim, clip_dim)

        self.shared_trunk = nn.Sequential(
            nn.Conv2d(trunk_in, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(TRUNK_OUT, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.dino_down_branch = nn.Sequential(
            nn.Conv2d(TRUNK_OUT, d_dino, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_dino, d_dino, kernel_size=3, padding=1),
        )
        self.dino_l4_branch = nn.Sequential(
            nn.ConvTranspose2d(TRUNK_OUT, d_dino, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(d_dino, d_dino, kernel_size=3, padding=1),
        )
        self.dino_l8_branch = nn.Sequential(
            nn.ConvTranspose2d(TRUNK_OUT, d_dino, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(d_dino, d_dino, kernel_size=3, padding=1),
        )
        self.clip_skip_adapters = nn.ModuleList([
            nn.Identity() if target_dim == clip_dim else nn.Conv2d(clip_dim, target_dim, kernel_size=1)
            for target_dim in clip_skip_dims
        ])
        for adapter in self.clip_skip_adapters:
            if isinstance(adapter, nn.Conv2d):
                nn.init.zeros_(adapter.weight)
                if adapter.bias is not None:
                    nn.init.zeros_(adapter.bias)
                n = min(adapter.out_channels, adapter.in_channels)
                with torch.no_grad():
                    for i in range(n):
                        adapter.weight[i, i, 0, 0] = 1.0

    def forward(self, image: torch.Tensor) -> dict:
        with torch.no_grad():
            stacked = extract_siglip_layers(self.siglip_model, image, self.siglip_layers)

        trunk_out = self.shared_trunk(stacked)   # (B, 512, H_grid, W_grid)

        return {
            "dino_down": self.dino_down_branch(trunk_out),
            "dino_L4": self.dino_l4_branch(trunk_out),
            "dino_L8": self.dino_l8_branch(trunk_out),
        }

    def adapt_clip_skip(self, skip: torch.Tensor, index: int) -> torch.Tensor:
        return self.clip_skip_adapters[index](skip)

    def trainable_parameters(self):
        params = (
            list(self.shared_trunk.parameters())
            + list(self.dino_down_branch.parameters())
            + list(self.dino_l4_branch.parameters())
            + list(self.dino_l8_branch.parameters())
            + list(self.clip_skip_adapters.parameters())
        )
        return params


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP inference — replaces gs_distill/inference.py for this backbone
# ─────────────────────────────────────────────────────────────────────────────

def siglip_inference(
    image, text_feats, student, siglip_model,
    ripd, clip_upsample1, clip_upsample2,
    dino_decod_proj1, dino_decod_proj2,
    skip_layer_indices=(3, 7),
):
    """
    Full inference using SigLIP features instead of CLIP.
    Mirrors gs_distill_inference() but uses get_siglip_skips().

    Args:
        image:        (B, 3, H, W) SigLIP-normalised.
        text_feats:   (B, T, P, C) text embeddings from CLIP text encoder (teacher's CLIP).
        student:      trained SigLIPStudent.
        siglip_model: frozen SiglipModel.
        ripd:         RIPD decoder from GSNet.
        ...           projection modules from GSNet (same as gs_distill_inference).
        skip_layer_indices: SigLIP encoder layer indices for decoder skip connections.

    Returns:
        logit: (B, T, H, W) segmentation logits.
    """
    with torch.no_grad():
        last_hidden_state, skips = get_siglip_skips(
            siglip_model, image, list(skip_layer_indices)
        )
        l0, l1 = skip_layer_indices

    student_out = student(image)
    predicted_dino_down = student_out["dino_down"]
    predicted_dino_L4 = student_out["dino_L4"]
    predicted_dino_L8 = student_out["dino_L8"]

    res4_raw = student.adapt_clip_skip(skips[l0], 0) if clip_upsample1 is not None else None
    res5_raw = student.adapt_clip_skip(skips[l1], 1) if clip_upsample2 is not None else None
    res4 = clip_upsample1(res4_raw) if res4_raw is not None else None
    res5 = clip_upsample2(res5_raw) if res5_raw is not None else None
    dino_L4_proj = dino_decod_proj1(predicted_dino_L4) if dino_decod_proj1 is not None else None
    dino_L8_proj = dino_decod_proj2(predicted_dino_L8) if dino_decod_proj2 is not None else None

    # res3: all 576 patch tokens from last_hidden_state — SigLIP has no CLS token
    # last_hidden_state: (B, 576, C) for siglip-base-patch16-384
    res3 = rearrange(last_hidden_state, "B (H W) C -> B C H W", H=24)

    clip_guidance = (res3, res4, res5)
    dino_guidance = [dino_L4_proj, dino_L8_proj]

    logit = ripd(res3, predicted_dino_down, text_feats, clip_guidance, dino_guidance)
    return logit


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """Image-only dataset for Phase 2 distillation (no labels)."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, resolution=384):
        self.resolution = resolution
        self.paths = sorted(
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        return TF.to_tensor(img)   # [0, 1] float32, normalise later


class SegDataset(Dataset):
    """Image + label dataset for Phase 3 segmentation fine-tuning."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384):
        self.resolution = resolution

        img_stems = {
            p.stem for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
        lbl_stems = {
            p.stem for p in Path(label_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
        stems = sorted(img_stems & lbl_stems)

        self.samples = []
        for stem in stems:
            for ext in (".png", ".jpg", ".tif"):
                ip = Path(image_dir) / (stem + ext)
                lp = Path(label_dir) / (stem + ext)
                if ip.exists() and lp.exists():
                    self.samples.append((str(ip), str(lp)))
                    break

        if not self.samples:
            raise RuntimeError(f"No matching image/label pairs in {image_dir} / {label_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        lbl = Image.open(lbl_path)
        lbl = TF.resize(lbl, (self.resolution, self.resolution),
                        interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Teacher setup
# ─────────────────────────────────────────────────────────────────────────────

_dino_last  = [None]
_dino_l4    = [None]
_dino_l8    = [None]


def _patch_dino(dino_model):
    original_fn = dino_model.get_intermediate_layers

    def patched_fn(imgs, n=12):
        result = original_fn(imgs, n)
        _dino_last[0] = rearrange(
            result[-1][:, 1:, :], "B (H W) C -> B C H W", H=48
        ).detach().cpu().half()
        _dino_l4[0] = rearrange(
            result[3][:, 1:, :], "B (H W) C -> B C H W", H=48
        ).detach().cpu().half()
        _dino_l8[0] = rearrange(
            result[7][:, 1:, :], "B (H W) C -> B C H W", H=48
        ).detach().cpu().half()
        return result

    dino_model.get_intermediate_layers = patched_fn

    def unpatch():
        dino_model.get_intermediate_layers = original_fn
    return unpatch


def build_teacher(config_file, weights_file, device):
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    from detectron2.modeling import build_model as d2_build
    model = d2_build(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()

    predictor = model.sem_seg_head.predictor
    predictor.test_class_texts = CLASSES_LandDiscover50K
    predictor.cache = None

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Text features (uses teacher CLIP text encoder — unchanged from finetune)
# ─────────────────────────────────────────────────────────────────────────────

def remove_gsnet_clip_cache_hooks(gsnet):
    """Remove GSNet's permanent CLIP image hooks; this script uses explicit hooks."""
    clip_model = gsnet.sem_seg_head.predictor.clip_model
    removed = 0
    for idx in getattr(gsnet, "layer_indexes", []):
        block = clip_model.visual.transformer.resblocks[idx]
        for hook_id, hook in list(block._forward_hooks.items()):
            closure = getattr(hook, "__closure__", None) or ()
            if any(cell.cell_contents is gsnet for cell in closure):
                del block._forward_hooks[hook_id]
                removed += 1
    if hasattr(gsnet, "layers"):
        gsnet.layers.clear()
    if removed:
        print(f"  Removed {removed} GSNet CLIP cache hooks.")


def release_unused_gsnet(gsnet):
    """Drop GSNet container references after extracting fine-tune modules."""
    gsnet.dino_model = None
    gsnet.backbone = None
    gsnet.sem_seg_head = None
    gsnet.upsample1 = None
    gsnet.upsample2 = None
    gsnet.dino_decod_proj1 = None
    gsnet.dino_decod_proj2 = None
    if hasattr(gsnet, "layers"):
        gsnet.layers.clear()


def build_text_features(class_json, clip_model, device):
    import clip as openai_clip
    with open(class_json) as f:
        class_names = json.load(f)
    templates = ["a photo of a {}."]
    all_feats = []
    with torch.no_grad():
        for name in class_names:
            texts = [t.format(name) for t in templates]
            tokens = openai_clip.tokenize(texts).to(device)
            feats = clip_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
    text_feats = torch.stack(all_feats, dim=0).unsqueeze(0)
    return text_feats   # (1, T, P, C)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Online distillation
# ─────────────────────────────────────────────────────────────────────────────

_BRANCH_KEYS = ("dino_down", "dino_L4", "dino_L8")


def run_distill_epoch(teacher, student, loader, optimizer, scaler, device, amp,
                      train=True, log_interval=50, epoch=0, use_wandb=False):
    student.train(train)
    total_loss = 0.0
    branch_totals = {k: 0.0 for k in _BRANCH_KEYS}
    n = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for step, images in enumerate(tqdm(loader, leave=False,
                                           desc="distill-train" if train else "distill-val")):
            images = images.to(device)

            with torch.no_grad():
                teacher([{"image": (img * 255.0).clamp(0, 255)} for img in images])
                dino_down = teacher.dino_down_sample(
                    _dino_last[0].to(device).float()
                )

            targets = {
                "dino_down": dino_down,
                "dino_L4": _dino_l4[0].to(device).float(),
                "dino_L8": _dino_l8[0].to(device).float(),
            }

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                pred = student(siglip_normalise(images))
                breakdown = distillation_loss_per_branch(pred, targets)
                loss = breakdown["total"]

            if train:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            for k in _BRANCH_KEYS:
                branch_totals[k] += breakdown[k]
            n += 1

            if train and (step + 1) % log_interval == 0:
                avg = total_loss / n
                tqdm.write(
                    f"  step {step+1:5d}  loss={avg:.4f}  "
                    f"down={branch_totals['dino_down']/n:.4f}  "
                    f"l4={branch_totals['dino_L4']/n:.4f}  "
                    f"l8={branch_totals['dino_L8']/n:.4f}"
                )
                if use_wandb:
                    global_step = epoch * len(loader) + step + 1
                    wandb.log({
                        "distill/step/loss":    avg,
                        "distill/step/dino_down": branch_totals["dino_down"] / n,
                        "distill/step/dino_l4": branch_totals["dino_L4"] / n,
                        "distill/step/dino_l8": branch_totals["dino_L8"] / n,
                    }, step=global_step)

    avg_loss     = total_loss / max(n, 1)
    avg_branches = {k: v / max(n, 1) for k, v in branch_totals.items()}
    return avg_loss, avg_branches


def run_distillation(args, teacher, student, device, output_dir, use_wandb):
    full_ds = ImageDataset(args.image_dir, resolution=384)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  [Distill] Train: {n_train}  Val: {n_val}")

    optimizer = torch.optim.AdamW(student.trainable_parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    warmup_epochs = max(1, args.distill_epochs // 10)
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "student_distill_best.pth")

    for epoch in range(args.distill_epochs):
        lr = cosine_lr(optimizer, epoch, args.distill_epochs, warmup_epochs, args.lr)
        t0 = time.time()

        train_loss, train_b = run_distill_epoch(
            teacher, student, train_loader, optimizer, scaler, device,
            args.amp, train=True, log_interval=args.log_interval,
            epoch=epoch, use_wandb=use_wandb,
        )
        val_loss, val_b = run_distill_epoch(
            teacher, student, val_loader, optimizer, scaler, device,
            args.amp, train=False, epoch=epoch, use_wandb=False,
        )

        elapsed = time.time() - t0
        print(
            f"[Distill] Epoch {epoch+1:3d}/{args.distill_epochs}  lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.0f}s)  "
            f"val_down={val_b['dino_down']:.4f}  "
            f"val_l4={val_b['dino_L4']:.4f}  val_l8={val_b['dino_L8']:.4f}"
        )

        if use_wandb:
            wandb.log({
                "distill/epoch":         epoch + 1,
                "distill/lr":            lr,
                "distill/epoch_time_s":  elapsed,
                "distill/train/loss":    train_loss,
                "distill/train/dino_down": train_b["dino_down"],
                "distill/train/dino_l4": train_b["dino_L4"],
                "distill/train/dino_l8": train_b["dino_L8"],
                "distill/val/loss":      val_loss,
                "distill/val/dino_down": val_b["dino_down"],
                "distill/val/dino_l4":   val_b["dino_L4"],
                "distill/val/dino_l8":   val_b["dino_L8"],
            }, step=epoch + 1)

        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }, os.path.join(output_dir, "student_distill_latest.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ [Distill] New best val loss: {val_loss:.4f} → {best_path}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(output_dir, f"student_distill_epoch{epoch+1:03d}.pth")
            torch.save({"epoch": epoch, "student": student.state_dict(),
                        "val_loss": val_loss, "args": vars(args)}, ckpt_path)
            print(f"  → [Distill] Periodic checkpoint: {ckpt_path}")

    print(f"\n[Distill] Done. Best val loss: {best_val_loss:.4f}")
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Segmentation fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(args, gsnet, student, siglip_model, device, output_dir, use_wandb):
    clip_model       = gsnet.sem_seg_head.predictor.clip_model
    clip_skip_indices = tuple(args.siglip_skip_layers)
    remove_gsnet_clip_cache_hooks(gsnet)

    ripd              = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1    = gsnet.upsample1
    clip_upsample2    = gsnet.upsample2
    dino_decod_proj1  = gsnet.dino_decod_proj1
    dino_decod_proj2  = gsnet.dino_decod_proj2

    for p in ripd.parameters():
        p.requires_grad = False

    decoder_proj_modules = [m for m in [clip_upsample1, clip_upsample2,
                                         dino_decod_proj1, dino_decod_proj2]
                             if m is not None]
    for m in decoder_proj_modules:
        for p in m.parameters():
            p.requires_grad = False

    release_unused_gsnet(gsnet)
    gc.collect()
    torch.cuda.empty_cache()

    # Text features from teacher CLIP.
    text_feats = build_text_features(args.class_json, clip_model, device)

    full_ds = SegDataset(args.image_dir, args.label_dir)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  [Finetune] Train: {n_train}  Val: {n_val}")

    def _student_head_params():
        return (
            list(student.shared_trunk.parameters())
            + list(student.dino_down_branch.parameters())
            + list(student.dino_l4_branch.parameters())
            + list(student.dino_l8_branch.parameters())
            + list(student.clip_skip_adapters.parameters())
        )

    def _decoder_proj_params():
        return [p for m in decoder_proj_modules for p in m.parameters()]

    active_params = _student_head_params()

    optimizer  = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler     = GradScaler(enabled=args.amp)
    ignore_idx = 255
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "finetune_best.pth")

    def _save(path, epoch, avg_val):
        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "ripd": None,
            "clip_upsample1":   None,
            "clip_upsample2":   None,
            "dino_decod_proj1": None,
            "dino_decod_proj2": None,
            "val_loss": avg_val,
            "args": vars(args),
        }, path)

    for epoch in range(args.finetune_epochs):
        student.train()
        ripd.eval()

        train_loss = 0.0
        n_train_batches = 0
        grad_accum = args.grad_accum
        optimizer.zero_grad(set_to_none=True)
        _all_siglip_layers = sorted(set(list(clip_skip_indices) + list(student.siglip_layers)))

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} [train]")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.detach().expand(B, -1, -1, -1)

            # ── SigLIP forward (frozen, one pass for both RIPD skips + student trunk) ──
            with torch.no_grad():
                with autocast(enabled=args.amp):
                    last_hidden_state, skips = get_siglip_skips(siglip_model, images,
                                                                 _all_siglip_layers)
                    l0, l1 = clip_skip_indices
                    _stacked = torch.cat([skips[l] for l in student.siglip_layers], dim=1)

            with autocast(enabled=args.amp):
                res4_raw = student.adapt_clip_skip(skips[l0], 0) if clip_upsample1 is not None else None
                res5_raw = student.adapt_clip_skip(skips[l1], 1) if clip_upsample2 is not None else None
                res4 = clip_upsample1(res4_raw) if res4_raw is not None else None
                res5 = clip_upsample2(res5_raw) if res5_raw is not None else None

            # ── Student: per-branch checkpointing ────────────────────────────────
            if student.training:
                with autocast(enabled=args.amp):
                    _trunk = grad_ckpt.checkpoint(student.shared_trunk, _stacked, use_reentrant=False)
                del _stacked

                def _down(_x):
                    with autocast(enabled=args.amp):
                        return student.dino_down_branch(_x)
                dino_down = grad_ckpt.checkpoint(_down, _trunk, use_reentrant=False)

                def _dl4(_x):
                    with autocast(enabled=args.amp):
                        return student.dino_l4_branch(_x)
                _dino_L4 = grad_ckpt.checkpoint(_dl4, _trunk, use_reentrant=False) if dino_decod_proj1 is not None else None

                def _dl8(_x):
                    with autocast(enabled=args.amp):
                        return student.dino_l8_branch(_x)
                _dino_L8 = grad_ckpt.checkpoint(_dl8, _trunk, use_reentrant=False) if dino_decod_proj2 is not None else None
                del _trunk
            else:
                with torch.no_grad(), autocast(enabled=args.amp):
                    _trunk = student.shared_trunk(_stacked)
                    del _stacked
                    dino_down = student.dino_down_branch(_trunk)
                    _dino_L4 = student.dino_l4_branch(_trunk) if dino_decod_proj1 is not None else None
                    _dino_L8 = student.dino_l8_branch(_trunk) if dino_decod_proj2 is not None else None
                    del _trunk

            with autocast(enabled=args.amp):
                dino_L4_proj = dino_decod_proj1(_dino_L4) if _dino_L4 is not None else None
                dino_L8_proj = dino_decod_proj2(_dino_L8) if _dino_L8 is not None else None
            del _dino_L4, _dino_L8

            res3 = rearrange(last_hidden_state, "B (H W) C -> B C H W", H=24)
            clip_guidance = (res3, res4, res5)

            logit = ripd(
                res3,
                dino_down,
                tf,
                clip_guidance,
                [dino_L4_proj, dino_L8_proj],
            )

            with autocast(enabled=args.amp):
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                          mode="bilinear", align_corners=False)
                del logit
                loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx) / grad_accum
                del logit_up
            loss_value = loss.detach().item() * grad_accum

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_last_step = (step + 1 == len(train_loader))
            if (step + 1) % grad_accum == 0 or is_last_step:
                if args.amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(active_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(active_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss_value
            n_train_batches += 1
            del loss, loss_value, dino_down, tf, clip_guidance, dino_L4_proj, dino_L8_proj
            del res3, res4, res5, last_hidden_state, skips, images, labels

        student.eval()
        ripd.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} [val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]
                tf = text_feats.expand(B, -1, -1, -1)
                with autocast(enabled=args.amp):
                    logit = siglip_inference(
                        image=images, text_feats=tf, student=student,
                        siglip_model=siglip_model, ripd=ripd,
                        clip_upsample1=clip_upsample1, clip_upsample2=clip_upsample2,
                        dino_decod_proj1=dino_decod_proj1, dino_decod_proj2=dino_decod_proj2,
                        skip_layer_indices=clip_skip_indices,
                    )
                    logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                              mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)
                val_loss += loss.item()
                n_val_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val   = val_loss   / max(n_val_batches,   1)
        print(f"[Finetune] Epoch {epoch+1:3d}/{args.finetune_epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            wandb.log({
                "finetune/epoch":      epoch + 1,
                "finetune/train/loss": avg_train,
                "finetune/val/loss":   avg_val,
            }, step=args.distill_epochs + epoch + 1)

        _save(os.path.join(output_dir, "finetune_latest.pth"), epoch, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save(best_path, epoch, avg_val)
            print(f"  ✓ [Finetune] New best val loss: {avg_val:.4f} → {best_path}")

    print(f"\n[Finetune] Done. Best val loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Args + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SigLIP-backed GS-Distill: Phase 2 + Phase 3")
    p.add_argument("--gsnet-config",    required=True)
    p.add_argument("--gsnet-weights",   required=True)
    p.add_argument("--image-dir",       required=True)
    p.add_argument("--label-dir",       required=True)
    p.add_argument("--class-json",      default="datasets/landdiscover.json")
    p.add_argument("--output-dir",      default="output/ashie/siglip")
    p.add_argument("--siglip-model",    default="google/siglip-base-patch16-384")
    p.add_argument("--distill-epochs",  type=int,   default=30)
    p.add_argument("--finetune-epochs", type=int,   default=15)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--d-dino",          type=int,   default=768)
    p.add_argument("--siglip-layers",   type=int,   nargs="+", default=SIGLIP_LAYERS_DEFAULT)
    p.add_argument("--siglip-skip-layers", type=int, nargs="+", default=[3, 7])
    p.add_argument("--val-fraction",    type=float, default=0.05)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--grad-accum",      type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum).")
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-interval",    type=int,   default=50)
    p.add_argument("--wandb-project",   default="gs-distill")
    p.add_argument("--wandb-run",       default=None)
    p.add_argument("--no-wandb",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # ── Load SigLIP (student backbone) ───────────────────────────────────────
    print(f"Loading SigLIP from {args.siglip_model} ...")
    siglip_model = AutoModel.from_pretrained(args.siglip_model).to(device)
    siglip_model.eval()
    for p in siglip_model.parameters():
        p.requires_grad = False

    native_res = _siglip_native_res(siglip_model)
    backbone_dim_check = siglip_model.config.vision_config.hidden_size
    print(f"  SigLIP native res={native_res}  backbone_dim={backbone_dim_check}"
          f"  layers={args.siglip_layers}")

    # ── Load GSNet teacher ────────────────────────────────────────────────────
    print(f"Loading GSNet teacher from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    teacher = teacher.to(device)

    unpatch_dino = _patch_dino(teacher.dino_model)
    clip_skip_dims = (
        teacher.upsample1.in_channels if teacher.upsample1 is not None else backbone_dim_check,
        teacher.upsample2.in_channels if teacher.upsample2 is not None else backbone_dim_check,
    )

    # ── Build student ─────────────────────────────────────────────────────────
    student = SigLIPStudent(
        siglip_model=siglip_model,
        d_dino=args.d_dino,
        siglip_layers=args.siglip_layers,
        clip_skip_dims=clip_skip_dims,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"Student trainable params: {n_params / 1e6:.2f}M")

    # ── Phase 2: Distillation ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 2 — Online distillation (SigLIP student ← GSNet teacher)")
    print("="*60)
    best_distill_ckpt = run_distillation(args, teacher, student, device,
                                          args.output_dir, use_wandb)

    unpatch_dino()

    # ── Reload best distill checkpoint for Phase 3 ────────────────────────────
    print(f"\nReloading best distill checkpoint: {best_distill_ckpt}")
    ckpt = torch.load(best_distill_ckpt, map_location=device)
    student.load_state_dict(ckpt["student"])

    # ── Phase 3: Segmentation fine-tune ───────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 3 — Segmentation fine-tune on LD50K")
    print("="*60)
    run_finetune(args, teacher, student, siglip_model, device,
                 args.output_dir, use_wandb)

    if use_wandb:
        wandb.finish()
    print("\nAll done.")


if __name__ == "__main__":
    main()
