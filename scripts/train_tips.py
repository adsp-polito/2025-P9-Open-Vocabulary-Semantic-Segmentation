"""
TIPSv2 backbone — combined Phase 2 (online distillation) + Phase 3 (segmentation fine-tune).

Replaces the OpenAI CLIP backbone with google/tipsv2-b14 (HuggingFace, trust_remote_code=True).
The student architecture, all 4 distillation targets, RIPD decoder, and LD50K pipeline
are identical to the CLIP-based GS-Distill pipeline.

Key differences from the CLIP pipeline:
  - TIPSv2 loaded via AutoModel with trust_remote_code=True
  - Layer path: model.vision_encoder.blocks[l]  (NOT vision_model.encoder.layers)
  - Hook output: (B, seq_len, C) — seq = [CLS, register, patch×1024]; patch tokens start at index 2
  - Image normalisation: mean=0.5, std=0.5  (same as SigLIP)
  - Text features: model.encode_text(list_of_strings) — takes raw strings, no tokenizer call
  - config.hidden_size at top level (not nested under vision_config)
  - Resolution: 448×448, patch 14 → 32×32 = 1024 patch tokens
  - GSNet teacher still uses original CLIP — loaded from the GSNet checkpoint as usual
  - No Phase 1 attention fine-tuning required for TIPSv2

Phase 2 (distillation) completes first, saves student_distill_best.pth, then
Phase 3 (fine-tune) starts automatically from that checkpoint.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/train_tips.py \\
        --gsnet-config  configs/vitb_384.yaml \\
        --gsnet-weights path/to/gsnet.pth \\
        --image-dir     path/to/LD50K/TR_Image \\
        --label-dir     path/to/LD50K/TR_Label \\
        --output-dir    output/ashie/tips \\
        [--distill-epochs 30] [--finetune-epochs 15] \\
        [--batch-size 4] [--lr 1e-5] [--amp] [--wandb-project gs-distill]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# TIPSv2 uses the same normalisation as SigLIP
_TIPS_MEAN = torch.tensor([0.5, 0.5, 0.5])
_TIPS_STD  = torch.tensor([0.5, 0.5, 0.5])

# TIPSv2-B14 @ 448px: 12 transformer layers (0–11), 32×32 = 1024 patch tokens.
# Mirror CLIP [4, 8, 10, 12] depth-wise → [4, 8, 10, 11] (max index = 11).
TIPS_LAYERS_DEFAULT = [4, 8, 10, 11]
TRUNK_OUT = 512

# Tokens before patch tokens in the TIPSv2 sequence: [CLS(1) + register(1)]
_TIPS_PREFIX_TOKENS = 2


# ─────────────────────────────────────────────────────────────────────────────
# TIPSv2 feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tips_native_res(tips_model) -> int:
    return tips_model.config.image_size


def extract_tips_layers(tips_model, image: torch.Tensor, layers: list) -> torch.Tensor:
    """
    Extract intermediate patch features from a TIPSv2 ViT for specified block indices.

    TIPSv2 sequence layout: [CLS(0), register(1), patch_0 … patch_1023]
    Patch tokens start at index _TIPS_PREFIX_TOKENS (2), not 1 like CLIP/SigLIP.

    Hook output shape: (B, seq_len, C) where seq_len = 1026 for B14@448px.

    Args:
        tips_model: frozen TIPSv2 model (google/tipsv2-b14).
        image: (B, 3, H, W) TIPSv2-normalised, any spatial size.
        layers: list of vision_encoder.blocks indices, e.g. [4, 8, 10, 11].

    Returns:
        (B, len(layers)*C, H_grid, W_grid) stacked spatial patch features.
    """
    native_res = _tips_native_res(tips_model)
    if image.shape[-2] != native_res or image.shape[-1] != native_res:
        image = F.interpolate(image, size=(native_res, native_res),
                              mode="bilinear", align_corners=False)

    captured = {}
    hooks = []
    for l in layers:
        def make_hook(idx):
            def hook(module, input, output):
                # output: (B, seq_len, C)
                captured[idx] = output
            return hook
        h = tips_model.vision_encoder.blocks[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        tips_model.vision_encoder(image)

    for h in hooks:
        h.remove()

    feature_maps = []
    for l in layers:
        feat = captured[l]                          # (B, seq_len, C)
        feat = feat[:, _TIPS_PREFIX_TOKENS:, :]     # drop CLS + register → (B, n_patches, C)
        # transpose to (n_patches, B, C) then reshape to (B, C, H, W)
        feat = feat.permute(1, 0, 2)               # (n_patches, B, C)
        n = feat.shape[0]
        H = W = int(n ** 0.5)
        feat = rearrange(feat, "(H W) B C -> B C H W", H=H, W=W)
        feature_maps.append(feat)

    return torch.cat(feature_maps, dim=1)   # (B, len(layers)*C, H_grid, W_grid)


def get_tips_skips(tips_model, image: torch.Tensor, layer_indices: list):
    """
    Return patch feature maps at given block indices and the full last hidden state.
    Mirrors get_clip_skips() / get_siglip_skips() for use in siglip_inference equivalent.

    Returns:
        last_hidden: (B, seq_len, C) full last-layer output (used for res3 guidance)
        skips: dict mapping layer_index -> (B, C, H_grid, W_grid)
    """
    native_res = _tips_native_res(tips_model)
    if image.shape[-2] != native_res or image.shape[-1] != native_res:
        image = F.interpolate(image, size=(native_res, native_res),
                              mode="bilinear", align_corners=False)

    captured = {}
    hooks = []
    for l in layer_indices:
        def make_hook(idx):
            def hook(module, input, output):
                captured[idx] = output   # (B, seq_len, C)
            return hook
        h = tips_model.vision_encoder.blocks[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        vit_out = tips_model.vision_encoder(image)

    for h in hooks:
        h.remove()

    # last_hidden_state from vision_encoder forward — use for res3 guidance
    last_hidden = vit_out  # TIPSv2 vision_encoder returns the token sequence directly

    skips = {}
    for l in layer_indices:
        feat = captured[l][:, _TIPS_PREFIX_TOKENS:, :]   # drop CLS + register
        feat = feat.permute(1, 0, 2)                      # (n_patches, B, C)
        n = feat.shape[0]
        H = W = int(n ** 0.5)
        skips[l] = rearrange(feat, "(H W) B C -> B C H W", H=H, W=W)

    return last_hidden, skips


def tips_normalise(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [0, 1] → TIPSv2-normalised."""
    mean = _TIPS_MEAN.to(images.device).view(1, 3, 1, 1)
    std  = _TIPS_STD.to(images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# TIPSv2 student — identical architecture to GSDistillStudent, TIPS backbone
# ─────────────────────────────────────────────────────────────────────────────

class TIPSStudent(nn.Module):
    """
    GS-Distill student with TIPSv2 ViT backbone instead of OpenAI CLIP.

    Architecture identical to GSDistillStudent / SigLIPStudent:
      - Frozen TIPSv2 ViT (multi-layer features)
      - Shared conv trunk (len(layers)*C → 512 channels)
      - Fusion head:  predicts fused_corr_embed  (B, hidden_dim, T, H, W)
      - CLIP head:    predicts clip_embed_corr   (B, hidden_dim, T, H, W)
      - DINO-L4 head: predicts dino_L4           (B, d_dino, 2H, 2W)
      - DINO-L8 head: predicts dino_L8           (B, d_dino, 2H, 2W)
    """

    def __init__(self, tips_model: nn.Module, hidden_dim: int = 128,
                 d_dino: int = 768, num_classes: int = 40,
                 tips_layers: list = None):
        super().__init__()

        self.tips_model = tips_model
        for p in self.tips_model.parameters():
            p.requires_grad = False

        self.tips_layers  = tips_layers if tips_layers is not None else TIPS_LAYERS_DEFAULT
        self.hidden_dim   = hidden_dim
        self.d_dino       = d_dino
        self.num_classes  = num_classes

        # hidden_size is top-level in TIPSv2 config (768 for B14)
        clip_dim  = tips_model.config.hidden_size
        trunk_in  = len(self.tips_layers) * clip_dim

        self.shared_trunk = nn.Sequential(
            nn.Conv2d(trunk_in, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(TRUNK_OUT, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.fusion_branch = nn.Conv2d(TRUNK_OUT, hidden_dim * num_classes,
                                       kernel_size=3, padding=1)
        self.clip_embed_branch = nn.Conv2d(TRUNK_OUT, hidden_dim * num_classes,
                                           kernel_size=3, padding=1)
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

    def forward(self, image: torch.Tensor) -> dict:
        with torch.no_grad():
            stacked = extract_tips_layers(self.tips_model, image, self.tips_layers)

        trunk_out = self.shared_trunk(stacked)   # (B, 512, H_grid, W_grid)

        def _corr_reshape(x):
            return rearrange(x, "B (C T) H W -> B C T H W",
                             C=self.hidden_dim, T=self.num_classes)

        return {
            "fused_corr_embed": _corr_reshape(self.fusion_branch(trunk_out)),
            "clip_embed_corr":  _corr_reshape(self.clip_embed_branch(trunk_out)),
            "dino_L4": self.dino_l4_branch(trunk_out),
            "dino_L8": self.dino_l8_branch(trunk_out),
        }

    def trainable_parameters(self):
        return (
            list(self.shared_trunk.parameters())
            + list(self.fusion_branch.parameters())
            + list(self.clip_embed_branch.parameters())
            + list(self.dino_l4_branch.parameters())
            + list(self.dino_l8_branch.parameters())
        )


# ─────────────────────────────────────────────────────────────────────────────
# TIPSv2 inference — replaces gs_distill_inference for this backbone
# ─────────────────────────────────────────────────────────────────────────────

def tips_inference(
    image, text_feats, student, tips_model,
    ripd, clip_upsample1, clip_upsample2,
    dino_decod_proj1, dino_decod_proj2,
    skip_layer_indices=(3, 7),
):
    """
    Full inference using TIPSv2 features instead of CLIP.
    Mirrors siglip_inference() but uses get_tips_skips().

    res3 guidance uses patch tokens from the TIPSv2 last hidden state
    (drop the 2 prefix tokens: CLS + register).
    """
    with torch.no_grad():
        last_hidden, skips = get_tips_skips(tips_model, image, list(skip_layer_indices))
        l0, l1 = skip_layer_indices
        res4_raw = skips[l0]
        res5_raw = skips[l1]
        res4 = clip_upsample1(res4_raw) if clip_upsample1 is not None else None
        res5 = clip_upsample2(res5_raw) if clip_upsample2 is not None else None

    student_out = student(image)
    fused_corr_embed  = student_out["fused_corr_embed"]
    predicted_dino_L4 = student_out["dino_L4"]
    predicted_dino_L8 = student_out["dino_L8"]

    dino_L4_proj = dino_decod_proj1(predicted_dino_L4) if dino_decod_proj1 is not None else None
    dino_L8_proj = dino_decod_proj2(predicted_dino_L8) if dino_decod_proj2 is not None else None

    # res3: patch tokens from last hidden state, drop 2 prefix tokens (CLS + register)
    # last_hidden shape depends on TIPSv2 vision_encoder return; handle both tensor and object
    if hasattr(last_hidden, "last_hidden_state"):
        patch_seq = last_hidden.last_hidden_state[:, _TIPS_PREFIX_TOKENS:, :]
    else:
        patch_seq = last_hidden[:, _TIPS_PREFIX_TOKENS:, :]   # (B, n_patches, C)

    res3 = rearrange(patch_seq, "B (H W) C -> B C H W", H=32)   # 448/14 = 32

    clip_guidance = {"res3": res3, "res4": res4, "res5": res5}
    dino_guidance = [dino_L4_proj, dino_L8_proj]

    logit = ripd.forward_from_fusion(
        fused_corr_embed=fused_corr_embed,
        text_feats=text_feats,
        appearance_guidance=clip_guidance,
        dino_guidance=dino_guidance,
    )
    return logit


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """Image-only dataset for Phase 2 distillation (no labels)."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, resolution=448):
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

    def __init__(self, image_dir, label_dir, resolution=448):
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
# Teacher setup (identical to train_distill_online.py / train_siglip.py)
# ─────────────────────────────────────────────────────────────────────────────

_hook_state = {"fused_corr_embed": None, "clip_embed_corr": None}
_dino_l4    = [None]
_dino_l8    = [None]


def _patch_ripd(ripd_module):
    original_forward = ripd_module.forward

    def patched_forward(img_feats, dino_feat, text_feats, appearance_guidance, dino_guidance):
        if dino_feat is not None and img_feats is not None:
            if ripd_module.fusion_type == 'query_guided':
                corr      = ripd_module.correlation(img_feats, text_feats)
                dino_corr = ripd_module.correlation(dino_feat, text_feats)
                fused_corr, clip_embed_corr, _ = ripd_module.corr_fusion_embed_seperate(
                    clip_corr=corr, dino_corr=dino_corr,
                )
                fused_corr_embed = fused_corr + clip_embed_corr
            elif ripd_module.fusion_type == 'simple_concatenation':
                simple_corr = ripd_module.simple_concatenation_corr(img_feats, dino_feat, text_feats)
                T = simple_corr.shape[2]
                fused_corr_embed = ripd_module.simple_concatenation_corr_embed(
                    rearrange(simple_corr, "B C T H W -> (B T) C H W"))
                fused_corr_embed = rearrange(fused_corr_embed, "(B T) C H W -> B C T H W", T=T)
                clip_embed_corr  = torch.zeros_like(fused_corr_embed)
            elif ripd_module.fusion_type == 'fusion_query':
                fused_feat = ripd_module.fusion_feats(torch.cat([img_feats, dino_feat], dim=1))
                corr = ripd_module.correlation(fused_feat, text_feats)
                fused_corr_embed = ripd_module.corr_embed(corr)
                clip_embed_corr  = torch.zeros_like(fused_corr_embed)
        elif dino_feat is not None:
            corr = ripd_module.correlation(dino_feat, text_feats)
            fused_corr_embed = ripd_module.corr_embed(corr)
            clip_embed_corr  = torch.zeros_like(fused_corr_embed)
        elif img_feats is not None:
            corr = ripd_module.correlation(img_feats, text_feats)
            fused_corr_embed = ripd_module.corr_embed(corr)
            clip_embed_corr  = torch.zeros_like(fused_corr_embed)

        _hook_state["fused_corr_embed"] = fused_corr_embed.detach().cpu().half()
        _hook_state["clip_embed_corr"]  = clip_embed_corr.detach().cpu().half()

        return original_forward(img_feats, dino_feat, text_feats, appearance_guidance, dino_guidance)

    ripd_module.forward = patched_forward

    def unpatch():
        ripd_module.forward = original_forward
    return unpatch


def _patch_dino(dino_model):
    original_fn = dino_model.get_intermediate_layers

    def patched_fn(imgs, n=12):
        result = original_fn(imgs, n)
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
# Text features — uses teacher CLIP text encoder (same as all other scripts)
# ─────────────────────────────────────────────────────────────────────────────

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

_BRANCH_KEYS = ("fused_corr_embed", "clip_embed_corr", "dino_L4", "dino_L8")


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

            targets = {
                "fused_corr_embed": _hook_state["fused_corr_embed"].to(device).float(),
                "clip_embed_corr":  _hook_state["clip_embed_corr"].to(device).float(),
                "dino_L4": _dino_l4[0].to(device).float(),
                "dino_L8": _dino_l8[0].to(device).float(),
            }

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                pred = student(tips_normalise(images))
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
                    f"fused={branch_totals['fused_corr_embed']/n:.4f}  "
                    f"clip={branch_totals['clip_embed_corr']/n:.4f}  "
                    f"l4={branch_totals['dino_L4']/n:.4f}  "
                    f"l8={branch_totals['dino_L8']/n:.4f}"
                )
                if use_wandb:
                    global_step = epoch * len(loader) + step + 1
                    wandb.log({
                        "distill/step/loss":    avg,
                        "distill/step/fused":   branch_totals["fused_corr_embed"] / n,
                        "distill/step/clip":    branch_totals["clip_embed_corr"]  / n,
                        "distill/step/dino_l4": branch_totals["dino_L4"] / n,
                        "distill/step/dino_l8": branch_totals["dino_L8"] / n,
                    }, step=global_step)

    avg_loss     = total_loss / max(n, 1)
    avg_branches = {k: v / max(n, 1) for k, v in branch_totals.items()}
    return avg_loss, avg_branches


def run_distillation(args, teacher, student, device, output_dir, use_wandb):
    full_ds = ImageDataset(args.image_dir, resolution=448)
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
            f"val_fused={val_b['fused_corr_embed']:.4f}  "
            f"val_clip={val_b['clip_embed_corr']:.4f}  "
            f"val_l4={val_b['dino_L4']:.4f}  val_l8={val_b['dino_L8']:.4f}"
        )

        if use_wandb:
            wandb.log({
                "distill/epoch":         epoch + 1,
                "distill/lr":            lr,
                "distill/epoch_time_s":  elapsed,
                "distill/train/loss":    train_loss,
                "distill/train/fused":   train_b["fused_corr_embed"],
                "distill/train/clip":    train_b["clip_embed_corr"],
                "distill/train/dino_l4": train_b["dino_L4"],
                "distill/train/dino_l8": train_b["dino_L8"],
                "distill/val/loss":      val_loss,
                "distill/val/fused":     val_b["fused_corr_embed"],
                "distill/val/clip":      val_b["clip_embed_corr"],
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

def run_finetune(args, gsnet, student, tips_model, device, output_dir, use_wandb):
    clip_model        = gsnet.sem_seg_head.predictor.clip_model
    clip_skip_indices = tuple(gsnet.layer_indexes)

    ripd              = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1    = gsnet.upsample1
    clip_upsample2    = gsnet.upsample2
    dino_decod_proj1  = gsnet.dino_decod_proj1
    dino_decod_proj2  = gsnet.dino_decod_proj2

    for p in ripd.parameters():
        p.requires_grad = True

    decoder_proj_modules = [m for m in [clip_upsample1, clip_upsample2,
                                         dino_decod_proj1, dino_decod_proj2]
                             if m is not None]
    for m in decoder_proj_modules:
        for p in m.parameters():
            p.requires_grad = True

    text_feats = build_text_features(args.class_json, clip_model, device)

    full_ds = SegDataset(args.image_dir, args.label_dir, resolution=448)
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
            + list(student.fusion_branch.parameters())
            + list(student.dino_l4_branch.parameters())
            + list(student.dino_l8_branch.parameters())
        )

    def _decoder_proj_params():
        return [p for m in decoder_proj_modules for p in m.parameters()]

    active_params = (
        _student_head_params()
        + list(ripd.parameters())
        + _decoder_proj_params()
    )

    optimizer  = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler     = GradScaler(enabled=args.amp)
    ignore_idx = 255
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "finetune_best.pth")

    def _save(path, epoch, avg_val):
        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "ripd": ripd.state_dict(),
            "clip_upsample1":   clip_upsample1.state_dict()   if clip_upsample1   is not None else None,
            "clip_upsample2":   clip_upsample2.state_dict()   if clip_upsample2   is not None else None,
            "dino_decod_proj1": dino_decod_proj1.state_dict() if dino_decod_proj1 is not None else None,
            "dino_decod_proj2": dino_decod_proj2.state_dict() if dino_decod_proj2 is not None else None,
            "val_loss": avg_val,
        }, path)

    for epoch in range(args.finetune_epochs):
        student.train()
        ripd.train()

        train_loss = 0.0
        n_train_batches = 0

        for images, labels in tqdm(train_loader,
                                    desc=f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} [train]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.expand(B, -1, -1, -1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                logit = tips_inference(
                    image=images, text_feats=tf, student=student,
                    tips_model=tips_model, ripd=ripd,
                    clip_upsample1=clip_upsample1, clip_upsample2=clip_upsample2,
                    dino_decod_proj1=dino_decod_proj1, dino_decod_proj2=dino_decod_proj2,
                    skip_layer_indices=clip_skip_indices,
                )
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                          mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(active_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(active_params, 1.0)
                optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        student.eval()
        ripd.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                        desc=f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} [val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]
                tf = text_feats.expand(B, -1, -1, -1)
                with autocast(enabled=args.amp):
                    logit = tips_inference(
                        image=images, text_feats=tf, student=student,
                        tips_model=tips_model, ripd=ripd,
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
        print(f"[Finetune] Epoch {epoch+1:3d}/{args.finetune_epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}")

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
    p = argparse.ArgumentParser(description="TIPSv2-backed GS-Distill: Phase 2 + Phase 3")
    p.add_argument("--gsnet-config",    required=True)
    p.add_argument("--gsnet-weights",   required=True)
    p.add_argument("--image-dir",       required=True)
    p.add_argument("--label-dir",       required=True)
    p.add_argument("--class-json",      default="datasets/landdiscover.json")
    p.add_argument("--output-dir",      default="output/ashie/tips")
    p.add_argument("--tips-model",      default="google/tipsv2-b14")
    p.add_argument("--distill-epochs",  type=int,   default=30)
    p.add_argument("--finetune-epochs", type=int,   default=15)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--d-dino",          type=int,   default=768)
    p.add_argument("--num-classes",     type=int,   default=40)
    p.add_argument("--tips-layers",     type=int,   nargs="+", default=TIPS_LAYERS_DEFAULT)
    p.add_argument("--val-fraction",    type=float, default=0.05)
    p.add_argument("--num-workers",     type=int,   default=4)
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

    # ── Load TIPSv2 (student backbone) ───────────────────────────────────────
    print(f"Loading TIPSv2 from {args.tips_model} ...")
    tips_model = AutoModel.from_pretrained(args.tips_model, trust_remote_code=True).to(device)
    tips_model.eval()
    for p in tips_model.parameters():
        p.requires_grad = False

    native_res = _tips_native_res(tips_model)
    hidden_dim_check = tips_model.config.hidden_size
    print(f"  TIPSv2 native res={native_res}  hidden_dim={hidden_dim_check}"
          f"  layers={args.tips_layers}")

    # ── Load GSNet teacher ────────────────────────────────────────────────────
    print(f"Loading GSNet teacher from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    teacher = teacher.to(device)

    unpatch_ripd = _patch_ripd(teacher.sem_seg_head.predictor.transformer)
    unpatch_dino = _patch_dino(teacher.dino_model)

    # ── Build student ─────────────────────────────────────────────────────────
    student = TIPSStudent(
        tips_model=tips_model,
        hidden_dim=args.hidden_dim,
        d_dino=args.d_dino,
        num_classes=args.num_classes,
        tips_layers=args.tips_layers,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"Student trainable params: {n_params / 1e6:.2f}M")

    # ── Phase 2: Distillation ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 2 — Online distillation (TIPSv2 student ← GSNet teacher)")
    print("="*60)
    best_distill_ckpt = run_distillation(args, teacher, student, device,
                                          args.output_dir, use_wandb)

    unpatch_ripd()
    unpatch_dino()

    # ── Reload best distill checkpoint for Phase 3 ────────────────────────────
    print(f"\nReloading best distill checkpoint: {best_distill_ckpt}")
    ckpt = torch.load(best_distill_ckpt, map_location=device)
    student.load_state_dict(ckpt["student"])

    # ── Phase 3: Segmentation fine-tune ───────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 3 — Segmentation fine-tune on LD50K")
    print("="*60)
    run_finetune(args, teacher, student, tips_model, device,
                 args.output_dir, use_wandb)

    if use_wandb:
        wandb.finish()
    print("\nAll done.")


if __name__ == "__main__":
    main()
