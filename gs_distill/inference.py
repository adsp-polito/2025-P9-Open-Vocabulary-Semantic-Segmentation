"""
GS-Distill inference pipeline.

Replaces the full GSNet Fusion Component (RSIB + QGFF) with:
  1. Frozen CLIP ViT-B/16  — provides base image features + decoder skip connections
  2. Trained student conv head — predicts fused_corr_embed and DINOv3 skip features
  3. Frozen RIPD decoder    — inherited unchanged from GSNet

Call gs_distill_inference() to get a segmentation logit tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import get_clip_skips


def gs_distill_inference(
    image: torch.Tensor,
    text_feats: torch.Tensor,
    student: nn.Module,
    clip_model: nn.Module,
    ripd: nn.Module,
    clip_upsample1: nn.Module,
    clip_upsample2: nn.Module,
    dino_decod_proj1: nn.Module,
    dino_decod_proj2: nn.Module,
    clip_skip_layer_indices=(3, 7),
    clip_resolution=(384, 384),
) -> torch.Tensor:
    """
    Full GS-Distill inference for a single image batch.

    Args:
        image:         (B, 3, H, W) CLIP-normalised, already at clip_resolution.
        text_feats:    (B, T, P, C) text embeddings from CLIP; T=num classes, P=prompts, C=512.
        student:       trained GSDistillStudent.
        clip_model:    frozen OpenAI CLIP model.
        ripd:          frozen RIPD decoder from GSNet.
        clip_upsample1: GSNet's upsample1 ConvTranspose2d (768→256, 2x) for res4 skip.
        clip_upsample2: GSNet's upsample2 ConvTranspose2d (768→128, 4x) for res5 skip.
        dino_decod_proj1: GSNet's dino_decod_proj1 Conv2d (768→256) for DINOv3 L4 skip.
        dino_decod_proj2: GSNet's dino_decod_proj2 ConvTranspose2d (768→128, 2x) for DINOv3 L8 skip.
        clip_skip_layer_indices: CLIP resblock indices used for decoder skip connections (res4, res5).
        clip_resolution: spatial size used for CLIP forward pass.

    Returns:
        logit: (B, T, H, W) segmentation logits.
    """
    device = image.device

    with torch.no_grad():
        # ── 1. CLIP forward: base features + decoder skip connections ──────────
        clip_features, clip_skips = get_clip_skips(
            clip_model, image, list(clip_skip_layer_indices)
        )
        # clip_features: (B, 577, 768)  (CLS + 576 patch tokens)
        # clip_skips:    {layer_idx: (B, 768, 24, 24)}

        l0, l1 = clip_skip_layer_indices
        res4_raw = clip_skips[l0]   # (B, 768, 24, 24)
        res5_raw = clip_skips[l1]   # (B, 768, 24, 24)

        # Project CLIP skip features for RIPD decoder guidance
        # Matches how GSNet.py lines 312-313 produce res4/res5
        res4 = clip_upsample1(res4_raw) if clip_upsample1 is not None else None  # (B,256,48,48)
        res5 = clip_upsample2(res5_raw) if clip_upsample2 is not None else None  # (B,128,96,96)

        # ── 2. Student conv head ─────────────────────────────────────────────
        student_out = student(image)
        fused_corr_embed     = student_out["fused_corr_embed"]   # (B, hidden_dim, T, 24, 24)
        predicted_dino_L4    = student_out["dino_L4"]            # (B, 768, 48, 48)
        predicted_dino_L8    = student_out["dino_L8"]            # (B, 768, 48, 48)

        # ── 3. Project predicted DINOv3 features for RIPD decoder guidance ───
        # Matches GSNet.py lines 300-301 (dino_decod_proj1 / dino_decod_proj2)
        dino_L4_proj = dino_decod_proj1(predicted_dino_L4) if dino_decod_proj1 is not None else None
        dino_L8_proj = dino_decod_proj2(predicted_dino_L8) if dino_decod_proj2 is not None else None

        clip_guidance = {
            "res3": rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24),
            "res4": res4,
            "res5": res5,
        }
        dino_guidance = [dino_L4_proj, dino_L8_proj]

        # ── 4. RIPD decoder (skip QGFF — feed pre-computed fused_corr_embed) ─
        logit = ripd.forward_from_fusion(
            fused_corr_embed=fused_corr_embed,
            text_feats=text_feats,
            appearance_guidance=clip_guidance,
            dino_guidance=dino_guidance,
        )

    return logit
