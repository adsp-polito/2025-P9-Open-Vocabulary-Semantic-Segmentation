"""
GS-Distill inference pipeline.

Replaces the full GSNet Fusion Component (RSIB + QGFF) with:
  1. Frozen CLIP ViT — provides base image features + decoder skip connections
  2. Trained student conv head — predicts DINO substitute features
  3. Frozen RIPD — computes dynamic CLIP/DINO correlations for the active text

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
    clip_skip_layer_indices=(7, 15),
    clip_resolution=(384, 384),
    clip_features=None,
    clip_skips=None,
    student_clip_stacked=None,
) -> torch.Tensor:
    """
    Full GS-Distill inference for a single image batch.

    Args:
        image:         (B, 3, H, W) CLIP-normalised, already at clip_resolution.
        text_feats:    (B, T, P, C) text embeddings from CLIP; T=num classes, P=prompts, C=512.
        student:       trained GSDistillStudent that predicts DINO substitutes.
        clip_model:    frozen OpenAI CLIP model.
        ripd:          frozen RIPD module from GSNet.
        clip_upsample1: GSNet's upsample1 ConvTranspose2d (768→256, 2x) for res4 skip.
        clip_upsample2: GSNet's upsample2 ConvTranspose2d (768→128, 4x) for res5 skip.
        dino_decod_proj1: GSNet's dino_decod_proj1 Conv2d (768→256) for DINOv3 L4 skip.
        dino_decod_proj2: GSNet's dino_decod_proj2 ConvTranspose2d (768→128, 2x) for DINOv3 L8 skip.
        clip_skip_layer_indices: CLIP resblock indices used for decoder skip connections (res4, res5).
        clip_resolution: spatial size used for CLIP forward pass.
        clip_features:  pre-computed dense CLIP features (B, seq, C); if provided, skips CLIP forward.
        clip_skips:     pre-computed skip dict {layer_idx: (B, C, H, W)}; required when clip_features is given.
        student_clip_stacked: pre-extracted (B, trunk_in, H_grid, W_grid) for student heads;
                              if provided, student.forward_from_features() is called instead of student().

    Returns:
        logit: (B, T, H, W) segmentation logits.
    """
    # ── 1. CLIP forward: base features + decoder skip connections ────────────
    if clip_features is None or clip_skips is None:
        all_layers = sorted(set(list(clip_skip_layer_indices) + student.clip_layers))
        with torch.no_grad():
            clip_features, clip_skips = get_clip_skips(clip_model, image, all_layers)
            if student_clip_stacked is None:
                student_clip_stacked = torch.cat(
                    [clip_skips[l] for l in student.clip_layers], dim=1
                )

    l0, l1 = clip_skip_layer_indices
    with torch.no_grad():
        res4 = clip_upsample1(clip_skips[l0]) if clip_upsample1 is not None else None
        res5 = clip_upsample2(clip_skips[l1]) if clip_upsample2 is not None else None

    # ── 2. Student DINO substitute heads ─────────────────────────────────────
    if student_clip_stacked is not None:
        student_out = student.forward_from_features(student_clip_stacked)
    else:
        student_out = student(image)
    predicted_dino_down = student_out["dino_down"]        # (B, 768, 24, 24)
    predicted_dino_L4 = student_out["dino_L4"]            # (B, 768, 48, 48)
    predicted_dino_L8 = student_out["dino_L8"]            # (B, 768, 48, 48)

    # ── 3. Project predicted DINOv3 features for RIPD decoder guidance ───────
    dino_L4_proj = dino_decod_proj1(predicted_dino_L4) if dino_decod_proj1 is not None else None
    dino_L8_proj = dino_decod_proj2(predicted_dino_L8) if dino_decod_proj2 is not None else None

    clip_res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
    clip_guidance = (clip_res3, res4, res5)
    dino_guidance = [dino_L4_proj, dino_L8_proj]

    # ── 4. Normal RIPD path: dynamic CLIP/DINO correlations + fusion ─────────
    logit = ripd(clip_res3, predicted_dino_down, text_feats, clip_guidance, dino_guidance)

    return logit
