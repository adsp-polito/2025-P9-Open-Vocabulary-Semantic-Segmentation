"""
RIPD-free Student-2 inference path.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .student2 import GSDistillStudent2


def gs_distill_s2_inference(
    image: torch.Tensor,
    text_feats: torch.Tensor,
    student2: GSDistillStudent2,
    clip_model: nn.Module,
    clip_skip_layer_indices: Tuple[int, int] = (7, 15),
    clip_features: Optional[torch.Tensor] = None,
    clip_skips: Optional[dict] = None,
    student_clip_stacked: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Student-2 inference without DINO, RIPD, or QGFF.

    Args:
        image: (B, 3, H, W) CLIP-normalised image.
        text_feats: (B, T, P, C) CLIP text features.
        student2: GSDistillStudent2 model.
        clip_model: kept for API symmetry with gs_distill_inference; the model
            uses student2.clip_model internally.
        clip_skip_layer_indices: CLIP skip layers used by the cost head decoder.
        clip_features/clip_skips/student_clip_stacked: optional cached CLIP outputs.

    Returns:
        (B, T, H_out, W_out) segmentation logits.
    """
    del clip_model
    return student2(
        image=image,
        text_feats=text_feats,
        clip_skip_indices=clip_skip_layer_indices,
        clip_features=clip_features,
        clip_skips=clip_skips,
        student_clip_stacked=student_clip_stacked,
    )
