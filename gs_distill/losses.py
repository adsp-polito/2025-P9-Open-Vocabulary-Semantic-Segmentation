"""
Distillation losses for GS-Distill.

distillation_loss  — temperature-scaled BCE on output logits (Phase 2 + Phase 3).
feature_distill_loss — cosine similarity on patch features (Phase 2 v2 only).

The feature loss gives the SpatialAdapter a direct spatial supervision signal:
it is trained to make TIPS patch features resemble GSNet's raw CLIP patch features
(same ViT-L/14 architecture, same 1024-dim space, same 24×24 grid).
This is more informative than output-level distillation alone because the adapter
receives dense per-patch gradient rather than a signal diluted through correlation
and decoder layers.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
) -> torch.Tensor:
    """
    Temperature-scaled binary cross-entropy distillation loss.

    Args:
        student_logits: (B, T, H, W) raw logits from student.
        teacher_logits: (B, T, H, W) raw logits from teacher cache.
        tau: temperature for softening distributions (default 4.0).

    Returns:
        Scalar loss tensor.
    """
    teacher_soft = torch.sigmoid(teacher_logits / tau)
    student_soft = student_logits / tau
    return F.binary_cross_entropy_with_logits(student_soft, teacher_soft) * (tau ** 2)


def feature_distill_loss(
    student_feats: torch.Tensor,
    teacher_clip_feats: torch.Tensor,
) -> torch.Tensor:
    """
    Feature-level distillation: 1 - mean cosine similarity between student
    patch features and cached GSNet CLIP patch features.

    Both tensors are L2-normalised along the channel dimension before comparison.
    student_feats is already normalised (adapter output after F.normalize);
    teacher_clip_feats are raw fp16 CLIP features from the cache and are
    normalised here.

    Cosine similarity is computed per patch location and averaged over all
    B × H × W locations, giving a dense spatial supervision signal.

    Args:
        student_feats:      (B, D, H, W)  L2-normalised adapter output.
        teacher_clip_feats: (B, D, H, W)  raw CLIP features from cache (fp16/fp32).

    Returns:
        Scalar loss tensor in [0, 2].  Perfect alignment → 0.
    """
    B, D, H, W = student_feats.shape
    t = F.normalize(
        teacher_clip_feats.to(dtype=student_feats.dtype, device=student_feats.device),
        dim=1,
    )
    # Reshape to (B*H*W, D) for vectorised cosine similarity
    s_flat = student_feats.permute(0, 2, 3, 1).reshape(-1, D)
    t_flat = t.permute(0, 2, 3, 1).reshape(-1, D)
    return 1.0 - F.cosine_similarity(s_flat, t_flat, dim=1).mean()
