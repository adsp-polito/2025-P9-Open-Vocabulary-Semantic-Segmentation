"""
Distillation loss for GS-Distill Phase 2.

Temperature-scaled BCE: L = BCE(student/τ, sigmoid(teacher/τ)) × τ²
This matches the knowledge distillation recipe from Hinton et al.,
adapted for per-pixel multi-label prediction.
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
