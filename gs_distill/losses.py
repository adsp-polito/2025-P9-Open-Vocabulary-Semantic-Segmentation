"""
Distillation loss for GS-Distill.

Per-branch: Smooth L1 + (1 - cosine_similarity) applied to flattened spatial
tensors. The dynamic CLIP path distills text-independent DINO substitute
features: dino_down, dino_L4, and dino_L8.
"""

import torch
import torch.nn.functional as F


def distillation_loss(pred: dict, target: dict) -> torch.Tensor:
    """
    Compute the combined distillation loss across all three prediction branches.

    Args:
        pred:   dict with keys 'dino_down', 'dino_L4', 'dino_L8'.
        target: dict with same keys (teacher cache loaded from .pt files).

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=next(iter(pred.values())).device)
    for key in ("dino_down", "dino_L4", "dino_L8"):
        p = pred[key].float()
        t = target[key].float()
        mse = F.smooth_l1_loss(p, t)
        cos = 1.0 - F.cosine_similarity(
            p.flatten(1),
            t.flatten(1),
            dim=-1,
        ).mean()
        loss = loss + mse + cos
    return loss


def distillation_loss_per_branch(pred: dict, target: dict) -> dict:
    """
    Same as distillation_loss but returns a per-branch breakdown for logging.

    Returns:
        dict with keys: 'dino_down', 'dino_L4', 'dino_L8', 'total'.
    """
    breakdown = {}
    total = torch.tensor(0.0, device=next(iter(pred.values())).device)
    for key in ("dino_down", "dino_L4", "dino_L8"):
        p = pred[key].float()
        t = target[key].float()
        mse = F.smooth_l1_loss(p, t)
        cos = 1.0 - F.cosine_similarity(p.flatten(1), t.flatten(1), dim=-1).mean()
        branch_loss = mse + cos
        breakdown[key] = branch_loss.item()
        total = total + branch_loss
    breakdown["total"] = total
    return breakdown
