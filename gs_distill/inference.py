"""
GS-Distill Phase 2 inference — TIPSDistillStudent forward pass.
"""

import torch


def tips_distill_inference(image: torch.Tensor, student) -> torch.Tensor:
    """
    Run student inference on a batch of images.

    Args:
        image:   (B, 3, H, W) in [0, 1] range, resized to 336×336.
        student: TIPSDistillStudent in eval mode.

    Returns:
        logits: (B, T, 96, 96)
    """
    with torch.no_grad():
        return student(image)
