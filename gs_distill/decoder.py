"""Lightweight segmentation decoder for the TIPS distillation student."""

import torch.nn as nn


class LightweightDecoder(nn.Module):
    """5-layer conv decoder: (B, T, 24, 24) → (B, T, 96, 96).

    Two 2× ConvTranspose upsampling stages with intermediate conv refinement.
    No skip connections — trained entirely from teacher soft-label supervision.

    Architecture:
        Conv2d(T → 128, 3×3)          (B, 128, 24, 24)
        ConvTranspose2d(128 → 64, 2×2) (B,  64, 48, 48)
        Conv2d(64 → 64, 3×3)           (B,  64, 48, 48)
        ConvTranspose2d(64 → 32, 2×2)  (B,  32, 96, 96)
        Conv2d(32 → T, 3×3)            (B,   T, 96, 96)
    """

    def __init__(self, num_classes: int = 40, hidden: int = 128):
        super().__init__()
        half = hidden // 2
        quarter = hidden // 4

        self.net = nn.Sequential(
            nn.Conv2d(num_classes, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, half, kernel_size=2, stride=2),
            nn.GroupNorm(8, half),
            nn.GELU(),
            nn.Conv2d(half, half, 3, padding=1),
            nn.GroupNorm(8, half),
            nn.GELU(),
            nn.ConvTranspose2d(half, quarter, kernel_size=2, stride=2),
            nn.GroupNorm(4, quarter),
            nn.GELU(),
            nn.Conv2d(quarter, num_classes, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
