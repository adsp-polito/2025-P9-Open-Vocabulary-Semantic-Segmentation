"""
Class-agnostic decoder for open-vocabulary segmentation.

Processes each class correlation map independently through shared weights.
K is never a fixed dimension in any weight tensor — works for any K at inference.

Input  (to forward): (B*K, 1, 24, 24)   — one scalar correlation map per class
Output (from forward): (B*K, 1, 96, 96)  — upsampled, one map per class

The caller is responsible for:
  - reshaping (B, K, 24, 24) → (B*K, 1, 24, 24) before calling forward
  - reshaping (B*K, 1, 96, 96) → (B, K, 96, 96) after calling forward

GroupNorm groups chosen so all channel counts are divisible:
  128 / 8 = 16  ✓
   64 / 8 =  8  ✓
   32 / 8 =  4  ✓
"""

import torch.nn as nn


class ClassAgnosticDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 24×24 → 24×24  (feature projection)
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # 24×24 → 48×48
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # 48×48 → 48×48  (spatial refinement)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # 48×48 → 96×96
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            # 96×96 → 96×96  (output)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B*K, 1, 24, 24)
        Returns:
            (B*K, 1, 96, 96)
        """
        return self.net(x)
