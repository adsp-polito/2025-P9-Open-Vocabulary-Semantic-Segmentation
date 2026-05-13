"""
Class-agnostic decoder and spatial projector for open-vocabulary segmentation.

SpatialProjector:
  Compresses 1024-dim L2-normalised TIPS features to 64-dim spatial context.
  Trained jointly in Phase 3. Gives the decoder texture/boundary signal.

ClassAgnosticDecoder:
  Processes each class correlation map independently through shared weights.
  Input: (B*K, 1+proj_dim, 24, 24)  — correlation map + tiled spatial context
  Output: (B*K, 1, 96, 96)

K never appears in any weight tensor — works for any K at inference.

The caller is responsible for:
  - computing spatial_ctx = projector(feats)               (B, 64, H, W)
  - tiling: spatial_ctx.unsqueeze(1).expand(-1,K,-1,-1,-1)
             .reshape(B*K, 64, H, W)
  - reshaping corr (B, K, H, W) → (B*K, 1, H, W)
  - concatenating → (B*K, 65, H, W) before calling forward
  - reshaping (B*K, 1, 96, 96) → (B, K, 96, 96) after calling forward

GroupNorm groups:
  128 / 8 = 16  ✓
   64 / 8 =  8  ✓
   32 / 8 =  4  ✓
    64 / 8 =  8  ✓  (projector)
"""

import torch.nn as nn


class SpatialProjector(nn.Module):
    """Projects L2-normalised TIPS features to a low-dim spatial context map.

    Applied once per image (not per class). The output is tiled K times before
    being concatenated with the per-class correlation maps.

    ~66K params for in_dim=1024, out_dim=64.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_dim, H, W)  L2-normalised TIPS patch features
        Returns:
            (B, out_dim, H, W)
        """
        return self.net(x)


class ClassAgnosticDecoder(nn.Module):
    """Upsamples per-class correlation maps from 24×24 to 96×96.

    Each class map is processed independently through shared conv weights,
    so K (number of classes) never appears in the weight tensors.

    Default in_channels=65 expects the correlation map (1 channel) concatenated
    with the projected spatial context (64 channels) from SpatialProjector.
    """

    def __init__(self, in_channels: int = 65):
        super().__init__()
        self.net = nn.Sequential(
            # 24×24 → 24×24  (feature fusion + projection)
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
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
            x: (B*K, in_channels, 24, 24)
        Returns:
            (B*K, 1, 96, 96)
        """
        return self.net(x)
