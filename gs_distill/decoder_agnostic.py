"""
Class-agnostic decoder and projectors for open-vocabulary segmentation.

MidProjector:
  Captures intermediate TIPS block features → 32-dim skip map.
  Applied once per image; tiled K times before decoder injection.
  ~33K params for in_dim=1024, out_dim=32.

SpatialProjector:
  Compresses 1024-dim L2-normalised TIPS features to 64-dim spatial context.
  Applied once per image; tiled K times before concatenation.
  ~66K params for in_dim=1024, out_dim=64.

ClassAgnosticDecoder (v2):
  Upsamples per-class correlation maps from 24×24 to 96×96 with a
  mid-level skip connection injected at the 48×48 stage.
  Input:    (B*K, 65, 24, 24)  — corr (1) + spatial ctx (64)
  mid_skip: (B*K, 32, 24, 24)  — upsampled to 48×48 internally
  Output:   (B*K, 1,  96, 96)

K never appears in any weight tensor — works for any K at inference.

The caller is responsible for:
  - computing spatial_ctx  = projector(feats)           (B, 64, 24, 24)
  - computing mid_skip_raw = mid_projector(mid_feats)   (B, 32, 24, 24)
  - tiling both unsqueeze(1).expand(-1,K,...).reshape(B*K, C, H, W)
  - reshaping corr (B, K, H, W) → (B*K, 1, H, W)
  - concatenating corr_flat + spatial_tiled → (B*K, 65, H, W)
  - calling decoder(dec_input, mid_skip=mid_skip_tiled)
  - reshaping (B*K, 1, 96, 96) → (B, K, 96, 96)

GroupNorm groups:
  256 / 8 = 32  ✓  (stage1)
  128 / 8 = 16  ✓  (up1, skip_fuse, refine48)
   64 / 8 =  8  ✓  (up2)
   64 / 8 =  8  ✓  (SpatialProjector)
   32 / 8 =  4  ✓  (MidProjector)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MidProjector(nn.Module):
    """Projects intermediate TIPS block features to a compact skip map.

    Hooked on `tips.vision_encoder.blocks[unfreeze_from - 1]`, capturing
    mid-level texture/boundary signal before the final transformer blocks.
    Output is injected at 48×48 in ClassAgnosticDecoder.

    ~33K params for in_dim=1024, out_dim=32.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_dim, H, W)  intermediate TIPS block patch features
        Returns:
            (B, out_dim, H, W)
        """
        return self.net(x)


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

    v2: Accepts optional mid_skip (B*K, 32, 24, 24) which is bilinearly
    upsampled to 48×48 and fused via cat+conv after the first upsample stage.

    Each class map is processed independently through shared conv weights,
    so K (number of classes) never appears in the weight tensors.

    Default in_channels=65 expects the correlation map (1 channel) concatenated
    with the projected spatial context (64 channels) from SpatialProjector.
    """

    def __init__(self, in_channels: int = 65, mid_channels: int = 32):
        super().__init__()
        # 24×24 → 24×24  (feature fusion)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )
        # 24×24 → 48×48
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        # Mid-level skip fusion at 48×48: (128 + mid_channels) → 128
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(128 + mid_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        # 48×48 spatial refinement
        self.refine48 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        # 48×48 → 96×96
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, mid_skip=None):
        """
        Args:
            x:        (B*K, in_channels, 24, 24)
            mid_skip: (B*K, mid_channels, 24, 24) optional mid-level skip features
        Returns:
            (B*K, 1, 96, 96)
        """
        x = self.stage1(x)   # (B*K, 256, 24, 24)
        x = self.up1(x)      # (B*K, 128, 48, 48)

        if mid_skip is not None:
            mid_up = F.interpolate(
                mid_skip, size=x.shape[-2:], mode="bilinear", align_corners=False
            )                # (B*K, 32, 48, 48)
            x = self.skip_fuse(torch.cat([x, mid_up], dim=1))  # (B*K, 128, 48, 48)

        x = self.refine48(x)  # (B*K, 128, 48, 48)
        x = self.up2(x)       # (B*K,  64, 96, 96)
        return self.out(x)    # (B*K,   1, 96, 96)
