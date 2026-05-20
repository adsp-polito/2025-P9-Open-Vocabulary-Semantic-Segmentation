"""
GSDistillStudent — lightweight DINO-stream substitute for GS-Distill.

The student keeps CLIP frozen and predicts text-independent DINO-like tensors
from CLIP intermediate features. RIPD then computes dynamic CLIP/DINO
correlations for whatever vocabulary is passed at inference time.

Architecture:
  - Frozen CLIP ViT-L/14@336px multi-layer features (default [8, 16, 20, 23])
  - Shared conv trunk
  - DINO-down head: predicts the 24x24 DINO tensor used by RIPD correlation
  - DINO-L4 head:   predicts DINOv3 layer-4 features for decoder guidance
  - DINO-L8 head:   predicts DINOv3 layer-8 features for decoder guidance
"""

import torch
import torch.nn as nn
from typing import List

from .utils import extract_clip_layers

CLIP_LAYERS = [8, 16, 20, 23]
TRUNK_OUT = 512


class GSDistillStudent(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        d_dino: int = 768,
        clip_layers: List[int] = None,
    ):
        """
        Args:
            clip_model: frozen OpenAI CLIP visual model.
            d_dino: DINOv3 feature dim after wrapper projection (768).
            clip_layers: CLIP resblock indices to hook.
        """
        super().__init__()

        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.clip_layers = clip_layers if clip_layers is not None else CLIP_LAYERS
        self.d_dino = d_dino

        # Infer CLIP hidden dim from the model (works for ViT-B/16=768, ViT-L/14=1024, etc.)
        clip_dim = clip_model.visual.transformer.resblocks[0].attn.out_proj.out_features
        trunk_in = len(self.clip_layers) * clip_dim

        self.shared_trunk = nn.Sequential(
            nn.Conv2d(trunk_in, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(TRUNK_OUT, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # DINO-down head: predict the 24x24 DINO feature map that RIPD uses
        # for dynamic text-conditioned DINO correlation.
        self.dino_down_branch = nn.Sequential(
            nn.Conv2d(TRUNK_OUT, d_dino, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_dino, d_dino, kernel_size=3, padding=1),
        )

        # DINO-L4 head: predict dino_L4 (B, d_dino, 48, 48).
        self.dino_l4_branch = nn.Sequential(
            nn.ConvTranspose2d(TRUNK_OUT, d_dino, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(d_dino, d_dino, kernel_size=3, padding=1),
        )

        # DINO-L8 head: predict dino_L8 (B, d_dino, 48, 48)
        self.dino_l8_branch = nn.Sequential(
            nn.ConvTranspose2d(TRUNK_OUT, d_dino, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(d_dino, d_dino, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> dict:
        """
        Args:
            image: (B, 3, H, W) CLIP-normalised, already resized to clip_resolution.

        Returns dict with keys:
            'dino_down':        (B, d_dino, 24, 24)
            'dino_L4':          (B, d_dino, 48, 48)
            'dino_L8':          (B, d_dino, 48, 48)
        """
        with torch.no_grad():
            stacked = extract_clip_layers(self.clip_model, image, self.clip_layers)
        return self.forward_from_features(stacked)

    def forward_from_features(self, stacked: torch.Tensor) -> dict:
        """
        Run only the trainable heads given a pre-extracted CLIP feature stack.

        Args:
            stacked: (B, len(clip_layers)*C, H_grid, W_grid) — already extracted
                     by an external get_clip_skips call so CLIP is not run again.

        Returns dict with keys:
            'dino_down':        (B, d_dino, 24, 24)
            'dino_L4':          (B, d_dino, 48, 48)
            'dino_L8':          (B, d_dino, 48, 48)
        """
        trunk_out = self.shared_trunk(stacked)   # (B, 512, 24, 24)

        return {
            "dino_down": self.dino_down_branch(trunk_out),
            "dino_L4": self.dino_l4_branch(trunk_out),
            "dino_L8": self.dino_l8_branch(trunk_out),
        }

    def trainable_parameters(self):
        """Return only the trainable (non-CLIP) parameters."""
        params = (
            list(self.shared_trunk.parameters())
            + list(self.dino_down_branch.parameters())
            + list(self.dino_l4_branch.parameters())
            + list(self.dino_l8_branch.parameters())
        )
        return params
