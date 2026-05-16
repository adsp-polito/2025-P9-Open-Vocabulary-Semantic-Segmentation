"""
GSDistillStudent — the lightweight student model for GS-Distill.

Architecture:
  - Frozen CLIP ViT-B/16  (multi-layer features from resblocks [4, 8, 10, 12])
  - Shared conv trunk     (4*768 → 512 channels, 24x24 spatial)
  - Fusion head:          predicts fused_corr_embed  (B, hidden_dim, T, 24, 24)
                          — CLIP+DINO fused correlation entering RIPD aggregators
  - CLIP head:            predicts clip_embed_corr   (B, hidden_dim, T, 24, 24)
                          — CLIP-only correlation embedding before DINO fusion
  - DINO-L4 head:         predicts DINOv3 layer-4 features (B, 768, 48, 48)
  - DINO-L8 head:         predicts DINOv3 layer-8 features (B, 768, 48, 48)

Key dimensions (from vitb_384.yaml / RIPD defaults):
  hidden_dim  = 128   (HIDDEN_DIMS)
  D_dino      = 768   (DINOv3Wrapper output after projection)
  T           = 40    (number of LD50K classes)
  spatial     = 24x24 (CLIP patch grid for ViT-B/16 @ 384px)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List

from .utils import extract_clip_layers

CLIP_LAYERS = [4, 8, 10, 12]
TRUNK_OUT = 512


class GSDistillStudent(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        hidden_dim: int = 128,
        d_dino: int = 768,
        num_classes: int = 40,
        clip_layers: List[int] = None,
    ):
        """
        Args:
            clip_model: frozen OpenAI CLIP visual model.
            hidden_dim: matches RIPD HIDDEN_DIMS (128 in vitb_384.yaml).
            d_dino: DINOv3 feature dim after wrapper projection (768).
            num_classes: number of vocabulary classes (40 for LD50K).
            clip_layers: which CLIP resblock indices to hook (default [4,8,10,12]).
        """
        super().__init__()

        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.clip_layers = clip_layers if clip_layers is not None else CLIP_LAYERS
        self.hidden_dim = hidden_dim
        self.d_dino = d_dino
        self.num_classes = num_classes

        # Infer CLIP hidden dim from the model (works for ViT-B/16=768, ViT-L/14=1024, etc.)
        clip_dim = clip_model.visual.transformer.resblocks[0].attn.out_proj.out_features
        trunk_in = len(self.clip_layers) * clip_dim

        self.shared_trunk = nn.Sequential(
            nn.Conv2d(trunk_in, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(TRUNK_OUT, TRUNK_OUT, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Fusion head: predict fused_corr_embed (B, hidden_dim, T, 24, 24)
        # Predicts hidden_dim * num_classes channels at 24x24, then reshape.
        self.fusion_branch = nn.Conv2d(
            TRUNK_OUT, hidden_dim * num_classes, kernel_size=3, padding=1
        )

        # CLIP head: predict clip_embed_corr (B, hidden_dim, T, 24, 24)
        # Same shape as fused_corr_embed but targets the CLIP-only signal before DINO fusion.
        self.clip_embed_branch = nn.Conv2d(
            TRUNK_OUT, hidden_dim * num_classes, kernel_size=3, padding=1
        )

        # DINO-L4 head: predict dino_L4 (B, d_dino, 48, 48) — 2x upsample from 24->48
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
            'fused_corr_embed': (B, hidden_dim, T, 24, 24)
            'dino_L4':          (B, d_dino, 48, 48)
            'dino_L8':          (B, d_dino, 48, 48)
        """
        B = image.shape[0]

        # Frozen CLIP multi-layer extraction → (B, trunk_in, 24, 24)
        with torch.no_grad():
            stacked = extract_clip_layers(self.clip_model, image, self.clip_layers)

        trunk_out = self.shared_trunk(stacked)   # (B, 512, 24, 24)

        def _corr_reshape(x):
            return rearrange(x, "B (C T) H W -> B C T H W", C=self.hidden_dim, T=self.num_classes)

        return {
            "fused_corr_embed": _corr_reshape(self.fusion_branch(trunk_out)),
            "clip_embed_corr":  _corr_reshape(self.clip_embed_branch(trunk_out)) if self.clip_embed_branch is not None else None,
            "dino_L4": self.dino_l4_branch(trunk_out),
            "dino_L8": self.dino_l8_branch(trunk_out),
        }

    def trainable_parameters(self):
        """Return only the trainable (non-CLIP) parameters."""
        return (
            list(self.shared_trunk.parameters())
            + list(self.fusion_branch.parameters())
            + list(self.clip_embed_branch.parameters())
            + list(self.dino_l4_branch.parameters())
            + list(self.dino_l8_branch.parameters())
        )
