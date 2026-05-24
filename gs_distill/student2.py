"""
Student-2 model for RIPD-free GS-Distill inference.

Student-2 keeps the frozen CLIP encoder and Student-1 DINO-substitute
branches, then replaces RIPD/QGFF with S2CostHead.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .cost_head import S2CostHead
from .student import CLIP_LAYERS, GSDistillStudent
from .utils import get_clip_skips


class GSDistillStudent2(GSDistillStudent):
    """
    Student-2: frozen CLIP + Student-1 DINO-substitute layers + S2CostHead.
    No RIPD, no QGFF, no DINOv3 at inference.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        d_dino: int = 768,
        clip_layers: Optional[List[int]] = None,
        head_kwargs: Optional[dict] = None,
        head_mode: str = "aggregator",
    ):
        super().__init__(
            clip_model=clip_model,
            d_dino=d_dino,
            clip_layers=clip_layers if clip_layers is not None else CLIP_LAYERS,
        )
        head_kwargs = dict(head_kwargs or {})

        clip_width = clip_model.visual.transformer.resblocks[0].attn.out_proj.out_features
        clip_dense_dim = int(getattr(clip_model.visual, "output_dim", clip_width))
        text_projection = getattr(clip_model, "text_projection", None)
        clip_text_dim = int(text_projection.shape[1]) if text_projection is not None else 512
        corr_dim = int(head_kwargs.pop("corr_dim", 256))

        self.corr_dim = corr_dim
        self.clip_cost_proj = nn.Conv2d(clip_dense_dim, corr_dim, kernel_size=1)
        self.dino_cost_proj = nn.Conv2d(d_dino, corr_dim, kernel_size=1)
        self.text_cost_proj = nn.Linear(clip_text_dim, corr_dim)

        self.cost_head = S2CostHead(
            clip_feat_dim=clip_dense_dim,
            clip_skip_dim=clip_width,
            clip_text_dim=clip_text_dim,
            dino_feat_dim=d_dino,
            dino_skip_dim=d_dino,
            head_mode=head_mode,
            **head_kwargs,
        )

        self.last_costs = None
        self.last_corr_embed = None

    def trainable_parameters(self):
        """Return all non-CLIP trainable Student-2 parameters."""
        params = (
            list(self.shared_trunk.parameters())
            + list(self.dino_down_branch.parameters())
            + list(self.dino_l4_branch.parameters())
            + list(self.dino_l8_branch.parameters())
            + list(self.clip_cost_proj.parameters())
            + list(self.dino_cost_proj.parameters())
            + list(self.text_cost_proj.parameters())
            + list(self.cost_head.parameters())
        )
        return params

    def set_student_backbone_trainable(self, trainable: bool):
        """Freeze or unfreeze the Student-1 trunk and substitute branches."""
        modules = [
            self.shared_trunk,
            self.dino_down_branch,
            self.dino_l4_branch,
            self.dino_l8_branch,
        ]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = trainable

    def _text_mean(self, text_feats: torch.Tensor, batch_size: int) -> torch.Tensor:
        if text_feats.dim() == 4:
            text = text_feats.mean(dim=2)
        elif text_feats.dim() == 3:
            text = text_feats
        else:
            raise ValueError(f"text_feats must be (B,T,P,C) or (B,T,C), got {text_feats.shape}")
        text = F.normalize(text, dim=-1)
        if text.shape[0] == 1 and batch_size > 1:
            text = text.expand(batch_size, -1, -1)
        if text.shape[0] != batch_size:
            raise ValueError(f"text batch {text.shape[0]} does not match image batch {batch_size}")
        return text

    def _cost_volumes(self, clip_res3, dino_down, text_mean):
        clip_feat = self.clip_cost_proj(clip_res3)
        dino_feat = self.dino_cost_proj(dino_down)
        if dino_feat.shape[-2:] != clip_feat.shape[-2:]:
            dino_feat = F.interpolate(
                dino_feat, size=clip_feat.shape[-2:], mode="bilinear", align_corners=False
            )

        text_feat = self.text_cost_proj(text_mean)
        clip_feat = F.normalize(clip_feat, dim=1)
        dino_feat = F.normalize(dino_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=-1)

        C_clip = torch.einsum("bchw,btc->bthw", clip_feat, text_feat)
        C_dino = torch.einsum("bchw,btc->bthw", dino_feat, text_feat)
        return C_clip, C_dino

    def forward(
        self,
        image: torch.Tensor,
        text_feats: torch.Tensor,
        clip_skip_indices: Tuple[int, int] = (7, 15),
        clip_features: Optional[torch.Tensor] = None,
        clip_skips: Optional[dict] = None,
        student_clip_stacked: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ):
        """
        Args:
            image: (B, 3, H, W) CLIP-normalised image.
            text_feats: (B, T, P, C) or (B, T, C) CLIP text features.
            clip_skip_indices: CLIP layers for decoder skip guidance.
            clip_features/clip_skips/student_clip_stacked: optional cached CLIP outputs.
            return_intermediate: return a dict with costs and features for distillation.
        """
        needed_layers = sorted(set(list(clip_skip_indices) + list(self.clip_layers)))
        if (
            clip_features is None
            or clip_skips is None
            or any(layer not in clip_skips for layer in needed_layers)
        ):
            with torch.no_grad():
                clip_features, clip_skips = get_clip_skips(
                    self.clip_model, image, needed_layers
                )

        if student_clip_stacked is None:
            student_clip_stacked = torch.cat(
                [clip_skips[layer] for layer in self.clip_layers], dim=1
            )

        student_out = self.forward_from_features(student_clip_stacked)
        dino_down = student_out["dino_down"]
        dino_L4 = student_out["dino_L4"]
        dino_L8 = student_out["dino_L8"]

        n_tokens = clip_features.shape[1] - 1
        H = W = int(n_tokens**0.5)
        clip_res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=H, W=W)

        B = image.shape[0]
        text_mean = self._text_mean(text_feats, B)
        C_clip, C_dino = self._cost_volumes(clip_res3, dino_down, text_mean)
        C_fused = self.cost_head.fuse_costs(C_clip, C_dino)

        l0, _ = clip_skip_indices
        logit = self.cost_head(
            C_fused=C_fused,
            E_Q=text_mean,
            F_G_res3=clip_res3,
            F_G_res4=clip_skips[l0],
            dino_L4=dino_L4,
        )
        self.last_corr_embed = self.cost_head.last_corr_embed
        self.last_costs = {
            "C_clip": C_clip,
            "C_dino": C_dino,
            "C_fused": C_fused,
        }

        if not return_intermediate:
            return logit

        return {
            "logit": logit,
            "C_clip": C_clip,
            "C_dino": C_dino,
            "C_fused": C_fused,
            "corr_embed": self.last_corr_embed,
            "clip_res3": clip_res3,
            "dino_down": dino_down,
            "dino_L4": dino_L4,
            "dino_L8": dino_L8,
        }
