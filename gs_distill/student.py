"""
TIPSDistillStudent — Phase 2 student model for GS-Distill.

Architecture (inspired by CLIP-DINOiser, ECCV 2024):
  ① Frozen TIPSv2-L/14 vision encoder → patch_tokens (B, 576, 1024)
  ② reshape → (B, 1024, 24, 24)
  ③ SpatialAdapter [trainable, ~0.5M params]
       bottleneck residual: 1×1 compress → 3×3 spatial mix → 1×1 expand + residual
       learns to inject DINOv3-like spatial structure into TIPS patch features
  ④ L2-normalise (channel dim)
  ⑤ einsum with frozen text_embeds (T, 1024) → correlation map (B, T, 24, 24)
  ⑥ LightweightDecoder [trainable, ~0.5M params] → logits (B, T, 96, 96)

Total trainable in Phase 2: ~1M params (adapter + decoder).
TIPS backbone (488M) is fully frozen.

What carries to Phase 3:
  - SpatialAdapter: preserved (carries the distilled DINOv3-like knowledge)
  - Decoder: reinitialized for K dataset classes
  - TIPS last 4 blocks: unfrozen and fine-tuned

At 336px input with patch_size=14: 336/14 = 24 patches per side → 24×24 grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .decoder import LightweightDecoder


# ── TIPS loader ───────────────────────────────────────────────────────────────

def _load_tips(tips_dir: str):
    """Load TIPSv2Model from a local directory, bypassing AutoModel.from_pretrained.

    AutoModel.from_pretrained crashes on safetensors files with no metadata header
    (transformers 4.41.x bug). We build a fake package namespace so that the relative
    imports inside modeling_tips.py resolve correctly, then load weights directly.
    """
    import sys, json, importlib.util, importlib.machinery
    from pathlib import Path
    from safetensors.torch import load_file

    tips_path = Path(tips_dir).resolve()
    pkg = "tips_local"

    if pkg not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
        pkg_mod  = importlib.util.module_from_spec(pkg_spec)
        pkg_mod.__path__    = [str(tips_path)]
        pkg_mod.__package__ = pkg
        sys.modules[pkg]    = pkg_mod

    for name in ("configuration_tips", "image_encoder", "text_encoder", "modeling_tips"):
        full_name = f"{pkg}.{name}"
        if full_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                full_name, str(tips_path / f"{name}.py")
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg
            sys.modules[full_name] = mod
            spec.loader.exec_module(mod)

    TIPSv2Config = sys.modules[f"{pkg}.configuration_tips"].TIPSv2Config
    TIPSv2Model  = sys.modules[f"{pkg}.modeling_tips"].TIPSv2Model

    with open(tips_path / "config.json") as f:
        cfg_dict = json.load(f)
    for key in ("model_type", "architectures", "auto_map"):
        cfg_dict.pop(key, None)
    cfg_dict["_name_or_path"] = str(tips_path)

    config = TIPSv2Config(**cfg_dict)
    model  = TIPSv2Model(config)
    model.load_state_dict(load_file(str(tips_path / "model.safetensors")), strict=True)
    return model


# ── LD50K class list ──────────────────────────────────────────────────────────

LD50K_CLASSES = [
    "background", "bare land", "grass", "pavement", "road",
    "tree", "water", "agriculture land", "buildings", "forest land",
    "barren land", "urban land", "large-vehicle", "swimming-pool", "helicopter",
    "bridge", "plane", "ship", "soccer-ball-field", "basketball-court",
    "ground-track-field", "small-vehicle", "baseball-diamond", "tennis-court", "roundabout",
    "storage-tank", "harbor", "container-crane", "airport", "helipad",
    "chimney", "expressway service area", "expresswalltoll station", "dam", "golf field",
    "overpass", "stadium", "train station", "vehicle", "windmill",
]


# ── Spatial Adapter ───────────────────────────────────────────────────────────

class SpatialAdapter(nn.Module):
    """Bottleneck residual adapter on top of TIPS patch features.

    Learns to inject DINOv3-like spatial structure (local patch correspondence,
    fine-grained boundaries) into TIPS features, inspired by CLIP-DINOiser
    (Wysoczańska et al., ECCV 2024).

    Architecture: 1×1 compress → 3×3 spatial mix → 1×1 expand + residual.
    The residual ensures training starts from unchanged TIPS features and
    the adapter can only improve spatial quality, never hurt text alignment.

    ~0.5M params for embed_dim=1024, bottleneck=256.
    """

    def __init__(self, embed_dim: int = 1024, bottleneck: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, bottleneck, kernel_size=1),
            nn.GroupNorm(8, bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.GroupNorm(8, bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, embed_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── Student model ─────────────────────────────────────────────────────────────

class TIPSDistillStudent(nn.Module):
    """
    Args:
        tips_dir:       path to TIPSv2-L/14 model directory.
        num_classes:    vocabulary size (40 for LD50K).
        hidden:         LightweightDecoder hidden channels (128).
        bottleneck:     SpatialAdapter bottleneck channels (256).
        class_texts:    class name strings; defaults to LD50K_CLASSES if None.
        unfreeze_blocks: number of TIPS transformer blocks to unfreeze from the end
                        (0 = fully frozen as in v1; 4 = v2 Phase 2; 8 = v2 Phase 3).
    """

    def __init__(
        self,
        tips_dir: str,
        num_classes: int = 40,
        hidden: int = 128,
        bottleneck: int = 256,
        class_texts: list = None,
        unfreeze_blocks: int = 0,
    ):
        super().__init__()

        tips = _load_tips(tips_dir)
        tips.eval()
        for p in tips.parameters():
            p.requires_grad = False

        if unfreeze_blocks > 0:
            total = len(tips.vision_encoder.blocks)
            for i in range(total - unfreeze_blocks, total):
                for p in tips.vision_encoder.blocks[i].parameters():
                    p.requires_grad = True

        self.tips = tips
        self.unfreeze_blocks = unfreeze_blocks

        self.adapter = SpatialAdapter(embed_dim=1024, bottleneck=bottleneck)
        self.decoder = LightweightDecoder(num_classes=num_classes, hidden=hidden)

        texts = class_texts if class_texts is not None else LD50K_CLASSES[:num_classes]
        with torch.no_grad():
            embeds = tips.encode_text(texts)       # (T, 1024)
            embeds = F.normalize(embeds, dim=-1)
        self.register_buffer("text_embeds", embeds)  # (T, 1024) frozen

    def forward(self, image: torch.Tensor, return_feats: bool = False):
        """
        Args:
            image:       (B, 3, H, W) in [0, 1], resized to 336×336.
            return_feats: when True, also return the L2-normalised adapter output
                         (B, 1024, 24, 24) for feature-level distillation loss.

        Returns:
            logits (B, T, 96, 96) — always.
            feats  (B, 1024, 24, 24) — only when return_feats=True.
        """
        if self.unfreeze_blocks > 0:
            _, _, patch_tokens = self.tips.vision_encoder(image)
        else:
            with torch.no_grad():
                _, _, patch_tokens = self.tips.vision_encoder(image)

        x = rearrange(patch_tokens, "B (H W) D -> B D H W", H=24, W=24)
        x = self.adapter(x)
        x = F.normalize(x, dim=1)

        corr = torch.einsum("b d h w, t d -> b t h w", x, self.text_embeds)
        logits = self.decoder(corr)

        if return_feats:
            return logits, x
        return logits

    def trainable_parameters(self):
        """Adapter + decoder — TIPS backbone excluded (even if some blocks are unfrozen)."""
        return list(self.adapter.parameters()) + list(self.decoder.parameters())

    def backbone_parameters(self):
        """Unfrozen TIPS block parameters — for a separate lower-LR optimizer group."""
        return [p for p in self.tips.parameters() if p.requires_grad]
