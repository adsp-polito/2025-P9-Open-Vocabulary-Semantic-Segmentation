"""
TIPSDistillStudent — the Phase 2 student model for GS-Distill.

Architecture:
  - Frozen TIPSv2-L/14 image encoder (ViT-L, patch_size=14)
  - Spatial correlation map: patch_tokens (B,576,1024) @ text_embeds (T,1024)
    → (B, T, 24, 24) cosine similarity map
  - LightweightDecoder: (B, T, 24, 24) → (B, T, 96, 96) segmentation logits

At 336px input with patch_size=14: 336/14 = 24 patches per side → 24×24 grid.
Text embeddings for 40 LD50K classes are pre-computed once and frozen as a buffer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .decoder import LightweightDecoder

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

    # Create a fake package pointing at tips_path so relative imports work
    if pkg not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        pkg_mod.__path__ = [str(tips_path)]
        pkg_mod.__package__ = pkg
        sys.modules[pkg] = pkg_mod

    # Load each sibling module under the fake package (order matters for imports)
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

    state_dict = load_file(str(tips_path / "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    return model


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


class TIPSDistillStudent(nn.Module):
    """
    Args:
        tips_dir:    path to TIPSv2-L/14 model directory (loaded via AutoModel).
        num_classes: number of vocabulary classes (40 for LD50K).
        hidden:      LightweightDecoder hidden channels (128).
        class_texts: list of class name strings; defaults to LD50K_CLASSES if None.
    """

    def __init__(
        self,
        tips_dir: str,
        num_classes: int = 40,
        hidden: int = 128,
        class_texts: list = None,
    ):
        super().__init__()

        tips = _load_tips(tips_dir)
        tips.eval()
        for p in tips.parameters():
            p.requires_grad = False
        self.tips = tips

        self.decoder = LightweightDecoder(num_classes=num_classes, hidden=hidden)

        texts = class_texts if class_texts is not None else LD50K_CLASSES[:num_classes]
        with torch.no_grad():
            embeds = tips.encode_text(texts)  # (T, 1024)
            embeds = F.normalize(embeds, dim=-1)
        self.register_buffer("text_embeds", embeds)  # (T, D) frozen

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:  (B, 3, H, W) in [0, 1] range, resized to 336×336.

        Returns:
            logits: (B, T, 96, 96) segmentation logits.
        """
        with torch.no_grad():
            _, _, patch_tokens = self.tips.vision_encoder(image)
            # patch_tokens: (B, N, D) = (B, 576, 1024)

        patch_feats = rearrange(patch_tokens, "B (H W) D -> B D H W", H=24, W=24)
        patch_feats = F.normalize(patch_feats, dim=1)  # L2-norm along channel dim

        # Cosine similarity: (B, D, H, W) × (T, D) → (B, T, H, W)
        corr = torch.einsum("b d h w, t d -> b t h w", patch_feats, self.text_embeds)

        return self.decoder(corr)  # (B, T, 96, 96)

    def trainable_parameters(self):
        return list(self.decoder.parameters())
