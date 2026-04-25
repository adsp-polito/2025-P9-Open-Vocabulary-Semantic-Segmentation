"""
Utilities for CLIP multi-layer feature extraction used by the student model.

CLIP ViT-B/16 produces patches at 24x24 spatial resolution (384/16=24).
We hook into resblocks[i] for i in [4, 8, 10, 12] to get intermediate features.
Each hook output has shape (seq_len, B, C) = (577, B, 768) in OpenAI CLIP's convention,
where index 0 is CLS and 1: are the 576 patch tokens.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import List


def extract_clip_layers(clip_model, image: torch.Tensor, layers: List[int]) -> torch.Tensor:
    """
    Extract intermediate patch features from CLIP ViT-B/16 for specified transformer block indices.

    Args:
        clip_model: OpenAI CLIP visual model (frozen).
        image: (B, 3, H, W) pre-normalised input, resized to CLIP resolution before this call.
        layers: list of block indices to hook, e.g. [4, 8, 10, 12].

    Returns:
        (B, len(layers)*C, H_grid, W_grid) stacked spatial features,
        where C=768, H_grid=W_grid=24 for ViT-B/16 @ 384px.
    """
    captured = {}

    hooks = []
    for l in layers:
        def make_hook(idx):
            def hook(module, input, output):
                # OpenAI CLIP resblocks return (seq_len, B, C)
                captured[idx] = output
            return hook
        h = clip_model.visual.transformer.resblocks[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        clip_model.encode_image(image, dense=True)

    for h in hooks:
        h.remove()

    feature_maps = []
    for l in layers:
        feat = captured[l]          # (seq_len, B, C) = (577, B, 768)
        feat = feat[1:, :, :]       # drop CLS → (576, B, 768)
        feat = rearrange(feat, "(H W) B C -> B C H W", H=24, W=24)
        feature_maps.append(feat)

    return torch.cat(feature_maps, dim=1)   # (B, len(layers)*768, 24, 24)


def get_clip_skips(clip_model, image: torch.Tensor, layer_indices: List[int]):
    """
    Return raw (seq_len, B, C) outputs at given block indices, for RIPD skip connections.
    Used at inference to pass res4/res5 to RIPD's decoder guidance projections.

    Returns:
        dict mapping layer_index -> (B, 768, 24, 24)
    """
    captured = {}
    hooks = []
    for l in layer_indices:
        def make_hook(idx):
            def hook(module, input, output):
                captured[idx] = output
            return hook
        h = clip_model.visual.transformer.resblocks[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        clip_features = clip_model.encode_image(image, dense=True)

    for h in hooks:
        h.remove()

    skips = {}
    for l in layer_indices:
        feat = captured[l][1:, :, :]    # (576, B, 768)
        skips[l] = rearrange(feat, "(H W) B C -> B C H W", H=24, W=24)

    return clip_features, skips
