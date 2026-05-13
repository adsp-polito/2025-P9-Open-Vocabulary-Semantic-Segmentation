"""
Utilities for CLIP multi-layer feature extraction used by the student model.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import List


def _clip_native_res(clip_model) -> int:
    """
    Infer CLIP's expected square input resolution from its positional embedding.
    Works for any ViT variant (ViT-B/16@384 → 384, ViT-L/14@336 → 336, etc.).
    """
    n_pos = clip_model.visual.positional_embedding.shape[0] - 1  # subtract CLS token
    h_grid = int(n_pos ** 0.5)
    patch_size = clip_model.visual.conv1.kernel_size[0]
    return h_grid * patch_size


def extract_clip_layers(clip_model, image: torch.Tensor, layers: List[int]) -> torch.Tensor:
    """
    Extract intermediate patch features from a CLIP ViT for specified transformer block indices.

    Automatically resizes the input image to the CLIP model's native resolution so this
    works for any ViT variant (ViT-B/16, ViT-L/14, etc.) regardless of input size.

    Args:
        clip_model: frozen OpenAI CLIP model.
        image: (B, 3, H, W) pre-normalised input (any spatial size).
        layers: list of resblock indices to hook, e.g. [4, 8, 10, 12].

    Returns:
        (B, len(layers)*C, H_grid, W_grid) stacked spatial features.
    """
    native_res = _clip_native_res(clip_model)
    if image.shape[-2] != native_res or image.shape[-1] != native_res:
        image = F.interpolate(image, size=(native_res, native_res),
                              mode="bilinear", align_corners=False)

    captured = {}
    hooks = []
    for l in layers:
        def make_hook(idx):
            def hook(module, input, output):
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
        feat = captured[l]           # (seq_len, B, C)
        feat = feat[1:, :, :]        # drop CLS token
        n = feat.shape[0]
        H = W = int(n ** 0.5)
        feat = rearrange(feat, "(H W) B C -> B C H W", H=H, W=W)
        feature_maps.append(feat)

    return torch.cat(feature_maps, dim=1)   # (B, len(layers)*C, H_grid, W_grid)


def get_clip_skips(clip_model, image: torch.Tensor, layer_indices: List[int]):
    """
    Return patch feature maps at given resblock indices, for RIPD skip connections.

    Automatically resizes the input image to the CLIP model's native resolution.

    Returns:
        clip_features: dense image features from encode_image
        skips: dict mapping layer_index -> (B, C, H_grid, W_grid)
    """
    native_res = _clip_native_res(clip_model)
    if image.shape[-2] != native_res or image.shape[-1] != native_res:
        image = F.interpolate(image, size=(native_res, native_res),
                              mode="bilinear", align_corners=False)

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
        feat = captured[l][1:, :, :]   # drop CLS token
        n = feat.shape[0]
        H = W = int(n ** 0.5)
        skips[l] = rearrange(feat, "(H W) B C -> B C H W", H=H, W=W)

    return clip_features, skips
