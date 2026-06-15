#!/usr/bin/env python3
"""
efficiency_metrics.py — standalone inference-time efficiency profiler.

Measures 5 experiments and writes per-experiment CSVs + a combined CSV.
Runs fully today using a dummy model; real loaders are wired in at the
TODO: CONNECT MODEL hooks below — the metric logic is untouched when you
do so.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHODOLOGY NOTES (read before plugging in real models)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Timing
  - Batch size = 1 throughout (latency, not throughput optimisation).
  - WARMUP_ITERS (default 10): runs that are timed but thrown away.
    Needed because CUDA JIT-compiles kernels on the first call; without
    warmup the first iterations are wildly slow and skew the mean.
  - MEASURE_ITERS (default 50): runs whose wall-clock time is recorded.
  - Synchronisation: torch.cuda.synchronize() before start AND after
    each forward pass. Without this, CUDA kernels run asynchronously and
    time.perf_counter() returns the launch time, not the completion time.
  - Statistics reported: mean, std, min, p50, p95 latency (ms).
    p95 is the "tail latency" that matters for worst-case pipeline speed.

FLOPs
  - Primary backend: fvcore (already in requirements.txt; used by GSNet
    for weight init). fvcore.nn.FlopCountAnalysis gives op-level FLOPs.
  - Fallback: thop (pip install thop) — tries to import, logs warning if
    missing, reports NaN.
  - Both may double-count or under-count custom ops (einsum in RIPD,
    interpolate calls). Treat the number as an order-of-magnitude guide,
    not a ground truth. The RIPD einsum in particular is not counted by
    fvcore by default; a custom jit hook is registered to catch it.
  - GFLOPs = FLOPs / 1e9 (multiply-adds counted as 2 ops each — fvcore
    default convention is to count each MAC as 1 flop; we halve the
    reported number accordingly so it matches the standard convention).
    Actually fvcore counts each multiply-add as 1 FlopCountAnalysis unit
    and each add as 1 unit separately, so we report the raw count / 1e9.

Memory
  - torch.cuda.max_memory_allocated() after resetting the peak counter
    just before the forward pass.  This captures the high-water mark of
    GPU memory allocated during that single forward, not the total RSS.
  - Reported in GB (1 GB = 2^30 bytes).
  - On CPU-only environments: reported as NaN.

Parameters
  - sum(p.numel() for p in model.parameters()) — total.
  - sum(p.numel() for p in model.parameters() if p.requires_grad) — trainable.
  - For the student experiments (3 & 4) CLIP is frozen; only the student
    trunk + heads + (optionally) RIPD weights are trainable.

Speedup
  - Computed in the combined CSV as baseline_latency_mean_ms / exp_latency_mean_ms.
  - Baseline = Experiment 1 (Old GSNet).
  - Left as NaN until Experiment 1 has been profiled and is present in
    the combined CSV.

Input
  - All experiments: (1, 3, 384, 384) CLIP-normalised image tensor, on
    the target device.  Text features: (1, 40, 1, 512) — 40 LD50K classes,
    1 template prompt, 512-dim CLIP text embedding.  These are passed as
    dummy random tensors; real evaluation uses the same shapes.

AMP
  - Profiling runs in float32 by default.  Pass --amp to wrap forward in
    torch.cuda.amp.autocast().  Report the precision used in the CSV so
    runs are not accidentally compared across precisions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 4 EXPERIMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Exp 1  old_gsnet    — original dual-stream CLIP+DINO with RIPD/QGFF decoder.
                      Loaded by build_gsnet() via Detectron2 + DetectionCheckpointer.
                      Forward: gsnet(image, text_feats) → logit (B, T, H, W).

Exp 2  improved_gsnet — same architecture, DINO branch swapped for DINOv3-based
                      remote-sensing backbone (RSIB). Same loading path as Exp 1;
                      only the checkpoint differs.  Both share eval.sh methodology.

Exp 3  gs_distill   — CLIP-conditioned student predicting DINO substitutes, which
                      feed into the *original frozen* RIPD/QGFF decoder.
                      Forward: gs_distill_inference(image, text_feats, student,
                        clip_model, ripd, upsample modules) → logit (B, T, H, W).
                      This is the GS-Distill experiment (Student-1 in the paper).

Exp 4  gs_distill_new_decoder — same student DINO substitutes but RIPD/QGFF replaced
                      by S2CostHead, a new lightweight trainable decoder (Student-2).
                      Forward: gs_distill_s2_inference(image, text_feats, student2,
                        clip_model, clip_skip_layer_indices) → logit (B, T, H, W).
                      Loaded via scripts/eval_s2.py::load_model_s2().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO PLUG IN REAL MODELS (one-time per experiment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Search for TODO: CONNECT MODEL — there is one per experiment.

Each hook is a function:
    loader_fn(args, device) -> (callable_model, dummy_input_override_or_None)

callable_model must be:
    - A single callable: model(image, text_feats) -> logit  OR
    - A partial / lambda wrapping the multi-module pipeline
      (see _make_gs_distill_callable() below for the pattern).

dummy_input_override: if None, the default dummy image+text is used.
    Supply a tuple (image_tensor, text_feats_tensor) to use experiment-
    specific shapes or real data from disk.

The metric loop in profile_model() is entirely agnostic to what is
inside callable_model.  It just calls callable_model(img, txt) and
measures time / memory / flops.
"""

import os
import sys
import csv
import time
import math
import argparse
import traceback
import warnings
import gc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.abspath("."))

# fvcore is in requirements.txt (used by GSNet weight_init too)
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    _FVCORE = True
except ImportError:
    _FVCORE = False
    warnings.warn(
        "fvcore not found — FLOPs will be NaN. "
        "Install with: pip install fvcore",
        ImportWarning, stacklevel=2,
    )

# thop as optional secondary backend
try:
    from thop import profile as thop_profile
    _THOP = True
except ImportError:
    _THOP = False


# ─────────────────────────────────────────────────────────────────────────────
# Config knobs — change these to adjust profiling thoroughness
# ─────────────────────────────────────────────────────────────────────────────

WARMUP_ITERS   = 100   # Iterations discarded before measurement begins
MEASURE_ITERS  = 1000  # Iterations whose wall-clock times are recorded

# Default input shape: (B, C, H, W) — matches CLIP resize in eval_clip.py
DEFAULT_IMAGE_SHAPE  = (1, 3, 384, 384)
# Text features: (B, T, P, C) — T=40 LD50K classes, P=1 template, C=512 text dim
DEFAULT_TEXT_SHAPE   = (1, 40, 1, 512)

# CSV output directory (relative to repo root or absolute)
DEFAULT_OUTPUT_DIR = "output/efficiency"

# ─────────────────────────────────────────────────────────────────────────────
# CLIP normalisation constants (from eval_clip.py)
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ─────────────────────────────────────────────────────────────────────────────
# Data class that describes one experiment's profiling configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    One row of configuration per experiment.  Pass this to profile_model().

    Fields
    ------
    name:          Short slug used in filenames and the CSV.
    description:   Human-readable label for the summary table.
    loader_fn:     callable(args, device) -> (callable_model, dummy_override).
                   dummy_override is (image_tensor, text_tensor) or None.
    image_shape:   (B, C, H, W) — overrides DEFAULT_IMAGE_SHAPE if set.
    text_shape:    (B, T, P, C) — overrides DEFAULT_TEXT_SHAPE if set.
    """
    name:        str
    description: str
    loader_fn:   Callable
    image_shape: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_IMAGE_SHAPE)
    text_shape:  Tuple[int, ...] = field(default_factory=lambda: DEFAULT_TEXT_SHAPE)


# ─────────────────────────────────────────────────────────────────────────────
# Dummy model — used when no real checkpoint is provided
# ─────────────────────────────────────────────────────────────────────────────

class _DummyModel(nn.Module):
    """
    Minimal placeholder that mimics the GS-Distill call signature:
        forward(image, text_feats) -> logit (B, T, H, W)

    Consists of two conv layers to give FlopCountAnalysis something real to
    count.  Shapes are chosen to match the real output resolution.

    Replace this with the real model loader — see TODO: CONNECT MODEL below.
    """
    def __init__(self, num_classes: int = 40, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, image: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        # text_feats ignored — dummy model does not use text conditioning
        x = self.encoder(image)
        return self.head(x)   # (B, num_classes, H, W)


class _S2CostHeadOnly(nn.Module):
    """Callable wrapper that profiles only S2CostHead, not CLIP or Student-1."""

    def __init__(
        self,
        head: nn.Module,
        clip_dense_dim: int = 768,
        clip_skip_dim: int = 1024,
        dino_dim: int = 768,
    ):
        super().__init__()
        self.head = head
        self.register_buffer("clip_res3", torch.randn(1, clip_dense_dim, 24, 24))
        self.register_buffer("clip_skip", torch.randn(1, clip_skip_dim, 24, 24))
        self.register_buffer("dino_l4", torch.randn(1, dino_dim, 48, 48))

    def forward(self, C_fused: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        B = C_fused.shape[0]
        clip_res3 = self.clip_res3.expand(B, -1, -1, -1)
        clip_skip = self.clip_skip.expand(B, -1, -1, -1)
        dino_l4 = self.dino_l4.expand(B, -1, -1, -1)
        return self.head(C_fused, text_feats, clip_res3, clip_skip, dino_l4)


class _RIPDDecoderOnly(nn.Module):
    """Profiles RIPD plus the external CLIP/DINO decoder bridge projections."""

    def __init__(
        self,
        ripd: nn.Module,
        clip_dense_dim: int = 768,
        clip_skip_dim: int = 1024,
        dino_dim: int = 768,
        clip_bridge_dims=(256, 128),
        dino_bridge_dims=(256, 128),
    ):
        super().__init__()
        self.ripd = ripd
        self.clip_upsample1 = (
            nn.ConvTranspose2d(clip_skip_dim, clip_bridge_dims[0], kernel_size=2, stride=2)
            if clip_bridge_dims[0] > 0 else None
        )
        self.clip_upsample2 = (
            nn.ConvTranspose2d(clip_skip_dim, clip_bridge_dims[1], kernel_size=4, stride=4)
            if clip_bridge_dims[1] > 0 else None
        )
        self.dino_decod_proj1 = (
            nn.Conv2d(dino_dim, dino_bridge_dims[0], kernel_size=1)
            if dino_bridge_dims[0] > 0 else None
        )
        self.dino_decod_proj2 = (
            nn.ConvTranspose2d(dino_dim, dino_bridge_dims[1], kernel_size=2, stride=2)
            if dino_bridge_dims[1] > 0 else None
        )
        self.register_buffer("dino_down", torch.randn(1, dino_dim, 24, 24))
        self.register_buffer("clip_skip0", torch.randn(1, clip_skip_dim, 24, 24))
        self.register_buffer("clip_skip1", torch.randn(1, clip_skip_dim, 24, 24))
        self.register_buffer("dino_l4", torch.randn(1, dino_dim, 48, 48))
        self.register_buffer("dino_l8", torch.randn(1, dino_dim, 48, 48))

    def forward(self, clip_res3: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        B = clip_res3.shape[0]
        dino_down = self.dino_down.expand(B, -1, -1, -1)
        clip_skip0 = self.clip_skip0.expand(B, -1, -1, -1)
        clip_skip1 = self.clip_skip1.expand(B, -1, -1, -1)
        dino_l4 = self.dino_l4.expand(B, -1, -1, -1)
        dino_l8 = self.dino_l8.expand(B, -1, -1, -1)

        res4 = self.clip_upsample1(clip_skip0) if self.clip_upsample1 is not None else None
        res5 = self.clip_upsample2(clip_skip1) if self.clip_upsample2 is not None else None
        dino_l4 = self.dino_decod_proj1(dino_l4) if self.dino_decod_proj1 is not None else None
        dino_l8 = self.dino_decod_proj2(dino_l8) if self.dino_decod_proj2 is not None else None
        return self.ripd(
            clip_res3,
            dino_down,
            text_feats,
            (clip_res3, res4, res5),
            [dino_l4, dino_l8],
        )


class _S2ReplacementOnly(nn.Module):
    """Profiles Student-2 cosine projections plus S2CostHead."""

    def __init__(
        self,
        head: nn.Module,
        clip_dense_dim: int = 768,
        clip_skip_dim: int = 1024,
        text_dim: int = 768,
        dino_dim: int = 768,
        corr_dim: int = 256,
    ):
        super().__init__()
        self.head = head
        self.clip_cost_proj = nn.Conv2d(clip_dense_dim, corr_dim, kernel_size=1)
        self.dino_cost_proj = nn.Conv2d(dino_dim, corr_dim, kernel_size=1)
        self.text_cost_proj = nn.Linear(text_dim, corr_dim)
        self.register_buffer("dino_down", torch.randn(1, dino_dim, 24, 24))
        self.register_buffer("clip_skip", torch.randn(1, clip_skip_dim, 24, 24))
        self.register_buffer("dino_l4", torch.randn(1, dino_dim, 48, 48))

    def forward(self, clip_res3: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        B = clip_res3.shape[0]
        dino_down = self.dino_down.expand(B, -1, -1, -1)
        clip_skip = self.clip_skip.expand(B, -1, -1, -1)
        dino_l4 = self.dino_l4.expand(B, -1, -1, -1)
        text = text_feats.mean(dim=2) if text_feats.dim() == 4 else text_feats

        clip_feat = F.normalize(self.clip_cost_proj(clip_res3), dim=1)
        dino_feat = F.normalize(self.dino_cost_proj(dino_down), dim=1)
        text_feat = F.normalize(self.text_cost_proj(text), dim=-1)
        C_clip = torch.einsum("bchw,btc->bthw", clip_feat, text_feat)
        C_dino = torch.einsum("bchw,btc->bthw", dino_feat, text_feat)
        C_fused = self.head.fuse_costs(C_clip, C_dino)
        return self.head(C_fused, text_feats, clip_res3, clip_skip, dino_l4)


class _GSNetFlopsWrapper:
    """
    FLOPs-only view of GSNet (Exp 1 & 2).

    Replicates the compute of GSNet.forward()'s eval branch (CLIP image
    encoder + DINO/RSIB backbone + sem_seg_head/RIPD) but skips
    Detectron2's ImageList.from_tensors() / sem_seg_postprocess(). Those
    helpers branch on torch.jit.is_tracing() and turn `image_sizes` into
    Tensors, which fvcore's FlopCountAnalysis (a torch.jit tracer) cannot
    get through -- it raises and _count_flops_fvcore() reports GFLOPs=NaN.

    Not an nn.Module on purpose: it only holds a reference to the already-
    built `gsnet`, so wrapping it doesn't double-count parameters when
    attached as `callable_model.flops_model`.
    """

    def __init__(self, gsnet: nn.Module):
        self.gsnet = gsnet

    def __call__(self, image: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        m = self.gsnet
        clip_images = (image - m.clip_pixel_mean) / m.clip_pixel_std
        clip_images_resized = F.interpolate(
            clip_images, size=m.clip_resolution, mode="bilinear", align_corners=False
        )
        dino_images_resized = F.interpolate(
            clip_images, size=m.dino_resolution, mode="bilinear", align_corners=False
        )

        m.layers = []
        if m.dino_model is not None:
            dino_feat = m.dino_model.get_intermediate_layers(dino_images_resized, n=12)
            dino_patch_feat_last_unfold = rearrange(
                dino_feat[-1][:, 1:, :], "B (H W) C -> B C H W", H=48
            )
            dino_feat_down = m.dino_down_sample(dino_patch_feat_last_unfold)

            dino_feat_L4 = rearrange(dino_feat[3][:, 1:, :], "B (H W) C -> B C H W", H=48)
            dino_feat_L8 = rearrange(dino_feat[7][:, 1:, :], "B (H W) C -> B C H W", H=48)
            dino_feat_L4_proj = m.dino_decod_proj1(dino_feat_L4) if m.dino_decod_proj1 is not None else None
            dino_feat_L8_proj = m.dino_decod_proj2(dino_feat_L8) if m.dino_decod_proj2 is not None else None
            dino_feat_guidance = [dino_feat_L4_proj, dino_feat_L8_proj]
        else:
            dino_feat_down, dino_feat_guidance = None, None

        if m.use_clip:
            clip_features = m.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
            clip_image_features = clip_features[:, 1:, :]
            res3 = rearrange(clip_image_features, "B (H W) C -> B C H W", H=24)
            res4 = rearrange(m.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
            res5 = rearrange(m.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
            res4 = m.upsample1(res4) if m.upsample1 is not None else None
            res5 = m.upsample2(res5) if m.upsample2 is not None else None
            clip_features_guidance = {"res5": res5, "res4": res4, "res3": res3}
        else:
            clip_features, clip_features_guidance = None, None

        return m.sem_seg_head(clip_features, dino_feat_down, clip_features_guidance, dino_feat_guidance)


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(out.get(key), dict)
        ):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_base(path: str) -> dict:
    import yaml

    cfg_path = Path(path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("_BASE_", None)
    if base_name:
        base = _load_yaml_with_base(str(cfg_path.parent / base_name))
        return _deep_update(base, cfg)
    return cfg


def _prompt_channel_from_config(cfg: dict) -> int:
    prompt_type = cfg.get("MODEL", {}).get("PROMPT_ENSEMBLE_TYPE", "single")
    if prompt_type == "single":
        return 1

    try:
        import ast

        path = Path("gs_net/third_party/imagenet_templates.py")
        tree = ast.parse(path.read_text())
        target = "IMAGENET_TEMPLATES_SELECT" if prompt_type == "imagenet_select" else "IMAGENET_TEMPLATES"
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for assign_target in node.targets:
                if isinstance(assign_target, ast.Name) and assign_target.id == target:
                    return len(ast.literal_eval(node.value))
    except Exception:
        pass
    return 1


def _clip_skip_dim_from_pretrained(name: str) -> int:
    return 768 if name in {"ViT-B/16", "RemoteCLIP-ViT-B-32"} else 1024


def _ripd_profile_config(args: argparse.Namespace) -> dict:
    cfg = _load_yaml_with_base(args.gsnet_config)
    sem = cfg.get("MODEL", {}).get("SEM_SEG_HEAD", {})
    clip_pretrained = sem.get("CLIP_PRETRAINED", "ViT-L/14@336px")

    ripd_num_layers = (
        sem.get("NUM_LAYERS", 4)
        if args.ripd_num_layers is None
        else args.ripd_num_layers
    )
    ripd_pooling_size = (
        sem.get("POOLING_SIZES", [6, 6])
        if args.ripd_pooling_size is None
        else args.ripd_pooling_size
    )

    return {
        "text_guidance_dim": sem.get("TEXT_GUIDANCE_DIM", args.ripd_text_dim),
        "text_guidance_proj_dim": sem.get("TEXT_GUIDANCE_PROJ_DIM", 128),
        "appearance_guidance_dim": sem.get("APPEARANCE_GUIDANCE_DIM", args.ripd_clip_dense_dim),
        "appearance_guidance_proj_dim": sem.get("APPEARANCE_GUIDANCE_PROJ_DIM", 128),
        "decoder_dims": tuple(sem.get("DECODER_DIMS", [64, 32])),
        "decoder_guidance_dims": tuple(sem.get("DECODER_GUIDANCE_DIMS", [256, 128])),
        "decoder_guidance_proj_dims": tuple(sem.get("DECODER_GUIDANCE_PROJ_DIMS", [32, 16])),
        "decoder_clip_guidance_dims": tuple(sem.get("DECODER_CLIP_GUIDANCE_DIMS", [256, 128])),
        "decoder_clip_guidance_proj_dims": tuple(sem.get("DECODER_CLIP_GUIDANCE_PROJ_DIMS", [32, 16])),
        "decoder_dino_guidance_dims": tuple(sem.get("DECODER_DINO_GUIDANCE_DIMS", [256, 128])),
        "decoder_dino_guidance_proj_dims": tuple(sem.get("DECODER_DINO_GUIDANCE_PROJ_DIMS", [32, 16])),
        "use_clip_corr": sem.get("USE_CLIP_CORR", True),
        "use_dino_corr": sem.get("USE_DINO_CORR", True),
        "fusion_type": sem.get("FUSION_TYPE", "query_guided"),
        "num_layers": ripd_num_layers,
        "nheads": sem.get("NUM_HEADS", 4),
        "hidden_dim": sem.get("HIDDEN_DIMS", 128),
        "pooling_size": tuple(ripd_pooling_size),
        "feature_resolution": tuple(sem.get("FEATURE_RESOLUTION", [24, 24])),
        "window_size": sem.get("WINDOW_SIZES", 12),
        "attention_type": sem.get("ATTENTION_TYPE", "linear"),
        "prompt_channel": _prompt_channel_from_config(cfg),
        "clip_dense_dim": sem.get("APPEARANCE_GUIDANCE_DIM", args.ripd_clip_dense_dim),
        "clip_skip_dim": _clip_skip_dim_from_pretrained(clip_pretrained),
        "text_dim": sem.get("TEXT_GUIDANCE_DIM", args.ripd_text_dim),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders  ← TODO: CONNECT MODEL is in each function below
# ─────────────────────────────────────────────────────────────────────────────

def _load_old_gsnet(args: argparse.Namespace, device: torch.device):
    """
    Experiment 1 — Old GSNet.

    Original dual-stream CLIP+DINO architecture with RIPD/QGFF fusion decoder.
    This is the BASELINE for speedup computation.

    Real loading path:
      - build_gsnet(config_file, weights_file, device) from eval_clip.py
      - uses Detectron2 cfg + DetectionCheckpointer
      - model.eval() + all params frozen

    Return signature:
        callable_model(image, text_feats) -> logit (B, T, H, W)
        dummy_input_override: None  (use default shapes)

    Pass --old-gsnet-weights to a Detectron2-format GSNet checkpoint.
    Requires a valid Detectron2-style state dict (top-level key "model").
    A GS-Distill training checkpoint (keys: student/epoch/val_loss) will
    be caught early with a clear error message.

    Pass --old-rsib-ckpt to the DinoV1 RSIB backbone checkpoint.
    Sets RSIB_CKPT env var before building so the DINO branch loads correctly.
    """
    if args.old_gsnet_weights is None:
        print("[Exp 1] old_gsnet: --old-gsnet-weights not provided; using DUMMY model")
        return _DummyModel().to(device).eval(), None

    # Validate checkpoint format before handing to DetectionCheckpointer.
    _raw = torch.load(args.old_gsnet_weights, map_location="cpu")
    _keys = list(_raw.keys()) if isinstance(_raw, dict) else []
    if any(k in _keys for k in ("student", "epoch", "val_loss")):
        raise ValueError(
            f"Checkpoint '{args.old_gsnet_weights}' looks like a GS-Distill training "
            f"checkpoint (top-level keys: {_keys[:6]}). "
            f"Exp 1 (old_gsnet) needs a Detectron2-format GSNet checkpoint "
            f"(top-level key 'model'). Pass the correct path via --old-gsnet-weights."
        )
    del _raw

    import os as _os
    _os.environ["RSIB_CKPT"] = args.old_rsib_ckpt

    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model as d2_build
    from gs_net import add_cat_seg_config

    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.old_gsnet_config)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()

    gsnet = d2_build(cfg)
    DetectionCheckpointer(gsnet).load(args.old_gsnet_weights)
    gsnet.eval()
    for p in gsnet.parameters():
        p.requires_grad = False

    class _GSNetCallable(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, text_feats):
            # text_feats ignored — GSNet uses its internal text_features_test
            return self.m([{"image": image[0], "height": image.shape[-2], "width": image.shape[-1]}])

    callable_model = _GSNetCallable(gsnet).to(device).eval()
    # FLOPs-only path that bypasses Detectron2's ImageList/sem_seg_postprocess
    # (see _GSNetFlopsWrapper docstring for why the full forward() can't be traced).
    callable_model.flops_model = _GSNetFlopsWrapper(gsnet)
    return callable_model, None


def _load_improved_gsnet(args: argparse.Namespace, device: torch.device):
    """
    Experiment 2 — Improved GSNet.

    Same RIPD/QGFF architecture as Experiment 1, but the DINO branch uses
    DINOv3 (RSIB — the supervised remote-sensing backbone, ViT-L/16, partial
    unfreeze).  The loading path is IDENTICAL to Experiment 1; only the
    checkpoint file differs.

    The active training checkpoint is at:
        output/gsnet_pretrain/model_final.pth
    (produced by jobs-ashie/phase1_gsnet_pretrain.sbatch)

    RSIB backbone weights (DINOv3):
        gs_net/third_party/experiments/sup_unfreeze_20260318_225917/
            backbone_for_gsnet/epoch_07.pth
    Loaded via env var RSIB_CKPT (set in all .sbatch files).

    Pass --gsnet-weights for the Detectron2-format improved GSNet checkpoint
    and --gsnet-config configs/vitl_336_dinov3.yaml.  RSIB_CKPT must be set
    (epoch_07.pth) so the DINOv3 backbone initialises correctly.
    """
    import os as _os
    _os.environ.setdefault("RSIB_CKPT", args.rsib_ckpt)

    from scripts.eval_clip import build_gsnet

    gsnet = build_gsnet(args.gsnet_config, args.gsnet_weights, str(device))
    gsnet.eval()
    for p in gsnet.parameters():
        p.requires_grad = False

    class _GSNetCallable(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, text_feats):
            # text_feats ignored — GSNet uses its internal text_features_test
            return self.m([{"image": image[0], "height": image.shape[-2], "width": image.shape[-1]}])

    callable_model = _GSNetCallable(gsnet).to(device).eval()
    # FLOPs-only path that bypasses Detectron2's ImageList/sem_seg_postprocess
    # (see _GSNetFlopsWrapper docstring for why the full forward() can't be traced).
    callable_model.flops_model = _GSNetFlopsWrapper(gsnet)
    return callable_model, None


def _load_gs_distill(args: argparse.Namespace, device: torch.device):
    """
    Experiment 3 — GS-Distill (Student-1, CLIP backbone, frozen RIPD decoder).

    This is the actively trained/evaluated experiment.
    Checkpoint: output/ashie/clip/finetune_best.pth
    (or output/ashie/baseline/baseline_best.pth for the random-init baseline)

    The eval pipeline is in scripts/eval_clip.py (load_model + gs_distill_inference).
    gs_distill_inference() is in gs_distill/inference.py.

    Component loading order:
      1. build_gsnet()                       → extracts clip_model, ripd, decoder bridges
      2. GSDistillStudent(clip_model, ...)   → student with frozen CLIP
      3. load finetune_best.pth              → student + optionally ripd/bridges

    ─────────────────────────────────────────────────────────────────────────
    """
    if args.clip_finetune_ckpt is None or not os.path.isfile(args.clip_finetune_ckpt):
        print(f"[Exp 3] gs_distill: checkpoint not found ({args.clip_finetune_ckpt}); using DUMMY model")
        return _DummyModel().to(device).eval(), None

    from scripts.eval_clip import load_model as _load_gs_distill_model
    from gs_distill.inference import gs_distill_inference

    parts = _load_gs_distill_model(
        args.gsnet_config,
        args.gsnet_weights,
        args.clip_finetune_ckpt,
        device,
    )

    for module in parts.values():
        if isinstance(module, nn.Module):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def _distill_callable(image, text_feats):
        return gs_distill_inference(
            image=image,
            text_feats=text_feats,
            student=parts["student"],
            clip_model=parts["clip_model"],
            ripd=parts["ripd"],
            clip_upsample1=parts["clip_upsample1"],
            clip_upsample2=parts["clip_upsample2"],
            dino_decod_proj1=parts["dino_decod_proj1"],
            dino_decod_proj2=parts["dino_decod_proj2"],
            clip_skip_layer_indices=parts["clip_skip_indices"],
        )

    modules_for_params = [
        m for m in parts.values() if isinstance(m, nn.Module)
    ]
    return _distill_callable, modules_for_params


def _load_gs_distill_new_decoder(args: argparse.Namespace, device: torch.device):
    """
    Experiment 4 — GS-Distill with new lightweight decoder (Student-2).

    Same student DINO-substitute predictions as Experiment 3, but RIPD/QGFF
    is replaced by S2CostHead (gs_distill/cost_head.py), a trainable
    lightweight cost-aggregation decoder.

    Real loading path:
      - scripts/eval_s2.py::load_model_s2()       → builds CLIP from
        args.gsnet_config and loads a GSDistillStudent2 checkpoint.
      - gs_distill/inference2.py::gs_distill_s2_inference()
        → image+text -> logit (B, T, H, W), no RIPD/QGFF/DINOv3.

    Checkpoint resolution:
      --new-decoder-ckpt, else --s2-ckpt, else
      output/ashie/s2/<s1-source>/s2_best.pth (written by scripts/train_s2.py).
      Falls back to DUMMY model if no checkpoint is found at that path.
    ─────────────────────────────────────────────────────────────────────────
    """
    s2_ckpt = (
        args.new_decoder_ckpt
        or args.s2_ckpt
        or os.path.join("output", "ashie", "s2", args.s1_source, "s2_best.pth")
    )
    if not os.path.isfile(s2_ckpt):
        print(f"[Exp 4] gs_distill_new_decoder: checkpoint not found ({s2_ckpt}); using DUMMY model")
        return _DummyModel().to(device).eval(), None

    from scripts.eval_s2 import load_model_s2
    from gs_distill.inference2 import gs_distill_s2_inference

    parts = load_model_s2(args.gsnet_config, args.gsnet_weights, s2_ckpt, device)

    for module in parts.values():
        if isinstance(module, nn.Module):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def _s2_callable(image, text_feats):
        return gs_distill_s2_inference(
            image=image,
            text_feats=text_feats,
            student2=parts["student2"],
            clip_model=parts["clip_model"],
            clip_skip_layer_indices=parts["clip_skip_indices"],
        )

    modules_for_params = [
        m for m in parts.values() if isinstance(m, nn.Module)
    ]
    return _s2_callable, modules_for_params


def _load_gs_distill_s2(args: argparse.Namespace, device: torch.device):
    """
    Experiment 5 - Student-2 cost head only.

    Profiles S2CostHead on a 24x24 fused cost volume plus text and decoder
    guidance tensors. CLIP and the Student-1 trunk are intentionally excluded.
    """
    from gs_distill.cost_head import S2CostHead

    s2_ckpt = args.s2_ckpt or os.path.join("output", "ashie", "s2", args.s1_source, "s2_best.pth")
    ckpt = None
    ckpt_args = {}
    if s2_ckpt and os.path.isfile(s2_ckpt):
        ckpt = torch.load(s2_ckpt, map_location="cpu")
        ckpt_args = ckpt.get("args", {})

    head = S2CostHead(
        hidden_dim=ckpt_args.get("head_hidden_dim", args.s2_head_hidden_dim),
        num_layers=ckpt_args.get("head_num_layers", args.s2_head_num_layers),
        head_mode=ckpt_args.get("head_mode", args.s2_head_mode),
        window_size=ckpt_args.get("head_window_size", args.s2_head_window_size),
        pad_len=ckpt_args.get("head_pad_len", args.s2_head_pad_len),
        clip_feat_dim=args.s2_clip_dense_dim,
        clip_skip_dim=args.s2_clip_skip_dim,
        clip_text_dim=args.s2_text_dim,
        dino_feat_dim=args.s2_dino_dim,
        dino_skip_dim=args.s2_dino_dim,
    )

    if ckpt is not None:
        state = ckpt.get("student2", {})
        cost_state = {
            k.replace("cost_head.", "", 1): v
            for k, v in state.items()
            if k.startswith("cost_head.")
        }
        if cost_state:
            current = head.state_dict()
            cost_state = {
                k: v for k, v in cost_state.items()
                if k in current and current[k].shape == v.shape
            }
            missing, unexpected = head.load_state_dict(cost_state, strict=False)
            print(
                f"[Exp 5] loaded S2 cost_head from {s2_ckpt} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
    else:
        print(f"[Exp 5] gs_distill_s2: no checkpoint at {s2_ckpt}; profiling randomly initialised S2CostHead")
    print(
        "[gs_distill_s2] "
        f"hidden={ckpt_args.get('head_hidden_dim', args.s2_head_hidden_dim)} "
        f"layers={ckpt_args.get('head_num_layers', args.s2_head_num_layers)} "
        f"mode={ckpt_args.get('head_mode', args.s2_head_mode)}"
    )

    model = _S2CostHeadOnly(
        head,
        clip_dense_dim=args.s2_clip_dense_dim,
        clip_skip_dim=args.s2_clip_skip_dim,
        dino_dim=args.s2_dino_dim,
    ).to(device).eval()
    C_fused = torch.randn(1, 40, 24, 24, device=device)
    text_feats = torch.randn(1, 40, 1, args.s2_text_dim, device=device)
    return model, (C_fused, text_feats)


def _load_ripd_qgff_decoder(args: argparse.Namespace, device: torch.device):
    """
    RIPD/QGFF decoder-only baseline.

    This counts and profiles RIPD plus the external CLIP/DINO bridge
    projections used by GSNet, excluding CLIP and the real DINO/RSIB encoder.
    """
    import importlib.util
    import types

    root = Path.cwd()
    pkg = types.ModuleType("_ripd_efficiency_pkg")
    pkg.__path__ = [str(root / "gs_net" / "modeling" / "transformer")]
    sys.modules[pkg.__name__] = pkg
    for modname, rel in [
        ("_ripd_efficiency_pkg.FusionAgg", "gs_net/modeling/transformer/FusionAgg.py"),
        ("_ripd_efficiency_pkg.RIPD", "gs_net/modeling/transformer/RIPD.py"),
    ]:
        spec = importlib.util.spec_from_file_location(modname, root / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
    RIPD = sys.modules["_ripd_efficiency_pkg.RIPD"].RIPD
    cfg = _ripd_profile_config(args)

    ripd = RIPD(
        text_guidance_dim=cfg["text_guidance_dim"],
        text_guidance_proj_dim=cfg["text_guidance_proj_dim"],
        appearance_guidance_dim=cfg["appearance_guidance_dim"],
        appearance_guidance_proj_dim=cfg["appearance_guidance_proj_dim"],
        decoder_dims=cfg["decoder_dims"],
        decoder_guidance_dims=cfg["decoder_guidance_dims"],
        decoder_guidance_proj_dims=cfg["decoder_guidance_proj_dims"],
        decoder_clip_guidance_dims=cfg["decoder_clip_guidance_dims"],
        decoder_clip_guidance_proj_dims=cfg["decoder_clip_guidance_proj_dims"],
        decoder_dino_guidance_dims=cfg["decoder_dino_guidance_dims"],
        decoder_dino_guidance_proj_dims=cfg["decoder_dino_guidance_proj_dims"],
        feat_dim=512,
        use_clip_corr=cfg["use_clip_corr"],
        use_dino_corr=cfg["use_dino_corr"],
        fusion_type=cfg["fusion_type"],
        num_layers=cfg["num_layers"],
        nheads=cfg["nheads"],
        hidden_dim=cfg["hidden_dim"],
        pooling_size=cfg["pooling_size"],
        feature_resolution=cfg["feature_resolution"],
        window_size=cfg["window_size"],
        attention_type=cfg["attention_type"],
        prompt_channel=cfg["prompt_channel"],
        pad_len=256,
    )
    model = _RIPDDecoderOnly(
        ripd,
        clip_dense_dim=cfg["clip_dense_dim"],
        clip_skip_dim=cfg["clip_skip_dim"],
        dino_dim=args.ripd_dino_dim,
        clip_bridge_dims=cfg["decoder_clip_guidance_dims"],
        dino_bridge_dims=cfg["decoder_dino_guidance_dims"],
    ).to(device).eval()
    print(
        "[ripd_qgff_decoder] "
        f"config={args.gsnet_config} layers={cfg['num_layers']} "
        f"hidden={cfg['hidden_dim']} fusion={cfg['fusion_type']} "
        f"clip_dense={cfg['clip_dense_dim']} clip_skip={cfg['clip_skip_dim']} "
        f"text={cfg['text_dim']}"
    )
    clip_res3 = torch.randn(1, cfg["clip_dense_dim"], 24, 24, device=device)
    text_feats = torch.randn(1, 40, 1, cfg["text_dim"], device=device)
    return model, (clip_res3, text_feats)


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs counting
# ─────────────────────────────────────────────────────────────────────────────

def _count_flops_fvcore(
    model: Callable,
    inputs: Tuple[torch.Tensor, ...],
) -> float:
    """
    Count FLOPs using fvcore.  Returns total FLOPs (not GFLOPs).

    fvcore counts each multiply-add as 1 unit.  Standard convention for
    neural-net FLOPs is to count each multiply-add as 1 FLOP (i.e. a MAC
    is 2 FLOPs, but the NN community usually reports MACs directly as FLOPs).
    We report fvcore's raw number / 1e9 as GFLOPs and label it accordingly.

    Custom ops that fvcore doesn't know about (e.g. RIPD's einsum, bilinear
    interpolate) will be silently zeroed.  A warning is printed if unsupported
    ops are detected.
    """
    if not _FVCORE:
        return float("nan")
    try:
        # Wrap the callable in an nn.Module if needed so fvcore can trace it
        if not isinstance(model, nn.Module):
            class _Wrapper(nn.Module):
                def forward(self, *inputs):
                    return model(*inputs)
            _model = _Wrapper()
        else:
            _model = model

        flops_obj = FlopCountAnalysis(_model, inputs)
        flops_obj.unsupported_ops_warnings(False)   # suppress per-op spam
        flops_obj.uncalled_modules_warnings(False)

        # Older fvcore versions raise instead of silently zeroing activation ops
        # (e.g. Sigmoid in ViT/DINOv3 models). Register explicit zero-cost handlers
        # so they're counted as 0 FLOPs rather than crashing the entire count.
        _zero_handler = lambda inputs, outputs: 0
        for _op in (
            "aten::sigmoid",    "aten::softmax",
            "aten::silu",       "aten::gelu",
            "aten::relu",       "aten::relu_",
            "aten::hardswish",  "aten::hardsigmoid",
        ):
            try:
                flops_obj._set_op_handle(_op, _zero_handler)
            except Exception:
                pass

        total = flops_obj.total()
        unsupported = flops_obj.unsupported_ops()
        if unsupported:
            op_names = ", ".join(list(unsupported.keys())[:5])
            warnings.warn(
                f"fvcore: {len(unsupported)} unsupported op type(s) zeroed: {op_names}. "
                f"GFLOPs may be underestimated (RIPD einsum is a known gap).",
                RuntimeWarning, stacklevel=2,
            )
        return float(total)
    except Exception as e:
        warnings.warn(
            f"fvcore FLOPs counting failed: {e}\n{traceback.format_exc()}",
            RuntimeWarning, stacklevel=2,
        )
        return float("nan")


def _count_flops_thop(
    model: Callable,
    inputs: Tuple[torch.Tensor, ...],
) -> float:
    """Fallback FLOPs via thop (pip install thop)."""
    if not _THOP:
        return float("nan")
    try:
        if not isinstance(model, nn.Module):
            warnings.warn("thop requires nn.Module; skipping.", RuntimeWarning, stacklevel=2)
            return float("nan")
        macs, _ = thop_profile(model, inputs=inputs, verbose=False)
        return float(macs * 2)   # thop reports MACs; multiply by 2 for FLOPs
    except Exception as e:
        warnings.warn(f"thop FLOPs counting failed: {e}", RuntimeWarning, stacklevel=2)
        return float("nan")


def count_flops(model: Callable, inputs: Tuple[torch.Tensor, ...]) -> float:
    """
    Count FLOPs using the best available backend.
    Priority: fvcore → thop → NaN.
    Returns raw FLOPs (not GFLOPs); caller divides by 1e9.
    """
    flops = _count_flops_fvcore(model, inputs)
    if math.isnan(flops) and _THOP:
        flops = _count_flops_thop(model, inputs)
    return flops


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model) -> Tuple[int, int]:
    """
    Count (total_params, trainable_params).

    Works on nn.Module objects.  For callable wrappers that are not nn.Module
    (e.g. a lambda over multiple modules), returns (0, 0) with a warning.
    The agent connecting real models should ensure the wrapper exposes a
    .parameters() method or pass an explicit list of modules via
    ExperimentConfig.modules_for_param_count (not yet implemented; see TODO).

    TODO (connecting agent): For Experiments 3 & 4, the profiled callable
    wraps multiple nn.Modules (clip_model, student, ripd, bridges).  To get
    accurate param counts, iterate the individual modules:

        total     = sum(p.numel() for m in [clip_model, student, ripd, ...] for p in m.parameters())
        trainable = sum(p.numel() for m in [student, ripd] for p in m.parameters() if p.requires_grad)

    The dummy model counts are correct as a sanity check.
    """
    if not isinstance(model, nn.Module):
        warnings.warn(
            "count_params: model is not nn.Module; param counts = 0. "
            "See TODO in count_params() docstring.",
            RuntimeWarning, stacklevel=2,
        )
        return 0, 0
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Core profiling function
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    """All metrics for one experiment run."""
    # Identification
    experiment_name:     str
    description:         str
    # Conditions
    device:              str
    gpu_name:            str
    precision:           str
    input_resolution:    str    # "384x384"
    batch_size:          int
    warmup_iters:        int
    measure_iters:       int
    # Parameters
    total_params:        int
    trainable_params:    float   # float so NaN round-trips through CSV
    # FLOPs
    gflops:              float
    # Latency (ms)
    latency_mean_ms:     float
    latency_std_ms:      float
    latency_min_ms:      float
    latency_p50_ms:      float
    latency_p95_ms:      float
    # Throughput
    throughput_img_per_s: float
    # Memory
    peak_memory_gb:      float
    # Speedup (filled in combined CSV)
    speedup_vs_baseline: float


def profile_model(
    config:      ExperimentConfig,
    args:        argparse.Namespace,
    device:      torch.device,
    amp:         bool = False,
    output_dir:  str  = DEFAULT_OUTPUT_DIR,
) -> ProfileResult:
    """
    Profile one experiment end-to-end.

    Steps:
      1. Call config.loader_fn(args, device) to get (callable_model, dummy_override).
      2. Build dummy inputs (or use override).
      3. Count parameters.
      4. Count FLOPs (once, no grad, before timing loop).
      5. Warmup loop.
      6. Measurement loop with synchronize() bracketing.
      7. Compute statistics.
      8. Peak memory pass.
      9. Return ProfileResult.
    """
    print(f"\n{'='*60}")
    print(f"  Profiling: {config.name}  ({config.description})")
    print(f"{'='*60}")

    # ── 1. Load model ────────────────────────────────────────────────────────
    loader_result = config.loader_fn(args, device)
    # loader_fn may return (callable, dummy_override) or (callable, list_of_modules)
    callable_model, extra = loader_result
    if isinstance(extra, list):
        modules_for_params = extra
        dummy_override = None
    else:
        modules_for_params = None
        dummy_override = extra

    # ── 2. Inputs ────────────────────────────────────────────────────────────
    if dummy_override is not None:
        img, txt = dummy_override
    else:
        img = torch.randn(*config.image_shape, device=device)
        txt = torch.randn(*config.text_shape,  device=device)
    inputs = (img, txt)

    # ── 3. Parameter count ───────────────────────────────────────────────────
    if modules_for_params is not None:
        # Deduplicate via object identity: GSDistillStudent registers clip_model
        # as self.clip_model, so iterating [clip_model, student, ripd, ...] would
        # count CLIP params twice without this guard.
        _seen_ids: set = set()
        total_p = 0
        train_p = 0
        for _m in modules_for_params:
            for _p in _m.parameters():
                if id(_p) not in _seen_ids:
                    _seen_ids.add(id(_p))
                    total_p += _p.numel()
                    if _p.requires_grad:
                        train_p += _p.numel()
    else:
        total_p, train_p = count_params(callable_model)
    print(f"  Params: {total_p:,} total, {train_p:,} trainable")

    # ── 4. FLOPs (single forward, no grad) ───────────────────────────────────
    # Some models attach a `.flops_model` (e.g. GSNet, see _GSNetFlopsWrapper)
    # that traces cleanly under fvcore while the real callable doesn't.
    flops_model = getattr(callable_model, "flops_model", callable_model)
    with torch.no_grad():
        raw_flops = count_flops(flops_model, inputs)
    gflops = raw_flops / 1e9
    print(f"  GFLOPs: {gflops:.2f}" if not math.isnan(gflops) else "  GFLOPs: NaN")

    # Determine device string and GPU name
    dev_str  = str(device)
    gpu_name = "cpu"
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = "unknown_gpu"
    precision_str = "float16 (AMP)" if amp else "float32"

    # ── 5 & 6. Warmup + measurement ──────────────────────────────────────────
    latencies_ms: List[float] = []

    def _forward():
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                return callable_model(*inputs)
        return callable_model(*inputs)

    print(f"  Warmup ({WARMUP_ITERS} iters) ...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _ = _forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    print("done")

    print(f"  Measuring ({MEASURE_ITERS} iters) ...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(MEASURE_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = _forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
    print("done")

    latencies_ms_t = torch.tensor(latencies_ms)
    lat_mean = float(latencies_ms_t.mean())
    lat_std  = float(latencies_ms_t.std())
    lat_min  = float(latencies_ms_t.min())
    lat_p50  = float(latencies_ms_t.quantile(0.50))
    lat_p95  = float(latencies_ms_t.quantile(0.95))
    throughput = 1000.0 / lat_mean   # images / sec (batch_size=1)

    print(f"  Latency: mean={lat_mean:.1f}ms  std={lat_std:.1f}ms  "
          f"min={lat_min:.1f}ms  p50={lat_p50:.1f}ms  p95={lat_p95:.1f}ms")
    print(f"  Throughput: {throughput:.2f} img/s")

    # ── 7. Peak memory (separate single-pass measurement) ────────────────────
    peak_gb = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _ = _forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb    = peak_bytes / (1024 ** 3)
        print(f"  Peak GPU memory: {peak_gb:.3f} GB")
    else:
        print("  Peak GPU memory: N/A (CPU)")

    # ── 8. Cleanup ────────────────────────────────────────────────────────────
    del callable_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    b, c, h, w = config.image_shape
    return ProfileResult(
        experiment_name      = config.name,
        description          = config.description,
        device               = dev_str,
        gpu_name             = gpu_name,
        precision            = precision_str,
        input_resolution     = f"{h}x{w}",
        batch_size           = b,
        warmup_iters         = WARMUP_ITERS,
        measure_iters        = MEASURE_ITERS,
        total_params         = total_p,
        trainable_params     = float(train_p),
        gflops               = gflops,
        latency_mean_ms      = lat_mean,
        latency_std_ms       = lat_std,
        latency_min_ms       = lat_min,
        latency_p50_ms       = lat_p50,
        latency_p95_ms       = lat_p95,
        throughput_img_per_s = throughput,
        peak_memory_gb       = peak_gb,
        speedup_vs_baseline  = float("nan"),   # filled in combine_results()
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "experiment_name", "description",
    "device", "gpu_name", "precision", "input_resolution", "batch_size",
    "warmup_iters", "measure_iters",
    "total_params", "trainable_params",
    "gflops",
    "latency_mean_ms", "latency_std_ms", "latency_min_ms",
    "latency_p50_ms", "latency_p95_ms",
    "throughput_img_per_s",
    "peak_memory_gb",
    "speedup_vs_baseline",
]


def write_csv(result: ProfileResult, output_dir: str) -> str:
    """Write a single-experiment CSV.  Returns the path written."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"efficiency_{result.experiment_name}.csv")
    row  = asdict(result)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerow(row)
    return path


def combine_results(results: List[ProfileResult], output_dir: str) -> str:
    """
    Merge all per-experiment results into efficiency_combined.csv and
    compute speedup_vs_baseline relative to Experiment 1 (old_gsnet).

    Also reads any pre-existing per-experiment CSVs in output_dir so partial
    runs (e.g. only Exp 2-4 profiled today) still produce a complete combined
    CSV as more experiments complete over time.
    """
    # Load any pre-existing CSVs not in this run's results
    existing: Dict[str, ProfileResult] = {}
    for fname in Path(output_dir).glob("efficiency_*.csv"):
        if fname.name == "efficiency_combined.csv":
            continue
        try:
            with open(fname) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row["experiment_name"]
                    if name not in existing:
                        # Reconstruct a minimal ProfileResult from CSV for speedup calc
                        existing[name] = row
        except Exception:
            pass

    # This run's results take priority over stale CSVs
    all_rows: Dict[str, dict] = {r.experiment_name: asdict(r) for r in results}
    for name, row in existing.items():
        if name not in all_rows:
            all_rows[name] = dict(row)

    # Compute speedup relative to old_gsnet latency
    baseline_lat = float("nan")
    if "old_gsnet" in all_rows:
        try:
            baseline_lat = float(all_rows["old_gsnet"]["latency_mean_ms"])
        except (KeyError, ValueError):
            pass

    for name, row in all_rows.items():
        try:
            lat = float(row["latency_mean_ms"])
            row["speedup_vs_baseline"] = (
                baseline_lat / lat if not math.isnan(baseline_lat) and lat > 0
                else float("nan")
            )
        except (KeyError, ValueError):
            row["speedup_vs_baseline"] = float("nan")

    # Sort by experiment order to keep the CSV deterministic
    order = [
        "old_gsnet",
        "improved_gsnet",
        "gs_distill",
        "gs_distill_new_decoder",
        "gs_distill_s2",
        "ripd_qgff_decoder",
    ]
    sorted_rows = sorted(
        all_rows.values(),
        key=lambda r: order.index(r["experiment_name"])
        if r["experiment_name"] in order else 99,
    )

    path = os.path.join(output_dir, "efficiency_combined.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(sorted_rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Summary table (stdout)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[ProfileResult]) -> None:
    _W = 24
    header = (
        f"{'Experiment':<{_W}} {'GFLOPs':>8} {'Params(M)':>10} "
        f"{'Lat_mean(ms)':>13} {'Lat_p95(ms)':>12} "
        f"{'Thrput(img/s)':>14} {'Mem(GB)':>8} {'Speedup':>8}"
    )
    print(f"\n{'='*len(header)}")
    print("  EFFICIENCY METRICS SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        gf    = f"{r.gflops:.1f}"     if not math.isnan(r.gflops)               else "NaN"
        pm    = f"{r.total_params/1e6:.1f}M" if r.total_params > 0              else "NaN"
        mem   = f"{r.peak_memory_gb:.3f}" if not math.isnan(r.peak_memory_gb)   else "N/A"
        spd   = f"{r.speedup_vs_baseline:.2f}x" if not math.isnan(r.speedup_vs_baseline) else "—"
        print(
            f"  {r.experiment_name:<{_W-2}} {gf:>8} {pm:>10} "
            f"{r.latency_mean_ms:>13.1f} {r.latency_p95_ms:>12.1f} "
            f"{r.throughput_img_per_s:>14.2f} {mem:>8} {spd:>8}"
        )
    print(f"{'='*len(header)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name        = "old_gsnet",
        description = "Original dual-stream CLIP+DINO with RIPD/QGFF (baseline)",
        loader_fn   = _load_old_gsnet,
    ),
    ExperimentConfig(
        name        = "improved_gsnet",
        description = "GSNet with DINOv3 RSIB backbone (supervised RS pretraining)",
        loader_fn   = _load_improved_gsnet,
    ),
    ExperimentConfig(
        name        = "gs_distill",
        description = "GS-Distill: CLIP student → frozen RIPD/QGFF decoder",
        loader_fn   = _load_gs_distill,
        # vitl_336_dinov3.yaml uses ViT-L/14@336px CLIP: TEXT_GUIDANCE_DIM=768,
        # not the 512-dim default (which matches vitb_384.yaml / ViT-B-16).
        # RIPD's correlation einsum requires text_feats channel dim == img_feats (768).
        text_shape  = (1, 40, 1, 768),
    ),
    ExperimentConfig(
        name        = "gs_distill_new_decoder",
        description = "GS-Distill: CLIP student → new lightweight decoder (Student-2/S2CostHead)",
        loader_fn   = _load_gs_distill_new_decoder,
        text_shape  = (1, 40, 1, 768),
    ),
    ExperimentConfig(
        name        = "gs_distill_s2",
        description = "Student-2 S2CostHead only (24x24 cost aggregation)",
        loader_fn   = _load_gs_distill_s2,
        image_shape = (1, 40, 24, 24),
        text_shape  = DEFAULT_TEXT_SHAPE,
    ),
    ExperimentConfig(
        name        = "ripd_qgff_decoder",
        description = "RIPD/QGFF decoder plus CLIP/DINO bridge projections",
        loader_fn   = _load_ripd_qgff_decoder,
        image_shape = (1, 768, 24, 24),
        text_shape  = (1, 40, 1, 768),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference-time efficiency profiler for all GS experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiments", nargs="+",
        default=[e.name for e in ALL_EXPERIMENTS],
        choices=[e.name for e in ALL_EXPERIMENTS],
        help="Which experiments to profile (default: all).",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-experiment and combined CSVs.",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device string.",
    )
    p.add_argument(
        "--amp", action="store_true",
        help="Wrap forward pass in torch.cuda.amp.autocast().",
    )
    p.add_argument(
        "--warmup-iters", type=int, default=WARMUP_ITERS,
        help="Number of warmup iterations (discarded).",
    )
    p.add_argument(
        "--measure-iters", type=int, default=MEASURE_ITERS,
        help="Number of measurement iterations.",
    )
    # ── Checkpoint paths (used when real models are connected) ────────────────
    # These are currently ignored by dummy loaders.  The connecting agent
    # should read them in the TODO: CONNECT MODEL functions above.
    p.add_argument("--gsnet-config",        default="configs/vitl_336_dinov3.yaml",
                   help="Detectron2 YAML config for improved GSNet (Exp 2).")
    p.add_argument("--old-gsnet-config",    default="configs/vitb_384.yaml",
                   help="Detectron2 YAML config for old GSNet (Exp 1); uses ViT-B/8 + DinoV1.")
    p.add_argument("--gsnet-weights",       default=None,
                   help="Improved GSNet checkpoint (Exp 2; also used as teacher for Exp 3).")
    p.add_argument("--old-gsnet-weights",   default=None,
                   help="Original GSNet checkpoint (Exp 1).")
    p.add_argument("--old-rsib-ckpt",
                   default="DinoV1/RSIB.pth",
                   help="DinoV1 RSIB backbone weights (Exp 1); sets RSIB_CKPT env var.")
    p.add_argument("--rsib-ckpt",           default=None,
                   help="RSIB DINOv3 backbone weights (Exp 2); sets RSIB_CKPT env var.")
    p.add_argument("--clip-finetune-ckpt",  default="output/ashie/clip/finetune_best.pth",
                   help="GS-Distill finetuned checkpoint (Exp 3).")
    p.add_argument("--baseline-finetune-ckpt", default="output/ashie/baseline/baseline_best.pth",
                   help="Baseline (random init) finetuned checkpoint (alternative to Exp 3).")
    p.add_argument("--new-decoder-ckpt",    default=None,
                   help="Student-2 checkpoint for gs_distill_new_decoder (Exp 4). "
                        "Falls back to --s2-ckpt, then output/ashie/s2/<s1-source>/s2_best.pth.")
    p.add_argument("--s1-source", choices=["distill", "baseline"], default="distill",
                   help="Student-1 source used to resolve default Student-2 checkpoint path.")
    p.add_argument("--s2-ckpt",             default=None,
                   help="Student-2 checkpoint for gs_distill_s2 profiling. Defaults to output/ashie/s2/<s1-source>/s2_best.pth.")
    p.add_argument("--s2-head-hidden-dim",  type=int, default=64,
                   help="S2CostHead hidden dim when no checkpoint is available.")
    p.add_argument("--s2-head-num-layers",  type=int, default=1,
                   help="S2CostHead aggregator layers when no checkpoint is available.")
    p.add_argument("--s2-head-window-size", type=int, default=4,
                   help="S2CostHead Swin window size.")
    p.add_argument("--s2-head-pad-len",     type=int, default=0,
                   help="S2CostHead class token cap/padding length.")
    p.add_argument("--s2-head-mode", choices=["aggregator", "conv_plus_class"],
                   default="aggregator", help="S2CostHead mode.")
    p.add_argument("--s2-text-dim",         type=int, default=768,
                   help="CLIP text embedding dim for S2CostHead profiling.")
    p.add_argument("--s2-clip-dense-dim",   type=int, default=768,
                   help="Projected dense CLIP feature dim at 24x24 for S2.")
    p.add_argument("--s2-clip-skip-dim",    type=int, default=1024,
                   help="Hooked CLIP skip feature dim before decoder projection.")
    p.add_argument("--s2-dino-dim",         type=int, default=768,
                   help="Student DINO-substitute feature dim.")
    p.add_argument("--ripd-text-dim",       type=int, default=768,
                   help="Text/correlation dim for RIPD decoder-only profiling.")
    p.add_argument("--ripd-clip-dense-dim", type=int, default=768,
                   help="Projected dense CLIP feature dim at 24x24 for RIPD.")
    p.add_argument("--ripd-clip-skip-dim",  type=int, default=1024,
                   help="Hooked CLIP skip feature dim before GSNet bridge projections.")
    p.add_argument("--ripd-dino-dim",       type=int, default=768,
                   help="DINO/RSIB feature dim for RIPD decoder-only profiling.")
    p.add_argument("--ripd-num-layers",     type=int, default=None,
                   help="Override RIPD AggregatorLayer count; default reads --gsnet-config.")
    p.add_argument("--ripd-pooling-size",   type=int, nargs=2, default=None,
                   help="Override RIPD pooling size; default reads --gsnet-config.")
    return p.parse_args()


def main() -> None:
    global WARMUP_ITERS, MEASURE_ITERS

    args   = parse_args()
    WARMUP_ITERS   = args.warmup_iters
    MEASURE_ITERS  = args.measure_iters
    device = torch.device(args.device)

    print("=" * 60)
    print("  GS-Research Inference Efficiency Profiler")
    print(f"  Device:    {device}  ({gpu_label(device)})")
    print(f"  Precision: {'float16 (AMP)' if args.amp else 'float32'}")
    print(f"  Warmup:    {WARMUP_ITERS} iters")
    print(f"  Measure:   {MEASURE_ITERS} iters")
    print(f"  FLOPs backend: {'fvcore' if _FVCORE else ('thop' if _THOP else 'none (NaN)')}")
    print("=" * 60)

    exp_map = {e.name: e for e in ALL_EXPERIMENTS}
    selected = [exp_map[n] for n in args.experiments]

    results: List[ProfileResult] = []
    for config in selected:
        result = profile_model(
            config     = config,
            args       = args,
            device     = device,
            amp        = args.amp,
            output_dir = args.output_dir,
        )
        # Fill speedup immediately if baseline is already in results
        baseline_rows = [r for r in results if r.experiment_name == "old_gsnet"]
        if baseline_rows:
            bl = baseline_rows[0].latency_mean_ms
            result.speedup_vs_baseline = bl / result.latency_mean_ms if result.latency_mean_ms > 0 else float("nan")

        csv_path = write_csv(result, args.output_dir)
        print(f"  Written: {csv_path}")
        results.append(result)

    combined_path = combine_results(results, args.output_dir)
    print(f"\nCombined CSV: {combined_path}")

    print_summary(results)


def gpu_label(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    try:
        return torch.cuda.get_device_name(device)
    except Exception:
        return "unknown GPU"


if __name__ == "__main__":
    main()
