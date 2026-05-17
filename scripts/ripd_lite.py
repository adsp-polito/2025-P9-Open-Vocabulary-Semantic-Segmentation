"""
ripd_lite.py - memory-efficient forward_from_fusion for GS-Distill fine-tuning.

This module is NOT a replacement class. It monkey-patches the forward_from_fusion
method on the already-loaded RIPD instance extracted from GSNet. The RIPD weights
and all sub-modules remain exactly as loaded from the checkpoint; only the Python
method dispatched during fine-tuning is swapped.

Changes vs original forward_from_fusion (RIPD.py lines 295-359):

1. pad_len bypass - ClassTransformerLayer pads T=40 to 256 and holds that tensor
   for backward. We force pad_len=0 on all AggregatorLayer attention sub-modules
   at patch time, eliminating the padding allocation per layer per backward.

2. Decoder split - Fusion_conv_decoer runs Fusiondecoder1 + Fusiondecoder2 + head
   as one unit. We split them into two separately checkpointed calls so each
   stage's input can be released before the next stage recomputes.

3. AggregatorLayer checkpointing - each layer is checkpointed individually so
   their Swin/window activations are freed before the next layer recomputes.

4. Optional class-streamed RIPD decoding - if --ripd-decoder-class-chunk-size
   is set, run Fusiondecoder1/2/head on class groups instead of all T classes at
   once. This is enabled only after checking that Fusiondecoder1/2 contain
   per-sample spatial ops, so splitting the flattened (B*T) batch preserves
   model behavior except for tiny floating-point differences.

Usage (in each train_*.py at the start of run_finetune / main):

    from scripts.ripd_lite import patch_ripd
    patch_ripd(ripd, args)         # args supplies .amp for autocast

After this call ripd.forward_from_fusion is replaced. Everything else (weights,
Fusiondecoder1/2, layers, head, projections) is untouched.
"""

import types

import torch
import torch.nn as nn
import torch.utils.checkpoint as grad_ckpt
from einops import rearrange
from torch.cuda.amp import autocast


_MB = 1024 ** 2
_CLASS_INDEPENDENT_DECODER_LEAVES = (
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.GroupNorm,
    nn.ReLU,
    nn.Identity,
)


def log_cuda_memory(label: str, reset_peak: bool = True):
    """
    Print current and peak CUDA allocation, then optionally reset peak tracking.
    This is a no-op on CPU-only runs.
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / _MB
    reserved = torch.cuda.memory_reserved() / _MB
    peak = torch.cuda.max_memory_allocated() / _MB
    print(
        f"[ripd_lite][mem] {label}: "
        f"allocated={allocated:.1f} MiB  reserved={reserved:.1f} MiB  peak={peak:.1f} MiB"
    )
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


def reset_cuda_memory_peak():
    """Reset CUDA peak allocation tracking. This is a no-op on CPU-only runs."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _disable_padding(ripd):
    """
    Set pad_len=0 on every ClassTransformerLayer inside ripd.layers.
    This prevents the T=40 to T=256 padding allocation in ClassTransformerLayer.
    """
    for agg_layer in ripd.layers:
        ctl = agg_layer.attention
        ctl.pad_len = 0
        ctl.padding_tokens = None
        ctl.padding_guidance = None


def _truncate_aggregator_layers(ripd, keep_layers: int):
    total_layers = len(ripd.layers)

    if keep_layers <= 0:
        return

    if keep_layers >= total_layers:
        print(f"[ripd_lite] Keeping all {total_layers} RIPD AggregatorLayers")
        return

    ripd.layers = nn.ModuleList(list(ripd.layers)[:keep_layers])
    ripd.num_layers = keep_layers
    print(f"[ripd_lite] Using first {keep_layers}/{total_layers} RIPD AggregatorLayers")


def _decoder_tail_class_independent(ripd):
    """
    Verify that Fusiondecoder1/2 do not mix flattened class samples.

    The RIPD decoder tail flattens classes into the batch axis:
    B,C,T,H,W -> (B*T),C,H,W. Conv/GroupNorm/ReLU spatial ops are per sample, so
    decoding smaller class chunks is mathematically equivalent. BatchNorm,
    attention, Linear layers over a reshaped class axis, or unknown leaves would
    make chunking behavior-changing, so they are rejected.
    """
    for decoder_name in ("Fusiondecoder1", "Fusiondecoder2"):
        decoder = getattr(ripd, decoder_name, None)
        if decoder is None:
            return False, f"{decoder_name} is missing."

        if decoder.__class__.__name__ != "FusionUP":
            return (
                False,
                f"{decoder_name} is {decoder.__class__.__name__}, not the expected FusionUP decoder.",
            )

        for module_name, module in decoder.named_modules():
            if module is decoder or len(list(module.children())) > 0:
                continue
            if isinstance(module, _CLASS_INDEPENDENT_DECODER_LEAVES):
                continue
            full_name = f"{decoder_name}.{module_name}" if module_name else decoder_name
            return (
                False,
                f"{full_name} is {module.__class__.__name__}; class chunking only supports "
                "per-sample Conv2d/ConvTranspose2d/GroupNorm/ReLU spatial ops.",
            )

    return True, "Fusiondecoder1/2 use only per-sample spatial ops."


def _make_forward_from_fusion(
    ripd,
    amp: bool,
    decoder_class_chunk_size: int = 0,
    memory_logging: bool = True,
):
    """
    Build and return the replacement forward_from_fusion bound method.
    Uses closure over ripd and amp so it needs no extra arguments at call time.
    """

    def _maybe_log(label: str):
        if memory_logging:
            log_cuda_memory(label)

    def forward_from_fusion(self, fused_corr_embed, text_feats, appearance_guidance, dino_guidance):
        _maybe_log("before RIPD forward")

        # Normalize appearance_guidance to tuple form.
        if isinstance(appearance_guidance, dict):
            res3 = appearance_guidance.get("res3")
            res4 = appearance_guidance.get("res4")
            res5 = appearance_guidance.get("res5")
            app_guid_tuple = (res3, res4, res5)
        else:
            app_guid_tuple = appearance_guidance

        # Guidance projections are cheap enough to keep outside checkpoints.
        with autocast(enabled=amp):
            projected_guidance = (
                self.guidance_projection(app_guid_tuple[0])
                if self.guidance_projection is not None and app_guid_tuple[0] is not None
                else None
            )

            if self.text_guidance_projection is not None:
                tf = text_feats.mean(dim=-2)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                projected_text_guidance = self.text_guidance_projection(tf)
            else:
                projected_text_guidance = None

            if self.CLIP_decoder_guidance_projection is not None:
                clip_skips = app_guid_tuple[1:]
                CLIP_proj = [
                    proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, clip_skips)
                    if g is not None
                ]
                while len(CLIP_proj) < 2:
                    CLIP_proj.append(None)
            else:
                CLIP_proj = [None, None]

            if self.DINO_decoder_guidance_projection is not None and dino_guidance is not None:
                DINO_proj = [
                    proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)
                    if g is not None
                ]
                while len(DINO_proj) < 2:
                    DINO_proj.append(None)
            else:
                DINO_proj = [None, None]

        # Per-AggregatorLayer checkpointing.
        _pg = projected_guidance if projected_guidance is not None else fused_corr_embed.new_zeros(1)
        _ptg = (
            projected_text_guidance
            if projected_text_guidance is not None
            else fused_corr_embed.new_zeros(1)
        )
        _has_pg = projected_guidance is not None
        _has_ptg = projected_text_guidance is not None

        for _layer in self.layers:
            def _agg(_fn, _x, _pg, _ptg, _has_pg=_has_pg, _has_ptg=_has_ptg):
                with autocast(enabled=amp):
                    return _fn(_x, _pg if _has_pg else None, _ptg if _has_ptg else None)

            fused_corr_embed = grad_ckpt.checkpoint(
                _agg,
                _layer,
                fused_corr_embed,
                _pg,
                _ptg,
                use_reentrant=False,
            )

        _maybe_log("after aggregator")

        _B = fused_corr_embed.shape[0]
        _T = fused_corr_embed.shape[2]

        _c0, _c1 = CLIP_proj[0], CLIP_proj[1]
        _d0, _d1 = DINO_proj[0], DINO_proj[1]
        _has_c0 = _c0 is not None
        _has_c1 = _c1 is not None
        _has_d0 = _d0 is not None
        _has_d1 = _d1 is not None
        _c0s = _c0 if _has_c0 else fused_corr_embed.new_zeros(1)
        _c1s = _c1 if _has_c1 else fused_corr_embed.new_zeros(1)
        _d0s = _d0 if _has_d0 else fused_corr_embed.new_zeros(1)
        _d1s = _d1 if _has_d1 else fused_corr_embed.new_zeros(1)

        def _dec1(_x, _cg, _dg, _has_c0=_has_c0, _has_d0=_has_d0):
            with autocast(enabled=amp):
                return self.Fusiondecoder1(_x, _cg if _has_c0 else None, _dg if _has_d0 else None)

        def _dec2(_x, _cg, _dg, _has_c1=_has_c1, _has_d1=_has_d1):
            with autocast(enabled=amp):
                return self.Fusiondecoder2(_x, _cg if _has_c1 else None, _dg if _has_d1 else None)

        def _decode_flat(_flat_in, _chunk_label=""):
            _flat = grad_ckpt.checkpoint(_dec1, _flat_in, _c0s, _d0s, use_reentrant=False)
            _maybe_log(f"after Fusiondecoder1{_chunk_label}")
            del _flat_in

            _flat = grad_ckpt.checkpoint(_dec2, _flat, _c1s, _d1s, use_reentrant=False)
            _maybe_log(f"after Fusiondecoder2{_chunk_label}")
            return _flat

        if decoder_class_chunk_size > 0 and decoder_class_chunk_size < _T:
            # Class-streamed RIPD decoding: Fusiondecoder1/2 are class-wise after
            # the B*T flatten, so process smaller T groups and restore B,T order.
            logit_chunks = []
            for _start in range(0, _T, decoder_class_chunk_size):
                _end = min(_start + decoder_class_chunk_size, _T)
                _chunk_t = _end - _start
                _flat_in = rearrange(
                    fused_corr_embed[:, :, _start:_end],
                    "B C T H W -> (B T) C H W",
                )
                _flat = _decode_flat(_flat_in, f" chunk {_start}:{_end}")
                with autocast(enabled=amp):
                    _flat = self.head(_flat)
                    logit_chunks.append(
                        rearrange(_flat, "(B T) () H W -> B T H W", B=_B, T=_chunk_t)
                    )
                del _flat
                if memory_logging:
                    reset_cuda_memory_peak()

            del fused_corr_embed
            logit = torch.cat(logit_chunks, dim=1)
        else:
            # Rearrange BEFORE checkpoint so the checkpoint input is already flat.
            # Doing it inside _dec1 would create two copies of fused_corr_embed
            # simultaneously during recompute.
            _flat_in = rearrange(fused_corr_embed, "B C T H W -> (B T) C H W")
            del fused_corr_embed

            _flat = _decode_flat(_flat_in)
            with autocast(enabled=amp):
                _flat = self.head(_flat)
                logit = rearrange(_flat, "(B T) () H W -> B T H W", B=_B)

        return logit

    return forward_from_fusion


def patch_ripd(ripd, args):
    """
    Monkey-patch ripd.forward_from_fusion with the memory-efficient version
    and disable ClassTransformerLayer padding.

    Args:
        ripd: the RIPD instance from gsnet.sem_seg_head.predictor.transformer.
        args: training args namespace. .amp is optional; .ripd_decoder_class_chunk_size
            enables class-streamed RIPD decoding when > 0.
    """
    _disable_padding(ripd)

    agg_layers = int(getattr(args, "ripd_agg_layers", 0) or 0)
    _truncate_aggregator_layers(ripd, agg_layers)

    chunk_size = int(getattr(args, "ripd_decoder_class_chunk_size", 0) or 0)
    if chunk_size < 0:
        raise ValueError("--ripd-decoder-class-chunk-size must be >= 0")

    if chunk_size > 0:
        ok, reason = _decoder_tail_class_independent(ripd)
        if not ok:
            raise RuntimeError(
                "Cannot enable class-streamed RIPD decoding because Fusiondecoder1/2 "
                f"may mix classes globally: {reason} Chunking would change model behavior."
            )
        print(f"[ripd_lite] Class-streamed RIPD decoding enabled: chunk_size={chunk_size}. {reason}")

    new_method = _make_forward_from_fusion(
        ripd,
        amp=bool(getattr(args, "amp", False)),
        decoder_class_chunk_size=chunk_size,
        memory_logging=bool(
            getattr(args, "ripd_memory_logging", hasattr(args, "ripd_decoder_class_chunk_size"))
        ),
    )
    ripd.forward_from_fusion = types.MethodType(new_method, ripd)
    print(
        "[ripd_lite] Patched forward_from_fusion on RIPD. "
        "pad_len=0, split decoder, per-layer ckpt."
    )
