"""
ripd_lite.py — memory-efficient forward_from_fusion for GS-Distill fine-tuning.

This module is NOT a replacement class. It monkey-patches the forward_from_fusion
method on the already-loaded RIPD instance extracted from GSNet. The RIPD weights
and all sub-modules remain exactly as loaded from the checkpoint — only the Python
method dispatched during fine-tuning is swapped.

Changes vs original forward_from_fusion (RIPD.py lines 295–359):

1. pad_len bypass — ClassTransformerLayer pads T=40 → 256 and holds that tensor
   for backward. We force pad_len=0 on all AggregatorLayer attention sub-modules
   at patch time, eliminating the ~6 MB padding allocation per layer per backward.

2. Decoder split — Fusion_conv_decoer runs Fusiondecoder1 + Fusiondecoder2 + head
   as one unit. During backward recompute the 96×96 output of Fusiondecoder2
   (22.5 MB) is allocated while Fusiondecoder1's input (11.25 MB) is still live
   = 33.75 MB peak. We split them into two separate checkpointed calls so each
   stage's input is freed before the next recomputes.

3. AggregatorLayer checkpointing — each of the 2 layers is checkpointed
   individually so their Swin window activations (~11 MB each) are freed before
   the next layer recomputes.

Usage (in each train_*.py at the start of run_finetune / main):

    from scripts.ripd_lite import patch_ripd
    patch_ripd(ripd, args)         # args supplies .amp for autocast

After this call ripd.forward_from_fusion is replaced. Everything else (weights,
Fusiondecoder1/2, layers, head, projections) is untouched.
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from torch.cuda.amp import autocast
from einops import rearrange
import types


def _disable_padding(ripd):
    """
    Set pad_len=0 on every ClassTransformerLayer inside ripd.layers.
    This prevents the T=40 → T=256 padding allocation in ClassTransformerLayer.forward.
    The padding was only needed to stabilise training of the teacher (GSNet) and
    provides no benefit during student fine-tuning where T is fixed at 40.
    """
    for agg_layer in ripd.layers:
        ctl = agg_layer.attention          # ClassTransformerLayer
        ctl.pad_len = 0
        ctl.padding_tokens = None
        ctl.padding_guidance = None


def _make_forward_from_fusion(ripd, amp: bool):
    """
    Build and return the replacement forward_from_fusion bound method.
    Uses closure over ripd and amp so it needs no extra arguments at call time.
    """

    def forward_from_fusion(self, fused_corr_embed, text_feats, appearance_guidance, dino_guidance):
        # ── Normalise appearance_guidance ──────────────────────────────────────
        if isinstance(appearance_guidance, dict):
            res3 = appearance_guidance.get("res3")
            res4 = appearance_guidance.get("res4")
            res5 = appearance_guidance.get("res5")
            app_guid_tuple = (res3, res4, res5)
        else:
            app_guid_tuple = appearance_guidance

        # ── Guidance projections (cheap, no checkpoint needed) ─────────────────
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

        # ── Per-AggregatorLayer checkpointing ─────────────────────────────────
        # Sentinel tensors: checkpoint() cannot receive None arguments.
        _pg  = projected_guidance      if projected_guidance      is not None else fused_corr_embed.new_zeros(1)
        _ptg = projected_text_guidance if projected_text_guidance is not None else fused_corr_embed.new_zeros(1)
        _has_pg  = projected_guidance      is not None
        _has_ptg = projected_text_guidance is not None

        for _layer in self.layers:
            def _agg(_fn, _x, _pg, _ptg, _has_pg=_has_pg, _has_ptg=_has_ptg):
                with autocast(enabled=amp):
                    return _fn(_x, _pg if _has_pg else None, _ptg if _has_ptg else None)
            fused_corr_embed = grad_ckpt.checkpoint(_agg, _layer, fused_corr_embed, _pg, _ptg, use_reentrant=False)

        # ── Decoder: two individually checkpointed stages ──────────────────────
        _B = fused_corr_embed.shape[0]
        _c0, _c1 = CLIP_proj[0], CLIP_proj[1]
        _d0, _d1 = DINO_proj[0], DINO_proj[1]
        _has_c0 = _c0 is not None;  _has_c1 = _c1 is not None
        _has_d0 = _d0 is not None;  _has_d1 = _d1 is not None
        _c0s = _c0 if _has_c0 else fused_corr_embed.new_zeros(1)
        _c1s = _c1 if _has_c1 else fused_corr_embed.new_zeros(1)
        _d0s = _d0 if _has_d0 else fused_corr_embed.new_zeros(1)
        _d1s = _d1 if _has_d1 else fused_corr_embed.new_zeros(1)

        def _dec1(_x, _cg, _dg, _has_c0=_has_c0, _has_d0=_has_d0):
            with autocast(enabled=amp):
                _x = rearrange(_x, 'B C T H W -> (B T) C H W')
                return self.Fusiondecoder1(_x, _cg if _has_c0 else None, _dg if _has_d0 else None)
        _flat = grad_ckpt.checkpoint(_dec1, fused_corr_embed, _c0s, _d0s, use_reentrant=False)

        def _dec2(_x, _cg, _dg, _has_c1=_has_c1, _has_d1=_has_d1):
            with autocast(enabled=amp):
                return self.Fusiondecoder2(_x, _cg if _has_c1 else None, _dg if _has_d1 else None)
        _flat = grad_ckpt.checkpoint(_dec2, _flat, _c1s, _d1s, use_reentrant=False)

        with autocast(enabled=amp):
            _flat = self.head(_flat)
            logit = rearrange(_flat, '(B T) () H W -> B T H W', B=_B)

        return logit

    return forward_from_fusion


def patch_ripd(ripd, args):
    """
    Monkey-patch ripd.forward_from_fusion with the memory-efficient version
    and disable ClassTransformerLayer padding.

    Args:
        ripd: the RIPD instance from gsnet.sem_seg_head.predictor.transformer
        args: training args namespace — must have .amp (bool)
    """
    _disable_padding(ripd)
    new_method = _make_forward_from_fusion(ripd, amp=args.amp)
    ripd.forward_from_fusion = types.MethodType(new_method, ripd)
    print(f"[ripd_lite] Patched forward_from_fusion on RIPD. pad_len=0, split decoder, per-layer ckpt.")
