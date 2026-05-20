# scripts/ - GS-Distill Training Pipeline

This folder contains the active GS-Distill training, evaluation, and SLURM entry points.
The active design no longer predicts fixed LD50K class-correlation slots.

## Current Architecture

GS-Distill keeps GSNet/RIPD's normal dynamic text-conditioned path:

```text
image backbone features + student DINO substitutes + text features
    -> RIPD.forward(...)
    -> dynamic CLIP/DINO correlations
    -> RIPD fusion/aggregators/decoder
    -> logits with T = current vocabulary size
```

The student predicts only text-independent DINO substitute tensors:

| Target | Teacher source | Shape |
|---|---|---|
| `dino_down` | `GSNet.dino_down_sample(DINO last layer)` | `(B, 768, 24, 24)` |
| `dino_L4` | DINOv3 intermediate layer 4 | `(B, 768, 48, 48)` |
| `dino_L8` | DINOv3 intermediate layer 8 | `(B, 768, 48, 48)` |

RIPD computes the correlation/fusion tensors internally for the active text features. Do not
reintroduce `forward_from_fusion`, fixed 40-slot padding, `ripd_lite`, RIPD layer truncation,
or class-chunked decoder patches in the active training path.

## Active Scripts

- `train_clip.py`: CLIP ViT-L/14@336 dynamic DINO-substitute distill + LD50K fine-tune.
- `train_baseline.py`: same CLIP dynamic architecture, random student init, LD50K fine-tune only.
- `train_siglip.py`: SigLIP image backbone variant, dynamic RIPD path.
- `train_tips.py`: TIPSv2 image backbone variant, dynamic RIPD path.
- `eval_clip.py`, `eval_siglip.py`, `eval_tips.py`: zero-shot eval on Potsdam, FloodNet, FLAIR, and FAST.
- `eval_finetune.py`: LD50K-focused CLIP checkpoint eval.

The active SLURM jobs are:

- `phase1_gsnet_pretrain.sbatch`
- `clip_distill_finetune.sbatch`
- `baseline_finetune.sbatch`
- `siglip_distill_finetune.sbatch`
- `tips_distill_finetune.sbatch`
- `eval_clip.sbatch`, `eval_baseline.sbatch`, `eval_siglip.sbatch`, `eval_tips.sbatch`

The old two-script jobs and `ripd_lite.py` were removed. Use the combined train scripts above.

## Backbone Notes

CLIP uses the GSNet checkpoint's CLIP image and text encoders. Default student layers are
`8 16 20 23`, and decoder skip indices come from `gsnet.layer_indexes` (`7 15` for ViT-L).

SigLIP and TIPS use their own image backbones, but text features are still built with the
teacher CLIP text encoder because RIPD is trained in that embedding space. Their decoder skip
indices are backbone-local (`3 7` by default). If the GSNet decoder bridge expects a different
skip width, the non-CLIP students include small 1x1 skip adapters.

TIPSv2 produces a native 32x32 patch grid; the active path resizes TIPS features to RIPD's
24x24 feature grid before correlation/fusion.

## Checkpoints

Distillation checkpoints save:

```python
{
    "epoch": int,
    "student": student.state_dict(),
    "val_loss": float,
    "args": dict,
}
```

Fine-tune checkpoints save:

```python
{
    "epoch": int,
    "student": student.state_dict(),
    "ripd": state_dict | None,
    "clip_upsample1": state_dict | None,
    "clip_upsample2": state_dict | None,
    "dino_decod_proj1": state_dict | None,
    "dino_decod_proj2": state_dict | None,
    "val_loss": float,
    "args": dict,
}
```

Old fixed-fusion checkpoints are incompatible with the active dynamic student heads.
