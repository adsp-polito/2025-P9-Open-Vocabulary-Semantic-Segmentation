# ViT-L/14@336px + DINOv3 Configuration Guide

## Overview

This setup uses **ViT-L/14@336px** (CLIP generalist) with **DINOv3 ViT-L/16** (specialist) for open-vocabulary remote sensing segmentation.

## Architecture Compatibility

### Dual-Stream Feature Dimensions

| Component | CLIP ViT-L/14@336px | DINOv3 ViT-L/16 |
|-----------|---------------------|-----------------|
| **Patch Size** | 14×14 | 16×16 |
| **Image Resolution** | 336×336 | 384×384 (resized) |
| **Feature Grid** | 24×24 | 48×48 → 24×24 (downsampled) |
| **Image Encoder Dim** | **1024** | **768** (projected from 1024) |
| **Text Encoder Dim** | 768 | N/A |
| **Layer Depth** | 24 layers | 24 layers |
| **Intermediate Layers** | [7, 15] | [3, 7] |

### How Feature Fusion Works

**Query-Guided Fusion** elegantly handles the dimension mismatch:

```python
# Step 1: Normalize and compute correlations separately
clip_corr = correlation(clip_features_1024d, text_features_768d)  # Shape: (B, P, T, H, W)
dino_corr = correlation(dino_features_768d, text_features_768d)   # Shape: (B, P, T, H, W)

# Step 2: Embed correlations separately
clip_embed = conv1(clip_corr)  # → (B, C, T, H, W)
dino_embed = conv2(dino_corr)  # → (B, C, T, H, W)

# Step 3: Fuse embedded correlations
fused = fusion_conv(cat([clip_embed, dino_embed], dim=1))
```

**Key insight:** Features are correlated with text *before* fusion, so different dimensions don't matter!

## Configuration File

Use the config: `configs/vitl_336_dinov3.yaml`

### Key Settings

```yaml
# CLIP Model
CLIP_PRETRAINED: 'ViT-L/14@336px'
CLIP_FINETUNE: 'attention'  # QV fine-tuning
CLIP_MULTIPLIER: 0.01        # Small learning rate

# DINOv3 Model
USE_DINO_CORR: True
DINO_FINETUNE: 'attention'   # QV fine-tuning
BACKBONE_MULTIPLIER: 0.0     # Frozen backbone

# Fusion Strategy
FUSION_TYPE: 'query_guided'  # Handles dimension mismatch
USE_CLIP_CORR: True
```

## Environment Setup

### 1. Set DINOv3 Checkpoint Path

```bash
export RSIB_CKPT="./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
```

### 2. Verify Checkpoint Exists

```bash
ls -lh $RSIB_CKPT
# Should show: ~1GB file
```

## Training Command

```bash
# Standard training
sh scripts/train.sh configs/vitl_336_dinov3.yaml 4 ./outputs/vitl_dinov3

# If you encounter OOM (Out of Memory):
# 1. Reduce batch size in config: IMS_PER_BATCH: 2
# 2. Or use gradient accumulation
```

## Evaluation Command

```bash
sh scripts/eval.sh configs/vitl_336_dinov3.yaml 4 ./outputs/vitl_dinov3
```

## Memory Requirements

| Configuration | VRAM Usage (est.) | Recommendation |
|---------------|-------------------|----------------|
| Batch size 4 | ~22-24 GB | RTX 3090 / A5000+ |
| Batch size 2 | ~12-14 GB | RTX 3080 / A4000+ |
| Batch size 1 | ~7-8 GB | RTX 3070+ |

### Memory Optimization Tips

1. **Reduce batch size:**
   ```yaml
   SOLVER:
     IMS_PER_BATCH: 2  # Instead of 4
   ```

2. **Use mixed precision training** (if implemented):
   ```bash
   # Add to training script
   --fp16
   ```

3. **Disable CLIP guidance if needed:**
   ```yaml
   DECODER_CLIP_GUIDANCE_DIMS: [0, 0]  # Saves ~2GB VRAM
   ```

## Expected Performance

### Advantages of ViT-L vs ViT-B

- **Better semantic understanding** from larger model
- **Finer spatial details** with 14×14 patches
- **Stronger open-vocabulary capabilities**
- **Higher resolution** input (336×336)

### Trade-offs

- **Slower training** (~1.5x compared to ViT-B/16)
- **Higher VRAM usage** (~2x compared to ViT-B/16)
- **Dimension mismatch handled** by query-guided fusion ✓

## Debugging

### Check Model Loads Correctly

```python
import torch
from gs_net.third_party import clip

# ViT-L will auto-download if not cached
model, preprocess = clip.load("ViT-L/14@336px", device="cuda", jit=False)
print(f"Visual encoder output dim: {model.visual.output_dim}")  # Should be 1024
print(f"Text encoder output dim: {model.ln_final.normalized_shape[0]}")  # Should be 768
```

### Verify Feature Shapes During Training

Add to GSNet.py after line 307:
```python
print(f"CLIP features shape: {clip_features.shape}")  # (B, 577, 1024) for 336px
print(f"DINO features shape: {dino_feat[-1].shape}")  # (B, 2305, 768)
```

## Architecture Flow

```
Input Image (384×384)
    │
    ├─→ Resize to 336×336 ─→ CLIP ViT-L/14@336px
    │                         │
    │                         ├─→ Image Features (B, 577, 1024)
    │                         │    ├─→ Layer 7  (intermediate)
    │                         │    └─→ Layer 15 (intermediate)
    │                         │
    │                         └─→ Text Features (B, T, 768)
    │
    └─→ Resize to 384×384 ─→ DINOv3 ViT-L/16
                              │
                              └─→ Image Features (B, 2305, 768)
                                   ├─→ Layer 3 (intermediate)
                                   └─→ Layer 7 (intermediate)

                    ↓ Query-Guided Fusion ↓

        Correlation with Text → Fused Embeddings
                    ↓
        Residual Information Preservation Decoder
                    ↓
        Segmentation Masks (B, C, H, W)
```

## Citation

If you use this configuration, cite both GSNet and the pretrained models:

```bibtex
@inproceedings{ye2025GSNet,
  title={Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation},
  author={Ye, Chengyang and Zhuge, Yunzhi and Zhang, Pingping},
  booktitle={AAAI},
  year={2025}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"
**Solution:** Reduce `IMS_PER_BATCH` to 2 or 1

### Issue: "FileNotFoundError: RSIB_CKPT"
**Solution:** Download DINOv3 checkpoint and set environment variable

### Issue: "Model downloading very slow"
**Solution:** Models auto-download to `~/.cache/clip/`. Pre-download with:
```bash
python -c "from gs_net.third_party import clip; clip.load('ViT-L/14@336px')"
```

### Issue: "Different mIoU from paper"
**Expected:** Small variations due to:
- Different CLIP model (ViT-L vs paper's ViT-B or RN101)
- DINOv3 vs DINOv1
- Hyperparameter differences

## Next Steps

1. ✓ Config file created: `configs/vitl_336_dinov3.yaml`
2. ✓ DINOv3 checkpoint set: `$RSIB_CKPT`
3. ⏳ Run training
4. ⏳ Evaluate on FloodNet
5. ⏳ Compare with ViT-B/16 baseline
