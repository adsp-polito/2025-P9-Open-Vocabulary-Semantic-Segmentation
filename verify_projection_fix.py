#!/usr/bin/env python3
"""
Verification script to check if DINOv3 projection layer fix is working.
Run this after training starts to verify the projection is trainable.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('./gs_net'))

from gs_net.GSNet import GSNet
from detectron2.config import get_cfg
from gs_net.config import add_cat_seg_config

def check_projection_trainability():
    """Check if DINOv3 dim_projection is trainable."""

    print("="*80)
    print("DINOv3 Projection Layer Verification")
    print("="*80)

    # Load config
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file("configs/rn101_224.yaml")

    # Build model
    print("\nBuilding model...")
    model = GSNet(cfg).to('cpu')

    # Check DINOv3 parameters
    print("\nChecking DINOv3 parameters:")
    print("-" * 80)

    trainable_params = 0
    frozen_params = 0
    projection_trainable = False

    for name, param in model.dino_model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            if "dim_projection" in name:
                projection_trainable = True
                print(f"✓ TRAINABLE: {name} - shape {param.shape}")
        else:
            frozen_params += param.numel()
            if "dim_projection" in name:
                print(f"✗ FROZEN: {name} - shape {param.shape}")

    print("-" * 80)
    print(f"\nTotal DINOv3 parameters:")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {frozen_params:,}")

    # Verify projection initialization
    print("\nChecking projection initialization:")
    proj_weight = model.dino_model.dim_projection.weight.data
    print(f"  Shape: {proj_weight.shape}")
    print(f"  Mean: {proj_weight.mean().item():.6f}")
    print(f"  Std: {proj_weight.std().item():.6f}")

    # Check if identity-like
    identity_block = proj_weight[:, :768]
    identity_similarity = (identity_block - 0.9 * torch.eye(768)).abs().mean().item()
    print(f"  Identity similarity (lower=better): {identity_similarity:.6f}")

    print("\n" + "="*80)
    if projection_trainable:
        print("✓✓✓ SUCCESS: dim_projection is TRAINABLE")
    else:
        print("✗✗✗ ERROR: dim_projection is FROZEN (training will fail!)")
    print("="*80)

    return projection_trainable

if __name__ == "__main__":
    try:
        success = check_projection_trainability()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
