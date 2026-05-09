---
name: GS-Distill project overview
description: What GS-Distill is doing — knowledge distillation pipeline for open-vocabulary semantic segmentation to replace expensive DINOv3 backbone with a lightweight student
type: project
---

GS-Distill is a 4-phase knowledge distillation pipeline that replaces the expensive DINOv3 (RSIB) backbone in GSNet with a lightweight student conv head, while preserving the RIPD decoder unchanged.

**Teacher model (GSNet)** fuses CLIP ViT-B/16 + DINOv3 features through QGFF inside RIPD to produce `fused_corr_embed`, plus DINOv3 skip features (layers 4 and 8) used in the decoder.

**Phase 1 — cache_teacher.py**: Runs frozen GSNet on all LD50K images, captures three tensors per image via monkey-patching: `fused_corr_embed` (hidden_dim × T × 24 × 24), `dino_L4` (768 × 48 × 48), `dino_L8` (768 × 48 × 48). Saves as `.pt` files.

**Phase 3 — train_distill.py**: Trains `GSDistillStudent` (gs_distill/student.py) to predict those three tensors from frozen CLIP multi-layer features only (layers 4, 8, 10, 12 → 4×768 → shared trunk → three branches). Loss = per-branch MSE + (1 − cosine_sim). No segmentation labels needed.

**Phase 4 — train_finetune.py** (optional): End-to-end fine-tune of student + frozen RIPD on LD50K with cross-entropy segmentation labels. Inference path (gs_distill/inference.py): student predicts `fused_corr_embed` + DINOv3 features → fed into `ripd.forward_from_fusion()`, bypassing QGFF/RSIB entirely.

**Why:** DINOv3 is expensive; CLIP is already being computed by the full model. Student learns to predict DINOv3's outputs from CLIP features that are already available, so inference cost drops significantly.

**Dataset:** LD50K = LandDiscover50K, a remote sensing segmentation dataset with 40 classes.
