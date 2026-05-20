---
name: GS-Distill project overview
description: Dynamic GS-Distill pipeline for open-vocabulary semantic segmentation
type: project
---

GS-Distill replaces GSNet's expensive DINOv3 image stream with a lightweight student while
leaving GSNet/RIPD's normal dynamic text-conditioned fusion path intact.

The active student does not predict fixed LD50K class slots. It predicts only text-independent
DINO substitute tensors:

- `dino_down`: `(B, 768, 24, 24)` for RIPD's DINO correlation branch.
- `dino_L4`: `(B, 768, 48, 48)` for DINO decoder guidance.
- `dino_L8`: `(B, 768, 48, 48)` for DINO decoder guidance.

At inference/fine-tune time, the pipeline calls normal `ripd(...)` with image features,
student-predicted DINO substitutes, decoder guidance, and the active dataset text features.
RIPD computes CLIP/DINO correlation, fusion, aggregation, and decoding dynamically, so logits
naturally have `T = len(current_dataset_classes)`.

Active training entry points:

- `scripts/train_clip.py`: CLIP ViT-L dynamic distill + LD50K fine-tune.
- `scripts/train_baseline.py`: CLIP dynamic architecture from random student init.
- `scripts/train_siglip.py`: SigLIP image-backbone variant using the same dynamic RIPD path.
- `scripts/train_tips.py`: TIPSv2 image-backbone variant using the same dynamic RIPD path.

Removed legacy pieces:

- `scripts/ripd_lite.py`
- old two-script distill/fine-tune SLURM jobs
- fixed 40-slot class padding/slicing in CLIP eval
- cached fixed-fusion dataset helper

Do not touch the real RIPD implementation in `gs_net/modeling/transformer/RIPD.py` for this
cleanup. Its internal dynamic correlation variable names are part of GSNet itself.
