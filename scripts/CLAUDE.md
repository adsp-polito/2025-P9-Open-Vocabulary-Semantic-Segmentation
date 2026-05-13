# scripts/ — GS-Distill Training Pipeline

This folder contains every training and evaluation script for the GS-Distill project.
The pipeline investigates whether knowledge distillation from a pretrained remote-sensing
vision-language model (GSNet) can produce a lighter student that still performs open-vocabulary
semantic segmentation on satellite imagery.

---

## The Big Picture

### Why distillation?

GSNet is a large dual-stream teacher: a fine-tuned CLIP ViT for vision-language alignment and
a frozen DINOv3 RSIB (Remote Sensing Image Backbone) for spatial detail. Running both streams
at inference is expensive. The goal is to train a smaller *student* that mimics what the teacher's
internal representations look like, then plug that student into the same RIPD decoder at test time.

### What the student learns

The teacher exposes four internal tensors per batch that capture the richest information in the
model. The student is trained to predict all four simultaneously:

| Target | Source inside teacher | Shape |
|---|---|---|
| `fused_corr_embed` | CLIP+DINO correlation after QGFF, entering RIPD | `(B, hidden, T, 24, 24)` |
| `clip_embed_corr` | CLIP-only correlation before DINO fusion | `(B, hidden, T, 24, 24)` |
| `dino_L4` | DINOv3 intermediate layer 4 | `(B, 768, 48, 48)` |
| `dino_L8` | DINOv3 intermediate layer 8 | `(B, 768, 48, 48)` |

These targets are captured via **monkey-patching** (not hooks) in `train_distill_online.py`:
- `_patch_ripd()` wraps `RIPD.forward` to intercept the tensors right after QGFF and stash them
  in `_hook_state` before calling the original forward.
- `_patch_dino()` wraps `dino_model.get_intermediate_layers` to extract layers 4 and 8 as a
  side-effect of the single DINOv3 forward that the teacher already performs.

Both patches are undone at the end of training by calling the returned `unpatch()` functions.

### Student architecture (GSDistillStudent)

Defined in `gs_distill/student.py`. Takes multi-scale CLIP features as input and predicts all
four teacher targets through a shared trunk and four specialist branches:

- **shared_trunk**: small conv stack operating on the concatenated multi-layer CLIP features
- **fusion_branch**: predicts `fused_corr_embed`
- **clip_embed_branch**: predicts `clip_embed_corr` (auxiliary; only active during distillation)
- **dino_l4_branch**: predicts `dino_L4`
- **dino_l8_branch**: predicts `dino_L8`

CLIP itself is always **frozen**. The student reuses the teacher's already-fine-tuned CLIP weights
(extracted from the GSNet checkpoint via `gsnet.sem_seg_head.predictor.clip_model`) so no separate
attention fine-tuning is needed.

### Segmentation fine-tune (Phase 3)

After distillation the student's trunk+branches produce features that approximate the teacher's
internal state. Phase 3 plugs this into the frozen RIPD decoder and trains end-to-end on LD50K
with cross-entropy against ground-truth segmentation labels. Only the student's three inference
branches (fusion, dino_l4, dino_l8) and optionally RIPD's parameters are active — the
`clip_embed_branch` is distillation-only and stays frozen at this stage.

The call path during fine-tune is `gs_distill_inference()` from `gs_distill/inference.py`, which
calls `student.forward_from_fusion()` → `ripd.forward_from_fusion()` → per-class logits.

### Evaluation

After fine-tuning each model variant is evaluated zero-shot on four benchmark datasets using
`SemSegEvaluator` from detectron2. Text features are built from each dataset's class list using
the **teacher's CLIP text encoder** (not the student's backbone), because RIPD was trained in
CLIP's embedding space and must stay there.

| Dataset | Split | Ignore label | Script target |
|---|---|---|---|
| Potsdam | test | 5 | `eval_clip.py`, `eval_siglip.py`, `eval_tips.py` |
| FloodNet | test | 0 | same |
| FLAIR | test | 12 | same |
| FAST | test | 255 | same |

---

## The Three Training Variants

All three variants share the same student architecture and RIPD decoder. They differ only in
which backbone drives the student's CLIP-side features.

### CLIP (original GS-Distill)

- Backbone: OpenAI `clip.load("ViT-L/14@336px")`, reused directly from the teacher checkpoint
- Layers extracted: `[4, 8, 10, 12]` (13-layer ViT-L has max index 12)
- Normalisation: CLIP standard (`mean=[0.481, 0.458, 0.408]`, `std=[0.269, 0.261, 0.276]`)
- Phase 2 → Phase 3 are **separate scripts** and separate jobs
  - Phase 2: `train_distill_online.py` → `output/distill/student_best.pth`
  - Phase 3: `train_finetune.py` (loads Phase 2 checkpoint) → `output/ashie/finetune/finetune_best.pth`
- Eval: `eval_clip.py`

### SigLIP

- Backbone: `google/siglip-base-patch16-384` via HuggingFace `transformers.AutoModel`
- Layers extracted: `[4, 8, 10, 11]` (12-layer ViT-B has max index 11; mirrors CLIP depth ratios)
- Layer path: `siglip_model.vision_model.encoder.layers[l]`, hook output `(B, seq, C)` — **no CLS
  token** in SigLIP, all 576 tokens are patches
- Normalisation: `mean=0.5, std=0.5` (SigLIP convention)
- Text features: `siglip_model.get_text_features()` via `AutoTokenizer`
- Resolution: 384×384, patch 16 → 24×24 grid
- Phase 2 + Phase 3 are **combined in one script** (`train_siglip.py`) and one job
  - Distill saves: `output/ashie/siglip/student_distill_best.pth`
  - Finetune saves: `output/ashie/siglip/finetune_best.pth`
- Eval: `eval_siglip.py` (text features still from teacher CLIP)

### TIPSv2

- Backbone: `google/tipsv2-b14` via HuggingFace `AutoModel(trust_remote_code=True)`
- Layers extracted: `[4, 8, 10, 11]` (12-layer ViT-B, same depth ratios)
- Layer path: `tips_model.vision_encoder.blocks[l]`, hook output `(B, seq, C)`
- Prefix tokens: 2 (CLS at index 0 + register at index 1); patch tokens start at index 2
- `vision_encoder(image)` returns a 3-tuple `(cls_token, register_tokens, patch_tokens)` in eval
  mode — use `patch_tokens` directly, do NOT call as a tensor
- Config: `config.embed_dim` (not `hidden_size`), `config.img_size` (not `image_size`)
- Normalisation: `mean=0.5, std=0.5`
- Text features: `tips_model.encode_text(list_of_strings)` — takes raw strings, no tokenizer step
- Resolution: 448×448, patch 14 → 32×32 grid = 1024 patch tokens
- Phase 2 + Phase 3 combined in one script (`train_tips.py`) and one job
  - Distill saves: `output/ashie/tips/student_distill_best.pth`
  - Finetune saves: `output/ashie/tips/finetune_best.pth`
- Eval: `eval_tips.py` (text features still from teacher CLIP)

### Baseline

- Same `GSDistillStudent` architecture as CLIP GS-Distill but with **randomly initialised**
  student weights — no distillation pre-training
- Purpose: ablation to measure how much the distillation step actually helps
- Script: `train_baseline.py` — goes straight to segmentation fine-tune on LD50K
- Output: `output/ashie/baseline/baseline_best.pth`
- Eval: `eval_clip.py` (same inference path as CLIP GS-Distill; RIPD and CLIP are identical)

---

## Submit Scripts and Their Job Chains

Each `.sh` wrapper calls `sbatch` and chains a dependent eval job with
`--dependency=afterok:<job_id>`. Eval only starts if training exits 0.

### `submit_distill.sh`

Covers Phase 1 (GSNet pretrain) and Phase 2 (student distillation). Phase 3 is launched
separately, either manually or by passing `--after <phase2_job_id>` to `submit_finetune.sh`.

```
phase1_gsnet_pretrain.sbatch   →   phase2_distill_online.sbatch
       (GSNet train)                   (train_distill_online.py)
                                       output/distill/student_best.pth
```

Flags:
- No args: submits both phases in sequence
- `--phase2-only`: skips Phase 1 (use when GSNet checkpoint already exists)

After Phase 2 finishes the script prints the exact command to launch Phase 3:
```
sh scripts/submit_finetune.sh --after <JOB2>
```

### `submit_finetune.sh`

Phase 3 for the original CLIP backbone, then auto-eval on 4 datasets.

```
phase3_finetune.sbatch    →   eval_clip.sbatch
 (train_finetune.py)           (eval_clip.py)
 output/ashie/finetune/        output/ashie/finetune/eval/
 finetune_best.pth
```

Flags:
- No args: submits immediately
- `--after <job_id>`: waits for `<job_id>` (Phase 2) before starting

Requires:
- `output/gsnet_pretrain/model_final.pth`
- `output/distill/student_best.pth`

### `submit_baseline.sh`

Baseline fine-tune (random init), then auto-eval on 4 datasets.

```
baseline_finetune.sbatch    →   eval_baseline.sbatch
  (train_baseline.py)             (eval_clip.py with baseline_best.pth)
  output/ashie/baseline/          output/ashie/baseline/eval/
  baseline_best.pth
```

Requires:
- `output/gsnet_pretrain/model_final.pth`

### `submit_siglip.sh`

SigLIP combined distill+finetune, then auto-eval on 4 datasets.

```
siglip_distill_finetune.sbatch    →   eval_siglip.sbatch
   (train_siglip.py)                     (eval_siglip.py)
   output/ashie/siglip/                  output/ashie/siglip/eval/
   student_distill_best.pth
   finetune_best.pth
```

Requires:
- `output/gsnet_pretrain/model_final.pth`
- HuggingFace `google/siglip-base-patch16-384` accessible at runtime (cached or mirrored)

### `submit_tips.sh`

TIPSv2 combined distill+finetune, then auto-eval on 4 datasets.

```
tips_distill_finetune.sbatch    →   eval_tips.sbatch
   (train_tips.py)                     (eval_tips.py)
   output/ashie/tips/                  output/ashie/tips/eval/
   student_distill_best.pth
   finetune_best.pth
```

Requires:
- `output/gsnet_pretrain/model_final.pth`
- HuggingFace `google/tipsv2-b14` accessible at runtime (`trust_remote_code=True`)

---

## Out-of-Folder Dependencies

These live **outside** `scripts/` and must be present for any script here to run.

### `gs_distill/` (package)

| Module | Used by | What it provides |
|---|---|---|
| `gs_distill/student.py` | `train_distill_online.py`, `train_finetune.py`, `train_baseline.py`, all eval scripts | `GSDistillStudent` class |
| `gs_distill/losses.py` | all distill scripts | `distillation_loss_per_branch(pred, targets)` — Smooth L1 + cosine per head |
| `gs_distill/inference.py` | `train_finetune.py`, `train_baseline.py`, `eval_clip.py`, `eval_siglip.py`, `eval_tips.py` | `gs_distill_inference()` — unified forward pass through student → RIPD → logits |

SigLIP and TIPSv2 scripts are self-contained and do **not** import from `gs_distill/` except
`losses.py` during Phase 2.

### `gs_net/` (package)

Provides `add_cat_seg_config` and the detectron2 model registration side-effects needed to
build the GSNet teacher. Every script imports `gs_net` at the top for this reason.

### `configs/vitl_336_dinov3.yaml`

The detectron2 config for the GSNet teacher. Hardcoded in every `.sbatch` file as
`--gsnet-config configs/vitl_336_dinov3.yaml`. Changing teacher architecture means
updating all `.sbatch` files.

### `output/gsnet_pretrain/model_final.pth`

The trained GSNet teacher checkpoint. Every script requires this. Produced by
`phase1_gsnet_pretrain.sbatch` (or a pre-existing model). All `.sbatch` files guard
against it being missing with an explicit `[[ -f ... ]]` check that exits 1.

### `gs_net/data/datasets/LandDiscover_50K/`

Training dataset. Two subdirectories used:
- `TR_Image/` — RGB satellite images at various resolutions (resized to backbone native res at runtime)
- `TR_Label/` — single-channel PNG label maps, class indices 0–39, 255 = ignore

The 40-class vocabulary (`CLASSES_LandDiscover50K`) is defined inline in each distillation
script and also referenced via `datasets/landdiscover.json` for text-feature building in the
fine-tune scripts.

### `gs_net/third_party/experiments/.../epoch_07.pth` (RSIB checkpoint)

DINOv3 remote sensing backbone weights. Must be exported as `RSIB_CKPT` before running any
script. Every `.sbatch` file sets this environment variable. Without it GSNet cannot be loaded.

### `detectron2/` (local submodule)

Used for `get_cfg`, `DetectionCheckpointer`, `DatasetCatalog`, `SemSegEvaluator`. Both
`sys.path.insert(0, './detectron2')` and `sys.path.insert(0, '.')` appear at the top of every
script to ensure the local copy is found before any system install.

### `datasets/landdiscover.json`

JSON list of 40 class name strings used by `build_text_features()` in fine-tune scripts to
tokenise prompts for CLIP text encoding. Default path; override with `--class-json`.

### `.env` (gitignored)

Sourced by every `.sbatch` file if present. Typical contents: `WANDB_API_KEY=...`. Safe to
omit if W&B logging is not needed (pass `--no-wandb` to any training script).

---

## SLURM Resource Defaults (all jobs)

| Setting | Value |
|---|---|
| Account | `intrn` |
| Partition | `RTX` (preemptable) |
| GPU | `gpu:2080ti:1` (11 GB VRAM) |
| Memory | 32 GB |
| Time limit — distillation jobs | 3 days |
| Time limit — finetune-only jobs | 2 days |
| Time limit — eval jobs | 6 hours |

All logs land in `jobs-ashie/logs/` with the SLURM job ID in the filename.

---

## Checkpoint Format Reference

All training scripts save dicts with a consistent structure. Eval scripts read these keys:

```python
# Phase 2 (distillation only)
{
    "epoch": int,
    "student": student.state_dict(),       # GSDistillStudent / SigLIPStudent / TIPSStudent
    "val_loss": float,
    "args": dict,                          # vars(args) — used by Phase 3 to rebuild student
}

# Phase 3 / baseline / SigLIP finetune / TIPSv2 finetune
{
    "epoch": int,
    "student": student.state_dict(),
    "ripd": state_dict | None,             # non-None only if --unfreeze-ripd was set
    "clip_upsample1": state_dict | None,
    "clip_upsample2": state_dict | None,
    "dino_decod_proj1": state_dict | None,
    "dino_decod_proj2": state_dict | None,
    "val_loss": float,
    "args": dict,
}
```

Eval scripts load each key with `ckpt.get(key)` and skip the corresponding module if `None`,
so checkpoints trained without `--unfreeze-ripd` still load correctly.
