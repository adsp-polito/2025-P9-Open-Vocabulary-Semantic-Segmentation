# Technical Report — Open-Vocabulary Remote Sensing Semantic Segmentation
## Architecture, Distillation Pipeline, and Evaluation

---

## 1. Teacher Model — GSNet

GSNet is the dual-stream teacher model we distill from. It was already trained and its weights are fixed throughout our work.

### Architecture
- **Two visual backbones running in parallel:**
  - **CLIP ViT-L/14@336px** — general vision-language backbone pretrained on internet-scale image-text pairs. Processes images at 336×336 resolution, produces patch tokens at 24×24 spatial resolution.
  - **DINOv3 ViT-L/16** — domain-specialist backbone pretrained via self-supervised learning on SAT-493M (a 493M remote sensing image dataset). Processes images at 384×384, produces patch tokens at 48×48, then downsampled to 24×24.
- **QGFF (Query-Guided Feature Fusion)** — fuses the two streams using query-guided attention so CLIP's semantic understanding guides DINOv3's spatial features.
- **RIPD (Region-Informed Pixel Decoder)** — a complex multi-scale decoder that uses intermediate features from both CLIP and DINOv3 at multiple resolutions (layer 4, layer 8, final) as guidance signals. Upsamples from 24×24 to 96×96.
- **Text encoder** — CLIP's text transformer encodes dataset-specific class names into 1024-dim embeddings. These are compared against visual features via dot product to produce per-class correlation maps.
- **Loss** — binary cross-entropy (BCE) per class, not softmax. Each class is treated as an independent binary classifier. This matters for distillation.
- **Total size** — approximately 860M parameters.
- **Training** — supervised on LandDiscover 50K (LD50K), a 50K remote sensing image dataset with 40 semantic classes.
- **Test-time inference** — sliding window: the image is tiled into overlapping 384×384 patches, each patch is processed independently, outputs are folded back and averaged with a global forward pass. This gives high-resolution predictions on large aerial images.

### GSNet Results (sliding window, Detectron2 eval)
| Dataset | mIoU |
|---|---|
| FloodNet | 42.63% |
| FAST | 16.61% |
| Potsdam | 45.75% |
| FLAIR | 20.00% |
| **Average** | **31.25%** |

---

## 2. Student Architecture

We design a lightweight student that:
1. Does not require DINOv3 (removes one entire backbone)
2. Uses a class-agnostic decoder (works for any number of classes at test time)
3. Is 16× smaller in trainable parameters

### Components

#### 2.1 Backbone — TIPSv2-L/14
- A CLIP-like model fine-tuned specifically for improved patch-level text alignment using the TIPS training objective.
- Same ViT-L/14 architecture as CLIP — 487.9M parameters total.
- Provides both a vision encoder (`encode_image`) and a text encoder (`encode_text`).
- Outputs patch tokens of shape (B, 576, 1024) → reshaped to (B, 1024, 24, 24) for spatial processing.
- **Most of TIPS is kept frozen** during training. Only the last 4 transformer blocks (~50.4M params) are unfrozen in Phase 3.

#### 2.2 SpatialAdapter — 1.117M parameters
```
Input:  (B, 1024, 24, 24)
Conv2d(1024 → 256, 1×1) → GroupNorm(8) → GELU
Conv2d(256  → 256, 3×3) → GroupNorm(8) → GELU
Conv2d(256  → 1024, 1×1)
Output: x + net(x)   ← residual connection
```
- A bottleneck residual convolution block inspired by CLIP-DINOiser.
- Injected after TIPS's patch tokens to add DINOv3-like spatial coherence (local smoothness, boundary awareness) to CLIP's patch features, without requiring DINOv3 at runtime.
- Trained in Phase 2 (distillation), then carried forward into Phase 3.

#### 2.3 SpatialProjector — 65.7K parameters
```
Input:  (B, 1024, 24, 24)   ← L2-normalised adapter output
Conv2d(1024 → 64, 1×1) → GroupNorm(8) → GELU
Output: (B, 64, 24, 24)
```
- Compresses the 1024-dim spatial features into a 64-dim context map.
- Applied once per image (not per class).
- Provides texture and boundary signal to the decoder.
- Trained from scratch in Phase 3.

#### 2.4 ClassAgnosticDecoder — 153.9K parameters
```
Input:  (B×K, 65, 24, 24)   ← 1 correlation channel + 64 spatial context channels
Conv2d(65 → 128, 3×3)   → GroupNorm(8) → GELU      [24×24]
ConvTranspose2d(128→64, 2×2, stride=2)→ GroupNorm → GELU   [48×48]
Conv2d(64 → 64, 3×3)    → GroupNorm(8) → GELU      [48×48]
ConvTranspose2d(64→32, 2×2, stride=2) → GroupNorm → GELU   [96×96]
Conv2d(32 → 1, 3×3)                                [96×96]
Output: (B×K, 1, 96×96) → reshape → (B, K, 96, 96)
```
- Processes each class correlation map independently through shared weights.
- K (number of classes) never appears in any weight tensor → works for any dataset with any number of classes at inference time.
- This is a key difference from GSNet's RIPD decoder, which is deeply coupled to the training class set and requires multi-scale CLIP+DINOv3 feature guidance.
- Trained from scratch in Phase 3.

### Full Forward Pass
```
image (B, 3, 336, 336)
  ↓  TIPS vision encoder (last 4 blocks unfrozen in Phase 3)
patch_tokens (B, 576, 1024)  →  reshape  →  feats (B, 1024, 24, 24)
  ↓  SpatialAdapter
feats (B, 1024, 24, 24)  →  L2-normalise
  ↓  SpatialProjector
spatial_ctx (B, 64, 24, 24)                ← context for decoder

  ↓  einsum with K text embeddings (each 1024-dim)
corr (B, K, 24, 24)                         ← per-class similarity maps

  ↓  tile spatial_ctx K times → (B×K, 64, 24, 24)
  ↓  concatenate with corr_flat (B×K, 1, 24, 24)
dec_input (B×K, 65, 24, 24)

  ↓  ClassAgnosticDecoder
logits (B, K, 96, 96)   →  upsample to original size  →  argmax  →  prediction
```

### Trainable Parameter Summary
| Component | Params | Phase trained |
|---|---|---|
| TIPS last 4 blocks | 50.4M | Phase 3 only |
| SpatialAdapter | 1.117M | Phase 2 + Phase 3 |
| SpatialProjector | 65.7K | Phase 3 only |
| ClassAgnosticDecoder | 153.9K | Phase 3 only |
| **Total trainable (Phase 3)** | **51.74M** | |

---

## 3. Three-Phase Training Pipeline

### Phase 1 — Cache Teacher Logits

**Purpose:** Run GSNet once on all training images and save the output logits. This avoids running GSNet (860M params) during student training, making training feasible.

**Why cache instead of running GSNet online during training?**
The alternative — online distillation — would mean running GSNet's forward pass on every training batch alongside the student. This is impractical for three reasons:
**memory**: GSNet (~860M params) and the student (~490M params) cannot both fit in GPU memory simultaneously on the 11GB 2080 Ti cards used in this project. 
**speed**: GSNet uses a sliding window at inference time, processing each image as multiple overlapping 384×384 tiles and combining them, making a single forward pass expensive; doing this on every batch across 25+ epochs would make training prohibitively long. 
**redundancy**: since GSNet is fully frozen and deterministic, its output for a given image never changes during training — running it repeatedly on the same images wastes computation. By caching once in Phase 1, we pay the GSNet inference cost exactly once per image and reuse those logits across all Phase 2 and Phase 3 training epochs at zero additional cost. The trade-off is that cached logits are static — they do not adapt as the student evolves — which is the defining characteristic of offline distillation as opposed to online distillation.

**What is cached:**
- For every image in LD50K training set (51,846 images): the raw pre-sigmoid logits from GSNet's output.
- Shape per image: (40, 96, 96) — 40 classes × 96×96 spatial resolution.
- Stored in fp16 to save disk space (~38MB per 1000 images).
- Verified range: [-17.8, +8.5] — raw logits, valid, no overflow.
- Location: `cache_logits/`

**What we did NOT cache — intermediate GSNet representations:**

GSNet produces many intermediate tensors during its forward pass that we chose not to cache:

| Intermediate tensor | Shape | Why not cached |
|---|---|---|
| CLIP patch tokens | (B, 576, 1024) at 24×24 | Student has no CLIP stream to consume them |
| DINOv3 patch tokens | (B, 2304, 768) at 48×48 → (B, 768, 24×24) | Student has no DINOv3 stream |
| QGFF fused features | (B, 768, 24×24) | Architecture-specific; incompatible with student |
| RIPD guidance (res3/res4/res5 from CLIP, L4/L8 from DINOv3) | Various | Multi-scale; no corresponding student layers |

Because the student is a fundamentally different architecture (single backbone, no fusion, no multi-scale decoder), none of GSNet's internal representations are directly consumable by the student. The only meaningful signal we can transfer is the final output — the per-class logits.

**Cache tensor sizes:**

The cached final logit tensor per image has shape **(40, 96, 96)** in fp16:
```
40 classes × 96 × 96 pixels × 2 bytes (fp16) = 737,280 bytes ≈ 0.70 MB (raw tensor)
```
Stored as a PyTorch `.pt` file with metadata overhead: **~1.43 MB per image**.

For the full LD50K training set of 51,846 images:
```
51,846 × 1.43 MB ≈ 72 GB  (measured on disk)
```
This is the cost of Phase 1 — a one-time 72 GB write to avoid running GSNet repeatedly during training.

**Key point:** These are raw logits, not probabilities. The student loss applies sigmoid at training time: `sigmoid(teacher_logits / τ)`.

---

### Phase 2 — Distillation (Adapter Training)

**Goal:** Train the SpatialAdapter to shape TIPS's patch features to resemble what GSNet's CLIP stream would produce for the same image.

**What is trained:** Only the SpatialAdapter (1.117M params). TIPS backbone is fully frozen.

**Loss — Temperature-Scaled BCE:**
```
L_distill = BCE(student_logits / τ,  sigmoid(teacher_logits / τ)) × τ²
```
- τ = 4.0 (temperature)
- The τ² factor compensates for the scaled gradient magnitude so effective learning rate stays consistent.
- **Why BCE and not KL divergence?** GSNet was trained with binary cross-entropy (each class independent), not softmax. The teacher's logits are therefore independent binary scores, not a probability distribution. KL divergence requires a proper distribution (summing to 1). Using BCE matches the teacher's training objective and treats each class correctly.

**Decoder in Phase 2:** A simple `LightweightDecoder` (K=40 fixed) was used only to produce the student's logits for the loss. This decoder is discarded after Phase 2 — only the SpatialAdapter weights carry forward.

**Results:**
- 36 epochs on LD50K
- Best validation loss: 2.5904 (at epoch 36)
- Loss curve: steep improvement in first 10 epochs, then near-plateau (adapter hit capacity limit with frozen backbone)
- Checkpoint: `output/distill/student_best_TIPS.pth`

---

### Phase 3 — GT Fine-Tuning (Mixed Loss)

**Goal:** Fine-tune the full student pipeline on ground truth labels from LD50K, while continuing to distill from cached teacher logits.

**What is trained:**
- SpatialAdapter — loaded from Phase 2 (distilled run) or random init (scratch ablation)
- SpatialProjector — always random init
- ClassAgnosticDecoder — always random init
- Last 4 TIPS transformer blocks — unfrozen

**Loss — Mixed:**
```
L = λ × L_distill  +  (1 - λ) × L_gt
L_distill = temperature-scaled BCE vs cached teacher logits (same as Phase 2)
L_gt      = CrossEntropy vs GT masks (40 classes, resized to 96×96)
λ = 0.5
```

**Training details:**
- 25 epochs, batch size 8, AMP (mixed precision)
- LR: 1×10⁻⁴ for adapter/projector/decoder, 1×10⁻⁵ for unfrozen TIPS blocks
- Gradient clipping: max norm = 1.0
- Data augmentation: random horizontal and vertical flips (training split only)
- Val split: 2,592 images held out (5% of dataset), train: 49,254

**Two parallel runs for ablation:**

| Run | Adapter init | Purpose |
|---|---|---|
| **Distilled** | Phase 2 checkpoint | Full 3-phase pipeline |
| **Scratch** | Random | Ablation — skips Phase 2 |

Both runs are otherwise identical. The comparison proves whether Phase 2 distillation contributes.

**Results:**
| | Distilled | Scratch |
|---|---|---|
| Best epoch | 24 | 20 |
| Best val loss | 1.4307 | 1.4287 |
| Time | ~8h40m | ~8h31m |

---

## 4. Decoder Design Comparison

| | **GSNet RIPD** | **Our ClassAgnosticDecoder** |
|---|---|---|
| Input features | Multi-scale CLIP (res3/4/5) + DINOv3 (L4, L8, final) | Single correlation map + projected spatial context |
| Number of classes | Fixed at training time (40) | Any number at test time (class-agnostic) |
| Architecture | Multi-scale attention + convolutional | Pure convolutional |
| Parameters | Millions (complex) | 153.9K |
| Output resolution | 96×96 → upsampled | 24×24 → 96×96 |
| Requires DINOv3 at test time | Yes | No |

The class-agnostic design is the key architectural decision: by processing each class's correlation map independently through shared weights, K disappears from the weight tensors. This is what allows the student to generalise to any test dataset with any class set.

---

## 5. Evaluation Protocol

### Student Evaluation
1. Load image, resize to 336×336 (bilinear), normalise to [0, 1].
2. TIPS vision encoder → patch tokens → reshape to (1, 1024, 24, 24).
3. SpatialAdapter → L2-normalise → SpatialProjector → spatial context.
4. Compute text embeddings for dataset's class names via TIPS text encoder.
5. Einsum → correlation maps → tile + concat → ClassAgnosticDecoder → logits (1, K, 96, 96).
6. Bilinear upsample to original image resolution → argmax → predicted class map.
7. Compute mIoU with ignore label (dataset-specific).

### GSNet Simple Evaluation (for fair comparison)
1. Load image at original resolution, pass as Detectron2 `batched_inputs` with `height` and `width` set to original dimensions.
2. GSNet normalises internally (clip_pixel_mean/std in [0,255] scale).
3. GSNet internally resizes: CLIP stream to 336×336, DINOv3 stream to 384×384.
4. `TEST.SLIDING_WINDOW = False` — no tiling, single forward pass.
5. `sem_seg_postprocess` upsamples output to original resolution.
6. Argmax → predicted class map → mIoU with same ignore labels.

**Why we ran GSNet under our protocol:** The original GSNet results use Detectron2's sliding window evaluation. To isolate how much of the gap is due to evaluation method vs model quality, we re-ran GSNet with a single-pass protocol identical in spirit to the student eval.

---

## 6. Results

### Full Comparison Table

| Dataset | GSNet (sliding) | GSNet (simple) | Student-Distilled | Student-Scratch |
|---|---|---|---|---|
| FloodNet | 42.63% | 41.27% | 9.03% | 9.57% |
| FAST     | 16.61% | 17.41% | 12.44% | 13.45% |
| Potsdam  | 45.75% | 25.67% | 21.32% | 18.95% |
| FLAIR    | 20.00% | 18.00% | 14.70% | 17.50% |
| **Avg**  | **31.25%** | **25.59%** | **14.37%** | **14.87%** |

---

## 7. Analysis and Discussion

### 7.1 Evaluation Protocol Gap is Dataset-Dependent
- **FloodNet** and **FAST**: GSNet-sliding ≈ GSNet-simple. The sliding window adds almost nothing here (~0–0.8%).
- **Potsdam**: GSNet drops from 45.75% → 25.67% without sliding window (−20%). Potsdam images are high-resolution ortho-photos where fine spatial detail is critical. The sliding window is essential there.
- **Implication:** Even under the same single-pass protocol, GSNet still outperforms the student by ~11% on average (25.59% vs ~14.6%). The gap is genuine, not just protocol artefact.

### 7.2 Distillation Did Not Clearly Help
The scratch model matches or slightly exceeds the distilled model on 3 out of 4 datasets. This is a surprising and important negative result.

**Possible explanations:**
- The SpatialAdapter (1.117M params) may be too small to meaningfully capture GSNet's spatial structure from distillation alone. Phase 2 val loss plateaued early (epoch ~10), suggesting the adapter reached its capacity limit with TIPS frozen.
- 25 epochs of GT fine-tuning with unfrozen TIPS blocks may be sufficient to erase any advantage from Phase 2, since the backbone itself adapts.
- The distillation target (GSNet's logits) may not encode the right inductive bias for the student's architecture (no DINOv3 dual stream).


### 7.3 Model Size vs Performance
| | GSNet | Our Student |
|---|---|---|
| Total params | ~860M | ~490M (mostly frozen) |
| Trainable params (Phase 3) | ~860M | 51.74M |
| Average mIoU (same protocol) | 25.59% | ~14.6% |
| Compression ratio (trainable) | 1× | **16.6×** |

The student achieves ~57% of GSNet's mIoU (same protocol) with 16× fewer trainable parameters and no DINOv3 dependency.

### 7.4 Open Questions for Discussion
1. **Would a larger adapter or more unfrozen TIPS blocks close the gap?** The current setup unfreezes only 4 of 24 blocks. Unfreezing more might improve performance at the cost of overfitting risk.
2. **Is the class-agnostic decoder the bottleneck?** 153.9K parameters upsampling from 24×24 is very constrained. A more powerful decoder might help more than distillation.
3. **Could online distillation (running GSNet during training, not caching) improve Phase 2?** Cached logits are fixed and do not respond to the student's evolution.
4. **Is the TIPS backbone actually better suited than CLIP for this task?** TIPS was fine-tuned for patch-text alignment, which may or may not benefit remote sensing.
5. **Potsdam anomaly:** Why does Potsdam drop so much under simple eval for both GSNet and the student? Is it purely image resolution, or is there something about the class distribution that sliding window helps with?

---

*Generated from training runs completed 2026-05-16. Checkpoints at `output_v2/finetune/{distilled,scratch}/student_best.pth`. GSNet simple eval at `results_v2/gsnet_simple/`.*
