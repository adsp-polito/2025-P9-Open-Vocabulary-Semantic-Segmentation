"""
Phase 4 (optional) — End-to-end fine-tune of student + RIPD on LD50K segmentation labels.

Loads:
  - trained student conv head (from distillation checkpoint)
  - frozen RIPD from GSNet checkpoint (optionally unfrozen)

Trains end-to-end on LD50K with cross-entropy segmentation loss.
CLIP stays frozen throughout.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'   # still needed to load GSNet; RSIB stays frozen

    python scripts/train_finetune.py \\
        --gsnet-config    configs/vitb_384.yaml \\
        --gsnet-weights   path/to/gsnet_checkpoint.pth \\
        --student-ckpt    output/distill/student_best.pth \\
        --image-dir       path/to/LD50K/images \\
        --label-dir       path/to/LD50K/labels \\
        --class-json      datasets/landdiscover.json \\
        --output-dir      output/finetune/ \\
        [--epochs 10] \\
        [--batch-size 8] \\
        [--lr 1e-5] \\
        [--unfreeze-ripd] \\
        [--amp] \\
        [--device cuda]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

import clip as openai_clip
import wandb

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_distill.student import GSDistillStudent
from gs_distill.utils import get_clip_skips
from einops import rearrange as _rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation dataset
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """
    Flat-folder RGB image + single-channel label dataset.
    Label pixels are class indices in [0, num_classes-1]; 255 = ignore.
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384,
                 clip_mean=(0.48145466, 0.4578275, 0.40821073),
                 clip_std=(0.26862954, 0.26130258, 0.27577711)):
        self.resolution = resolution
        self.clip_mean = clip_mean
        self.clip_std  = clip_std

        img_stems = {
            p.stem for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
        lbl_stems = {
            p.stem for p in Path(label_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
        stems = sorted(img_stems & lbl_stems)

        self.samples = []
        for stem in stems:
            for ext in (".png", ".jpg", ".tif"):
                ip = Path(image_dir) / (stem + ext)
                lp = Path(label_dir) / (stem + ext)
                if ip.exists() and lp.exists():
                    self.samples.append((str(ip), str(lp)))
                    break

        if not self.samples:
            raise RuntimeError(f"No matching image/label pairs in {image_dir} / {label_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.clip_mean, std=self.clip_std)

        lbl = Image.open(lbl_path)
        lbl = TF.resize(lbl, (self.resolution, self.resolution), interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--student-ckpt",   required=True)
    p.add_argument("--image-dir",      required=True)
    p.add_argument("--label-dir",      required=True)
    p.add_argument("--class-json",     default="datasets/landdiscover.json")
    p.add_argument("--output-dir",     default="output/finetune")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--unfreeze-ripd",          action="store_true")
    p.add_argument("--warmup-decoder-epochs",  type=int, default=0,
                   help="Train RIPD decoder alone for N epochs, then unfreeze student heads.")
    p.add_argument("--grad-accum",     type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum).")
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction",   type=float, default=0.05)
    p.add_argument("--data-fraction",  type=float, default=1.0,
                   help="Fraction of dataset to use (e.g. 0.5 = 50%%). Fixed-seed random subset.")
    p.add_argument("--data-seed",      type=int,   default=42)
    p.add_argument("--wandb-project",  default="gs-distill")
    p.add_argument("--wandb-run",      default=None, help="W&B run name (auto if omitted)")
    p.add_argument("--no-wandb",       action="store_true", help="Disable W&B logging")
    return p.parse_args()


def build_gsnet(config_file, weights_file, device):
    from gs_net import add_cat_seg_config
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    from detectron2.modeling import build_model as d2_build
    model = d2_build(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()
    return model


def build_text_features(class_json, clip_model, device):
    with open(class_json) as f:
        class_names = json.load(f)
    templates = ["a photo of a {}."]
    all_feats = []
    with torch.no_grad():
        for name in class_names:
            texts = [t.format(name) for t in templates]
            tokens = openai_clip.tokenize(texts).to(device)
            feats = clip_model.encode_text(tokens).float()  # (P, C)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
    # (T, P, C) → (1, T, P, C) batch dim
    text_feats = torch.stack(all_feats, dim=0).unsqueeze(0)
    return text_feats   # (1, T, P, C)


def create_subset(dataset, fraction, seed):
    from torch.utils.data import Subset
    from collections import defaultdict

    rng = np.random.RandomState(seed)
    n = len(dataset)

    print(f"Building stratification index for {n} images ...")

    # Pass 1: collect the set of classes present in each image and global class frequency
    image_classes = []
    class_freq = defaultdict(int)
    for _, lbl_path in dataset.samples:
        lbl = np.array(Image.open(lbl_path))
        present = set(lbl[lbl != 255].tolist())
        image_classes.append(present)
        for c in present:
            class_freq[c] += 1

    # Pass 2: assign each image to its rarest globally-present class so that
    # every class that appears anywhere in the dataset becomes its own stratum.
    strata = []
    for present in image_classes:
        if present:
            stratum = min(present, key=lambda c: class_freq[c])
        else:
            stratum = -1
        strata.append(stratum)

    # Sample proportionally from each stratum
    stratum_buckets = defaultdict(list)
    for idx, s in enumerate(strata):
        stratum_buckets[s].append(idx)

    selected = []
    for s, idxs in sorted(stratum_buckets.items()):
        k = max(1, round(len(idxs) * fraction))
        chosen = rng.choice(idxs, min(k, len(idxs)), replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    print(f"Dataset subset (stratified): {len(selected)} / {n} images ({fraction*100:.0f}%)")
    print(f"  Strata covered: {len(stratum_buckets)} classes")
    return Subset(dataset, selected)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load GSNet (to extract CLIP, RIPD + projection layers) ──────────────
    print("Loading GSNet ...")
    gsnet = build_gsnet(args.gsnet_config, args.gsnet_weights, str(device))
    gsnet = gsnet.to(device)

    # Reuse the teacher's fine-tuned CLIP — correct architecture (ViT-L/14@336)
    # and the attention tuning from GSNet pretraining.
    clip_model = gsnet.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Layer indices for CLIP skip extraction depend on architecture
    # (ViT-B/16: [3,7], ViT-L/14@336: [7,15]) — read directly from the loaded model.
    clip_skip_indices = tuple(gsnet.layer_indexes)

    # forward_from_fusion lives on the RIPD transformer, not on the predictor wrapper.
    ripd            = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1  = gsnet.upsample1
    clip_upsample2  = gsnet.upsample2
    dino_decod_proj1 = gsnet.dino_decod_proj1
    dino_decod_proj2 = gsnet.dino_decod_proj2

    # RSIB and backbone are not used during fine-tune — free VRAM before training.
    gsnet.dino_model = None
    gsnet.backbone = None
    torch.cuda.empty_cache()

    warmup        = args.warmup_decoder_epochs
    ripd_unfrozen = args.unfreeze_ripd or warmup > 0

    for p in ripd.parameters():
        p.requires_grad = ripd_unfrozen

    decoder_proj_modules = [m for m in [clip_upsample1, clip_upsample2,
                                         dino_decod_proj1, dino_decod_proj2]
                             if m is not None]
    for m in decoder_proj_modules:
        for p in m.parameters():
            p.requires_grad = ripd_unfrozen

    # ── Load student ─────────────────────────────────────────────────────────
    print(f"Loading student from {args.student_ckpt} ...")
    ckpt = torch.load(args.student_ckpt, map_location=device)
    student_args = ckpt.get("args", {})
    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=student_args.get("hidden_dim", 128),
        d_dino=student_args.get("d_dino", 768),
        num_classes=student_args.get("num_classes", 40),
        clip_layers=student_args.get("clip_layers", [4, 8, 10, 12]),
    ).to(device)
    student.load_state_dict(ckpt["student"])

    if warmup > 0:
        # Phase 1: student fully frozen while the decoder warms up alone.
        for p in student.parameters():
            p.requires_grad = False
        print(f"  RIPD decoder unfrozen. Student heads frozen for {warmup}-epoch warmup.")
    elif ripd_unfrozen:
        print("  RIPD unfrozen for fine-tuning.")

    # ── Text features (frozen) ────────────────────────────────────────────────
    text_feats = build_text_features(args.class_json, clip_model, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = SegDataset(args.image_dir, args.label_dir)
    if args.data_fraction < 1.0:
        full_ds = create_subset(full_ds, args.data_fraction, args.data_seed)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"  Train: {n_train}  Val: {n_val}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # shared_trunk + the three task heads; clip_embed_branch stays frozen
    # (distillation-only auxiliary branch, not used in inference).
    def _student_head_params():
        return (
            list(student.shared_trunk.parameters())
            + list(student.fusion_branch.parameters())
            + list(student.dino_l4_branch.parameters())
            + list(student.dino_l8_branch.parameters())
        )

    def _decoder_proj_params():
        return [p for m in decoder_proj_modules for p in m.parameters()]

    if warmup > 0:
        active_params = list(ripd.parameters()) + _decoder_proj_params()
    else:
        active_params = _student_head_params()
        if ripd_unfrozen:
            active_params += list(ripd.parameters()) + _decoder_proj_params()

    optimizer  = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler     = GradScaler(enabled=args.amp)
    ignore_idx = 255

    best_val_loss = float("inf")

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        # ── Warmup transition: unfreeze student heads after warmup epochs ─────
        if warmup > 0 and epoch == warmup:
            for p in _student_head_params():
                p.requires_grad = True
            active_params = _student_head_params() + list(ripd.parameters()) + _decoder_proj_params()
            optimizer = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
            print(f"  Epoch {epoch+1}: student heads unfrozen — joint fine-tuning begins.")

        student_training = (warmup == 0) or (epoch >= warmup)
        student.train() if student_training else student.eval()
        if ripd_unfrozen:
            ripd.train()

        train_loss = 0.0
        n_train_batches = 0
        grad_accum = args.grad_accum
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.expand(B, -1, -1, -1)   # (B, T, P, C)

            # ── student + CLIP skips (cheap, under autocast) ──────────────────
            with autocast(enabled=args.amp):
                with torch.no_grad():
                    clip_features, clip_skips = get_clip_skips(
                        clip_model, images, list(clip_skip_indices))
                    l0, l1 = clip_skip_indices
                    res4 = clip_upsample1(clip_skips[l0]) if clip_upsample1 is not None else None
                    res5 = clip_upsample2(clip_skips[l1]) if clip_upsample2 is not None else None

                if student.training:
                    student_out  = grad_ckpt.checkpoint(student, images, use_reentrant=False)
                else:
                    with torch.no_grad():
                        student_out = student(images)
                fused_corr_embed = student_out["fused_corr_embed"]
                dino_L4_proj     = dino_decod_proj1(student_out["dino_L4"]) if dino_decod_proj1 is not None else None
                dino_L8_proj     = dino_decod_proj2(student_out["dino_L8"]) if dino_decod_proj2 is not None else None
                # Free the full student_out dict — only the extracted tensors are needed
                del student_out

            clip_guidance = {
                "res3": _rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24),
                "res4": res4,
                "res5": res5,
            }
            # Free raw CLIP skips — clip_guidance holds the only tensors needed downstream
            del clip_features, clip_skips

            # ── RIPD decoder: checkpoint to avoid retaining (B*T,C,H,W) graph ─
            _has_dg0 = dino_L4_proj is not None
            _has_dg1 = dino_L8_proj is not None
            _dg0 = dino_L4_proj if _has_dg0 else fused_corr_embed.new_zeros(1)
            _dg1 = dino_L8_proj if _has_dg1 else fused_corr_embed.new_zeros(1)

            # -- inline forward_from_fusion: guidance projections (cheap, no checkpoint) --
            with autocast(enabled=args.amp):
                _res3 = clip_guidance["res3"]
                _res4 = clip_guidance["res4"]
                _res5 = clip_guidance["res5"]
                _projected_guidance = ripd.guidance_projection(_res3) if ripd.guidance_projection is not None else None
                if ripd.text_guidance_projection is not None:
                    _tf_mean = tf.mean(dim=-2)
                    _tf_mean = _tf_mean / _tf_mean.norm(dim=-1, keepdim=True)
                    _projected_text_guidance = ripd.text_guidance_projection(_tf_mean)
                else:
                    _projected_text_guidance = None
                if ripd.CLIP_decoder_guidance_projection is not None:
                    _CLIP_proj = [proj(g) for proj, g in zip(ripd.CLIP_decoder_guidance_projection, [_res4, _res5]) if g is not None]
                    while len(_CLIP_proj) < 2:
                        _CLIP_proj.append(None)
                else:
                    _CLIP_proj = [None, None]
                if ripd.DINO_decoder_guidance_projection is not None:
                    _DINO_proj = [proj(g) for proj, g in zip(ripd.DINO_decoder_guidance_projection, [dino_L4_proj, dino_L8_proj]) if g is not None]
                    while len(_DINO_proj) < 2:
                        _DINO_proj.append(None)
                else:
                    _DINO_proj = [None, None]

            # -- per-AggregatorLayer checkpointing: each layer's ~11 MB freed before next --
            _pg  = _projected_guidance      if _projected_guidance      is not None else fused_corr_embed.new_zeros(1)
            _ptg = _projected_text_guidance if _projected_text_guidance is not None else fused_corr_embed.new_zeros(1)
            _has_pg  = _projected_guidance      is not None
            _has_ptg = _projected_text_guidance is not None

            for _layer in ripd.layers:
                def _agg(_fn, _x, _pg, _ptg):
                    with autocast(enabled=args.amp):
                        return _fn(_x, _pg if _has_pg else None, _ptg if _has_ptg else None)
                fused_corr_embed = grad_ckpt.checkpoint(_agg, _layer, fused_corr_embed, _pg, _ptg, use_reentrant=False)

            def _decoder(_x):
                with autocast(enabled=args.amp):
                    return ripd.Fusion_conv_decoer(_x, _CLIP_proj, _DINO_proj)
            logit = grad_ckpt.checkpoint(_decoder, fused_corr_embed, use_reentrant=False)

            # logit: (B, T, H, W)  labels: (B, H, W)
            with autocast(enabled=args.amp):
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                         mode="bilinear", align_corners=False)
                del logit
                loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx) / grad_accum
                del logit_up

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_last_step = (step + 1 == len(train_loader))
            if (step + 1) % grad_accum == 0 or is_last_step:
                if args.amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(active_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(active_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * grad_accum
            n_train_batches += 1

        # Validation
        student.eval()
        if ripd_unfrozen:
            ripd.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]
                tf = text_feats.expand(B, -1, -1, -1)
                with autocast(enabled=args.amp):
                    logit = gs_distill_inference(
                        image=images, text_feats=tf, student=student,
                        clip_model=clip_model, ripd=ripd,
                        clip_upsample1=clip_upsample1, clip_upsample2=clip_upsample2,
                        dino_decod_proj1=dino_decod_proj1, dino_decod_proj2=dino_decod_proj2,
                        clip_skip_layer_indices=clip_skip_indices,
                    )
                    logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                             mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)
                val_loss += loss.item()
                n_val_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val   = val_loss   / max(n_val_batches,   1)
        phase     = "warmup" if (warmup > 0 and epoch < warmup) else "joint"
        print(f"Epoch {epoch+1:3d}/{args.epochs}  [{phase}]  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            wandb.log({
                "epoch":       epoch + 1,
                "train/loss":  avg_train,
                "val/loss":    avg_val,
                "phase":       phase,
                "best_val_loss": best_val_loss,
            }, step=epoch + 1)

        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "ripd": ripd.state_dict() if ripd_unfrozen else None,
            "clip_upsample1":   clip_upsample1.state_dict()  if ripd_unfrozen and clip_upsample1  is not None else None,
            "clip_upsample2":   clip_upsample2.state_dict()  if ripd_unfrozen and clip_upsample2  is not None else None,
            "dino_decod_proj1": dino_decod_proj1.state_dict() if ripd_unfrozen and dino_decod_proj1 is not None else None,
            "dino_decod_proj2": dino_decod_proj2.state_dict() if ripd_unfrozen and dino_decod_proj2 is not None else None,
            "val_loss": avg_val,
        }, os.path.join(args.output_dir, "finetune_latest.pth"))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.output_dir, "finetune_best.pth")
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "ripd": ripd.state_dict() if ripd_unfrozen else None,
                "clip_upsample1":   clip_upsample1.state_dict()  if ripd_unfrozen and clip_upsample1  is not None else None,
                "clip_upsample2":   clip_upsample2.state_dict()  if ripd_unfrozen and clip_upsample2  is not None else None,
                "dino_decod_proj1": dino_decod_proj1.state_dict() if ripd_unfrozen and dino_decod_proj1 is not None else None,
                "dino_decod_proj2": dino_decod_proj2.state_dict() if ripd_unfrozen and dino_decod_proj2 is not None else None,
                "val_loss": avg_val,
            }, best_path)
            print(f"  ✓ New best val loss: {avg_val:.4f} → {best_path}")

    if use_wandb:
        wandb.finish()
    print(f"\nFine-tuning complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
