"""
CLIP backbone — combined Phase 2 (online distillation) + Phase 3 (segmentation fine-tune).

Runs online DINO-substitute distillation and LD50K segmentation fine-tuning in one script.

Phase 2 (distillation) completes first, saves student_distill_best.pth, then
Phase 3 (fine-tune) starts automatically from that checkpoint.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/train_clip.py \\
        --gsnet-config  configs/vitl_336_dinov3.yaml \\
        --gsnet-weights path/to/gsnet.pth \\
        --image-dir     path/to/LD50K/TR_Image \\
        --label-dir     path/to/LD50K/TR_Label \\
        --output-dir    output/ashie/clip \\
        [--distill-epochs 30] [--finetune-epochs 15] \\
        [--batch-size 4] [--lr 1e-5] [--amp] [--wandb-project gs-distill]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

import wandb

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from gs_net import add_cat_seg_config
import gs_net  # noqa: side-effect registrations

from gs_distill.student import GSDistillStudent
from gs_distill.losses import distillation_loss_per_branch
from gs_distill.inference import gs_distill_inference
from gs_distill.utils import get_clip_skips


# ─────────────────────────────────────────────────────────────────────────────
# LD50K vocabulary
# ─────────────────────────────────────────────────────────────────────────────
CLASSES_LandDiscover50K = [
    'background', 'bare land', 'grass', 'pavement', 'road', 'tree', 'water',
    'agriculture land', 'buildings', 'forest land', 'barren land', 'urban land',
    'large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
    'plane', 'ship', 'soccer-ball-field', 'basketball-court',
    'ground-track-field', 'small-vehicle', 'baseball-diamond',
    'tennis-court', 'roundabout', 'storage-tank', 'harbor',
    'container-crane', 'airport', 'helipad', 'chimney',
    'expressway service area', 'expresswalltoll station', 'dam',
    'golf field', 'overpass', 'stadium', 'train station',
    'vehicle', 'windmill',
]

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def clip_normalise(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [0, 1] → CLIP-normalised."""
    mean = _CLIP_MEAN.to(images.device).view(1, 3, 1, 1)
    std  = _CLIP_STD.to(images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Teacher capture state (written by patches, read by training loop)
# ─────────────────────────────────────────────────────────────────────────────
_dino_last  = [None]
_dino_l4    = [None]
_dino_l8    = [None]


# ─────────────────────────────────────────────────────────────────────────────
# Teacher DINO capture patch
# ─────────────────────────────────────────────────────────────────────────────

def _patch_dino(dino_model):
    original_fn = dino_model.get_intermediate_layers

    def patched_fn(imgs, n=12):
        result = original_fn(imgs, n)
        _dino_last[0] = rearrange(
            result[-1][:, 1:, :], "B (H W) C -> B C H W", H=48
        ).detach().cpu().half()
        _dino_l4[0] = rearrange(
            result[3][:, 1:, :], "B (H W) C -> B C H W", H=48
        ).detach().cpu().half()
        _dino_l8[0] = rearrange(
            result[7][:, 1:, :], "B (H W) C -> B C H W", H=48
        ).detach().cpu().half()
        return result

    dino_model.get_intermediate_layers = patched_fn

    def unpatch():
        dino_model.get_intermediate_layers = original_fn
    return unpatch


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """Image-only dataset for Phase 2 distillation (no labels)."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, resolution=384):
        self.resolution = resolution
        self.paths = sorted(
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        return TF.to_tensor(img)   # [0, 1] float32, normalise later


class SegDataset(Dataset):
    """Image + label dataset for Phase 3 segmentation fine-tuning."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384):
        self.resolution = resolution

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
        img = TF.normalize(img, mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])

        lbl = Image.open(lbl_path)
        lbl = TF.resize(lbl, (self.resolution, self.resolution),
                        interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Teacher setup
# ─────────────────────────────────────────────────────────────────────────────

def build_teacher(config_file, weights_file, device):
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    from detectron2.modeling import build_model as d2_build
    model = d2_build(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()

    predictor = model.sem_seg_head.predictor
    predictor.test_class_texts = CLASSES_LandDiscover50K
    predictor.cache = None

    return model


def remove_gsnet_clip_cache_hooks(gsnet):
    """Remove GSNet's permanent CLIP skip hooks; fine-tune uses temporary hooks."""
    clip_model = gsnet.sem_seg_head.predictor.clip_model
    removed = 0
    for idx in getattr(gsnet, "layer_indexes", []):
        block = clip_model.visual.transformer.resblocks[idx]
        for hook_id, hook in list(block._forward_hooks.items()):
            closure = getattr(hook, "__closure__", None) or ()
            if any(cell.cell_contents is gsnet for cell in closure):
                del block._forward_hooks[hook_id]
                removed += 1
    if hasattr(gsnet, "layers"):
        gsnet.layers.clear()
    if removed:
        print(f"  Removed {removed} GSNet CLIP cache hooks.")


def release_unused_gsnet(gsnet):
    """Drop GSNet container references after extracting fine-tune modules."""
    gsnet.dino_model = None
    gsnet.backbone = None
    gsnet.sem_seg_head = None
    gsnet.upsample1 = None
    gsnet.upsample2 = None
    gsnet.dino_decod_proj1 = None
    gsnet.dino_decod_proj2 = None
    if hasattr(gsnet, "layers"):
        gsnet.layers.clear()


def build_text_features(class_json, clip_model, device):
    from gs_net.third_party import clip as openai_clip
    with open(class_json) as f:
        class_names = json.load(f)
    templates = ["a photo of a {}."]
    all_feats = []
    with torch.no_grad():
        for name in class_names:
            texts = [t.format(name) for t in templates]
            tokens = openai_clip.tokenize(texts).to(device)
            feats = clip_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
    text_feats = torch.stack(all_feats, dim=0).unsqueeze(0)
    return text_feats   # (1, T, P, C)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Online distillation
# ─────────────────────────────────────────────────────────────────────────────

_BRANCH_KEYS = ("dino_down", "dino_L4", "dino_L8")


def run_distill_epoch(teacher, student, loader, optimizer, scaler, device, amp,
                      train=True, log_interval=50, epoch=0, use_wandb=False):
    student.train(train)
    total_loss = 0.0
    branch_totals = {k: 0.0 for k in _BRANCH_KEYS}
    n = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for step, images in enumerate(tqdm(loader, leave=False,
                                           desc="distill-train" if train else "distill-val")):
            images = images.to(device)

            with torch.no_grad():
                teacher([{"image": (img * 255.0).clamp(0, 255)} for img in images])
                dino_down = teacher.dino_down_sample(
                    _dino_last[0].to(device).float()
                )

            targets = {
                "dino_down": dino_down,
                "dino_L4": _dino_l4[0].to(device).float(),
                "dino_L8": _dino_l8[0].to(device).float(),
            }

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                pred = student(clip_normalise(images))
                breakdown = distillation_loss_per_branch(pred, targets)
                loss = breakdown["total"]

            if train:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            for k in _BRANCH_KEYS:
                branch_totals[k] += breakdown[k]
            n += 1

            if train and (step + 1) % log_interval == 0:
                avg = total_loss / n
                tqdm.write(
                    f"  step {step+1:5d}  loss={avg:.4f}  "
                    f"down={branch_totals['dino_down']/n:.4f}  "
                    f"l4={branch_totals['dino_L4']/n:.4f}  "
                    f"l8={branch_totals['dino_L8']/n:.4f}"
                )
                if use_wandb:
                    global_step = epoch * len(loader) + step + 1
                    wandb.log({
                        "distill/step/loss":    avg,
                        "distill/step/dino_down": branch_totals["dino_down"] / n,
                        "distill/step/dino_l4": branch_totals["dino_L4"] / n,
                        "distill/step/dino_l8": branch_totals["dino_L8"] / n,
                    }, step=global_step)

    avg_loss     = total_loss / max(n, 1)
    avg_branches = {k: v / max(n, 1) for k, v in branch_totals.items()}
    return avg_loss, avg_branches


def run_distillation(args, teacher, student, device, output_dir, use_wandb):
    full_ds = ImageDataset(args.image_dir, resolution=384)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  [Distill] Train: {n_train}  Val: {n_val}")

    optimizer = torch.optim.AdamW(student.trainable_parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    warmup_epochs = max(1, args.distill_epochs // 10)
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "student_distill_best.pth")

    for epoch in range(args.distill_epochs):
        lr = cosine_lr(optimizer, epoch, args.distill_epochs, warmup_epochs, args.lr)
        t0 = time.time()

        train_loss, train_b = run_distill_epoch(
            teacher, student, train_loader, optimizer, scaler, device,
            args.amp, train=True, log_interval=args.log_interval,
            epoch=epoch, use_wandb=use_wandb,
        )
        val_loss, val_b = run_distill_epoch(
            teacher, student, val_loader, optimizer, scaler, device,
            args.amp, train=False, epoch=epoch, use_wandb=False,
        )

        elapsed = time.time() - t0
        print(
            f"[Distill] Epoch {epoch+1:3d}/{args.distill_epochs}  lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.0f}s)  "
            f"val_down={val_b['dino_down']:.4f}  "
            f"val_l4={val_b['dino_L4']:.4f}  val_l8={val_b['dino_L8']:.4f}"
        )

        if use_wandb:
            wandb.log({
                "distill/epoch":         epoch + 1,
                "distill/lr":            lr,
                "distill/epoch_time_s":  elapsed,
                "distill/train/loss":    train_loss,
                "distill/train/dino_down": train_b["dino_down"],
                "distill/train/dino_l4": train_b["dino_L4"],
                "distill/train/dino_l8": train_b["dino_L8"],
                "distill/val/loss":      val_loss,
                "distill/val/dino_down": val_b["dino_down"],
                "distill/val/dino_l4":   val_b["dino_L4"],
                "distill/val/dino_l8":   val_b["dino_L8"],
            }, step=epoch + 1)

        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }, os.path.join(output_dir, "student_distill_latest.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ [Distill] New best val loss: {val_loss:.4f} → {best_path}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(output_dir, f"student_distill_epoch{epoch+1:03d}.pth")
            torch.save({"epoch": epoch, "student": student.state_dict(),
                        "val_loss": val_loss, "args": vars(args)}, ckpt_path)
            print(f"  → [Distill] Periodic checkpoint: {ckpt_path}")

    print(f"\n[Distill] Done. Best val loss: {best_val_loss:.4f}")
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Segmentation fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(args, gsnet, student, device, output_dir, use_wandb):
    clip_model       = gsnet.sem_seg_head.predictor.clip_model
    clip_skip_indices = tuple(gsnet.layer_indexes)
    remove_gsnet_clip_cache_hooks(gsnet)

    ripd              = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1    = gsnet.upsample1
    clip_upsample2    = gsnet.upsample2
    dino_decod_proj1  = gsnet.dino_decod_proj1
    dino_decod_proj2  = gsnet.dino_decod_proj2

    ripd_unfrozen         = args.unfreeze_ripd or args.warmup_decoder_epochs > 0
    decoder_proj_unfrozen = ripd_unfrozen or args.train_decoder_projections

    decoder_proj_modules = [m for m in [clip_upsample1, clip_upsample2,
                                         dino_decod_proj1, dino_decod_proj2]
                             if m is not None]

    release_unused_gsnet(gsnet)
    gc.collect()

    clip_model = clip_model.to(device)
    ripd = ripd.to(device)
    for m in decoder_proj_modules:
        m.to(device)
    torch.cuda.empty_cache()

    for p in ripd.parameters():
        p.requires_grad = ripd_unfrozen
    for m in decoder_proj_modules:
        for p in m.parameters():
            p.requires_grad = decoder_proj_unfrozen

    warmup = args.warmup_decoder_epochs
    if warmup > 0:
        for p in student.parameters():
            p.requires_grad = False
        print(f"  RIPD decoder unfrozen. Student heads frozen for {warmup}-epoch warmup.")
    elif ripd_unfrozen:
        print("  RIPD unfrozen for fine-tuning.")
    else:
        print("  RIPD frozen; using checkpoint-loaded decoder weights.")
    if decoder_proj_unfrozen:
        print("  Decoder bridge projections trainable.")

    text_feats = build_text_features(args.class_json, clip_model, device)

    full_ds = SegDataset(args.image_dir, args.label_dir)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  [Finetune] Train: {n_train}  Val: {n_val}")

    def _student_head_params():
        return list(student.trainable_parameters())

    def _decoder_proj_params():
        return [p for m in decoder_proj_modules for p in m.parameters()]

    if warmup > 0:
        active_params = []
        if ripd_unfrozen:
            active_params += list(ripd.parameters())
        if decoder_proj_unfrozen:
            active_params += _decoder_proj_params()
    else:
        active_params = _student_head_params()
        if ripd_unfrozen:
            active_params += list(ripd.parameters())
        if decoder_proj_unfrozen:
            active_params += _decoder_proj_params()

    optimizer  = torch.optim.AdamW(active_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler     = GradScaler(enabled=args.amp)
    ignore_idx = 255
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "finetune_best.pth")

    _all_clip_layers = sorted(set(list(clip_skip_indices) + list(student.clip_layers)))

    def _save(path, epoch, avg_val):
        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "ripd": ripd.state_dict() if ripd_unfrozen else None,
            "clip_upsample1":   clip_upsample1.state_dict()  if decoder_proj_unfrozen and clip_upsample1  is not None else None,
            "clip_upsample2":   clip_upsample2.state_dict()  if decoder_proj_unfrozen and clip_upsample2  is not None else None,
            "dino_decod_proj1": dino_decod_proj1.state_dict() if decoder_proj_unfrozen and dino_decod_proj1 is not None else None,
            "dino_decod_proj2": dino_decod_proj2.state_dict() if decoder_proj_unfrozen and dino_decod_proj2 is not None else None,
            "val_loss": avg_val,
            "args": vars(args),
        }, path)

    for epoch in range(args.finetune_epochs):
        if warmup > 0 and epoch == warmup:
            for p in _student_head_params():
                p.requires_grad = True
            active_params = _student_head_params()
            if ripd_unfrozen:
                active_params += list(ripd.parameters())
            if decoder_proj_unfrozen:
                active_params += _decoder_proj_params()
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

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} [train]")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.detach().expand(B, -1, -1, -1)

            with torch.no_grad():
                with autocast(enabled=args.amp):
                    clip_features, _all_skips = get_clip_skips(
                        clip_model, images, _all_clip_layers)
                    l0, l1 = clip_skip_indices
                    res4 = clip_upsample1(_all_skips[l0]) if clip_upsample1 is not None else None
                    res5 = clip_upsample2(_all_skips[l1]) if clip_upsample2 is not None else None
                    _stacked = torch.cat([_all_skips[l] for l in student.clip_layers], dim=1)
                    del _all_skips
                    # _stacked is reused by student heads below — no second CLIP pass needed

            if student.training:
                with autocast(enabled=args.amp):
                    _trunk = grad_ckpt.checkpoint(student.shared_trunk, _stacked, use_reentrant=False)
                del _stacked

                def _down(_x):
                    with autocast(enabled=args.amp):
                        return student.dino_down_branch(_x)
                dino_down = grad_ckpt.checkpoint(_down, _trunk, use_reentrant=False)

                def _dl4(_x):
                    with autocast(enabled=args.amp):
                        return student.dino_l4_branch(_x)
                _dino_L4 = grad_ckpt.checkpoint(_dl4, _trunk, use_reentrant=False) if dino_decod_proj1 is not None else None

                def _dl8(_x):
                    with autocast(enabled=args.amp):
                        return student.dino_l8_branch(_x)
                _dino_L8 = grad_ckpt.checkpoint(_dl8, _trunk, use_reentrant=False) if dino_decod_proj2 is not None else None
                del _trunk
            else:
                with torch.no_grad(), autocast(enabled=args.amp):
                    _trunk = student.shared_trunk(_stacked)
                    del _stacked
                    dino_down = student.dino_down_branch(_trunk)
                    _dino_L4 = student.dino_l4_branch(_trunk) if dino_decod_proj1 is not None else None
                    _dino_L8 = student.dino_l8_branch(_trunk) if dino_decod_proj2 is not None else None
                    del _trunk

            with autocast(enabled=args.amp):
                dino_L4_proj = dino_decod_proj1(_dino_L4) if _dino_L4 is not None else None
                dino_L8_proj = dino_decod_proj2(_dino_L8) if _dino_L8 is not None else None
            del _dino_L4, _dino_L8

            clip_res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
            clip_guidance = (clip_res3, res4, res5)
            del clip_features

            logit = ripd(
                clip_res3,
                dino_down,
                tf,
                clip_guidance,
                [dino_L4_proj, dino_L8_proj],
            )

            with autocast(enabled=args.amp):
                logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                          mode="bilinear", align_corners=False)
                del logit
                loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx) / grad_accum
                del logit_up
            loss_value = loss.detach().item() * grad_accum

            try:
                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError:
                raise

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

            train_loss += loss_value
            n_train_batches += 1
            del loss, loss_value, dino_down, tf, clip_res3, clip_guidance, dino_L4_proj, dino_L8_proj
            del res4, res5, images, labels

        student.eval()
        if ripd_unfrozen:
            ripd.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} [val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]
                tf = text_feats.detach().expand(B, -1, -1, -1)
                with autocast(enabled=args.amp):
                    clip_feats_val, all_skips_val = get_clip_skips(
                        clip_model, images, _all_clip_layers)
                    stacked_val = torch.cat(
                        [all_skips_val[l] for l in student.clip_layers], dim=1)
                    logit = gs_distill_inference(
                        image=images, text_feats=tf, student=student,
                        clip_model=clip_model, ripd=ripd,
                        clip_upsample1=clip_upsample1, clip_upsample2=clip_upsample2,
                        dino_decod_proj1=dino_decod_proj1, dino_decod_proj2=dino_decod_proj2,
                        clip_skip_layer_indices=clip_skip_indices,
                        clip_features=clip_feats_val, clip_skips=all_skips_val,
                        student_clip_stacked=stacked_val,
                    )
                    del clip_feats_val, all_skips_val, stacked_val
                    logit_up = F.interpolate(logit, size=labels.shape[-2:],
                                              mode="bilinear", align_corners=False)
                    del logit
                    loss = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)
                    del logit_up
                val_loss += loss.item()
                n_val_batches += 1
                del loss, tf, images, labels

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val   = val_loss   / max(n_val_batches,   1)
        phase     = "warmup" if (warmup > 0 and epoch < warmup) else "joint"
        print(f"[Finetune] Epoch {epoch+1:3d}/{args.finetune_epochs}  [{phase}]  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            wandb.log({
                "finetune/epoch":      epoch + 1,
                "finetune/train/loss": avg_train,
                "finetune/val/loss":   avg_val,
                "finetune/phase":      phase,
            }, step=args.distill_epochs + epoch + 1)

        _save(os.path.join(output_dir, "finetune_latest.pth"), epoch, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save(best_path, epoch, avg_val)
            print(f"  ✓ [Finetune] New best val loss: {avg_val:.4f} → {best_path}")

    print(f"\n[Finetune] Done. Best val loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Args + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLIP-backed GS-Distill: Phase 2 + Phase 3")
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--image-dir",      required=True)
    p.add_argument("--label-dir",      required=True)
    p.add_argument("--class-json",     default="datasets/landdiscover.json")
    p.add_argument("--output-dir",     default="output/ashie/clip")
    p.add_argument("--distill-epochs", type=int,   default=30)
    p.add_argument("--finetune-epochs",type=int,   default=15)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--d-dino",         type=int,   default=768)
    p.add_argument("--clip-layers",    type=int,   nargs="+", default=[8, 16, 20, 23])
    p.add_argument("--val-fraction",   type=float, default=0.05)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--grad-accum",     type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum).")
    p.add_argument("--unfreeze-ripd",            action="store_true")
    p.add_argument("--train-decoder-projections", action="store_true",
                   help="Train GSNet CLIP/DINO decoder bridge projections while keeping RIPD frozen.")
    p.add_argument("--warmup-decoder-epochs",    type=int, default=0,
                   help="Train RIPD decoder alone for N epochs, then unfreeze student heads.")
    p.add_argument("--skip-distill",    action="store_true",
                   help="Skip Phase 2; load --distill-ckpt directly and run Phase 3 only.")
    p.add_argument("--distill-ckpt",    default=None,
                   help="Path to existing distill checkpoint (used with --skip-distill).")
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-interval",   type=int,   default=50)
    p.add_argument("--wandb-project",  default="gs-distill")
    p.add_argument("--wandb-run",      default=None)
    p.add_argument("--no-wandb",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # ── Load GSNet teacher ────────────────────────────────────────────────────
    print(f"Loading GSNet teacher from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    teacher = teacher.to(device)

    unpatch_dino = _patch_dino(teacher.dino_model)

    # ── Build student (reuse teacher's fine-tuned CLIP backbone) ─────────────
    print("Reusing teacher's fine-tuned CLIP as student backbone ...")
    clip_model = teacher.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    student = GSDistillStudent(
        clip_model=clip_model,
        d_dino=args.d_dino,
        clip_layers=args.clip_layers,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"Student trainable params: {n_params / 1e6:.2f}M")

    if args.skip_distill:
        distill_ckpt_path = args.distill_ckpt or os.path.join(args.output_dir, "student_distill_best.pth")
        print(f"\nSkipping Phase 2 — loading distill checkpoint: {distill_ckpt_path}")
        if not os.path.isfile(distill_ckpt_path):
            print(f"[ERROR] Distill checkpoint not found: {distill_ckpt_path}"); sys.exit(1)
        ckpt = torch.load(distill_ckpt_path, map_location=device)
        student.load_state_dict(ckpt["student"])
        unpatch_dino()
    else:
        # ── Phase 2: Distillation ─────────────────────────────────────────────
        print("\n" + "="*60)
        print("Phase 2 — Online distillation (CLIP student ← GSNet teacher)")
        print("="*60)
        best_distill_ckpt = run_distillation(args, teacher, student, device,
                                              args.output_dir, use_wandb)
        unpatch_dino()

        # ── Reload best distill checkpoint for Phase 3 ────────────────────────
        print(f"\nReloading best distill checkpoint: {best_distill_ckpt}")
        ckpt = torch.load(best_distill_ckpt, map_location=device)
        student.load_state_dict(ckpt["student"])

    # ── Phase 3: Segmentation fine-tune ───────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 3 — Segmentation fine-tune on LD50K")
    print("="*60)
    run_finetune(args, teacher, student, device, args.output_dir, use_wandb)

    if use_wandb:
        wandb.finish()
    print("\nAll done.")


if __name__ == "__main__":
    main()
