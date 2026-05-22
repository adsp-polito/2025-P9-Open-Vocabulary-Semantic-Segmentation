"""
Baseline — same architecture as GS-Distill but NO distillation pre-training.

Builds a GSDistillStudent (identical shared trunk + DINO substitute heads) with randomly
initialised weights, then fine-tunes directly on LD50K segmentation labels.

This is the apples-to-apples benchmark for GS-Distill: same model, same
fine-tuning procedure, only difference is the student starts from random
weights instead of a distilled checkpoint.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/train_baseline.py \\
        --gsnet-config    configs/vitl_336_dinov3.yaml \\
        --gsnet-weights   path/to/gsnet_checkpoint.pth \\
        --image-dir       path/to/LD50K/images \\
        --label-dir       path/to/LD50K/labels \\
        --output-dir      output/baseline/ \\
        [--epochs 15] \\
        [--batch-size 1] \\
        [--lr 1e-5] \\
        [--amp] \\
        [--device cuda]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import gc
import json
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

try:
    import wandb
except ImportError:
    wandb = None

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_net.third_party import clip as openai_clip
from gs_distill.inference import gs_distill_inference
from gs_distill.student import GSDistillStudent
from gs_distill.utils import get_clip_skips
from einops import rearrange as _rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation dataset
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384,
                 clip_mean=(0.48145466, 0.4578275, 0.40821073),
                 clip_std=(0.26862954, 0.26130258, 0.27577711)):
        self.resolution = resolution
        self.clip_mean  = clip_mean
        self.clip_std   = clip_std

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
    p.add_argument("--image-dir",      required=True)
    p.add_argument("--label-dir",      required=True)
    p.add_argument("--class-json",     default="datasets/landdiscover.json")
    p.add_argument("--output-dir",     default="output/baseline")
    p.add_argument("--epochs",         type=int,   default=15)
    p.add_argument("--batch-size",     type=int,   default=1)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--d-dino",         type=int,   default=768)
    p.add_argument("--clip-layers",    type=int,   nargs="+", default=[8, 16, 20, 23])
    p.add_argument("--unfreeze-ripd",          action="store_true")
    p.add_argument("--train-decoder-projections", action="store_true",
                   help="Train GSNet CLIP/DINO decoder bridge projections while keeping RIPD frozen.")
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
    """Drop GSNet container references after extracting the modules fine-tune needs."""
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
    text_feats = torch.stack(all_feats, dim=0).unsqueeze(0)  # (1, T, P, C)
    return text_feats


def create_subset(dataset, fraction, seed):
    from torch.utils.data import Subset
    from collections import defaultdict

    rng = np.random.RandomState(seed)
    n = len(dataset)

    print(f"Building stratification index for {n} images ...")

    image_classes = []
    class_freq = defaultdict(int)
    for _, lbl_path in dataset.samples:
        lbl = np.array(Image.open(lbl_path))
        present = set(lbl[lbl != 255].tolist())
        image_classes.append(present)
        for c in present:
            class_freq[c] += 1

    strata = []
    for present in image_classes:
        if present:
            stratum = min(present, key=lambda c: class_freq[c])
        else:
            stratum = -1
        strata.append(stratum)

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

    # ── Load GSNet on CPU (to extract CLIP, RIPD + projection layers) ────────
    print("Loading GSNet ...")
    gsnet = build_gsnet(args.gsnet_config, args.gsnet_weights, "cpu")

    clip_model = gsnet.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_skip_indices = tuple(gsnet.layer_indexes)
    remove_gsnet_clip_cache_hooks(gsnet)

    ripd             = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1   = gsnet.upsample1
    clip_upsample2   = gsnet.upsample2
    dino_decod_proj1 = gsnet.dino_decod_proj1
    dino_decod_proj2 = gsnet.dino_decod_proj2

    decoder_proj_modules = [m for m in [clip_upsample1, clip_upsample2,
                                         dino_decod_proj1, dino_decod_proj2]
                             if m is not None]

    release_unused_gsnet(gsnet)
    del gsnet
    gc.collect()

    clip_model = clip_model.to(device)
    ripd = ripd.to(device)
    for m in decoder_proj_modules:
        m.to(device)
    torch.cuda.empty_cache()

    warmup        = args.warmup_decoder_epochs
    ripd_unfrozen = args.unfreeze_ripd or warmup > 0
    decoder_proj_unfrozen = ripd_unfrozen or args.train_decoder_projections

    for p in ripd.parameters():
        p.requires_grad = ripd_unfrozen

    for m in decoder_proj_modules:
        for p in m.parameters():
            p.requires_grad = decoder_proj_unfrozen

    # ── Build student from scratch (random init — no distillation) ───────────
    print("Building baseline student (random init, same arch as GS-Distill) ...")
    student = GSDistillStudent(
        clip_model=clip_model,
        d_dino=args.d_dino,
        clip_layers=args.clip_layers,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"  Trainable params: {n_params / 1e6:.2f}M")

    if warmup > 0:
        for p in student.parameters():
            p.requires_grad = False
        print(f"  Student frozen for {warmup}-epoch RIPD warmup.")
    elif ripd_unfrozen:
        print("  RIPD unfrozen for joint fine-tuning.")
    else:
        print("  RIPD frozen; student heads train against frozen RIPD decoder.")
    if decoder_proj_unfrozen:
        print("  Decoder bridge projections trainable.")

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

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb and wandb is not None
    if wandb is None and not args.no_wandb:
        print("W&B is not installed; continuing without W&B logging.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
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
        _all_clip_layers = sorted(set(list(clip_skip_indices) + list(student.clip_layers)))

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.detach().expand(B, -1, -1, -1)

            # ── CLIP forward (frozen, single pass for RIPD skips + student trunk) ──
            with torch.no_grad():
                with autocast(enabled=args.amp):
                    clip_features, _all_skips = get_clip_skips(
                        clip_model, images, _all_clip_layers)
                    l0, l1 = clip_skip_indices
                    clip_skips = {l: _all_skips[l] for l in clip_skip_indices}
                    _stacked = torch.cat([_all_skips[l] for l in student.clip_layers], dim=1)
                    del _all_skips

            with autocast(enabled=args.amp):
                res4 = clip_upsample1(clip_skips[l0]) if clip_upsample1 is not None else None
                res5 = clip_upsample2(clip_skips[l1]) if clip_upsample2 is not None else None

            # ── Student: per-branch checkpointing ────────────────────────────────
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

            clip_res3 = _rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
            clip_guidance = (clip_res3, res4, res5)
            del clip_features, clip_skips

            logit = ripd(
                clip_res3.float(),
                dino_down.float(),
                tf.float(),
                tuple(x.float() if x is not None else None for x in clip_guidance),
                [x.float() if x is not None else None for x in [dino_L4_proj, dino_L8_proj]],
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
            del res4, res5
            del images, labels

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
        print(f"Epoch {epoch+1:3d}/{args.epochs}  [{phase}]  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            wandb.log({
                "epoch":         epoch + 1,
                "train/loss":    avg_train,
                "val/loss":      avg_val,
                "phase":         phase,
                "best_val_loss": best_val_loss,
            }, step=epoch + 1)

        torch.save({
            "epoch":   epoch,
            "student": student.state_dict(),
            "ripd":    ripd.state_dict() if ripd_unfrozen else None,
            "clip_upsample1":   clip_upsample1.state_dict()  if decoder_proj_unfrozen and clip_upsample1  is not None else None,
            "clip_upsample2":   clip_upsample2.state_dict()  if decoder_proj_unfrozen and clip_upsample2  is not None else None,
            "dino_decod_proj1": dino_decod_proj1.state_dict() if decoder_proj_unfrozen and dino_decod_proj1 is not None else None,
            "dino_decod_proj2": dino_decod_proj2.state_dict() if decoder_proj_unfrozen and dino_decod_proj2 is not None else None,
            "val_loss": avg_val,
            "args": vars(args),
        }, os.path.join(args.output_dir, "baseline_latest.pth"))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.output_dir, "baseline_best.pth")
            torch.save({
                "epoch":   epoch,
                "student": student.state_dict(),
                "ripd":    ripd.state_dict() if ripd_unfrozen else None,
                "clip_upsample1":   clip_upsample1.state_dict()  if decoder_proj_unfrozen and clip_upsample1  is not None else None,
                "clip_upsample2":   clip_upsample2.state_dict()  if decoder_proj_unfrozen and clip_upsample2  is not None else None,
                "dino_decod_proj1": dino_decod_proj1.state_dict() if decoder_proj_unfrozen and dino_decod_proj1 is not None else None,
                "dino_decod_proj2": dino_decod_proj2.state_dict() if decoder_proj_unfrozen and dino_decod_proj2 is not None else None,
                "val_loss": avg_val,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ New best val loss: {avg_val:.4f} → {best_path}")

    if use_wandb:
        wandb.finish()
    print(f"\nBaseline training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
