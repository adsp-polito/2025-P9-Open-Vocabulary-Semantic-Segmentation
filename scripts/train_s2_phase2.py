#!/usr/bin/env python3
"""
Student-2 Phase 2: Joint fine-tuning with mixed loss.

Loads the Phase 1 warm-up checkpoint and jointly trains the decoder +
Student-1 branches (CLIP stays frozen) using:
    L = 0.5 * L_KL(student, cached_teacher) + 0.5 * L_CE(student, GT_mask)

No live GSNet teacher needed — distillation signal comes from pre-cached logits.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("./detectron2"))
sys.path.insert(0, os.path.abspath("."))

import argparse
import json
import time
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from gs_distill.student2 import GSDistillStudent2
from gs_distill.utils import get_clip_skips
from scripts.train_clip import build_text_features, build_teacher, cosine_lr, remove_gsnet_clip_cache_hooks


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Phase2Dataset(Dataset):
    """Triplets: image + cached teacher logits + GT segmentation mask."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, cache_dir, label_dir, resolution=384):
        self.resolution = resolution
        cache_dir = Path(cache_dir)
        label_dir = Path(label_dir)

        cache_stems = {p.stem for p in cache_dir.glob("*.pt")}
        lbl_stems = {
            p.stem for p in label_dir.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }
        valid_stems = cache_stems & lbl_stems

        self.samples = []
        for p in sorted(Path(image_dir).rglob("*")):
            if p.suffix.lower() not in self.EXTENSIONS:
                continue
            if p.stem not in valid_stems:
                continue
            lbl_path = None
            for ext in (".png", ".jpg", ".tif", ".tiff"):
                candidate = label_dir / (p.stem + ext)
                if candidate.exists():
                    lbl_path = candidate
                    break
            if lbl_path is None:
                continue
            self.samples.append((
                str(p),
                str(cache_dir / (p.stem + ".pt")),
                str(lbl_path),
            ))

        if not self.samples:
            raise RuntimeError(
                f"No triplets found.\n"
                f"  image_dir: {image_dir}\n"
                f"  cache_dir: {cache_dir}\n"
                f"  label_dir: {label_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cache_path, lbl_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        cached = torch.load(cache_path, map_location="cpu")
        logits = cached["logits"].float()   # (40, 96, 96)

        lbl = Image.open(lbl_path)
        lbl = TF.resize(lbl, (self.resolution, self.resolution),
                        interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()  # (H, W)

        return img, logits, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def response_distil_loss(student_logits, teacher_logits, tau=4.0):
    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(
            teacher_logits, size=student_logits.shape[-2:],
            mode="bilinear", align_corners=False,
        )
    s = student_logits.float() / tau
    t = teacher_logits.float() / tau
    student_lp = F.log_softmax(s, dim=1)
    teacher_p  = F.softmax(t,  dim=1)
    # Per-pixel mean KL (sum over classes, mean over B*H*W positions) so the
    # distillation loss is on the SAME scale as the per-pixel CE loss. The old
    # reduction="batchmean" divided only by batch (=1), turning KL into a near-sum
    # over ~368k positions (~1570) that dwarfed CE (~4.6) ~340x — CE never learned.
    B, T, H, W = student_lp.shape
    student_lp = student_lp.permute(0, 2, 3, 1).reshape(-1, T)
    teacher_p  = teacher_p.permute(0, 2, 3, 1).reshape(-1, T)
    return F.kl_div(student_lp, teacher_p, reduction="batchmean") * (tau ** 2)


def seg_loss(student_logits, labels, ignore_index=255):
    logit_up = F.interpolate(
        student_logits.float(), size=labels.shape[-2:],
        mode="bilinear", align_corners=False,
    )
    return F.cross_entropy(logit_up, labels, ignore_index=ignore_index)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_phase1_weights(student2, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("student2", ckpt)
    missing, unexpected = student2.load_state_dict(state, strict=False)
    print(f"Loaded Phase 1 weights from {ckpt_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return ckpt.get("val_loss", None)


def save_ckpt(path, epoch, student2, val_loss, args):
    torch.save(
        {
            "epoch": epoch,
            "student2": student2.state_dict(),
            "val_loss": float(val_loss),
            "args": vars(args),
        },
        path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    loader, student2, text_feats, optimizer, scaler,
    device, amp, clip_skip_indices, tau, lam_distill, lam_ce,
    train, epoch, total_epochs, grad_accum,
):
    student2.train(train)
    student2.clip_model.eval()
    # CLIP always frozen
    for p in student2.clip_model.parameters():
        p.requires_grad_(False)

    total_loss = total_kl = total_ce = 0.0
    n_batches = 0
    desc = "train" if train else "val  "
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    if train:
        optimizer.zero_grad(set_to_none=True)

    with grad_ctx:
        for step, (images, teacher_logits, labels) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1:3d}/{total_epochs} [{desc}]", leave=False)
        ):
            images         = images.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            labels         = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.detach().expand(B, -1, -1, -1)

            with autocast(enabled=amp):
                student_logits = student2(
                    image=images,
                    text_feats=tf,
                    clip_skip_indices=clip_skip_indices,
                )
                l_kl = response_distil_loss(student_logits, teacher_logits, tau=tau)
                l_ce = seg_loss(student_logits, labels)
                loss = lam_distill * l_kl + lam_ce * l_ce
                loss_for_backward = loss / grad_accum

            if train:
                if amp:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                is_last = step + 1 == len(loader)
                if (step + 1) % grad_accum == 0 or is_last:
                    if amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], 1.0
                    )
                    if amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach())
            total_kl   += float(l_kl.detach())
            total_ce   += float(l_ce.detach())
            n_batches  += 1

            if train and step % 1000 == 0:
                alloc = torch.cuda.memory_allocated() / 1024**2
                print(f"\n  [MEM step={step}] {alloc:.1f} MiB", flush=True)

    n = max(n_batches, 1)
    return total_loss / n, total_kl / n, total_ce / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Student-2 Phase 2: joint fine-tuning (decoder + branches)."
    )
    p.add_argument("--gsnet-config",  required=True)
    p.add_argument("--gsnet-weights", required=True)
    p.add_argument("--phase1-ckpt",   required=True,
                   help="Phase 1 warm-up checkpoint to initialise from.")
    p.add_argument("--image-dir",  required=True)
    p.add_argument("--cache-dir",  default="cache_logits")
    p.add_argument("--label-dir",  required=True)
    p.add_argument("--class-json", default="datasets/landdiscover.json")
    p.add_argument("--output-dir", default="output/ashie/s2/phase2")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--lr-decoder",   type=float, default=1e-4,
                   help="LR for cost head + projection layers.")
    p.add_argument("--lr-branches",  type=float, default=1e-5,
                   help="LR for Student-1 substitute branches (lower — already trained).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tau",          type=float, default=4.0)
    p.add_argument("--lam-distill",  type=float, default=0.5,
                   help="Weight for KL distillation loss.")
    p.add_argument("--lam-ce",       type=float, default=0.5,
                   help="Weight for CE segmentation loss.")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--clip-layers",       type=int, nargs="+", default=[8, 16, 20, 23])
    p.add_argument("--clip-skip-indices", type=int, nargs=2,   default=[7, 15])
    p.add_argument("--head-hidden-dim",  type=int,   default=64)
    p.add_argument("--head-num-layers",  type=int,   default=2)
    p.add_argument("--head-window-size", type=int,   default=4)
    p.add_argument("--head-pad-len",     type=int,   default=0)
    p.add_argument("--head-dec-dims",    type=int,   nargs=2, default=[32, 8])
    p.add_argument("--d-dino", type=int, default=768)
    p.add_argument("--amp",       action="store_true")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Enable gradient checkpointing in S2CostHead aggregator.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available — training requires a GPU. "
            "Check that SLURM allocated a GPU and CUDA_VISIBLE_DEVICES is set correctly."
        )
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds = Phase2Dataset(args.image_dir, args.cache_dir, args.label_dir)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Dataset: {len(full_ds)} triplets  train={n_train}  val={n_val}")

    # ── Build model (same arch as Phase 1) ───────────────────────────────────
    print(f"Loading GSNet to extract CLIP from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    clip_model = teacher.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    # Remove GSNet's permanent CLIP skip hooks before nulling sem_seg_head.
    # Without this, each encode_image call appends ~2.4 MB to teacher.layers
    # indefinitely (hooks APPEND, never overwrite), causing a GPU memory leak.
    remove_gsnet_clip_cache_hooks(teacher)
    teacher.dino_model = None
    teacher.backbone = None
    teacher.sem_seg_head = None
    teacher.upsample1 = None
    teacher.upsample2 = None
    teacher.dino_decod_proj1 = None
    teacher.dino_decod_proj2 = None
    del teacher
    torch.cuda.empty_cache()

    head_kwargs = {
        "hidden_dim":             args.head_hidden_dim,
        "num_layers":             args.head_num_layers,
        "window_size":            args.head_window_size,
        "pad_len":                args.head_pad_len,
        "gradient_checkpointing": args.grad_ckpt,
        "dec_dims":               tuple(args.head_dec_dims),
    }
    student2 = GSDistillStudent2(
        clip_model=clip_model,
        d_dino=args.d_dino,
        clip_layers=args.clip_layers,
        head_kwargs=head_kwargs,
    ).to(device)

    phase1_val_loss = load_phase1_weights(student2, args.phase1_ckpt, device)
    print(f"Phase 1 best val loss was: {phase1_val_loss}")

    # CLIP fp16 (same as Phase 1)
    clip_model.half()
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.float()

    # ── Two param groups: decoder (higher LR) + branches (lower LR) ─────────
    decoder_params = [
        p for m in [
            student2.clip_cost_proj, student2.dino_cost_proj,
            student2.text_cost_proj, student2.cost_head,
        ]
        for p in m.parameters()
    ]
    branch_params = [
        p for m in [
            student2.shared_trunk, student2.dino_down_branch,
            student2.dino_l4_branch, student2.dino_l8_branch,
        ]
        for p in m.parameters()
    ]
    for p in decoder_params + branch_params:
        p.requires_grad = True

    n_decoder = sum(p.numel() for p in decoder_params)
    n_branch  = sum(p.numel() for p in branch_params)
    print(f"Trainable: decoder={n_decoder/1e6:.2f}M  branches={n_branch/1e6:.2f}M")

    # ── Text features ─────────────────────────────────────────────────────────
    text_feats = build_text_features(args.class_json, clip_model, device)

    # ── Optimizer (two param groups) ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_params, "lr": args.lr_decoder},
            {"params": branch_params,  "lr": args.lr_branches},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp)
    clip_skip_indices = tuple(args.clip_skip_indices)
    warmup_epochs = max(1, args.epochs // 10)

    best_val  = float("inf")
    best_path   = os.path.join(args.output_dir, "s2_phase2_best.pth")
    latest_path = os.path.join(args.output_dir, "s2_phase2_latest.pth")

    print(f"\nPhase 2: {args.epochs} epochs")
    print(f"  lr_decoder={args.lr_decoder:.1e}  lr_branches={args.lr_branches:.1e}")
    print(f"  loss = {args.lam_distill}×KL + {args.lam_ce}×CE  tau={args.tau}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    for epoch in range(args.epochs):
        # cosine schedule applied per-group to preserve the LR ratio
        t = epoch / max(args.epochs - 1, 1)
        warmup = min(1.0, (epoch + 1) / max(warmup_epochs, 1))
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        scale  = warmup * cosine
        optimizer.param_groups[0]["lr"] = args.lr_decoder  * scale
        optimizer.param_groups[1]["lr"] = args.lr_branches * scale

        t0 = time.time()
        train_loss, train_kl, train_ce = run_epoch(
            train_loader, student2, text_feats, optimizer, scaler,
            device, args.amp, clip_skip_indices, args.tau,
            args.lam_distill, args.lam_ce,
            train=True, epoch=epoch, total_epochs=args.epochs,
            grad_accum=args.grad_accum,
        )
        val_loss, val_kl, val_ce = run_epoch(
            val_loader, student2, text_feats, optimizer, scaler,
            device, args.amp, clip_skip_indices, args.tau,
            args.lam_distill, args.lam_ce,
            train=False, epoch=epoch, total_epochs=args.epochs,
            grad_accum=args.grad_accum,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.4f} (kl={train_kl:.4f} ce={train_ce:.4f})  "
            f"val={val_loss:.4f} (kl={val_kl:.4f} ce={val_ce:.4f})  "
            f"({elapsed:.0f}s)"
        )

        save_ckpt(latest_path, epoch, student2, val_loss, args)
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(best_path, epoch, student2, best_val, args)
            print(f"  New best val loss: {best_val:.4f}  -> {best_path}")

    print(f"\nPhase 2 done. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
