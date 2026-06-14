#!/usr/bin/env python3
"""
Student-2 Phase 3: Decoder consolidation via CE-only fine-tuning.

Loads the Phase 2 checkpoint and trains ONLY the decoder (cost_head +
projection layers) with CE loss against GT masks.  S1 branches and CLIP
are fully frozen — same freeze profile as Phase 1.

Motivation: after Phase 2's mixed KL+CE loss the decoder may still be
pulled toward the teacher's soft predictions rather than hard GT labels.
This short CE-only pass lets it consolidate without the KL regulariser.

Saves weights to a NEW directory (--output-dir) so Phase 2 checkpoints
are never modified.  The saved checkpoint contains three separate keys
so you can diff or reload components independently:
    "clip"       : clip_model.state_dict()
    "s1_layers"  : shared_trunk + dino_*_branch state dicts
    "s2_decoder" : clip_cost_proj + dino_cost_proj + text_cost_proj + cost_head
"""

import sys
import os

sys.path.insert(0, os.path.abspath("./detectron2"))
sys.path.insert(0, os.path.abspath("."))

import argparse
import math
import time
from pathlib import Path

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
from scripts.train_clip import build_text_features, build_teacher, cosine_lr, remove_gsnet_clip_cache_hooks


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (same as Phase 2 — needs GT labels)
# ─────────────────────────────────────────────────────────────────────────────

class Phase3Dataset(Dataset):
    """Image + GT segmentation mask pairs."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, label_dir, resolution=384):
        self.resolution = resolution
        label_dir = Path(label_dir)

        lbl_stems = {
            p.stem for p in label_dir.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        }

        self.samples = []
        for p in sorted(Path(image_dir).rglob("*")):
            if p.suffix.lower() not in self.EXTENSIONS:
                continue
            if p.stem not in lbl_stems:
                continue
            lbl_path = None
            for ext in (".png", ".jpg", ".tif", ".tiff"):
                candidate = label_dir / (p.stem + ext)
                if candidate.exists():
                    lbl_path = candidate
                    break
            if lbl_path is None:
                continue
            self.samples.append((str(p), str(lbl_path)))

        if not self.samples:
            raise RuntimeError(
                f"No image/label pairs found.\n"
                f"  image_dir: {image_dir}\n"
                f"  label_dir: {label_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (self.resolution, self.resolution))
        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        lbl = Image.open(lbl_path)
        lbl = TF.resize(lbl, (self.resolution, self.resolution),
                        interpolation=TF.InterpolationMode.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()

        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def seg_loss(student_logits, labels, ignore_index=255):
    logit_up = F.interpolate(
        student_logits.float(), size=labels.shape[-2:],
        mode="bilinear", align_corners=False,
    )
    return F.cross_entropy(logit_up, labels, ignore_index=ignore_index)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_phase2_weights(student2, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("student2", ckpt)
    missing, unexpected = student2.load_state_dict(state, strict=False)
    print(f"Loaded Phase 2 weights from {ckpt_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return ckpt.get("val_loss", None)


def save_ckpt(path, epoch, student2, val_loss, args):
    """Save with separate keys for clip / s1_layers / s2_decoder."""
    s1_state = {}
    for name, module in [
        ("shared_trunk",     student2.shared_trunk),
        ("dino_down_branch", student2.dino_down_branch),
        ("dino_l4_branch",   student2.dino_l4_branch),
        ("dino_l8_branch",   student2.dino_l8_branch),
    ]:
        for k, v in module.state_dict().items():
            s1_state[f"{name}.{k}"] = v

    decoder_state = {}
    for name, module in [
        ("clip_cost_proj", student2.clip_cost_proj),
        ("dino_cost_proj", student2.dino_cost_proj),
        ("text_cost_proj", student2.text_cost_proj),
        ("cost_head",      student2.cost_head),
    ]:
        for k, v in module.state_dict().items():
            decoder_state[f"{name}.{k}"] = v

    torch.save(
        {
            "epoch":      epoch,
            "clip":       student2.clip_model.state_dict(),
            "s1_layers":  s1_state,
            "s2_decoder": decoder_state,
            # full state dict so the checkpoint can be loaded like phase 1/2
            "student2":   student2.state_dict(),
            "val_loss":   float(val_loss),
            "args":       vars(args),
        },
        path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    loader, student2, text_feats, optimizer, scaler,
    device, amp, clip_skip_indices, decoder_params,
    train, epoch, total_epochs, grad_accum,
):
    student2.train(train)
    # CLIP and S1 branches always frozen
    student2.clip_model.eval()
    for m in [student2.shared_trunk, student2.dino_down_branch,
              student2.dino_l4_branch, student2.dino_l8_branch]:
        m.eval()

    total_loss = 0.0
    n_batches = 0
    desc = "train" if train else "val  "
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    if train:
        optimizer.zero_grad(set_to_none=True)

    with grad_ctx:
        for step, (images, labels) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1:3d}/{total_epochs} [{desc}]", leave=False)
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.detach().expand(B, -1, -1, -1)

            with autocast(enabled=amp):
                student_logits = student2(
                    image=images,
                    text_feats=tf,
                    clip_skip_indices=clip_skip_indices,
                )
                loss = seg_loss(student_logits, labels)
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
                    nn.utils.clip_grad_norm_(decoder_params, 1.0)
                    if amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach())
            n_batches += 1

            if train and step % 1000 == 0:
                alloc = torch.cuda.memory_allocated() / 1024**2
                print(f"\n  [MEM step={step}] {alloc:.1f} MiB", flush=True)

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Student-2 Phase 3: CE-only decoder consolidation (S1 frozen)."
    )
    p.add_argument("--gsnet-config",  required=True)
    p.add_argument("--gsnet-weights", required=True)
    p.add_argument("--phase2-ckpt",   required=True,
                   help="Phase 2 best checkpoint to initialise from (never overwritten).")
    p.add_argument("--image-dir",  required=True)
    p.add_argument("--label-dir",  required=True)
    p.add_argument("--class-json", default="datasets/landdiscover.json")
    p.add_argument("--output-dir", default="output/ashie/s2/phase3")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=1e-5,
                   help="Decoder LR (keep small — S1 is frozen, decoder only refines).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
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
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available — training requires a GPU."
        )
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds = Phase3Dataset(args.image_dir, args.label_dir)
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
    print(f"Dataset: {len(full_ds)} pairs  train={n_train}  val={n_val}")

    # ── Extract CLIP from GSNet ───────────────────────────────────────────────
    print(f"Loading GSNet to extract CLIP from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    clip_model = teacher.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
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

    phase2_val_loss = load_phase2_weights(student2, args.phase2_ckpt, device)
    print(f"Phase 2 best val loss was: {phase2_val_loss}")

    # Freeze CLIP and S1 branches — only decoder trains
    for p in student2.clip_model.parameters():
        p.requires_grad_(False)
    student2.set_student_backbone_trainable(False)

    clip_model.half()
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.float()

    decoder_params = [
        p for m in [
            student2.clip_cost_proj, student2.dino_cost_proj,
            student2.text_cost_proj, student2.cost_head,
        ]
        for p in m.parameters()
    ]
    for p in decoder_params:
        p.requires_grad_(True)

    n_decoder = sum(p.numel() for p in decoder_params)
    n_total   = sum(p.numel() for p in student2.parameters())
    print(f"Student-2: {n_total/1e6:.2f}M total  |  {n_decoder/1e6:.2f}M decoder trainable")
    print("S1 branches and CLIP are frozen.")

    # ── Text features ─────────────────────────────────────────────────────────
    text_feats = build_text_features(args.class_json, clip_model, device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        decoder_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)
    clip_skip_indices = tuple(args.clip_skip_indices)
    warmup_epochs = max(1, args.epochs // 5)

    best_val    = float("inf")
    best_path   = os.path.join(args.output_dir, "s2_phase3_best.pth")
    latest_path = os.path.join(args.output_dir, "s2_phase3_latest.pth")

    print(f"\nPhase 3: {args.epochs} epochs  lr={args.lr:.1e}  loss=CE only")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, warmup_epochs, args.lr)
        t0 = time.time()

        train_loss = run_epoch(
            train_loader, student2, text_feats, optimizer, scaler,
            device, args.amp, clip_skip_indices, decoder_params,
            train=True, epoch=epoch, total_epochs=args.epochs,
            grad_accum=args.grad_accum,
        )
        val_loss = run_epoch(
            val_loader, student2, text_feats, optimizer, scaler,
            device, args.amp, clip_skip_indices, decoder_params,
            train=False, epoch=epoch, total_epochs=args.epochs,
            grad_accum=args.grad_accum,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"lr={lr:.2e}  train_ce={train_loss:.4f}  val_ce={val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )

        save_ckpt(latest_path, epoch, student2, val_loss, args)
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(best_path, epoch, student2, best_val, args)
            print(f"  New best val CE: {best_val:.4f}  -> {best_path}")

    print(f"\nPhase 3 done. Best val CE: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")
    print("Phase 2 weights in --phase2-ckpt were not modified.")


if __name__ == "__main__":
    main()
