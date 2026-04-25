"""
Phase 3 — Train the student conv head via knowledge distillation.

The student is supervised entirely by the cached teacher outputs (no segmentation labels).

Usage:
    python scripts/train_distill.py \\
        --image-dir  path/to/LD50K/images \\
        --cache-dir  cache/ \\
        --clip-pretrained ViT-B/16 \\
        --output-dir output/distill/ \\
        [--epochs 30] \\
        [--batch-size 16] \\
        [--lr 1e-4] \\
        [--hidden-dim 128] \\
        [--num-classes 40] \\
        [--clip-layers 4 8 10 12] \\
        [--val-fraction 0.05] \\
        [--amp] \\
        [--device cuda]

The best checkpoint (by validation loss) is saved as {output-dir}/student_best.pth.
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import clip as openai_clip   # pip install git+https://github.com/openai/CLIP

from gs_distill.student import GSDistillStudent
from gs_distill.data    import build_dataloaders
from gs_distill.losses  import distillation_loss_per_branch


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir",       required=True)
    p.add_argument("--cache-dir",       required=True)
    p.add_argument("--output-dir",      default="output/distill")
    p.add_argument("--clip-pretrained", default="ViT-B/16")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--d-dino",          type=int,   default=768)
    p.add_argument("--num-classes",     type=int,   default=40)
    p.add_argument("--clip-layers",     type=int,   nargs="+", default=[4, 8, 10, 12])
    p.add_argument("--val-fraction",    type=float, default=0.05)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--amp",             action="store_true", help="Mixed-precision training")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",          default=None, help="Path to checkpoint to resume from")
    p.add_argument("--log-interval",    type=int,   default=50)
    return p.parse_args()


def cosine_lr_schedule(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    """In-place cosine decay with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def run_epoch(student, loader, optimizer, scaler, device, amp, train=True, log_interval=50):
    student.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_loss = 0.0
    counts = {"fused_corr_embed": 0.0, "dino_L4": 0.0, "dino_L8": 0.0}
    n_batches = 0

    with ctx:
        for step, (images, targets) in enumerate(tqdm(loader, leave=False, desc="train" if train else "val")):
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                pred = student(images)
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
            for k in counts:
                counts[k] += breakdown[k]
            n_batches += 1

            if train and (step + 1) % log_interval == 0:
                avg = total_loss / n_batches
                tqdm.write(
                    f"  step {step+1:5d}  loss={avg:.4f}  "
                    f"fused={counts['fused_corr_embed']/n_batches:.4f}  "
                    f"l4={counts['dino_L4']/n_batches:.4f}  "
                    f"l8={counts['dino_L8']/n_batches:.4f}"
                )

    avg_loss = total_loss / max(n_batches, 1)
    avg_per_branch = {k: v / max(n_batches, 1) for k, v in counts.items()}
    return avg_loss, avg_per_branch


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load frozen CLIP ─────────────────────────────────────────────────────
    print(f"Loading CLIP {args.clip_pretrained} ...")
    clip_model, _ = openai_clip.load(args.clip_pretrained, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ── Build student ────────────────────────────────────────────────────────
    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=args.hidden_dim,
        d_dino=args.d_dino,
        num_classes=args.num_classes,
        clip_layers=args.clip_layers,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"Student trainable parameters: {n_params/1e6:.2f}M")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["student"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    print("Building dataloaders ...")
    train_loader, val_loader = build_dataloaders(
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        resolution=384,
    )
    print(f"  Train: {len(train_loader.dataset)} samples  |  Val: {len(val_loader.dataset)} samples")

    warmup_epochs = max(1, args.epochs // 10)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr_schedule(optimizer, epoch, args.epochs, warmup_epochs, args.lr)

        t0 = time.time()
        train_loss, train_branches = run_epoch(
            student, train_loader, optimizer, scaler, device, args.amp,
            train=True, log_interval=args.log_interval,
        )
        val_loss, val_branches = run_epoch(
            student, val_loader, optimizer, scaler, device, args.amp,
            train=False,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"({elapsed:.0f}s)  "
            f"val_fused={val_branches['fused_corr_embed']:.4f}  "
            f"val_l4={val_branches['dino_L4']:.4f}  "
            f"val_l8={val_branches['dino_L8']:.4f}"
        )

        # Save latest checkpoint
        ckpt_path = os.path.join(args.output_dir, "student_latest.pth")
        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, ckpt_path)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "student_best.pth")
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ New best val loss: {val_loss:.4f} → saved to {best_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
