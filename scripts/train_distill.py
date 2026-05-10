"""
Phase 2 — Train the LightweightDecoder student via knowledge distillation.

The student (frozen TIPSv2-L/14 + trainable LightweightDecoder) is supervised
entirely by the cached teacher logits (no segmentation labels needed).

Usage:
    python scripts/train_distill.py \\
        --tips-dir    checkpoints/tipsv2-l14 \\
        --image-dir   gs_net/data/datasets/LandDiscover_50K/TR_Image \\
        --cache-dir   cache_logits/ \\
        --output-dir  output/distill/ \\
        [--epochs 40] [--batch-size 16] [--lr 3e-4] [--tau 4.0] [--amp] \\
        [--resume output/distill/student_latest.pth]

Output layout:
    output/distill/
        checkpoints/
            student_ep005.pth  ... student_ep040.pth   (every 5 epochs, weights only)
        student_best.pth        (best val loss, weights only)
        student_latest.pth      (latest epoch, weights + optimizer, for resuming)
        training_log.json       (per-epoch losses + run settings, updated each epoch)
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from gs_distill.student import TIPSDistillStudent
from gs_distill.data    import build_dataloaders
from gs_distill.losses  import distillation_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tips-dir",      required=True,  help="Path to TIPSv2-L/14 model directory")
    p.add_argument("--image-dir",     required=True,  help="LD50K TR_Image directory")
    p.add_argument("--cache-dir",     required=True,  help="Teacher logit cache directory")
    p.add_argument("--output-dir",    default="output/distill")
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--tau",           type=float, default=4.0,  help="Distillation temperature")
    p.add_argument("--num-classes",   type=int,   default=40)
    p.add_argument("--hidden-dim",    type=int,   default=128)
    p.add_argument("--resolution",    type=int,   default=336)
    p.add_argument("--val-fraction",  type=float, default=0.05)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--warmup-epochs", type=int,   default=4)
    p.add_argument("--save-interval", type=int,   default=5,
                   help="Save a numbered checkpoint every N epochs (default 5)")
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",        default=None, help="Checkpoint path to resume from")
    p.add_argument("--log-interval",  type=int,   default=50)
    return p.parse_args()


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def run_epoch(student, loader, optimizer, scaler, device, amp, tau, train, log_interval=50):
    student.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    total_loss, n_batches = 0.0, 0

    with ctx:
        for step, (images, teacher_logits) in enumerate(
            tqdm(loader, leave=False, desc="train" if train else "val")
        ):
            images         = images.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                student_logits = student(images)
                loss = distillation_loss(student_logits, teacher_logits, tau=tau)

            if train:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(student.trainable_parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.trainable_parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if train and (step + 1) % log_interval == 0:
                tqdm.write(f"  step {step+1:5d}  loss={total_loss/n_batches:.4f}")

    return total_loss / max(n_batches, 1)


def save_weights_only(ckpt: dict, path: str):
    """Save checkpoint without optimizer state."""
    torch.save({k: v for k, v in ckpt.items() if k != "optimizer"}, path)


def main():
    args = parse_args()

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    # ── Build student ─────────────────────────────────────────────────────────
    print(f"Loading TIPSv2-L/14 from {args.tips_dir} ...")
    student = TIPSDistillStudent(
        tips_dir=args.tips_dir,
        num_classes=args.num_classes,
        hidden=args.hidden_dim,
    ).to(device)

    n_trainable = sum(p.numel() for p in student.trainable_parameters())
    n_total     = sum(p.numel() for p in student.parameters())
    print(f"Parameters — trainable (decoder): {n_trainable/1e6:.2f}M  |  total: {n_total/1e6:.2f}M")

    # ── Optimizer & scaler ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp)

    start_epoch   = 0
    best_val_loss = float("inf")
    best_epoch    = -1
    epoch_log     = []

    if args.resume:
        resume_ckpt   = torch.load(args.resume, map_location=device)
        student.load_state_dict(resume_ckpt["student"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch   = resume_ckpt.get("epoch", 0) + 1
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        epoch_log     = resume_ckpt.get("epoch_log", [])
        best_epoch    = resume_ckpt.get("best_epoch", -1)
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("Building dataloaders ...")
    train_loader, val_loader = build_dataloaders(
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        resolution=args.resolution,
    )
    print(f"  Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}")

    run_start = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)

        t0 = time.time()
        train_loss = run_epoch(
            student, train_loader, optimizer, scaler, device,
            args.amp, args.tau, train=True, log_interval=args.log_interval,
        )
        val_loss = run_epoch(
            student, val_loader, optimizer, scaler, device,
            args.amp, args.tau, train=False,
        )
        elapsed = time.time() - t0
        ep1 = epoch + 1

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = ep1

        print(
            f"Epoch {ep1:3d}/{args.epochs}  lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"({'best' if is_best else f'{elapsed:.0f}s'})"
        )

        # ── Full checkpoint for resuming (every epoch, overwrites) ────────────
        full_ckpt = {
            "epoch":         epoch,
            "student":       student.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch":    best_epoch,
            "epoch_log":     epoch_log,
            "args":          vars(args),
        }
        torch.save(full_ckpt, out_dir / "student_latest.pth")

        # ── Periodic checkpoint every save_interval epochs ────────────────────
        if ep1 % args.save_interval == 0:
            path = ckpt_dir / f"student_ep{ep1:03d}.pth"
            save_weights_only(full_ckpt, str(path))
            print(f"  Checkpoint saved -> checkpoints/student_ep{ep1:03d}.pth")

        # ── Best checkpoint ───────────────────────────────────────────────────
        if is_best:
            save_weights_only(full_ckpt, str(out_dir / "student_best.pth"))
            print(f"  New best val loss: {val_loss:.4f} -> student_best.pth")

        # ── Update training log (written every epoch) ─────────────────────────
        epoch_log.append({
            "epoch":      ep1,
            "lr":         round(lr, 8),
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "is_best":    is_best,
            "epoch_time_s": round(elapsed, 1),
        })

        log_data = {
            "run_settings": {
                "tips_dir":      args.tips_dir,
                "image_dir":     args.image_dir,
                "cache_dir":     args.cache_dir,
                "output_dir":    args.output_dir,
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "tau":           args.tau,
                "num_classes":   args.num_classes,
                "hidden_dim":    args.hidden_dim,
                "resolution":    args.resolution,
                "val_fraction":  args.val_fraction,
                "warmup_epochs": args.warmup_epochs,
                "save_interval": args.save_interval,
                "amp":           args.amp,
                "device":        args.device,
                "trainable_params_M": round(n_trainable / 1e6, 3),
            },
            "epochs": epoch_log,
            "summary": {
                "best_epoch":     best_epoch,
                "best_val_loss":  round(best_val_loss, 6),
                "last_epoch":     ep1,
                "total_time_min": round((time.time() - run_start) / 60, 1),
                "completed":      ep1 == args.epochs,
                "started_at":     datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        }
        with open(out_dir / "training_log.json", "w") as f:
            json.dump(log_data, f, indent=2)

    total_min = (time.time() - run_start) / 60
    print(f"\nDone. Best epoch: {best_epoch}  val_loss: {best_val_loss:.4f}  "
          f"Total time: {total_min:.1f} min")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Log:         {out_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
