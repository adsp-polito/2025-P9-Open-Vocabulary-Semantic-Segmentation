"""
Phase 2 (v2) — Distillation with feature-level supervision and partial TIPS unfreezing.

v2 changes vs v1:
  - TIPS last --unfreeze-blocks blocks unfrozen (default 4), trained at backbone-lr (1e-5).
  - Feature distillation loss added: 1 - cosine_sim(adapter_out, cached_clip_feats).
    Combined loss: L = L_logit + feat-loss-weight * L_feat  (default weight 0.5).
  - Two optimizer param groups: backbone (1e-5) and head/adapter/decoder (3e-4).
  - Output goes to output_v3/distill/ to keep v1 results separate.

Usage:
    python scripts/train_distill.py \\
        --tips-dir        checkpoints/tipsv2-l14 \\
        --image-dir       gs_net/data/datasets/LandDiscover_50K/TR_Image \\
        --cache-dir       cache_logits/ \\
        --clip-feats-dir  cache_clip_feats/ \\
        --output-dir      output_v3/distill/ \\
        [--unfreeze-blocks 4] [--backbone-lr 1e-5] [--feat-loss-weight 0.5] \\
        [--epochs 40] [--batch-size 8] [--lr 3e-4] [--tau 4.0] [--amp]
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
from gs_distill.losses  import distillation_loss, feature_distill_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tips-dir",          required=True)
    p.add_argument("--image-dir",         required=True)
    p.add_argument("--cache-dir",         required=True,  help="Teacher logit cache dir")
    p.add_argument("--clip-feats-dir",    default=None,   help="CLIP feature cache dir (v2)")
    p.add_argument("--output-dir",        default="output_v3/distill")
    p.add_argument("--epochs",            type=int,   default=40)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=3e-4,  help="Head/adapter LR")
    p.add_argument("--backbone-lr",       type=float, default=1e-5,  help="Unfrozen TIPS blocks LR")
    p.add_argument("--unfreeze-blocks",   type=int,   default=4,     help="TIPS blocks to unfreeze")
    p.add_argument("--feat-loss-weight",  type=float, default=0.5,   help="Weight for L_feat")
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--tau",               type=float, default=4.0)
    p.add_argument("--num-classes",       type=int,   default=40)
    p.add_argument("--hidden-dim",        type=int,   default=128)
    p.add_argument("--bottleneck",        type=int,   default=256)
    p.add_argument("--resolution",        type=int,   default=336)
    p.add_argument("--val-fraction",      type=float, default=0.05)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--warmup-epochs",     type=int,   default=4)
    p.add_argument("--save-interval",     type=int,   default=5)
    p.add_argument("--amp",               action="store_true")
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",            default=None)
    p.add_argument("--log-interval",      type=int,   default=50)
    return p.parse_args()


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lrs, min_lr=1e-6):
    """Apply cosine schedule independently to each param group using its base_lr."""
    lrs = []
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        pg["lr"] = lr
        lrs.append(lr)
    return lrs


def run_epoch(student, loader, optimizer, scaler, device, amp, tau,
              train, use_feats, feat_weight, log_interval=50):
    student.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    total_loss, total_l_logit, total_l_feat = 0.0, 0.0, 0.0
    n_batches = 0

    with ctx:
        for step, batch in enumerate(
            tqdm(loader, leave=False, desc="train" if train else "val")
        ):
            if use_feats:
                images, teacher_logits, clip_feats = batch
                clip_feats = clip_feats.to(device, non_blocking=True)
            else:
                images, teacher_logits = batch
                clip_feats = None

            images         = images.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                if use_feats:
                    student_logits, adapted_feats = student(images, return_feats=True)
                    l_logit = distillation_loss(student_logits, teacher_logits, tau=tau)
                    l_feat  = feature_distill_loss(adapted_feats, clip_feats)
                    loss    = l_logit + feat_weight * l_feat
                else:
                    student_logits = student(images)
                    l_logit = distillation_loss(student_logits, teacher_logits, tau=tau)
                    l_feat  = torch.tensor(0.0)
                    loss    = l_logit

            if train:
                all_params = student.trainable_parameters() + student.backbone_parameters()
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()

            total_loss   += loss.item()
            total_l_logit += l_logit.item()
            total_l_feat  += l_feat.item()
            n_batches     += 1

            if train and (step + 1) % log_interval == 0:
                tqdm.write(
                    f"  step {step+1:5d}  loss={total_loss/n_batches:.4f}"
                    f"  l_logit={total_l_logit/n_batches:.4f}"
                    + (f"  l_feat={total_l_feat/n_batches:.4f}" if use_feats else "")
                )

    N = max(n_batches, 1)
    return total_loss / N, total_l_logit / N, total_l_feat / N


def save_weights_only(ckpt: dict, path: str):
    """Save checkpoint without optimizer state."""
    torch.save({k: v for k, v in ckpt.items() if k != "optimizer"}, path)


def main():
    args = parse_args()

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    device    = torch.device(args.device)
    use_feats = args.clip_feats_dir is not None

    # ── Build student ─────────────────────────────────────────────────────────
    print(f"Loading TIPSv2-L/14 from {args.tips_dir} ...")
    student = TIPSDistillStudent(
        tips_dir=args.tips_dir,
        num_classes=args.num_classes,
        hidden=args.hidden_dim,
        bottleneck=args.bottleneck,
        unfreeze_blocks=args.unfreeze_blocks,
    ).to(device)

    n_head     = sum(p.numel() for p in student.trainable_parameters())
    n_backbone = sum(p.numel() for p in student.backbone_parameters())
    n_total    = sum(p.numel() for p in student.parameters())
    print(f"Parameters — head/adapter: {n_head/1e6:.2f}M  "
          f"| backbone (unfrozen): {n_backbone/1e6:.2f}M  "
          f"| total: {n_total/1e6:.2f}M")
    print(f"Feature distillation: {'ON  (clip_feats_dir=' + args.clip_feats_dir + ')' if use_feats else 'OFF'}")

    # ── Optimizer: two param groups when backbone is partially unfrozen ───────
    param_groups = [{"params": student.trainable_parameters(), "lr": args.lr}]
    if n_backbone > 0:
        param_groups.append({"params": student.backbone_parameters(), "lr": args.backbone_lr})
    base_lrs = [args.lr] + ([args.backbone_lr] if n_backbone > 0 else [])

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler    = GradScaler(enabled=args.amp)

    start_epoch   = 0
    best_val_loss = float("inf")
    best_epoch    = -1
    epoch_log     = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["student"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epoch_log     = ckpt.get("epoch_log", [])
        best_epoch    = ckpt.get("best_epoch", -1)
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
        clip_feats_dir=args.clip_feats_dir,
    )
    print(f"  Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}")

    run_start = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        lrs = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, base_lrs)

        t0 = time.time()
        train_loss, tl_logit, tl_feat = run_epoch(
            student, train_loader, optimizer, scaler, device,
            args.amp, args.tau, train=True,
            use_feats=use_feats, feat_weight=args.feat_loss_weight,
            log_interval=args.log_interval,
        )
        val_loss, vl_logit, vl_feat = run_epoch(
            student, val_loader, optimizer, scaler, device,
            args.amp, args.tau, train=False,
            use_feats=use_feats, feat_weight=args.feat_loss_weight,
        )
        elapsed = time.time() - t0
        ep1 = epoch + 1

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = ep1

        lr_str = f"lr={lrs[0]:.1e}" + (f"/bb={lrs[1]:.1e}" if len(lrs) > 1 else "")
        print(
            f"Epoch {ep1:3d}/{args.epochs}  {lr_str}  "
            f"train={train_loss:.4f}(l={tl_logit:.3f},f={tl_feat:.3f})  "
            f"val={val_loss:.4f}(l={vl_logit:.3f},f={vl_feat:.3f})  "
            f"{'** best **' if is_best else f'{elapsed:.0f}s'}"
        )

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

        if ep1 % args.save_interval == 0:
            save_weights_only(full_ckpt, str(ckpt_dir / f"student_ep{ep1:03d}.pth"))
            print(f"  Checkpoint -> checkpoints/student_ep{ep1:03d}.pth")

        if is_best:
            save_weights_only(full_ckpt, str(out_dir / "student_best.pth"))
            print(f"  Best -> student_best.pth")

        epoch_log.append({
            "epoch": ep1, "lr": round(lrs[0], 8),
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
            "l_logit_train": round(tl_logit, 6), "l_feat_train": round(tl_feat, 6),
            "l_logit_val": round(vl_logit, 6),   "l_feat_val": round(vl_feat, 6),
            "is_best": is_best, "epoch_time_s": round(elapsed, 1),
        })
        with open(out_dir / "training_log.json", "w") as f:
            json.dump({
                "run_settings": vars(args),
                "epochs": epoch_log,
                "summary": {
                    "best_epoch": best_epoch,
                    "best_val_loss": round(best_val_loss, 6),
                    "last_epoch": ep1,
                    "total_time_min": round((time.time() - run_start) / 60, 1),
                    "completed": ep1 == args.epochs,
                },
            }, f, indent=2)

    total_min = (time.time() - run_start) / 60
    print(f"\nDone. Best epoch: {best_epoch}  val_loss: {best_val_loss:.4f}  "
          f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
