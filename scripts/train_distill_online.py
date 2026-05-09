"""
Online distillation — teacher forward and student update happen together each batch.
No disk cache required.

For each batch the frozen GSNet teacher captures four tensors:
  fused_corr_embed  (hidden_dim, T, 24, 24)  — CLIP+DINO correlation entering RIPD aggregators
  clip_embed_corr   (hidden_dim, T, 24, 24)  — CLIP-only correlation embedding (before DINO fusion)
  dino_L4           (768, 48, 48)             — DINOv3 intermediate layer 4
  dino_L8           (768, 48, 48)             — DINOv3 intermediate layer 8

The student (frozen CLIP multi-layer → 4 specialist heads) predicts all four.
Loss: Smooth L1 + cosine per head.

Usage:
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/train_distill_online.py \\
        --gsnet-config  configs/vitb_384.yaml \\
        --gsnet-weights path/to/gsnet.pth \\
        --image-dir     path/to/LD50K/images \\
        --output-dir    output/distill/ \\
        [--epochs 30] [--batch-size 8] [--lr 1e-4] [--amp] [--device cuda]
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
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

import wandb
import clip as openai_clip

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from gs_net import add_cat_seg_config
import gs_net  # noqa: side-effect registrations

from gs_distill.student import GSDistillStudent
from gs_distill.losses import distillation_loss_per_branch


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

# ─────────────────────────────────────────────────────────────────────────────
# Teacher capture state (written by patches, read by training loop)
# ─────────────────────────────────────────────────────────────────────────────
_hook_state = {"fused_corr_embed": None, "clip_embed_corr": None}
_dino_l4    = [None]
_dino_l8    = [None]


# ─────────────────────────────────────────────────────────────────────────────
# Teacher patches
# ─────────────────────────────────────────────────────────────────────────────

def _patch_ripd(ripd_module):
    """
    Patch RIPD.forward to capture fused_corr_embed and clip_embed_corr
    immediately after QGFF, before the AggregatorLayers run.
    """
    original_forward = ripd_module.forward

    def patched_forward(img_feats, dino_feat, text_feats, appearance_guidance, dino_guidance):
        if dino_feat is not None and img_feats is not None:
            if ripd_module.fusion_type == 'query_guided':
                corr      = ripd_module.correlation(img_feats, text_feats)
                dino_corr = ripd_module.correlation(dino_feat, text_feats)
                fused_corr, clip_embed_corr, _ = ripd_module.corr_fusion_embed_seperate(
                    clip_corr=corr, dino_corr=dino_corr,
                )
                fused_corr_embed = fused_corr + clip_embed_corr
            elif ripd_module.fusion_type == 'simple_concatenation':
                simple_corr = ripd_module.simple_concatenation_corr(img_feats, dino_feat, text_feats)
                T = simple_corr.shape[2]
                fused_corr_embed = ripd_module.simple_concatenation_corr_embed(
                    rearrange(simple_corr, "B C T H W -> (B T) C H W"))
                fused_corr_embed = rearrange(fused_corr_embed, "(B T) C H W -> B C T H W", T=T)
                clip_embed_corr  = torch.zeros_like(fused_corr_embed)
            elif ripd_module.fusion_type == 'fusion_query':
                fused_feat = ripd_module.fusion_feats(torch.cat([img_feats, dino_feat], dim=1))
                corr = ripd_module.correlation(fused_feat, text_feats)
                fused_corr_embed = ripd_module.corr_embed(corr)
                clip_embed_corr  = torch.zeros_like(fused_corr_embed)
        elif dino_feat is not None:
            corr = ripd_module.correlation(dino_feat, text_feats)
            fused_corr_embed = ripd_module.corr_embed(corr)
            clip_embed_corr  = torch.zeros_like(fused_corr_embed)
        elif img_feats is not None:
            corr = ripd_module.correlation(img_feats, text_feats)
            fused_corr_embed = ripd_module.corr_embed(corr)
            clip_embed_corr  = torch.zeros_like(fused_corr_embed)

        _hook_state["fused_corr_embed"] = fused_corr_embed.detach().cpu().half()
        _hook_state["clip_embed_corr"]  = clip_embed_corr.detach().cpu().half()

        return original_forward(img_feats, dino_feat, text_feats, appearance_guidance, dino_guidance)

    ripd_module.forward = patched_forward

    def unpatch():
        ripd_module.forward = original_forward
    return unpatch


def _patch_dino(dino_model):
    """
    Patch dino_model.get_intermediate_layers so that L4 and L8 features
    are captured as a side effect of the single DINOv3 forward pass.
    """
    original_fn = dino_model.get_intermediate_layers

    def patched_fn(imgs, n=12):
        result = original_fn(imgs, n)
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
# Image-only dataset — raw [0, 1] tensors, no normalisation applied here
# ─────────────────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
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
        return TF.to_tensor(img)   # [0, 1] float32


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def clip_normalise(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [0, 1] → CLIP-normalised."""
    mean = _CLIP_MEAN.to(images.device).view(1, 3, 1, 1)
    std  = _CLIP_STD.to(images.device).view(1, 3, 1, 1)
    return (images - mean) / std


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

    # Force LD50K text vocabulary regardless of TEST_CLASS_JSON in the config
    predictor = model.sem_seg_head.predictor
    predictor.test_class_texts = CLASSES_LandDiscover50K
    predictor.cache = None

    return model


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
# Training / validation epoch
# ─────────────────────────────────────────────────────────────────────────────
_BRANCH_KEYS = ("fused_corr_embed", "clip_embed_corr", "dino_L4", "dino_L8")


def run_epoch(teacher, student, loader, optimizer, scaler, device, amp,
              train=True, log_interval=50, epoch=0, use_wandb=False):
    student.train(train)
    total_loss = 0.0
    branch_totals = {k: 0.0 for k in _BRANCH_KEYS}
    n = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for step, images in enumerate(tqdm(loader, leave=False, desc="train" if train else "val")):
            images = images.to(device)

            # ── Teacher forward (always no_grad) ───────────────────────────
            with torch.no_grad():
                teacher([{"image": (img * 255.0).clamp(0, 255)} for img in images])

            targets = {
                "fused_corr_embed": _hook_state["fused_corr_embed"].to(device).float(),
                "clip_embed_corr":  _hook_state["clip_embed_corr"].to(device).float(),
                "dino_L4": _dino_l4[0].to(device).float(),
                "dino_L8": _dino_l8[0].to(device).float(),
            }

            # ── Student forward ────────────────────────────────────────────
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
                    f"fused={branch_totals['fused_corr_embed']/n:.4f}  "
                    f"clip={branch_totals['clip_embed_corr']/n:.4f}  "
                    f"l4={branch_totals['dino_L4']/n:.4f}  "
                    f"l8={branch_totals['dino_L8']/n:.4f}"
                )
                if use_wandb:
                    global_step = epoch * len(loader) + step + 1
                    wandb.log({
                        "step/loss":    avg,
                        "step/fused":   branch_totals["fused_corr_embed"] / n,
                        "step/clip":    branch_totals["clip_embed_corr"]  / n,
                        "step/dino_l4": branch_totals["dino_L4"] / n,
                        "step/dino_l8": branch_totals["dino_L8"] / n,
                    }, step=global_step)

    avg_loss     = total_loss / max(n, 1)
    avg_branches = {k: v / max(n, 1) for k, v in branch_totals.items()}
    return avg_loss, avg_branches


# ─────────────────────────────────────────────────────────────────────────────
# Args + main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",    required=True)
    p.add_argument("--gsnet-weights",   required=True)
    p.add_argument("--image-dir",       required=True)
    p.add_argument("--output-dir",      default="output/distill")
    p.add_argument("--clip-pretrained", default="ViT-B/16")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--d-dino",          type=int,   default=768)
    p.add_argument("--num-classes",     type=int,   default=40)
    p.add_argument("--clip-layers",     type=int,   nargs="+", default=[4, 8, 10, 12])
    p.add_argument("--val-fraction",    type=float, default=0.05)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",          default=None)
    p.add_argument("--log-interval",    type=int,   default=50)
    p.add_argument("--wandb-project",   default="gs-distill")
    p.add_argument("--wandb-run",       default=None, help="W&B run name (auto if omitted)")
    p.add_argument("--no-wandb",        action="store_true", help="Disable W&B logging")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Teacher ──────────────────────────────────────────────────────────────
    print(f"Loading teacher from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    teacher = teacher.to(device)

    unpatch_ripd = _patch_ripd(teacher.sem_seg_head.predictor.transformer)
    unpatch_dino = _patch_dino(teacher.dino_model)

    # ── Student ───────────────────────────────────────────────────────────────
    print(f"Loading CLIP {args.clip_pretrained} for student ...")
    clip_model, _ = openai_clip.load(args.clip_pretrained, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=args.hidden_dim,
        d_dino=args.d_dino,
        num_classes=args.num_classes,
        clip_layers=args.clip_layers,
    ).to(device)

    n_params = sum(p.numel() for p in student.trainable_parameters())
    print(f"Student trainable params: {n_params / 1e6:.2f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp)
    start_epoch, best_val_loss = 0, float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["student"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Data ──────────────────────────────────────────────────────────────────
    full_ds = ImageDataset(args.image_dir, resolution=384)
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
    print(f"  Train: {n_train}  Val: {n_val}")

    warmup_epochs = max(1, args.epochs // 10)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, warmup_epochs, args.lr)
        t0 = time.time()

        train_loss, train_b = run_epoch(
            teacher, student, train_loader, optimizer, scaler,
            device, args.amp, train=True, log_interval=args.log_interval,
            epoch=epoch, use_wandb=use_wandb,
        )
        val_loss, val_b = run_epoch(
            teacher, student, val_loader, optimizer, scaler,
            device, args.amp, train=False,
            epoch=epoch, use_wandb=False,   # val has no step-level logs
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.0f}s)  "
            f"val_fused={val_b['fused_corr_embed']:.4f}  "
            f"val_clip={val_b['clip_embed_corr']:.4f}  "
            f"val_l4={val_b['dino_L4']:.4f}  "
            f"val_l8={val_b['dino_L8']:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch":            epoch + 1,
                "lr":               lr,
                "epoch_time_s":     elapsed,
                "train/loss":       train_loss,
                "train/fused":      train_b["fused_corr_embed"],
                "train/clip":       train_b["clip_embed_corr"],
                "train/dino_l4":    train_b["dino_L4"],
                "train/dino_l8":    train_b["dino_L8"],
                "val/loss":         val_loss,
                "val/fused":        val_b["fused_corr_embed"],
                "val/clip":         val_b["clip_embed_corr"],
                "val/dino_l4":      val_b["dino_L4"],
                "val/dino_l8":      val_b["dino_L8"],
                "best_val_loss":    best_val_loss,
            }, step=epoch + 1)

        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, os.path.join(args.output_dir, "student_latest.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "student_best.pth")
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ New best val loss: {val_loss:.4f} → {best_path}")

    unpatch_ripd()
    unpatch_dino()
    if use_wandb:
        wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
