"""
Phase 3 — Fine-tune the distilled student on LD50K with GT supervision.

Mixed loss:
  L = λ × L_distill  +  (1-λ) × L_gt
  L_distill = temperature-scaled BCE vs cached teacher logits (40-class, 96x96)
  L_gt      = CrossEntropy vs LD50K GT masks (40-class, resized to 96x96)

Architecture:
  TIPS vision encoder (last N blocks unfrozen) → patch tokens
  → SpatialAdapter → L2-normalise
  → SpatialProjector → spatial context (B, 64, 24, 24)
  → einsum with text embeddings → correlation (B, K, 24, 24)
  → tile + concat → (B*K, 65, 24, 24)
  → ClassAgnosticDecoder → (B*K, 1, 96, 96) → (B, K, 96, 96)

For the scratch ablation baseline, pass --no-distill-init: adapter starts from
random weights while TIPS still loads its pretrained weights.

Projector and decoder always start from random init in both runs.

Usage:
    python scripts/train_finetune.py \\
        --checkpoint  output/distill/student_best.pth \\
        --tips-dir    checkpoints/tipsv2-l14 \\
        --image-dir   gs_net/data/datasets/LandDiscover_50K/TR_Image \\
        --gt-dir      gs_net/data/datasets/LandDiscover_50K/GT_ID \\
        --cache-dir   cache_logits/ \\
        --output-dir  output/finetune \\
        [--no-distill-init] \\
        [--epochs 25] [--batch-size 8] [--lr 1e-4] [--backbone-lr 1e-5] [--amp]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from gs_distill.decoder_agnostic import ClassAgnosticDecoder, SpatialProjector


# ── SpatialAdapter (inline — same architecture as Phase 2) ────────────────────

class SpatialAdapter(nn.Module):
    def __init__(self, embed_dim=1024, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, bottleneck, kernel_size=1),
            nn.GroupNorm(8, bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.GroupNorm(8, bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, embed_dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


# ── TIPS loader ───────────────────────────────────────────────────────────────

def _load_tips(tips_dir: str):
    tips_path = Path(tips_dir)
    pkg = "tips_local"
    if pkg not in sys.modules:
        pkg_mod = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(pkg, None, is_package=True)
        )
        pkg_mod.__path__ = [str(tips_path)]
        pkg_mod.__package__ = pkg
        sys.modules[pkg] = pkg_mod

    def _load_module(name):
        full_name = f"{pkg}.{name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        spec = importlib.util.spec_from_file_location(
            full_name, str(tips_path / f"{name}.py"),
            submodule_search_locations=[str(tips_path)]
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_module("configuration_tips")
    _load_module("image_encoder")
    _load_module("text_encoder")
    tips_mod = _load_module("modeling_tips")
    config_mod = sys.modules[f"{pkg}.configuration_tips"]
    config = config_mod.TIPSv2Config.from_pretrained(tips_dir)
    return tips_mod.TIPSv2Model(config)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LD50KFinetuneDataset(Dataset):
    """
    Returns (image, gt_mask_96, teacher_logits) triples from LD50K.
      image:           (3, 336, 336) float32 in [0, 1]
      gt_mask_96:      (96, 96)      int64   class labels 0-39
      teacher_logits:  (40, 96, 96)  float32

    Augmentation (train=True): random horizontal and vertical flips applied
    consistently to all three tensors.
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, gt_dir, cache_dir,
                 resolution=336, gt_size=96, augment=False):
        self.resolution = resolution
        self.gt_size    = gt_size
        self.augment    = augment

        img_map   = {p.stem: p for p in Path(image_dir).iterdir()
                     if p.suffix.lower() in self.EXTENSIONS}
        gt_map    = {p.stem: p for p in Path(gt_dir).iterdir()
                     if p.suffix.lower() in self.EXTENSIONS}
        cache_map = {p.stem: p for p in Path(cache_dir).glob("*.pt")}

        common = set(img_map) & set(gt_map) & set(cache_map)
        if not common:
            raise RuntimeError(
                f"No three-way matches.\n"
                f"  images={len(img_map)} gts={len(gt_map)} caches={len(cache_map)}"
            )
        self.samples = [(img_map[s], gt_map[s], cache_map[s]) for s in sorted(common)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, cache_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        image = torch.from_numpy(np.array(img)).float() / 255.0
        image = image.permute(2, 0, 1)  # (3, H, W)

        gt = Image.open(gt_path).convert("L")
        gt = gt.resize((self.gt_size, self.gt_size), Image.NEAREST)
        gt_mask = torch.from_numpy(np.array(gt)).long()  # (96, 96)

        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        teacher_logits = cache["logits"].float()  # (40, 96, 96)

        if self.augment:
            # Random horizontal flip — applied consistently to all spatial tensors
            if random.random() < 0.5:
                image          = torch.flip(image, dims=[-1])
                gt_mask        = torch.flip(gt_mask.unsqueeze(0), dims=[-1]).squeeze(0)
                teacher_logits = torch.flip(teacher_logits, dims=[-1])
            # Random vertical flip
            if random.random() < 0.5:
                image          = torch.flip(image, dims=[-2])
                gt_mask        = torch.flip(gt_mask.unsqueeze(0), dims=[-2]).squeeze(0)
                teacher_logits = torch.flip(teacher_logits, dims=[-2])

        return image, gt_mask, teacher_logits


# ── Loss ──────────────────────────────────────────────────────────────────────

def mixed_loss(student_logits, teacher_logits, gt_mask, tau=4.0, lam=0.5):
    t_soft    = torch.sigmoid(teacher_logits / tau)
    s_soft    = student_logits / tau
    l_distill = F.binary_cross_entropy_with_logits(s_soft, t_soft) * (tau ** 2)
    l_gt      = F.cross_entropy(student_logits, gt_mask, ignore_index=255)
    return lam * l_distill + (1.0 - lam) * l_gt, l_distill.item(), l_gt.item()


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, base_lrs, min_lr=1e-7):
    scale = 0.5 * (1 + math.cos(math.pi * epoch / max(1, total_epochs)))
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = min_lr + (base - min_lr) * scale


# ── Epoch runner ──────────────────────────────────────────────────────────────

def run_epoch(tips, adapter, projector, decoder, text_emb, loader,
              optimizer, scaler, device, amp, tau, lam, train=True):
    tips.train(train)
    adapter.train(train)
    projector.train(train)
    decoder.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    total, d_sum, g_sum, n = 0.0, 0.0, 0.0, 0

    with ctx:
        for images, gt_masks, teacher_logits in tqdm(
            loader, desc="train" if train else "val ", leave=False
        ):
            images         = images.to(device, non_blocking=True)
            gt_masks       = gt_masks.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                # TIPS encoder
                out     = tips.encode_image(images)
                patches = out.patch_tokens.float()         # (B, N, 1024)
                B, N, D = patches.shape
                H = W   = int(N ** 0.5)                    # 24
                feats   = patches.permute(0, 2, 1).reshape(B, D, H, W)

                # SpatialAdapter + L2-normalise
                feats = adapter(feats)                     # (B, 1024, 24, 24)
                feats = F.normalize(feats, dim=1)

                # Spatial context branch
                spatial_ctx = projector(feats)             # (B, 64, 24, 24)

                # Correlation branch
                K    = text_emb.shape[0]
                corr = torch.einsum(
                    "b d h w, t d -> b t h w", feats, text_emb
                )                                          # (B, K, 24, 24)

                # Tile spatial context and concatenate with correlation maps
                spatial_tiled = (
                    spatial_ctx.unsqueeze(1)
                    .expand(-1, K, -1, -1, -1)
                    .reshape(B * K, 64, H, W)
                )                                          # (B*K, 64, 24, 24)
                corr_flat = corr.reshape(B * K, 1, H, W)  # (B*K, 1,  24, 24)
                dec_input = torch.cat(
                    [corr_flat, spatial_tiled], dim=1
                )                                          # (B*K, 65, 24, 24)

                # ClassAgnosticDecoder
                out_flat = decoder(dec_input)              # (B*K, 1,  96, 96)
                logits   = out_flat.reshape(B, K, 96, 96) # (B,   K,  96, 96)

                loss, ld, lg = mixed_loss(
                    logits, teacher_logits, gt_masks, tau, lam
                )

            if train:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(tips.parameters()) +
                        list(adapter.parameters()) +
                        list(projector.parameters()) +
                        list(decoder.parameters()),
                        1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(tips.parameters()) +
                        list(adapter.parameters()) +
                        list(projector.parameters()) +
                        list(decoder.parameters()),
                        1.0,
                    )
                    optimizer.step()

            total += loss.item()
            d_sum += ld
            g_sum += lg
            n += 1

    return total / max(n, 1), d_sum / max(n, 1), g_sum / max(n, 1)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      default="output/distill/student_best.pth")
    p.add_argument("--tips-dir",        default="checkpoints/tipsv2-l14")
    p.add_argument("--image-dir",       required=True)
    p.add_argument("--gt-dir",          required=True)
    p.add_argument("--cache-dir",       required=True)
    p.add_argument("--output-dir",      default="output/finetune")
    p.add_argument("--no-distill-init", action="store_true",
                   help="Random adapter init — scratch ablation baseline")
    p.add_argument("--epochs",          type=int,   default=25)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4,
                   help="LR for adapter, projector, decoder")
    p.add_argument("--backbone-lr",     type=float, default=1e-5,
                   help="LR for unfrozen TIPS blocks")
    p.add_argument("--unfreeze-blocks", type=int,   default=4)
    p.add_argument("--tau",             type=float, default=4.0)
    p.add_argument("--lam",             type=float, default=0.5,
                   help="Weight for distillation loss (1-lam for GT CE)")
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--val-fraction",    type=float, default=0.05)
    p.add_argument("--num-workers",     type=int,   default=6)
    p.add_argument("--resolution",      type=int,   default=336)
    p.add_argument("--num-classes",     type=int,   default=40)
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    tag     = "scratch" if args.no_distill_init else "distilled"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f"Phase 3 fine-tuning — init: {tag.upper()}")
    print(f"Output: {out_dir}\n")

    # ── Load Phase 2 checkpoint ───────────────────────────────────────────────
    print(f"Loading Phase 2 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd   = ckpt["student"]

    # ── Load TIPS ─────────────────────────────────────────────────────────────
    print(f"Loading TIPS from {args.tips_dir} ...")
    tips    = _load_tips(args.tips_dir)
    tips_sd = {k[len("tips."):]: v for k, v in sd.items() if k.startswith("tips.")}
    tips.load_state_dict(tips_sd, strict=True)
    tips.to(device)

    for param in tips.parameters():
        param.requires_grad = False
    total_blocks  = len(tips.vision_encoder.blocks)
    unfreeze_from = total_blocks - args.unfreeze_blocks
    for i in range(unfreeze_from, total_blocks):
        for param in tips.vision_encoder.blocks[i].parameters():
            param.requires_grad = True

    n_backbone = sum(p.numel() for p in tips.parameters() if p.requires_grad)
    print(f"  TIPS last {args.unfreeze_blocks} blocks unfrozen: {n_backbone/1e6:.1f}M params")

    # ── Adapter ───────────────────────────────────────────────────────────────
    adapter = SpatialAdapter(embed_dim=1024, bottleneck=256).to(device)

    if not args.no_distill_init:
        adapter.load_state_dict(
            {k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")},
            strict=True,
        )
        print("  Adapter: loaded from Phase 2 checkpoint (distilled init)")
    else:
        print("  Adapter: randomly initialised (scratch baseline)")

    # ── Projector + Decoder (always random init) ──────────────────────────────
    projector = SpatialProjector(in_dim=1024, out_dim=64).to(device)
    decoder   = ClassAgnosticDecoder(in_channels=65).to(device)
    print("  SpatialProjector + ClassAgnosticDecoder: randomly initialised")

    # ── Text embeddings ───────────────────────────────────────────────────────
    ld50k_classes = json.load(open("datasets/landdiscover.json"))
    with torch.no_grad():
        text_emb = tips.encode_text(ld50k_classes)
        text_emb = F.normalize(text_emb.float(), dim=-1).to(device)
    print(f"  Text embeddings: {text_emb.shape}")

    n_head = (sum(p.numel() for p in adapter.parameters()) +
              sum(p.numel() for p in projector.parameters()) +
              sum(p.numel() for p in decoder.parameters()))
    print(f"  Total trainable: {(n_backbone + n_head)/1e6:.2f}M params\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building dataset ...")
    full_ds = LD50KFinetuneDataset(
        args.image_dir, args.gt_dir, args.cache_dir,
        resolution=args.resolution, augment=True,
    )
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val

    # Validation split does not use augmentation
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds = torch.utils.data.Subset(full_ds, train_indices.indices)

    # Val dataset without augmentation
    val_ds_raw = LD50KFinetuneDataset(
        args.image_dir, args.gt_dir, args.cache_dir,
        resolution=args.resolution, augment=False,
    )
    val_ds = torch.utils.data.Subset(val_ds_raw, val_indices.indices)

    print(f"  Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in tips.parameters() if p.requires_grad],
             "lr": args.backbone_lr},
            {"params": (list(adapter.parameters()) +
                        list(projector.parameters()) +
                        list(decoder.parameters())),
             "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [args.backbone_lr, args.lr]
    scaler   = GradScaler(enabled=args.amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    log           = []
    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        cosine_lr(optimizer, epoch - 1, args.epochs, base_lrs)
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        t0 = time.time()
        tr_loss, tr_d, tr_g = run_epoch(
            tips, adapter, projector, decoder, text_emb,
            train_loader, optimizer, scaler, device, args.amp,
            args.tau, args.lam, train=True,
        )
        vl_loss, vl_d, vl_g = run_epoch(
            tips, adapter, projector, decoder, text_emb,
            val_loader, optimizer, scaler, device, args.amp,
            args.tau, args.lam, train=False,
        )
        elapsed = time.time() - t0
        is_best = vl_loss < best_val_loss

        ckpt_data = {
            "epoch":     epoch,
            "tag":       tag,
            "tips":      tips.state_dict(),
            "adapter":   adapter.state_dict(),
            "projector": projector.state_dict(),
            "decoder":   decoder.state_dict(),
            "val_loss":  vl_loss,
            "args":      vars(args),
        }
        if is_best:
            best_val_loss = vl_loss
            best_epoch    = epoch
            torch.save(ckpt_data, out_dir / "student_best.pth")
        torch.save(ckpt_data, out_dir / "student_latest.pth")

        entry = {
            "epoch":        epoch,
            "backbone_lr":  lrs[0],
            "head_lr":      lrs[1],
            "train_loss":   round(tr_loss, 6),
            "train_distill":round(tr_d,    6),
            "train_gt":     round(tr_g,    6),
            "val_loss":     round(vl_loss, 6),
            "val_distill":  round(vl_d,    6),
            "val_gt":       round(vl_g,    6),
            "is_best":      is_best,
            "epoch_time_s": round(elapsed, 1),
        }
        log.append(entry)
        print(
            f"Ep {epoch:2d}/{args.epochs}  "
            f"bb_lr={lrs[0]:.1e}  head_lr={lrs[1]:.1e}  "
            f"train={tr_loss:.4f}(d={tr_d:.3f} gt={tr_g:.3f})  "
            f"val={vl_loss:.4f}(d={vl_d:.3f} gt={vl_g:.3f})  "
            f"{'BEST' if is_best else '    '}  {elapsed:.0f}s"
        )

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "run_settings": vars(args),
                "tag":          tag,
                "best_epoch":   best_epoch,
                "best_val_loss":best_val_loss,
                "epochs":       log,
            },
            f, indent=2,
        )

    print(f"\nDone. Best epoch {best_epoch}  val_loss={best_val_loss:.4f}")
    print(f"Checkpoint → {out_dir}/student_best.pth")
    print(f"Log        → {log_path}")
    print(f"Eval cmd:  python scripts/eval_decoder.py "
          f"--checkpoint {out_dir}/student_best.pth")


if __name__ == "__main__":
    main()
