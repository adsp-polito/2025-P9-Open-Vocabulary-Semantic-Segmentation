"""
Retrain only the class-agnostic decoder on top of frozen TIPS + frozen SpatialAdapter.

Loads student_best.pth (Phase 2), freezes TIPS + adapter completely,
attaches a new ClassAgnosticDecoder (random init), trains only the decoder
on cached LD50K teacher logits using the same temperature-scaled BCE loss.

The reshape (B, K, 24, 24) ↔ (B*K, 1, ...) is handled here, not inside the decoder.

Output: output/decoder_retrain/
  decoder_best.pth       — best checkpoint by val loss
  decoder_latest.pth
  checkpoints/decoder_ep005.pth … decoder_ep020.pth
  training_log.json

Usage:
    python scripts/train_decoder.py \\
        --student-ckpt  output/distill/student_best.pth \\
        --tips-dir      checkpoints/tipsv2-l14 \\
        --image-dir     gs_net/data/datasets/LandDiscover_50K/TR_Image \\
        --cache-dir     cache_logits/ \\
        --output-dir    output/decoder_retrain \\
        [--epochs 20] [--batch-size 16] [--lr 3e-4] [--amp]
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

from gs_distill.decoder_agnostic import ClassAgnosticDecoder


# ── SpatialAdapter (inline — same as Phase 2, needed for loading) ─────────────

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

class CachedLogitsDataset(Dataset):
    """Pairs LD50K images with cached teacher logits. No GT masks needed."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir, cache_dir, resolution=336):
        self.resolution = resolution
        img_map   = {p.stem: p for p in Path(image_dir).iterdir()
                     if p.suffix.lower() in self.EXTENSIONS}
        cache_map = {p.stem: p for p in Path(cache_dir).glob("*.pt")}
        common = set(img_map) & set(cache_map)
        if not common:
            raise RuntimeError(f"No image/cache pairs. images={len(img_map)} caches={len(cache_map)}")
        self.samples = [(img_map[s], cache_map[s]) for s in sorted(common)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cache_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        image = torch.from_numpy(np.array(img)).float() / 255.0
        image = image.permute(2, 0, 1)  # (3, 336, 336) in [0,1]
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        teacher_logits = cache["logits"].float()  # (40, 96, 96)
        return image, teacher_logits


# ── Loss ──────────────────────────────────────────────────────────────────────

def distill_loss(student_logits, teacher_logits, tau=4.0):
    """Temperature-scaled BCE. student_logits and teacher_logits: (B, 40, 96, 96)."""
    t_soft = torch.sigmoid(teacher_logits / tau)
    s_soft = student_logits / tau
    return F.binary_cross_entropy_with_logits(s_soft, t_soft) * (tau ** 2)


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_epochs, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ── Epoch runner ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(tips, adapter, images, text_emb, device):
    """Frozen forward: image → (B, K, 24, 24) correlation."""
    out     = tips.encode_image(images)
    patches = out.patch_tokens.float()           # (B, N, 1024)
    B, N, D = patches.shape
    H = W   = int(N ** 0.5)
    feats   = patches.permute(0, 2, 1).reshape(B, D, H, W)  # (B, 1024, 24, 24)
    feats   = adapter(feats)
    feats   = F.normalize(feats, dim=1)
    corr    = torch.einsum("b d h w, t d -> b t h w", feats, text_emb)  # (B, K, 24, 24)
    return corr


def run_epoch(tips, adapter, decoder, text_emb, loader,
              optimizer, scaler, device, amp, tau, train=True):
    decoder.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    total, n = 0.0, 0

    with ctx:
        for images, teacher_logits in tqdm(
            loader, desc="train" if train else "val ", leave=False
        ):
            images         = images.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            B, K = teacher_logits.shape[:2]   # K=40

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                # Frozen TIPS + adapter → correlation
                corr = extract_features(tips, adapter, images, text_emb, device)
                # (B, K, 24, 24) → (B*K, 1, 24, 24) → decoder → (B*K, 1, 96, 96)
                corr_flat = corr.reshape(B * K, 1, corr.shape[2], corr.shape[3])
                out_flat  = decoder(corr_flat)                  # (B*K, 1, 96, 96)
                logits    = out_flat.reshape(B, K, 96, 96)      # (B, K, 96, 96)
                loss      = distill_loss(logits, teacher_logits, tau)

            if train:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()

            total += loss.item()
            n += 1

    return total / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student-ckpt", default="output/distill/student_best.pth")
    p.add_argument("--tips-dir",     default="checkpoints/tipsv2-l14")
    p.add_argument("--image-dir",    required=True)
    p.add_argument("--cache-dir",    required=True)
    p.add_argument("--output-dir",   default="output/decoder_retrain")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup-epochs",type=int,   default=2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--num-workers",  type=int,   default=6)
    p.add_argument("--resolution",   type=int,   default=336)
    p.add_argument("--tau",          type=float, default=4.0)
    p.add_argument("--save-interval",type=int,   default=5)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    device  = torch.device(args.device)

    print("Decoder retrain — frozen TIPS + frozen adapter + new ClassAgnosticDecoder")
    print(f"Output: {out_dir}\n")

    # ── Load Phase 2 checkpoint ───────────────────────────────────────────────
    print(f"Loading Phase 2 checkpoint: {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    sd   = ckpt["student"]
    print(f"  Phase 2 epoch: {ckpt.get('epoch','?')}  val_loss: {ckpt.get('val_loss','?'):.4f}")

    # ── Load TIPS (frozen) ────────────────────────────────────────────────────
    print(f"Loading TIPS from {args.tips_dir} ...")
    tips    = _load_tips(args.tips_dir)
    tips_sd = {k[len("tips."):]: v for k, v in sd.items() if k.startswith("tips.")}
    tips.load_state_dict(tips_sd, strict=True)
    tips.eval().to(device)
    for p in tips.parameters():
        p.requires_grad = False

    # ── Load SpatialAdapter (frozen) ──────────────────────────────────────────
    adapter    = SpatialAdapter(embed_dim=1024, bottleneck=256).to(device)
    adapter_sd = {k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")}
    adapter.load_state_dict(adapter_sd, strict=True)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False
    print("  TIPS + SpatialAdapter: loaded and frozen")

    # ── New class-agnostic decoder (random init) ──────────────────────────────
    decoder = ClassAgnosticDecoder().to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  ClassAgnosticDecoder: {n_params/1e3:.1f}K params (random init)\n")

    # ── 40 LD50K text embeddings (pre-computed once, frozen) ──────────────────
    ld50k_classes = json.load(open("datasets/landdiscover.json"))
    with torch.no_grad():
        text_emb = tips.encode_text(ld50k_classes)
        text_emb = F.normalize(text_emb.float(), dim=-1).to(device)  # (40, 1024)
    print(f"  LD50K text embeddings: {text_emb.shape}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nBuilding dataset ...")
    full_ds = CachedLogitsDataset(args.image_dir, args.cache_dir, args.resolution)
    n_val   = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Optimizer (decoder only) ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    log           = []
    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(optimizer, epoch - 1, args.epochs, args.lr,
                       args.warmup_epochs)

        t0 = time.time()
        tr_loss = run_epoch(tips, adapter, decoder, text_emb,
                            train_loader, optimizer, scaler, device,
                            args.amp, args.tau, train=True)
        vl_loss = run_epoch(tips, adapter, decoder, text_emb,
                            val_loader, optimizer, scaler, device,
                            args.amp, args.tau, train=False)
        elapsed = time.time() - t0
        is_best = vl_loss < best_val_loss

        # Save best
        if is_best:
            best_val_loss = vl_loss
            best_epoch    = epoch
            torch.save({"epoch": epoch, "decoder": decoder.state_dict(),
                        "val_loss": vl_loss, "args": vars(args)},
                       out_dir / "decoder_best.pth")

        # Save latest
        torch.save({"epoch": epoch, "decoder": decoder.state_dict(),
                    "val_loss": vl_loss, "args": vars(args)},
                   out_dir / "decoder_latest.pth")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({"epoch": epoch, "decoder": decoder.state_dict(),
                        "val_loss": vl_loss},
                       ckpt_dir / f"decoder_ep{epoch:03d}.pth")

        entry = {
            "epoch": epoch, "lr": round(lr, 8),
            "train_loss": round(tr_loss, 6),
            "val_loss":   round(vl_loss, 6),
            "is_best": is_best,
            "epoch_time_s": round(elapsed, 1),
        }
        log.append(entry)
        print(f"Ep {epoch:2d}/{args.epochs}  lr={lr:.2e}  "
              f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
              f"{'BEST' if is_best else '    '}  {elapsed:.0f}s")

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(out_dir / "training_log.json", "w") as f:
        json.dump({"run_settings": vars(args), "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss, "epochs": log}, f, indent=2)

    print(f"\nDone. Best epoch {best_epoch}  val_loss={best_val_loss:.4f}")
    print(f"Checkpoint → {out_dir}/decoder_best.pth")


if __name__ == "__main__":
    main()
