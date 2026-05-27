#!/usr/bin/env python3
"""
Student-2 Phase 1: Decoder warm-up via response-based distillation.

Trains only the S2CostHead + projection layers against pre-cached teacher
logits.  No live GSNet teacher or DINOv3 is needed.

  Input:   Student-1 distilled checkpoint  (branches already trained)
  Targets: cache_logits/<stem>.pt  {"logits": (40,96,96) fp16}
  Output:  output/ashie/s2/phase1/s2_phase1_best.pth

Student-1 trunk / substitute branches are frozen throughout.
CLIP is always frozen.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("./detectron2"))
sys.path.insert(0, os.path.abspath("."))

import argparse
import gc
import json
import math
import time
from pathlib import Path

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
from scripts.train_clip import build_text_features, build_teacher, cosine_lr


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CachedLogitDataset(Dataset):
    """Pairs LD50K images with pre-cached teacher logit .pt files."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir: str, cache_dir: str, resolution: int = 384):
        self.resolution = resolution
        cache_dir = Path(cache_dir)

        cache_stems = {p.stem for p in cache_dir.glob("*.pt")}
        self.samples = []
        for p in sorted(Path(image_dir).rglob("*")):
            if p.suffix.lower() in self.EXTENSIONS and p.stem in cache_stems:
                self.samples.append((str(p), str(cache_dir / (p.stem + ".pt"))))

        if not self.samples:
            raise RuntimeError(
                f"No image/cache pairs found.\n"
                f"  image_dir: {image_dir}\n"
                f"  cache_dir: {cache_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cache_path = self.samples[idx]

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
        return img, logits


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def response_distil_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
) -> torch.Tensor:
    """
    Temperature-scaled KL divergence.

    student_logits: (B, T, H, W)
    teacher_logits: (B, T, H, W)
    """
    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(
            teacher_logits, size=student_logits.shape[-2:],
            mode="bilinear", align_corners=False,
        )

    s = student_logits.float() / tau
    t = teacher_logits.float() / tau

    # KL over the class dimension (dim=1 is T)
    student_lp = F.log_softmax(s, dim=1)
    teacher_p  = F.softmax(t,  dim=1)
    kl = F.kl_div(student_lp, teacher_p, reduction="batchmean")
    return kl * (tau ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_s1_weights(student2: GSDistillStudent2, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # finetune checkpoint stores weights under "student"
    state = ckpt.get("student2", ckpt.get("student", ckpt))
    missing, unexpected = student2.load_state_dict(state, strict=False)
    cost_missing = [k for k in missing if "cost_head" in k or "cost_proj" in k]
    other_missing = [k for k in missing if k not in cost_missing]
    print(f"Loaded Student-1 weights from {ckpt_path}")
    print(f"  Cost-head/proj keys randomly initialised: {len(cost_missing)}")
    if other_missing:
        print(f"  Other missing keys: {len(other_missing)}")
    if unexpected:
        print(f"  Ignored unexpected keys: {len(unexpected)}")


def save_ckpt(path: str, epoch: int, student2: GSDistillStudent2, val_loss: float, args):
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
    loader,
    student2: GSDistillStudent2,
    text_feats: torch.Tensor,
    optimizer,
    scaler: GradScaler,
    device,
    amp: bool,
    clip_skip_indices,
    tau: float,
    train: bool,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
):
    student2.train(train)
    # CLIP and branches always frozen
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
        for step, (images, teacher_logits) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1:3d}/{total_epochs} [{desc}]", leave=False)
        ):
            images = images.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            B = images.shape[0]
            tf = text_feats.detach().expand(B, -1, -1, -1)

            _diag = train and step < 8   # phase-point trace for first 8 steps
            if _diag:
                torch.cuda.synchronize()
                _m0 = torch.cuda.memory_allocated() / 1024**2

            try:
                with autocast(enabled=amp):
                    student_logits = student2(
                        image=images,
                        text_feats=tf,
                        clip_skip_indices=clip_skip_indices,
                    )   # (B, T, H, W)
                    loss = response_distil_loss(student_logits, teacher_logits, tau=tau)
                    loss_for_backward = loss / grad_accum

                if _diag:
                    torch.cuda.synchronize()
                    _m1 = torch.cuda.memory_allocated() / 1024**2

                if train:
                    if amp:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

                    if _diag:
                        torch.cuda.synchronize()
                        _m2 = torch.cuda.memory_allocated() / 1024**2

                    is_last = step + 1 == len(loader)
                    if (step + 1) % grad_accum == 0 or is_last:
                        if amp:
                            scaler.unscale_(optimizer)
                        decoder_params = [
                            p for m in [
                                student2.clip_cost_proj,
                                student2.dino_cost_proj,
                                student2.text_cost_proj,
                                student2.cost_head,
                            ]
                            for p in m.parameters()
                        ]
                        nn.utils.clip_grad_norm_(decoder_params, 1.0)
                        if amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if _diag:
                        torch.cuda.synchronize()
                        _m3 = torch.cuda.memory_allocated() / 1024**2
                        print(
                            f"\n  [PHASE step={step}] "
                            f"pre={_m0:.1f}  +fwd={_m1-_m0:+.1f}  "
                            f"+bwd={_m2-_m1:+.1f}  +opt={_m3-_m2:+.1f}  "
                            f"final={_m3:.1f} MiB",
                            flush=True,
                        )

                    if (step + 1) % 200 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(
                    f"\n[OOM] step={step}  epoch={epoch+1}  "
                    f"reserved={torch.cuda.memory_reserved()/1024**2:.0f} MiB  "
                    f"allocated={torch.cuda.memory_allocated()/1024**2:.0f} MiB\n"
                    + torch.cuda.memory_summary(abbreviated=True)
                )
                raise

            total_loss += float(loss.detach().item())
            n_batches += 1

            # ── Memory diagnostics (first 30 steps + every 1000) ──────────────
            if train and (step < 30 or step % 1000 == 0):
                torch.cuda.synchronize()
                stats = torch.cuda.memory_stats()
                alloc_mib = torch.cuda.memory_allocated() / 1024**2
                n_alive  = stats.get("allocation.current", 0)
                print(
                    f"\n  [MEM step={step}] {alloc_mib:.1f} MiB  "
                    f"live_allocs={n_alive}",
                    flush=True,
                )

            # ── Memory snapshot at step 5 and 15 to find what accumulates ──
            if train and step in (5, 15):
                torch.cuda.memory._dump_snapshot(
                    f"mem_snapshot_step{step}.pickle"
                )
                print(
                    f"\n[SNAPSHOT] saved mem_snapshot_step{step}.pickle",
                    flush=True,
                )

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Student-2 Phase 1: decoder warm-up from cached teacher logits."
    )
    p.add_argument("--gsnet-config", required=True,
                   help="GSNet detectron2 config YAML (needed to load GSNet CLIP fork).")
    p.add_argument("--gsnet-weights", required=True,
                   help="GSNet checkpoint (CLIP extracted from it; rest is discarded).")
    p.add_argument("--s1-ckpt", default="output/ashie/clip/finetune_best.pth",
                   help="Student-1 distilled+finetuned checkpoint.")
    p.add_argument("--image-dir", required=True,
                   help="LD50K training images (TR_Image).")
    p.add_argument("--cache-dir", default="cache_logits",
                   help="Directory with pre-cached teacher logit .pt files.")
    p.add_argument("--class-json", default="datasets/landdiscover.json")
    p.add_argument("--output-dir", default="output/ashie/s2/phase1")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Decoder learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=4.0,
                   help="Temperature for response-based distillation.")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--clip-layers", type=int, nargs="+", default=[8, 16, 20, 23])
    p.add_argument("--clip-skip-indices", type=int, nargs=2, default=[7, 15],
                   help="CLIP layers used as decoder skip connections.")
    p.add_argument("--head-hidden-dim", type=int, default=128,
                   help="S2CostHead hidden dimension (default 128 = scaled up).")
    p.add_argument("--head-num-layers", type=int, default=2,
                   help="Number of aggregator layers (default 2).")
    p.add_argument("--head-window-size", type=int, default=4)
    p.add_argument("--head-pad-len", type=int, default=0)
    p.add_argument("--head-dec-dims", type=int, nargs=2, default=[32, 16],
                   metavar=("DIM1", "DIM2"),
                   help="Decoder channel dims for stage1 (24→48) and stage2 (48→96).")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Enable gradient checkpointing in aggregator layers (reduces VRAM at ~30%% compute cost).")
    p.add_argument("--d-dino", type=int, default=768)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds = CachedLogitDataset(args.image_dir, args.cache_dir)
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
    print(f"Dataset: {len(full_ds)} samples  train={n_train}  val={n_val}")

    # ── Extract CLIP from GSNet (needed for the custom dense=True extension) ──
    # The GSNet CLIP fork supports encode_image(dense=True) for spatial features.
    # We load the full GSNet once, pull out CLIP, then discard the rest.
    # No teacher inference happens during training — only cached logits are used.
    print(f"Loading GSNet to extract CLIP model from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device))
    clip_model = teacher.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    # Free everything except CLIP
    teacher.dino_model = None
    teacher.backbone = None
    teacher.sem_seg_head = None
    teacher.upsample1 = None
    teacher.upsample2 = None
    teacher.dino_decod_proj1 = None
    teacher.dino_decod_proj2 = None
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print("GSNet loaded; CLIP extracted, rest discarded.")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    head_kwargs = {
        "hidden_dim":              args.head_hidden_dim,
        "num_layers":              args.head_num_layers,
        "window_size":             args.head_window_size,
        "pad_len":                 args.head_pad_len,
        "gradient_checkpointing":  args.grad_ckpt,
        "dec_dims":                tuple(args.head_dec_dims),
    }
    student2 = GSDistillStudent2(
        clip_model=clip_model,
        d_dino=args.d_dino,
        clip_layers=args.clip_layers,
        head_kwargs=head_kwargs,
    ).to(device)

    load_s1_weights(student2, args.s1_ckpt, device)

    # Freeze CLIP and Student-1 branches; only decoder is trainable
    for p in student2.clip_model.parameters():
        p.requires_grad = False
    student2.set_student_backbone_trainable(False)

    # Convert frozen CLIP to fp16: saves ~725 MB and halves CLIP's per-step
    # intermediate allocations. Must happen after load_s1_weights (which loads
    # fp32 CLIP weights from the checkpoint) so the conversion sticks.
    # The GSNet CLIP fork's custom LayerNorm (model_vpt.py) casts input to
    # float32 before calling super().forward(), so LayerNorm weights must stay
    # fp32 to avoid a dtype mismatch — this is standard practice for ViTs.
    clip_model.half()
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.float()
    mem_after = torch.cuda.memory_allocated() / 1024**2
    print(f"CLIP converted to fp16 (LayerNorm stays fp32).  GPU allocated: {mem_after:.0f} MiB")
    decoder_params = [
        p for m in [
            student2.clip_cost_proj,
            student2.dino_cost_proj,
            student2.text_cost_proj,
            student2.cost_head,
        ]
        for p in m.parameters()
    ]
    for p in decoder_params:
        p.requires_grad = True

    n_decoder = sum(p.numel() for p in decoder_params)
    n_total   = sum(p.numel() for p in student2.parameters())
    print(f"Student-2: {n_total/1e6:.2f}M total  |  {n_decoder/1e6:.2f}M decoder trainable")
    print(f"Decoder config: hidden_dim={args.head_hidden_dim}  num_layers={args.head_num_layers}")

    # ── Text features (built once, 40-class LD50K vocabulary) ────────────────
    text_feats = build_text_features(
        args.class_json, clip_model, device
    )   # (1, 40, P, C)
    print(f"Text features: {text_feats.shape}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        decoder_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)
    clip_skip_indices = tuple(args.clip_skip_indices)
    warmup_epochs = max(1, args.epochs // 5)

    best_val = float("inf")
    best_path   = os.path.join(args.output_dir, "s2_phase1_best.pth")
    latest_path = os.path.join(args.output_dir, "s2_phase1_latest.pth")

    print(f"\nPhase 1: {args.epochs} epochs  lr={args.lr:.1e}  tau={args.tau}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, warmup_epochs, args.lr)
        t0 = time.time()

        train_loss = run_epoch(
            train_loader, student2, text_feats, optimizer, scaler,
            device, args.amp, clip_skip_indices, args.tau,
            train=True, epoch=epoch, total_epochs=args.epochs,
            grad_accum=args.grad_accum,
        )
        val_loss = run_epoch(
            val_loader, student2, text_feats, optimizer, scaler,
            device, args.amp, clip_skip_indices, args.tau,
            train=False, epoch=epoch, total_epochs=args.epochs,
            grad_accum=args.grad_accum,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"lr={lr:.2e}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )

        save_ckpt(latest_path, epoch, student2, val_loss, args)
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(best_path, epoch, student2, best_val, args)
            print(f"  New best val loss: {best_val:.4f}  -> {best_path}")

    print(f"\nPhase 1 done. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
