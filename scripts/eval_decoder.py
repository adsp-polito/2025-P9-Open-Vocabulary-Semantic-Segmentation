"""
Evaluate the class-agnostic decoder on 4 test datasets.

Pipeline per image:
  TIPS vision encoder (frozen) → patch tokens (B, N, 1024)
  → reshape (B, 1024, 24, 24)
  → SpatialAdapter (frozen, from Phase 2 checkpoint)
  → L2-normalise
  → einsum with K dataset class embeddings → (B, K, 24, 24)
  → reshape (B*K, 1, 24, 24)
  → ClassAgnosticDecoder → (B*K, 1, 96, 96)
  → reshape (B, K, 96, 96)
  → upsample to GT size → argmax → mIoU

Loads:
  TIPS + SpatialAdapter : output/distill/student_best.pth  (Phase 2 format)
  ClassAgnosticDecoder  : output/decoder_retrain/decoder_best.pth

Results saved to: results/decoder_agnostic/
  {dataset}.json   — per-class IoU + mIoU
  summary.json     — one-line mIoU per dataset + comparison to baselines

Usage:
  python scripts/eval_decoder.py \\
      --student-ckpt  output/distill/student_best.pth \\
      --decoder-ckpt  output/decoder_retrain/decoder_best.pth \\
      --tips-dir      checkpoints/tipsv2-l14 \\
      --data-root     gs_net/data/datasets \\
      --results-dir   results/decoder_agnostic \\
      [--device cuda] [--batch-size 4]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from gs_distill.decoder_agnostic import ClassAgnosticDecoder


# ── Dataset configs ───────────────────────────────────────────────────────────

DATASETS = {
    "FloodNet": {
        "img_dir":   "FloodNet/val+test/img",
        "mask_dir":  "FloodNet/val+test/lbl",
        "img_ext":   ".jpg",
        "mask_ext":  ".png",
        "classes":   ["Background","building-flooded","building-non-flooded",
                      "road-flooded","road-non-flooded","water","tree",
                      "vehicle","pool","grass"],
        "ignore_label": 0,
    },
    "FAST": {
        "img_dir":   "FAST/val/images",
        "mask_dir":  "FAST/val/semlabels/gray",
        "img_ext":   ".png",
        "mask_ext":  ".png",
        "classes":   ["A220","A321","A330","A350","ARJ21","Baseball-Field",
                      "Basketball-Court","Boeing737","Boeing747","Boeing777",
                      "Boeing787","Bridge","Bus","C919","Cargo-Truck",
                      "Dry-Cargo-Ship","Dump-Truck","Engineering-Ship","Excavator",
                      "Fishing-Boat","Football-Field","Intersection",
                      "Liquid-Cargo-Ship","Motorboat","other-airplane","other-ship",
                      "other-vehicle","Passenger-Ship","Roundabout","Small-Car",
                      "Tennis-Court","Tractor","Trailer","Truck-Tractor","Tugboat",
                      "Van","Warship"],
        "ignore_label": 255,
    },
    "Potsdam": {
        "img_dir":   "PotsdamSplit/img_dir/val",
        "mask_dir":  "PotsdamSplit/ann_dir/val",
        "img_ext":   ".png",
        "mask_ext":  ".png",
        "classes":   ["impervious surface","building","low vegetation","tree","car"],
        "ignore_label": 5,
    },
    "FLAIR": {
        "img_dir":   "FLAIR_test/image",
        "mask_dir":  "FLAIR_test/mask",
        "img_ext":   ".png",
        "mask_ext":  ".png",
        "classes":   ["building","pervious-surface","impervious-surface","bare soil",
                      "water","coniferous","deciduous","brushwood","vineyard",
                      "herbaceous vegetation","agricultural land","plowed land"],
        "ignore_label": 12,
    },
}


# ── Model components ──────────────────────────────────────────────────────────

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


# ── IoU computation ───────────────────────────────────────────────────────────

def compute_miou(all_preds, all_gts, num_classes, ignore_label):
    intersection = np.zeros(num_classes, dtype=np.float64)
    union        = np.zeros(num_classes, dtype=np.float64)

    for pred, gt in zip(all_preds, all_gts):
        gt   = gt.astype(np.int64)
        pred = pred.astype(np.int64)
        valid  = gt != ignore_label
        pred_v = pred[valid]
        gt_v   = gt[valid]
        for c in range(num_classes):
            tp = int(np.sum((pred_v == c) & (gt_v == c)))
            fp = int(np.sum((pred_v == c) & (gt_v != c)))
            fn = int(np.sum((pred_v != c) & (gt_v == c)))
            intersection[c] += tp
            union[c]        += tp + fp + fn

    iou_per_class = np.where(union > 0, intersection / union, np.nan)
    miou = float(np.nanmean(iou_per_class))
    return miou, iou_per_class


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_pairs(data_root, cfg):
    img_dir  = Path(data_root) / cfg["img_dir"]
    mask_dir = Path(data_root) / cfg["mask_dir"]
    pairs = []
    for img_path in sorted(img_dir.glob(f"*{cfg['img_ext']}")):
        mask_path = mask_dir / (img_path.stem + cfg["mask_ext"])
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_image(path, resolution=336):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((resolution, resolution), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (3, H, W) in [0, 1]
    return tensor, (orig_h, orig_w)


def load_mask(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


# ── Main eval loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dataset(name, cfg, tips, adapter, decoder, data_root, device, batch_size, resolution):
    class_names  = cfg["classes"]
    num_classes  = len(class_names)
    ignore_label = cfg["ignore_label"]

    print(f"\n{'─'*60}")
    print(f"Dataset: {name}  ({num_classes} classes, ignore={ignore_label})")

    # Pre-compute text embeddings for this dataset's K classes
    text_out = tips.encode_text(class_names)                      # (K, 1024)
    text_emb = F.normalize(text_out.float(), dim=-1).to(device)  # (K, 1024)

    pairs = load_pairs(data_root, cfg)
    print(f"Images:  {len(pairs)}")

    all_preds, all_gts = [], []

    for i in tqdm(range(0, len(pairs), batch_size), desc=name, leave=True):
        batch_pairs = pairs[i : i + batch_size]
        images, orig_sizes, masks = [], [], []

        for img_path, mask_path in batch_pairs:
            img, (oh, ow) = load_image(img_path, resolution)
            images.append(img)
            orig_sizes.append((oh, ow))
            masks.append(load_mask(mask_path))

        imgs = torch.stack(images).to(device)  # (B, 3, 336, 336)
        B    = imgs.shape[0]
        K    = num_classes

        # TIPS vision encoder
        out     = tips.encode_image(imgs)
        patches = out.patch_tokens.float()                  # (B, N, 1024)
        _, N, D = patches.shape
        H = W   = int(N ** 0.5)                            # 24
        feats   = patches.permute(0, 2, 1).reshape(B, D, H, W)  # (B, 1024, 24, 24)

        # SpatialAdapter + L2-normalise
        feats = adapter(feats)
        feats = F.normalize(feats, dim=1)                  # (B, 1024, 24, 24)

        # Correlation: (B, K, 24, 24)
        corr = torch.einsum("b d h w, k d -> b k h w", feats, text_emb)

        # Class-agnostic decoder: (B, K, 24, 24) → (B*K, 1, 24, 24) → (B*K, 1, 96, 96)
        corr_flat = corr.reshape(B * K, 1, H, W)
        out_flat  = decoder(corr_flat)                     # (B*K, 1, 96, 96)
        logits    = out_flat.reshape(B, K, 96, 96)        # (B, K, 96, 96)

        for j, (oh, ow) in enumerate(orig_sizes):
            logit = F.interpolate(
                logits[j:j+1], size=(oh, ow), mode="bilinear", align_corners=False
            )[0]                                           # (K, oh, ow)
            pred = logit.argmax(dim=0).cpu().numpy().astype(np.uint8)
            all_preds.append(pred)
            all_gts.append(masks[j])

    miou, iou_per_class = compute_miou(all_preds, all_gts, num_classes, ignore_label)

    result = {
        "dataset":      name,
        "num_classes":  num_classes,
        "ignore_label": ignore_label,
        "miou":         round(miou * 100, 2),
        "per_class_iou": {
            cls: (round(float(iou) * 100, 2) if not np.isnan(iou) else None)
            for cls, iou in zip(class_names, iou_per_class)
        },
    }
    print(f"mIoU: {result['miou']:.2f}%")
    for cls, iou in result["per_class_iou"].items():
        val = f"{iou:.2f}%" if iou is not None else "—"
        print(f"  {cls:<35} {val}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-ckpt",  default="output/distill/student_best.pth")
    p.add_argument("--decoder-ckpt",  default="output/decoder_retrain/decoder_best.pth")
    p.add_argument("--tips-dir",      default="checkpoints/tipsv2-l14")
    p.add_argument("--data-root",     default="gs_net/data/datasets")
    p.add_argument("--results-dir",   default="results/decoder_agnostic")
    p.add_argument("--datasets",      nargs="+", default=list(DATASETS.keys()),
                   help="Subset of datasets to evaluate")
    p.add_argument("--batch-size",    type=int, default=4)
    p.add_argument("--resolution",    type=int, default=336)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load Phase 2 checkpoint (TIPS + SpatialAdapter) ──────────────────────
    print(f"Loading Phase 2 student checkpoint: {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch','?')}  val_loss: {ckpt.get('val_loss','?'):.4f}")

    # Phase 2 format: single "student" dict with "tips.*" and "adapter.*" keys
    if "student" in ckpt:
        sd = ckpt["student"]
        tips_sd  = {k[len("tips."):]: v for k, v in sd.items() if k.startswith("tips.")}
        adapt_sd = {k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")}
    else:
        tips_sd  = ckpt["tips"]
        adapt_sd = ckpt["adapter"]

    # ── Load TIPS ─────────────────────────────────────────────────────────────
    print(f"Loading TIPS from {args.tips_dir} ...")
    tips = _load_tips(args.tips_dir)
    tips.load_state_dict(tips_sd, strict=True)
    tips.eval().to(device)
    for param in tips.parameters():
        param.requires_grad = False

    # ── Load SpatialAdapter ───────────────────────────────────────────────────
    adapter = SpatialAdapter(embed_dim=1024, bottleneck=256)
    adapter.load_state_dict(adapt_sd, strict=True)
    adapter.eval().to(device)
    for param in adapter.parameters():
        param.requires_grad = False

    # ── Load ClassAgnosticDecoder ─────────────────────────────────────────────
    print(f"Loading decoder checkpoint: {args.decoder_ckpt}")
    dec_ckpt = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    dec_epoch = dec_ckpt.get("epoch", "?")
    dec_loss  = dec_ckpt.get("val_loss", float("nan"))
    print(f"  Decoder epoch: {dec_epoch}  val_loss: {dec_loss:.4f}")

    decoder = ClassAgnosticDecoder()
    decoder.load_state_dict(dec_ckpt["decoder"], strict=True)
    decoder.eval().to(device)
    for param in decoder.parameters():
        param.requires_grad = False

    print(f"All models loaded on {device}.\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    summary = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"Unknown dataset '{name}', skipping.")
            continue
        cfg = DATASETS[name]
        result = eval_dataset(
            name, cfg, tips, adapter, decoder,
            args.data_root, device, args.batch_size, args.resolution
        )
        summary[name] = result["miou"]

        out_path = Path(args.results_dir) / f"{name.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    teacher = {"FloodNet": 48.40, "FAST": 18.53, "Potsdam": 44.11, "FLAIR": 28.18}
    zeroshot = {"FloodNet": 4.27,  "FAST": 1.03,  "Potsdam": 13.27, "FLAIR": 4.65}

    summary_path = Path(args.results_dir) / "summary.json"
    summary_out  = {
        "mode":            "class-agnostic decoder (distilled Phase 2 + retrained decoder)",
        "student_ckpt":    args.student_ckpt,
        "decoder_ckpt":    args.decoder_ckpt,
        "decoder_epoch":   dec_epoch,
        "results":         summary,
        "baselines": {
            "teacher":  teacher,
            "zeroshot": zeroshot,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\n{'='*65}")
    print(f"DECODER EVAL RESULTS (class-agnostic, distilled Phase 2 student)")
    print(f"{'='*65}")
    print(f"{'Dataset':<12} {'With Decoder':>14} {'Zero-shot':>11} {'Teacher':>10}")
    print(f"{'─'*50}")
    for ds, miou in summary.items():
        zs = zeroshot.get(ds, "—")
        t  = teacher.get(ds, "—")
        print(f"{ds:<12} {miou:>13.2f}%  {zs:>10}%  {t:>9}%")
    avg = sum(summary.values()) / len(summary) if summary else 0
    avg_zs = sum(zeroshot[d] for d in summary if d in zeroshot) / len(summary)
    avg_t  = sum(teacher[d]  for d in summary if d in teacher)  / len(summary)
    print(f"{'─'*50}")
    print(f"{'Average':<12} {avg:>13.2f}%  {avg_zs:>10.2f}%  {avg_t:>9.2f}%")
    print(f"\nFull results → {args.results_dir}/")


if __name__ == "__main__":
    main()
