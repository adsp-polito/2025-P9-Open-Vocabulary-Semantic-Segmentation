#!/usr/bin/env python3
"""
qual_viz.py — qualitative comparison of all 4 models on 10 seeded images per dataset.

Output layout:
  output/qual_viz/
    {dataset}/{image_stem}/
      00_image.png          — original RGB image
      01_gt.png             — ground-truth label (colourised)
      02_old_gsnet.png      — Exp 1 prediction (colourised)
      03_improved_gsnet.png — Exp 2 prediction (colourised)
      04_gs_distill.png     — Exp 3 prediction (colourised)
      05_gs_distill_s2.png  — Exp 4 prediction (colourised)

Models are loaded one at a time (same pattern as efficiency_metrics.py) and
unloaded before the next is loaded to avoid OOM on a single GPU.

Usage:
    python scripts/qual_viz.py \
        --old-gsnet-config   configs/vitb_384.yaml \
        --old-gsnet-weights  gs_net_base/GSNet_base.pth \
        --old-rsib-ckpt      DinoV1/RSIB.pth \
        --gsnet-config       configs/vitl_336_dinov3.yaml \
        --gsnet-weights      output/gsnet_pretrain/model_final.pth \
        --clip-finetune-ckpt output/ashie/clip/finetune_best.pth \
        --s2-ckpt            output/ashie/s2/distill/s2_best.pth \
        --output-dir         output/qual_viz \
        [--seed 42] [--n-images 10] [--amp] [--device cuda]
"""

import os
import sys
import gc
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath("./detectron2"))
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from gs_net import add_cat_seg_config
import gs_net  # noqa: dataset registrations
from gs_net.third_party import clip as openai_clip

from gs_distill.student import GSDistillStudent
from gs_distill.student2 import GSDistillStudent2
from gs_distill.inference import gs_distill_inference
from gs_distill.inference2 import gs_distill_s2_inference

from scripts.eval_clip import (
    DATASET_CONFIGS, CLIP_MEAN, CLIP_RES, CLIP_STD,
    build_gsnet, remove_gsnet_clip_cache_hooks, release_unused_gsnet,
    load_model as load_model_s1,
    build_text_features as build_text_features_clip,
)
from scripts.eval_s2 import (
    load_model_s2,
    build_text_features as build_text_features_s2,
    build_clip_from_config,
)

# ── Colour palettes (one RGB per class) ───────────────────────────────────────
# Derived from standard remote-sensing / Detectron2 conventions.
# We generate a deterministic palette for any dataset by cycling through a
# fixed set of visually distinct colours.

_BASE_PALETTE = [
    (128,  64, 128), (244,  35, 232), ( 70,  70,  70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170,  30), (220, 220,   0),
    (107, 142,  35), (152, 251, 152), ( 70, 130, 180), (220,  20,  60),
    (255,   0,   0), (  0,   0, 142), (  0,   0,  70), (  0,  60, 100),
    (  0,  80, 100), (  0,   0, 230), (119,  11,  32), ( 80,  90,  80),
    (150,  30, 150), ( 35, 142, 107), (152, 251,  52), ( 30, 180, 220),
    (220,  20, 220), (  0, 142, 255), ( 70,   0, 142), (100,  60,   0),
    ( 32, 119,  11), ( 90,  80,  80), ( 30, 150, 150), (255, 142,   0),
    (  0, 220, 142), (142,   0, 255), (142, 255,   0), (  0,  70, 255),
    (255,  70,   0), (142,   0,  70),
]


def make_palette(n_classes: int):
    """Return a list of n_classes (R,G,B) tuples."""
    palette = []
    for i in range(n_classes):
        palette.append(_BASE_PALETTE[i % len(_BASE_PALETTE)])
    return palette


def label_to_color(label_hw: np.ndarray, palette, ignore_label: int = 255) -> np.ndarray:
    """
    Convert an (H, W) integer label map to an (H, W, 3) uint8 RGB image.
    Pixels with label == ignore_label are painted black.
    """
    h, w = label_hw.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        mask = label_hw == cls_idx
        rgb[mask] = color
    return rgb


# ── Dataset helpers ───────────────────────────────────────────────────────────

def register_dataset(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    with open(cfg["class_json"]) as f:
        class_names = json.load(f)
    DatasetCatalog.register(
        dataset_name,
        lambda c=cfg: load_sem_seg(
            c["gt_dir"], c["image_dir"],
            gt_ext=c["gt_ext"], image_ext=c["image_ext"],
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        stuff_classes=class_names,
        image_root=cfg["image_dir"],
        sem_seg_root=cfg["gt_dir"],
        evaluator_type="sem_seg",
        ignore_label=cfg["ignore_label"],
    )
    return class_names


def sample_entries(dataset_name: str, n: int, seed: int):
    entries = DatasetCatalog.get(dataset_name)
    rng = random.Random(seed)
    return rng.sample(entries, min(n, len(entries)))


def preprocess_image(img_pil: Image.Image, device: torch.device):
    """Resize to CLIP_RES, normalise, return (1,3,H,W) tensor."""
    img_resized = img_pil.resize((CLIP_RES, CLIP_RES), Image.BILINEAR)
    img_t = transforms.ToTensor()(img_resized)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD).view(3, 1, 1)
    img_t = (img_t - mean) / std
    return img_t.unsqueeze(0).to(device)


def load_gt(sem_seg_file: str, ignore_label: int):
    """Return an (H, W) numpy uint8 label map."""
    gt = np.array(Image.open(sem_seg_file))
    return gt


def save_coloured(arr_hw: np.ndarray, palette, ignore_label: int, path: str):
    rgb = label_to_color(arr_hw, palette, ignore_label)
    Image.fromarray(rgb).save(path)


def logit_to_label(logit: torch.Tensor) -> np.ndarray:
    """(1, T, H, W) or (T, H, W) logit → (H, W) argmax label numpy array."""
    if logit.dim() == 4:
        logit = logit[0]
    return logit.argmax(dim=0).cpu().numpy().astype(np.uint8)


# ── Model 1: old_gsnet ────────────────────────────────────────────────────────

def run_old_gsnet(args, entries_by_dataset, device, amp, output_dir):
    print("\n" + "="*60)
    print("  Model 1/4: old_gsnet")
    print("="*60)

    if args.old_gsnet_weights is None:
        print("  [SKIP] --old-gsnet-weights not provided.")
        return

    import os as _os
    _os.environ["RSIB_CKPT"] = args.old_rsib_ckpt

    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.old_gsnet_config)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()

    from detectron2.modeling import build_model as d2_build
    gsnet = d2_build(cfg)
    DetectionCheckpointer(gsnet).load(args.old_gsnet_weights)
    gsnet.eval()
    for p in gsnet.parameters():
        p.requires_grad = False
    gsnet = gsnet.to(device)

    for dataset_name, entries in entries_by_dataset.items():
        cfg_ds = DATASET_CONFIGS[dataset_name]
        class_names = json.load(open(cfg_ds["class_json"]))
        palette = make_palette(len(class_names))
        ignore_label = cfg_ds["ignore_label"]

        for entry in entries:
            stem = Path(entry["file_name"]).stem
            out_folder = Path(output_dir) / dataset_name / stem
            out_folder.mkdir(parents=True, exist_ok=True)

            img_pil = Image.open(entry["file_name"]).convert("RGB")
            h, w = img_pil.size[1], img_pil.size[0]

            with torch.no_grad():
                result = gsnet([{
                    "image": transforms.ToTensor()(img_pil).to(device) * 255,
                    "height": h,
                    "width": w,
                }])
            logit = result[0]["sem_seg"]  # (T, H, W) already postprocessed
            pred = logit.argmax(dim=0).cpu().numpy().astype(np.uint8)
            save_coloured(pred, palette, ignore_label,
                          str(out_folder / "02_old_gsnet.png"))
            print(f"    [{dataset_name}] {stem} done")

    del gsnet
    gc.collect()
    torch.cuda.empty_cache()


# ── Model 2: improved_gsnet ───────────────────────────────────────────────────

def run_improved_gsnet(args, entries_by_dataset, device, amp, output_dir):
    print("\n" + "="*60)
    print("  Model 2/4: improved_gsnet")
    print("="*60)

    if args.gsnet_weights is None:
        print("  [SKIP] --gsnet-weights not provided.")
        return

    if args.rsib_ckpt:
        os.environ["RSIB_CKPT"] = args.rsib_ckpt

    gsnet = build_gsnet(args.gsnet_config, args.gsnet_weights, str(device))
    gsnet.eval()
    for p in gsnet.parameters():
        p.requires_grad = False

    for dataset_name, entries in entries_by_dataset.items():
        cfg_ds = DATASET_CONFIGS[dataset_name]
        class_names = json.load(open(cfg_ds["class_json"]))
        palette = make_palette(len(class_names))
        ignore_label = cfg_ds["ignore_label"]

        for entry in entries:
            stem = Path(entry["file_name"]).stem
            out_folder = Path(output_dir) / dataset_name / stem
            out_folder.mkdir(parents=True, exist_ok=True)

            img_pil = Image.open(entry["file_name"]).convert("RGB")
            h, w = img_pil.size[1], img_pil.size[0]

            with torch.no_grad():
                result = gsnet([{
                    "image": transforms.ToTensor()(img_pil).to(device) * 255,
                    "height": h,
                    "width": w,
                }])
            logit = result[0]["sem_seg"]
            pred = logit.argmax(dim=0).cpu().numpy().astype(np.uint8)
            save_coloured(pred, palette, ignore_label,
                          str(out_folder / "03_improved_gsnet.png"))
            print(f"    [{dataset_name}] {stem} done")

    del gsnet
    gc.collect()
    torch.cuda.empty_cache()


# ── Model 3: gs_distill (Student-1) ──────────────────────────────────────────

def run_gs_distill(args, entries_by_dataset, device, amp, output_dir):
    print("\n" + "="*60)
    print("  Model 3/4: gs_distill (Student-1)")
    print("="*60)

    if args.clip_finetune_ckpt is None or not os.path.isfile(args.clip_finetune_ckpt):
        print(f"  [SKIP] --clip-finetune-ckpt not found ({args.clip_finetune_ckpt}).")
        return

    parts = load_model_s1(
        args.gsnet_config, args.gsnet_weights, args.clip_finetune_ckpt, device
    )
    for m in parts.values():
        if hasattr(m, "eval"):
            m.eval()
        if hasattr(m, "parameters"):
            for p in m.parameters():
                p.requires_grad = False

    for dataset_name, entries in entries_by_dataset.items():
        cfg_ds = DATASET_CONFIGS[dataset_name]
        class_names = json.load(open(cfg_ds["class_json"]))
        palette = make_palette(len(class_names))
        ignore_label = cfg_ds["ignore_label"]
        text_feats = build_text_features_clip(cfg_ds["class_json"], parts["clip_model"], device)

        for entry in entries:
            stem = Path(entry["file_name"]).stem
            out_folder = Path(output_dir) / dataset_name / stem
            out_folder.mkdir(parents=True, exist_ok=True)

            img_pil = Image.open(entry["file_name"]).convert("RGB")
            h, w = img_pil.size[1], img_pil.size[0]
            img_t = preprocess_image(img_pil, device)

            with torch.no_grad():
                from torch.cuda.amp import autocast
                with autocast(enabled=amp):
                    logit = gs_distill_inference(
                        image=img_t,
                        text_feats=text_feats,
                        student=parts["student"],
                        clip_model=parts["clip_model"],
                        ripd=parts["ripd"],
                        clip_upsample1=parts["clip_upsample1"],
                        clip_upsample2=parts["clip_upsample2"],
                        dino_decod_proj1=parts["dino_decod_proj1"],
                        dino_decod_proj2=parts["dino_decod_proj2"],
                        clip_skip_layer_indices=parts["clip_skip_indices"],
                    )
                logit_up = F.interpolate(
                    logit.float(), size=(h, w), mode="bilinear", align_corners=False
                )
            pred = logit_to_label(logit_up)
            save_coloured(pred, palette, ignore_label,
                          str(out_folder / "04_gs_distill.png"))
            print(f"    [{dataset_name}] {stem} done")

    for m in parts.values():
        del m
    del parts
    gc.collect()
    torch.cuda.empty_cache()


# ── Model 4: gs_distill_new_decoder (Student-2) ───────────────────────────────

def run_gs_distill_s2(args, entries_by_dataset, device, amp, output_dir):
    print("\n" + "="*60)
    print("  Model 4/4: gs_distill_new_decoder (Student-2)")
    print("="*60)

    s2_ckpt = args.s2_ckpt
    if s2_ckpt is None or not os.path.isfile(s2_ckpt):
        print(f"  [SKIP] --s2-ckpt not found ({s2_ckpt}).")
        return

    parts = load_model_s2(args.gsnet_config, args.gsnet_weights, s2_ckpt, device)
    # Put CLIP back to fp32 for inference (same fix as efficiency_metrics.py)
    parts["clip_model"].float()
    for m in parts.values():
        if hasattr(m, "eval"):
            m.eval()
        if hasattr(m, "parameters"):
            for p in m.parameters():
                p.requires_grad = False

    for dataset_name, entries in entries_by_dataset.items():
        cfg_ds = DATASET_CONFIGS[dataset_name]
        class_names = json.load(open(cfg_ds["class_json"]))
        palette = make_palette(len(class_names))
        ignore_label = cfg_ds["ignore_label"]
        text_feats = build_text_features_s2(cfg_ds["class_json"], parts["clip_model"], device)

        for entry in entries:
            stem = Path(entry["file_name"]).stem
            out_folder = Path(output_dir) / dataset_name / stem
            out_folder.mkdir(parents=True, exist_ok=True)

            img_pil = Image.open(entry["file_name"]).convert("RGB")
            h, w = img_pil.size[1], img_pil.size[0]
            img_t = preprocess_image(img_pil, device)

            with torch.no_grad():
                from torch.cuda.amp import autocast
                with autocast(enabled=amp):
                    logit = gs_distill_s2_inference(
                        image=img_t,
                        text_feats=text_feats,
                        student2=parts["student2"],
                        clip_model=parts["clip_model"],
                        clip_skip_layer_indices=parts["clip_skip_indices"],
                    )
                logit_up = F.interpolate(
                    logit.float(), size=(h, w), mode="bilinear", align_corners=False
                )
            pred = logit_to_label(logit_up)
            save_coloured(pred, palette, ignore_label,
                          str(out_folder / "05_gs_distill_s2.png"))
            print(f"    [{dataset_name}] {stem} done")

    for m in parts.values():
        del m
    del parts
    gc.collect()
    torch.cuda.empty_cache()


# ── Save original image + GT (done once, independent of models) ───────────────

def save_image_and_gt(entries_by_dataset, output_dir):
    print("\n[Phase 0] Saving original images and GT labels ...")
    for dataset_name, entries in entries_by_dataset.items():
        cfg_ds = DATASET_CONFIGS[dataset_name]
        class_names = json.load(open(cfg_ds["class_json"]))
        palette = make_palette(len(class_names))
        ignore_label = cfg_ds["ignore_label"]

        for entry in entries:
            stem = Path(entry["file_name"]).stem
            out_folder = Path(output_dir) / dataset_name / stem
            out_folder.mkdir(parents=True, exist_ok=True)

            # Original image
            img_pil = Image.open(entry["file_name"]).convert("RGB")
            img_pil.save(str(out_folder / "00_image.png"))

            # GT label
            gt_key = "sem_seg_file_name" if "sem_seg_file_name" in entry else "sem_seg_file"
            gt = load_gt(entry[gt_key], ignore_label)
            save_coloured(gt, palette, ignore_label, str(out_folder / "01_gt.png"))

            print(f"  [{dataset_name}] {stem}: image + GT saved")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Qualitative comparison: 4 models × 10 images × 4 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--old-gsnet-config",    default="configs/vitb_384.yaml")
    p.add_argument("--old-gsnet-weights",   default=None)
    p.add_argument("--old-rsib-ckpt",       default="DinoV1/RSIB.pth")
    p.add_argument("--gsnet-config",        default="configs/vitl_336_dinov3.yaml")
    p.add_argument("--gsnet-weights",       default=None)
    p.add_argument("--rsib-ckpt",           default=None)
    p.add_argument("--clip-finetune-ckpt",  default="output/ashie/clip/finetune_best.pth")
    p.add_argument("--s2-ckpt",             default=None)
    p.add_argument("--output-dir",          default="output/qual_viz")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--n-images",type=int,   default=10)
    p.add_argument("--amp",     action="store_true")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
    )
    return p.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    device = torch.device(args.device)

    print("="*60)
    print("  GS-Research Qualitative Visualisation")
    print(f"  Datasets:  {args.datasets}")
    print(f"  N images:  {args.n_images}  (seed={args.seed})")
    print(f"  Device:    {device}")
    print(f"  AMP:       {args.amp}")
    print(f"  Output:    {args.output_dir}")
    print("="*60)

    # ── Register datasets and sample entries ──────────────────────────────────
    entries_by_dataset = {}
    for ds in args.datasets:
        class_names = register_dataset(ds)
        sampled = sample_entries(ds, args.n_images, args.seed)
        entries_by_dataset[ds] = sampled
        print(f"  {ds}: {len(sampled)} images sampled from {len(DatasetCatalog.get(ds))} total")

    # ── Phase 0: save original images + GT ───────────────────────────────────
    save_image_and_gt(entries_by_dataset, args.output_dir)

    # ── Phases 1-4: run each model, save predictions ─────────────────────────
    run_old_gsnet(args, entries_by_dataset, device, args.amp, args.output_dir)
    run_improved_gsnet(args, entries_by_dataset, device, args.amp, args.output_dir)
    run_gs_distill(args, entries_by_dataset, device, args.amp, args.output_dir)
    run_gs_distill_s2(args, entries_by_dataset, device, args.amp, args.output_dir)

    print("\n" + "="*60)
    print(f"  Done. Output: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
