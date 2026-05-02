#!/usr/bin/env python3
"""
Evaluate Talk2DINO-ViTB on remote-sensing semantic segmentation benchmarks.

Usage:
    python eval_talk2dino.py                          # runs ALL datasets
    python eval_talk2dino.py potsdam_all FloodNet     # runs only these two
    python eval_talk2dino.py FAST_val --max-images 50 # quick test on 50 imgs
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.evaluation import SemSegEvaluator

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DETECTRON2_DATASETS = os.environ.get(
    "DETECTRON2_DATASETS", "../data/datasets"
)
TALK2DINO_DIR = os.environ.get("TALK2DINO_DIR", "Talk2DINO-ViTB")
TALK2DINO_SRC = os.environ.get("TALK2DINO_SRC", TALK2DINO_DIR)

DATASET_CONFIGS = {
    "potsdam_all": {
        "classes": ["impervious surface", "building", "low vegetation", "tree", "car", "clutter"],
        "test_classes": ["impervious surface", "building", "low vegetation", "tree", "car"],
        "image_dir": os.path.join(DETECTRON2_DATASETS, "PotsdamSplit", "img_dir", "val"),
        "gt_dir": os.path.join(DETECTRON2_DATASETS, "PotsdamSplit", "ann_dir", "val"),
        "gt_ext": "png", "image_ext": "png", "ignore_label": 5,
    },
    "FloodNet": {
        "classes": ["Background", "building-flooded", "building-non-flooded", "road-flooded",
                     "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"],
        "test_classes": ["Background", "building-flooded", "building-non-flooded", "road-flooded",
                          "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"],
        "image_dir": os.path.join(DETECTRON2_DATASETS, "FloodNet", "val+test", "img"),
        "gt_dir": os.path.join(DETECTRON2_DATASETS, "FloodNet", "val+test", "lbl"),
        "gt_ext": "png", "image_ext": "jpg", "ignore_label": 0,
    },
    "FLAIR_test": {
        "classes": ["building", "pervious surface", "impervious surface", "bare soil",
                     "water", "coniferous", "deciduous", "brushwood", "vineyard",
                     "herbaceous vegetation", "agricultural land", "plowed land", "other"],
        "test_classes": ["building", "pervious-surface", "impervious-surface", "bare soil",
                          "water", "coniferous", "deciduous", "brushwood", "vineyard",
                          "herbaceous vegetation", "agricultural land", "plowed land"],
        "image_dir": os.path.join(DETECTRON2_DATASETS, "FLAIR_test", "image"),
        "gt_dir": os.path.join(DETECTRON2_DATASETS, "FLAIR_test", "mask"),
        "gt_ext": "png", "image_ext": "png", "ignore_label": 12,
    },
    "FAST_val": {
        "classes": ["A220", "A321", "A330", "A350", "ARJ21", "Baseball-Field", "Basketball-Court",
                     "Boeing737", "Boeing747", "Boeing777", "Boeing787", "Bridge", "Bus", "C919",
                     "Cargo-Truck", "Dry-Cargo-Ship", "Dump-Truck", "Engineering-Ship", "Excavator",
                     "Fishing-Boat", "Football-Field", "Intersection", "Liquid-Cargo-Ship", "Motorboat",
                     "other-airplane", "other-ship", "other-vehicle", "Passenger-Ship", "Roundabout",
                     "Small-Car", "Tennis-Court", "Tractor", "Trailer", "Truck-Tractor", "Tugboat",
                     "Van", "Warship"],
        "test_classes": ["A220", "A321", "A330", "A350", "ARJ21", "Baseball-Field", "Basketball-Court",
                          "Boeing737", "Boeing747", "Boeing777", "Boeing787", "Bridge", "Bus", "C919",
                          "Cargo-Truck", "Dry-Cargo-Ship", "Dump-Truck", "Engineering-Ship", "Excavator",
                          "Fishing-Boat", "Football-Field", "Intersection", "Liquid-Cargo-Ship", "Motorboat",
                          "other-airplane", "other-ship", "other-vehicle", "Passenger-Ship", "Roundabout",
                          "Small-Car", "Tennis-Court", "Tractor", "Trailer", "Truck-Tractor", "Tugboat",
                          "Van", "Warship"],
        "image_dir": os.path.join(DETECTRON2_DATASETS, "FAST", "val", "images"),
        "gt_dir": os.path.join(DETECTRON2_DATASETS, "FAST", "val", "semlabels", "gray"),
        "gt_ext": "png", "image_ext": "png", "ignore_label": 255,
    },
}


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
def load_model(device):
    import shutil
    import tempfile

    # Copy Talk2DINO source files into an isolated temp package so that
    # third_party/clip.py cannot shadow the pip-installed 'clip' package.
    tmp_root = tempfile.mkdtemp(prefix="talk2dino_")
    pkg_dir = os.path.join(tmp_root, "talk2dino_vitb")
    os.makedirs(pkg_dir)
    for f in os.listdir(TALK2DINO_DIR):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(os.path.abspath(TALK2DINO_DIR), f), pkg_dir)
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()

    # Put ONLY the temp dir on sys.path (at the front), hide CWD + '' entries
    saved_path = sys.path[:]
    cwd = os.getcwd()
    sys.path = [tmp_root] + [p for p in sys.path
                              if p not in ("", ".", cwd, os.path.abspath(cwd))]
    import importlib
    importlib.invalidate_caches()

    from talk2dino_vitb.configuration_talk2dino import Talk2DINOConfig
    from talk2dino_vitb.modeling_talk2dino import Talk2DINO
    from safetensors.torch import load_file

    # Restore original sys.path
    sys.path = saved_path
    importlib.invalidate_caches()

    config = Talk2DINOConfig.from_pretrained(TALK2DINO_DIR)
    model = Talk2DINO(config)
    model = model.to_empty(device=device)
    state_dict = load_file(
        os.path.join(TALK2DINO_DIR, "model.safetensors"), device=device
    )
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    print(f"Model loaded on {device}")
    return model


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate_dataset(model, dataset_name, max_images, device):
    cfg = DATASET_CONFIGS[dataset_name]

    # Register dataset (idempotent)
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

    DatasetCatalog.register(
        dataset_name,
        lambda c=cfg: load_sem_seg(
            c["gt_dir"], c["image_dir"], gt_ext=c["gt_ext"], image_ext=c["image_ext"]
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        stuff_classes=cfg["classes"],
        image_root=cfg["image_dir"],
        seg_seg_root=cfg["gt_dir"],
        evaluator_type="sem_seg",
        ignore_label=cfg["ignore_label"],
    )

    dataset_dicts = DatasetCatalog.get(dataset_name)
    num_classes = len(cfg["test_classes"])

    # Text embeddings
    with torch.no_grad():
        text_embeds = torch.cat(
            [
                model.encode_text(c) / model.encode_text(c).norm(dim=-1, keepdim=True)
                for c in cfg["test_classes"]
            ],
            dim=0,
        )

    evaluator = SemSegEvaluator(
        dataset_name, distributed=False, output_dir=f"./eval_output/{dataset_name}"
    )
    evaluator.reset()

    n = min(len(dataset_dicts), max_images) if max_images else len(dataset_dicts)
    for i, entry in enumerate(dataset_dicts):
        if i >= n:
            break

        img_pil = Image.open(entry["file_name"]).convert("RGB")
        image = (transforms.ToTensor()(img_pil) * 255).to(torch.uint8).to(device)
        h, w = img_pil.size[1], img_pil.size[0]

        with torch.no_grad():
            image_embed = model.encode_image(image)
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        seg_logits = (image_embed @ text_embeds.T)[0]
        ps = int(round(np.sqrt(seg_logits.shape[0])))
        seg_logits = torch.nn.functional.interpolate(
            seg_logits.reshape(ps, ps, num_classes).permute(2, 0, 1).unsqueeze(0).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]

        evaluator.process(
            [{"file_name": entry["file_name"]}], [{"sem_seg": seg_logits.cpu()}]
        )
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i + 1}/{n}]")

    results = evaluator.evaluate()
    print(f"\n{'=' * 60}")
    print(f"  Talk2DINO on {dataset_name} ({n} images)")
    print(f"{'=' * 60}")
    for k, v in results["sem_seg"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()
    return results


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Talk2DINO on semantic segmentation benchmarks."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        help="Datasets to evaluate: {}. Default: all.".format(
            ", ".join(DATASET_CONFIGS.keys())
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap images per dataset (e.g. 50 for quick test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = args.datasets or list(DATASET_CONFIGS.keys())
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            sys.exit(f"Unknown dataset: {ds}. Choose from: {list(DATASET_CONFIGS.keys())}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    for ds in datasets:
        print(f"\n>>> Evaluating: {ds}")
        evaluate_dataset(model, ds, args.max_images, device)


if __name__ == "__main__":
    main()