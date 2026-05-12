#!/usr/bin/env python3
"""
Evaluate the GS-Distill finetuned model on LD50K using Detectron2's
SemSegEvaluator — same pattern as eval_talk2dino.py.

Usage:
    export DETECTRON2_DATASETS='gs_net/data/datasets'
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/eval_finetune.py \\
        --gsnet-config  configs/vitl_336_dinov3.yaml \\
        --gsnet-weights output/gsnet_pretrain/model_final.pth \\
        --finetune-ckpt output/finetune/finetune_best.pth \\
        [--dataset LandDiscover50K_test] \\
        [--output-dir output/finetune/eval] \\
        [--amp]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.evaluation import SemSegEvaluator

import clip as openai_clip

from gs_distill.student import GSDistillStudent
from gs_distill.inference import gs_distill_inference


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs — mirrors DATASET_CONFIGS in eval_talk2dino.py
# ─────────────────────────────────────────────────────────────────────────────

DETECTRON2_DATASETS = os.environ.get("DETECTRON2_DATASETS", "gs_net/data/datasets")

LD50K_CLASSES = [
    "background", "bare land", "grass", "pavement", "road", "tree", "water",
    "agriculture land", "buildings", "forest land", "barren land", "urban land",
    "large-vehicle", "swimming-pool", "helicopter", "bridge",
    "plane", "ship", "soccer-ball-field", "basketball-court",
    "ground-track-field", "small-vehicle", "baseball-diamond",
    "tennis-court", "roundabout", "storage-tank", "harbor",
    "container-crane", "airport", "helipad", "chimney",
    "expressway service area", "expressway toll station", "dam",
    "golf field", "overpass", "stadium", "train station",
    "vehicle", "windmill",
]

DATASET_CONFIGS = {
    "LandDiscover50K_test": {
        "classes":      LD50K_CLASSES,
        "image_dir":    os.path.join(DETECTRON2_DATASETS, "LandDiscover_50K", "TE_Image"),
        "gt_dir":       os.path.join(DETECTRON2_DATASETS, "LandDiscover_50K", "TE_Label"),
        "gt_ext":       "png",
        "image_ext":    "png",
        "ignore_label": 255,
    },
    "LandDiscover50K_train": {
        "classes":      LD50K_CLASSES,
        "image_dir":    os.path.join(DETECTRON2_DATASETS, "LandDiscover_50K", "TR_Image"),
        "gt_dir":       os.path.join(DETECTRON2_DATASETS, "LandDiscover_50K", "TR_Label"),
        "gt_ext":       "png",
        "image_ext":    "png",
        "ignore_label": 255,
    },
}

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
CLIP_RES  = 384


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def build_gsnet(config_file, weights_file, device):
    from gs_net import add_cat_seg_config
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    from detectron2.modeling import build_model as d2_build
    model = d2_build(cfg)
    DetectionCheckpointer(model).load(weights_file)
    model.eval()
    return model


def load_model(gsnet_config, gsnet_weights, finetune_ckpt, device):
    print("Loading GSNet ...")
    gsnet = build_gsnet(gsnet_config, gsnet_weights, str(device))
    gsnet = gsnet.to(device)

    clip_model = gsnet.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_skip_indices = tuple(gsnet.layer_indexes)
    ripd             = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1   = gsnet.upsample1
    clip_upsample2   = gsnet.upsample2
    dino_decod_proj1 = gsnet.dino_decod_proj1
    dino_decod_proj2 = gsnet.dino_decod_proj2

    print(f"Loading finetuned student from {finetune_ckpt} ...")
    ckpt = torch.load(finetune_ckpt, map_location=device)
    student_args = ckpt.get("args", {})
    student = GSDistillStudent(
        clip_model=clip_model,
        hidden_dim=student_args.get("hidden_dim", 128),
        d_dino=student_args.get("d_dino", 768),
        num_classes=student_args.get("num_classes", 40),
        clip_layers=student_args.get("clip_layers", [4, 8, 10, 12]),
    ).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()

    if ckpt.get("ripd") is not None:
        ripd.load_state_dict(ckpt["ripd"])
        print("  RIPD weights loaded from finetune checkpoint.")
    ripd.eval()

    return dict(
        clip_model=clip_model,
        student=student,
        ripd=ripd,
        clip_upsample1=clip_upsample1,
        clip_upsample2=clip_upsample2,
        dino_decod_proj1=dino_decod_proj1,
        dino_decod_proj2=dino_decod_proj2,
        clip_skip_indices=clip_skip_indices,
    )


def build_text_features(class_names, clip_model, device):
    templates = ["a photo of a {}."]
    all_feats = []
    with torch.no_grad():
        for name in class_names:
            texts = [t.format(name) for t in templates]
            tokens = openai_clip.tokenize(texts).to(device)
            feats = clip_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
    return torch.stack(all_feats, dim=0).unsqueeze(0)   # (1, T, P, C)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — mirrors evaluate_dataset() in eval_talk2dino.py
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(model_parts, dataset_name, output_dir, amp, device):
    cfg = DATASET_CONFIGS[dataset_name]

    # Register dataset (idempotent)
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

    DatasetCatalog.register(
        dataset_name,
        lambda c=cfg: load_sem_seg(
            c["gt_dir"], c["image_dir"],
            gt_ext=c["gt_ext"], image_ext=c["image_ext"],
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        stuff_classes=cfg["classes"],
        image_root=cfg["image_dir"],
        sem_seg_root=cfg["gt_dir"],
        evaluator_type="sem_seg",
        ignore_label=cfg["ignore_label"],
    )

    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Text features (frozen, computed once)
    text_feats = build_text_features(
        cfg["classes"], model_parts["clip_model"], device
    )   # (1, T, P, C)

    eval_out = os.path.join(output_dir, dataset_name)
    evaluator = SemSegEvaluator(dataset_name, distributed=False, output_dir=eval_out)
    evaluator.reset()

    to_tensor = transforms.ToTensor()

    for i, entry in enumerate(dataset_dicts):
        img_pil = Image.open(entry["file_name"]).convert("RGB")
        h, w = img_pil.size[1], img_pil.size[0]

        # Preprocess for CLIP (resize + normalise)
        img_resized = img_pil.resize((CLIP_RES, CLIP_RES), Image.BILINEAR)
        img_t = to_tensor(img_resized)                              # (3, 384, 384)
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std  = torch.tensor(CLIP_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(device)                      # (1, 3, 384, 384)

        with torch.no_grad():
            with autocast(enabled=amp):
                logit = gs_distill_inference(
                    image=img_t,
                    text_feats=text_feats,
                    student=model_parts["student"],
                    clip_model=model_parts["clip_model"],
                    ripd=model_parts["ripd"],
                    clip_upsample1=model_parts["clip_upsample1"],
                    clip_upsample2=model_parts["clip_upsample2"],
                    dino_decod_proj1=model_parts["dino_decod_proj1"],
                    dino_decod_proj2=model_parts["dino_decod_proj2"],
                    clip_skip_layer_indices=model_parts["clip_skip_indices"],
                )   # (1, T, H, W)

            # Upsample to original image resolution
            logit_up = F.interpolate(
                logit.float(), size=(h, w), mode="bilinear", align_corners=False
            )[0]   # (T, H, W)

        evaluator.process(
            [{"file_name": entry["file_name"]}],
            [{"sem_seg": logit_up.cpu()}],
        )

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset_dicts):
            print(f"  [{i + 1}/{len(dataset_dicts)}]")

    results = evaluator.evaluate()
    print(f"\n{'=' * 60}")
    print(f"  GS-Distill finetune on {dataset_name}")
    print(f"{'=' * 60}")
    for k, v in results["sem_seg"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--finetune-ckpt",  required=True)
    p.add_argument("--dataset",        default="LandDiscover50K_test",
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--output-dir",     default="output/finetune/eval")
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model_parts = load_model(
        args.gsnet_config, args.gsnet_weights, args.finetune_ckpt, device
    )

    evaluate_dataset(model_parts, args.dataset, args.output_dir, args.amp, device)


if __name__ == "__main__":
    main()
