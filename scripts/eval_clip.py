#!/usr/bin/env python3
"""
Evaluate the GS-Distill finetuned model (CLIP backbone) on all 4 benchmark datasets:
  Potsdam, FloodNet, FLAIR, FAST

Zero-shot: text features built from each dataset's class list.
Student + RIPD weights loaded from finetune_best.pth.

Usage:
    export DETECTRON2_DATASETS='gs_net/data/datasets'
    export RSIB_CKPT='path/to/dinov3.pth'

    python scripts/eval_clip.py \\
        --gsnet-config  configs/vitl_336_dinov3.yaml \\
        --gsnet-weights output/gsnet_pretrain/model_final.pth \\
        --finetune-ckpt output/ashie/finetune/finetune_best.pth \\
        --output-dir    output/ashie/finetune/eval \\
        [--amp]
"""

import sys, os
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json

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

from gs_net import add_cat_seg_config
import gs_net  # noqa: triggers dataset registrations

from gs_distill.student import GSDistillStudent
from gs_distill.inference import gs_distill_inference


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs — matches register_*.py in gs_net/data/datasets/
# ─────────────────────────────────────────────────────────────────────────────

DETECTRON2_DATASETS = os.environ.get("DETECTRON2_DATASETS", "gs_net/data/datasets")

DATASET_CONFIGS = {
    "PotsdamSplit": {
        "class_json":   "datasets/potsdam.json",
        "ignore_label": 5,
        "image_ext":    "png",
        "gt_ext":       "png",
        "image_dir":    os.path.join(DETECTRON2_DATASETS, "PotsdamSplit", "img_dir/val"),
        "gt_dir":       os.path.join(DETECTRON2_DATASETS, "PotsdamSplit", "ann_dir/val"),
    },
    "FloodNet": {
        "class_json":   "datasets/floodnet.json",
        "ignore_label": 0,
        "image_ext":    "jpg",
        "gt_ext":       "png",
        "image_dir":    os.path.join(DETECTRON2_DATASETS, "FloodNet", "val+test", "img"),
        "gt_dir":       os.path.join(DETECTRON2_DATASETS, "FloodNet", "val+test", "lbl"),
    },
    "FLAIR_test": {
        "class_json":   "datasets/flair.json",
        "ignore_label": 12,
        "image_ext":    "png",
        "gt_ext":       "png",
        "image_dir":    os.path.join(DETECTRON2_DATASETS, "FLAIR_test", "image"),
        "gt_dir":       os.path.join(DETECTRON2_DATASETS, "FLAIR_test", "mask"),
    },
    "FAST": {
        "class_json":   "datasets/fast.json",
        "ignore_label": 255,
        "image_ext":    "png",
        "gt_ext":       "png",
        "image_dir":    os.path.join(DETECTRON2_DATASETS, "FAST", "val", "images"),
        "gt_dir":       os.path.join(DETECTRON2_DATASETS, "FAST", "val", "semlabels", "gray"),
    },
}

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
CLIP_RES  = 384


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def build_gsnet(config_file, weights_file, device):
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

    clip_skip_indices  = tuple(gsnet.layer_indexes)
    ripd               = gsnet.sem_seg_head.predictor.transformer
    clip_upsample1     = gsnet.upsample1
    clip_upsample2     = gsnet.upsample2
    dino_decod_proj1   = gsnet.dino_decod_proj1
    dino_decod_proj2   = gsnet.dino_decod_proj2

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
    if ckpt.get("clip_upsample1") is not None:
        clip_upsample1.load_state_dict(ckpt["clip_upsample1"])
    if ckpt.get("clip_upsample2") is not None:
        clip_upsample2.load_state_dict(ckpt["clip_upsample2"])
    if ckpt.get("dino_decod_proj1") is not None:
        dino_decod_proj1.load_state_dict(ckpt["dino_decod_proj1"])
    if ckpt.get("dino_decod_proj2") is not None:
        dino_decod_proj2.load_state_dict(ckpt["dino_decod_proj2"])
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


def build_text_features(class_json, clip_model, device):
    with open(class_json) as f:
        class_names = json.load(f)
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
# Per-dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(dataset_name, model_parts, output_dir, amp, device):
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

    text_feats = build_text_features(cfg["class_json"], model_parts["clip_model"], device)

    eval_out = os.path.join(output_dir, dataset_name)
    os.makedirs(eval_out, exist_ok=True)
    evaluator = SemSegEvaluator(dataset_name, distributed=False, output_dir=eval_out)
    evaluator.reset()

    dataset_dicts = DatasetCatalog.get(dataset_name)
    to_tensor = transforms.ToTensor()

    print(f"\n[{dataset_name}] evaluating {len(dataset_dicts)} images ...")
    for i, entry in enumerate(dataset_dicts):
        img_pil = Image.open(entry["file_name"]).convert("RGB")
        h, w = img_pil.size[1], img_pil.size[0]

        img_resized = img_pil.resize((CLIP_RES, CLIP_RES), Image.BILINEAR)
        img_t = to_tensor(img_resized)
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std  = torch.tensor(CLIP_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(device)

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
                )
            logit_up = F.interpolate(
                logit.float(), size=(h, w), mode="bilinear", align_corners=False
            )[0]

        evaluator.process(
            [{"file_name": entry["file_name"]}],
            [{"sem_seg": logit_up.cpu()}],
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(dataset_dicts):
            print(f"  [{i+1}/{len(dataset_dicts)}]")

    results = evaluator.evaluate()
    miou = results["sem_seg"].get("mIoU", float("nan"))
    print(f"\n{'='*60}")
    print(f"  CLIP GS-Distill | {dataset_name}  mIoU={miou:.4f}")
    print(f"{'='*60}")
    for k, v in results["sem_seg"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config",   required=True)
    p.add_argument("--gsnet-weights",  required=True)
    p.add_argument("--finetune-ckpt",  required=True)
    p.add_argument("--output-dir",     default="output/ashie/finetune/eval")
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

    all_results = {}
    for dataset_name in DATASET_CONFIGS:
        all_results[dataset_name] = evaluate_dataset(
            dataset_name, model_parts, args.output_dir, args.amp, device
        )

    print(f"\n{'='*60}")
    print("  SUMMARY — CLIP GS-Distill")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        miou = res["sem_seg"].get("mIoU", float("nan"))
        print(f"  {ds:<20} mIoU = {miou:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
