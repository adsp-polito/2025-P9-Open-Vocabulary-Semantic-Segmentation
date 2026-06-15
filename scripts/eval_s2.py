#!/usr/bin/env python3
"""
Evaluate Student-2 on the same benchmark suite as eval_clip.py.

Student-2 evaluation loads CLIP + the Student-2 checkpoint and does not
instantiate RIPD, QGFF, DINOv3, or GSNet teacher modules.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("./detectron2"))
sys.path.insert(0, os.path.abspath("."))

import argparse
import json

try:
    import wandb
except ImportError:
    wandb = None

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.evaluation import SemSegEvaluator

from gs_net import add_cat_seg_config
import gs_net  # noqa: dataset registrations
from gs_net.third_party import clip as openai_clip

from gs_distill.student2 import GSDistillStudent2
from gs_distill.inference2 import gs_distill_s2_inference
from scripts.eval_clip import DATASET_CONFIGS, CLIP_MEAN, CLIP_RES, CLIP_STD


def _clip_skip_indices_from_cfg(cfg):
    clip_pretrained = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
    if clip_pretrained in {"ViT-B/16", "RemoteCLIP-ViT-B-32"}:
        return (3, 7)
    return (7, 15)


def build_clip_from_config(config_file, device):
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()

    clip_pretrained = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
    prompt_depth = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
    prompt_length = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

    if clip_pretrained in {"ViT-G", "ViT-H"}:
        import open_clip

        name, pretrain = (
            ("ViT-H-14", "laion2b_s32b_b79k")
            if clip_pretrained == "ViT-H"
            else ("ViT-bigG-14", "laion2b_s39b_b160k")
        )
        clip_model, _, _ = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device=device,
            force_image_size=336,
        )
    elif clip_pretrained == "RemoteCLIP-ViT-B-32":
        raise NotImplementedError(
            "RemoteCLIP direct loading is not wired in eval_s2.py; use an OpenAI CLIP config."
        )
    else:
        clip_model, _ = openai_clip.load(
            clip_pretrained,
            device=device,
            jit=False,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length,
        )

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model, _clip_skip_indices_from_cfg(cfg)


def load_model_s2(gsnet_config, gsnet_weights, s2_ckpt, device):
    del gsnet_weights
    print("Building CLIP backbone from config (no GSNet/RIPD/DINO modules) ...")
    clip_model, clip_skip_indices = build_clip_from_config(gsnet_config, device)

    print(f"Loading Student-2 checkpoint from {s2_ckpt} ...", flush=True)
    ckpt = torch.load(s2_ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})
    head_kwargs = {
        "hidden_dim": ckpt_args.get("head_hidden_dim", 64),
        "num_layers": ckpt_args.get("head_num_layers", 1),
        "window_size": ckpt_args.get("head_window_size", 4),
        "pad_len":     ckpt_args.get("head_pad_len", 0),
        "dec_dims":    tuple(ckpt_args.get("head_dec_dims", [32, 16])),
    }
    student2 = GSDistillStudent2(
        clip_model=clip_model,
        d_dino=ckpt_args.get("d_dino", 768),
        clip_layers=ckpt_args.get("clip_layers", [8, 16, 20, 23]),
        head_kwargs=head_kwargs,
        head_mode=ckpt_args.get("head_mode", "aggregator"),
    ).to(device)
    student2.load_state_dict(ckpt["student2"])
    student2.eval()

    # Match training: CLIP fp16, LayerNorm stays fp32 to avoid dtype mismatch
    clip_model.half()
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.float()

    print(
        f"  Student-2 loaded. (epoch {ckpt.get('epoch', '?')}, "
        f"val_loss {ckpt.get('val_loss', float('nan')):.4f})",
        flush=True,
    )

    return {
        "clip_model": clip_model,
        "student2": student2,
        "clip_skip_indices": clip_skip_indices,
    }


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
    return torch.stack(all_feats, dim=0).unsqueeze(0)


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
            c["gt_dir"],
            c["image_dir"],
            gt_ext=c["gt_ext"],
            image_ext=c["image_ext"],
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

    print(f"\n[{dataset_name}] evaluating {len(dataset_dicts)} images ...", flush=True)
    for entry in tqdm(dataset_dicts, desc=dataset_name, unit="img"):
        img_pil = Image.open(entry["file_name"]).convert("RGB")
        h, w = img_pil.size[1], img_pil.size[0]

        img_resized = img_pil.resize((CLIP_RES, CLIP_RES), Image.BILINEAR)
        img_t = to_tensor(img_resized)
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(enabled=amp):
                logit = gs_distill_s2_inference(
                    image=img_t,
                    text_feats=text_feats,
                    student2=model_parts["student2"],
                    clip_model=model_parts["clip_model"],
                    clip_skip_layer_indices=model_parts["clip_skip_indices"],
                )
            logit_up = F.interpolate(
                logit.float(), size=(h, w), mode="bilinear", align_corners=False
            )[0]

        evaluator.process(
            [{"file_name": entry["file_name"]}],
            [{"sem_seg": logit_up.cpu()}],
        )

    results = evaluator.evaluate()
    miou = results["sem_seg"].get("mIoU", float("nan"))
    print(f"\n{'='*60}")
    print(f"  Student-2 GS-Distill | {dataset_name}  mIoU={miou:.4f}")
    print(f"{'='*60}")
    for k, v in results["sem_seg"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsnet-config", required=True)
    p.add_argument("--gsnet-weights", required=True)
    p.add_argument("--s2-ckpt", required=True)
    p.add_argument("--output-dir", default="output/ashie/s2/eval")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-project", default="gs-distill")
    p.add_argument("--wandb-run", default="eval-s2")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model_parts = load_model_s2(
        args.gsnet_config,
        args.gsnet_weights,
        args.s2_ckpt,
        device,
    )

    all_results = {}
    for dataset_name in DATASET_CONFIGS:
        all_results[dataset_name] = evaluate_dataset(
            dataset_name, model_parts, args.output_dir, args.amp, device
        )

    print(f"\n{'='*60}")
    print("  SUMMARY - Student-2 GS-Distill")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        miou = res["sem_seg"].get("mIoU", float("nan"))
        print(f"  {ds:<20} mIoU = {miou:.4f}")
    print(f"{'='*60}")

    if not args.no_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        for ds, res in all_results.items():
            wandb.log({f"{ds}/{k}": v for k, v in res["sem_seg"].items()})
        wandb.finish()
    elif wandb is None and not args.no_wandb:
        print("W&B not installed; skipping logging.")


if __name__ == "__main__":
    main()
