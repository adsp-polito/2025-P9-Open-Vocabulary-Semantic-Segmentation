#!/usr/bin/env python3
"""
Train Student-2 with a two-stage RIPD-free decoder schedule.

Student-2 starts from a Student-1 checkpoint, keeps CLIP frozen, and replaces
RIPD/QGFF with S2CostHead. Stage A freezes the Student-1 substitute layers and
distills the new decoder from the offline GSNet teacher. Stage B optionally
unfreezes the substitute layers and fine-tunes the whole deployable Student-2
stack except CLIP.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from einops import rearrange
from tqdm import tqdm

try:
    import wandb
except ImportError:
    class _WandbStub:
        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self):
            return None

    wandb = _WandbStub()
    sys.modules["wandb"] = wandb
    _WANDB_AVAILABLE = False
else:
    _WANDB_AVAILABLE = True

from scripts.train_clip import (
    SegDataset,
    build_teacher,
    build_text_features,
    cosine_lr,
    remove_gsnet_clip_cache_hooks,
)
from gs_distill.student2 import GSDistillStudent2
from gs_distill.inference2 import gs_distill_s2_inference
from gs_distill.utils import get_clip_skips


FORBIDDEN_KEYS = {
    "ripd",
    "dino_model",
    "backbone",
    "sem_seg_head",
    "upsample1",
    "upsample2",
    "dino_decod_proj1",
    "dino_decod_proj2",
}


def _assert_clean_ckpt(state_dict):
    bad = FORBIDDEN_KEYS & set(state_dict.keys())
    assert not bad, f"Checkpoint contains forbidden teacher-only keys: {bad}"
    student_state = state_dict.get("student2", {})
    forbidden_fragments = {
        "ripd",
        "dino_model",
        "backbone",
        "sem_seg_head",
        "upsample1",
        "upsample2",
        "dino_decod_proj1",
        "dino_decod_proj2",
        "transformer.layers",
        "predictor.transformer",
    }
    bad_nested = [
        key for key in student_state
        if any(fragment in key for fragment in forbidden_fragments)
    ]
    assert not bad_nested, (
        "Checkpoint contains forbidden teacher-only state_dict keys: "
        f"{bad_nested[:10]}"
    )


def _load_class_names(class_json):
    with open(class_json) as f:
        return json.load(f)


def _load_s1_weights(student2, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Student-1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("student2", ckpt.get("student", ckpt))
    missing, unexpected = student2.load_state_dict(state, strict=False)
    missing_cost = [k for k in missing if k.startswith("cost_head")]
    print(f"Loaded Student-1 weights from {ckpt_path}")
    if missing_cost:
        print(f"  Cost-head keys randomly initialised: {len(missing_cost)}")
    other_missing = [k for k in missing if not k.startswith("cost_head")]
    if other_missing:
        print(f"  Other missing keys: {len(other_missing)}")
    if unexpected:
            print(f"  Ignored unexpected keys: {len(unexpected)}")


def _resolve_s1_ckpt(args):
    if args.s1_ckpt is not None:
        return args.s1_ckpt
    if args.s1_source == "distill":
        return args.s1_distill_ckpt
    if args.s1_source == "baseline":
        return args.s1_baseline_ckpt
    raise ValueError("--s1-source custom requires --s1-ckpt PATH")


def _register_teacher_feature_hook(teacher):
    capture = {"feat": None}
    ripd = teacher.sem_seg_head.predictor.transformer
    if hasattr(ripd, "layers") and len(ripd.layers) > 0:
        module = ripd.layers[-1]
    else:
        module = ripd

    def _hook(_module, _inputs, output):
        capture["feat"] = output

    handle = module.register_forward_hook(_hook)
    return capture, handle


def _teacher_logits(
    teacher,
    images,
    clip_features,
    clip_skips,
    clip_skip_indices,
):
    if teacher.dino_model is None:
        raise RuntimeError("Student-2 training requires the offline GSNet teacher DINO branch.")

    dino_images = F.interpolate(
        images, size=teacher.dino_resolution, mode="bilinear", align_corners=False
    )
    dino_feat = teacher.dino_model.get_intermediate_layers(dino_images, n=12)
    dino_last = rearrange(dino_feat[-1][:, 1:, :], "B (H W) C -> B C H W", H=48)
    dino_down = teacher.dino_down_sample(dino_last)

    dino_l4 = rearrange(dino_feat[3][:, 1:, :], "B (H W) C -> B C H W", H=48)
    dino_l8 = rearrange(dino_feat[7][:, 1:, :], "B (H W) C -> B C H W", H=48)
    dino_l4 = teacher.dino_decod_proj1(dino_l4) if teacher.dino_decod_proj1 is not None else None
    dino_l8 = teacher.dino_decod_proj2(dino_l8) if teacher.dino_decod_proj2 is not None else None

    n_tokens = clip_features.shape[1] - 1
    H = W = int(n_tokens**0.5)
    clip_res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=H, W=W)
    l0, l1 = clip_skip_indices
    res4 = teacher.upsample1(clip_skips[l0]) if teacher.upsample1 is not None else None
    res5 = teacher.upsample2(clip_skips[l1]) if teacher.upsample2 is not None else None
    clip_guidance = {"res5": res5, "res4": res4, "res3": clip_res3}

    return teacher.sem_seg_head(
        clip_features,
        dino_down,
        clip_guidance,
        [dino_l4, dino_l8],
    )


def _soft_logit_loss(student_logits, teacher_logits):
    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(
            teacher_logits,
            size=student_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    teacher_probs = F.softmax(teacher_logits.float(), dim=1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1).mean()


def _align_feature_channels(student_feat, teacher_feat):
    cs = student_feat.shape[1]
    ct = teacher_feat.shape[1]
    if cs == ct:
        return student_feat, teacher_feat
    if ct % cs == 0:
        ratio = ct // cs
        teacher_feat = teacher_feat.view(
            teacher_feat.shape[0], cs, ratio, *teacher_feat.shape[2:]
        ).mean(dim=2)
        return student_feat, teacher_feat
    if cs % ct == 0:
        ratio = cs // ct
        student_feat = student_feat.view(
            student_feat.shape[0], ct, ratio, *student_feat.shape[2:]
        ).mean(dim=2)
        return student_feat, teacher_feat
    c = min(cs, ct)
    return student_feat[:, :c], teacher_feat[:, :c]


def _feature_loss(student_feat, teacher_feat):
    if teacher_feat is None:
        raise RuntimeError("Teacher feature hook did not capture RIPD aggregation output.")
    teacher_feat = teacher_feat.detach().to(device=student_feat.device, dtype=student_feat.dtype)

    if teacher_feat.shape[2] != student_feat.shape[2]:
        t = min(teacher_feat.shape[2], student_feat.shape[2])
        teacher_feat = teacher_feat[:, :, :t]
        student_feat = student_feat[:, :, :t]

    if teacher_feat.shape[-2:] != student_feat.shape[-2:]:
        B, C, T, _, _ = teacher_feat.shape
        teacher_feat = rearrange(teacher_feat, "B C T H W -> (B T) C H W")
        teacher_feat = F.interpolate(
            teacher_feat,
            size=student_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        teacher_feat = rearrange(teacher_feat, "(B T) C H W -> B C T H W", B=B, T=T)

    student_feat, teacher_feat = _align_feature_channels(student_feat, teacher_feat)
    return F.smooth_l1_loss(student_feat.float(), teacher_feat.float())


def _save(path, epoch, student2, avg_val, args):
    ckpt = {
        "epoch": epoch,
        "student2": student2.state_dict(),
        "val_loss": float(avg_val),
        "args": vars(args),
    }
    _assert_clean_ckpt(ckpt)
    torch.save(ckpt, path)


def _set_decoder_trainable(student2, trainable=True):
    for module in [
        student2.clip_cost_proj,
        student2.dino_cost_proj,
        student2.text_cost_proj,
        student2.cost_head,
    ]:
        for p in module.parameters():
            p.requires_grad = trainable


def _student_backbone_modules(student2):
    return [
        student2.shared_trunk,
        student2.dino_down_branch,
        student2.dino_l4_branch,
        student2.dino_l8_branch,
    ]


def _prepare_stage_trainability(student2, freeze_student_backbone):
    student2.set_student_backbone_trainable(not freeze_student_backbone)
    _set_decoder_trainable(student2, True)
    for p in student2.clip_model.parameters():
        p.requires_grad = False
    active_params = [p for p in student2.trainable_parameters() if p.requires_grad]
    if not active_params:
        raise RuntimeError("No trainable Student-2 parameters for this stage.")
    return active_params


def _set_student2_mode(student2, train, freeze_student_backbone):
    student2.train(train)
    student2.clip_model.eval()
    if freeze_student_backbone:
        for module in _student_backbone_modules(student2):
            module.eval()


def _load_student2_state(student2, path, device):
    ckpt = torch.load(path, map_location=device)
    student2.load_state_dict(ckpt["student2"], strict=True)
    return ckpt


def run_finetune_s2(args, teacher, student2, device, output_dir, use_wandb):
    clip_model = teacher.sem_seg_head.predictor.clip_model
    clip_skip_indices = tuple(teacher.layer_indexes)
    remove_gsnet_clip_cache_hooks(teacher)

    class_names = _load_class_names(args.class_json)
    predictor = teacher.sem_seg_head.predictor
    predictor.test_class_texts = class_names
    predictor.cache = None

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    student2 = student2.to(device)
    for p in student2.clip_model.parameters():
        p.requires_grad = False

    text_feats = build_text_features(args.class_json, clip_model, device)

    full_ds = SegDataset(args.image_dir, args.label_dir)
    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"  [Student-2] Train: {n_train}  Val: {n_val}")

    scaler = GradScaler(enabled=args.amp)
    ignore_idx = 255
    all_clip_layers = sorted(set(list(clip_skip_indices) + list(student2.clip_layers)))
    total_epochs = args.distill_epochs + args.finetune_epochs
    if total_epochs <= 0:
        raise ValueError("At least one of --distill-epochs or --finetune-epochs must be > 0.")

    capture, hook_handle = _register_teacher_feature_hook(teacher)

    def _run_epoch(loader, stage, stage_epoch, train, optimizer, active_params):
        _set_student2_mode(student2, train, stage["freeze_student_backbone"])
        total = {"loss": 0.0, "seg": 0.0, "logit": 0.0, "feat": 0.0}
        n_batches = 0
        grad_accum = args.grad_accum
        if train:
            optimizer.zero_grad(set_to_none=True)

        desc = "train" if train else "val"
        grad_ctx = torch.enable_grad() if train else torch.no_grad()
        with grad_ctx:
            for step, (images, labels) in enumerate(
                tqdm(
                    loader,
                    desc=(
                        f"[{stage['tag']}] Epoch {stage_epoch+1}/"
                        f"{stage['epochs']} [{desc}]"
                    ),
                )
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]
                tf = text_feats.detach().expand(B, -1, -1, -1)

                with torch.no_grad(), autocast(enabled=args.amp):
                    clip_features, clip_skips = get_clip_skips(
                        clip_model, images, all_clip_layers
                    )
                    capture["feat"] = None
                    teacher_logit = _teacher_logits(
                        teacher, images, clip_features, clip_skips, clip_skip_indices
                    )
                    teacher_feat = capture["feat"]
                    stacked = torch.cat(
                        [clip_skips[layer] for layer in student2.clip_layers], dim=1
                    )

                with autocast(enabled=args.amp):
                    out = student2(
                        image=images,
                        text_feats=tf,
                        clip_skip_indices=clip_skip_indices,
                        clip_features=clip_features,
                        clip_skips=clip_skips,
                        student_clip_stacked=stacked,
                        return_intermediate=True,
                    )
                    logit = out["logit"]
                    logit_up = F.interpolate(
                        logit,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss_seg = F.cross_entropy(logit_up, labels, ignore_index=ignore_idx)
                    loss_logit = _soft_logit_loss(logit, teacher_logit)
                    if stage["lambda_feat"] > 0:
                        loss_feat = _feature_loss(out["corr_embed"], teacher_feat)
                    else:
                        loss_feat = logit.new_zeros(())
                    loss = (
                        stage["lambda_seg"] * loss_seg
                        + stage["lambda_logit"] * loss_logit
                        + stage["lambda_feat"] * loss_feat
                    )
                    loss_for_backward = loss / grad_accum

                if train:
                    if args.amp:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

                    is_last_step = step + 1 == len(loader)
                    if (step + 1) % grad_accum == 0 or is_last_step:
                        if args.amp:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(active_params, 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(active_params, 1.0)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                total["loss"] += float(loss.detach().item())
                total["seg"] += float(loss_seg.detach().item())
                total["logit"] += float(loss_logit.detach().item())
                total["feat"] += float(loss_feat.detach().item())
                n_batches += 1

                del (
                    images,
                    labels,
                    tf,
                    clip_features,
                    clip_skips,
                    stacked,
                    teacher_logit,
                    teacher_feat,
                    out,
                    logit,
                    logit_up,
                    loss,
                    loss_for_backward,
                    loss_seg,
                    loss_logit,
                    loss_feat,
                )

        return {k: v / max(n_batches, 1) for k, v in total.items()}

    def _run_stage(stage, global_epoch):
        active_params = _prepare_stage_trainability(
            student2,
            freeze_student_backbone=stage["freeze_student_backbone"],
        )
        optimizer = torch.optim.AdamW(
            active_params,
            lr=stage["lr"],
            weight_decay=args.weight_decay,
        )
        warmup_epochs = max(1, stage["epochs"] // 10)
        stage_best = float("inf")
        best_path = os.path.join(output_dir, f"{stage['prefix']}_best.pth")
        latest_path = os.path.join(output_dir, f"{stage['prefix']}_latest.pth")
        n_active = sum(p.numel() for p in active_params)

        freeze_text = "frozen" if stage["freeze_student_backbone"] else "trainable"
        print(
            f"\n[{stage['tag']}] {stage['name']}\n"
            f"  Student-1 substitute layers: {freeze_text}\n"
            f"  Active params: {n_active / 1e6:.2f}M\n"
            f"  lr={stage['lr']:.2e} "
            f"loss={stage['lambda_seg']}*CE + "
            f"{stage['lambda_logit']}*KL + "
            f"{stage['lambda_feat']}*feature"
        )

        for stage_epoch in range(stage["epochs"]):
            lr = cosine_lr(
                optimizer,
                stage_epoch,
                stage["epochs"],
                warmup_epochs,
                stage["lr"],
            )
            t0 = time.time()
            train_stats = _run_epoch(
                train_loader,
                stage,
                stage_epoch,
                train=True,
                optimizer=optimizer,
                active_params=active_params,
            )
            val_stats = _run_epoch(
                val_loader,
                stage,
                stage_epoch,
                train=False,
                optimizer=optimizer,
                active_params=active_params,
            )
            elapsed = time.time() - t0

            print(
                f"[{stage['tag']}] Epoch {stage_epoch+1:3d}/{stage['epochs']} "
                f"lr={lr:.2e} train={train_stats['loss']:.4f} "
                f"val={val_stats['loss']:.4f} "
                f"val_seg={val_stats['seg']:.4f} "
                f"val_logit={val_stats['logit']:.4f} "
                f"val_feat={val_stats['feat']:.4f} "
                f"({elapsed:.0f}s)"
            )

            if use_wandb:
                wandb.log(
                    {
                        "s2/global_epoch": global_epoch + stage_epoch + 1,
                        f"s2/{stage['key']}/epoch": stage_epoch + 1,
                        f"s2/{stage['key']}/lr": lr,
                        f"s2/{stage['key']}/train/loss": train_stats["loss"],
                        f"s2/{stage['key']}/train/seg": train_stats["seg"],
                        f"s2/{stage['key']}/train/logit": train_stats["logit"],
                        f"s2/{stage['key']}/train/feat": train_stats["feat"],
                        f"s2/{stage['key']}/val/loss": val_stats["loss"],
                        f"s2/{stage['key']}/val/seg": val_stats["seg"],
                        f"s2/{stage['key']}/val/logit": val_stats["logit"],
                        f"s2/{stage['key']}/val/feat": val_stats["feat"],
                    },
                    step=global_epoch + stage_epoch + 1,
                )

            absolute_epoch = global_epoch + stage_epoch
            _save(latest_path, absolute_epoch, student2, val_stats["loss"], args)
            if val_stats["loss"] < stage_best:
                stage_best = val_stats["loss"]
                _save(best_path, absolute_epoch, student2, stage_best, args)
                print(f"  [{stage['tag']}] New best val loss: {stage_best:.4f} -> {best_path}")

        return best_path, stage_best, global_epoch + stage["epochs"]

    stage_a = {
        "key": "stage_a",
        "tag": "S2 Stage A",
        "name": "decoder distillation warm-up",
        "prefix": "s2_stage_a",
        "epochs": args.distill_epochs,
        "lr": args.distill_lr,
        "freeze_student_backbone": True,
        "lambda_seg": args.distill_lambda_seg,
        "lambda_logit": args.distill_lambda_logit,
        "lambda_feat": args.distill_lambda_feat,
    }
    stage_b = {
        "key": "stage_b",
        "tag": "S2 Stage B",
        "name": "segmentation fine-tune",
        "prefix": "s2",
        "epochs": args.finetune_epochs,
        "lr": args.lr,
        "freeze_student_backbone": args.freeze_student_backbone,
        "lambda_seg": 1.0,
        "lambda_logit": args.lambda_logit,
        "lambda_feat": args.lambda_feat,
    }

    final_best_path = None
    final_best_val = float("inf")
    global_epoch = 0

    try:
        if stage_a["epochs"] > 0:
            stage_a_best_path, stage_a_best_val, global_epoch = _run_stage(stage_a, global_epoch)
            _load_student2_state(student2, stage_a_best_path, device)
            print(f"  [S2 Stage A] Restored best decoder warm-up checkpoint: {stage_a_best_path}")
            final_best_path = stage_a_best_path
            final_best_val = stage_a_best_val

        if stage_b["epochs"] > 0:
            stage_b_best_path, stage_b_best_val, global_epoch = _run_stage(stage_b, global_epoch)
            final_best_path = stage_b_best_path
            final_best_val = stage_b_best_val

        if final_best_path is None:
            raise RuntimeError("No Student-2 training stage was run.")

        if final_best_path != os.path.join(output_dir, "s2_best.pth"):
            best_ckpt = _load_student2_state(student2, final_best_path, device)
            _save(
                os.path.join(output_dir, "s2_best.pth"),
                best_ckpt["epoch"],
                student2,
                best_ckpt["val_loss"],
                args,
            )
            final_best_path = os.path.join(output_dir, "s2_best.pth")
            final_best_val = best_ckpt["val_loss"]
    finally:
        hook_handle.remove()

    print(f"\n[Student-2] Done. Best val loss: {final_best_val:.4f}")
    return final_best_path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Student-2 two-stage training: Stage A distills the new decoder with "
            "Student-1 substitute layers frozen; Stage B fine-tunes the deployable "
            "Student-2 model except CLIP."
        )
    )
    p.add_argument("--gsnet-config", required=True)
    p.add_argument("--gsnet-weights", required=True)
    p.add_argument(
        "--s1-source",
        choices=["distill", "baseline", "custom"],
        default="distill",
        help=(
            "Which Student-1 checkpoint to initialise from. distill uses the "
            "GS-Distill Student-1 checkpoint; baseline uses the random-init "
            "Student-1 baseline checkpoint; custom requires --s1-ckpt."
        ),
    )
    p.add_argument(
        "--s1-ckpt",
        default=None,
        help="Explicit Student-1 checkpoint path. Overrides --s1-source defaults.",
    )
    p.add_argument(
        "--s1-distill-ckpt",
        default="output/ashie/clip/finetune_best.pth",
        help="Default checkpoint used when --s1-source distill.",
    )
    p.add_argument(
        "--s1-baseline-ckpt",
        default="output/ashie/baseline/baseline_best.pth",
        help="Default checkpoint used when --s1-source baseline.",
    )
    p.add_argument("--image-dir", required=True)
    p.add_argument("--label-dir", required=True)
    p.add_argument("--class-json", default="datasets/landdiscover.json")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to output/ashie/s2/<s1-source>.",
    )
    p.add_argument(
        "--distill-epochs",
        type=int,
        default=5,
        help="Stage A epochs: freeze Student-1 substitute layers and distill the new decoder.",
    )
    p.add_argument(
        "--finetune-epochs",
        type=int,
        default=15,
        help="Stage B epochs: fine-tune Student-2 after decoder distillation.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--distill-lr",
        type=float,
        default=5e-5,
        help="Stage A learning rate for the randomly initialised S2 decoder/projections.",
    )
    p.add_argument("--lr", type=float, default=1e-5, help="Stage B fine-tuning learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-dino", type=int, default=768)
    p.add_argument("--clip-layers", type=int, nargs="+", default=[8, 16, 20, 23])
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument(
        "--freeze-student-backbone",
        action="store_true",
        help=(
            "Keep Student-1 trunk/branches frozen during Stage B too. Stage A always "
            "freezes them; default Stage B unfreezes them for best accuracy recovery."
        ),
    )
    p.add_argument(
        "--distill-lambda-seg",
        type=float,
        default=0.5,
        help="Stage A CE weight. Set to 0 for pure teacher-only decoder distillation.",
    )
    p.add_argument(
        "--distill-lambda-logit",
        type=float,
        default=1.0,
        help="Stage A KL weight against GSNet teacher logits.",
    )
    p.add_argument(
        "--distill-lambda-feat",
        type=float,
        default=0.1,
        help="Stage A SmoothL1 weight against the teacher RIPD aggregation feature.",
    )
    p.add_argument(
        "--lambda-logit",
        type=float,
        default=0.25,
        help="Stage B KL weight against GSNet teacher logits.",
    )
    p.add_argument(
        "--lambda-feat",
        type=float,
        default=0.0,
        help="Stage B feature-distillation weight. Default disables feature pinning during joint fine-tune.",
    )
    p.add_argument("--head-mode", choices=["aggregator", "conv_plus_class"], default="aggregator")
    p.add_argument("--head-hidden-dim", type=int, default=64)
    p.add_argument("--head-num-layers", type=int, default=1)
    p.add_argument("--head-window-size", type=int, default=4)
    p.add_argument("--head-pad-len", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-project", default="gs-distill")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.resolved_s1_ckpt = _resolve_s1_ckpt(args)
    if args.output_dir is None:
        args.output_dir = os.path.join("output", "ashie", "s2", args.s1_source)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if not _WANDB_AVAILABLE and not args.no_wandb:
        print("W&B not installed; continuing without logging.")
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    print(f"Loading offline GSNet teacher from {args.gsnet_weights} ...")
    teacher = build_teacher(args.gsnet_config, args.gsnet_weights, str(device)).to(device)

    clip_model = teacher.sem_seg_head.predictor.clip_model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    head_kwargs = {
        "hidden_dim": args.head_hidden_dim,
        "num_layers": args.head_num_layers,
        "window_size": args.head_window_size,
        "pad_len": args.head_pad_len,
    }
    student2 = GSDistillStudent2(
        clip_model=clip_model,
        d_dino=args.d_dino,
        clip_layers=args.clip_layers,
        head_kwargs=head_kwargs,
        head_mode=args.head_mode,
    )
    print(f"Student-1 source: {args.s1_source}")
    print(f"Student-1 checkpoint: {args.resolved_s1_ckpt}")
    _load_s1_weights(student2, args.resolved_s1_ckpt, device)

    n_trainable = sum(p.numel() for p in student2.trainable_parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in student2.cost_head.parameters())
    print(f"Student-2 trainable params before freeze flag: {n_trainable / 1e6:.2f}M")
    print(f"S2CostHead params: {n_head / 1e6:.2f}M")

    run_finetune_s2(args, teacher, student2, device, args.output_dir, use_wandb)

    if use_wandb:
        wandb.finish()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nAll done.")


if __name__ == "__main__":
    main()
