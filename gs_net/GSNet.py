# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
import os
sys.path.insert(0, os.path.abspath('./detectron2'))

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange
from .vision_transformer import vit_base
from .dinov3_wrapper import DINOv3Wrapper
import os

def BuildRSIB(Weights):
    """
    Build and load RSIB (Remote Sensing Image Backbone) - DINOv3 only.
    
    Args:
        Weights: Path to DINOv3 checkpoint file.
                 Expected: ./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
               
    Returns:
        DINOv3Wrapper instance
        
    Raises:
        FileNotFoundError: If checkpoint file not found
    """
    if not os.path.isfile(Weights):
        raise FileNotFoundError(f"Checkpoint not found at {Weights}. Please verify the path exists.")
    
    print(f"\n{'='*80}")
    print("Loading DINOv3 Backbone (ViT-L/16)")
    print(f"  Checkpoint: {Weights}")
    print(f"  Patch Size: 16×16 → upsampled to 48×48")
    print(f"  Features: 1024 dims → projected to 768 dims")
    print(f"  Output: 12 × (B, 2305, 768)")
    print(f"{'='*80}\n")
    
    model = DINOv3Wrapper(checkpoint_path=Weights)
    return model


@META_ARCH_REGISTRY.register()
class GSNet(nn.Module):
    @configurable
    
    
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        dino: nn.Module,
        use_clip: bool,
        clip_decod_guid_dim: list,
        dino_decod_guid_dim: list,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.dino_model = dino
        self.use_clip = use_clip

        self.backbone = backbone
        self.clip_decod_dim = clip_decod_guid_dim
        self.dino_decod_dim = dino_decod_guid_dim
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if clip_finetune == "freezeIMG":
                if "attn" in name:
                    # QV fine-tuning for attention blocks
                    params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                elif "position" in name:
                    params.requires_grad = True
                else: params.requires_grad = False
                if "visual" in name:
                    params.requires_grad = False
                
            elif "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            
            else:
                params.requires_grad = False


        self.sliding_window = sliding_window
        if clip_pretrained == "ViT-B/16":
            self.clip_resolution = (384, 384)
        elif clip_pretrained == "RemoteCLIP-ViT-B-32":
            self.clip_resolution = (768,768)
        elif clip_pretrained == "RN101" or clip_pretrained == "RN50":
            # SOLUTION: Increase ResNet resolution from 224 to 384 to match DINOv3
            # Trade-off: Slightly different from CLIP pretraining (224), but better feature alignment
            # CLIP ResNet was pretrained at 224×224, but can handle 384×384 with interpolated pos encoding
            self.clip_resolution = (384, 384)  # Changed from (224, 224)
            print(f"[CLIP ResNet] Using 384×384 resolution (instead of pretrain 224×224)")
            print(f"  → Better alignment with DINOv3 and input images")
            print(f"  → May have slight domain shift from pretraining")
        else:
            self.clip_resolution = (336, 336)
        self.dino_resolution = (384,384)
        # RN101 uses 512-dim embeddings, ViT-B/16 uses 768-dim, ViT-L uses 1024-dim
        if clip_pretrained == "RN101" or clip_pretrained == "RN50":
            self.proj_dim = 512
        elif clip_pretrained == "ViT-B/16" or clip_pretrained == "RemoteCLIP-ViT-B-32":
            self.proj_dim = 768
        else:
            self.proj_dim = 1024

        print(f"[GSNet] CLIP Configuration:")
        print(f"  - Model: {clip_pretrained}")
        print(f"  - Resolution: {self.clip_resolution}")
        print(f"  - Projection Dim: {self.proj_dim}")
        print(f"  - Architecture: {'ResNet' if clip_pretrained in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'] else 'Vision Transformer'}")

        # Determine if using ResNet architecture
        self.is_resnet = clip_pretrained in ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"]

        # For ResNet, we need projection layers to match channel dimensions
        # RN101 layer structure: layer1=256ch, layer2=512ch, layer3=1024ch, layer4=2048ch
        if self.is_resnet:
            # Project ResNet features to match expected dimensions
            # SOLUTION 1 FIX: Added layer4 projection to restore spatial uniqueness in res3
            # Previously, res3 used expanded global features (all identical across spatial locations)
            # Now, res3 uses layer4 spatial features (unique per location) for proper correlation
            self.resnet_layer4_proj = nn.Conv2d(2048, self.proj_dim, kernel_size=1) if self.use_clip else None  # NEW: for res3
            self.resnet_layer2_proj = nn.Conv2d(512, self.proj_dim, kernel_size=1) if self.use_clip else None   # for res4
            self.resnet_layer3_proj = nn.Conv2d(1024, self.proj_dim, kernel_size=1) if self.use_clip else None  # for res5
            self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2) if self.use_clip and self.clip_decod_dim[0]!=0 else None
            self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4) if self.use_clip and self.clip_decod_dim[1]!=0 else None

            # CRITICAL FIX: Smart initialization for ResNet projection layers
            # Problem: Random init destroys pretrained ResNet features
            # Solution: Use small weights to preserve features while allowing learning
            if self.use_clip:
                print(f"\n[CLIP ResNet] Initializing projection layers with feature-preserving weights:")

                # Layer4 projection: 2048 → proj_dim (512 for RN101)
                # Use smaller weights since we're downsampling 4x
                if self.resnet_layer4_proj is not None:
                    nn.init.xavier_uniform_(self.resnet_layer4_proj.weight, gain=0.02)
                    if self.resnet_layer4_proj.bias is not None:
                        nn.init.zeros_(self.resnet_layer4_proj.bias)
                    print(f"  ✓ layer4_proj: 2048→{self.proj_dim} (gain=0.02)")

                # Layer2 projection: 512 → proj_dim
                # Identity-like since input/output dims are similar
                if self.resnet_layer2_proj is not None:
                    if self.proj_dim == 512:
                        # Same dims: use near-identity
                        nn.init.eye_(self.resnet_layer2_proj.weight.squeeze())
                        self.resnet_layer2_proj.weight.data *= 0.9
                    else:
                        # Different dims: use small weights
                        nn.init.xavier_uniform_(self.resnet_layer2_proj.weight, gain=0.02)
                    if self.resnet_layer2_proj.bias is not None:
                        nn.init.zeros_(self.resnet_layer2_proj.bias)
                    print(f"  ✓ layer2_proj: 512→{self.proj_dim} ({'identity' if self.proj_dim == 512 else 'gain=0.02'})")

                # Layer3 projection: 1024 → proj_dim
                if self.resnet_layer3_proj is not None:
                    nn.init.xavier_uniform_(self.resnet_layer3_proj.weight, gain=0.02)
                    if self.resnet_layer3_proj.bias is not None:
                        nn.init.zeros_(self.resnet_layer3_proj.bias)
                    print(f"  ✓ layer3_proj: 1024→{self.proj_dim} (gain=0.02)")

                print(f"  → All projections are TRAINABLE and will adapt during training")

                # SOLUTION: Add feature normalization for CNN→Decoder compatibility
                # Problem: ResNet features have different statistics than ViT features
                # - ResNet: local, sparse activations (ReLU-based)
                # - ViT: global, dense activations (GELU-based)
                # Solution: LayerNorm to standardize feature distributions
                print(f"\n[Architecture Bridge] Adding normalization for CNN→Transformer decoder:")
                self.clip_feature_norm = nn.LayerNorm(self.proj_dim)
                print(f"  ✓ LayerNorm({self.proj_dim}) - normalizes ResNet features to match decoder expectations")
        else:
            # For ViT, use standard upsample layers
            self.resnet_layer2_proj = None
            self.resnet_layer3_proj = None
            self.clip_feature_norm = None  # ViT doesn't need this
            self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2) if self.use_clip and self.clip_decod_dim[0]!=0 else None
            self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4) if self.use_clip and self.clip_decod_dim[1]!=0 else None

        self.dino_decod_proj1 = nn.Conv2d(in_channels = 768, out_channels=256, kernel_size=1, stride=1, padding=0) if self.dino_model and self.dino_decod_dim[0]!=0 else None
        self.dino_decod_proj2 = nn.ConvTranspose2d(in_channels= 768, out_channels=128, kernel_size=2, stride=2) if self.dino_model and self.dino_decod_dim[0]!=0 else None

        self.dino_down_sample = nn.Conv2d(in_channels=768, out_channels=self.proj_dim, kernel_size=2, stride=2, padding=0) if self.dino_model else None

        # Register forward hooks for intermediate features
        # Note: self.is_resnet already set at line 163
        if not self.is_resnet:
            # For ViT models, use transformer block hooks
            self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" or clip_pretrained == "RemoteCLIP-ViT-B-32" else [7, 15]
            self.layers = []
            if self.use_clip:
                for l in self.layer_indexes:
                    self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        else:
            # For ResNet models, use layer hooks instead
            # Note: ResNet features are CNN-based (local receptive fields) vs ViT features which are
            # attention-based (global receptive fields). This architectural difference is acceptable:
            # - ResNet excels at capturing local patterns and sharp boundaries
            # - DINOv3 provides complementary global/transformer-based features
            # - The decoder can adapt to work with both feature types
            self.layers = []
            if self.use_clip:
                # SOLUTION 1 FIX: Hook layer2, layer3, AND layer4 for spatial features
                # layer4 (2048ch, 7×7) → used for res3 (main spatial feature with unique locations)
                # layer2 (512ch, 28×28) → used for res4 (decoder guidance)
                # layer3 (1024ch, 14×14) → used for res5 (decoder guidance)
                # Why layer4? It's the deepest spatial layer before attnpool collapses to 1×1,
                # providing the most semantic features while maintaining spatial structure
                self.sem_seg_head.predictor.clip_model.visual.layer2.register_forward_hook(lambda m, _, o: self.layers.append(o))
                self.sem_seg_head.predictor.clip_model.visual.layer3.register_forward_hook(lambda m, _, o: self.layers.append(o))
                self.sem_seg_head.predictor.clip_model.visual.layer4.register_forward_hook(lambda m, _, o: self.layers.append(o))  # NEW!


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        if cfg.MODEL.SEM_SEG_HEAD.USE_DINO_CORR:
            # Use environment variable if set, otherwise use default path
            rsib_ckpt = os.getenv('RSIB_CKPT', './dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
            dino = BuildRSIB(rsib_ckpt)
            dino_ft = cfg.MODEL.SEM_SEG_HEAD.DINO_FINETUNE
            for name, params in dino.named_parameters():
                if dino_ft == "attention":

                    if "attn.qkv.weight" in name:
                        params.requires_grad = True
                    elif "pos_embed" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif dino_ft == "full":
                    params.requires_grad = True
                else:
                    # CRITICAL FIX: Always keep dim_projection trainable even when freezing backbone
                    if "dim_projection" in name:
                        params.requires_grad = True
                        print(f"[DINOv3] Keeping projection layer trainable: {name}")
                    else:
                        params.requires_grad = False

        else:
            dino = None
            

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "dino": dino, 
            "use_clip":cfg.MODEL.SEM_SEG_HEAD.USE_CLIP_CORR, 
            "clip_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS,
            "dino_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_DIMS
            
        }

    @property
    def device(self):
        return self.pixel_mean.device
    # @profile(precision=4,stream=open('./log.txt','w+',encoding="utf-8"))
    def forward(self, batched_inputs):

        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        if self.training:
            images = [x["image"].to(self.device) for x in batched_inputs]
            # images_shape: 384*384
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        
            self.layers = []

            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
            dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )
            # clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        elif not self.sliding_window:
            with torch.no_grad():
                images = [x["image"].to(self.device) for x in batched_inputs]
                clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
                clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
            
                self.layers = []

                clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
                dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )
        elif self.sliding_window:
            with torch.no_grad():
                kernel=384
                overlap=0.333
                out_res=[640, 640]
                images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
                stride = int(kernel * (1 - overlap))
                unfold = nn.Unfold(kernel_size=kernel, stride=stride)
                fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

                image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
                image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
                global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
                image = torch.cat((image, global_image), dim=0)

                images = (image - self.pixel_mean) / self.pixel_std
                clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
                clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
                dino_images_resized = F.interpolate(clip_images, size=self.dino_resolution, mode='bilinear', align_corners=False, )
                self.layers = []
                
        
        if self.dino_model is not None:
            dino_feat = self.dino_model.get_intermediate_layers(dino_images_resized, n=12) # actually only 12 layers, but use a large num to avoid ambiguity
            dino_patch_feat_last_unfold = rearrange(dino_feat[-1][:,1:,:],"B (H W) C -> B C H W", H=48)
            dino_feat_down = self.dino_down_sample(dino_patch_feat_last_unfold) # B,512,24,24
            dino_feat_L4 = rearrange(dino_feat[3][:,1:,:],"B (H W) C -> B C H W", H=48)
            dino_feat_L8 = rearrange(dino_feat[7][:,1:,:],"B (H W) C -> B C H W", H=48)
            
            dino_feat_L4_proj = self.dino_decod_proj1(dino_feat_L4) if self.dino_decod_proj1 is not None else None
            dino_feat_L8_proj = self.dino_decod_proj2(dino_feat_L8) if self.dino_decod_proj2 is not None else None
            dino_feat_guidance = [dino_feat_L4_proj,dino_feat_L8_proj]
        else:
            dino_feat_down, dino_feat_guidance = None, None
        
        if self.use_clip:
            clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)

            if self.is_resnet:
                # ============================================================================
                # SOLUTION 1 FIX: Use layer4 spatial features for res3
                # ============================================================================
                # PROBLEM (old code): res3 was created by expanding global features, making
                # all 576 spatial locations identical. This broke spatial reasoning in the
                # correlation module - every location had the same text-image similarity!
                #
                # SOLUTION: Use layer4 (deepest spatial layer) for res3 instead of global.
                # This provides unique features per spatial location, enabling proper
                # spatial discrimination in correlation and attention mechanisms.
                # ============================================================================

                # Extract hooked features from ResNet layers
                # self.layers[0] = layer2 (512ch, ~28×28) → for res4 (decoder guidance)
                # self.layers[1] = layer3 (1024ch, ~14×14) → for res5 (decoder guidance)
                # self.layers[2] = layer4 (2048ch, ~7×7) → for res3 (main spatial feature) ← NEW!
                layer2_resnet = self.layers[0]  # (B, 512, H, W)
                layer3_resnet = self.layers[1]  # (B, 1024, H, W)
                layer4_resnet = self.layers[2]  # (B, 2048, H, W) ← NEW!

                # ============================================================================
                # Create res3 from layer4 spatial features (FIXED VERSION)
                # ============================================================================
                # Interpolate layer4 to 24×24 and project to 512 channels
                # This gives us UNIQUE features per spatial location (not all identical!)
                res3_temp = F.interpolate(layer4_resnet, size=(24, 24), mode='bilinear', align_corners=False)  # (B, 2048, 24, 24)
                res3 = self.resnet_layer4_proj(res3_temp)  # (B, 512, 24, 24) with UNIQUE spatial features ✓

                # VERIFICATION: Check spatial uniqueness (only in training, first batch)
                if self.training and not hasattr(self, '_verified_spatial_uniqueness'):
                    with torch.no_grad():
                        spatial_tokens = res3.reshape(res3.shape[0], res3.shape[1], -1)  # (B, C, 576)
                        # Check if different spatial locations have different features
                        loc0 = spatial_tokens[:, :, 0]  # First location
                        loc100 = spatial_tokens[:, :, 100]  # Middle location
                        difference = (loc0 - loc100).abs().mean().item()
                        if difference < 0.01:
                            print(f"⚠️  WARNING: Spatial features are too similar! diff={difference:.6f}")
                            print(f"    This suggests projection may not be working correctly")
                        else:
                            print(f"✓ Spatial discrimination verified: diff={difference:.4f} (good!)")
                    self._verified_spatial_uniqueness = True

                # Create proper token sequence for head: CLS token + spatial tokens from res3
                clip_image_features = rearrange(res3, "B C H W -> B (H W) C")  # (B, 576, 512) - each location unique!
                cls_token = clip_features.unsqueeze(1)  # (B, 1, 512) - global feature from attnpool

                # SOLUTION: Apply feature normalization for CNN→Decoder compatibility
                # Normalizes ResNet features to have similar statistics as ViT features
                if self.clip_feature_norm is not None:
                    clip_image_features = self.clip_feature_norm(clip_image_features)  # (B, 576, 512) - normalized
                    cls_token = self.clip_feature_norm(cls_token)  # (B, 1, 512) - normalized

                clip_features = torch.cat([cls_token, clip_image_features], dim=1)  # (B, 577, 512)

                # ============================================================================
                # OLD CODE (COMMENTED FOR REFERENCE - DO NOT DELETE)
                # ============================================================================
                # This was the problematic code that created identical spatial tokens:
                #
                # clip_image_features = clip_features.unsqueeze(1)  # (B, 1, 512) - ONE global feature
                # spatial_tokens = clip_image_features.expand(B, 24*24, self.proj_dim)  # Copy 576 times ← PROBLEM!
                # clip_features = torch.cat([clip_image_features, spatial_tokens], dim=1)  # All identical!
                # res3 = clip_image_features.squeeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 24, 24)
                #
                # Why this was wrong:
                # - spatial_tokens[:, 0, :] == spatial_tokens[:, 575, :] (all identical)
                # - Correlation module got same value at every spatial location
                # - No spatial discrimination possible!
                # ============================================================================

                # Process decoder guidance features (res4, res5) - unchanged from before
                # 1. Interpolate to target spatial size (24×24)
                res4 = F.interpolate(layer2_resnet, size=(24, 24), mode='bilinear', align_corners=False)
                res5 = F.interpolate(layer3_resnet, size=(24, 24), mode='bilinear', align_corners=False)

                # 2. Project to expected channel dimensions (proj_dim: 512 for RN101, 768 for ViT-B, etc.)
                res4 = self.resnet_layer2_proj(res4) if self.resnet_layer2_proj is not None else res4
                res5 = self.resnet_layer3_proj(res5) if self.resnet_layer3_proj is not None else res5

                # 3. Apply upsample layers to match decoder expectations
                res4 = self.upsample1(res4) if self.upsample1 is not None else None  # (B, 256, 48, 48)
                res5 = self.upsample2(res5) if self.upsample2 is not None else None  # (B, 128, 96, 96)
            else:
                # For ViT: features are patch tokens that need reshaping
                clip_image_features = clip_features[:, 1:, :]
                res3 = rearrange(clip_image_features, "B (H W) C -> B C H W", H=24)
                res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
                res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
                res4 = self.upsample1(res4) if self.upsample1 is not None else None
                res5 = self.upsample2(res5) if self.upsample2 is not None else None

            clip_features_guidance = {'res5': res5, 'res4': res4, 'res3': res3,}
        else:
            clip_features, clip_features_guidance=None, None

        outputs = self.sem_seg_head(clip_features,dino_feat_down, clip_features_guidance, dino_feat_guidance)
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses
        elif self.sliding_window:
            with torch.no_grad():
                outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
                outputs = outputs.sigmoid()
                
                global_output = outputs[-1:]
                global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
                outputs = outputs[:-1]
                outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
                outputs = (outputs + global_output) / 2.

                height = batched_inputs[0].get("height", out_res[0])
                width = batched_inputs[0].get("width", out_res[1])
                output = sem_seg_postprocess(outputs[0], out_res, height, width)
                return [{'sem_seg': output}]
        
        else:
            with torch.no_grad():
                outputs = outputs.sigmoid()
                image_size = clip_images.image_sizes[0]
                height = batched_inputs[0].get("height", image_size[0])
                width = batched_inputs[0].get("width", image_size[1])

                output = sem_seg_postprocess(outputs[0], image_size, height, width)
                processed_results = [{'sem_seg': output}]
                return processed_results