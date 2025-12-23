import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List
import os


class DINOv3Wrapper(nn.Module):
    """
    Wrapper for DINOv3 (ViT-L/16) to match DINOv1 (ViT-B/8) interface.
    
    Architecture differences handled:
    - Patch size: 16×16 (DINOv3) vs 8×8 (DINOv1)
    - Feature dimension: 1024 (DINOv3) vs 768 (DINOv1)
    - Depth: 24 blocks (DINOv3) vs 12 blocks (DINOv1)
    - Output grid: 24×24 (DINOv3) → upsampled to 48×48 (DINOv1)
    
    Attributes:
        model: timm ViT-L/16 model loaded from checkpoint
        dim_projection: Linear layer for 1024→768 dimension reduction
        block_indices: List of block indices to extract from (evenly sampled)
        hooks: List of registered hook handles
        activations: Storage for hooked activations during forward pass
    """
    
    def __init__(self, checkpoint_path: str = "./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"):
        """
        Initialize DINOv3Wrapper with local checkpoint.
        
        Args:
            checkpoint_path: Path to local DINOv3 checkpoint file.
                           Default: "./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
                           
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails or timm version is too old
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        
        # 0. Verify timm version supports DINOv3
        try:
            timm_version = timm.__version__
            print(f"timm version: {timm_version}")
            # DINOv3 support added in timm 1.0.20
            version_parts = timm_version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            patch = int(version_parts[2].split('dev')[0]) if len(version_parts) > 2 else 0
            
            if major < 1 or (major == 1 and minor == 0 and patch < 20):
                raise RuntimeError(
                    f"timm version {timm_version} is too old. "
                    f"DINOv3 requires timm >= 1.0.20. "
                    f"Please upgrade: pip install --upgrade timm"
                )
        except Exception as e:
            print(f"⚠️  Warning: Could not verify timm version: {e}")
        
        # 1. Create base ViT-L/16 model via timm
        print(f"Creating ViT-L/16 model from timm...")
        try:
            self.model = timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=False, num_classes=0)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to create model 'vit_large_patch16_dinov3.sat493m'. "
                f"This usually means your timm version doesn't support DINOv3. "
                f"Please upgrade: pip install --upgrade timm>=1.0.20\n"
                f"Original error: {e}"
            )
        
        # Verify model architecture
        print(f"Model architecture:")
        print(f"  - Number of blocks: {len(self.model.blocks)}")
        print(f"  - Embedding dim: {self.model.embed_dim}")
        print(f"  - Patch size: {self.model.patch_embed.patch_size}")
        
        # Verify we have 24 blocks for ViT-L
        assert len(self.model.blocks) == 24, \
            f"Expected 24 blocks for ViT-L, got {len(self.model.blocks)}"
        assert self.model.embed_dim == 1024, \
            f"Expected 1024 dim for ViT-L, got {self.model.embed_dim}"
        
        # 2. Load local checkpoint
        print(f"\nLoading checkpoint from: {self.checkpoint_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                f"Please verify the path exists."
            )
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # 3. Handle different checkpoint formats
        state_dict = self._extract_state_dict(checkpoint)
        
        # 4. Load with flexible key matching
        print(f"Loading state dict into model...")
        msg = self.model.load_state_dict(state_dict, strict=False)
        
        # Check if critical keys are missing
        if msg.missing_keys:
            print(f"⚠️  Warning: {len(msg.missing_keys)} missing keys")
            # Print first few missing keys for debugging
            if len(msg.missing_keys) <= 10:
                for key in msg.missing_keys:
                    print(f"    - {key}")
            else:
                print(f"    First 10: {msg.missing_keys[:10]}")
                print(f"    ... and {len(msg.missing_keys) - 10} more")
        
        if msg.unexpected_keys:
            print(f"⚠️  Warning: {len(msg.unexpected_keys)} unexpected keys")
            if len(msg.unexpected_keys) <= 10:
                for key in msg.unexpected_keys:
                    print(f"    - {key}")
            else:
                print(f"    First 10: {msg.unexpected_keys[:10]}")
                print(f"    ... and {len(msg.unexpected_keys) - 10} more")
        
        if not msg.missing_keys and not msg.unexpected_keys:
            print("✓ Model loaded successfully with all keys matched!")
        else:
            print("⚠️  Model loaded with some key mismatches (this may be OK)")
        
        # 5. Set to eval mode
        self.model.eval()
        
        # 6. Create dimension projection: 1024 → 768
        # SOLUTION: Use identity-like initialization to preserve DINOv3 features
        # Project first 768 dims with identity, learn residual for remaining 256 dims
        self.dim_projection = nn.Linear(1024, 768, bias=False)

        # Initialize as: out = W @ in, where W[:768, :768] ≈ I (identity for first 768 dims)
        # This preserves most DINOv3 info while allowing learning
        with torch.no_grad():
            # Start with small random weights
            self.dim_projection.weight.data.zero_()
            # Set first 768x768 block to identity (preserve main features)
            self.dim_projection.weight.data[:, :768] = torch.eye(768) * 0.9
            # Add small noise for remaining 256 dims (learn from extra capacity)
            self.dim_projection.weight.data[:, 768:] = torch.randn(768, 256) * 0.01

        print(f"\n[DINOv3 Projection] Initialized with identity-preserving weights:")
        print(f"  - First 768 dims: 0.9×Identity (preserve features)")
        print(f"  - Last 256 dims: Random (learn compression)")
        print(f"  - Projection is TRAINABLE (will adapt during training)")
        
        # 7. Set up layer index mapping for 24 blocks
        # Map 12 output indices to 24 block indices (evenly spaced)
        # 24 blocks with 0-based indexing: [0, 1, 2, ..., 23]
        # Sample every 2nd block: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        self.block_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        
        # Verify block indices are valid
        assert max(self.block_indices) < len(self.model.blocks), \
            f"Block index {max(self.block_indices)} out of range for {len(self.model.blocks)} blocks"
        assert len(self.block_indices) == 12, \
            f"Expected 12 block indices, got {len(self.block_indices)}"
        
        print(f"\nBlock sampling strategy:")
        print(f"  - Total blocks: {len(self.model.blocks)}")
        print(f"  - Sampled blocks: {self.block_indices}")
        
        # 8. Initialize hook storage
        self.hooks = []
        self.activations = []
        
        print("\n" + "="*80)
        print("DINOv3Wrapper initialized successfully!")
        print("="*80 + "\n")
    
    def _extract_state_dict(self, checkpoint):
        """
        Extract state_dict from various checkpoint formats.
        
        Handles:
        - Direct state_dict
        - Wrapped in 'model' key
        - Wrapped in 'state_dict' key
        - Teacher-student format
        
        Args:
            checkpoint: Loaded checkpoint (dict or state_dict)
            
        Returns:
            state_dict: Model weights dictionary
        """
        if isinstance(checkpoint, dict):
            # Check for common wrapper keys
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                print(f"  Checkpoint format: wrapped in 'model' key")
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                print(f"  Checkpoint format: wrapped in 'state_dict' key")
                state_dict = checkpoint['state_dict']
            elif 'teacher' in checkpoint and isinstance(checkpoint['teacher'], dict):
                # DINOv1-style teacher checkpoint
                print(f"  Checkpoint format: teacher-student (using 'teacher')")
                state_dict = checkpoint['teacher']
            # Check if it's already a state_dict
            elif 'pos_embed' in checkpoint or any('blocks' in k for k in checkpoint.keys()):
                print(f"  Checkpoint format: direct state_dict")
                state_dict = checkpoint
            else:
                # Try to be more flexible - take the first dict-like value
                print(f"  Checkpoint keys: {list(checkpoint.keys())[:10]}")
                raise ValueError(
                    f"Unknown checkpoint format. Expected keys: 'model', 'state_dict', 'teacher', "
                    f"or direct state_dict. Got: {list(checkpoint.keys())[:5]}"
                )
        else:
            raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
        
        # Remove common prefixes if present
        original_keys = len(state_dict)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        if len(state_dict) != original_keys:
            print(f"  Removed prefixes from state_dict keys")
        
        print(f"  State dict contains {len(state_dict)} keys")
        
        return state_dict
    
    def _save_activation(self, module, input, output):
        """
        Hook callback to save intermediate activations.
        
        Args:
            module: The module being hooked
            input: Module input (unused)
            output: Module output to save
        """
        self.activations.append(output)
    
    def get_intermediate_layers(self, x: torch.Tensor, n: int = 12) -> List[torch.Tensor]:
        """
        Extract intermediate layer features from DINOv3 and adapt to DINOv1 format.
        
        This is the main interface method that replaces DINOv1's get_intermediate_layers().
        
        Processing pipeline:
        1. Register forward hooks on selected blocks
        2. Forward pass through model (hooks collect activations)
        3. For each hooked activation:
           - Ensure CLS token is present
           - Project dimension: 1024 → 768
           - Separate CLS and patches (skip 4 register tokens)
           - Upsample spatial resolution: 24×24 → 48×48
           - Reshape to sequence format: (B, 2305, 768)
        4. Remove hooks and return outputs
        
        Args:
            x: Input tensor, shape (B, 3, 384, 384)
            n: Number of layers to extract (default 12, full depth)
               Currently unused but kept for API compatibility with DINOv1
               
        Returns:
            List of 12 tensors, each shape (B, 2305, 768)
            - 2305 = 1 CLS token + 48×48 patches
            - 768 = projected feature dimension
            
        Example:
            >>> wrapper = DINOv3Wrapper(checkpoint_path)
            >>> x = torch.randn(2, 3, 384, 384)
            >>> features = wrapper.get_intermediate_layers(x, n=12)
            >>> len(features)
            12
            >>> features[0].shape
            torch.Size([2, 2305, 768])
        """
        # Clear previous activations
        self.activations = []
        
        # Register hooks on selected blocks
        hooks = []
        for block_idx in self.block_indices:
            hook = self.model.blocks[block_idx].register_forward_hook(self._save_activation)
            hooks.append(hook)
        
        # Forward pass through model (hooks will collect activations)
        with torch.no_grad():
            _ = self.model(x)
        
        # Process each activation
        outputs = []
        for activation in self.activations:
            # activation shape: (B, 581, 1024) for DINOv3 with register tokens
            # Structure: [CLS, REG1, REG2, REG3, REG4, PATCH1, ..., PATCH576]
            # Total: 1 CLS + 4 registers + 576 patches = 581 tokens

            B = activation.shape[0]

            # Project dimensions: 1024 → 768
            activation = self.dim_projection(activation)
            # activation shape: (B, 581, 768)

            # Separate CLS, register tokens, and patches
            cls_token = activation[:, :1, :].contiguous()  # (B, 1, 768)
            # Skip register tokens (indices 1-4) and take only patch tokens (indices 5-)
            patches = activation[:, 5:, :].contiguous()    # (B, 576, 768)

            # Verify patch count
            assert patches.shape[1] == 576, \
                f"Expected 576 patches (24×24), got {patches.shape[1]}"

            # Reshape patches to spatial format for upsampling
            # (B, 576, 768) → (B, 24, 24, 768) → (B, 768, 24, 24)
            patches = patches.view(B, 24, 24, 768).permute(0, 3, 1, 2)  # (B, 768, 24, 24)
            
            # Upsample spatial resolution: 24×24 → 48×48
            patches = F.interpolate(patches, scale_factor=2, mode='bilinear', align_corners=False)
            # patches shape: (B, 768, 48, 48)
            
            # Reshape back to sequence format
            patches = patches.permute(0, 2, 3, 1).reshape(B, -1, 768)  # (B, 2304, 768)
            
            # Verify upsampled patch count
            assert patches.shape[1] == 2304, \
                f"Expected 2304 patches (48×48), got {patches.shape[1]}"
            
            # Concatenate CLS token back
            output = torch.cat([cls_token, patches], dim=1)  # (B, 2305, 768)
            # Ensure output is contiguous for downstream operations (like einops rearrange)
            output = output.contiguous()
            outputs.append(output)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Verify output count and shapes
        assert len(outputs) == 12, f"Expected 12 outputs, got {len(outputs)}"
        for i, output in enumerate(outputs):
            assert output.shape[1] == 2305, \
                f"Output {i} has wrong sequence length: {output.shape[1]}"
            assert output.shape[2] == 768, \
                f"Output {i} has wrong feature dim: {output.shape[2]}"
        
        return outputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (extract final CLS token only).
        
        Args:
            x: Input tensor, shape (B, 3, H, W)
            
        Returns:
            CLS token features, shape (B, 1024)
        """
        return self.model(x)
    
    def to(self, *args, **kwargs):
        """Move model and adapters to specified device/dtype."""
        super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        self.dim_projection = self.dim_projection.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        """Move model to CUDA."""
        self.to(torch.device('cuda', device) if device is not None else torch.device('cuda'))
        return self
    
    def cpu(self):
        """Move model to CPU."""
        self.to(torch.device('cpu'))
        return self


def test_wrapper():
    """
    Simple test function to verify wrapper works correctly.
    
    This can be run independently to debug wrapper initialization and forward pass.
    """
    print("\n" + "="*80)
    print("Testing DINOv3Wrapper")
    print("="*80 + "\n")
    
    # Test 1: Load wrapper
    print("Test 1: Loading wrapper...")
    try:
        checkpoint_path = "./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        wrapper = DINOv3Wrapper(checkpoint_path)
        print("✓ Wrapper loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load wrapper: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Move to device
    print("Test 2: Moving to CUDA...")
    try:
        if torch.cuda.is_available():
            wrapper = wrapper.cuda()
            print("✓ Model moved to CUDA\n")
        else:
            print("⚠ CUDA not available, using CPU\n")
    except Exception as e:
        print(f"✗ Failed to move to device: {e}\n")
        return
    
    # Test 3: Forward pass with dummy input
    print("Test 3: Forward pass with dummy input...")
    try:
        x = torch.randn(2, 3, 384, 384)
        if torch.cuda.is_available():
            x = x.cuda()
        
        features = wrapper.get_intermediate_layers(x, n=12)
        
        # Check output
        assert len(features) == 12, f"Expected 12 features, got {len(features)}"
        assert features[0].shape == (2, 2305, 768), f"Wrong output shape: {features[0].shape}"
        for i, feat in enumerate(features):
            assert feat.shape == (2, 2305, 768), f"Feature {i} has wrong shape: {feat.shape}"
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output count: {len(features)}")
        print(f"  - Output shape: {features[0].shape}\n")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Check feature values
    print("Test 4: Checking feature statistics...")
    try:
        mean_val = features[0].mean().item()
        std_val = features[0].std().item()
        min_val = features[0].min().item()
        max_val = features[0].max().item()
        
        print(f"✓ Feature statistics:")
        print(f"  - Mean: {mean_val:.6f}")
        print(f"  - Std: {std_val:.6f}")
        print(f"  - Min: {min_val:.6f}")
        print(f"  - Max: {max_val:.6f}\n")
        
        # Sanity checks
        if abs(mean_val) > 100:
            print(f"⚠️  Warning: Mean value seems unusually large")
        if std_val < 0.01 or std_val > 100:
            print(f"⚠️  Warning: Std value seems unusual")
            
    except Exception as e:
        print(f"✗ Statistics check failed: {e}\n")
        return
    
    print("="*80)
    print("All tests passed! ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_wrapper()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from typing import List
# import os


# class DINOv3Wrapper(nn.Module):
#     """
#     Wrapper for DINOv3 (ViT-L/16) to match DINOv1 (ViT-B/8) interface.
    
#     Architecture differences handled:
#     - Patch size: 16×16 (DINOv3) vs 8×8 (DINOv1)
#     - Feature dimension: 1024 (DINOv3) vs 768 (DINOv1)
#     - Depth: 24 blocks (DINOv3) vs 12 blocks (DINOv1)
#     - Output grid: 24×24 (DINOv3) → upsampled to 48×48 (DINOv1)
    
#     Attributes:
#         model: timm ViT-L/16 model loaded from checkpoint
#         dim_projection: Linear layer for 1024→768 dimension reduction
#         block_indices: List of block indices to extract from (evenly sampled)
#         hooks: List of registered hook handles
#         activations: Storage for hooked activations during forward pass
#     """
    
#     def __init__(self, checkpoint_path: str = "./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"):
#         """
#         Initialize DINOv3Wrapper with local checkpoint.
        
#         Args:
#             checkpoint_path: Path to local DINOv3 checkpoint file.
#                            Default: "./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
                           
#         Raises:
#             FileNotFoundError: If checkpoint file doesn't exist
#             RuntimeError: If checkpoint loading fails
#         """
#         super().__init__()
        
#         self.checkpoint_path = checkpoint_path
        
#         # 1. Create base ViT-L/16 model via timm
#         print(f"Creating ViT-L/16 model from timm...")
#         self.model = timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=False, num_classes=0)
        
#         # 2. Load local checkpoint
#         print(f"Loading checkpoint from: {self.checkpoint_path}")
#         if not os.path.isfile(self.checkpoint_path):
#             raise FileNotFoundError(
#                 f"Checkpoint not found at {self.checkpoint_path}. "
#                 f"Please verify the path exists."
#             )
        
#         checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
#         # 3. Handle different checkpoint formats
#         state_dict = self._extract_state_dict(checkpoint)
        
#         # 4. Load with flexible key matching
#         print(f"Loading state dict into model...")
#         msg = self.model.load_state_dict(state_dict, strict=False)
#         print(f"Model loaded with msg: {msg}")
        
#         # 5. Set to eval mode
#         self.model.eval()
        
#         # 6. Create dimension projection: 1024 → 768
#         # Initialized with small weights (no training needed)
#         self.dim_projection = nn.Linear(1024, 768, bias=False)
#         self.dim_projection.weight.data.normal_(0, 0.01)
        
#         # 7. Set up layer index mapping for 24 blocks
#         # Map 12 output indices to 24 block indices (evenly spaced)
#         # 24 blocks with 0-based indexing: [0, 1, 2, ..., 23]
#         # Sample every 2nd block: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
#         self.block_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        
#         # 8. Initialize hook storage
#         self.hooks = []
#         self.activations = []
    
#     def _extract_state_dict(self, checkpoint):
#         """
#         Extract state_dict from various checkpoint formats.
        
#         Handles:
#         - Direct state_dict
#         - Wrapped in 'model' key
#         - Wrapped in 'state_dict' key
#         - Teacher-student format
        
#         Args:
#             checkpoint: Loaded checkpoint (dict or state_dict)
            
#         Returns:
#             state_dict: Model weights dictionary
#         """
#         if isinstance(checkpoint, dict):
#             # Check for common wrapper keys
#             if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
#                 state_dict = checkpoint['model']
#             elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
#                 state_dict = checkpoint['state_dict']
#             elif 'teacher' in checkpoint and isinstance(checkpoint['teacher'], dict):
#                 # DINOv1-style teacher checkpoint
#                 state_dict = checkpoint['teacher']
#             # Check if it's already a state_dict
#             elif 'pos_embed' in checkpoint or any('blocks' in k for k in checkpoint.keys()):
#                 state_dict = checkpoint
#             else:
#                 raise ValueError(
#                     f"Unknown checkpoint format. Expected keys: 'model', 'state_dict', 'teacher', "
#                     f"or direct state_dict. Got: {list(checkpoint.keys())[:5]}"
#                 )
#         else:
#             raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
        
#         # Remove common prefixes if present
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
#         return state_dict
    
#     def _save_activation(self, module, input, output):
#         """
#         Hook callback to save intermediate activations.
        
#         Args:
#             module: The module being hooked
#             input: Module input (unused)
#             output: Module output to save
#         """
#         self.activations.append(output)
    
#     def get_intermediate_layers(self, x: torch.Tensor, n: int = 12) -> List[torch.Tensor]:
#         """
#         Extract intermediate layer features from DINOv3 and adapt to DINOv1 format.
        
#         This is the main interface method that replaces DINOv1's get_intermediate_layers().
        
#         Processing pipeline:
#         1. Register forward hooks on selected blocks
#         2. Forward pass through model (hooks collect activations)
#         3. For each hooked activation:
#            - Ensure CLS token is present
#            - Project dimension: 1024 → 768
#            - Separate CLS and patches
#            - Upsample spatial resolution: 24×24 → 48×48
#            - Reshape to sequence format: (B, 2305, 768)
#         4. Remove hooks and return outputs
        
#         Args:
#             x: Input tensor, shape (B, 3, 384, 384)
#             n: Number of layers to extract (default 12, full depth)
#                Currently unused but kept for API compatibility with DINOv1
               
#         Returns:
#             List of 12 tensors, each shape (B, 2305, 768)
#             - 2305 = 1 CLS token + 48×48 patches
#             - 768 = projected feature dimension
            
#         Example:
#             >>> wrapper = DINOv3Wrapper(checkpoint_path)
#             >>> x = torch.randn(2, 3, 384, 384)
#             >>> features = wrapper.get_intermediate_layers(x, n=12)
#             >>> len(features)
#             12
#             >>> features[0].shape
#             torch.Size([2, 2305, 768])
#         """
#         # Clear previous activations
#         self.activations = []
        
#         # Register hooks on selected blocks
#         hooks = []
#         for block_idx in self.block_indices:
#             hook = self.model.blocks[block_idx].register_forward_hook(self._save_activation)
#             hooks.append(hook)
        
#         # Forward pass through model (hooks will collect activations)
#         with torch.no_grad():
#             _ = self.model(x)
        
#         # Process each activation
#         outputs = []
#         for activation in self.activations:
#             # activation shape: (B, 581, 1024) for DINOv3 with register tokens
#             # Structure: [CLS, REG1, REG2, REG3, REG4, PATCH1, ..., PATCH576]
#             # Total: 1 CLS + 4 registers + 576 patches = 581 tokens

#             B = activation.shape[0]

#             # Project dimensions: 1024 → 768
#             activation = self.dim_projection(activation)
#             # activation shape: (B, 581, 768)

#             # Separate CLS, register tokens, and patches
#             cls_token = activation[:, :1, :]  # (B, 1, 768)
#             # Skip register tokens (indices 1-4) and take only patch tokens (indices 5-)
#             patches = activation[:, 5:, :]    # (B, 576, 768)
            
#             # Reshape patches to spatial format for upsampling
#             # (B, 576, 768) → (B, 24, 24, 768) → (B, 768, 24, 24)
#             B = patches.shape[0]
#             patches = patches.view(B, 24, 24, 768).permute(0, 3, 1, 2)  # (B, 768, 24, 24)
            
#             # Upsample spatial resolution: 24×24 → 48×48
#             patches = F.interpolate(patches, scale_factor=2, mode='bilinear', align_corners=False)
#             # patches shape: (B, 768, 48, 48)
            
#             # Reshape back to sequence format
#             patches = patches.permute(0, 2, 3, 1).reshape(B, -1, 768)  # (B, 2304, 768)
            
#             # Concatenate CLS token back
#             output = torch.cat([cls_token, patches], dim=1)  # (B, 2305, 768)
#             outputs.append(output)
        
#         # Remove hooks
#         for hook in hooks:
#             hook.remove()
        
#         return outputs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Standard forward pass (extract final CLS token only).
        
#         Args:
#             x: Input tensor, shape (B, 3, H, W)
            
#         Returns:
#             CLS token features, shape (B, 1024)
#         """
#         return self.model(x)
    
#     def to(self, *args, **kwargs):
#         """Move model and adapters to specified device/dtype."""
#         super().to(*args, **kwargs)
#         self.model = self.model.to(*args, **kwargs)
#         self.dim_projection = self.dim_projection.to(*args, **kwargs)
#         return self
    
#     def cuda(self, device=None):
#         """Move model to CUDA."""
#         self.to(torch.device('cuda', device) if device is not None else torch.device('cuda'))
#         return self
    
#     def cpu(self):
#         """Move model to CPU."""
#         self.to(torch.device('cpu'))
#         return self


# def test_wrapper():
#     """
#     Simple test function to verify wrapper works correctly.
    
#     This can be run independently to debug wrapper initialization and forward pass.
#     """
#     print("\n" + "="*80)
#     print("Testing DINOv3Wrapper")
#     print("="*80 + "\n")
    
#     # Test 1: Load wrapper
#     print("Test 1: Loading wrapper...")
#     try:
#         checkpoint_path = "./dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
#         wrapper = DINOv3Wrapper(checkpoint_path)
#         print("✓ Wrapper loaded successfully\n")
#     except Exception as e:
#         print(f"✗ Failed to load wrapper: {e}\n")
#         return
    
#     # Test 2: Move to device
#     print("Test 2: Moving to CUDA...")
#     try:
#         if torch.cuda.is_available():
#             wrapper = wrapper.cuda()
#             print("✓ Model moved to CUDA\n")
#         else:
#             print("⚠ CUDA not available, skipping device test\n")
#     except Exception as e:
#         print(f"✗ Failed to move to device: {e}\n")
#         return
    
#     # Test 3: Forward pass with dummy input
#     print("Test 3: Forward pass with dummy input...")
#     try:
#         x = torch.randn(2, 3, 384, 384)
#         if torch.cuda.is_available():
#             x = x.cuda()
        
#         features = wrapper.get_intermediate_layers(x, n=12)
        
#         # Check output
#         assert len(features) == 12, f"Expected 12 features, got {len(features)}"
#         assert features[0].shape == (2, 2305, 768), f"Wrong output shape: {features[0].shape}"
#         for i, feat in enumerate(features):
#             assert feat.shape == (2, 2305, 768), f"Feature {i} has wrong shape: {feat.shape}"
        
#         print(f"✓ Forward pass successful")
#         print(f"  - Input shape: {x.shape}")
#         print(f"  - Output count: {len(features)}")
#         print(f"  - Output shape: {features[0].shape}\n")
#     except Exception as e:
#         print(f"✗ Forward pass failed: {e}\n")
#         return
    
#     # Test 4: Check feature values
#     print("Test 4: Checking feature statistics...")
#     try:
#         mean_val = features[0].mean().item()
#         std_val = features[0].std().item()
#         min_val = features[0].min().item()
#         max_val = features[0].max().item()
        
#         print(f"✓ Feature statistics:")
#         print(f"  - Mean: {mean_val:.6f}")
#         print(f"  - Std: {std_val:.6f}")
#         print(f"  - Min: {min_val:.6f}")
#         print(f"  - Max: {max_val:.6f}\n")
#     except Exception as e:
#         print(f"✗ Statistics check failed: {e}\n")
#         return
    
#     print("="*80)
#     print("All tests passed! ✓")
#     print("="*80 + "\n")


# if __name__ == "__main__":
#     test_wrapper()
