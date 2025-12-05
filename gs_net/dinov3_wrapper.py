

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
            RuntimeError: If checkpoint loading fails
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        
        # 1. Create base ViT-L/16 model via timm
        print(f"Creating ViT-L/16 model from timm...")
        self.model = timm.create_model('vit_l16', pretrained=False, num_classes=0)
        
        # 2. Load local checkpoint
        print(f"Loading checkpoint from: {self.checkpoint_path}")
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
        print(f"Model loaded with msg: {msg}")
        
        # 5. Set to eval mode
        self.model.eval()
        
        # 6. Create dimension projection: 1024 → 768
        # Initialized with small weights (no training needed)
        self.dim_projection = nn.Linear(1024, 768, bias=False)
        self.dim_projection.weight.data.normal_(0, 0.01)
        
        # 7. Set up layer index mapping for 24 blocks
        # Map 12 output indices to 24 block indices (evenly spaced)
        # Formula: block_idx = 2 * (output_idx + 1)
        # Output 0 → Block 2, Output 1 → Block 4, ..., Output 11 → Block 24
        self.block_indices = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        
        # 8. Initialize hook storage
        self.hooks = []
        self.activations = []
    
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
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                state_dict = checkpoint['state_dict']
            elif 'teacher' in checkpoint and isinstance(checkpoint['teacher'], dict):
                # DINOv1-style teacher checkpoint
                state_dict = checkpoint['teacher']
            # Check if it's already a state_dict
            elif 'pos_embed' in checkpoint or any('blocks' in k for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                raise ValueError(
                    f"Unknown checkpoint format. Expected keys: 'model', 'state_dict', 'teacher', "
                    f"or direct state_dict. Got: {list(checkpoint.keys())[:5]}"
                )
        else:
            raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
        
        # Remove common prefixes if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
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
           - Separate CLS and patches
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
            # activation shape: (B, 577, 1024) or (B, 576, 1024)
            # Need to ensure CLS token is present
            
            # Ensure CLS token is present
            if activation.shape[1] == 576:  # Patches only (should not happen, but handle it)
                # Add dummy CLS token at beginning
                B = activation.shape[0]
                cls_token = torch.zeros((B, 1, activation.shape[2]), device=activation.device, dtype=activation.dtype)
                activation = torch.cat([cls_token, activation], dim=1)
            
            # activation shape: (B, 577, 1024)
            # Project dimensions: 1024 → 768
            activation = self.dim_projection(activation)
            # activation shape: (B, 577, 768)
            
            # Separate CLS and patches
            cls_token = activation[:, :1, :]  # (B, 1, 768)
            patches = activation[:, 1:, :]    # (B, 576, 768)
            
            # Reshape patches to spatial format for upsampling
            # (B, 576, 768) → (B, 24, 24, 768) → (B, 768, 24, 24)
            B = patches.shape[0]
            patches = patches.view(B, 24, 24, 768).permute(0, 3, 1, 2)  # (B, 768, 24, 24)
            
            # Upsample spatial resolution: 24×24 → 48×48
            patches = F.interpolate(patches, scale_factor=2, mode='bilinear', align_corners=False)
            # patches shape: (B, 768, 48, 48)
            
            # Reshape back to sequence format
            patches = patches.permute(0, 2, 3, 1).reshape(B, -1, 768)  # (B, 2304, 768)
            
            # Concatenate CLS token back
            output = torch.cat([cls_token, patches], dim=1)  # (B, 2305, 768)
            outputs.append(output)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
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
        return
    
    # Test 2: Move to device
    print("Test 2: Moving to CUDA...")
    try:
        if torch.cuda.is_available():
            wrapper = wrapper.cuda()
            print("✓ Model moved to CUDA\n")
        else:
            print("⚠ CUDA not available, skipping device test\n")
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
    except Exception as e:
        print(f"✗ Statistics check failed: {e}\n")
        return
    
    print("="*80)
    print("All tests passed! ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_wrapper()
