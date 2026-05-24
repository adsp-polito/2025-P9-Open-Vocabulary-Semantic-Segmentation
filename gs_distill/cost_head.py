"""
Lightweight CAT-Seg-inspired cost aggregation head for Student-2.

The head consumes a fused class cost volume and produces RIPD-free
segmentation logits. Tensor convention inside the aggregator is
``(B, C, T, H, W)`` where ``T`` is the active text/class dimension.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def _gn_groups(channels: int) -> int:
    groups = max(1, channels // 16)
    while channels % groups != 0:
        groups -= 1
    return groups


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: image height
        W: image width

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        appearance_guidance_dim,
        num_heads,
        head_dim=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, dim + guidance_dim)
            mask: (num_windows, N, N) or None
        """
        B_, N, _ = x.shape

        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, : self.dim]).reshape(B_, N, self.num_heads, -1)
        v = v.permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        appearance_guidance_dim,
        input_resolution,
        num_heads=4,
        head_dim=None,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.appearance_guidance_dim = appearance_guidance_dim
        self.input_resolution = _to_2tuple(input_resolution)
        self.window_size = window_size[0] if isinstance(window_size, tuple) else window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            appearance_guidance_dim=appearance_guidance_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=_to_2tuple(self.window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            ):
                for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                ):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.appearance_guidance_dim > 0:
            if appearance_guidance is None:
                appearance_guidance = x.new_zeros(B, H * W, self.appearance_guidance_dim)
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, x_windows.shape[-1]
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinBlockWrapper(nn.Module):
    def __init__(
        self,
        dim,
        appearance_guidance_dim,
        input_resolution,
        nheads=4,
        window_size=4,
    ):
        super().__init__()
        self.appearance_guidance_dim = appearance_guidance_dim
        self.input_resolution = _to_2tuple(input_resolution)
        self.block_1 = SwinTransformerBlock(
            dim,
            appearance_guidance_dim,
            self.input_resolution,
            num_heads=nheads,
            window_size=window_size,
            shift_size=0,
        )
        self.block_2 = SwinTransformerBlock(
            dim,
            appearance_guidance_dim,
            self.input_resolution,
            num_heads=nheads,
            window_size=window_size,
            shift_size=window_size // 2,
        )
        self.guidance_norm = (
            nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None
        )

    def forward(self, x, appearance_guidance):
        """
        Args:
            x: (B, C, T, H, W)
            appearance_guidance: (B, Cg, H, W)
        """
        B, C, T, H, W = x.shape
        if (H, W) != self.input_resolution:
            raise ValueError(
                f"SwinBlockWrapper expected {self.input_resolution}, got {(H, W)}"
            )

        x = rearrange(x, "B C T H W -> (B T) (H W) C")
        if self.appearance_guidance_dim > 0:
            if appearance_guidance is not None:
                if appearance_guidance.shape[-2:] != (H, W):
                    appearance_guidance = F.interpolate(
                        appearance_guidance, size=(H, W), mode="bilinear", align_corners=False
                    )
                g = repeat(appearance_guidance, "B C H W -> (B T) (H W) C", T=T)
                g = self.guidance_norm(g)
            else:
                g = x.new_zeros(B * T, H * W, self.appearance_guidance_dim)
        else:
            g = None

        x = self.block_1(x, g)
        x = self.block_2(x, g)
        return rearrange(x, "(B T) (H W) C -> B C T H W", B=B, T=T, H=H, W=W)


def elu_feature_map(x):
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q, k, v):
        q = elu_feature_map(q)
        k = elu_feature_map(k)
        v_length = v.size(1)
        v = v / v_length
        KV = torch.einsum("nshd,nshv->nhdv", k, v)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)
        out = torch.einsum("nlhd,nhdv,nlh->nlhv", q, KV, Z) * v_length
        return out.contiguous()


class ClassTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        text_guidance_dim,
        nheads,
        pooling_size=(4, 4),
        pad_len=0,
    ):
        super().__init__()
        assert hidden_dim % nheads == 0, "hidden_dim must be divisible by nheads"
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.text_guidance_dim = text_guidance_dim
        self.q = nn.Linear(hidden_dim + text_guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + text_guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.attn = LinearAttention()
        self.nheads = nheads
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pad_len = pad_len
        self.padding_tokens = (
            nn.Parameter(torch.zeros(1, hidden_dim, 1, 1, 1)) if pad_len > 0 else None
        )
        self.padding_guidance = (
            nn.Parameter(torch.zeros(1, 1, text_guidance_dim))
            if pad_len > 0 and text_guidance_dim > 0
            else None
        )

    def _pool_features(self, x):
        B = x.size(0)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.pool(x)
        return rearrange(x, "(B T) C H W -> B C T H W", B=B)

    def _select_or_pad(self, x_pool, guidance, x_ref):
        B, C, T, Hp, Wp = x_pool.shape
        selected_idx = None
        orig_T = T

        if self.text_guidance_dim > 0 and guidance is None:
            guidance = x_ref.new_zeros(B, T, self.text_guidance_dim)

        if self.pad_len <= 0:
            return x_pool, guidance, selected_idx, orig_T

        if T > self.pad_len:
            scores = x_ref.detach().float().mean(dim=1).mean(dim=(-1, -2))
            selected_idx = scores.topk(self.pad_len, dim=1).indices.sort(dim=1).values
            gather_x = selected_idx[:, None, :, None, None].expand(
                B, C, self.pad_len, Hp, Wp
            )
            x_pool = x_pool.gather(dim=2, index=gather_x)
            if guidance is not None:
                gather_g = selected_idx[:, :, None].expand(
                    B, self.pad_len, guidance.shape[-1]
                )
                guidance = guidance.gather(dim=1, index=gather_g)
            return x_pool, guidance, selected_idx, orig_T

        if T < self.pad_len:
            pad_count = self.pad_len - T
            pad_tokens = self.padding_tokens.expand(B, C, pad_count, Hp, Wp)
            x_pool = torch.cat([x_pool, pad_tokens.to(dtype=x_pool.dtype)], dim=2)
            if self.text_guidance_dim > 0:
                pad_g = self.padding_guidance.expand(B, pad_count, self.text_guidance_dim)
                guidance = torch.cat([guidance, pad_g.to(dtype=guidance.dtype)], dim=1)

        return x_pool, guidance, selected_idx, orig_T

    def forward(self, x, guidance):
        """
        Args:
            x: (B, C, T, H, W)
            guidance: (B, T, Cg)
        """
        B, C, T, H, W = x.size()
        xp = self._pool_features(x)
        xp, guidance, selected_idx, orig_T = self._select_or_pad(xp, guidance, x)
        _, _, T_eff, Hp, Wp = xp.shape

        xp = rearrange(xp, "B C T H W -> (B H W) T C")
        xn = self.norm1(xp)
        if self.text_guidance_dim > 0:
            g = repeat(guidance, "B T C -> (B H W) T C", H=Hp, W=Wp)
            q = self.q(torch.cat([xn, g], dim=-1))
            k = self.k(torch.cat([xn, g], dim=-1))
        else:
            q = self.q(xn)
            k = self.k(xn)
        v = self.v(xn)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.nheads)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.nheads)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.nheads)
        a = rearrange(self.attn(q, k, v), "B L H D -> B L (H D)")

        xp = xp + a
        xp = xp + self.MLP(self.norm2(xp))
        xp = rearrange(xp, "(B H W) T C -> (B T) C H W", H=Hp, W=Wp)
        xp = F.interpolate(xp, size=(H, W), mode="bilinear", align_corners=True)
        xp = rearrange(xp, "(B T) C H W -> B C T H W", B=B, T=T_eff)

        if selected_idx is not None:
            scattered = x.new_zeros(B, C, orig_T, H, W)
            scatter_idx = selected_idx[:, None, :, None, None].expand(
                B, C, self.pad_len, H, W
            )
            scattered.scatter_(2, scatter_idx, xp)
            xp = scattered
        elif self.pad_len > 0 and orig_T < self.pad_len:
            xp = xp[:, :, :orig_T]

        return x + xp


class AggregatorLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        text_guidance_dim=32,
        appearance_guidance_dim=32,
        nheads=4,
        input_resolution=(24, 24),
        pooling_size=(4, 4),
        window_size=4,
        pad_len=0,
    ):
        super().__init__()
        self.spatial = SwinBlockWrapper(
            hidden_dim,
            appearance_guidance_dim,
            input_resolution,
            nheads=nheads,
            window_size=window_size,
        )
        self.cls = ClassTransformerLayer(
            hidden_dim,
            text_guidance_dim,
            nheads=nheads,
            pooling_size=pooling_size,
            pad_len=pad_len,
        )

    def forward(self, x, appearance_guidance, text_guidance):
        x = self.spatial(x, appearance_guidance)
        x = self.cls(x, text_guidance)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid=None):
        super().__init__()
        mid = mid or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, guidance_ch):
        super().__init__()
        if guidance_ch >= in_ch:
            raise ValueError("guidance_ch must be smaller than in_ch")
        self.guidance_ch = guidance_ch
        self.up = nn.ConvTranspose2d(
            in_ch, in_ch - guidance_ch, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, guidance, batch_size: Optional[int] = None):
        x = self.up(x)
        if self.guidance_ch > 0:
            if guidance is None:
                if batch_size is None:
                    raise ValueError("batch_size is required when guidance is None")
                guidance = x.new_zeros(
                    batch_size, self.guidance_ch, x.shape[-2], x.shape[-1]
                )
            if guidance.shape[-2:] != x.shape[-2:]:
                guidance = F.interpolate(
                    guidance, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)


class CostResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(hidden_dim), hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        y = rearrange(x, "B C T H W -> (B T) C H W")
        y = self.net(y)
        y = rearrange(y, "(B T) C H W -> B C T H W", B=B, T=T)
        return x + y


class S2CostHead(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=1,
        nheads=4,
        window_size=4,
        pooling_size=(4, 4),
        pad_len=0,
        appearance_guidance_proj_dim=32,
        text_guidance_proj_dim=32,
        dec_dims=(32, 16),
        clip_feat_dim=1024,
        clip_text_dim=512,
        dino_feat_dim=768,
        clip_skip_dim=None,
        dino_skip_dim=None,
        input_resolution=(24, 24),
        decoder_guidance_proj_dims=None,
        head_mode="aggregator",
    ):
        super().__init__()
        if head_mode not in {"aggregator", "conv_plus_class"}:
            raise ValueError(f"Unknown head_mode: {head_mode}")
        self.hidden_dim = hidden_dim
        self.head_mode = head_mode
        self.text_guidance_proj_dim = text_guidance_proj_dim
        self.clip_text_dim = clip_text_dim
        self.input_resolution = _to_2tuple(input_resolution)

        clip_skip_dim = clip_skip_dim or clip_feat_dim
        dino_skip_dim = dino_skip_dim or dino_feat_dim
        if decoder_guidance_proj_dims is None:
            decoder_guidance_proj_dims = (
                max(4, hidden_dim // 4),
                max(4, dec_dims[0] // 4),
            )

        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.gate = nn.Conv2d(2, 1, kernel_size=1)

        self.appearance_proj = nn.Sequential(
            nn.Conv2d(clip_feat_dim, appearance_guidance_proj_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(clip_text_dim, text_guidance_proj_dim),
            nn.ReLU(inplace=True),
        )

        if head_mode == "aggregator":
            self.layers = nn.ModuleList(
                [
                    AggregatorLayer(
                        hidden_dim=hidden_dim,
                        text_guidance_dim=text_guidance_proj_dim,
                        appearance_guidance_dim=appearance_guidance_proj_dim,
                        nheads=nheads,
                        input_resolution=self.input_resolution,
                        pooling_size=pooling_size,
                        window_size=window_size,
                        pad_len=pad_len,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.conv_blocks = None
            self.class_layer = None
        else:
            self.layers = nn.ModuleList()
            self.conv_blocks = nn.ModuleList(
                [CostResidualBlock(hidden_dim), CostResidualBlock(hidden_dim)]
            )
            self.class_layer = ClassTransformerLayer(
                hidden_dim,
                text_guidance_proj_dim,
                nheads=nheads,
                pooling_size=pooling_size,
                pad_len=pad_len,
            )

        self.dec_proj0 = nn.Sequential(
            nn.Conv2d(clip_skip_dim, decoder_guidance_proj_dims[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_proj1 = nn.Sequential(
            nn.Conv2d(dino_skip_dim, decoder_guidance_proj_dims[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = Up(hidden_dim, dec_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(dec_dims[0], dec_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(dec_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.last_corr_embed = None

    def corr_embed(self, x):
        B = x.shape[0]
        x = rearrange(x, "B P T H W -> (B T) P H W")
        x = self.conv1(x)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def fuse_costs(self, C_clip, C_dino):
        if C_clip.shape != C_dino.shape:
            raise ValueError(
                f"C_clip and C_dino must have the same shape, got {C_clip.shape} and {C_dino.shape}"
            )
        B, T, H, W = C_clip.shape
        pair = torch.stack([C_clip, C_dino], dim=2)
        pair = rearrange(pair, "B T C H W -> (B T) C H W")
        gate = torch.sigmoid(self.gate(pair))
        gate = rearrange(gate, "(B T) 1 H W -> B T H W", B=B, T=T)
        return gate * C_clip + (1.0 - gate) * C_dino

    def _text_guidance(self, E_Q):
        if E_Q.dim() == 4:
            E_Q = E_Q.mean(dim=2)
            E_Q = F.normalize(E_Q, dim=-1)
        if E_Q.shape[-1] == self.text_guidance_proj_dim:
            return E_Q
        if E_Q.shape[-1] != self.clip_text_dim:
            raise ValueError(
                f"Expected text dim {self.clip_text_dim} or {self.text_guidance_proj_dim}, got {E_Q.shape[-1]}"
            )
        return self.text_proj(E_Q)

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.decoder1(x, guidance[0], batch_size=B)
        x = self.decoder2(x, guidance[1], batch_size=B)
        x = self.head(x)
        return rearrange(x, "(B T) 1 H W -> B T H W", B=B)

    def forward(self, C_fused, E_Q, F_G_res3, F_G_res4, dino_L4):
        """
        Args:
            C_fused: (B, T, H, W)
            E_Q: (B, T, C) or (B, T, P, C)
            F_G_res3: dense CLIP feature map at cost resolution
            F_G_res4: CLIP skip used by decoder stage 1
            dino_L4: Student-1 DINO-substitute skip used by decoder stage 2
        """
        if C_fused.dim() == 4:
            C_fused = C_fused.unsqueeze(1)
        x = self.corr_embed(C_fused)
        self.last_corr_embed = x

        H, W = x.shape[-2:]
        app_g = self.appearance_proj(F_G_res3)
        if app_g.shape[-2:] != (H, W):
            app_g = F.interpolate(app_g, size=(H, W), mode="bilinear", align_corners=False)
        text_g = self._text_guidance(E_Q)

        if self.head_mode == "aggregator":
            for layer in self.layers:
                x = layer(x, app_g, text_g)
        else:
            for block in self.conv_blocks:
                x = block(x)
            x = self.class_layer(x, text_g)

        dec_g = [
            self.dec_proj0(F_G_res4) if F_G_res4 is not None else None,
            self.dec_proj1(dino_L4) if dino_L4 is not None else None,
        ]
        return self.conv_decoder(x, dec_g)
