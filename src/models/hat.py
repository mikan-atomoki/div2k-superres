"""
HAT: Hybrid Attention Transformer for Image Super-Resolution
Standalone implementation (no basicsr dependency).
Reference: "Activating More Pixels in Image Super-Resolution Transformer" (CVPR 2023)
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def to_2tuple(x):
    if isinstance(x, Sequence) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """Channel Attention Block: Conv-LeakyReLU-Conv-ChannelAttention."""

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# ---------------------------------------------------------------------------
# Window Self-Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# HAB: Hybrid Attention Block (Window SA + Channel Attention)
# ---------------------------------------------------------------------------

class HAB(nn.Module):
    """Hybrid Attention Block combining Window Self-Attention with Channel Attention."""

    def __init__(self, dim: int, input_resolution: tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, compress_ratio: int = 3,
                 squeeze_factor: int = 30, conv_scale: float = 0.01,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # Compute attention mask for shifted window
        if self.shift_size > 0:
            h, w = self.input_resolution
            img_mask = torch.zeros((1, h, w, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv branch (channel attention)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c)

        # Combine attention + conv
        x = shortcut + self.drop_path(x) + conv_x * self.conv_scale

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# OCAB: Overlapping Cross-Attention Block
# ---------------------------------------------------------------------------

class OCAB(nn.Module):
    """Overlapping Cross-Attention Block for inter-window information exchange."""

    def __init__(self, dim: int, input_resolution: tuple[int, int], window_size: int,
                 overlap_ratio: float, num_heads: int, qkv_bias: bool = True,
                 qk_scale: float | None = None, mlp_ratio: float = 2.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

        # Relative position bias for cross attention
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Relative position index
        coords_h_q = torch.arange(self.window_size)
        coords_w_q = torch.arange(self.window_size)
        coords_q = torch.stack(torch.meshgrid(coords_h_q, coords_w_q, indexing="ij"))
        coords_q_flatten = torch.flatten(coords_q, 1)

        coords_h_k = torch.arange(self.overlap_win_size)
        coords_w_k = torch.arange(self.overlap_win_size)
        coords_k = torch.stack(torch.meshgrid(coords_h_k, coords_w_k, indexing="ij"))
        coords_k_flatten = torch.flatten(coords_k, 1)

        relative_coords = coords_q_flatten[:, :, None] - coords_k_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.overlap_win_size - 1
        relative_coords[:, :, 1] += self.overlap_win_size - 1
        relative_coords[:, :, 0] *= window_size + self.overlap_win_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, B, C, H, W
        q = qkv[0].permute(0, 2, 3, 1)  # B, H, W, C
        kv = torch.cat([qkv[1], qkv[2]], dim=1)  # B, 2C, H, W

        # Partition q into windows
        q = window_partition(q, self.window_size)
        q = q.view(-1, self.window_size * self.window_size, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        # Unfold kv with overlap
        kv = self.unfold(kv)  # B, 2C*overlap_win_size^2, nW
        kv = rearrange(
            kv, "b (nc owsh owsw) nw -> (b nw) nc owsh owsw",
            nc=2 * c, owsh=self.overlap_win_size, owsw=self.overlap_win_size,
        )
        kv = rearrange(
            kv, "bnw (two c) owsh owsw -> two bnw (owsh owsw) c",
            two=2,
        )
        k, v = kv.unbind(0)
        k = k.view(-1, self.overlap_win_size * self.overlap_win_size, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(-1, self.overlap_win_size * self.overlap_win_size, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, c)

        # Merge windows
        x = x.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(x, self.window_size, h, w)
        x = x.view(b, h * w, c)

        x = self.proj(x)

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# AttenBlocks: Sequence of HABs (used inside RHAG)
# ---------------------------------------------------------------------------

class AttenBlocks(nn.Module):
    """A sequence of HAB blocks."""

    def __init__(self, dim: int, input_resolution: tuple[int, int], depth: int,
                 num_heads: int, window_size: int, compress_ratio: int,
                 squeeze_factor: int, conv_scale: float, overlap_ratio: float,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float | list[float] = 0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                conv_scale=conv_scale, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.overlap_attn = OCAB(
            dim=dim, input_resolution=input_resolution, window_size=window_size,
            overlap_ratio=overlap_ratio, num_heads=num_heads, qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.overlap_attn(x, x_size)
        return x


# ---------------------------------------------------------------------------
# RHAG: Residual Hybrid Attention Group
# ---------------------------------------------------------------------------

class RHAG(nn.Module):
    """Residual Hybrid Attention Group: AttenBlocks + Conv + residual."""

    def __init__(self, dim: int, input_resolution: tuple[int, int], depth: int,
                 num_heads: int, window_size: int, compress_ratio: int,
                 squeeze_factor: int, conv_scale: float, overlap_ratio: float,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float | list[float] = 0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.residual_group = AttenBlocks(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size,
            compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
            conv_scale=conv_scale, overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer,
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return self.patch_embed(
            self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
        ) + x


# ---------------------------------------------------------------------------
# Patch Embed / Unembed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to Patch Embedding (flatten spatial dims)."""

    def __init__(self, embed_dim: int = 96, norm_layer: type | None = None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B HW C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Patch Embedding to Image (unflatten spatial dims)."""

    def __init__(self, embed_dim: int = 96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, *x_size)


# ---------------------------------------------------------------------------
# Upsample modules
# ---------------------------------------------------------------------------

class Upsample(nn.Sequential):
    """Upsample via PixelShuffle: Conv -> PixelShuffle (repeated for each 2x step)."""

    def __init__(self, scale: int, num_feat: int):
        layers = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale: {scale}. Supported: powers of 2 or 3.")
        super().__init__(*layers)


# ---------------------------------------------------------------------------
# HAT: Main model
# ---------------------------------------------------------------------------

class HAT(nn.Module):
    """
    Hybrid Attention Transformer (HAT) for image super-resolution.

    Args:
        img_size: Input image spatial size (used to compute attention masks).
        in_chans: Number of input image channels.
        embed_dim: Patch embedding dimension.
        depths: Depth of each RHAG.
        num_heads: Number of attention heads in each RHAG.
        window_size: Window size for self-attention.
        compress_ratio: Compress ratio for CAB.
        squeeze_factor: Squeeze factor for channel attention.
        conv_scale: Scale for conv branch in HAB.
        overlap_ratio: Overlap ratio for OCAB.
        mlp_ratio: MLP hidden dim ratio.
        upscale: Upscale factor (2, 3, or 4).
        img_range: Image pixel range (default 1.0).
        upsampler: Upsampler type, 'pixelshuffle' or 'pixelshuffledirect'.
        resi_connection: Residual connection type, '1conv' or '3conv'.
    """

    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        window_size: int = 16,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        overlap_ratio: float = 0.5,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
        **kwargs,
    ):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        # Mean shift
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.register_buffer("mean", torch.Tensor(rgb_mean).view(1, 3, 1, 1))

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Patches resolution (for computing attn masks)
        patches_resolution = [img_size, img_size]

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build RHAG layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        # Residual connection conv
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # Reconstruction
        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad image to be divisible by window_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # Mean shift
        x = (x - self.mean) * self.img_range

        # Pad
        x = self.check_image_size(x)

        # Shallow feature
        x = self.conv_first(x)
        # Deep feature
        x = self.conv_after_body(self.forward_features(x)) + x
        # Reconstruction
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        # Mean unshift
        x = x / self.img_range + self.mean

        return x[:, :, : h * self.upscale, : w * self.upscale]
