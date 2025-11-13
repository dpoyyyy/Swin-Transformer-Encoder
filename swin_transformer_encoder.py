# --------------------------------------------------------
# Swin Transformer Blocks & BINA Encoder
# Based on: Swin Transformer (Microsoft Research Asia) — MIT License
# Original Repo: https://github.com/microsoft/Swin-Transformer
# Adapted by Danial Farshbaf for the BINA Project
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Tuple, List, Optional


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used in Vision Transformer and related architectures.

    Applies two linear transformations with GELU activation and dropout in between.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden layer features.
        out_features (int, optional): Number of output features.
        act_layer (nn.Module, optional): Activation function class. Default is nn.GELU.
        drop (float, optional): Dropout rate.

    Returns:
        torch.Tensor: Transformed tensor with same or different channel dimensions.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: int):
    """
    Splits feature map into non-overlapping local windows.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): Window size.

    Returns:
        torch.Tensor: Flattened windows of shape (num_windows * B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    ws = min(H, W, window_size)
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Reverses the window partition process to reconstruct spatial dimensions.

    Args:
        windows (torch.Tensor): Tensor of shape (num_windows * B, window_size, window_size, C).
        window_size (int): Window size.
        H (int): Original height.
        W (int): Original width.

    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA) module.

    Implements relative position bias and efficient local attention computation
    within non-overlapping windows.

    Args:
        dim (int): Number of input channels.
        window_size (Tuple[int, int]): Height and width of the attention window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, adds bias to QKV projections.
        attn_drop (float, optional): Dropout ratio on attention weights.
        proj_drop (float, optional): Dropout ratio after linear projection.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads,
                             N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer Block with windowed multi-head self-attention (W-MSA) or shifted windows (SW-MSA).

    Each block consists of:
        - Layer normalization
        - W-MSA or SW-MSA attention
        - Residual connections
        - Feed-forward MLP sub-layer

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size for self-attention.
        shift_size (int): Shifted window size.
        mlp_ratio (float): Hidden layer expansion ratio in MLP.
        qkv_bias (bool): Whether to use bias in QKV projections.
        drop (float): Dropout rate.
        attn_drop (float): Dropout on attention probabilities.
        drop_path (float): Drop path rate for stochastic depth.
        act_layer (nn.Module): Activation function.
        norm_layer (nn.Module): Normalization layer.
        use_checkpoint (bool): Whether to use gradient checkpointing for memory efficiency.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, mask_matrix=None):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=mask_matrix)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer that concatenates features of 2x2 neighboring patches and applies linear projection.

    Args:
        input_resolution (Tuple[int, int]): Resolution of input feature map.
        dim (int): Number of input channels.

    Returns:
        torch.Tensor: Downsampled tensor with doubled channel dimension.
    """

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """
    A hierarchical stage in Swin Transformer composed of multiple SwinTransformerBlocks and optional PatchMerging.

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple[int, int]): Spatial resolution of input.
        depth (int): Number of SwinTransformerBlocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Expansion ratio in MLP.
        qkv_bias (bool): Whether to use bias in QKV projections.
        drop (float): Dropout rate.
        attn_drop (float): Dropout on attention probabilities.
        drop_path (float or list): Drop path rate(s).
        norm_layer (nn.Module): Normalization layer.
        downsample (nn.Module, optional): PatchMerging layer for downsampling.
        use_checkpoint (bool): Whether to use gradient checkpointing.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
                                 norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)
        ])

        self.downsample = downsample(
            input_resolution, dim) if downsample is not None else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x

class BINA_Swin_Encoder(nn.Module):
    """
    Main encoder module based on Swin Transformer for BINA project.

    Designed to process spatiotemporal data (video or multi-frame input).
    Generates skip features at multiple scales and a final bottleneck representation.

    Args:
        img_size (int): Input image size.
        patch_size (int): Size of initial patch embedding.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension of first stage.
        depths (List[int]): Number of Swin blocks per stage.
        num_heads (List[int]): Number of attention heads per stage.
        window_size (int): Attention window size.
        drop_path_rate (float): Drop path rate.
        pretrained_weights_path (str, optional): Path to pretrained Swin weights.
        use_checkpoint (bool): Whether to enable checkpointing for memory saving.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: Skip connections and final bottleneck tensor.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 drop_path_rate=0.1,
                 pretrained_weights_path=None,
                 use_checkpoint=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(p=0.)

        self.layers = nn.ModuleList()
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(img_size // patch_size //
                                  (2 ** i_layer), img_size // patch_size // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        if pretrained_weights_path is not None:
            self._load_pretrained_weights(pretrained_weights_path)

    def _load_pretrained_weights(self, pretrained_weights_path: str):
        """
        Loads pretrained Swin Transformer weights and adapts them to the BINA encoder if needed.

        Args:
            pretrained_weights_path (str): Path to pretrained model checkpoint.

        Notes:
            - Adapts input projection weights if input channels differ.
            - Ignores head layers from pretrained checkpoint.
        """
        state_dict = torch.load(pretrained_weights_path, map_location='cpu')

        if 'model' in state_dict:
            state_dict = state_dict['model']

        if 'patch_embed.proj.weight' in state_dict:
            old_weight = state_dict['patch_embed.proj.weight']
            if old_weight.shape[1] != self.in_chans:
                if old_weight.shape[1] == 3:
                    new_weight = torch.zeros_like(
                        self.patch_embed.weight, dtype=old_weight.dtype)
                    new_weight[:, :3, :, :] = old_weight
                    mean_rgb = old_weight.mean(dim=1, keepdim=True)
                    new_weight[:, 3:, :, :] = mean_rgb
                    state_dict['patch_embed.proj.weight'] = new_weight
                else:
                    del state_dict['patch_embed.proj.weight']

        state_dict = {k: v for k, v in state_dict.items()
                      if 'head' not in k and 'fc' not in k}
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x_video):
        """
        Forward pass of BINA Swin Encoder.

        Args:
            x_video (torch.Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]:
                - skip_outputs_flat: List of skip connection features.
                - bottleneck_img_flat: Final bottleneck feature map.
        """
        B, T, C, H, W = x_video.shape
        x = x_video.view(B * T, C, H, W)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        skip_outputs_flat = []

        H_layer = H // self.patch_size
        W_layer = W // self.patch_size

        for i, layer in enumerate(self.layers):
            x_out, x = layer(x, H_layer, W_layer)
            skip_outputs_flat.append(
                x_out.transpose(1, 2).view(B * T, -1, H_layer, W_layer))
            if layer.downsample is not None:
                H_layer, W_layer = H_layer // 2, W_layer // 2

        bottleneck = self.norm(x)
        bottleneck_img_flat = bottleneck.transpose(
            1, 2).view(B * T, -1, H_layer, W_layer)

        return skip_outputs_flat, bottleneck_img_flat
