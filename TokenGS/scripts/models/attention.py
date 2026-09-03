# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Attention and patch embedding layers used by the inference graph."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(0.0)

    def forward(self, value: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(value)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Attention dimension must be divisible by the head count")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, value: Tensor) -> Tensor:
        batch, count, channels = value.shape
        qkv = self.qkv(value).reshape(batch, count, 3, self.num_heads, self.head_dim)
        query, key, val = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        result = F.scaled_dot_product_attention(self.q_norm(query), self.k_norm(key), val)
        result = result.transpose(1, 2).reshape(batch, count, channels)
        return self.proj_drop(self.proj(result))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.gamma


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_chans: int,
        embed_dim: int,
        norm_layer: Callable[..., nn.Module] | None,
    ):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = True
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, value: Tensor) -> Tensor:
        height, width = value.shape[-2:]
        if height % self.patch_size[0] or width % self.patch_size[1]:
            raise ValueError("Input dimensions must be divisible by the patch size")
        return self.norm(self.proj(value).flatten(2).transpose(1, 2))
