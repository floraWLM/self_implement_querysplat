# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""QuerySplat decoder and encoder-memory projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scripts.models.attention import Attention, LayerScale, Mlp
from scripts.options import Options


class DecoderBlock(nn.Module):
    class SelfAttnBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, qkv_bias: bool, qk_norm: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(dim, num_heads, qkv_bias, qk_norm)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.gs_self_attn(self.norm(tokens))

    class CrossAttnBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, qkv_bias: bool, q_norm: bool):
            super().__init__()
            self.num_heads = num_heads
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.out_proj = nn.Linear(dim, dim)

        def forward(self, tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            query = rearrange(
                self.q_proj(self.gs_token_norm(tokens)),
                "b n (h d) -> b h n d",
                h=self.num_heads,
            )
            result = F.scaled_dot_product_attention(self.q_norm(query), keys, values)
            return self.out_proj(rearrange(result, "b h n d -> b n (h d)"))

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float,
    ):
        super().__init__()
        self.gs_self_attn = self.SelfAttnBlock(dim, num_heads, qkv_bias, qk_norm)
        self.gs_self_attn_scale = LayerScale(dim, init_values)
        self.gs_cross_attn = self.CrossAttnBlock(dim, num_heads, qkv_bias, qk_norm)
        self.gs_cross_attn_scale = LayerScale(dim, init_values)
        self.mlp = self.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = LayerScale(dim, init_values)

    def forward(self, tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.gs_cross_attn_scale(self.gs_cross_attn(tokens, keys, values))
        tokens = tokens + self.gs_self_attn_scale(self.gs_self_attn(tokens))
        return tokens + self.mlp_scale(self.mlp(tokens))


class EncDecBackbone(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.encoder_norm = nn.LayerNorm(opt.enc_embed_dim)
        self.kv_proj = nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim * 2, bias=True)
        self.k_proj_norm = nn.LayerNorm(opt.enc_embed_dim // opt.enc_num_heads)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    opt.enc_embed_dim,
                    opt.enc_num_heads,
                    opt.mlp_ratio,
                    qkv_bias=True,
                    ffn_bias=True,
                    qk_norm=True,
                    init_values=5e-3 * opt.gs_token_std,
                )
                for _ in range(opt.dec_depth)
            ]
        )

    def _encode_to_kv(self, features: torch.Tensor, run_encoder: bool = False):
        if run_encoder:
            raise ValueError("The released inference graph has no trainable RGB encoder stack")
        features = self.encoder_norm(features)
        keys, values = rearrange(
            self.kv_proj(features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        return self.k_proj_norm(keys), values
