# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Gaussian activation head used by the released inference graph."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

from scripts.options import Options


def effective_gaussian_sh_rest_scale(opt: Options, current_step=None) -> float:
    return float(opt.gaussian_sh_rest_scale)


class PartialClipActivationHead(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.num_gaussians_per_token = opt.num_gaussians_per_token
        self.scale_cap = opt.gaussian_scale_cap
        self.scale_shift = 1 - math.log(self.scale_cap)
        self.output_dims = 10
        self.deconv = nn.Linear(
            opt.enc_embed_dim,
            self.output_dims * self.num_gaussians_per_token,
            bias=True,
        )
        trunc_normal_(self.deconv.weight, std=0.002)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, tokens: torch.Tensor, current_step=None) -> torch.Tensor:
        batch = tokens.shape[0]
        values = rearrange(
            self.deconv(tokens),
            "b n (p c) -> b c (n p)",
            p=self.num_gaussians_per_token,
        )
        values = values.reshape(batch, self.output_dims, -1).permute(0, 2, 1).contiguous()
        position, scaling, rotation = values.split([3, 3, 4], dim=-1)
        position = torch.sign(position) * torch.expm1(torch.abs(position))
        scaling = torch.exp(
            (scaling - self.scale_shift).clamp(max=math.log(self.scale_cap))
        )
        rotation = F.normalize(rotation, dim=-1)
        return torch.cat([position, scaling, rotation], dim=-1)
