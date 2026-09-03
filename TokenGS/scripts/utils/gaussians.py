# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Gaussians:
    """Structured views over a single tensor of Gaussian parameters."""
    raw: torch.Tensor
    xyz: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    rgb: torch.Tensor

    @staticmethod
    def color_to_rgb(raw: torch.Tensor, sh_degree: int = 0) -> torch.Tensor:
        colors = raw[..., 11:]
        if sh_degree <= 0:
            return colors[..., :3]
        sh_coeffs = colors.reshape(*colors.shape[:-1], -1, 3)
        return (sh_coeffs[..., 0, :] * 0.28209479177387814 + 0.5).clamp(0, 1)

    @classmethod
    def from_raw(cls, raw: torch.Tensor) -> Gaussians:
        assert raw.shape[-1] >= 14, (
            f"Raw gaussians must have at least 14 channels, got {raw.shape[-1]}."
        )
        return cls(
            raw=raw,
            xyz=raw[..., 0:3],
            opacity=raw[..., 3:4],
            scaling=raw[..., 4:7],
            rotation=raw[..., 7:11],
            rgb=raw[..., 11:14],
        )
