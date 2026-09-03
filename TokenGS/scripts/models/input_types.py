# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Tensor containers used by the inference graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelInputEncoder:
    images_rgb: torch.Tensor
    images_rgb_unnormalized: torch.Tensor


@dataclass
class EncoderLatent:
    keys: torch.Tensor
    values: torch.Tensor
    post_rgb_keys: torch.Tensor
    post_rgb_values: torch.Tensor
    eye_token: torch.Tensor | None = None
    layer_weights: torch.Tensor | None = None


@dataclass
class ModelInputDecoder:
    cam_view: torch.Tensor
    intrinsics: torch.Tensor


@dataclass
class ModelInput:
    encoder: ModelInputEncoder
    decoder: ModelInputDecoder

    @property
    def batch_size(self) -> int:
        return self.encoder.images_rgb.shape[0]
