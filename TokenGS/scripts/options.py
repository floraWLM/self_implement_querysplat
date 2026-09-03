# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Strict inference-only model configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Options:
    img_size: tuple[int, int] = (512, 512)
    patch_size: int = 8
    dec_depth: int = 12
    enc_embed_dim: int = 2048
    enc_num_heads: int = 16
    mlp_ratio: float = 4.0
    bg_color: str = "grey"
    gaussian_scale_cap: float = 0.075
    gaussian_z_offset: float = 1.0
    gaussian_save_opacity_threshold: float = 0.005
    gaussian_sh_degree: int = 1
    gaussian_sh_rest_scale: float = 0.1
    num_gs_tokens: int = 8192
    num_gaussians_per_token: int = 64
    token_dim: int = 2048
    gs_token_std: float = 0.01
    num_input_views: int = 4
    vggt_checkpoint: str = "checkpoints/vggt_omega_1b_512.pt"
    vggt_input_size: int = 512
    vggt_intermediate_layers: tuple[int, ...] = (4, 11, 17, 23)
    vggt_keep_camera_token: bool = True
    vggt_keep_scene_token: bool = True
    vggt_use_eye_layer_mixer: bool = True
    vggt_original_encoder_patch_size: int = 16
    vggt_original_encoder_post_decoder_depth: int = 6
    vggt_original_encoder_post_use_color_tokens: bool = True
    vggt_original_encoder_post_detach_query: bool = False
    vggt_original_encoder_post_output_params: tuple[str, ...] = ("rgb", "alpha")
    znear: float = 0.025
    zfar: float = 125.0

    def evolve(self, **changes: Any) -> "Options":
        return replace(self, **changes)


_YAML_FIELDS = {
    "img_size",
    "patch_size",
    "dec_depth",
    "enc_embed_dim",
    "enc_num_heads",
    "mlp_ratio",
    "bg_color",
    "gaussian_scale_cap",
    "gaussian_z_offset",
    "gaussian_save_opacity_threshold",
    "gaussian_sh_degree",
    "gaussian_sh_rest_scale",
    "num_gs_tokens",
    "num_gaussians_per_token",
    "token_dim",
    "gs_token_std",
    "num_input_views",
    "vggt_checkpoint",
    "vggt_input_size",
    "vggt_intermediate_layers",
    "vggt_keep_camera_token",
    "vggt_keep_scene_token",
    "vggt_use_eye_layer_mixer",
    "vggt_original_encoder_patch_size",
    "vggt_original_encoder_post_decoder_depth",
    "vggt_original_encoder_post_use_color_tokens",
    "vggt_original_encoder_post_detach_query",
    "vggt_original_encoder_post_output_params",
    "znear",
    "zfar",
}


def load_options_yaml(path: str | Path) -> Options:
    with open(path, "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Inference config must be a YAML mapping")
    unknown = set(payload) - _YAML_FIELDS
    if unknown:
        raise ValueError(f"Unknown inference config fields: {sorted(unknown)}")

    normalized = dict(payload)
    for key in ("img_size", "vggt_intermediate_layers", "vggt_original_encoder_post_output_params"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    options = Options(**normalized)
    if set(options.vggt_original_encoder_post_output_params) != {"rgb", "alpha"}:
        raise ValueError("This checkpoint requires post output parameters [rgb, alpha]")
    if options.img_size != (512, 512) or options.vggt_input_size != 512:
        raise ValueError("This checkpoint requires 512x512 QuerySplat and VGGT-Omega inputs")
    if options.num_gs_tokens != 8192 or options.num_gaussians_per_token != 64:
        raise ValueError("This checkpoint requires 8192 GS tokens and 64 Gaussians per token")
    return options
