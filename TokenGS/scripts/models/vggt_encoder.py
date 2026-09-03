# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Frozen VGGT-Omega feature, camera, and depth encoder for inference."""

from __future__ import annotations

import hashlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scripts.options import Options


_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OMEGA_SOURCE = os.path.join(_PACKAGE_ROOT, "third_party", "vggt_omega")
if _OMEGA_SOURCE not in sys.path:
    sys.path.insert(0, _OMEGA_SOURCE)

from vggt_omega.models.aggregator import (  # noqa: E402
    Aggregator,
    slice_expand_and_flatten,
)
from vggt_omega.models.heads.camera_head import CameraHead  # noqa: E402
from vggt_omega.models.heads.dense_head import DenseHead  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


VGGT_OMEGA_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return torch.amp.autocast(device_type=device.type, enabled=False)


class VGGTEncoderOutput:
    def __init__(
        self,
        tokens: torch.Tensor,
        eye_token: torch.Tensor | None,
        layer_weights: torch.Tensor | None,
        camera_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        tokens_per_view: int,
        special_tokens_per_view: int,
    ):
        self.tokens = tokens
        self.eye_token = eye_token
        self.layer_weights = layer_weights
        self.camera_tokens = camera_tokens
        self.scene_tokens = scene_tokens
        self.tokens_per_view = tokens_per_view
        self.special_tokens_per_view = special_tokens_per_view


class VGGTEncoder(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.vggt_input_size = opt.vggt_input_size
        self.agg_embed_dim = 1024
        self.agg_out_dim = 2048

        needed_layers = tuple(sorted(set(opt.vggt_intermediate_layers) | {23}))
        self.aggregator = Aggregator(
            patch_size=16,
            embed_dim=self.agg_embed_dim,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            cached_layer_indices=needed_layers,
        )
        self.camera_head = CameraHead(dim_in=self.agg_out_dim)
        self.depth_head = DenseHead(
            dim_in=self.agg_out_dim,
            patch_size=16,
            intermediate_layer_idx=list(opt.vggt_intermediate_layers),
        )

        checkpoint = opt.vggt_checkpoint
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(_PACKAGE_ROOT, checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != VGGT_OMEGA_SHA256:
            raise ValueError(
                "VGGT-Omega checkpoint checksum mismatch: "
                f"expected {VGGT_OMEGA_SHA256}, got {actual_sha256}"
            )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.aggregator.load_state_dict(
            {k.removeprefix("aggregator."): v for k, v in state.items() if k.startswith("aggregator.")},
            strict=True,
        )
        self.camera_head.load_state_dict(
            {k.removeprefix("camera_head."): v for k, v in state.items() if k.startswith("camera_head.")},
            strict=True,
        )
        self.depth_head.load_state_dict(
            {k.removeprefix("dense_head."): v for k, v in state.items() if k.startswith("dense_head.")},
            strict=True,
        )
        del state

        for module in (self.aggregator, self.camera_head, self.depth_head):
            module.requires_grad_(False)
            module.eval()

        self.feature_proj = nn.Identity()
        self.eye_token = nn.Parameter(opt.gs_token_std * torch.randn(1, opt.enc_embed_dim))
        self.eye_layer_mlp = nn.Sequential(
            nn.LayerNorm(opt.enc_embed_dim),
            nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim // 2),
            nn.GELU(),
            nn.Linear(opt.enc_embed_dim // 2, len(opt.vggt_intermediate_layers)),
        )
        self.output_norm = nn.LayerNorm(opt.enc_embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()
        self.camera_head.eval()
        self.depth_head.eval()
        return self

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB images, got {channels} channels")
        images = images.reshape(batch * views, channels, height, width)
        if (height, width) != (self.vggt_input_size, self.vggt_input_size):
            images = F.interpolate(
                images,
                size=(self.vggt_input_size, self.vggt_input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return images.reshape(batch, views, 3, self.vggt_input_size, self.vggt_input_size)

    def _run_aggregator_with_images(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor | None], int, torch.Tensor]:
        prepared = self._prepare_images(images)
        with torch.no_grad():
            output_list, patch_start = self.aggregator(prepared)
        return output_list, patch_start, prepared

    def _run_selected_layers(
        self,
        images: torch.Tensor,
        layer_indices: set[int],
    ) -> tuple[dict[int, torch.Tensor], int]:
        prepared = self._prepare_images(images)
        agg = self.aggregator
        batch, views, channels, height, width = prepared.shape
        prepared = (prepared - agg._resnet_mean) / agg._resnet_std
        prepared = prepared.view(batch * views, channels, height, width)

        camera_token = slice_expand_and_flatten(agg.camera_token, batch, views)
        register_token = slice_expand_and_flatten(agg.register_token, batch, views)
        patch_tokens = agg.patch_embed(prepared)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        grid = (height // agg.patch_size, width // agg.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = agg.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        output_by_layer: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for block_idx in range(agg.depth):
                tokens, frame_tokens = agg._run_frame_block(
                    tokens,
                    batch,
                    views,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    frame_rope,
                )
                tokens = agg._run_inter_frame_attention_block(
                    tokens,
                    batch,
                    views,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    agg.inter_frame_attention_types[block_idx],
                )
                if block_idx in layer_indices:
                    output_by_layer[block_idx] = torch.cat([frame_tokens, tokens], dim=-1)
        missing = layer_indices - output_by_layer.keys()
        if missing:
            raise RuntimeError(f"VGGT-Omega did not produce layers: {sorted(missing)}")
        return output_by_layer, agg.patch_token_start

    def _tokens_from_layers(
        self,
        output_by_layer: dict[int, torch.Tensor],
        patch_start: int,
        batch_size: int,
    ) -> VGGTEncoderOutput:
        patch_layers = [
            self.feature_proj(output_by_layer[index][:, :, patch_start:, :])
            for index in self.opt.vggt_intermediate_layers
        ]
        patch_stack = torch.stack(patch_layers, dim=2)
        eye = self.eye_token.expand(batch_size, -1).to(device=patch_stack.device)
        logits = self.eye_layer_mlp(eye.float()).to(patch_stack.dtype)
        layer_weights = torch.softmax(logits, dim=-1)
        patch_tokens = (patch_stack * layer_weights[:, None, :, None, None]).sum(dim=2)

        final_layer = output_by_layer[self.aggregator.depth - 1]
        camera_token = self.feature_proj(final_layer[:, :, 0:1, :])
        scene_tokens = self.feature_proj(final_layer[:, :, 1:patch_start, :])
        tokens = torch.cat([camera_token, scene_tokens, patch_tokens], dim=2)
        tokens_per_view = tokens.shape[2]
        special_tokens_per_view = camera_token.shape[2] + scene_tokens.shape[2]

        camera_token = self.output_norm(camera_token)
        scene_tokens = self.output_norm(scene_tokens)
        tokens = self.output_norm(tokens)
        tokens = rearrange(tokens, "b v p c -> b (v p) c")
        return VGGTEncoderOutput(
            tokens=tokens,
            eye_token=eye.to(dtype=tokens.dtype),
            layer_weights=layer_weights,
            camera_tokens=camera_token,
            scene_tokens=scene_tokens,
            tokens_per_view=tokens_per_view,
            special_tokens_per_view=special_tokens_per_view,
        )

    def _decode_cameras(
        self,
        final_layer: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad(), _autocast_disabled(final_layer.device):
            pose_enc = self.camera_head(
                [final_layer.float()],
                patch_token_start=self.aggregator.patch_token_start,
            )
        extrinsic, intrinsic = encoding_to_camera(pose_enc.float(), image_size_hw=image_hw)
        batch, views = extrinsic.shape[:2]
        w2c = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype).view(1, 1, 4, 4)
        w2c = w2c.repeat(batch, views, 1, 1)
        w2c[:, :, :3, :4] = extrinsic
        cam_view = w2c.transpose(-2, -1)
        intrinsics = torch.stack(
            [intrinsic[..., 0, 0], intrinsic[..., 1, 1], intrinsic[..., 0, 2], intrinsic[..., 1, 2]],
            dim=-1,
        )
        return cam_view, intrinsics, pose_enc

    def _decode_depths(
        self,
        output_list: list[torch.Tensor | None],
        patch_start: int,
        prepared_images: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), _autocast_disabled(prepared_images.device):
            head_inputs = [None if value is None else value.float() for value in output_list]
            depth, confidence = self.depth_head(
                head_inputs,
                images=prepared_images.float(),
                patch_token_start=patch_start,
            )
        batch, views = depth.shape[:2]
        depth = depth.permute(0, 1, 4, 2, 3).contiguous()
        confidence = confidence.unsqueeze(2).contiguous()
        if depth.shape[-2:] != image_hw:
            depth = F.interpolate(
                depth.view(batch * views, 1, *depth.shape[-2:]),
                size=image_hw,
                mode="bilinear",
                align_corners=False,
            ).view(batch, views, 1, *image_hw)
            confidence = F.interpolate(
                confidence.view(batch * views, 1, *confidence.shape[-2:]),
                size=image_hw,
                mode="bilinear",
                align_corners=False,
            ).view(batch, views, 1, *image_hw)
        return depth, confidence

    def forward_with_cameras(
        self,
        images: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[VGGTEncoderOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        indices = set(self.opt.vggt_intermediate_layers) | {self.aggregator.depth - 1}
        output_by_layer, patch_start = self._run_selected_layers(images, indices)
        output = self._tokens_from_layers(output_by_layer, patch_start, batch)
        cam_view, intrinsics, pose_enc = self._decode_cameras(
            output_by_layer[self.aggregator.depth - 1], image_hw
        )
        return output, cam_view, intrinsics, pose_enc

    def predict_cameras_and_depths(
        self,
        images: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_list, patch_start, prepared = self._run_aggregator_with_images(images)
        final_layer = output_list[self.aggregator.depth - 1]
        if final_layer is None:
            raise RuntimeError("VGGT-Omega final layer was not cached")
        cam_view, intrinsics, pose_enc = self._decode_cameras(final_layer, image_hw)
        depth, confidence = self._decode_depths(output_list, patch_start, prepared, image_hw)
        return cam_view, intrinsics, pose_enc, depth, confidence
