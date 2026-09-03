"""Training-only QuerySplat dual-branch forward built on self-calibrated VGM output."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.models.input_types import EncoderLatent, ModelInputDecoder, ModelInputEncoder
from scripts.models.querysplat import QuerySplat
from scripts.training.self_calibration import (
    SelfCalibratedVGMOutput,
    forward_self_calibrated_vgm,
)


@dataclass(frozen=True)
class DualBranchForwardOutput:
    self_calibration: SelfCalibratedVGMOutput
    latent: EncoderLatent
    gaussians: torch.Tensor
    supervision_decoder: ModelInputDecoder
    supervision_images: torch.Tensor
    render_results: dict[str, torch.Tensor]


def _validate_training_images(
    input_normalized: torch.Tensor,
    input_raw: torch.Tensor,
    images_all: torch.Tensor,
) -> None:
    for name, value in (
        ("input_normalized", input_normalized),
        ("input_raw", input_raw),
        ("images_all", images_all),
    ):
        if value.ndim != 5 or value.shape[2] != 3:
            raise ValueError(f"{name} must have shape [B,V,3,H,W], got {tuple(value.shape)}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite floating-point tensor")
    if input_normalized.shape != input_raw.shape:
        raise ValueError("input_normalized and input_raw must have identical shapes")
    if images_all.shape[0] != input_raw.shape[0] or images_all.shape[1] <= input_raw.shape[1]:
        raise ValueError("images_all must contain input views followed by supervision views")
    if images_all.shape[2:] != input_raw.shape[2:]:
        raise ValueError("input and all-view image dimensions must match")
    if not torch.equal(images_all[:, : input_raw.shape[1]], input_raw):
        raise ValueError("the first images_all views must exactly match input_raw and order")
    if float(input_raw.min()) < 0 or float(input_raw.max()) > 1:
        raise ValueError("input_raw must be in [0,1]")
    if float(images_all.min()) < 0 or float(images_all.max()) > 1:
        raise ValueError("images_all must be in [0,1]")


def build_dual_branch_latent(
    model: QuerySplat,
    input_normalized: torch.Tensor,
    input_raw: torch.Tensor,
    self_calibration: SelfCalibratedVGMOutput,
) -> EncoderLatent:
    """Build geometry and appearance memories without another VGM forward."""
    input_pass = self_calibration.input_pass
    batch, views = input_raw.shape[:2]
    if input_pass.cam_view.shape[:2] != (batch, views):
        raise ValueError("input-only VGM cameras do not match the input image views")
    if input_pass.intrinsics.shape[:2] != (batch, views):
        raise ValueError("input-only VGM intrinsics do not match the input image views")

    geometry = input_pass.geometry
    geometry_keys, geometry_values = model.enc_dec_backbone._encode_to_kv(
        geometry.tokens,
        run_encoder=False,
    )

    encoder_input = ModelInputEncoder(
        images_rgb=input_normalized,
        images_rgb_unnormalized=input_raw,
    )
    plucker = model._plucker_from_vggt_cameras(
        input_pass.cam_view,
        input_pass.intrinsics,
        dtype=input_normalized.dtype,
        device=input_normalized.device,
    )
    appearance_tokens = model._original_encoder_tokens(encoder_input, plucker)
    appearance_keys, appearance_values = model._original_tokens_to_kv(appearance_tokens)

    return EncoderLatent(
        keys=geometry_keys,
        values=geometry_values,
        post_rgb_keys=appearance_keys,
        post_rgb_values=appearance_values,
        eye_token=geometry.eye_token,
        layer_weights=geometry.layer_weights,
    )


def decode_and_render_from_self_calibration(
    model: QuerySplat,
    input_normalized: torch.Tensor,
    input_raw: torch.Tensor,
    images_all: torch.Tensor,
    self_calibration: SelfCalibratedVGMOutput,
) -> DualBranchForwardOutput:
    """Decode Gaussians and render only the held-out supervision views."""
    _validate_training_images(input_normalized, input_raw, images_all)
    num_input_views = input_raw.shape[1]
    if self_calibration.aligned_all_cam_view.shape[:2] != images_all.shape[:2]:
        raise ValueError("aligned all-view cameras do not match images_all")
    if self_calibration.all_view_cameras.intrinsics.shape[:2] != images_all.shape[:2]:
        raise ValueError("all-view VGM intrinsics do not match images_all")

    latent = build_dual_branch_latent(
        model,
        input_normalized=input_normalized,
        input_raw=input_raw,
        self_calibration=self_calibration,
    )
    gaussians = model.forward_decoder(latent)
    supervision_decoder = ModelInputDecoder(
        cam_view=self_calibration.aligned_all_cam_view[:, num_input_views:],
        intrinsics=self_calibration.all_view_cameras.intrinsics[:, num_input_views:],
    )
    supervision_images = images_all[:, num_input_views:]
    render_results = model.render_gaussians(gaussians, supervision_decoder)
    return DualBranchForwardOutput(
        self_calibration=self_calibration,
        latent=latent,
        gaussians=gaussians,
        supervision_decoder=supervision_decoder,
        supervision_images=supervision_images,
        render_results=render_results,
    )


def forward_querysplat_training(
    model: QuerySplat,
    input_normalized: torch.Tensor,
    input_raw: torch.Tensor,
    images_all: torch.Tensor,
) -> DualBranchForwardOutput:
    """Run two VGM passes, dual-branch decoding, and supervision rendering."""
    _validate_training_images(input_normalized, input_raw, images_all)
    self_calibration = forward_self_calibrated_vgm(
        model.vggt_encoder,
        input_images=input_raw,
        images_all=images_all,
        image_hw=tuple(input_raw.shape[-2:]),
    )
    return decode_and_render_from_self_calibration(
        model,
        input_normalized=input_normalized,
        input_raw=input_raw,
        images_all=images_all,
        self_calibration=self_calibration,
    )
