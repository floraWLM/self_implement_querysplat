"""Two-pass VGM self-calibration in the input-only coordinate frame."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from scripts.models.vggt_encoder import VGGTEncoder
from scripts.training.vggt_input_pass import VGGTInputPassOutput, forward_vggt_input_once


@dataclass(frozen=True)
class CameraOnlyOutput:
    """Camera products from the isolated all-view VGM pass."""

    cam_view: torch.Tensor
    intrinsics: torch.Tensor
    pose_enc: torch.Tensor


@dataclass(frozen=True)
class Sim3Transform:
    """Similarity mapping ``x_target = scale * rotation @ x_source + translation``."""

    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor


@dataclass(frozen=True)
class CameraAlignmentMetrics:
    center_rmse: torch.Tensor
    rotation_mean_degrees: torch.Tensor
    rotation_max_degrees: torch.Tensor


@dataclass(frozen=True)
class SelfCalibratedVGMOutput:
    """Geometry from input views plus all-view cameras aligned to its frame."""

    input_pass: VGGTInputPassOutput
    all_view_cameras: CameraOnlyOutput
    aligned_all_cam_view: torch.Tensor
    sim3_all_to_input: Sim3Transform
    shared_input_metrics: CameraAlignmentMetrics


def _validate_cam_view(cam_view: torch.Tensor, name: str) -> None:
    if cam_view.ndim != 4 or cam_view.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape [B,V,4,4], got {tuple(cam_view.shape)}")
    if not cam_view.is_floating_point() or not torch.isfinite(cam_view).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def _autocast_disabled(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.amp.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def cam_view_to_c2w(cam_view: torch.Tensor) -> torch.Tensor:
    """Convert QuerySplat's transposed world-to-camera matrices to c2w."""
    _validate_cam_view(cam_view, "cam_view")
    working = cam_view.float() if cam_view.dtype in {torch.float16, torch.bfloat16} else cam_view
    with _autocast_disabled(working):
        return torch.linalg.inv(working.transpose(-2, -1))


def c2w_to_cam_view(c2w: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world matrices to QuerySplat renderer convention."""
    _validate_cam_view(c2w, "c2w")
    working = c2w.float() if c2w.dtype in {torch.float16, torch.bfloat16} else c2w
    with _autocast_disabled(working):
        return torch.linalg.inv(working).transpose(-2, -1)


def estimate_sim3_from_cameras(
    source_cam_view: torch.Tensor,
    target_cam_view: torch.Tensor,
    minimum_center_variance: float = 1e-8,
) -> Sim3Transform:
    """Fit the all-view-frame -> input-only-frame Sim(3) from shared cameras.

    Camera orientations determine the global rotation, which remains observable
    for nearly collinear camera paths. Camera centers then determine scale and
    translation by least squares.
    """
    _validate_cam_view(source_cam_view, "source_cam_view")
    _validate_cam_view(target_cam_view, "target_cam_view")
    if source_cam_view.shape != target_cam_view.shape:
        raise ValueError(
            "shared source and target cameras must have identical shapes, got "
            f"{tuple(source_cam_view.shape)} and {tuple(target_cam_view.shape)}"
        )
    if source_cam_view.shape[1] < 2:
        raise ValueError("at least two shared input cameras are required to estimate scale")

    calculation_dtype = torch.float64 if source_cam_view.dtype == torch.float64 else torch.float32
    with _autocast_disabled(source_cam_view):
        source_c2w = cam_view_to_c2w(source_cam_view.to(calculation_dtype))
        target_c2w = cam_view_to_c2w(target_cam_view.to(calculation_dtype))
        source_rotations = source_c2w[..., :3, :3]
        target_rotations = target_c2w[..., :3, :3]

        rotation_candidates = target_rotations @ source_rotations.transpose(-2, -1)
        rotation_sum = rotation_candidates.sum(dim=1)
        u, _, vh = torch.linalg.svd(rotation_sum)
        correction = torch.ones(
            source_cam_view.shape[0], 3, device=rotation_sum.device, dtype=rotation_sum.dtype
        )
        correction[:, -1] = torch.det(u @ vh)
        rotation = u @ torch.diag_embed(correction) @ vh

        source_centers = source_c2w[..., :3, 3]
        target_centers = target_c2w[..., :3, 3]
        source_mean = source_centers.mean(dim=1)
        target_mean = target_centers.mean(dim=1)
        source_centered = source_centers - source_mean[:, None]
        target_centered = target_centers - target_mean[:, None]
        rotated_source = torch.einsum("bvi,bji->bvj", source_centered, rotation)

        denominator = source_centered.square().sum(dim=(-2, -1))
        if bool((denominator <= minimum_center_variance).any()):
            raise ValueError("shared source camera centers have insufficient baseline for scale")
        numerator = (rotated_source * target_centered).sum(dim=(-2, -1))
        scale = numerator / denominator
        if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("estimated Sim(3) scale must be finite and positive")

        rotated_source_mean = torch.einsum("bi,bji->bj", source_mean, rotation)
        translation = target_mean - scale[:, None] * rotated_source_mean
    return Sim3Transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )


def apply_sim3_to_cameras(
    source_cam_view: torch.Tensor,
    transform: Sim3Transform,
) -> torch.Tensor:
    """Move cameras into the target frame while preserving camera intrinsics."""
    source_c2w = cam_view_to_c2w(source_cam_view)
    batch = source_cam_view.shape[0]
    if transform.scale.shape != (batch,):
        raise ValueError(f"scale must have shape [{batch}], got {tuple(transform.scale.shape)}")
    if transform.rotation.shape != (batch, 3, 3):
        raise ValueError(
            f"rotation must have shape [{batch},3,3], got {tuple(transform.rotation.shape)}"
        )
    if transform.translation.shape != (batch, 3):
        raise ValueError(
            f"translation must have shape [{batch},3], got {tuple(transform.translation.shape)}"
        )

    with _autocast_disabled(source_c2w):
        scale = transform.scale.to(device=source_c2w.device, dtype=source_c2w.dtype)
        rotation = transform.rotation.to(device=source_c2w.device, dtype=source_c2w.dtype)
        translation = transform.translation.to(device=source_c2w.device, dtype=source_c2w.dtype)
        aligned_c2w = torch.eye(
            4, device=source_c2w.device, dtype=source_c2w.dtype
        ).view(1, 1, 4, 4).repeat(*source_c2w.shape[:2], 1, 1)
        aligned_c2w[..., :3, :3] = rotation[:, None] @ source_c2w[..., :3, :3]
        aligned_c2w[..., :3, 3] = (
            scale[:, None, None]
            * torch.einsum("bvi,bji->bvj", source_c2w[..., :3, 3], rotation)
            + translation[:, None]
        )
    return c2w_to_cam_view(aligned_c2w)


def camera_alignment_metrics(
    aligned_cam_view: torch.Tensor,
    target_cam_view: torch.Tensor,
) -> CameraAlignmentMetrics:
    """Measure shared-camera center and orientation residuals after alignment."""
    if aligned_cam_view.shape != target_cam_view.shape:
        raise ValueError("aligned and target cameras must have identical shapes")
    with _autocast_disabled(aligned_cam_view):
        aligned_c2w = cam_view_to_c2w(aligned_cam_view.float())
        target_c2w = cam_view_to_c2w(target_cam_view.float())
        center_error = aligned_c2w[..., :3, 3] - target_c2w[..., :3, 3]
        center_rmse = center_error.square().sum(dim=-1).mean(dim=-1).sqrt()

        relative_rotation = (
            target_c2w[..., :3, :3] @ aligned_c2w[..., :3, :3].transpose(-2, -1)
        )
        trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))
    return CameraAlignmentMetrics(
        center_rmse=center_rmse,
        rotation_mean_degrees=angles.mean(dim=-1),
        rotation_max_degrees=angles.max(dim=-1).values,
    )


def forward_vggt_camera_only(
    encoder: VGGTEncoder,
    images_all: torch.Tensor,
    image_hw: tuple[int, int],
) -> CameraOnlyOutput:
    """Run the isolated all-view VGM pass and expose camera products only."""
    final_index = encoder.aggregator.depth - 1
    output_by_layer, _ = encoder._run_selected_layers(images_all, {final_index})
    cam_view, intrinsics, pose_enc = encoder._decode_cameras(
        output_by_layer[final_index], image_hw
    )
    return CameraOnlyOutput(
        cam_view=cam_view,
        intrinsics=intrinsics,
        pose_enc=pose_enc,
    )


def forward_self_calibrated_vgm(
    encoder: VGGTEncoder,
    input_images: torch.Tensor,
    images_all: torch.Tensor,
    image_hw: tuple[int, int],
) -> SelfCalibratedVGMOutput:
    """Run isolated input/all-view passes and align all cameras to input frame."""
    if input_images.ndim != 5 or images_all.ndim != 5:
        raise ValueError("input_images and images_all must have shape [B,V,3,H,W]")
    num_input_views = input_images.shape[1]
    if images_all.shape[0] != input_images.shape[0] or images_all.shape[1] < num_input_views:
        raise ValueError("images_all must contain the input views for every batch item")
    if not torch.equal(images_all[:, :num_input_views], input_images):
        raise ValueError("the first images_all views must exactly match input_images and order")

    input_pass = forward_vggt_input_once(encoder, input_images, image_hw)
    all_view_cameras = forward_vggt_camera_only(encoder, images_all, image_hw)
    transform = estimate_sim3_from_cameras(
        source_cam_view=all_view_cameras.cam_view[:, :num_input_views],
        target_cam_view=input_pass.cam_view,
    )
    aligned_all_cam_view = apply_sim3_to_cameras(all_view_cameras.cam_view, transform)
    metrics = camera_alignment_metrics(
        aligned_all_cam_view[:, :num_input_views], input_pass.cam_view
    )
    return SelfCalibratedVGMOutput(
        input_pass=input_pass,
        all_view_cameras=all_view_cameras,
        aligned_all_cam_view=aligned_all_cam_view,
        sim3_all_to_input=transform,
        shared_input_metrics=metrics,
    )
