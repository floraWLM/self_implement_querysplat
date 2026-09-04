"""QuerySplat training losses and explicit stage-dependent weight schedules."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.rendering.fused_ssim import fused_ssim
from scripts.training.dual_branch_forward import DualBranchForwardOutput
from scripts.training.self_calibration import cam_view_to_c2w


@dataclass(frozen=True)
class LinearWeightSchedule:
    initial: float
    final: float
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("schedule requires 0 <= start_step <= end_step")
        if self.initial < 0 or self.final < 0:
            raise ValueError("loss weights must be non-negative")

    def value(self, step: int) -> float:
        if step < 0:
            raise ValueError("training step must be non-negative")
        if self.end_step == self.start_step:
            return float(self.initial if step < self.start_step else self.final)
        if step <= self.start_step:
            return float(self.initial)
        if step >= self.end_step:
            return float(self.final)
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return float(self.initial + progress * (self.final - self.initial))


@dataclass(frozen=True)
class QuerySplatLossConfig:
    lambda_ssim: float
    lambda_visibility: float
    lpips_schedule: LinearWeightSchedule
    chamfer_schedule: LinearWeightSchedule
    opacity_schedule: LinearWeightSchedule
    opacity_floor: float = 0.1
    epsilon: float = 1e-6
    visibility_distance_threshold: float = 1.0
    max_depth_points: int = 8192
    max_gaussian_chamfer_points: int = 8192
    chamfer_chunk_size: int = 1024
    minimum_depth_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_ssim < 0 or self.lambda_visibility < 0:
            raise ValueError("fixed loss weights must be non-negative")
        if not 0 < self.opacity_floor < 1:
            raise ValueError("opacity_floor must lie in (0,1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if min(
            self.max_depth_points,
            self.max_gaussian_chamfer_points,
            self.chamfer_chunk_size,
        ) <= 0:
            raise ValueError("Chamfer sampling and chunk sizes must be positive")


def build_lpips_vgg(device: torch.device | str) -> nn.Module:
    """Build the frozen perceptual network explicitly before DDP wrapping."""
    from lpips import LPIPS

    model = LPIPS(net="vgg").to(device=device)
    model.requires_grad_(False)
    model.eval()
    return model


def _autocast_disabled(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.amp.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def compute_photometric_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float,
    lambda_lpips: float,
    lpips_model: nn.Module | None,
    ssim_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = fused_ssim,
) -> dict[str, torch.Tensor]:
    if predicted.shape != target.shape or predicted.ndim != 5 or predicted.shape[2] != 3:
        raise ValueError("predicted and target RGB must share shape [B,V,3,H,W]")
    batch, views, channels, height, width = predicted.shape
    predicted_flat = predicted.reshape(batch * views, channels, height, width)
    target_flat = target.reshape(batch * views, channels, height, width)
    loss_l1 = F.l1_loss(predicted, target)
    loss_ssim = (1.0 - ssim_function(predicted_flat, target_flat)) / 2.0

    if lambda_lpips > 0:
        if lpips_model is None:
            raise ValueError("LPIPS schedule is active but lpips_model was not provided")
        with _autocast_disabled(predicted):
            loss_lpips = lpips_model(
                target_flat.float(),
                predicted_flat.float(),
                normalize=True,
            ).mean()
    else:
        loss_lpips = predicted.new_zeros(())
    loss_photo = loss_l1 + lambda_ssim * loss_ssim + lambda_lpips * loss_lpips
    return {
        "loss_photo": loss_photo,
        "loss_l1": loss_l1,
        "loss_ssim": loss_ssim,
        "loss_lpips": loss_lpips,
    }


def compute_visibility_loss(
    gaussian_centers: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    znear: float,
    distance_threshold: float,
    zfar: float | None = None,
) -> torch.Tensor:
    """Penalize centers outside every relevant frustum or behind every camera."""
    if gaussian_centers.ndim != 3 or gaussian_centers.shape[-1] != 3:
        raise ValueError("gaussian_centers must have shape [B,N,3]")
    if cam_view.shape[:2] != intrinsics.shape[:2] or cam_view.shape[0] != gaussian_centers.shape[0]:
        raise ValueError("camera, intrinsics, and Gaussian batches/views do not match")
    if znear <= 0:
        raise ValueError("znear must be positive")
    if zfar is not None and zfar <= znear:
        raise ValueError("zfar must be greater than znear")

    with _autocast_disabled(gaussian_centers):
        centers = gaussian_centers.float()
        world_to_camera = cam_view.float().transpose(-2, -1)
        rotation = world_to_camera[..., :3, :3]
        translation = world_to_camera[..., :3, 3]
        camera_points = torch.einsum("bvij,bnj->bvni", rotation, centers)
        camera_points = camera_points + translation[:, :, None]
        x, y, z = camera_points.unbind(dim=-1)
        safe_z = z.clamp_min(znear)
        fx, fy, cx, cy = intrinsics.float().unbind(dim=-1)
        pixel_x = fx[:, :, None] * x / safe_z + cx[:, :, None]
        pixel_y = fy[:, :, None] * y / safe_z + cy[:, :, None]
        height, width = image_hw
        normalized_x = pixel_x / width * 2.0 - 1.0
        normalized_y = pixel_y / height * 2.0 - 1.0
        outside = F.relu(normalized_x.abs() - 1.0) + F.relu(normalized_y.abs() - 1.0)
        behind = F.relu((znear - z) / znear)
        beyond_far = torch.zeros_like(z) if zfar is None else F.relu((z - zfar) / zfar)
        penalty = outside + behind + beyond_far
        if distance_threshold > 0:
            penalty = penalty.clamp(max=distance_threshold)
        return penalty.min(dim=1).values.mean()


def sample_vggt_depth_pointcloud(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    max_points: int,
    minimum_confidence: float = 0.0,
) -> list[torch.Tensor]:
    """Back-project deterministic input-only VGGT z-depth samples into its frame."""
    if depth.ndim != 5 or depth.shape[2] != 1 or confidence.shape != depth.shape:
        raise ValueError("depth and confidence must share shape [B,V,1,H,W]")
    if cam_view.shape[:2] != depth.shape[:2] or intrinsics.shape[:2] != depth.shape[:2]:
        raise ValueError("depth cameras/intrinsics must match depth batch and views")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    with _autocast_disabled(depth):
        depth = depth[:, :, 0].float()
        confidence = confidence[:, :, 0].float()
        batch, views, height, width = depth.shape
        total = views * height * width
        sample_count = min(max_points, total)
        indices = torch.linspace(0, total - 1, sample_count, device=depth.device).long()
        view_index = torch.div(indices, height * width, rounding_mode="floor")
        pixel_index = indices.remainder(height * width)
        pixel_y = torch.div(pixel_index, width, rounding_mode="floor").float() + 0.5
        pixel_x = pixel_index.remainder(width).float() + 0.5
        sampled_depth = depth.reshape(batch, total)[:, indices]
        sampled_confidence = confidence.reshape(batch, total)[:, indices]

        fx = intrinsics.float()[:, view_index, 0]
        fy = intrinsics.float()[:, view_index, 1]
        cx = intrinsics.float()[:, view_index, 2]
        cy = intrinsics.float()[:, view_index, 3]
        x_camera = (pixel_x[None] - cx) / fx * sampled_depth
        y_camera = (pixel_y[None] - cy) / fy * sampled_depth
        points_camera = torch.stack([x_camera, y_camera, sampled_depth], dim=-1)

        c2w = cam_view_to_c2w(cam_view.float())
        selected_rotation = c2w[:, view_index, :3, :3]
        selected_translation = c2w[:, view_index, :3, 3]
        points_world = torch.einsum("bnij,bnj->bni", selected_rotation, points_camera)
        points_world = points_world + selected_translation
        valid = (
            torch.isfinite(points_world).all(dim=-1)
            & torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & torch.isfinite(sampled_confidence)
            & (sampled_confidence >= minimum_confidence)
        )
        pointclouds = [points_world[index, valid[index]] for index in range(batch)]
    if any(points.shape[0] == 0 for points in pointclouds):
        raise ValueError("VGGT depth sampling produced an empty pseudo point cloud")
    return pointclouds


def _deterministic_subsample(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    indices = torch.linspace(0, points.shape[0] - 1, max_points, device=points.device).long()
    return points[indices]


def _mean_nearest_squared_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    values = []
    target_squared = target.square().sum(dim=-1).unsqueeze(0)
    for start in range(0, source.shape[0], chunk_size):
        source_chunk = source[start : start + chunk_size]
        distances = (
            source_chunk.square().sum(dim=-1, keepdim=True)
            + target_squared
            - 2.0 * (source_chunk @ target.transpose(0, 1))
        ).clamp_min(0)
        values.append(distances.min(dim=-1).values)
    return torch.cat(values).mean()


def compute_bidirectional_chamfer_loss(
    gaussian_centers: torch.Tensor,
    pseudo_pointclouds: list[torch.Tensor],
    max_gaussian_points: int,
    chunk_size: int,
) -> torch.Tensor:
    if len(pseudo_pointclouds) != gaussian_centers.shape[0]:
        raise ValueError("one pseudo point cloud is required per Gaussian batch item")
    losses = []
    with _autocast_disabled(gaussian_centers):
        for centers, points in zip(gaussian_centers.float(), pseudo_pointclouds):
            centers = _deterministic_subsample(centers, max_gaussian_points)
            points = points.float()
            if centers.shape[0] == 0 or points.shape[0] == 0:
                raise ValueError("Chamfer inputs must be non-empty")
            center_to_depth = _mean_nearest_squared_distance(centers, points, chunk_size)
            depth_to_center = _mean_nearest_squared_distance(points, centers, chunk_size)
            losses.append(center_to_depth + depth_to_center)
    return torch.stack(losses).mean()


def compute_opacity_floor_loss(
    opacity: torch.Tensor,
    opacity_floor: float,
    epsilon: float,
) -> torch.Tensor:
    with _autocast_disabled(opacity):
        opacity = opacity.float()
        floor = opacity.new_tensor(opacity_floor).log()
        return F.relu(floor - opacity.clamp_min(epsilon).log()).mean()


def compute_querysplat_losses(
    output: DualBranchForwardOutput,
    step: int,
    config: QuerySplatLossConfig,
    znear: float,
    zfar: float | None = None,
    lpips_model: nn.Module | None = None,
    ssim_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = fused_ssim,
) -> dict[str, torch.Tensor]:
    """Compute the full scheduled QuerySplat objective for one training step."""
    weights = {
        "weight_lpips": config.lpips_schedule.value(step),
        "weight_chamfer": config.chamfer_schedule.value(step),
        "weight_opacity": config.opacity_schedule.value(step),
    }
    photometric = compute_photometric_losses(
        output.render_results["images_pred"],
        output.supervision_images,
        lambda_ssim=config.lambda_ssim,
        lambda_lpips=weights["weight_lpips"],
        lpips_model=lpips_model,
        ssim_function=ssim_function,
    )

    input_pass = output.self_calibration.input_pass
    relevant_cam_view = torch.cat(
        [input_pass.cam_view, output.supervision_decoder.cam_view], dim=1
    )
    relevant_intrinsics = torch.cat(
        [input_pass.intrinsics, output.supervision_decoder.intrinsics], dim=1
    )
    gaussian_centers = output.gaussians[..., :3]
    loss_visibility = compute_visibility_loss(
        gaussian_centers,
        relevant_cam_view,
        relevant_intrinsics,
        image_hw=tuple(output.supervision_images.shape[-2:]),
        znear=znear,
        distance_threshold=config.visibility_distance_threshold,
        zfar=zfar,
    )

    if weights["weight_chamfer"] > 0:
        pseudo_pointclouds = sample_vggt_depth_pointcloud(
            input_pass.depth,
            input_pass.depth_confidence,
            input_pass.cam_view,
            input_pass.intrinsics,
            max_points=config.max_depth_points,
            minimum_confidence=config.minimum_depth_confidence,
        )
        loss_chamfer = compute_bidirectional_chamfer_loss(
            gaussian_centers,
            pseudo_pointclouds,
            max_gaussian_points=config.max_gaussian_chamfer_points,
            chunk_size=config.chamfer_chunk_size,
        )
    else:
        loss_chamfer = gaussian_centers.new_zeros(())
    loss_opacity = compute_opacity_floor_loss(
        output.gaussians[..., 3],
        opacity_floor=config.opacity_floor,
        epsilon=config.epsilon,
    )

    total = (
        photometric["loss_photo"]
        + config.lambda_visibility * loss_visibility
        + weights["weight_chamfer"] * loss_chamfer
        + weights["weight_opacity"] * loss_opacity
    )
    weight_tensors = {name: total.new_tensor(value) for name, value in weights.items()}
    return {
        "loss": total,
        **photometric,
        "loss_visibility": loss_visibility,
        "loss_chamfer": loss_chamfer,
        "loss_opacity_floor": loss_opacity,
        **weight_tensors,
    }
