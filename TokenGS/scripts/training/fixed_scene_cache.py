"""Cache frozen VGM products while preserving trainable geometry adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.models.vggt_encoder import VGGTEncoder
from scripts.training.self_calibration import (
    CameraAlignmentMetrics,
    CameraOnlyOutput,
    SelfCalibratedVGMOutput,
    Sim3Transform,
    apply_sim3_to_cameras,
    camera_alignment_metrics,
    estimate_sim3_from_cameras,
    forward_vggt_camera_only,
)
from scripts.training.vggt_input_pass import VGGTInputPassOutput


@dataclass(frozen=True)
class FixedSceneVGMCache:
    input_layers: dict[int, torch.Tensor]
    patch_start: int
    input_cam_view: torch.Tensor
    input_intrinsics: torch.Tensor
    input_pose_enc: torch.Tensor
    input_depth: torch.Tensor
    input_depth_confidence: torch.Tensor
    all_view_cameras: CameraOnlyOutput
    aligned_all_cam_view: torch.Tensor
    sim3_all_to_input: Sim3Transform
    shared_input_metrics: CameraAlignmentMetrics


@torch.no_grad()
def prepare_fixed_scene_vgm_cache(
    encoder: VGGTEncoder,
    input_images: torch.Tensor,
    images_all: torch.Tensor,
    image_hw: tuple[int, int],
) -> FixedSceneVGMCache:
    """Run the two frozen VGM passes once for repeated fixed-scene optimization."""
    if input_images.ndim != 5 or images_all.ndim != 5:
        raise ValueError("input_images and images_all must have shape [B,V,3,H,W]")
    num_input_views = input_images.shape[1]
    if not torch.equal(images_all[:, :num_input_views], input_images):
        raise ValueError("images_all must begin with the exact input views and order")

    output_list, patch_start, prepared = encoder._run_aggregator_with_images(input_images)
    required_layers = set(encoder.opt.vggt_intermediate_layers) | {
        encoder.aggregator.depth - 1
    }
    missing = [index for index in required_layers if output_list[index] is None]
    if missing:
        raise RuntimeError(f"VGGT-Omega did not cache required layers: {sorted(missing)}")
    input_layers = {index: output_list[index].detach() for index in required_layers}
    final_layer = input_layers[encoder.aggregator.depth - 1]
    input_cam_view, input_intrinsics, input_pose_enc = encoder._decode_cameras(
        final_layer, image_hw
    )
    input_depth, input_depth_confidence = encoder._decode_depths(
        output_list, patch_start, prepared, image_hw
    )

    all_view_cameras = forward_vggt_camera_only(encoder, images_all, image_hw)
    transform = estimate_sim3_from_cameras(
        all_view_cameras.cam_view[:, :num_input_views], input_cam_view
    )
    aligned_all_cam_view = apply_sim3_to_cameras(all_view_cameras.cam_view, transform)
    metrics = camera_alignment_metrics(
        aligned_all_cam_view[:, :num_input_views], input_cam_view
    )
    return FixedSceneVGMCache(
        input_layers=input_layers,
        patch_start=patch_start,
        input_cam_view=input_cam_view.detach(),
        input_intrinsics=input_intrinsics.detach(),
        input_pose_enc=input_pose_enc.detach(),
        input_depth=input_depth.detach(),
        input_depth_confidence=input_depth_confidence.detach(),
        all_view_cameras=CameraOnlyOutput(
            cam_view=all_view_cameras.cam_view.detach(),
            intrinsics=all_view_cameras.intrinsics.detach(),
            pose_enc=all_view_cameras.pose_enc.detach(),
        ),
        aligned_all_cam_view=aligned_all_cam_view.detach(),
        sim3_all_to_input=Sim3Transform(
            scale=transform.scale.detach(),
            rotation=transform.rotation.detach(),
            translation=transform.translation.detach(),
        ),
        shared_input_metrics=CameraAlignmentMetrics(
            center_rmse=metrics.center_rmse.detach(),
            rotation_mean_degrees=metrics.rotation_mean_degrees.detach(),
            rotation_max_degrees=metrics.rotation_max_degrees.detach(),
        ),
    )


def materialize_fixed_scene_self_calibration(
    encoder: VGGTEncoder,
    cache: FixedSceneVGMCache,
) -> SelfCalibratedVGMOutput:
    """Reapply trainable layer fusion/norm to cached frozen input features each step."""
    first_layer = next(iter(cache.input_layers.values()))
    geometry = encoder._tokens_from_layers(
        cache.input_layers,
        cache.patch_start,
        batch_size=first_layer.shape[0],
    )
    input_pass = VGGTInputPassOutput(
        geometry=geometry,
        cam_view=cache.input_cam_view,
        intrinsics=cache.input_intrinsics,
        pose_enc=cache.input_pose_enc,
        depth=cache.input_depth,
        depth_confidence=cache.input_depth_confidence,
    )
    return SelfCalibratedVGMOutput(
        input_pass=input_pass,
        all_view_cameras=cache.all_view_cameras,
        aligned_all_cam_view=cache.aligned_all_cam_view,
        sim3_all_to_input=cache.sim3_all_to_input,
        shared_input_metrics=cache.shared_input_metrics,
    )
