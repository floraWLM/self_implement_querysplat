"""Training adapter for a single input-only VGGT-Omega pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.models.vggt_encoder import VGGTEncoder, VGGTEncoderOutput


@dataclass(frozen=True)
class VGGTInputPassOutput:
    geometry: VGGTEncoderOutput
    cam_view: torch.Tensor
    intrinsics: torch.Tensor
    pose_enc: torch.Tensor
    depth: torch.Tensor
    depth_confidence: torch.Tensor


def forward_vggt_input_once(
    encoder: VGGTEncoder,
    images: torch.Tensor,
    image_hw: tuple[int, int],
) -> VGGTInputPassOutput:
    """Reuse one frozen aggregator result for geometry, camera, and depth.

    The released inference helpers compute geometry/cameras and camera/depth in
    separate calls. Training needs all products every step, so this adapter
    composes the existing private building blocks without changing the
    released model or its state-dict keys.
    """
    output_list, patch_start, prepared = encoder._run_aggregator_with_images(images)
    required_layers = set(encoder.opt.vggt_intermediate_layers) | {
        encoder.aggregator.depth - 1
    }
    missing_layers = [index for index in required_layers if output_list[index] is None]
    if missing_layers:
        raise RuntimeError(
            f"VGGT-Omega did not cache required layers: {sorted(missing_layers)}"
        )
    output_by_layer = {index: output_list[index] for index in required_layers}

    geometry = encoder._tokens_from_layers(
        output_by_layer,
        patch_start,
        batch_size=images.shape[0],
    )
    final_layer = output_by_layer[encoder.aggregator.depth - 1]
    cam_view, intrinsics, pose_enc = encoder._decode_cameras(final_layer, image_hw)
    depth, depth_confidence = encoder._decode_depths(
        output_list,
        patch_start,
        prepared,
        image_hw,
    )
    return VGGTInputPassOutput(
        geometry=geometry,
        cam_view=cam_view,
        intrinsics=intrinsics,
        pose_enc=pose_enc,
        depth=depth,
        depth_confidence=depth_confidence,
    )
