"""Training-only adapters around the unchanged QuerySplat inference graph."""

from .vggt_input_pass import VGGTInputPassOutput, forward_vggt_input_once
from .self_calibration import (
    CameraAlignmentMetrics,
    CameraOnlyOutput,
    SelfCalibratedVGMOutput,
    Sim3Transform,
    apply_sim3_to_cameras,
    camera_alignment_metrics,
    cam_view_to_c2w,
    c2w_to_cam_view,
    estimate_sim3_from_cameras,
    forward_self_calibrated_vgm,
    forward_vggt_camera_only,
)
from .dual_branch_forward import (
    DualBranchForwardOutput,
    build_dual_branch_latent,
    decode_and_render_from_self_calibration,
    forward_querysplat_training,
)
from .losses import (
    LinearWeightSchedule,
    QuerySplatLossConfig,
    build_lpips_vgg,
    compute_bidirectional_chamfer_loss,
    compute_opacity_floor_loss,
    compute_photometric_losses,
    compute_querysplat_losses,
    compute_visibility_loss,
    sample_vggt_depth_pointcloud,
)

__all__ = [
    "CameraAlignmentMetrics",
    "CameraOnlyOutput",
    "DualBranchForwardOutput",
    "LinearWeightSchedule",
    "QuerySplatLossConfig",
    "SelfCalibratedVGMOutput",
    "Sim3Transform",
    "VGGTInputPassOutput",
    "apply_sim3_to_cameras",
    "build_dual_branch_latent",
    "build_lpips_vgg",
    "camera_alignment_metrics",
    "cam_view_to_c2w",
    "c2w_to_cam_view",
    "decode_and_render_from_self_calibration",
    "compute_bidirectional_chamfer_loss",
    "compute_opacity_floor_loss",
    "compute_photometric_losses",
    "compute_querysplat_losses",
    "compute_visibility_loss",
    "estimate_sim3_from_cameras",
    "forward_self_calibrated_vgm",
    "forward_querysplat_training",
    "forward_vggt_camera_only",
    "forward_vggt_input_once",
    "sample_vggt_depth_pointcloud",
]
