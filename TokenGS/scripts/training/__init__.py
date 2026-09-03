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

__all__ = [
    "CameraAlignmentMetrics",
    "CameraOnlyOutput",
    "SelfCalibratedVGMOutput",
    "Sim3Transform",
    "VGGTInputPassOutput",
    "apply_sim3_to_cameras",
    "camera_alignment_metrics",
    "cam_view_to_c2w",
    "c2w_to_cam_view",
    "estimate_sim3_from_cameras",
    "forward_self_calibrated_vgm",
    "forward_vggt_camera_only",
    "forward_vggt_input_once",
]
