# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Output writers used by custom-image inference."""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.options import Options
from scripts.utils.gaussians import Gaussians

def opacity_threshold_suffix(threshold: float) -> str:
    """Return a concise, filesystem-safe threshold suffix for multi-PLY export."""
    return f"_opacity{threshold:g}"

def _intrinsics_4_to_k(intrinsics: torch.Tensor) -> torch.Tensor:
    Ks = torch.zeros(
        (*intrinsics.shape[:-1], 3, 3),
        dtype=intrinsics.dtype,
        device=intrinsics.device,
    )
    Ks[..., 0, 0] = intrinsics[..., 0]
    Ks[..., 1, 1] = intrinsics[..., 1]
    Ks[..., 0, 2] = intrinsics[..., 2]
    Ks[..., 1, 2] = intrinsics[..., 3]
    Ks[..., 2, 2] = 1
    return Ks

def preprocessed_input_frame_path(view_index: int, source: str) -> str:
    src_stem = os.path.splitext(os.path.basename(source))[0]
    filename = f"view{view_index}_{src_stem}_preprocessed.png"
    return os.path.join("input_frames", filename)

def save_predicted_input_cameras(
    predicted_input_cameras: dict[str, torch.Tensor],
    input_frame_sources: list[tuple[str, str]],
    output_dir: str,
):
    """Save VGGT-predicted input-view cameras in readable and tensor formats."""
    cam_view = predicted_input_cameras["cam_view"][0].detach().cpu().float()
    intrinsics = predicted_input_cameras["intrinsics"][0].detach().cpu().float()
    pose_enc = predicted_input_cameras["pose_enc"][0].detach().cpu().float()
    w2c = cam_view.transpose(-2, -1)
    c2w = torch.linalg.inv(w2c)
    K = _intrinsics_4_to_k(intrinsics)

    frame_names = [name for name, _ in input_frame_sources]
    frame_sources = [
        preprocessed_input_frame_path(i, src) for i, (_, src) in enumerate(input_frame_sources)
    ]
    if cam_view.shape[0] != len(frame_names):
        raise ValueError(
            f"Predicted {cam_view.shape[0]} input cameras, but got "
            f"{len(frame_names)} input frame sources."
        )
    payload = {
        "convention": {
            "cam_view": "QuerySplat renderer convention: transposed OpenCV world-to-camera matrix, shape [V,4,4].",
            "w2c": "OpenCV world-to-camera matrix, shape [V,4,4].",
            "c2w": "Inverse of w2c, shape [V,4,4].",
            "intrinsics": "[fx, fy, cx, cy] in rendered/preprocessed image pixels, shape [V,4].",
            "K": "3x3 pinhole intrinsics matrix in rendered/preprocessed image pixels, shape [V,3,3].",
            "pose_enc": "Raw VGGT camera-head output, shape [V,9].",
        },
        "frames": [
            {
                "index": i,
                "name": frame_names[i],
                "source": frame_sources[i],
                "cam_view": cam_view[i].tolist(),
                "w2c": w2c[i].tolist(),
                "c2w": c2w[i].tolist(),
                "intrinsics": intrinsics[i].tolist(),
                "K": K[i].tolist(),
                "pose_enc": pose_enc[i].tolist(),
            }
            for i in range(len(frame_names))
        ],
    }

    json_path = os.path.join(output_dir, "predicted_input_cameras.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    np.savez(
        os.path.join(output_dir, "predicted_input_cameras.npz"),
        cam_view=cam_view.numpy(),
        w2c=w2c.numpy(),
        c2w=c2w.numpy(),
        intrinsics=intrinsics.numpy(),
        K=K.numpy(),
        pose_enc=pose_enc.numpy(),
        frame_names=np.asarray(frame_names),
        frame_sources=np.asarray(frame_sources),
    )

def save_colored_pointcloud(
    gaussians: torch.Tensor,
    path: str,
    opacity_threshold: float,
    opt: Options | None = None,
):
    """Save Gaussian centers as a simple colored PLY (vertex-only, no GS attributes)."""
    from plyfile import PlyData, PlyElement

    g = gaussians[0].detach().cpu()
    xyz = g[:, 0:3].numpy()
    # gaussians[..., 3] is already activated opacity in [0, 1].
    opacity = g[:, 3].numpy()
    rgb_tensor = Gaussians.color_to_rgb(
        g,
        sh_degree=int(getattr(opt, "gaussian_sh_degree", 0)) if opt is not None else 0,
    )
    rgb = rgb_tensor.numpy()

    mask = opacity >= opacity_threshold
    xyz = xyz[mask]
    rgb_u8 = (rgb[mask].clip(0, 1) * 255).astype(np.uint8)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    verts = np.empty(len(xyz), dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = (
        rgb_u8[:, 0],
        rgb_u8[:, 1],
        rgb_u8[:, 2],
    )
    PlyData([PlyElement.describe(verts, "vertex")]).write(path)

def sample_depth_image_grid(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    target_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Select the highest-confidence pixel in each stride-sized image cell."""
    batch, views, _, height, width = depth.shape
    stride = max(1, int((max(views * height * width, 1) / target_points) ** 0.5))
    device, dtype = depth.device, depth.dtype
    if stride == 1:
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        x_grid = x_grid.view(1, 1, 1, height, width).expand_as(depth).contiguous()
        y_grid = y_grid.view(1, 1, 1, height, width).expand_as(depth).contiguous()
        return depth, confidence, x_grid, y_grid, stride

    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    depth_padded = depth
    if pad_h:
        depth_padded = torch.cat(
            [depth_padded, depth_padded[..., -1:, :].expand(*depth_padded.shape[:-2], pad_h, width)],
            dim=-2,
        )
    if pad_w:
        depth_padded = torch.cat(
            [depth_padded, depth_padded[..., :, -1:].expand(*depth_padded.shape[:-1], pad_w)],
            dim=-1,
        )
    confidence_padded = F.pad(confidence, (0, pad_w, 0, pad_h), value=float("-inf"))
    padded_h, padded_w = depth_padded.shape[-2:]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(padded_h, device=device, dtype=dtype).clamp_max(height - 1),
        torch.arange(padded_w, device=device, dtype=dtype).clamp_max(width - 1),
        indexing="ij",
    )
    x_grid = x_grid.view(1, 1, 1, padded_h, padded_w).expand(batch, views, 1, padded_h, padded_w)
    y_grid = y_grid.view(1, 1, 1, padded_h, padded_w).expand(batch, views, 1, padded_h, padded_w)

    def cells(value: torch.Tensor) -> torch.Tensor:
        value = value.view(batch, views, 1, padded_h // stride, stride, padded_w // stride, stride)
        value = value.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        return value.view(batch, views, 1, padded_h // stride, padded_w // stride, stride * stride)

    depth_cells = cells(depth_padded)
    confidence_cells = cells(confidence_padded)
    x_cells = cells(x_grid)
    y_cells = cells(y_grid)
    best = confidence_cells.argmax(dim=-1, keepdim=True)
    gather = lambda value: torch.gather(value, dim=-1, index=best).squeeze(-1).contiguous()
    return gather(depth_cells), gather(confidence_cells), gather(x_cells), gather(y_cells), stride


def world_points_from_sampled_depth(
    depth: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    batch, views = depth.shape[:2]
    dtype, device = depth.dtype, depth.device
    z = depth[:, :, 0]
    x = pixel_x[:, :, 0].to(device=device, dtype=dtype)
    y = pixel_y[:, :, 0].to(device=device, dtype=dtype)
    fx = intrinsics[..., 0].view(batch, views, 1, 1).to(device=device, dtype=dtype)
    fy = intrinsics[..., 1].view(batch, views, 1, 1).to(device=device, dtype=dtype)
    cx = intrinsics[..., 2].view(batch, views, 1, 1).to(device=device, dtype=dtype)
    cy = intrinsics[..., 3].view(batch, views, 1, 1).to(device=device, dtype=dtype)
    camera_points = torch.stack(
        [(x - cx) / fx.clamp_min(1e-6) * z, (y - cy) / fy.clamp_min(1e-6) * z, z],
        dim=-1,
    )
    c2w = torch.linalg.inv(cam_view.transpose(-2, -1).float()).to(dtype=dtype)
    translation = c2w[..., :3, 3]
    return (
        torch.einsum("bvhwj,bvij->bvhwi", camera_points, c2w[..., :3, :3])
        + translation[:, :, None, None, :]
    )


def save_vggt_depth_pointcloud(
    prediction: dict[str, torch.Tensor],
    input_images: torch.Tensor,
    input_sources: list[tuple[str, str]],
    output_dir: str,
    target_points: int,
) -> tuple[str, str, int]:
    from plyfile import PlyData, PlyElement

    depth = prediction["depth"].detach()
    confidence = prediction["depth_conf"].detach()
    cam_view = prediction["cam_view"].detach()
    intrinsics = prediction["intrinsics"].detach()
    original_shape = list(depth.shape)
    rgb = input_images.detach().to(device=depth.device, dtype=torch.float32)
    if rgb.shape[-2:] != depth.shape[-2:]:
        batch, views = rgb.shape[:2]
        rgb = F.interpolate(
            rgb.view(batch * views, 3, *rgb.shape[-2:]),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view(batch, views, 3, *depth.shape[-2:])

    depth, confidence, pixel_x, pixel_y, stride = sample_depth_image_grid(
        depth, confidence, target_points
    )
    world_points = world_points_from_sampled_depth(depth, pixel_x, pixel_y, cam_view, intrinsics)
    batch, views, _, full_height, full_width = original_shape
    rgb_flat = rgb.permute(0, 1, 3, 4, 2).contiguous().view(
        batch, views, full_height * full_width, 3
    )
    flat_index = (
        pixel_y[:, :, 0].long().clamp(0, full_height - 1) * full_width
        + pixel_x[:, :, 0].long().clamp(0, full_width - 1)
    )
    sampled_rgb = torch.gather(
        rgb_flat,
        dim=2,
        index=flat_index.view(batch, views, -1, 1).expand(-1, -1, -1, 3),
    ).view(batch, views, *depth.shape[-2:], 3)

    valid = (
        torch.isfinite(world_points).all(dim=-1)
        & torch.isfinite(depth[:, :, 0])
        & (depth[:, :, 0] > 0)
        & torch.isfinite(confidence[:, :, 0])
    )
    sampled_height, sampled_width = depth.shape[-2:]
    view_index = torch.arange(views, device=depth.device).view(1, views, 1, 1)
    view_index = view_index.expand(batch, views, sampled_height, sampled_width)
    xyz = world_points[valid].float().cpu().numpy()
    rgb_u8 = (sampled_rgb[valid].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    point_views = view_index[valid].int().cpu().numpy()

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("view_index", "i4"),
    ]
    vertices = np.empty(len(xyz), dtype=dtype)
    if len(xyz):
        vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertices["red"], vertices["green"], vertices["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
        vertices["view_index"] = point_views
    ply_path = os.path.join(output_dir, "vggt_depth_pointcloud.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)

    camera_cpu = cam_view[0].cpu().float()
    intrinsics_cpu = intrinsics[0].cpu().float()
    w2c = camera_cpu.transpose(-2, -1)
    c2w = torch.linalg.inv(w2c)
    counts = valid[0].sum(dim=(-1, -2)).cpu().tolist()
    metadata_path = os.path.join(output_dir, "vggt_depth_pointcloud.json")
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "pointcloud": os.path.basename(ply_path),
                "sampling": "image_grid",
                "target_points_per_scene": target_points,
                "sampling_stride": stride,
                "total_points": len(xyz),
                "mean_nearest_neighbor_distance": mean_pointcloud_nn_distance_np(xyz),
                "original_depth_shape": original_shape,
                "depth_shape": list(depth.shape),
                "frames": [
                    {
                        "index": index,
                        "name": name,
                        "source": preprocessed_input_frame_path(index, source),
                        "valid_points": int(counts[index]),
                        "cam_view": camera_cpu[index].tolist(),
                        "w2c": w2c[index].tolist(),
                        "c2w": c2w[index].tolist(),
                        "intrinsics": intrinsics_cpu[index].tolist(),
                    }
                    for index, (name, source) in enumerate(input_sources)
                ],
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return ply_path, metadata_path, len(xyz)

def mean_pointcloud_nn_distance_np(
    points: np.ndarray,
    chunk_size: int = 8192,
) -> float | None:
    """Mean nearest-neighbor distance for a saved point cloud."""
    if points.shape[0] < 2:
        return None
    points_f32 = points.astype(np.float32, copy=False)
    try:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(points_f32).query(points_f32, k=2, workers=-1)
        return float(dists[:, 1].mean())
    except ImportError:
        pass

    nearest = []
    for start in range(0, points_f32.shape[0], chunk_size):
        end = min(start + chunk_size, points_f32.shape[0])
        chunk = points_f32[start:end]
        diff = chunk[:, None, :] - points_f32[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        dists[rows, cols] = np.inf
        nearest.append(dists.min(axis=1))
    return float(np.concatenate(nearest, axis=0).mean())

def _depth_visualization_range(values: np.ndarray, positive_only: bool) -> tuple[float, float]:
    valid = np.isfinite(values)
    if positive_only:
        valid &= values > 0
    samples = values[valid]
    if samples.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(samples, (1.0, 99.0)).astype(np.float64)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(samples.min())
        high = float(samples.max())
    if high <= low:
        high = low + max(abs(low) * 1e-6, 1e-6)
    return float(low), float(high)

def _save_scalar_visualization(
    values: np.ndarray,
    path: str,
    positive_only: bool,
) -> tuple[float, float]:
    low, high = _depth_visualization_range(values, positive_only)
    valid = np.isfinite(values)
    if positive_only:
        valid &= values > 0
    image = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        scaled = np.clip((values[valid] - low) / (high - low), 0.0, 1.0)
        image[valid] = np.rint(scaled * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)
    return low, high

def save_vggt_input_depths(
    vggt_prediction: dict[str, torch.Tensor],
    input_frame_sources: list[tuple[str, str]],
    output_dir: str,
) -> tuple[str, str, list[str], list[str]]:
    """Save raw VGGT input depths plus readable depth/confidence visualizations."""
    required = {"depth", "depth_conf", "cam_view", "intrinsics", "pose_enc"}
    missing = required.difference(vggt_prediction)
    if missing:
        raise ValueError(f"VGGT depth export is missing prediction fields: {sorted(missing)}")

    depth = vggt_prediction["depth"].detach().float().cpu()
    depth_conf = vggt_prediction["depth_conf"].detach().float().cpu()
    cam_view = vggt_prediction["cam_view"].detach().float().cpu()
    intrinsics = vggt_prediction["intrinsics"].detach().float().cpu()
    pose_enc = vggt_prediction["pose_enc"].detach().float().cpu()
    if depth.shape[0] != 1 or depth_conf.shape[0] != 1:
        raise ValueError("VGGT input depth export supports batch size 1 only.")
    if depth.shape != depth_conf.shape or depth.shape[1] != len(input_frame_sources):
        raise ValueError(
            "VGGT depth export shape does not match the selected input views: "
            f"depth={tuple(depth.shape)}, depth_conf={tuple(depth_conf.shape)}, "
            f"frames={len(input_frame_sources)}"
        )

    depth_dir = os.path.join(output_dir, "vggt_input_depths")
    os.makedirs(depth_dir, exist_ok=True)
    depth_np = depth[0, :, 0].numpy()
    confidence_np = depth_conf[0, :, 0].numpy()
    raw_path = os.path.join(depth_dir, "vggt_input_depths.npz")
    np.savez_compressed(
        raw_path,
        depth=depth_np,
        depth_confidence=confidence_np,
        cam_view=cam_view[0].numpy(),
        intrinsics=intrinsics[0].numpy(),
        pose_enc=pose_enc[0].numpy(),
    )

    depth_paths: list[str] = []
    confidence_paths: list[str] = []
    frame_metadata = []
    cam_view_cpu = cam_view[0]
    intrinsics_cpu = intrinsics[0]
    pose_enc_cpu = pose_enc[0]
    w2c = cam_view_cpu.transpose(-2, -1)
    c2w = torch.linalg.inv(w2c)
    for index, (name, source) in enumerate(input_frame_sources):
        depth_filename = f"depth_view{index:02d}.png"
        confidence_filename = f"depth_confidence_view{index:02d}.png"
        depth_path = os.path.join(depth_dir, depth_filename)
        confidence_path = os.path.join(depth_dir, confidence_filename)
        depth_low, depth_high = _save_scalar_visualization(
            depth_np[index],
            depth_path,
            positive_only=True,
        )
        confidence_low, confidence_high = _save_scalar_visualization(
            confidence_np[index],
            confidence_path,
            positive_only=False,
        )
        depth_paths.append(depth_path)
        confidence_paths.append(confidence_path)
        finite_depth = depth_np[index][np.isfinite(depth_np[index]) & (depth_np[index] > 0)]
        finite_confidence = confidence_np[index][np.isfinite(confidence_np[index])]
        frame_metadata.append(
            {
                "index": index,
                "name": name,
                "source": preprocessed_input_frame_path(index, source),
                "depth_image": depth_filename,
                "depth_confidence_image": confidence_filename,
                "depth_visualization_range": [depth_low, depth_high],
                "depth_confidence_visualization_range": [confidence_low, confidence_high],
                "positive_finite_depth_pixels": int(finite_depth.size),
                "depth_min": None if finite_depth.size == 0 else float(finite_depth.min()),
                "depth_max": None if finite_depth.size == 0 else float(finite_depth.max()),
                "confidence_min": (
                    None if finite_confidence.size == 0 else float(finite_confidence.min())
                ),
                "confidence_max": (
                    None if finite_confidence.size == 0 else float(finite_confidence.max())
                ),
                "cam_view": cam_view_cpu[index].tolist(),
                "w2c": w2c[index].tolist(),
                "c2w": c2w[index].tolist(),
                "intrinsics": intrinsics_cpu[index].tolist(),
                "pose_enc": pose_enc_cpu[index].tolist(),
            }
        )

    metadata_path = os.path.join(depth_dir, "vggt_input_depths.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_tensors": os.path.basename(raw_path),
                "depth_shape": list(depth_np.shape),
                "depth_confidence_shape": list(confidence_np.shape),
                "convention": {
                    "depth": "VGGT input-view depth in the VGGT-predicted camera coordinate system.",
                    "depth_confidence": "Raw VGGT depth confidence; it is not normalized to [0, 1].",
                    "depth_images": "8-bit visualizations clipped to each view's 1st-99th percentile positive finite depth range; invalid and non-positive depths are black.",
                    "depth_confidence_images": "8-bit visualizations clipped to each view's 1st-99th percentile finite confidence range; invalid values are black.",
                    "cam_view": "QuerySplat renderer convention: transposed OpenCV world-to-camera matrix.",
                    "intrinsics": "[fx, fy, cx, cy] in preprocessed/render image pixels.",
                },
                "frames": frame_metadata,
            },
            f,
            indent=2,
        )
        f.write("\n")
    return raw_path, metadata_path, depth_paths, confidence_paths

def save_rgb_tensor_image(image: torch.Tensor, path: str):
    """Save one CHW RGB tensor in [0, 1] as an 8-bit PNG."""
    image = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    image_u8 = (image * 255).astype(np.uint8)
    Image.fromarray(image_u8).save(path)

def save_gaussian_alpha_distribution(
    gaussians: torch.Tensor,
    output_dir: str,
    thresholds: tuple[float, ...] = (
        5e-1,
        1e-1,
        5e-2,
        1e-2,
        5e-3,
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
        5e-6,
        1e-6,
    ),
) -> str:
    """Save a log-scale alpha histogram with threshold percentages annotated."""
    import matplotlib.pyplot as plt

    alpha = gaussians[0, :, 3].detach().float().cpu().clamp(0, 1).numpy()
    total = int(alpha.size)
    positive_alpha = alpha[alpha > 0]
    min_positive = float(positive_alpha.min()) if positive_alpha.size else None
    alpha_floor = min_positive if min_positive is not None else 1e-8
    alpha_floor = max(alpha_floor, 1e-8)

    bin_min = min(alpha_floor, min(thresholds))
    log_min_exp = int(np.floor(np.log10(bin_min)))
    log_min_exp = min(log_min_exp, -6)
    bin_edges = []
    for exp in range(log_min_exp, 0):
        bin_edges.extend((10.0**exp, 5.0 * 10.0**exp))
    bin_edges.append(1.0)
    bin_edges = np.array(sorted(set(bin_edges)), dtype=np.float64)
    clipped_alpha = np.clip(alpha, bin_edges[0], 1.0)
    weights = np.ones_like(clipped_alpha, dtype=np.float64) / max(total, 1)
    hist, edges = np.histogram(clipped_alpha, bins=bin_edges, weights=weights)

    threshold_stats = {}
    for threshold in thresholds:
        mask = alpha > threshold
        threshold_stats[f"{threshold:g}"] = {
            "count": int(mask.sum()),
            "percent": float(mask.mean() * 100.0),
        }
    bin_stats = []
    for bin_idx, (left, right, ratio) in enumerate(zip(edges[:-1], edges[1:], hist)):
        mask = (clipped_alpha >= left) & (clipped_alpha < right)
        if right == edges[-1]:
            mask = (clipped_alpha >= left) & (clipped_alpha <= right)
        bin_stats.append(
            {
                "range": f"[{left:g}, {right:g}{']' if right == edges[-1] else ')'}",
                "contains_alpha_below_range": bool(bin_idx == 0 and (alpha < left).any()),
                "count": int(mask.sum()),
                "percent": float(mask.mean() * 100.0),
            }
        )
    json_path = os.path.join(output_dir, "gaussian_alpha_distribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_gaussians": total,
                "alpha_channel": "gaussians[..., 3]",
                "statistics": {
                    "min": float(alpha.min()) if total else None,
                    "min_positive": min_positive,
                    "max": float(alpha.max()) if total else None,
                    "mean": float(alpha.mean()) if total else None,
                    "median": float(np.median(alpha)) if total else None,
                    "zero_count": int((alpha <= 0).sum()),
                    "zero_percent": float((alpha <= 0).mean() * 100.0) if total else 0.0,
                },
                "thresholds_alpha_greater_than": threshold_stats,
                "log_bins": bin_stats,
            },
            f,
            indent=2,
        )
        f.write("\n")

    png_path = os.path.join(output_dir, "gaussian_alpha_distribution.png")
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.hist(clipped_alpha, bins=bin_edges, weights=weights * 100.0, color="#377eb8", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlim(1.0, bin_edges[0])
    ax.set_xlabel("Gaussian alpha (log scale)")
    ax.set_ylabel("Percentage of Gaussians (%)")
    ax.set_title("Final 3DGS Alpha Distribution")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.6, alpha=0.4)

    tick_values = [1.0]
    for exp in range(-1, log_min_exp - 1, -1):
        tick_values.extend((5.0 * 10.0**exp, 10.0**exp))
    tick_labels = [f"{value:g}" for value in tick_values]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    y_max = max(float((hist * 100.0).max()) if hist.size else 0.0, 1.0)
    ax.set_ylim(0.0, y_max * 1.35)
    for threshold in thresholds:
        if threshold < bin_edges[0]:
            continue
        pct = threshold_stats[f"{threshold:g}"]["percent"]
        ax.axvline(threshold, color="#e41a1c", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.text(
            threshold,
            y_max * 1.03,
            f">{threshold:g}: {pct:.2f}%",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            color="#7f0000",
        )

    text_lines = [
        f"total GS: {total}",
        f"zero alpha: {((alpha <= 0).mean() * 100.0 if total else 0.0):.2f}%",
    ]
    text_lines.extend(
        f"alpha > {threshold:g}: {threshold_stats[f'{threshold:g}']['percent']:.2f}%"
        for threshold in thresholds
    )
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return png_path

def save_gaussian_scale_distribution(
    gaussians: torch.Tensor,
    output_dir: str,
    max_scale: float,
    num_bins: int = 50,
) -> str:
    """Save a linear scale histogram and cumulative percentages up to each bin."""
    import matplotlib.pyplot as plt

    if max_scale <= 0:
        raise ValueError(f"max_scale must be positive, got {max_scale}")
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    scale_xyz = (
        gaussians[0, :, 4:7]
        .detach()
        .float()
        .cpu()
        .clamp_min(0)
        .numpy()
        .astype(np.float64, copy=False)
    )
    scale = scale_xyz.reshape(-1)
    total_gaussians = int(scale_xyz.shape[0])
    total = int(scale.size)
    positive_scale = scale[scale > 0]
    min_positive = float(positive_scale.min()) if positive_scale.size else None
    max_scale = float(max_scale)
    scale_eps = max(abs(max_scale) * 1e-6, 1e-12)
    bin_edges = np.linspace(0.0, max_scale, num_bins + 1, dtype=np.float64)
    clipped_scale = np.clip(scale, 0.0, max_scale)
    hist_counts, edges = np.histogram(clipped_scale, bins=bin_edges)
    hist_percent = hist_counts.astype(np.float64) / max(total, 1) * 100.0
    cumulative_counts = np.cumsum(hist_counts)
    cumulative_percent = cumulative_counts.astype(np.float64) / max(total, 1) * 100.0

    bin_stats = []
    cumulative_stats = []
    for bin_idx, (left, right, count, percent, cum_count, cum_percent) in enumerate(
        zip(
            edges[:-1],
            edges[1:],
            hist_counts,
            hist_percent,
            cumulative_counts,
            cumulative_percent,
        )
    ):
        mask = (clipped_scale >= left) & (clipped_scale < right)
        if right == edges[-1]:
            mask = (clipped_scale >= left) & (clipped_scale <= right)
        contains_above = bool(
            bin_idx == len(hist_counts) - 1
            and (scale > right + scale_eps).any()
        )
        bin_stats.append(
            {
                "range": f"[{left:g}, {right:g}{']' if right == edges[-1] else ')'}",
                "contains_scale_above_range": contains_above,
                "count": int(count),
                "percent": float(percent),
            }
        )
        cumulative_stats.append(
            {
                "scale_lte": float(right),
                "count": int(cum_count),
                "percent": float(cum_percent),
            }
        )

    json_path = os.path.join(output_dir, "gaussian_scale_distribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_gaussians": total_gaussians,
                "total_scale_values": total,
                "scale_channels": "gaussians[..., 4:7]",
                "scale_metric": "all three scale components flattened",
                "max_scale_from_config": max_scale,
                "bin_count": int(num_bins),
                "statistics": {
                    "min": float(scale.min()) if total else None,
                    "min_positive": min_positive,
                    "max": float(scale.max()) if total else None,
                    "mean": float(scale.mean()) if total else None,
                    "median": float(np.median(scale)) if total else None,
                    "zero_count": int((scale <= 0).sum()),
                    "zero_percent": float((scale <= 0).mean() * 100.0) if total else 0.0,
                    "above_max_scale_count": int((scale > max_scale + scale_eps).sum()),
                    "above_max_scale_percent": float((scale > max_scale + scale_eps).mean() * 100.0) if total else 0.0,
                    "component_min": [float(v) for v in scale_xyz.min(axis=0)] if total else None,
                    "component_max": [float(v) for v in scale_xyz.max(axis=0)] if total else None,
                    "component_mean": [float(v) for v in scale_xyz.mean(axis=0)] if total else None,
                    "component_median": [float(v) for v in np.median(scale_xyz, axis=0)] if total else None,
                },
                "linear_bins": bin_stats,
                "cumulative_scale_less_equal": cumulative_stats,
            },
            f,
            indent=2,
        )
        f.write("\n")

    png_path = os.path.join(output_dir, "gaussian_scale_distribution.png")
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    bin_width = max_scale / float(num_bins)
    ax.bar(
        edges[:-1],
        hist_percent,
        width=bin_width,
        align="edge",
        color="#4daf4a",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlim(0.0, max_scale)
    ax.set_xlabel("Gaussian scale component")
    ax.set_ylabel("Percentage of Gaussians (%)")
    ax.set_title("Final 3DGS Scale Distribution")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.25)

    tick_values = np.linspace(0.0, max_scale, 11)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{value:g}" for value in tick_values], rotation=45, ha="right")

    y_max = max(float(hist_percent.max()) if hist_percent.size else 0.0, 1.0)
    ax.set_ylim(0.0, y_max * 1.35)

    cumulative_samples = np.linspace(0, num_bins - 1, 10, dtype=int)
    cumulative_samples = sorted(set(int(idx) for idx in cumulative_samples))
    text_lines = [
        f"total scale values: {total}",
        f"max scale config: {max_scale:g}",
        f"zero scale: {((scale <= 0).mean() * 100.0 if total else 0.0):.2f}%",
        f"> max scale: {((scale > max_scale + scale_eps).mean() * 100.0 if total else 0.0):.2f}%",
    ]
    text_lines.extend(
        f"scale <= {edges[idx + 1]:g}: {cumulative_percent[idx]:.2f}%"
        for idx in cumulative_samples
    )
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return png_path

def _cuda_sync(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def timed_inference_call(device: torch.device, fn):
    _cuda_sync(device)
    start_time = time.perf_counter()
    result = fn()
    _cuda_sync(device)
    return result, time.perf_counter() - start_time

def save_inference_timing(timing_info: dict, output_dir: str):
    json_path = os.path.join(output_dir, "inference_timing.json")
    txt_path = os.path.join(output_dir, "inference_timing.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)
        f.write("\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Inference timing\n")
        for key, value in timing_info.items():
            if isinstance(value, float) and key.endswith("_seconds"):
                f.write(f"{key}: {value:.6f} s\n")
            elif isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

def save_tto_losses(tto_losses: dict, output_dir: str):
    json_path = os.path.join(output_dir, "tto_losses.json")
    txt_path = os.path.join(output_dir, "tto_losses.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tto_losses, f, indent=2)
        f.write("\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TTO final losses\n")
        for key in ("L1", "SSIM", "LPIPS"):
            value = tto_losses.get(key)
            if value is None:
                f.write(f"{key}: None\n")
            else:
                f.write(f"{key}: {value:.6f}\n")
