"""Validate two-pass VGM Sim(3) self-calibration on the fixed DL3DV sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.models.vggt_encoder import VGGTEncoder
from scripts.options import load_options_yaml
from scripts.training import cam_view_to_c2w, forward_self_calibrated_vgm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated two-pass VGM camera alignment")
    parser.add_argument(
        "--images-all",
        type=Path,
        default=Path("../outputs/dl3dv_loader_smoke/images_all.pt"),
    )
    parser.add_argument("--num-input-views", type=int, default=4)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/querysplat_vggto_1B_512_8192.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/vggt_omega_1b_512.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/self_calibration_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this node")

    images_path = args.images_all.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    for name, path in (
        ("images_all", images_path),
        ("config", config_path),
        ("checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    images_all = torch.load(images_path, map_location="cpu", weights_only=True)
    if images_all.ndim == 4:
        images_all = images_all.unsqueeze(0)
    if images_all.ndim != 5 or images_all.shape[2] != 3:
        raise ValueError(f"expected images_all [B,V,3,H,W], got {tuple(images_all.shape)}")
    if not 2 <= args.num_input_views < images_all.shape[1]:
        raise ValueError("num_input_views must leave at least one supervision view")
    if not images_all.is_floating_point() or not torch.isfinite(images_all).all():
        raise ValueError("images_all must be finite floating point")
    if float(images_all.min()) < 0 or float(images_all.max()) > 1:
        raise ValueError("images_all must be in [0,1]")

    height, width = images_all.shape[-2:]
    options = load_options_yaml(config_path).evolve(
        vggt_checkpoint=str(checkpoint_path),
        num_input_views=args.num_input_views,
    )
    model = VGGTEncoder(options).to(device).eval()
    images_all = images_all.to(device)
    input_images = images_all[:, : args.num_input_views]

    with torch.inference_mode():
        output = forward_self_calibrated_vgm(
            model,
            input_images=input_images,
            images_all=images_all,
            image_hw=(height, width),
        )

    transform = output.sim3_all_to_input
    tensors = {
        "input_cam_view": output.input_pass.cam_view.detach().cpu(),
        "input_intrinsics": output.input_pass.intrinsics.detach().cpu(),
        "all_cam_view_unaligned": output.all_view_cameras.cam_view.detach().cpu(),
        "all_cam_view_aligned": output.aligned_all_cam_view.detach().cpu(),
        "all_intrinsics": output.all_view_cameras.intrinsics.detach().cpu(),
        "supervision_cam_view_aligned": output.aligned_all_cam_view[
            :, args.num_input_views :
        ].detach().cpu(),
        "supervision_intrinsics": output.all_view_cameras.intrinsics[
            :, args.num_input_views :
        ].detach().cpu(),
        "sim3_scale": transform.scale.detach().cpu(),
        "sim3_rotation": transform.rotation.detach().cpu(),
        "sim3_translation": transform.translation.detach().cpu(),
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("self-calibration produced NaN or Inf")
    if not bool((tensors["sim3_scale"] > 0).all()):
        raise RuntimeError("self-calibration produced a non-positive scale")

    rotation = tensors["sim3_rotation"].float()
    identity = torch.eye(3).expand_as(rotation)
    orthogonality_error = float(
        (rotation @ rotation.transpose(-2, -1) - identity).abs().max()
    )
    determinant_error = float((torch.det(rotation) - 1.0).abs().max())
    input_c2w = cam_view_to_c2w(tensors["input_cam_view"]).float()
    input_centers = input_c2w[..., :3, 3]
    centered = input_centers - input_centers.mean(dim=1, keepdim=True)
    input_center_rms = centered.square().sum(dim=-1).mean(dim=-1).sqrt()
    center_rmse = output.shared_input_metrics.center_rmse.detach().cpu()
    relative_center_rmse = center_rmse / input_center_rms.clamp_min(1e-8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    products_path = args.output_dir / "self_calibration.pt"
    metadata_path = args.output_dir / "metadata.json"
    torch.save(tensors, products_path)
    metadata = {
        "images_all": str(images_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "images_all_shape": list(images_all.shape),
        "num_input_views": args.num_input_views,
        "num_supervision_views": images_all.shape[1] - args.num_input_views,
        "vgm_forward_count": 2,
        "input_pass_outputs_geometry": True,
        "all_view_pass_public_fields": list(output.all_view_cameras.__dataclass_fields__),
        "sim3_direction": "all-view frame -> input-only frame",
        "sim3_scale": tensors["sim3_scale"].tolist(),
        "sim3_rotation_determinant": torch.det(rotation).tolist(),
        "sim3_rotation_orthogonality_max_error": orthogonality_error,
        "sim3_rotation_determinant_max_error": determinant_error,
        "shared_center_rmse": center_rmse.tolist(),
        "shared_relative_center_rmse": relative_center_rmse.tolist(),
        "shared_rotation_mean_degrees": (
            output.shared_input_metrics.rotation_mean_degrees.detach().cpu().tolist()
        ),
        "shared_rotation_max_degrees": (
            output.shared_input_metrics.rotation_max_degrees.detach().cpu().tolist()
        ),
        "products": {name: list(value.shape) for name, value in tensors.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved products: {products_path.resolve()}")


if __name__ == "__main__":
    main()
