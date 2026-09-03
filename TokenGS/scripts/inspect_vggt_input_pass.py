"""Run the frozen VGGT-Omega input-only pass on a saved image tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from scripts.models.vggt_encoder import VGGTEncoder
from scripts.options import load_options_yaml
from scripts.training import forward_vggt_input_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract QuerySplat geometry features, cameras, and depth once."
    )
    parser.add_argument(
        "--input-tensor",
        type=Path,
        default=Path("../outputs/dl3dv_loader_smoke/input_raw.pt"),
    )
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
        default=Path("../outputs/vggt_input_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize_maps_for_grid(maps: torch.Tensor) -> torch.Tensor:
    """Percentile-normalize each single-channel map for diagnostic PNGs."""
    normalized = []
    for value in maps.reshape(-1, 1, *maps.shape[-2:]).float().cpu():
        finite = torch.isfinite(value)
        valid_values = value[finite]
        if valid_values.numel() == 0:
            normalized.append(torch.zeros_like(value))
            continue
        lower = torch.quantile(valid_values, 0.01)
        upper = torch.quantile(valid_values, 0.99)
        denominator = (upper - lower).clamp_min(1e-8)
        normalized.append(((value - lower) / denominator).clamp(0, 1))
    return torch.stack(normalized)


def save_map_grid(maps: torch.Tensor, path: Path) -> None:
    grid = make_grid(normalize_maps_for_grid(maps), nrow=maps.shape[1], padding=2)
    pixels = (
        grid[0]
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(pixels, mode="L").save(path)


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    finite = value[torch.isfinite(value)].float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(finite.min()),
        "median": float(finite.median()),
        "max": float(finite.max()),
        "all_finite": bool(torch.isfinite(value).all()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this node")

    input_path = args.input_tensor.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input tensor not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"QuerySplat config not found: {config_path}")

    input_raw = torch.load(input_path, map_location="cpu", weights_only=True)
    if input_raw.ndim == 4:
        input_raw = input_raw.unsqueeze(0)
    if input_raw.ndim != 5 or input_raw.shape[2] != 3:
        raise ValueError(f"expected [B,V,3,H,W] input RGB, got {tuple(input_raw.shape)}")
    if not input_raw.is_floating_point() or not torch.isfinite(input_raw).all():
        raise ValueError("input_raw must be a finite floating-point tensor")
    if float(input_raw.min()) < 0 or float(input_raw.max()) > 1:
        raise ValueError("input_raw must be in [0,1]")

    height, width = input_raw.shape[-2:]
    options = load_options_yaml(config_path).evolve(
        vggt_checkpoint=str(checkpoint_path),
        num_input_views=input_raw.shape[1],
    )
    model = VGGTEncoder(options).to(device)
    model.eval()

    frozen_modules = (model.aggregator, model.camera_head, model.depth_head)
    if any(parameter.requires_grad for module in frozen_modules for parameter in module.parameters()):
        raise RuntimeError("a frozen VGGT backbone/head parameter unexpectedly requires gradients")
    adaptive_parameters = list(model.eye_layer_mlp.parameters()) + [model.eye_token]
    if not all(parameter.requires_grad for parameter in adaptive_parameters):
        raise RuntimeError("QuerySplat eye token/layer mixer must remain trainable")

    with torch.inference_mode():
        output = forward_vggt_input_once(
            model,
            input_raw.to(device),
            image_hw=(height, width),
        )

    tensors = {
        "geometry_tokens": output.geometry.tokens.detach().cpu(),
        "eye_token": output.geometry.eye_token.detach().cpu(),
        "layer_weights": output.geometry.layer_weights.detach().cpu(),
        "camera_tokens": output.geometry.camera_tokens.detach().cpu(),
        "scene_tokens": output.geometry.scene_tokens.detach().cpu(),
        "cam_view": output.cam_view.detach().cpu(),
        "intrinsics": output.intrinsics.detach().cpu(),
        "pose_enc": output.pose_enc.detach().cpu(),
        "depth": output.depth.detach().cpu(),
        "depth_confidence": output.depth_confidence.detach().cpu(),
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("VGGT input pass produced NaN or Inf")

    batch, views = input_raw.shape[:2]
    expected_shapes = {
        "cam_view": (batch, views, 4, 4),
        "intrinsics": (batch, views, 4),
        "depth": (batch, views, 1, height, width),
        "depth_confidence": (batch, views, 1, height, width),
    }
    for name, expected in expected_shapes.items():
        if tuple(tensors[name].shape) != expected:
            raise RuntimeError(f"{name} has shape {tuple(tensors[name].shape)}, expected {expected}")

    w2c = tensors["cam_view"].transpose(-2, -1)
    identity = torch.eye(4).expand_as(w2c)
    camera_inverse_error = float((w2c @ torch.linalg.inv(w2c) - identity).abs().max())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    products_path = args.output_dir / "vggt_input_pass.pt"
    metadata_path = args.output_dir / "metadata.json"
    depth_grid_path = args.output_dir / "depth_grid.png"
    confidence_grid_path = args.output_dir / "depth_confidence_grid.png"
    torch.save(tensors, products_path)
    save_map_grid(tensors["depth"], depth_grid_path)
    save_map_grid(tensors["depth_confidence"], confidence_grid_path)

    metadata = {
        "input_tensor": str(input_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "input_shape": list(input_raw.shape),
        "aggregator_forward_count": 1,
        "tokens_per_view": output.geometry.tokens_per_view,
        "special_tokens_per_view": output.geometry.special_tokens_per_view,
        "camera_inverse_max_error": camera_inverse_error,
        "frozen_vggt_parameter_count": sum(
            parameter.numel()
            for module in frozen_modules
            for parameter in module.parameters()
        ),
        "trainable_adapter_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "products": {name: tensor_stats(value) for name, value in tensors.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved products: {products_path.resolve()}")


if __name__ == "__main__":
    main()
