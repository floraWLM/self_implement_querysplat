"""Run the 1024-query dual-branch training forward and gradient smoke test."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid

from scripts.models.querysplat import QuerySplat
from scripts.options import load_options_yaml
from scripts.training import forward_querysplat_training


BASE_QUERY_COUNT = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate base-stage geometry/appearance decoding and rendering"
    )
    parser.add_argument(
        "--images-all",
        type=Path,
        default=Path("../outputs/dl3dv_loader_smoke/images_all.pt"),
    )
    parser.add_argument(
        "--input-normalized",
        type=Path,
        default=Path("../outputs/dl3dv_loader_smoke/input_normalized.pt"),
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
        help="Frozen VGGT-Omega checkpoint; the final QuerySplat checkpoint is not loaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/dual_branch_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    return parser.parse_args()


def load_batched_tensor(path: Path, name: str) -> torch.Tensor:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5 or value.shape[2] != 3:
        raise ValueError(f"{name} must have shape [B,V,3,H,W], got {tuple(value.shape)}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")
    return value


def gradient_stats(parameters) -> dict[str, int | float | bool]:
    parameters = list(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if gradients:
        finite = bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all())
        squared_norm = float(
            torch.stack(
                [gradient.detach().float().square().sum() for gradient in gradients]
            ).sum()
        )
    else:
        finite = True
        squared_norm = 0.0
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "tensors_with_grad": len(gradients),
        "all_finite": finite,
        "l2_norm": squared_norm**0.5,
    }


def named_gradient_groups(model: QuerySplat):
    geometry = [model.gs_tokens]
    geometry.extend(model.enc_dec_backbone.parameters())
    geometry.extend(model.non_rgb_activation_head.parameters())

    appearance = [model.post_rgb_gs_tokens]
    for module in (
        model.patch_embed,
        model.patch_plucker_embed,
        model.original_encoder_norm,
        model.original_kv_proj,
        model.original_k_proj_norm,
        model.post_rgb_decoder_blocks,
        model.post_rgb_delta_head,
    ):
        appearance.extend(module.parameters())

    vggt_adapter = [model.vggt_encoder.eye_token]
    vggt_adapter.extend(model.vggt_encoder.eye_layer_mlp.parameters())
    vggt_adapter.extend(model.vggt_encoder.output_norm.parameters())
    return {
        "geometry": geometry,
        "appearance": appearance,
        "vggt_adapter": vggt_adapter,
    }


def save_rgb_grid(images: torch.Tensor, path: Path) -> None:
    batch, views = images.shape[:2]
    grid = make_grid(
        images.detach().float().cpu().reshape(batch * views, 3, *images.shape[-2:]).clamp(0, 1),
        nrow=views,
        padding=2,
    )
    pixels = grid.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this node")
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 smoke testing requires CUDA")

    images_all_path = args.images_all.expanduser().resolve()
    normalized_path = args.input_normalized.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")

    images_all = load_batched_tensor(images_all_path, "images_all")
    input_normalized = load_batched_tensor(normalized_path, "input_normalized")
    if not 2 <= args.num_input_views < images_all.shape[1]:
        raise ValueError("num_input_views must leave at least one supervision view")
    input_raw = images_all[:, : args.num_input_views]
    if input_normalized.shape != input_raw.shape:
        raise ValueError("saved input_normalized does not match the requested input views")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    torch.testing.assert_close(input_normalized, (input_raw - mean) / std)

    options = load_options_yaml(config_path).evolve(
        vggt_checkpoint=str(checkpoint_path),
        num_input_views=args.num_input_views,
        num_gs_tokens=BASE_QUERY_COUNT,
    )
    model = QuerySplat(options).to(device).train()
    images_all = images_all.to(device)
    input_raw = input_raw.to(device)
    input_normalized = input_normalized.to(device)
    model.zero_grad(set_to_none=True)

    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )
    with precision_context:
        output = forward_querysplat_training(
            model,
            input_normalized=input_normalized,
            input_raw=input_raw,
            images_all=images_all,
        )
        loss_l1 = F.l1_loss(output.render_results["images_pred"], output.supervision_images)

    expected_gaussians = BASE_QUERY_COUNT * options.num_gaussians_per_token
    expected_gaussian_dim = 11 + 3 * (options.gaussian_sh_degree + 1) ** 2
    if output.gaussians.shape != (
        images_all.shape[0],
        expected_gaussians,
        expected_gaussian_dim,
    ):
        raise RuntimeError(f"unexpected Gaussian shape: {tuple(output.gaussians.shape)}")
    expected_render_shape = (
        images_all.shape[0],
        images_all.shape[1] - args.num_input_views,
        3,
        *images_all.shape[-2:],
    )
    if output.render_results["images_pred"].shape != expected_render_shape:
        raise RuntimeError(
            f"unexpected supervision render shape: {tuple(output.render_results['images_pred'].shape)}"
        )
    checked_tensors = [
        output.gaussians,
        output.render_results["images_pred"],
        output.render_results["alphas_pred"],
        output.render_results["depths_pred"],
        loss_l1,
    ]
    if not all(bool(torch.isfinite(value).all()) for value in checked_tensors):
        raise RuntimeError("dual-branch forward produced NaN or Inf")

    loss_l1.backward()
    gradients = {
        name: gradient_stats(parameters)
        for name, parameters in named_gradient_groups(model).items()
    }
    for name, stats in gradients.items():
        if stats["tensors_with_grad"] == 0 or not stats["all_finite"] or stats["l2_norm"] <= 0:
            raise RuntimeError(f"invalid {name} gradients: {stats}")

    frozen_modules = (
        model.vggt_encoder.aggregator,
        model.vggt_encoder.camera_head,
        model.vggt_encoder.depth_head,
    )
    frozen_parameters = [
        parameter for module in frozen_modules for parameter in module.parameters()
    ]
    if any(parameter.requires_grad or parameter.grad is not None for parameter in frozen_parameters):
        raise RuntimeError("frozen VGM parameters unexpectedly received gradients")

    calibration = output.self_calibration
    products = {
        "gaussians": output.gaussians.detach().cpu(),
        "images_pred": output.render_results["images_pred"].detach().cpu(),
        "supervision_images": output.supervision_images.detach().cpu(),
        "supervision_cam_view": output.supervision_decoder.cam_view.detach().cpu(),
        "supervision_intrinsics": output.supervision_decoder.intrinsics.detach().cpu(),
        "sim3_scale": calibration.sim3_all_to_input.scale.detach().cpu(),
        "sim3_rotation": calibration.sim3_all_to_input.rotation.detach().cpu(),
        "sim3_translation": calibration.sim3_all_to_input.translation.detach().cpu(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products_path = args.output_dir / "dual_branch_forward.pt"
    metadata_path = args.output_dir / "metadata.json"
    render_path = args.output_dir / "supervision_render_grid.png"
    target_path = args.output_dir / "supervision_target_grid.png"
    torch.save(products, products_path)
    save_rgb_grid(products["images_pred"], render_path)
    save_rgb_grid(products["supervision_images"], target_path)

    metadata = {
        "images_all": str(images_all_path),
        "input_normalized": str(normalized_path),
        "config_source": str(config_path),
        "vggt_checkpoint": str(checkpoint_path),
        "querysplat_checkpoint_loaded": False,
        "device": str(device),
        "precision": args.precision,
        "num_input_views": args.num_input_views,
        "num_supervision_views": images_all.shape[1] - args.num_input_views,
        "num_geometry_queries": BASE_QUERY_COUNT,
        "gaussians_per_query": options.num_gaussians_per_token,
        "gaussians_shape": list(output.gaussians.shape),
        "render_shape": list(output.render_results["images_pred"].shape),
        "loss_l1": float(loss_l1.detach().cpu()),
        "sim3_scale": products["sim3_scale"].tolist(),
        "shared_relative_alignment": {
            "center_rmse": calibration.shared_input_metrics.center_rmse.detach().cpu().tolist(),
            "rotation_mean_degrees": (
                calibration.shared_input_metrics.rotation_mean_degrees.detach().cpu().tolist()
            ),
        },
        "gradient_groups": gradients,
        "frozen_vggt_parameter_count": sum(
            parameter.numel() for parameter in frozen_parameters
        ),
        "frozen_vggt_has_grad": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved products: {products_path.resolve()}")


if __name__ == "__main__":
    main()
