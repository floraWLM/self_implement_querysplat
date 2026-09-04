"""Run all QuerySplat loss terms and a full backward on the fixed DL3DV sample."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from scripts.inspect_dual_branch_forward import (
    BASE_QUERY_COUNT,
    gradient_stats,
    load_batched_tensor,
    named_gradient_groups,
)
from scripts.models.querysplat import QuerySplat
from scripts.options import load_options_yaml
from scripts.training import (
    LinearWeightSchedule,
    QuerySplatLossConfig,
    build_lpips_vgg,
    compute_querysplat_losses,
    forward_querysplat_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the complete scheduled loss system")
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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/full_loss_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--early-reg-end-step", type=int, default=10000)
    parser.add_argument("--lpips-start-step", type=int, default=1000)
    parser.add_argument("--lpips-ramp-end-step", type=int, default=10000)
    parser.add_argument("--max-depth-points", type=int, default=2048)
    parser.add_argument("--max-gaussian-chamfer-points", type=int, default=2048)
    parser.add_argument("--chamfer-chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the renderer and fused SSIM smoke test require CUDA")
    if args.step < 0:
        raise ValueError("step must be non-negative")
    if not 0 <= args.lpips_start_step <= args.lpips_ramp_end_step:
        raise ValueError("invalid LPIPS ramp")
    if args.early_reg_end_step <= 0:
        raise ValueError("early-reg-end-step must be positive")

    images_path = args.images_all.expanduser().resolve()
    normalized_path = args.input_normalized.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")

    images_all = load_batched_tensor(images_path, "images_all")
    input_normalized = load_batched_tensor(normalized_path, "input_normalized")
    if not 2 <= args.num_input_views < images_all.shape[1]:
        raise ValueError("num_input_views must leave at least one supervision view")
    input_raw = images_all[:, : args.num_input_views]
    if input_normalized.shape != input_raw.shape:
        raise ValueError("input_normalized does not match the selected input views")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    torch.testing.assert_close(input_normalized, (input_raw - mean) / std)

    loss_config = QuerySplatLossConfig(
        lambda_ssim=0.2,
        lambda_visibility=1.0,
        lpips_schedule=LinearWeightSchedule(
            0.0, 0.05, args.lpips_start_step, args.lpips_ramp_end_step
        ),
        chamfer_schedule=LinearWeightSchedule(
            1.0, 0.0, 0, args.early_reg_end_step
        ),
        opacity_schedule=LinearWeightSchedule(
            0.1, 0.0, 0, args.early_reg_end_step
        ),
        opacity_floor=0.1,
        visibility_distance_threshold=1.0,
        max_depth_points=args.max_depth_points,
        max_gaussian_chamfer_points=args.max_gaussian_chamfer_points,
        chamfer_chunk_size=args.chamfer_chunk_size,
    )
    active_lpips_weight = loss_config.lpips_schedule.value(args.step)
    lpips_model = build_lpips_vgg(device) if active_lpips_weight > 0 else None

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
    losses = compute_querysplat_losses(
        output,
        step=args.step,
        config=loss_config,
        znear=options.znear,
        zfar=options.zfar,
        lpips_model=lpips_model,
    )
    if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
        raise RuntimeError("full loss system produced NaN or Inf")
    if any(float(losses[name]) <= 0 for name in ("weight_lpips", "weight_chamfer", "weight_opacity")):
        raise RuntimeError("smoke step must exercise all three scheduled weights")

    losses["loss"].backward()
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
    if lpips_model is not None and any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in lpips_model.parameters()
    ):
        raise RuntimeError("frozen LPIPS parameters unexpectedly received gradients")

    scalar_names = (
        "loss", "loss_photo", "loss_l1", "loss_ssim", "loss_lpips",
        "loss_visibility", "loss_chamfer", "loss_opacity_floor",
        "weight_lpips", "weight_chamfer", "weight_opacity",
    )
    scalar_values = {
        name: float(losses[name].detach().float().cpu()) for name in scalar_names
    }
    weighted_terms = {
        "photo": scalar_values["loss_photo"],
        "visibility": loss_config.lambda_visibility * scalar_values["loss_visibility"],
        "chamfer": scalar_values["weight_chamfer"] * scalar_values["loss_chamfer"],
        "opacity_floor": scalar_values["weight_opacity"]
        * scalar_values["loss_opacity_floor"],
    }
    metadata = {
        "images_all": str(images_path),
        "input_normalized": str(normalized_path),
        "config_source": str(config_path),
        "vggt_checkpoint": str(checkpoint_path),
        "querysplat_checkpoint_loaded": False,
        "device": str(device),
        "precision": args.precision,
        "step": args.step,
        "num_geometry_queries": BASE_QUERY_COUNT,
        "gaussians_shape": list(output.gaussians.shape),
        "render_shape": list(output.render_results["images_pred"].shape),
        "losses": scalar_values,
        "weighted_terms": weighted_terms,
        "schedule": {
            "early_reg_end_step": args.early_reg_end_step,
            "lpips_start_step": args.lpips_start_step,
            "lpips_ramp_end_step": args.lpips_ramp_end_step,
        },
        "chamfer_sampling": {
            "max_depth_points": args.max_depth_points,
            "max_gaussian_points": args.max_gaussian_chamfer_points,
            "chunk_size": args.chamfer_chunk_size,
        },
        "gradient_groups": gradients,
        "frozen_vggt_has_grad": False,
        "frozen_lpips_has_grad": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    products_path = args.output_dir / "full_loss.pt"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    torch.save({name: losses[name].detach().cpu() for name in scalar_names}, products_path)
    print(json.dumps(metadata, indent=2))
    print(f"Saved products: {products_path.resolve()}")


if __name__ == "__main__":
    main()
