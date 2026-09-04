"""Overfit QuerySplat's 1024-query base stage on the fixed DL3DV scene."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from scripts.inspect_dual_branch_forward import BASE_QUERY_COUNT, load_batched_tensor, save_rgb_grid
from scripts.models.querysplat import QuerySplat
from scripts.options import load_options_yaml
from scripts.training import (
    ExponentialMovingAverage,
    LinearWeightSchedule,
    OptimizerConfig,
    QuerySplatLossConfig,
    build_adamw,
    build_lpips_vgg,
    build_warmup_cosine_scheduler,
    compute_querysplat_losses,
    decode_and_render_from_self_calibration,
    load_training_checkpoint,
    materialize_fixed_scene_self_calibration,
    prepare_fixed_scene_vgm_cache,
    save_training_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-scene QuerySplat overfit diagnostic")
    parser.add_argument(
        "--images-all", type=Path,
        default=Path("../outputs/dl3dv_loader_smoke/images_all.pt"),
    )
    parser.add_argument(
        "--input-normalized", type=Path,
        default=Path("../outputs/dl3dv_loader_smoke/input_normalized.pt"),
    )
    parser.add_argument("--num-input-views", type=int, default=4)
    parser.add_argument(
        "--config", type=Path,
        default=Path("checkpoints/querysplat_vggto_1B_512_8192.yaml"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/vggt_omega_1b_512.pt"),
    )
    parser.add_argument("--workspace", type=Path, default=Path("../outputs/fixed_scene_overfit"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--early-reg-end-step", type=int, default=50)
    parser.add_argument("--lpips-start-step", type=int, default=20)
    parser.add_argument("--lpips-ramp-end-step", type=int, default=50)
    parser.add_argument("--max-depth-points", type=int, default=2048)
    parser.add_argument("--max-gaussian-chamfer-points", type=int, default=2048)
    parser.add_argument("--chamfer-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or not 0 <= args.warmup_steps < args.steps:
        raise ValueError("steps must be positive and warmup_steps must be in [0,steps)")
    if args.gradient_clip <= 0 or args.checkpoint_every <= 0 or args.log_every <= 0:
        raise ValueError("gradient clip and logging/checkpoint intervals must be positive")
    if not 0 <= args.lpips_start_step <= args.lpips_ramp_end_step:
        raise ValueError("invalid LPIPS schedule")
    if args.early_reg_end_step <= 0:
        raise ValueError("early-reg-end-step must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed-scene training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    images_path = args.images_all.expanduser().resolve()
    normalized_path = args.input_normalized.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    vggt_path = args.checkpoint.expanduser().resolve()
    for name, path in (
        ("images_all", images_path),
        ("input_normalized", normalized_path),
        ("config", config_path),
        ("VGGT checkpoint", vggt_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    images_all = load_batched_tensor(images_path, "images_all")
    input_normalized = load_batched_tensor(normalized_path, "input_normalized")
    if not 2 <= args.num_input_views < images_all.shape[1]:
        raise ValueError("num_input_views must leave at least one supervision view")
    input_raw = images_all[:, : args.num_input_views]
    if input_normalized.shape != input_raw.shape:
        raise ValueError("input_normalized does not match selected input views")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    torch.testing.assert_close(input_normalized, (input_raw - mean) / std)

    options = load_options_yaml(config_path).evolve(
        vggt_checkpoint=str(vggt_path),
        num_input_views=args.num_input_views,
        num_gs_tokens=BASE_QUERY_COUNT,
    )
    model = QuerySplat(options).to(device).train()
    images_all = images_all.to(device)
    input_raw = input_raw.to(device)
    input_normalized = input_normalized.to(device)

    def precision_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.precision == "bf16" else nullcontext()
        )

    print("Preparing two-pass frozen VGM cache...")
    with precision_context():
        vgm_cache = prepare_fixed_scene_vgm_cache(
            model.vggt_encoder,
            input_images=input_raw,
            images_all=images_all,
            image_hw=tuple(images_all.shape[-2:]),
        )
    print("VGM cache ready; later steps do not rerun the 1B backbone.")

    loss_config = QuerySplatLossConfig(
        lambda_ssim=0.2,
        lambda_visibility=1.0,
        lpips_schedule=LinearWeightSchedule(
            0.0, 0.05, args.lpips_start_step, args.lpips_ramp_end_step
        ),
        chamfer_schedule=LinearWeightSchedule(1.0, 0.0, 0, args.early_reg_end_step),
        opacity_schedule=LinearWeightSchedule(0.1, 0.0, 0, args.early_reg_end_step),
        opacity_floor=0.1,
        visibility_distance_threshold=1.0,
        max_depth_points=args.max_depth_points,
        max_gaussian_chamfer_points=args.max_gaussian_chamfer_points,
        chamfer_chunk_size=args.chamfer_chunk_size,
    )
    lpips_model = build_lpips_vgg(device)
    optimizer = build_adamw(
        model,
        OptimizerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        minimum_lr_ratio=args.minimum_lr_ratio,
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    global_step = 0
    if args.resume is not None:
        global_step, checkpoint_extra = load_training_checkpoint(
            args.resume.expanduser().resolve(), model, optimizer, scheduler, ema
        )
        expected = {"total_steps": args.steps, "num_queries": BASE_QUERY_COUNT}
        if any(checkpoint_extra.get(key) != value for key, value in expected.items()):
            raise ValueError("resume checkpoint does not match steps/query configuration")
        print(f"Resumed at global step {global_step}")
    if global_step > args.steps:
        raise ValueError("checkpoint global step exceeds requested total steps")

    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoint_path = workspace / "latest.pt"
    log_path = workspace / "metrics.jsonl"
    if global_step == 0:
        log_path.write_text("", encoding="utf-8")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss_history: list[float] = []
    while global_step < args.steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        with precision_context():
            self_calibration = materialize_fixed_scene_self_calibration(
                model.vggt_encoder, vgm_cache
            )
            output = decode_and_render_from_self_calibration(
                model,
                input_normalized=input_normalized,
                input_raw=input_raw,
                images_all=images_all,
                self_calibration=self_calibration,
            )
        if global_step == 0:
            save_rgb_grid(
                output.render_results["images_pred"],
                workspace / "initial_train_render.png",
            )
        losses = compute_querysplat_losses(
            output,
            step=global_step,
            config=loss_config,
            znear=options.znear,
            zfar=options.zfar,
            lpips_model=lpips_model,
        )
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"non-finite loss at step {global_step}")
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, max_norm=args.gradient_clip
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {global_step}")
        optimizer.step()
        scheduler.step()
        ema.update(model)
        global_step += 1

        scalar_losses = {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
        }
        loss_history.append(scalar_losses["loss"])
        record = {
            "step": global_step,
            "lr": learning_rate,
            "grad_norm_before_clip": float(grad_norm.detach().float().cpu()),
            **scalar_losses,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if global_step == 1 or global_step % args.log_every == 0:
            print(
                f"step={global_step}/{args.steps} lr={learning_rate:.3e} "
                f"loss={scalar_losses['loss']:.6f} photo={scalar_losses['loss_photo']:.6f} "
                f"cd={scalar_losses['loss_chamfer']:.6f} grad={record['grad_norm_before_clip']:.6f}"
            )
        if global_step % args.checkpoint_every == 0 or global_step == args.steps:
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                ema,
                global_step=global_step,
                extra={
                    "total_steps": args.steps,
                    "num_queries": BASE_QUERY_COUNT,
                    "precision": args.precision,
                },
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    model.eval()
    with torch.no_grad(), precision_context():
        self_calibration = materialize_fixed_scene_self_calibration(
            model.vggt_encoder, vgm_cache
        )
        final_output = decode_and_render_from_self_calibration(
            model,
            input_normalized,
            input_raw,
            images_all,
            self_calibration,
        )
    save_rgb_grid(
        final_output.render_results["images_pred"], workspace / "final_train_render.png"
    )
    save_rgb_grid(final_output.supervision_images, workspace / "target.png")
    summary = {
        "completed_steps": global_step,
        "initial_loss_this_run": loss_history[0] if loss_history else None,
        "final_loss_this_run": loss_history[-1] if loss_history else None,
        "best_loss_this_run": min(loss_history) if loss_history else None,
        "checkpoint": str(checkpoint_path),
        "metrics": str(log_path),
        "ema_decay": args.ema_decay,
        "gradient_clip": args.gradient_clip,
        "vggt_cached": True,
        "sim3_scale": vgm_cache.sim3_all_to_input.scale.detach().cpu().tolist(),
    }
    (workspace / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
