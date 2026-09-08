"""Multi-scene DL3DV training entry point for QuerySplat's 1024-query base stage."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.inspect_dual_branch_forward import BASE_QUERY_COUNT, save_rgb_grid
from scripts.models.querysplat import QuerySplat
from scripts.options import load_options_yaml
from scripts.training import (
    ExponentialMovingAverage,
    LinearWeightSchedule,
    OptimizerConfig,
    QuerySplatLossConfig,
    QuerySplatTrainingModule,
    build_adamw,
    build_dl3dv_dataloader,
    build_lpips_vgg,
    build_warmup_cosine_scheduler,
    compute_querysplat_losses,
    load_training_checkpoint,
    prepare_querysplat_images,
    save_training_checkpoint,
    set_dl3dv_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QuerySplat on dynamically sampled DL3DV scenes")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("checkpoints/querysplat_vggto_1B_512_8192.yaml"),
    )
    parser.add_argument(
        "--vgm-checkpoint", type=Path,
        default=Path("checkpoints/vggt_omega_1b_512.pt"),
    )
    parser.add_argument("--workspace", type=Path, default=Path("../outputs/querysplat_dl3dv_base"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-input-views", type=int, default=4)
    parser.add_argument("--num-supervision-views", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--image-every", type=int, default=500)
    parser.add_argument("--early-reg-end-step", type=int, default=30_000)
    parser.add_argument("--lpips-start-step", type=int, default=10_000)
    parser.add_argument("--lpips-ramp-end-step", type=int, default=30_000)
    parser.add_argument("--max-depth-points", type=int, default=8_192)
    parser.add_argument("--max-gaussian-chamfer-points", type=int, default=8_192)
    parser.add_argument("--chamfer-chunk-size", type=int, default=1_024)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_input_views != 4:
        raise ValueError("Step 8 is the four-input-view, 1024-query base stage")
    if args.num_supervision_views < 1:
        raise ValueError("at least one supervision view is required")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.total_steps <= 0 or not 0 <= args.warmup_steps < args.total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    if min(args.checkpoint_every, args.log_every, args.image_every) <= 0:
        raise ValueError("logging and checkpoint intervals must be positive")
    if args.gradient_clip <= 0 or args.early_reg_end_step <= 0:
        raise ValueError("gradient clip and early regularization endpoint must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0 <= args.minimum_lr_ratio <= 1 or not 0 <= args.ema_decay < 1:
        raise ValueError("minimum LR ratio and EMA decay must lie in [0,1]")
    if min(
        args.max_depth_points,
        args.max_gaussian_chamfer_points,
        args.chamfer_chunk_size,
    ) <= 0:
        raise ValueError("Chamfer sampling and chunk sizes must be positive")
    if not 0 <= args.lpips_start_step <= args.lpips_ramp_end_step:
        raise ValueError("invalid LPIPS schedule")


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("multi-scene QuerySplat training requires CUDA")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid WORLD_SIZE/RANK environment")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def reduce_scalars(values: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    names = sorted(values)
    packed = torch.stack([values[name].detach().float() for name in names])
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed /= world_size
    return {name: float(value.cpu()) for name, value in zip(names, packed)}


def local_rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}


def gather_rng_states(device: torch.device, world_size: int) -> list[dict[str, torch.Tensor]]:
    state = local_rng_state(device)
    if world_size == 1:
        return [state]
    states: list[Any] = [None] * world_size
    dist.all_gather_object(states, state)
    return states


def restore_rank_rng_state(states: list[dict[str, torch.Tensor]], rank: int, device: torch.device) -> None:
    if len(states) <= rank:
        raise ValueError("checkpoint does not contain RNG state for this rank")
    torch.set_rng_state(states[rank]["torch"])
    torch.cuda.set_rng_state(states[rank]["cuda"], device)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        config_path = args.config.expanduser().resolve()
        vgm_path = args.vgm_checkpoint.expanduser().resolve()
        for label, path in (("config", config_path), ("VGM checkpoint", vgm_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        options = load_options_yaml(config_path).evolve(
            vggt_checkpoint=str(vgm_path),
            num_input_views=args.num_input_views,
            num_gs_tokens=BASE_QUERY_COUNT,
        )
        model = QuerySplat(options).to(device).train()
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
            total_steps=args.total_steps,
            minimum_lr_ratio=args.minimum_lr_ratio,
        )
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        loader = build_dl3dv_dataloader(
            args.data_root,
            num_input_views=args.num_input_views,
            num_supervision_views=args.num_supervision_views,
            image_size=options.img_size[0],
            num_workers=args.num_workers,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        steps_per_epoch = len(loader.dataloader)
        resume_contract = {
            "total_steps": args.total_steps,
            "num_queries": BASE_QUERY_COUNT,
            "num_input_views": args.num_input_views,
            "num_supervision_views": args.num_supervision_views,
            "world_size": world_size,
            "steps_per_epoch": steps_per_epoch,
        }
        global_step = 0
        resume_rng_states = None
        if args.resume is not None:
            global_step, extra = load_training_checkpoint(
                args.resume.expanduser().resolve(), model, optimizer, scheduler, ema
            )
            if any(extra.get(key) != value for key, value in resume_contract.items()):
                raise ValueError("resume checkpoint does not match the current training contract")
            resume_rng_states = extra["rank_rng_states"]
            if is_main:
                print(f"Resumed at global step {global_step}")
        if global_step > args.total_steps:
            raise ValueError("checkpoint step exceeds total_steps")

        training_graph = QuerySplatTrainingModule(model)
        if world_size > 1:
            training_graph = DistributedDataParallel(
                training_graph,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
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
        lpips_model = build_lpips_vgg(device)
        # Restore only after all module construction, which may consume RNG.
        if resume_rng_states is None:
            torch.manual_seed(args.seed + rank)
            torch.cuda.manual_seed(args.seed + rank)
        else:
            restore_rank_rng_state(resume_rng_states, rank, device)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        workspace = args.workspace.expanduser().resolve()
        checkpoint_path = workspace / "latest.pt"
        metrics_path = workspace / "metrics.jsonl"
        images_dir = workspace / "images"
        if is_main:
            workspace.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            if global_step == 0:
                metrics_path.write_text("", encoding="utf-8")
            (workspace / "config.json").write_text(
                json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
            )
        barrier(world_size)

        def precision_context():
            return (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16" else nullcontext()
            )

        epoch = global_step // steps_per_epoch
        offset = global_step % steps_per_epoch
        validated_loader_contract = False
        while global_step < args.total_steps:
            set_dl3dv_epoch(loader, epoch)
            for batch_index, loader_batch in enumerate(loader.dataloader):
                if batch_index < offset:
                    continue
                image_batch = prepare_querysplat_images(
                    loader_batch,
                    args.num_input_views,
                    validate_provider_contract=not validated_loader_contract,
                )
                validated_loader_contract = True
                images_all = torch.cat(
                    [image_batch.input_raw, image_batch.supervision_images], dim=1
                ).to(device, non_blocking=True)
                input_raw = image_batch.input_raw.to(device, non_blocking=True)
                input_normalized = image_batch.input_normalized.to(device, non_blocking=True)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                learning_rate = optimizer.param_groups[0]["lr"]
                with precision_context():
                    output = training_graph(input_normalized, input_raw, images_all)
                losses = compute_querysplat_losses(
                    output,
                    step=global_step,
                    config=loss_config,
                    znear=options.znear,
                    zfar=options.zfar,
                    lpips_model=lpips_model,
                )
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError(f"rank {rank}: non-finite loss at step {global_step}")
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=args.gradient_clip
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"rank {rank}: non-finite gradient at step {global_step}")
                optimizer.step()
                scheduler.step()
                ema.update(model)
                global_step += 1

                averaged = reduce_scalars(losses, world_size)
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": learning_rate,
                    "grad_norm_before_clip_rank0": float(grad_norm.detach().float().cpu()),
                    **averaged,
                }
                if is_main:
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    if global_step == 1 or global_step % args.log_every == 0:
                        print(
                            f"step={global_step}/{args.total_steps} epoch={epoch} "
                            f"lr={learning_rate:.3e} loss={averaged['loss']:.6f} "
                            f"photo={averaged['loss_photo']:.6f} "
                            f"grad={record['grad_norm_before_clip_rank0']:.6f}"
                        )
                    if global_step == 1 or global_step % args.image_every == 0:
                        save_rgb_grid(
                            output.render_results["images_pred"],
                            images_dir / f"pred_{global_step:07d}.png",
                        )
                        save_rgb_grid(
                            output.supervision_images,
                            images_dir / f"target_{global_step:07d}.png",
                        )

                should_checkpoint = (
                    global_step % args.checkpoint_every == 0
                    or global_step == args.total_steps
                )
                if should_checkpoint:
                    rank_rng_states = gather_rng_states(device, world_size)
                    barrier(world_size)
                    if is_main:
                        save_training_checkpoint(
                            checkpoint_path,
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            global_step=global_step,
                            extra={**resume_contract, "rank_rng_states": rank_rng_states},
                        )
                        print(f"Saved checkpoint: {checkpoint_path}")
                    barrier(world_size)
                if global_step >= args.total_steps:
                    break
            epoch += 1
            offset = 0

        if is_main:
            summary = {
                **resume_contract,
                "completed_steps": global_step,
                "checkpoint": str(checkpoint_path),
                "metrics": str(metrics_path),
                "dynamic_view_sampling": True,
                "ddp": world_size > 1,
                "ema_decay": args.ema_decay,
            }
            (workspace / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, indent=2))
        barrier(world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
