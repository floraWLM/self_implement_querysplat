# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Custom-image inference for the released QuerySplat VGGT-Omega model."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file

from scripts.models import ModelInput, ModelInputDecoder, ModelInputEncoder, QuerySplat
from scripts.options import Options, load_options_yaml
from scripts.output_utils import (
    opacity_threshold_suffix,
    preprocessed_input_frame_path,
    save_colored_pointcloud,
    save_gaussian_alpha_distribution,
    save_gaussian_scale_distribution,
    save_inference_timing,
    save_predicted_input_cameras,
    save_rgb_tensor_image,
    save_tto_losses,
    save_vggt_depth_pointcloud,
    save_vggt_input_depths,
    timed_inference_call,
)
from scripts.rendering.gs import GaussianRenderer
from scripts.utils.data import ImageTransform


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VENDOR_PREFIXES = (
    "vggt_encoder.aggregator.",
    "vggt_encoder.camera_head.",
    "vggt_encoder.depth_head.",
)


def parse_args():
    parser = argparse.ArgumentParser(description="QuerySplat custom-image inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--category",
        choices=("custom",),
        default="custom",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--gaussian_save_opacity_threshold",
        nargs="+",
        default=["0.05"],
    )
    parser.add_argument("--save_gaussian_alpha_distribution", action="store_true")
    parser.add_argument("--save_gaussian_scale_distribution", action="store_true")
    parser.add_argument("--save_vggt_depth_pointcloud", action="store_true")
    parser.add_argument("--vggt_depth_pointcloud_target_points", type=int)
    parser.add_argument("--save_vggt_input_depths", action="store_true")
    parser.add_argument("--save_predicted_input_cameras", action="store_true")
    parser.add_argument("--use_tto", action="store_true")
    parser.add_argument("--tto_n_steps", type=int, default=20)
    parser.add_argument("--tto_lr", type=float, default=5e-3)
    parser.add_argument("--tto_lpips_weight", type=float, default=0.05)
    parser.add_argument(
        "--tto_save_step",
        nargs="*",
        default=None,
        help="TTO steps whose Gaussian PLYs should also be saved.",
    )
    return parser.parse_args()


def parse_number_list(raw_values: list[str] | None, value_type, option: str):
    values = []
    for raw in raw_values or []:
        text = raw.strip().strip("{}[]()")
        for item in text.replace(",", " ").split():
            try:
                values.append(value_type(item))
            except ValueError as error:
                raise ValueError(f"Invalid value for {option}: {item!r}") from error
    return values


def list_input_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")
    paths = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not paths:
        raise ValueError(f"No supported images found in: {folder}")
    return paths


def preprocess_images(paths: list[str], opt: Options) -> torch.Tensor:
    transform = ImageTransform(crop_size=opt.img_size, sample_size=opt.img_size, max_crop=True)
    images = []
    for path in paths:
        image = np.array(Image.open(path).convert("RGB"), copy=True)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor, _, _, _ = transform.preprocess_images(tensor.unsqueeze(0))
        images.append(tensor[0])
    return torch.stack(images)


def build_model_input(images: torch.Tensor, decoder: ModelInputDecoder | None = None) -> ModelInput:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    normalized = normalize(images.clone()).unsqueeze(0)
    raw = images.unsqueeze(0)
    encoder = ModelInputEncoder(images_rgb=normalized, images_rgb_unnormalized=raw)
    if decoder is None:
        decoder = ModelInputDecoder(cam_view=torch.empty(0), intrinsics=torch.empty(0))
    return ModelInput(encoder=encoder, decoder=decoder)


def load_querysplat_checkpoint(model: QuerySplat, path: str) -> None:
    if not path.endswith(".safetensors"):
        raise ValueError("QuerySplat checkpoint must use the .safetensors format")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"QuerySplat checkpoint not found: {path}")
    checkpoint = load_file(path, device="cpu")
    if len(checkpoint) != 521:
        raise ValueError(f"Expected 521 QuerySplat tensors, got {len(checkpoint)}")
    forbidden = [key for key in checkpoint if key.startswith(VENDOR_PREFIXES)]
    if forbidden:
        raise ValueError(f"QuerySplat checkpoint contains VGGT-Omega tensors: {forbidden[:3]}")
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected QuerySplat keys: {incompatible.unexpected_keys}")
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith(VENDOR_PREFIXES)
    ]
    if invalid_missing:
        raise ValueError(f"Missing QuerySplat keys: {invalid_missing}")
    del checkpoint


def predict_cameras_and_depths(model: QuerySplat, model_input: ModelInput):
    cam_view, intrinsics, pose_enc, depth, depth_conf = (
        model.vggt_encoder.predict_cameras_and_depths(
            model_input.encoder.images_rgb_unnormalized,
            image_hw=model.img_size,
        )
    )
    return {
        "cam_view": cam_view,
        "intrinsics": intrinsics,
        "pose_enc": pose_enc,
        "depth": depth,
        "depth_conf": depth_conf,
    }


def validate_args(args, tto_steps: list[int], thresholds: list[float]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This release requires a CUDA device")
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Opacity thresholds must be in [0, 1]")
    if args.vggt_depth_pointcloud_target_points is not None:
        if not args.save_vggt_depth_pointcloud:
            raise ValueError("--vggt_depth_pointcloud_target_points requires --save_vggt_depth_pointcloud")
        if args.vggt_depth_pointcloud_target_points <= 0:
            raise ValueError("--vggt_depth_pointcloud_target_points must be positive")
    if args.save_vggt_depth_pointcloud and args.vggt_depth_pointcloud_target_points is None:
        raise ValueError("--save_vggt_depth_pointcloud requires --vggt_depth_pointcloud_target_points")
    if args.tto_n_steps < 0 or args.tto_lr <= 0 or args.tto_lpips_weight < 0:
        raise ValueError("TTO steps/LPIPS weight must be non-negative and TTO LR must be positive")
    if tto_steps and not args.use_tto:
        raise ValueError("--tto_save_step requires --use_tto")
    if any(step < 0 or step > args.tto_n_steps for step in tto_steps):
        raise ValueError("TTO save steps must fall between 0 and --tto_n_steps")


def main():
    args = parse_args()
    thresholds = sorted(set(parse_number_list(
        args.gaussian_save_opacity_threshold,
        float,
        "--gaussian_save_opacity_threshold",
    )))
    tto_steps = sorted(set(parse_number_list(args.tto_save_step, int, "--tto_save_step")))
    validate_args(args, tto_steps, thresholds)

    options = load_options_yaml(args.config)
    image_paths = list_input_images(args.input_folder)
    options = options.evolve(num_input_views=len(image_paths))
    images = preprocess_images(image_paths, options)
    sources = [(os.path.splitext(os.path.basename(path))[0], path) for path in image_paths]
    print(f"Using {len(image_paths)} custom input images")

    device = torch.device("cuda")
    model = QuerySplat(options).to(device)
    load_querysplat_checkpoint(model, args.checkpoint)
    model.eval()
    model_input = build_model_input(images)
    model_input = ModelInput(
        encoder=ModelInputEncoder(
            images_rgb=model_input.encoder.images_rgb.to(device),
            images_rgb_unnormalized=model_input.encoder.images_rgb_unnormalized.to(device),
        ),
        decoder=model_input.decoder,
    )

    with torch.no_grad():
        prediction, camera_seconds = timed_inference_call(
            device,
            lambda: predict_cameras_and_depths(model, model_input),
        )
    model_input = ModelInput(
        encoder=model_input.encoder,
        decoder=ModelInputDecoder(
            cam_view=prediction["cam_view"],
            intrinsics=prediction["intrinsics"],
        ),
    )

    if args.use_tto:
        gaussians, reconstruction_seconds = timed_inference_call(
            device,
            lambda: model.forward_tto_memory(
                model_input,
                n_steps=args.tto_n_steps,
                lr=args.tto_lr,
                lpips_weight=args.tto_lpips_weight,
                save_steps=tto_steps,
            ),
        )
    else:
        with torch.inference_mode():
            gaussians, reconstruction_seconds = timed_inference_call(
                device,
                lambda: model.forward_gaussians(model_input),
            )

    with torch.inference_mode():
        rendered, render_seconds = timed_inference_call(
            device,
            lambda: model.render_gaussians(gaussians, model_input.decoder),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    input_dir = os.path.join(args.output_dir, "input_frames")
    render_dir = os.path.join(args.output_dir, "rendered")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    for index, (_, source) in enumerate(sources):
        save_rgb_tensor_image(
            model_input.encoder.images_rgb_unnormalized[0, index],
            os.path.join(args.output_dir, preprocessed_input_frame_path(index, source)),
        )
    for index, image in enumerate(rendered["images_pred"][0].detach().cpu()):
        pixels = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(os.path.join(render_dir, f"render_view{index}.png"))

    if args.save_predicted_input_cameras:
        save_predicted_input_cameras(prediction, sources, args.output_dir)
    if args.save_vggt_input_depths:
        save_vggt_input_depths(prediction, sources, args.output_dir)
    if args.save_vggt_depth_pointcloud:
        save_vggt_depth_pointcloud(
            prediction,
            model_input.encoder.images_rgb_unnormalized,
            sources,
            args.output_dir,
            args.vggt_depth_pointcloud_target_points,
        )

    save_colored_pointcloud(
        gaussians,
        os.path.join(args.output_dir, "pointcloud.ply"),
        options.gaussian_save_opacity_threshold,
        options,
    )
    if args.save_gaussian_alpha_distribution:
        save_gaussian_alpha_distribution(gaussians, args.output_dir)
    if args.save_gaussian_scale_distribution:
        save_gaussian_scale_distribution(
            gaussians,
            args.output_dir,
            max_scale=options.gaussian_scale_cap,
        )

    renderer = GaussianRenderer(options)
    suffix_thresholds = len(thresholds) > 1

    def save_plys(value: torch.Tensor, stem: str) -> list[str]:
        saved = []
        base, extension = os.path.splitext(stem)
        for threshold in thresholds:
            name = (
                f"{base}{opacity_threshold_suffix(threshold)}{extension}"
                if suffix_thresholds
                else stem
            )
            renderer.save_ply(value, os.path.join(args.output_dir, name), threshold)
            saved.append(name)
        return saved

    saved_plys = save_plys(gaussians, "gaussians.ply")
    for step in tto_steps:
        captured = getattr(model, "last_tto_step_gaussians", {}).get(step)
        if captured is None:
            raise RuntimeError(f"Requested TTO step {step} was not captured")
        saved_plys.extend(save_plys(captured, f"gaussians_tto_step{step:04d}.ply"))

    timing = {
        "device": str(device),
        "num_input_views": len(image_paths),
        "camera_and_depth_seconds": camera_seconds,
        "image_to_3dgs_seconds": reconstruction_seconds,
        "render_seconds": render_seconds,
        "use_tto": args.use_tto,
        "tto_n_steps": args.tto_n_steps if args.use_tto else 0,
        "tto_lr": args.tto_lr if args.use_tto else None,
        "tto_lpips_weight": args.tto_lpips_weight if args.use_tto else None,
        "tto_save_steps": tto_steps,
        "gaussian_save_opacity_thresholds": thresholds,
    }
    save_inference_timing(timing, args.output_dir)
    if args.use_tto:
        save_tto_losses(model.last_tto_losses, args.output_dir)
    with open(os.path.join(args.output_dir, "camera_info.txt"), "w", encoding="utf-8") as stream:
        stream.write("Input and render cameras are predicted by the frozen VGGT-Omega camera head.\n")
        stream.write(f"num_input_views: {len(image_paths)}\n")

    print(f"Results saved to {args.output_dir}")
    print(f"3DGS files: {', '.join(saved_plys)}")
    print(
        f"Timing: camera/depth={camera_seconds:.6f}s, "
        f"image-to-3DGS={reconstruction_seconds:.6f}s, render={render_seconds:.6f}s"
    )


if __name__ == "__main__":
    main()
