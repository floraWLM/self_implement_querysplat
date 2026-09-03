"""Load one deterministic DL3DV sample through the TokenGS Provider.

This is a data-pipeline smoke test for the QuerySplat reimplementation.  It
materializes ``images_all`` without constructing a model or moving data to a
GPU, so loader and preprocessing failures can be diagnosed independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from tokengs.data.provider import Provider
from tokengs.options import Options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one scene's images_all tensor with the TokenGS DL3DV loader."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing unpacked DL3DV scene directories or canonical split ZIPs.",
    )
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--num-input-views", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dl3dv_loader_smoke"),
    )
    return parser.parse_args()


def save_grid(images: torch.Tensor, path: Path) -> None:
    grid = make_grid(images, nrow=images.shape[0], padding=2)
    pixels = (
        grid.permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(pixels).save(path)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"DL3DV data root does not exist: {data_root}")
    if args.num_input_views < 2:
        raise ValueError("TokenGS static view sampling requires at least two input views")
    if args.num_views < args.num_input_views:
        raise ValueError("--num-views must be at least --num-input-views")

    options = Options(
        evaluating=True,
        dataset_kwargs={"root_path": str(data_root)},
        num_input_views=args.num_input_views,
        num_views=args.num_views,
        img_size=(args.image_size, args.image_size),
        batch_size=1,
        seed=args.seed,
        random_reflect=False,
    )
    provider = Provider("dl3dv_scaled_1.0", options, training=False)
    if not 0 <= args.scene_index < len(provider):
        raise IndexError(
            f"scene index {args.scene_index} is outside [0, {len(provider) - 1}]"
        )

    # Use get_item deliberately: Provider.__getitem__ retries a random scene on
    # failure, which would hide the exact scene and error in this smoke test.
    sample = provider.get_item(args.scene_index)
    images_all = sample["images_all"].detach().cpu().contiguous()

    expected_shape = (args.num_views, 3, args.image_size, args.image_size)
    if tuple(images_all.shape) != expected_shape:
        raise RuntimeError(
            f"images_all has shape {tuple(images_all.shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(images_all).all():
        raise RuntimeError("images_all contains NaN or Inf")
    image_min = float(images_all.min())
    image_max = float(images_all.max())
    if image_min < 0.0 or image_max > 1.0:
        raise RuntimeError(
            f"images_all must be in [0, 1], observed [{image_min}, {image_max}]"
        )

    scene_path = Path(provider.dataset.sample_list[args.scene_index]).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / "images_all.pt"
    grid_path = args.output_dir / "images_all_grid.png"
    metadata_path = args.output_dir / "metadata.json"
    torch.save(images_all, tensor_path)
    save_grid(images_all, grid_path)

    metadata = {
        "data_root": str(data_root),
        "scene_index": args.scene_index,
        "scene_path": str(scene_path),
        "provider_length": len(provider),
        "seed": args.seed,
        "num_input_views": args.num_input_views,
        "num_views": args.num_views,
        "images_all_shape": list(images_all.shape),
        "images_all_dtype": str(images_all.dtype),
        "images_all_min": image_min,
        "images_all_max": image_max,
        "sample_keys": sorted(sample),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"Saved tensor: {tensor_path.resolve()}")
    print(f"Saved grid:   {grid_path.resolve()}")


if __name__ == "__main__":
    main()
