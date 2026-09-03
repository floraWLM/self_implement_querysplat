"""Image-only batch contract between the TokenGS loader and QuerySplat."""

from __future__ import annotations

from dataclasses import dataclass

import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class QuerySplatImageBatch:
    """The three image tensors consumed by QuerySplat training.

    Tensors may be unbatched ``[V, 3, H, W]`` or batched
    ``[B, V, 3, H, W]``.  Only the input views are normalized because the
    frozen VGM and reconstruction supervision both consume RGB in ``[0, 1]``.
    """

    input_raw: torch.Tensor
    input_normalized: torch.Tensor
    supervision_images: torch.Tensor


def split_querysplat_images(
    images_all: torch.Tensor,
    num_input_views: int,
) -> QuerySplatImageBatch:
    """Split ordered loader images and normalize the QuerySplat RGB branch."""
    if images_all.ndim not in (4, 5):
        raise ValueError(
            "images_all must have shape [V, 3, H, W] or [B, V, 3, H, W], "
            f"got {tuple(images_all.shape)}"
        )
    if not images_all.is_floating_point():
        raise TypeError(f"images_all must be floating point, got {images_all.dtype}")

    view_dim = images_all.ndim - 4
    channel_dim = view_dim + 1
    num_views = images_all.shape[view_dim]
    if images_all.shape[channel_dim] != 3:
        raise ValueError(f"images_all must contain RGB images, got {tuple(images_all.shape)}")
    if not 0 < num_input_views < num_views:
        raise ValueError(
            f"num_input_views must be in [1, {num_views - 1}], got {num_input_views}"
        )
    if not torch.isfinite(images_all).all():
        raise ValueError("images_all contains NaN or Inf")
    value_min = float(images_all.min())
    value_max = float(images_all.max())
    if value_min < 0.0 or value_max > 1.0:
        raise ValueError(
            f"images_all must be in [0, 1], observed [{value_min}, {value_max}]"
        )

    input_raw = images_all.narrow(view_dim, 0, num_input_views).contiguous()
    supervision_images = images_all.narrow(
        view_dim, num_input_views, num_views - num_input_views
    ).contiguous()

    stats_shape = [1] * images_all.ndim
    stats_shape[channel_dim] = 3
    mean = images_all.new_tensor(IMAGENET_MEAN).view(stats_shape)
    std = images_all.new_tensor(IMAGENET_STD).view(stats_shape)
    input_normalized = ((input_raw - mean) / std).contiguous()

    return QuerySplatImageBatch(
        input_raw=input_raw,
        input_normalized=input_normalized,
        supervision_images=supervision_images,
    )
