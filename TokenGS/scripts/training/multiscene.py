"""DL3DV batch adaptation and DDP-safe QuerySplat training wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.models.querysplat import QuerySplat
from scripts.training.dual_branch_forward import (
    DualBranchForwardOutput,
    forward_querysplat_training,
)
from tokengs.data.provider import Provider
from tokengs.data.querysplat_images import (
    QuerySplatImageBatch,
    split_querysplat_images,
)


@dataclass
class DL3DVProviderOptions:
    """Minimal TokenGS Provider protocol without importing its Tyro CLI."""

    evaluating: bool = False
    dataset_kwargs: dict[str, str] | None = None
    num_input_views: int = 4
    num_views: int = 8
    img_size: tuple[int, int] = (512, 512)
    batch_size: int = 1
    seed: int = 42
    random_reflect: bool = True
    camera_normalization_method: str = "first_cam"
    camera_scale_method: str = "constant"
    pointmap_trim_lo: float = 0.0
    pointmap_trim_hi: float = 1.0
    time_embedding: bool = False
    time_embedding_dim: int = 2
    use_interp_target: bool = False


@dataclass(frozen=True)
class DL3DVLoaderBundle:
    dataloader: DataLoader
    dataset: Provider
    sampler: DistributedSampler


class QuerySplatTrainingModule(nn.Module):
    """Put the complete two-pass graph inside DDP's forward boundary."""

    def __init__(self, model: QuerySplat):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_normalized: torch.Tensor,
        input_raw: torch.Tensor,
        images_all: torch.Tensor,
    ) -> DualBranchForwardOutput:
        return forward_querysplat_training(
            self.model,
            input_normalized=input_normalized,
            input_raw=input_raw,
            images_all=images_all,
        )


def build_dl3dv_dataloader(
    data_root: str | Path,
    *,
    num_input_views: int,
    num_supervision_views: int,
    image_size: int,
    num_workers: int,
    seed: int,
    rank: int,
    world_size: int,
) -> DL3DVLoaderBundle:
    """Build one rank's loader while keeping TokenGS view sampling unchanged."""
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"DL3DV root not found: {data_root}")
    if num_input_views < 2 or num_supervision_views < 1:
        raise ValueError("training requires at least two input and one supervision view")
    if image_size <= 0 or num_workers < 0:
        raise ValueError("image_size must be positive and num_workers non-negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must lie in [0, world_size)")

    options = DL3DVProviderOptions(
        evaluating=False,
        dataset_kwargs={"root_path": str(data_root)},
        num_input_views=num_input_views,
        num_views=num_input_views + num_supervision_views,
        img_size=(image_size, image_size),
        batch_size=1,
        seed=seed + rank,
        random_reflect=True,
    )
    dataset = Provider("dl3dv_scaled_1.0", options, training=True)
    if len(dataset) < world_size:
        raise ValueError(
            f"training split has {len(dataset)} scenes but world_size is {world_size}; "
            "DistributedSampler(drop_last=True) would leave ranks empty"
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed + rank),
    )
    if len(dataloader) == 0:
        raise ValueError("DL3DV dataloader is empty")
    return DL3DVLoaderBundle(dataloader=dataloader, dataset=dataset, sampler=sampler)


def set_dl3dv_epoch(bundle: DL3DVLoaderBundle, epoch: int) -> None:
    """Reseed both scene shuffling and TokenGS frame/view sampling."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    bundle.sampler.set_epoch(epoch)
    bundle.dataset.set_rng_epoch(epoch)


def prepare_querysplat_images(
    loader_batch: dict[str, Any],
    num_input_views: int,
    *,
    validate_provider_contract: bool = False,
) -> QuerySplatImageBatch:
    """Use images_all as the single source of truth for both training branches."""
    if "images_all" not in loader_batch:
        raise KeyError("TokenGS loader batch is missing images_all")
    images = split_querysplat_images(loader_batch["images_all"], num_input_views)
    if validate_provider_contract:
        required = {"images_input", "images_output", "input"}
        missing = required - set(loader_batch)
        if missing:
            raise KeyError(f"TokenGS loader batch is missing fields: {sorted(missing)}")
        torch.testing.assert_close(images.input_raw, loader_batch["images_input"])
        torch.testing.assert_close(images.supervision_images, loader_batch["images_output"])
        torch.testing.assert_close(
            images.input_normalized,
            loader_batch["input"][:, :num_input_views, :3],
        )
    return images
