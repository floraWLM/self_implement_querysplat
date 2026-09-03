# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Image preprocessing and camera-ray helpers used by inference."""

import torch
import torchvision.transforms as transforms


class ImageTransform:
    def __init__(self, crop_size, sample_size, max_crop):
        self.crop_size = crop_size
        self.max_crop = max_crop
        self.crop_transform = transforms.CenterCrop(crop_size)
        self.resize_transform = transforms.Resize(sample_size)

    def preprocess_images(self, images):
        original_height, original_width = images.shape[-2:]
        if self.max_crop:
            ratio = min(
                original_height / self.crop_size[0],
                original_width / self.crop_size[1],
            )
            crop_size = (
                int(self.crop_size[0] * ratio),
                int(self.crop_size[1] * ratio),
            )
            self.crop_transform = transforms.CenterCrop(crop_size)
        images = self.crop_transform(images)
        cropped_height, cropped_width = images.shape[-2:]
        shift = (
            (cropped_width - original_width) / 2,
            (cropped_height - original_height) / 2,
        )
        images = self.resize_transform(images)
        resized_height, resized_width = images.shape[-2:]
        scale = (
            resized_width / cropped_width,
            resized_height / cropped_height,
        )
        flip = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
        return images, shift, scale, flip


def get_rays(intrinsics, c2w, height, width, device):
    batch, views = intrinsics.shape[:2]
    y, x = torch.meshgrid(
        torch.linspace(0, height - 1, height, device=device, dtype=intrinsics.dtype),
        torch.linspace(0, width - 1, width, device=device, dtype=intrinsics.dtype),
        indexing="ij",
    )
    x = x.reshape(1, 1, height * width).expand(batch, views, -1) + 0.5
    y = y.reshape(1, 1, height * width).expand(batch, views, -1) + 0.5
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    z = torch.ones_like(x)
    directions = torch.stack(((x - cx) / fx * z, (y - cy) / fy * z, z), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3][..., None, :].expand_as(rays_d)
    return rays_o, rays_d
