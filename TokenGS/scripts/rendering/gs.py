# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable Gaussian rendering and PLY export for inference/TTO."""

import numpy as np
import torch
from gsplat.rendering import rasterization


class GaussianRenderer:
    def __init__(self, opt):
        self.opt = opt

    def _sh_degree(self) -> int:
        return int(self.opt.gaussian_sh_degree)

    def render(self, gaussians, cam_view, bg_color, intrinsics):
        batch, views = cam_view.shape[:2]
        means = gaussians[..., 0:3].contiguous().float()
        opacity = gaussians[..., 3].contiguous().float()
        scales = gaussians[..., 4:7].contiguous().float()
        rotations = gaussians[..., 7:11].contiguous().float()
        colors = gaussians[..., 11:].contiguous().float()
        colors = colors.reshape(*colors.shape[:-1], -1, 3)

        viewmats = cam_view.float().transpose(3, 2)
        intrinsics = intrinsics.to(device=means.device, dtype=means.dtype)
        backgrounds = bg_color[None, None].repeat(batch, views, 1).to(means)
        height, width = self.opt.img_size
        camera_matrices = torch.zeros(
            (*intrinsics.shape[:2], 3, 3),
            dtype=means.dtype,
            device=means.device,
        )
        camera_matrices[..., 0, 0] = intrinsics[..., 0]
        camera_matrices[..., 1, 1] = intrinsics[..., 1]
        camera_matrices[..., 0, 2] = intrinsics[..., 2]
        camera_matrices[..., 1, 2] = intrinsics[..., 3]
        camera_matrices[..., 2, 2] = 1.0

        images = []
        alphas = []
        depths = []
        means2d = []
        for index in range(batch):
            rendered, alpha, info = rasterization(
                means=means[index],
                quats=rotations[index],
                scales=scales[index],
                opacities=opacity[index],
                colors=colors[index],
                viewmats=viewmats[index],
                Ks=camera_matrices[index],
                width=width,
                height=height,
                near_plane=self.opt.znear,
                far_plane=self.opt.zfar,
                packed=False,
                backgrounds=backgrounds[index],
                render_mode="RGB+ED",
                sh_degree=self._sh_degree(),
            )
            for image, alpha_map, projected in zip(rendered, alpha, info["means2d"]):
                images.append(image[..., :3].permute(2, 0, 1))
                depths.append(image[..., 3:].permute(2, 0, 1))
                alphas.append(alpha_map.permute(2, 0, 1))
                means2d.append(projected)
        return {
            "images_pred": torch.stack(images).view(batch, views, 3, height, width),
            "alphas_pred": torch.stack(alphas).view(batch, views, 1, height, width),
            "depths_pred": torch.stack(depths).view(batch, views, 1, height, width),
            "means2d_pred": torch.stack(means2d).view(batch, views, *means2d[0].shape),
        }

    def save_ply(self, gaussians, path, opacity_threshold: float):
        if gaussians.shape[0] != 1:
            raise ValueError("PLY export supports batch size one")
        from plyfile import PlyData, PlyElement

        means = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        sh = gaussians[0, :, 11:].contiguous().float().reshape(means.shape[0], -1, 3)
        keep = opacity.squeeze(-1) >= opacity_threshold
        means, opacity, scales, rotations, sh = (
            value[keep] for value in (means, opacity, scales, rotations, sh)
        )

        opacity = torch.logit(opacity.clamp(1e-8, 1 - 1e-8))
        scales = torch.log(scales + 1e-8)
        xyz = means.detach().cpu().numpy()
        f_dc = sh[:, :1].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = sh[:, 1:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacity_np = opacity.detach().cpu().numpy()
        scale_np = scales.detach().cpu().numpy()
        rotation_np = rotations.detach().cpu().numpy()

        names = ["x", "y", "z"]
        names += [f"f_dc_{index}" for index in range(f_dc.shape[1])]
        names += [f"f_rest_{index}" for index in range(f_rest.shape[1])]
        names += ["opacity"]
        names += [f"scale_{index}" for index in range(scale_np.shape[1])]
        names += [f"rot_{index}" for index in range(rotation_np.shape[1])]
        elements = np.empty(xyz.shape[0], dtype=[(name, "f4") for name in names])
        attributes = np.concatenate(
            (xyz, f_dc, f_rest, opacity_np, scale_np, rotation_np),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
