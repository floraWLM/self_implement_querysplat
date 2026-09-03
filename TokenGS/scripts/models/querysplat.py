# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only QuerySplat graph for the released VGGT-Omega checkpoint."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scripts.models.activations import PartialClipActivationHead, effective_gaussian_sh_rest_scale
from scripts.models.attention import PatchEmbed
from scripts.models.enc_dec import DecoderBlock, EncDecBackbone
from scripts.models.input_types import EncoderLatent, ModelInput, ModelInputDecoder, ModelInputEncoder
from scripts.models.vggt_encoder import VGGTEncoder
from scripts.options import Options
from scripts.rendering.fused_ssim import fused_ssim
from scripts.rendering.gs import GaussianRenderer
from scripts.utils.data import get_rays


@contextmanager
def _frozen_parameters(model: nn.Module) -> Iterator[None]:
    flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return torch.amp.autocast(device_type=device.type, enabled=False)


class QuerySplat(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.img_size = tuple(opt.img_size)
        norm_layer = partial(nn.LayerNorm, bias=True)

        self.patch_embed = PatchEmbed(
            img_size=opt.img_size,
            patch_size=opt.vggt_original_encoder_patch_size,
            in_chans=3,
            embed_dim=opt.enc_embed_dim,
            norm_layer=norm_layer,
        )
        self.patch_plucker_embed = PatchEmbed(
            img_size=opt.img_size,
            patch_size=opt.vggt_original_encoder_patch_size,
            in_chans=6,
            embed_dim=opt.enc_embed_dim,
            norm_layer=norm_layer,
        )
        self.vggt_encoder = VGGTEncoder(opt)
        self.enc_dec_backbone = EncDecBackbone(opt)
        self.original_encoder_norm = nn.LayerNorm(opt.enc_embed_dim)
        self.original_kv_proj = nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim * 2, bias=True)
        self.original_k_proj_norm = nn.LayerNorm(opt.enc_embed_dim // opt.enc_num_heads)
        self.gs = GaussianRenderer(opt)

        self.post_output_params = tuple(opt.vggt_original_encoder_post_output_params)
        self.color_dim = 3 * (opt.gaussian_sh_degree + 1) ** 2
        self.post_output_dim = self.color_dim + 1
        self.non_rgb_activation_head = PartialClipActivationHead(opt)
        self.post_rgb_gs_tokens = nn.Parameter(
            torch.zeros(opt.num_gs_tokens, opt.enc_embed_dim)
        )
        self.post_rgb_decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    opt.enc_embed_dim,
                    opt.enc_num_heads,
                    opt.mlp_ratio,
                    qkv_bias=True,
                    ffn_bias=True,
                    init_values=5e-3 * opt.gs_token_std,
                    qk_norm=True,
                )
                for _ in range(opt.vggt_original_encoder_post_decoder_depth)
            ]
        )
        self.post_rgb_delta_head = nn.Linear(
            opt.enc_embed_dim,
            self.post_output_dim * opt.num_gaussians_per_token,
            bias=True,
        )
        self.gs_tokens = nn.Parameter(
            opt.gs_token_std * torch.randn(opt.num_gs_tokens, opt.token_dim)
        )

    def _background_color(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.opt.bg_color == "white":
            return torch.ones(3, dtype=dtype, device=device)
        if self.opt.bg_color == "black":
            return torch.zeros(3, dtype=dtype, device=device)
        if self.opt.bg_color == "grey":
            return torch.full((3,), 0.5, dtype=dtype, device=device)
        raise ValueError(f"Invalid background color: {self.opt.bg_color}")

    @staticmethod
    def _cam_view_to_c2w(cam_view: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(cam_view.transpose(-2, -1))

    def _plucker_from_vggt_cameras(
        self,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        c2w = self._cam_view_to_c2w(cam_view).to(device=device, dtype=dtype)
        intrinsics = intrinsics.to(device=device, dtype=dtype)
        height, width = self.img_size
        rays_o, rays_d = get_rays(intrinsics, c2w, height, width, device=device)
        batch, views = intrinsics.shape[:2]
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
        return plucker.reshape(batch, views, height, width, 6).permute(0, 1, 4, 2, 3).contiguous()

    def _original_encoder_tokens(
        self,
        encoder_input: ModelInputEncoder,
        plucker: torch.Tensor,
    ) -> torch.Tensor:
        images = encoder_input.images_rgb
        batch, views, _, height, width = images.shape
        patch_size = self.opt.vggt_original_encoder_patch_size
        if height % patch_size or width % patch_size:
            raise ValueError(f"Image size {(height, width)} is not divisible by {patch_size}")
        rgb_tokens = self.patch_embed(images.reshape(batch * views, 3, height, width))
        ray_tokens = self.patch_plucker_embed(plucker.reshape(batch * views, 6, height, width))
        tokens = rgb_tokens + ray_tokens
        return rearrange(tokens, "(b v) n c -> b (v n) c", b=batch, v=views)

    def _original_tokens_to_kv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.original_encoder_norm(tokens)
        keys, values = rearrange(
            self.original_kv_proj(tokens),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        return self.original_k_proj_norm(keys), values

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        vggt_output, cam_view, intrinsics, _ = self.vggt_encoder.forward_with_cameras(
            encoder_input.images_rgb_unnormalized,
            image_hw=self.img_size,
        )
        keys, values = self.enc_dec_backbone._encode_to_kv(
            vggt_output.tokens,
            run_encoder=False,
        )
        plucker = self._plucker_from_vggt_cameras(
            cam_view,
            intrinsics,
            dtype=encoder_input.images_rgb.dtype,
            device=encoder_input.images_rgb.device,
        )
        rgb_tokens = self._original_encoder_tokens(encoder_input, plucker)
        post_rgb_keys, post_rgb_values = self._original_tokens_to_kv(rgb_tokens)
        return EncoderLatent(
            keys=keys,
            values=values,
            post_rgb_keys=post_rgb_keys,
            post_rgb_values=post_rgb_values,
            eye_token=vggt_output.eye_token,
            layer_weights=vggt_output.layer_weights,
        )

    def get_gs_tokens(self, batch_size: int) -> torch.Tensor:
        return self.gs_tokens.unsqueeze(0).repeat(batch_size, 1, 1)

    def _forward_decoder_tokens(self, latent: EncoderLatent) -> torch.Tensor:
        tokens = self.get_gs_tokens(latent.keys.shape[0])
        if latent.eye_token is not None:
            tokens = torch.cat([latent.eye_token.unsqueeze(1), tokens], dim=1)
        for layer in self.enc_dec_backbone.decoder_blocks:
            tokens = layer(tokens, latent.keys, latent.values)
        if latent.eye_token is not None:
            tokens = tokens[:, 1:]
        return tokens

    def _decode_partial_gaussians(self, tokens: torch.Tensor) -> torch.Tensor:
        gaussians = self.non_rgb_activation_head(tokens, current_step=None)
        gaussians[..., 2] += self.opt.gaussian_z_offset
        return gaussians

    def _post_output(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        output = self.post_rgb_delta_head(tokens)
        output = rearrange(
            output,
            "b n (p c) -> b c (n p)",
            p=self.opt.num_gaussians_per_token,
            c=self.post_output_dim,
        )
        return output.reshape(batch, self.post_output_dim, -1).permute(0, 2, 1).contiguous()

    def _activate_post_output(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_color = raw[..., : self.color_dim]
        sh_basis_count = (self.opt.gaussian_sh_degree + 1) ** 2
        sh = raw_color.reshape(*raw_color.shape[:-1], sh_basis_count, 3)
        dc_rgb = 0.5 * torch.tanh(sh[..., 0, :]) + 0.5
        dc = (dc_rgb - 0.5) / 0.28209479177387814
        rest = effective_gaussian_sh_rest_scale(self.opt, None) * torch.tanh(sh[..., 1:, :])
        color = torch.cat([dc.unsqueeze(-2), rest], dim=-2).flatten(start_dim=-2)
        opacity = torch.sigmoid(raw[..., self.color_dim : self.color_dim + 1] - 2.0)
        return color, opacity

    def forward_decoder(self, latent: EncoderLatent) -> torch.Tensor:
        geometry_tokens = self._forward_decoder_tokens(latent)
        partial = self._decode_partial_gaussians(geometry_tokens)
        color_tokens = geometry_tokens
        if self.opt.vggt_original_encoder_post_detach_query:
            color_tokens = color_tokens.detach()
        if self.opt.vggt_original_encoder_post_use_color_tokens:
            color_tokens = color_tokens + self.post_rgb_gs_tokens.unsqueeze(0).to(
                device=color_tokens.device,
                dtype=color_tokens.dtype,
            )
        for layer in self.post_rgb_decoder_blocks:
            color_tokens = layer(color_tokens, latent.post_rgb_keys, latent.post_rgb_values)
        color, opacity = self._activate_post_output(self._post_output(color_tokens))
        position, scale, rotation = partial.split([3, 3, 4], dim=-1)
        return torch.cat([position, opacity, scale, rotation, color], dim=-1)

    def forward_gaussians(self, model_input: ModelInput) -> torch.Tensor:
        return self.forward_decoder(self.forward_encoder(model_input.encoder))

    def render_gaussians(self, gaussians: torch.Tensor, decoder: ModelInputDecoder) -> dict:
        return self.gs.render(
            gaussians,
            decoder.cam_view,
            bg_color=self._background_color(gaussians.dtype, gaussians.device),
            intrinsics=decoder.intrinsics,
        )

    def forward_tto_memory(
        self,
        model_input: ModelInput,
        n_steps: int,
        lr: float,
        lpips_weight: float,
        save_steps: list[int],
    ) -> torch.Tensor:
        if n_steps < 0:
            raise ValueError("--tto_n_steps must be non-negative")
        if lpips_weight > 0:
            self._ensure_lpips(model_input.encoder.images_rgb_unnormalized.device)
        with _frozen_parameters(self):
            return self._optimize_memory(model_input, n_steps, lr, lpips_weight, set(save_steps))

    def _optimize_memory(
        self,
        model_input: ModelInput,
        n_steps: int,
        lr: float,
        lpips_weight: float,
        save_steps: set[int],
    ) -> torch.Tensor:
        with torch.no_grad():
            latent = self.forward_encoder(model_input.encoder)
        geo_keys = latent.keys.detach().requires_grad_(True)
        geo_values = latent.values.detach().requires_grad_(True)
        rgb_keys = latent.post_rgb_keys.detach().requires_grad_(True)
        rgb_values = latent.post_rgb_values.detach().requires_grad_(True)
        eye_token = latent.eye_token.detach() if latent.eye_token is not None else None
        layer_weights = latent.layer_weights.detach() if latent.layer_weights is not None else None
        ground_truth = model_input.encoder.images_rgb_unnormalized
        self.last_tto_step_gaussians: dict[int, torch.Tensor] = {}

        def decode() -> torch.Tensor:
            current = EncoderLatent(
                keys=geo_keys,
                values=geo_values,
                post_rgb_keys=rgb_keys,
                post_rgb_values=rgb_values,
                eye_token=eye_token,
                layer_weights=layer_weights,
            )
            return self.forward_decoder(current)

        def capture(step: int) -> None:
            if step in save_steps:
                with torch.no_grad():
                    self.last_tto_step_gaussians[step] = decode().detach().cpu()

        capture(0)
        optimizer = torch.optim.Adam([geo_keys, geo_values, rgb_keys, rgb_values], lr=lr)
        with torch.set_grad_enabled(True):
            for index in range(n_steps):
                optimizer.zero_grad()
                gaussians = decode()
                rendered = self.render_gaussians(gaussians, model_input.decoder)
                losses = self._tto_losses(rendered["images_pred"], ground_truth, lpips_weight)
                losses["loss_total"].backward()
                optimizer.step()
                step = index + 1
                lpips_text = "nan" if losses["loss_lpips"] is None else f"{losses['loss_lpips'].item():.6f}"
                print(
                    f"TTO [{step}/{n_steps}] L1={losses['loss_l1'].item():.6f} "
                    f"SSIM={losses['loss_ssim'].item():.6f} LPIPS={lpips_text} "
                    f"total={losses['loss_total'].item():.6f}"
                )
                capture(step)

        with torch.no_grad():
            gaussians = decode()
            rendered = self.render_gaussians(gaussians, model_input.decoder)
            losses = self._tto_losses(rendered["images_pred"], ground_truth, lpips_weight)
            self.last_tto_losses = {
                "L1": float(losses["loss_l1"].cpu()),
                "SSIM": float(losses["loss_ssim"].cpu()),
                "LPIPS": None if losses["loss_lpips"] is None else float(losses["loss_lpips"].cpu()),
            }
        return gaussians

    def _ensure_lpips(self, device: torch.device) -> None:
        if hasattr(self, "lpips_loss"):
            return
        from lpips import LPIPS

        self.lpips_loss = LPIPS(net="vgg").to(device=device)
        self.lpips_loss.requires_grad_(False)
        self.lpips_loss.eval()

    def _tto_losses(
        self,
        predicted: torch.Tensor,
        ground_truth: torch.Tensor,
        lpips_weight: float,
    ) -> dict[str, torch.Tensor | None]:
        height, width = self.img_size
        loss_l1 = F.l1_loss(predicted, ground_truth)
        ssim = fused_ssim(
            predicted.reshape(-1, 3, height, width),
            ground_truth.reshape(-1, 3, height, width),
        )
        loss_ssim = (1 - ssim) / 2
        loss_total = loss_l1 + 0.2 * loss_ssim
        loss_lpips = None
        if lpips_weight > 0:
            with _autocast_disabled(predicted.device):
                loss_lpips = self.lpips_loss(
                    ground_truth.reshape(-1, 3, height, width).float(),
                    predicted.reshape(-1, 3, height, width).float(),
                ).mean()
            loss_total = loss_total + lpips_weight * loss_lpips
        return {
            "loss_total": loss_total,
            "loss_l1": loss_l1,
            "loss_ssim": loss_ssim,
            "loss_lpips": loss_lpips,
        }
