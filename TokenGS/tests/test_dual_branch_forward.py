import unittest

import torch

from scripts.models.vggt_encoder import VGGTEncoderOutput
from scripts.training.dual_branch_forward import decode_and_render_from_self_calibration
from scripts.training.self_calibration import (
    CameraAlignmentMetrics,
    CameraOnlyOutput,
    SelfCalibratedVGMOutput,
    Sim3Transform,
)
from scripts.training.vggt_input_pass import VGGTInputPassOutput


class FakeBackbone:
    def __init__(self, owner):
        self.owner = owner

    def _encode_to_kv(self, features, run_encoder=False):
        self.owner.geometry_features = features
        return features + 1, features + 2


class FakeQuerySplat:
    def __init__(self):
        self.enc_dec_backbone = FakeBackbone(self)

    def _plucker_from_vggt_cameras(self, cam_view, intrinsics, dtype, device):
        self.plucker_cam_view = cam_view
        self.plucker_intrinsics = intrinsics
        batch, views = cam_view.shape[:2]
        return torch.ones(batch, views, 6, 2, 2, dtype=dtype, device=device)

    def _original_encoder_tokens(self, encoder_input, plucker):
        self.appearance_images = encoder_input.images_rgb
        self.appearance_raw_images = encoder_input.images_rgb_unnormalized
        return encoder_input.images_rgb.mean(dim=(-2, -1))

    def _original_tokens_to_kv(self, tokens):
        return tokens + 3, tokens + 4

    def forward_decoder(self, latent):
        self.latent = latent
        return latent.keys.mean().expand(latent.keys.shape[0], 2, 15)

    def render_gaussians(self, gaussians, decoder):
        self.render_decoder = decoder
        batch, views = decoder.cam_view.shape[:2]
        return {"images_pred": gaussians.mean().expand(batch, views, 3, 2, 2)}


def make_self_calibration(batch=1, input_views=2, all_views=4):
    input_cam_view = torch.eye(4).view(1, 1, 4, 4).repeat(batch, input_views, 1, 1)
    all_cam_view = torch.eye(4).view(1, 1, 4, 4).repeat(batch, all_views, 1, 1)
    all_cam_view[:, input_views:, 3, 0] = torch.tensor([0.5, 1.0])
    geometry = VGGTEncoderOutput(
        tokens=torch.full((batch, 5, 8), 7.0),
        eye_token=torch.ones(batch, 8),
        layer_weights=torch.ones(batch, 4) / 4,
        camera_tokens=torch.ones(batch, input_views, 1, 8),
        scene_tokens=torch.ones(batch, input_views, 1, 8),
        tokens_per_view=1,
        special_tokens_per_view=1,
    )
    input_pass = VGGTInputPassOutput(
        geometry=geometry,
        cam_view=input_cam_view,
        intrinsics=torch.full((batch, input_views, 4), 10.0),
        pose_enc=torch.ones(batch, input_views, 9),
        depth=torch.ones(batch, input_views, 1, 2, 2),
        depth_confidence=torch.ones(batch, input_views, 1, 2, 2),
    )
    return SelfCalibratedVGMOutput(
        input_pass=input_pass,
        all_view_cameras=CameraOnlyOutput(
            cam_view=all_cam_view,
            intrinsics=torch.full((batch, all_views, 4), 20.0),
            pose_enc=torch.ones(batch, all_views, 9),
        ),
        aligned_all_cam_view=all_cam_view,
        sim3_all_to_input=Sim3Transform(
            scale=torch.ones(batch),
            rotation=torch.eye(3).repeat(batch, 1, 1),
            translation=torch.zeros(batch, 3),
        ),
        shared_input_metrics=CameraAlignmentMetrics(
            center_rmse=torch.zeros(batch),
            rotation_mean_degrees=torch.zeros(batch),
            rotation_max_degrees=torch.zeros(batch),
        ),
    )


class TestDualBranchForward(unittest.TestCase):
    def test_branches_use_input_only_and_render_supervision_only(self):
        model = FakeQuerySplat()
        images_all = torch.rand(1, 4, 3, 2, 2)
        input_raw = images_all[:, :2]
        input_normalized = input_raw * 2 - 1
        calibration = make_self_calibration()

        output = decode_and_render_from_self_calibration(
            model,
            input_normalized=input_normalized,
            input_raw=input_raw,
            images_all=images_all,
            self_calibration=calibration,
        )

        torch.testing.assert_close(model.geometry_features, calibration.input_pass.geometry.tokens)
        torch.testing.assert_close(model.appearance_images, input_normalized)
        torch.testing.assert_close(model.appearance_raw_images, input_raw)
        torch.testing.assert_close(model.plucker_cam_view, calibration.input_pass.cam_view)
        torch.testing.assert_close(model.plucker_intrinsics, calibration.input_pass.intrinsics)
        torch.testing.assert_close(
            output.supervision_decoder.cam_view, calibration.aligned_all_cam_view[:, 2:]
        )
        torch.testing.assert_close(
            output.supervision_decoder.intrinsics,
            calibration.all_view_cameras.intrinsics[:, 2:],
        )
        torch.testing.assert_close(output.supervision_images, images_all[:, 2:])
        self.assertEqual(output.gaussians.shape, (1, 2, 15))
        self.assertEqual(output.render_results["images_pred"].shape, (1, 2, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
