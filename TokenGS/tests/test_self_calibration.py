import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.models.vggt_encoder import VGGTEncoderOutput
from scripts.training.self_calibration import (
    CameraOnlyOutput,
    Sim3Transform,
    apply_sim3_to_cameras,
    camera_alignment_metrics,
    c2w_to_cam_view,
    estimate_sim3_from_cameras,
    forward_self_calibrated_vgm,
)


def rotation_z(angle: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
        [cosine, -sine, zero, sine, cosine, zero, zero, zero, one], dim=-1
    ).reshape(*angle.shape, 3, 3)


class FakeTwoPassEncoder(nn.Module):
    def __init__(self, input_cam_view, all_cam_view):
        super().__init__()
        self.opt = SimpleNamespace(vggt_intermediate_layers=(4, 11, 17, 23))
        self.aggregator = SimpleNamespace(depth=24)
        self.input_cam_view = input_cam_view
        self.all_cam_view = all_cam_view
        self.input_calls = 0
        self.all_view_calls = 0

    def _run_aggregator_with_images(self, images):
        self.input_calls += 1
        outputs = [None] * 24
        for index in self.opt.vggt_intermediate_layers:
            outputs[index] = torch.tensor([1.0])
        return outputs, 17, images

    def _run_selected_layers(self, images, layer_indices):
        self.all_view_calls += 1
        return {23: torch.tensor([2.0])}, 17

    def _tokens_from_layers(self, output_by_layer, patch_start, batch_size):
        views = self.input_cam_view.shape[1]
        return VGGTEncoderOutput(
            tokens=torch.ones(batch_size, views, 8),
            eye_token=torch.ones(batch_size, 8),
            layer_weights=torch.ones(batch_size, 4) / 4,
            camera_tokens=torch.ones(batch_size, views, 1, 8),
            scene_tokens=torch.ones(batch_size, views, 1, 8),
            tokens_per_view=1,
            special_tokens_per_view=1,
        )

    def _decode_cameras(self, final_layer, image_hw):
        cam_view = self.input_cam_view if float(final_layer[0]) == 1.0 else self.all_cam_view
        batch, views = cam_view.shape[:2]
        return cam_view, torch.ones(batch, views, 4), torch.ones(batch, views, 9)

    def _decode_depths(self, output_list, patch_start, prepared_images, image_hw):
        batch, views = prepared_images.shape[:2]
        height, width = image_hw
        shape = (batch, views, 1, height, width)
        return torch.ones(shape), torch.ones(shape)


class TestSelfCalibration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        batch, views = 2, 6
        source_c2w = torch.eye(4).view(1, 1, 4, 4).repeat(batch, views, 1, 1)
        source_c2w[..., :3, :3] = rotation_z(
            torch.linspace(-0.3, 0.4, views).repeat(batch, 1)
        )
        centers = torch.tensor(
            [
                [-1.0, 0.1, 0.2],
                [-0.4, 0.5, -0.1],
                [0.2, -0.3, 0.4],
                [0.8, 0.6, 0.1],
                [1.2, -0.2, -0.4],
                [1.7, 0.3, 0.5],
            ]
        )
        source_c2w[..., :3, 3] = centers.unsqueeze(0) + torch.tensor(
            [[[0.0, 0.0, 0.0]], [[0.3, -0.2, 0.1]]]
        )
        self.source_cam_view = c2w_to_cam_view(source_c2w)
        self.expected = Sim3Transform(
            scale=torch.tensor([2.5, 0.7]),
            rotation=rotation_z(torch.tensor([0.6, -0.45])),
            translation=torch.tensor([[0.4, -1.2, 2.0], [-0.3, 0.8, -0.5]]),
        )
        self.target_cam_view = apply_sim3_to_cameras(self.source_cam_view, self.expected)

    def test_recovers_known_sim3_from_shared_cameras(self):
        shared_views = 4
        estimated = estimate_sim3_from_cameras(
            self.source_cam_view[:, :shared_views],
            self.target_cam_view[:, :shared_views],
        )

        torch.testing.assert_close(estimated.scale, self.expected.scale, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            estimated.rotation, self.expected.rotation, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            estimated.translation, self.expected.translation, atol=1e-5, rtol=1e-5
        )

        aligned = apply_sim3_to_cameras(self.source_cam_view, estimated)
        torch.testing.assert_close(aligned, self.target_cam_view, atol=2e-5, rtol=2e-5)
        metrics = camera_alignment_metrics(
            aligned[:, :shared_views], self.target_cam_view[:, :shared_views]
        )
        torch.testing.assert_close(metrics.center_rmse, torch.zeros(2), atol=2e-5, rtol=0)
        self.assertTrue(bool((metrics.rotation_max_degrees < 0.05).all()))

    def test_rejects_zero_camera_baseline(self):
        repeated = self.source_cam_view[:, :1].repeat(1, 2, 1, 1)
        with self.assertRaisesRegex(ValueError, "insufficient baseline"):
            estimate_sim3_from_cameras(repeated, repeated)

    def test_two_pass_contract_keeps_all_view_output_camera_only(self):
        input_views = 4
        model = FakeTwoPassEncoder(
            self.target_cam_view[:, :input_views], self.source_cam_view
        )
        images_all = torch.rand(2, 6, 3, 8, 8)
        result = forward_self_calibrated_vgm(
            model,
            input_images=images_all[:, :input_views],
            images_all=images_all,
            image_hw=(8, 8),
        )

        self.assertEqual(model.input_calls, 1)
        self.assertEqual(model.all_view_calls, 1)
        self.assertIsInstance(result.all_view_cameras, CameraOnlyOutput)
        self.assertEqual(
            set(result.all_view_cameras.__dataclass_fields__),
            {"cam_view", "intrinsics", "pose_enc"},
        )
        torch.testing.assert_close(
            result.aligned_all_cam_view, self.target_cam_view, atol=2e-5, rtol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
