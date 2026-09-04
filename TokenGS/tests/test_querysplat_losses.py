import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.training.losses import (
    LinearWeightSchedule,
    QuerySplatLossConfig,
    compute_bidirectional_chamfer_loss,
    compute_opacity_floor_loss,
    compute_photometric_losses,
    compute_querysplat_losses,
    compute_visibility_loss,
    sample_vggt_depth_pointcloud,
)


class FakeLPIPS(nn.Module):
    def forward(self, target, predicted, normalize=False):
        if target.dtype != torch.float32 or predicted.dtype != torch.float32:
            raise AssertionError("LPIPS must run in FP32")
        if not normalize:
            raise AssertionError("LPIPS must normalize [0,1] RGB inputs")
        return (target - predicted).abs().mean(dim=(1, 2, 3), keepdim=True)


def fake_ssim(predicted, target):
    return predicted.new_tensor(0.6)


class TestQuerySplatLosses(unittest.TestCase):
    def test_linear_weight_schedule(self):
        schedule = LinearWeightSchedule(1.0, 0.0, 10, 20)
        self.assertEqual(schedule.value(0), 1.0)
        self.assertEqual(schedule.value(10), 1.0)
        self.assertAlmostEqual(schedule.value(15), 0.5)
        self.assertEqual(schedule.value(20), 0.0)
        instant = LinearWeightSchedule(1.0, 0.0, 5, 5)
        self.assertEqual(instant.value(4), 1.0)
        self.assertEqual(instant.value(5), 0.0)

    def test_photometric_terms_and_fp32_lpips(self):
        predicted = torch.zeros(1, 2, 3, 4, 4, requires_grad=True)
        target = torch.ones_like(predicted)
        losses = compute_photometric_losses(
            predicted,
            target,
            lambda_ssim=0.2,
            lambda_lpips=0.05,
            lpips_model=FakeLPIPS(),
            ssim_function=fake_ssim,
        )
        self.assertAlmostEqual(float(losses["loss_l1"].detach()), 1.0)
        self.assertAlmostEqual(float(losses["loss_ssim"].detach()), 0.2)
        self.assertAlmostEqual(float(losses["loss_lpips"].detach()), 1.0)
        self.assertAlmostEqual(float(losses["loss_photo"].detach()), 1.09, places=5)
        losses["loss_photo"].backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_visibility_penalizes_outside_and_behind_centers(self):
        cam_view = torch.eye(4).view(1, 1, 4, 4)
        intrinsics = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        visible = torch.tensor([[[0.0, 0.0, 1.0]]])
        invalid = torch.tensor(
            [[[4.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 11.0]]]
        )
        self.assertEqual(
            float(
                compute_visibility_loss(
                    visible, cam_view, intrinsics, (2, 2), 0.1, 1.0, zfar=10.0
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                compute_visibility_loss(
                    invalid, cam_view, intrinsics, (2, 2), 0.1, 1.0, zfar=10.0
                )
            ),
            0.0,
        )

    def test_depth_backprojection_and_bidirectional_chamfer(self):
        depth = torch.ones(1, 1, 1, 1, 1)
        confidence = torch.ones_like(depth)
        cam_view = torch.eye(4).view(1, 1, 4, 4)
        intrinsics = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        points = sample_vggt_depth_pointcloud(
            depth, confidence, cam_view, intrinsics, max_points=1
        )
        torch.testing.assert_close(points[0], torch.tensor([[0.5, 0.5, 1.0]]))
        centers = points[0].view(1, 1, 3).clone().requires_grad_(True)
        loss = compute_bidirectional_chamfer_loss(
            centers, points, max_gaussian_points=8, chunk_size=2
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0)
        shifted_loss = compute_bidirectional_chamfer_loss(
            centers + 1.0, points, max_gaussian_points=8, chunk_size=2
        )
        self.assertGreater(float(shifted_loss.detach()), 0.0)
        shifted_loss.backward()
        self.assertTrue(bool(torch.isfinite(centers.grad).all()))

    def test_opacity_floor_matches_log_hinge(self):
        opacity = torch.tensor([0.05, 0.1, 0.2])
        loss = compute_opacity_floor_loss(opacity, opacity_floor=0.1, epsilon=1e-6)
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))) / 3, places=6)

    def test_full_scheduled_objective(self):
        predicted = torch.zeros(1, 1, 3, 2, 2, requires_grad=True)
        target = torch.ones_like(predicted)
        cam_view = torch.eye(4).view(1, 1, 4, 4)
        intrinsics = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        depth = torch.ones(1, 1, 1, 2, 2)
        gaussians = torch.zeros(1, 2, 23)
        gaussians[..., 2] = 1.0
        gaussians[..., 3] = 0.05
        gaussians.requires_grad_(True)
        output = SimpleNamespace(
            render_results={"images_pred": predicted},
            supervision_images=target,
            gaussians=gaussians,
            supervision_decoder=SimpleNamespace(cam_view=cam_view, intrinsics=intrinsics),
            self_calibration=SimpleNamespace(
                input_pass=SimpleNamespace(
                    cam_view=cam_view,
                    intrinsics=intrinsics,
                    depth=depth,
                    depth_confidence=torch.ones_like(depth),
                )
            ),
        )
        config = QuerySplatLossConfig(
            lambda_ssim=0.2,
            lambda_visibility=1.0,
            lpips_schedule=LinearWeightSchedule(0.0, 0.05, 0, 10),
            chamfer_schedule=LinearWeightSchedule(1.0, 0.0, 0, 10),
            opacity_schedule=LinearWeightSchedule(0.1, 0.0, 0, 10),
            max_depth_points=4,
            max_gaussian_chamfer_points=2,
            chamfer_chunk_size=2,
        )
        losses = compute_querysplat_losses(
            output,
            step=5,
            config=config,
            znear=0.1,
            lpips_model=FakeLPIPS(),
            ssim_function=fake_ssim,
        )
        expected_keys = {
            "loss", "loss_photo", "loss_l1", "loss_ssim", "loss_lpips",
            "loss_visibility", "loss_chamfer", "loss_opacity_floor",
            "weight_lpips", "weight_chamfer", "weight_opacity",
        }
        self.assertEqual(set(losses), expected_keys)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
        self.assertAlmostEqual(float(losses["weight_lpips"]), 0.025)
        self.assertAlmostEqual(float(losses["weight_chamfer"]), 0.5)
        self.assertAlmostEqual(float(losses["weight_opacity"]), 0.05)
        losses["loss"].backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        self.assertTrue(bool(torch.isfinite(gaussians.grad).all()))


if __name__ == "__main__":
    unittest.main()
