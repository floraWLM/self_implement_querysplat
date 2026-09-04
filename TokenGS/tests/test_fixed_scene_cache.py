import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.models.vggt_encoder import VGGTEncoderOutput
from scripts.training.fixed_scene_cache import (
    materialize_fixed_scene_self_calibration,
    prepare_fixed_scene_vgm_cache,
)
from scripts.training.self_calibration import c2w_to_cam_view


class FakeCachedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt = SimpleNamespace(vggt_intermediate_layers=(4, 23))
        self.aggregator = SimpleNamespace(depth=24)
        self.mix = nn.Parameter(torch.tensor(2.0))
        c2w = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
        c2w[0, :, 0, 3] = torch.tensor([0.0, 1.0, 2.0])
        self.all_cameras = c2w_to_cam_view(c2w)
        self.input_calls = 0
        self.all_calls = 0

    def _run_aggregator_with_images(self, images):
        self.input_calls += 1
        outputs = [None] * 24
        outputs[4] = torch.tensor([4.0])
        outputs[23] = torch.tensor([1.0])
        return outputs, 2, images

    def _run_selected_layers(self, images, layer_indices):
        self.all_calls += 1
        return {23: torch.tensor([2.0])}, 2

    def _decode_cameras(self, final_layer, image_hw):
        cameras = self.all_cameras[:, :2] if float(final_layer[0]) == 1.0 else self.all_cameras
        batch, views = cameras.shape[:2]
        return cameras, torch.ones(batch, views, 4), torch.ones(batch, views, 9)

    def _decode_depths(self, output_list, patch_start, prepared_images, image_hw):
        height, width = image_hw
        shape = (1, 2, 1, height, width)
        return torch.ones(shape), torch.ones(shape)

    def _tokens_from_layers(self, output_by_layer, patch_start, batch_size):
        tokens = output_by_layer[23].reshape(1, 1, 1) * self.mix
        return VGGTEncoderOutput(
            tokens=tokens,
            eye_token=None,
            layer_weights=None,
            camera_tokens=tokens.unsqueeze(1),
            scene_tokens=tokens.unsqueeze(1),
            tokens_per_view=1,
            special_tokens_per_view=1,
        )


class TestFixedSceneCache(unittest.TestCase):
    def test_frozen_vgm_runs_once_but_trainable_adapter_remains_live(self):
        encoder = FakeCachedEncoder()
        images_all = torch.rand(1, 3, 3, 2, 2)
        cache = prepare_fixed_scene_vgm_cache(
            encoder,
            input_images=images_all[:, :2],
            images_all=images_all,
            image_hw=(2, 2),
        )
        self.assertEqual(encoder.input_calls, 1)
        self.assertEqual(encoder.all_calls, 1)
        self.assertTrue(all(not value.requires_grad for value in cache.input_layers.values()))

        first = materialize_fixed_scene_self_calibration(encoder, cache)
        second = materialize_fixed_scene_self_calibration(encoder, cache)
        (first.input_pass.geometry.tokens + second.input_pass.geometry.tokens).sum().backward()
        self.assertIsNotNone(encoder.mix.grad)
        self.assertGreater(float(encoder.mix.grad), 0.0)
        self.assertEqual(encoder.input_calls, 1)
        self.assertEqual(encoder.all_calls, 1)


if __name__ == "__main__":
    unittest.main()
