import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.models.vggt_encoder import VGGTEncoder, VGGTEncoderOutput
from scripts.training.vggt_input_pass import (
    VGGTInputPassOutput,
    forward_vggt_input_once,
)


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 24


class FakeVGGTEncoder(VGGTEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.opt = SimpleNamespace(vggt_intermediate_layers=(4, 11, 17, 23))
        self.aggregator = FakeAggregator()
        self.calls = 0

    def _run_aggregator_with_images(self, images):
        self.calls += 1
        outputs = [None] * 24
        for index in self.opt.vggt_intermediate_layers:
            outputs[index] = torch.full((1,), float(index))
        return outputs, 17, images

    def _tokens_from_layers(self, output_by_layer, patch_start, batch_size):
        return VGGTEncoderOutput(
            tokens=torch.ones(batch_size, 8, 16),
            eye_token=torch.ones(batch_size, 16),
            layer_weights=torch.ones(batch_size, 4) / 4,
            camera_tokens=torch.ones(batch_size, 2, 1, 16),
            scene_tokens=torch.ones(batch_size, 2, 16, 16),
            tokens_per_view=4,
            special_tokens_per_view=2,
        )

    def _decode_cameras(self, final_layer, image_hw):
        return (
            torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
            torch.ones(1, 2, 4),
            torch.ones(1, 2, 9),
        )

    def _decode_depths(self, output_list, patch_start, prepared_images, image_hw):
        height, width = image_hw
        return torch.ones(1, 2, 1, height, width), torch.ones(1, 2, 1, height, width)


class TestVGGTInputPass(unittest.TestCase):
    def test_one_aggregator_pass_feeds_all_products(self):
        model = FakeVGGTEncoder()

        output = forward_vggt_input_once(
            model,
            torch.rand(1, 2, 3, 8, 8),
            image_hw=(8, 8),
        )

        self.assertIsInstance(output, VGGTInputPassOutput)
        self.assertEqual(model.calls, 1)
        self.assertEqual(output.geometry.tokens.shape, (1, 8, 16))
        self.assertEqual(output.cam_view.shape, (1, 2, 4, 4))
        self.assertEqual(output.intrinsics.shape, (1, 2, 4))
        self.assertEqual(output.depth.shape, (1, 2, 1, 8, 8))
        self.assertEqual(output.depth_confidence.shape, (1, 2, 1, 8, 8))


if __name__ == "__main__":
    unittest.main()
