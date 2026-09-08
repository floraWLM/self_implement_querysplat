import unittest
from unittest import mock

import torch
import torch.nn as nn

from scripts.training.multiscene import (
    QuerySplatTrainingModule,
    prepare_querysplat_images,
)
from tokengs.data.querysplat_images import IMAGENET_MEAN, IMAGENET_STD


class TestMultiSceneTraining(unittest.TestCase):
    def test_provider_batch_is_split_from_images_all(self):
        torch.manual_seed(9)
        images_all = torch.rand(1, 8, 3, 6, 7)
        mean = images_all.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = images_all.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        loader_batch = {
            "images_all": images_all,
            "images_input": images_all[:, :4],
            "images_output": images_all[:, 4:],
            "input": torch.cat(
                [(images_all - mean) / std, torch.zeros(1, 8, 6, 6, 7)], dim=2
            ),
        }
        result = prepare_querysplat_images(
            loader_batch, 4, validate_provider_contract=True
        )
        torch.testing.assert_close(result.input_raw, images_all[:, :4])
        torch.testing.assert_close(result.supervision_images, images_all[:, 4:])
        torch.testing.assert_close(result.input_normalized, (images_all[:, :4] - mean) / std)

    def test_complete_training_forward_stays_inside_wrapper(self):
        model = nn.Linear(2, 2)
        wrapper = QuerySplatTrainingModule(model)
        sentinel = object()
        with mock.patch(
            "scripts.training.multiscene.forward_querysplat_training",
            return_value=sentinel,
        ) as forward:
            result = wrapper(
                torch.zeros(1, 2), torch.ones(1, 2), torch.full((1, 2), 2.0)
            )
        self.assertIs(result, sentinel)
        self.assertIs(forward.call_args.args[0], model)


if __name__ == "__main__":
    unittest.main()
