import unittest
from unittest import mock

import torch
import torch.nn as nn

from scripts.training.multiscene import (
    QuerySplatTrainingModule,
    ResumableDistributedSampler,
    prepare_querysplat_images,
)
from tokengs.data.querysplat_images import IMAGENET_MEAN, IMAGENET_STD


class TestMultiSceneTraining(unittest.TestCase):
    def test_resume_sampler_matches_uninterrupted_suffix_on_all_ranks(self):
        for rank in range(4):
            sampler = ResumableDistributedSampler(
                list(range(101)), num_replicas=4, rank=rank,
                seed=42, shuffle=True, drop_last=True,
            )
            sampler.set_epoch(3)
            full = list(sampler)
            sampler.start_index = 10
            self.assertEqual(list(sampler), full[10:])
            self.assertEqual(len(sampler), len(full))
            sampler.start_index = len(full)
            self.assertEqual(list(sampler), [])

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


class TestProviderResume(unittest.TestCase):
    def provider(self):
        from types import SimpleNamespace
        from tokengs.data.provider import Provider
        provider = Provider.__new__(Provider)
        provider.opt = SimpleNamespace(seed=42, indexed_sampling=True, max_sample_attempts=3)
        provider.dataset = list(range(20))
        provider.set_rng_epoch(2)
        return provider

    def test_indexed_sampling_reproduces_views_and_augmentation_without_replay(self):
        provider = self.provider()
        provider.get_item = lambda idx: (int(provider.rng.integers(1000000)),
                                         float(torch.rand((), generator=provider.generator)))
        expected = provider[11]
        for idx in range(10):
            provider[idx]
        self.assertEqual(provider[11], expected)
        restored = self.provider()
        restored.get_item = lambda idx: (int(restored.rng.integers(1000000)),
                                        float(torch.rand((), generator=restored.generator)))
        self.assertEqual(restored[11], expected)

    def test_bad_dataset_fails_after_bounded_retries(self):
        provider = self.provider()
        provider.get_item = mock.Mock(side_effect=ValueError("bad scene"))
        with self.assertWarns(RuntimeWarning):
            with self.assertRaisesRegex(RuntimeError, "3 attempts"):
                provider[0]
        self.assertEqual(provider.get_item.call_count, 3)


if __name__ == "__main__":
    unittest.main()
