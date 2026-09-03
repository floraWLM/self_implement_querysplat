import unittest

import torch

from tokengs.data.querysplat_images import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    split_querysplat_images,
)


class TestQuerySplatImages(unittest.TestCase):
    def test_unbatched_split_preserves_raw_views_and_normalizes_inputs(self):
        images_all = torch.linspace(0, 1, 6 * 3 * 2 * 2).reshape(6, 3, 2, 2)

        result = split_querysplat_images(images_all, num_input_views=4)

        torch.testing.assert_close(result.input_raw, images_all[:4])
        torch.testing.assert_close(result.supervision_images, images_all[4:])
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        torch.testing.assert_close(result.input_normalized, (images_all[:4] - mean) / std)

    def test_batched_split_uses_view_dimension_one(self):
        images_all = torch.rand(2, 8, 3, 4, 5)

        result = split_querysplat_images(images_all, num_input_views=3)

        self.assertEqual(result.input_raw.shape, (2, 3, 3, 4, 5))
        self.assertEqual(result.input_normalized.shape, (2, 3, 3, 4, 5))
        self.assertEqual(result.supervision_images.shape, (2, 5, 3, 4, 5))
        torch.testing.assert_close(result.input_raw, images_all[:, :3])
        torch.testing.assert_close(result.supervision_images, images_all[:, 3:])

    def test_rejects_invalid_contracts(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            split_querysplat_images(torch.rand(3, 4, 5), 1)
        with self.assertRaisesRegex(ValueError, "RGB"):
            split_querysplat_images(torch.rand(4, 1, 2, 2), 2)
        with self.assertRaisesRegex(ValueError, "num_input_views"):
            split_querysplat_images(torch.rand(4, 3, 2, 2), 4)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            split_querysplat_images(torch.full((4, 3, 2, 2), 2.0), 2)


if __name__ == "__main__":
    unittest.main()
