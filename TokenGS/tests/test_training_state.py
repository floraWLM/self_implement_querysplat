import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts.training.state import (
    ExponentialMovingAverage,
    OptimizerConfig,
    build_adamw,
    build_warmup_cosine_scheduler,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))
        self.bias = nn.Parameter(torch.tensor([0.5]))
        self.frozen = nn.Parameter(torch.tensor([3.0]), requires_grad=False)

    def forward(self):
        return self.weight.square().sum() + self.bias.square().sum()


class TestTrainingState(unittest.TestCase):
    def test_optimizer_groups_warmup_cosine_clip_and_ema(self):
        model = TinyModel()
        optimizer = build_adamw(
            model,
            OptimizerConfig(learning_rate=1e-3, weight_decay=0.05),
            fused=False,
        )
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.05, 0.0])
        scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)
        ema = ExponentialMovingAverage(model, decay=0.5)
        initial_average = ema.shadow["weight"].clone()

        optimizer.zero_grad(set_to_none=True)
        model().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertGreater(float(norm), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        expected = 0.5 * initial_average + 0.5 * model.weight.detach()
        torch.testing.assert_close(ema.shadow["weight"], expected)
        self.assertIsNone(model.frozen.grad)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_checkpoint_restores_model_optimizer_scheduler_ema_and_step(self):
        torch.manual_seed(3)
        model = TinyModel()
        optimizer = build_adamw(model, OptimizerConfig(learning_rate=1e-3), fused=False)
        scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=1, total_steps=5)
        ema = ExponentialMovingAverage(model, decay=0.9)
        optimizer.zero_grad(set_to_none=True)
        model().backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        expected_weight = model.weight.detach().clone()
        expected_ema = ema.shadow["weight"].clone()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_training_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                ema,
                global_step=1,
                extra={"num_queries": 1024},
            )
            with torch.no_grad():
                model.weight.add_(10)
            ema.shadow["weight"].zero_()
            step, extra = load_training_checkpoint(path, model, optimizer, scheduler, ema)

        self.assertEqual(step, 1)
        self.assertEqual(extra, {"num_queries": 1024})
        torch.testing.assert_close(model.weight, expected_weight)
        torch.testing.assert_close(ema.shadow["weight"], expected_ema)
        self.assertEqual(scheduler.last_epoch, 1)


if __name__ == "__main__":
    unittest.main()
