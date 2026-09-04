"""Optimizer, warmup-cosine schedule, EMA, and resumable training checkpoints."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must lie in [0,1)")


def build_adamw(
    model: nn.Module,
    config: OptimizerConfig,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Match TokenGS decay/no-decay grouping while excluding frozen VGM parameters."""
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if fused is None:
        fused = any(parameter.is_cuda for parameter in decay + no_decay)
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=fused,
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("scheduler requires 0 <= warmup_steps < total_steps")
    if not 0 <= minimum_lr_ratio <= 1:
        raise ValueError("minimum_lr_ratio must lie in [0,1]")

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


class ExponentialMovingAverage:
    """EMA over trainable parameters only; frozen VGGT weights are referenced externally."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must lie in [0,1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("model has no trainable parameters for EMA")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if not set(parameters).issuperset(self.shadow):
            raise ValueError("EMA parameter names no longer match the model")
        for name, average in self.shadow.items():
            value = parameters[name].detach().to(device=average.device, dtype=average.dtype)
            average.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            parameters[name].copy_(average.to(parameters[name]))

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any], model: nn.Module) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("checkpoint EMA decay does not match current configuration")
        parameters = dict(model.named_parameters())
        saved = state["shadow"]
        if set(saved) != set(self.shadow):
            raise ValueError("checkpoint EMA parameter names do not match the model")
        for name, value in saved.items():
            if value.shape != parameters[name].shape:
                raise ValueError(f"EMA shape mismatch for {name}")
            self.shadow[name] = value.to(
                device=parameters[name].device,
                dtype=parameters[name].dtype,
            ).clone()


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    global_step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format_version": 1,
        "global_step": global_step,
        "trainable_model": _trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "extra": extra or {},
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
) -> tuple[int, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    saved = payload["trainable_model"]
    if set(saved) != set(current):
        raise ValueError("checkpoint trainable parameter names do not match the model")
    with torch.no_grad():
        for name, value in saved.items():
            if value.shape != current[name].shape:
                raise ValueError(f"checkpoint shape mismatch for {name}")
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    ema.load_state_dict(payload["ema"], model)
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["global_step"]), dict(payload.get("extra", {}))
