from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class _EMAModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)


class EMA(nn.Module):
    """
    Exponential moving average wrapper used by KLDM+ training.

    Two modes are supported:
        - legacy fixed decay with delayed start
        - Karras power EMA with beta(step) = (1 - 1 / step) ** (1 + gamma)

    The KLDM+ YAML configs can switch to the power schedule by setting
    `ema.type: power` and a `gamma` value.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        start_epoch: int = 500,
        gamma: float | None = None,
    ) -> None:
        super().__init__()
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.gamma = None if gamma is None else float(gamma)
        self.online_model = [model]
        self.ema_model = _EMAModule(model)
        self.register_buffer("_num_updates", torch.zeros((), dtype=torch.long))

    @property
    def model(self) -> nn.Module:
        return self.online_model[0]

    @property
    def num_updates(self) -> int:
        return int(self._num_updates.item())

    @property
    def uses_power_schedule(self) -> bool:
        return self.gamma is not None

    def _power_beta(self, step: int) -> float:
        return (1.0 - 1.0 / float(step)) ** (1.0 + float(self.gamma))

    @torch.no_grad()
    def _copy_model_state(self, model: nn.Module) -> None:
        self.ema_model.module.load_state_dict(model.state_dict(), strict=False)

    @torch.no_grad()
    def _apply_decay(self, model: nn.Module, beta: float) -> None:
        ema_state = self.ema_model.module.state_dict()
        model_state = model.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach().to(
                device=ema_value.device,
                dtype=ema_value.dtype,
            )
            if torch.is_floating_point(ema_value):
                ema_value.mul_(beta).add_(model_value, alpha=1.0 - beta)
            else:
                ema_value.copy_(model_value)

    @torch.no_grad()
    def update(self, model: nn.Module | None = None, current_epoch: int | None = None) -> None:
        model = self.model if model is None else model

        if not self.uses_power_schedule and current_epoch is not None and current_epoch <= self.start_epoch:
            return

        next_step = self.num_updates + 1
        self._num_updates.fill_(next_step)

        if next_step == 1:
            self._copy_model_state(model)
            return

        beta = self.decay if not self.uses_power_schedule else self._power_beta(next_step)
        self._apply_decay(model, beta)

    @torch.no_grad()
    def copy_ema_to_model(self, model: nn.Module | None = None) -> None:
        model = self.model if model is None else model
        model.load_state_dict(self.ema_model.module.state_dict(), strict=False)

    @contextmanager
    def average_parameters(self, model: nn.Module | None = None) -> Iterator[None]:
        model = self.model if model is None else model
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.copy_ema_to_model(model)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Older checkpoints saved by torch.swa_utils.AveragedModel expose
        # `ema_model.n_averaged` instead of our `_num_updates` buffer.
        state_dict = dict(state_dict)
        legacy_updates = state_dict.get("ema_model.n_averaged")
        if legacy_updates is not None and "_num_updates" not in state_dict:
            state_dict["_num_updates"] = legacy_updates.to(dtype=torch.long)
        return super().load_state_dict(state_dict, strict=strict)


ExponentialMovingAverage = EMA
