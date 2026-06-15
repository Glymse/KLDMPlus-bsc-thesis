from __future__ import annotations

from typing import Any

import torch

try:
    from torch_geometric.data import Batch, Data
except ImportError:  # pragma: no cover
    Batch = Data = Any

"Mode is either uniform or log_uniform, not nesseclary covered in theises. But is a possible ablation. See the loss_vs_time analytics"

class TimeSampler:
    # Keeps one RNG per device so timestep sampling stays reproducible across resume.
    def __init__(self, *, mode: str = "uniform", lower_bound: float = 1e-3, seed: int = 2002) -> None:
        if mode not in {"uniform", "log_uniform"}:
            raise ValueError(f"time sampler mode must be 'uniform' or 'log_uniform', got {mode!r}.")
        self.mode = str(mode)
        self.lower_bound = float(lower_bound)
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}

    # creates the seeded generator for the requested device.
    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    # Samples one diffusion time per graph and returns unit weights for the loss.
    def sample(self, batch: Batch | Data) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch.pos.device
        dtype = batch.pos.dtype
        num_graphs = int(batch.num_graphs)
        generator = self._generator_for(device)

        u = torch.rand(
            num_graphs,
            1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        if self.mode == "uniform":
            t = self.lower_bound + (1.0 - self.lower_bound) * u
        else:
            log_low = torch.log(torch.as_tensor(self.lower_bound, device=device, dtype=dtype))
            t = torch.exp(log_low + (0.0 - log_low) * u)

        weights = torch.ones(num_graphs, 1, device=device, dtype=dtype)
        return t, weights

    # Saves RNG state so a resumed run continues the same timestep stream.
    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "generator_states": {
                key: generator.get_state().detach().cpu()
                for key, generator in self._generators.items()
            },
        }

    # Restores RNG state from a checkpoint when available.
    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return
        self.mode = str(state_dict.get("mode", self.mode))
        self._generators.clear()
        for key, state in dict(state_dict.get("generator_states", {})).items():
            try:
                device = torch.device(key)
                generator = torch.Generator(device=device)
                generator.set_state(state.detach().cpu())
            except Exception:
                continue
            self._generators[str(device)] = generator
