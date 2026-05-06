from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch, Data


@dataclass
class TimeSamplerOutput:
    t: torch.Tensor
    bins: torch.Tensor
    weights: torch.Tensor
    probs: torch.Tensor


class KLDMUniformTimeSampler:
    """
    Drop-in replacement for sample_times(...) when we want sampler objects.

    This preserves the current KLDM behavior:
        t_g ~ Uniform(lower_bound, 1)
    """

    def __init__(self, lower_bound: float = 1e-3, seed: int = 2002) -> None:
        self.lower_bound = float(lower_bound)
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}

    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def sample(self, batch: Batch | Data) -> TimeSamplerOutput:
        device = batch.pos.device
        dtype = batch.pos.dtype
        num_graphs = int(batch.num_graphs)
        generator = self._generator_for(device)

        t = self.lower_bound + (1.0 - self.lower_bound) * torch.rand(
            num_graphs,
            1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        return TimeSamplerOutput(
            t=t,
            bins=torch.zeros(num_graphs, device=device, dtype=torch.long),
            weights=torch.ones(num_graphs, 1, device=device, dtype=dtype),
            probs=torch.ones(1, device=device, dtype=dtype),
        )

    def update(self, **kwargs) -> None:
        del kwargs

    def diagnostics(self) -> dict[str, float]:
        return {}


class LossSecondMomentTimeSampler:
    """
    KLDM adaptation of OpenAI's LossSecondMomentResampler.

    Copied idea:
        q_i ∝ sqrt(E[L_i^2])

    KLDM-specific additions:
        - continuous time is discretized into bins
        - velocity and lattice losses are tracked separately
        - probabilities are mixed with uniform and clipped
    """

    def __init__(
        self,
        *,
        n_bins: int = 64,
        lower_bound: float = 1e-3,
        history_per_bin: int = 10,
        alpha: float = 0.5,
        adaptive_power: float = 0.5,
        min_prob: float = 0.002,
        max_prob: float = 0.10,
        velocity_weight: float = 0.7,
        lattice_weight: float = 0.3,
        use_importance_weights: bool = False,
        clip_importance_weights: bool = True,
        weight_clip_min: float = 0.5,
        weight_clip_max: float = 2.0,
        seed: int = 2002,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_bins = int(n_bins)
        self.lower_bound = float(lower_bound)
        self.history_per_bin = int(history_per_bin)
        self.alpha = float(alpha)
        self.adaptive_power = float(adaptive_power)
        if self.adaptive_power <= 0.0:
            raise ValueError("adaptive_power must be > 0")
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.velocity_weight = float(velocity_weight)
        self.lattice_weight = float(lattice_weight)
        self.use_importance_weights = bool(use_importance_weights)
        self.clip_importance_weights = bool(clip_importance_weights)
        self.weight_clip_min = float(weight_clip_min)
        self.weight_clip_max = float(weight_clip_max)
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}

        self.loss_v_history = torch.zeros(
            self.n_bins,
            self.history_per_bin,
            device=device,
            dtype=torch.float64,
        )
        self.loss_l_history = torch.zeros(
            self.n_bins,
            self.history_per_bin,
            device=device,
            dtype=torch.float64,
        )
        self.loss_counts = torch.zeros(
            self.n_bins,
            device=device,
            dtype=torch.long,
        )

    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def warmed_up(self) -> bool:
        return bool((self.loss_counts >= self.history_per_bin).all().item())

    def _second_moments(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Keep branch scales comparable so the lattice spike does not dominate
        # purely because of magnitude.
        moments_v = (self.loss_v_history ** 2).mean(dim=-1).to(device=device, dtype=dtype)
        moments_l = (self.loss_l_history ** 2).mean(dim=-1).to(device=device, dtype=dtype)
        eps = torch.as_tensor(1e-12, device=device, dtype=dtype)

        moments_v = moments_v / moments_v.mean().clamp_min(eps)
        moments_l = moments_l / moments_l.mean().clamp_min(eps)

        return self.velocity_weight * moments_v + self.lattice_weight * moments_l

    def probabilities(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.warmed_up():
            return torch.full((self.n_bins,), 1.0 / self.n_bins, device=device, dtype=dtype)

        moments = self._second_moments(device=device, dtype=dtype)
        adaptive = moments.clamp_min(1e-12).pow(self.adaptive_power)
        adaptive = adaptive / adaptive.sum().clamp_min(1e-12)

        uniform = torch.full_like(adaptive, 1.0 / self.n_bins)
        probs = (1.0 - self.alpha) * uniform + self.alpha * adaptive
        probs = probs.clamp(min=self.min_prob, max=self.max_prob)
        probs = probs / probs.sum().clamp_min(1e-12)
        return probs

    def sample(self, batch: Batch | Data) -> TimeSamplerOutput:
        device = batch.pos.device
        dtype = batch.pos.dtype
        num_graphs = int(batch.num_graphs)
        generator = self._generator_for(device)

        probs = self.probabilities(device=device, dtype=dtype)
        bins = torch.multinomial(
            probs,
            num_samples=num_graphs,
            replacement=True,
            generator=generator,
        )

        u = torch.rand(num_graphs, device=device, dtype=dtype, generator=generator)
        bin_width = (1.0 - self.lower_bound) / self.n_bins
        t = self.lower_bound + (bins.to(dtype) + u) * bin_width
        t = t[:, None]

        if self.use_importance_weights:
            selected_probs = probs[bins]
            weights = 1.0 / (self.n_bins * selected_probs)
            if self.clip_importance_weights:
                weights = weights.clamp(self.weight_clip_min, self.weight_clip_max)
        else:
            weights = torch.ones(num_graphs, device=device, dtype=dtype)

        return TimeSamplerOutput(
            t=t,
            bins=bins,
            weights=weights[:, None].to(dtype=dtype),
            probs=probs.detach(),
        )

    @torch.no_grad()
    def update(
        self,
        *,
        bins: torch.Tensor,
        loss_v_graph: torch.Tensor,
        loss_l_graph: torch.Tensor,
    ) -> None:
        device = self.loss_v_history.device
        bins = bins.detach().to(device=device, dtype=torch.long)
        loss_v_graph = loss_v_graph.detach().to(device=device, dtype=torch.float64)
        loss_l_graph = loss_l_graph.detach().to(device=device, dtype=torch.float64)

        for bin_id, loss_v, loss_l in zip(bins.tolist(), loss_v_graph, loss_l_graph):
            count = int(self.loss_counts[bin_id].item())
            if count < self.history_per_bin:
                self.loss_v_history[bin_id, count] = loss_v
                self.loss_l_history[bin_id, count] = loss_l
                self.loss_counts[bin_id] += 1
            else:
                self.loss_v_history[bin_id, :-1] = self.loss_v_history[bin_id, 1:].clone()
                self.loss_l_history[bin_id, :-1] = self.loss_l_history[bin_id, 1:].clone()
                self.loss_v_history[bin_id, -1] = loss_v
                self.loss_l_history[bin_id, -1] = loss_l

    def diagnostics(self) -> dict[str, float]:
        device = self.loss_v_history.device
        probs = self.probabilities(device=device, dtype=torch.float64)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        effective_bins = torch.exp(entropy)
        return {
            "time_sampler/p_min": float(probs.min().item()),
            "time_sampler/p_max": float(probs.max().item()),
            "time_sampler/entropy": float(entropy.item()),
            "time_sampler/effective_bins": float(effective_bins.item()),
            "time_sampler/warmed_up": float(self.warmed_up()),
        }
