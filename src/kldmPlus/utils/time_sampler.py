from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch, Data


@dataclass
class TimeSamplerOutput:
    t: torch.Tensor
    bins: torch.Tensor
    weights: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor | None = None
    entropies: torch.Tensor | None = None
    policy_alpha: torch.Tensor | None = None
    policy_beta: torch.Tensor | None = None


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

    def before_model_update(self, *, batch: Batch | Data, model) -> None:
        del batch, model

    def after_model_update(
        self,
        *,
        batch: Batch | Data,
        model,
        sampled_time: TimeSamplerOutput,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        del batch, model, sampled_time, metrics

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        del state_dict

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

    def before_model_update(self, *, batch: Batch | Data, model) -> None:
        del batch, model

    def after_model_update(
        self,
        *,
        batch: Batch | Data,
        model,
        sampled_time: TimeSamplerOutput,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        del batch, model
        self.update(
            bins=sampled_time.bins,
            loss_v_graph=metrics["loss_v_graph"],
            loss_l_graph=metrics["loss_l_graph"],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss_v_history": self.loss_v_history,
            "loss_l_history": self.loss_l_history,
            "loss_counts": self.loss_counts,
        }

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return
        device = self.loss_v_history.device
        self.loss_v_history.copy_(state_dict["loss_v_history"].to(device=device, dtype=torch.float64))
        self.loss_l_history.copy_(state_dict["loss_l_history"].to(device=device, dtype=torch.float64))
        self.loss_counts.copy_(state_dict["loss_counts"].to(device=device, dtype=torch.long))

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


def _graphwise_mean(values: torch.Tensor, index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if values.ndim == 1:
        values = values[:, None]
    sums = torch.zeros(
        num_graphs,
        values.shape[-1],
        device=values.device,
        dtype=values.dtype,
    )
    sums = sums.index_add(0, index, values)
    counts = torch.bincount(index, minlength=num_graphs).to(
        device=values.device,
        dtype=values.dtype,
    ).clamp_min(1.0)[:, None]
    return sums / counts


def _graphwise_std(
    values: torch.Tensor,
    index: torch.Tensor,
    num_graphs: int,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim == 1:
        values = values[:, None]
    if mean is None:
        mean = _graphwise_mean(values, index, num_graphs)
    second_moment = _graphwise_mean(values.square(), index, num_graphs)
    var = (second_moment - mean.square()).clamp_min(1e-12)
    return torch.sqrt(var)


class GraphTimeBetaPolicy(nn.Module):
    """
    Graph-level Beta policy for continuous KLDM++ training times.

    Inspired by adaptive paper:
    nearby timesteps tend to behave similarly, so we predict a smooth density on
    [lower_bound, 1] instead of an unrelated categorical distribution.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        min_concentration: float,
    ) -> None:
        super().__init__()
        if hidden_depth <= 0:
            raise ValueError("hidden_depth must be positive.")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(hidden_depth - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 2)
        self.min_concentration = float(min_concentration)

        nn.init.zeros_(self.output.weight)
        # Match the reference adaptive implementation: the final actor layer is
        # zero-initialized with a 0.5 bias before the softplus transform.
        nn.init.constant_(self.output.bias, 0.5)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        raw = self.output(hidden)
        alpha, beta = torch.chunk(raw, 2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha).squeeze(-1) + self.min_concentration
        beta = torch.nn.functional.softplus(beta).squeeze(-1) + self.min_concentration
        return alpha, beta


class AdaptiveReinforceTimeSampler:
    """
    Adaptive REINFORCE timestep sampler for KLDM++.

    Inspired by adaptive paper:
    instead of only tracking which times currently have large losses, the policy
    learns which sampled training times produce the largest before/after
    improvement on a small set of probe times.
    """

    def __init__(
        self,
        *,
        lower_bound: float = 1e-3,
        policy_hidden_dim: int = 128,
        policy_hidden_depth: int = 2,
        min_concentration: float = 0.25,
        policy_lr: float = 2e-5,
        entropy_coef: float = 1e-2,
        policy_update_every: int = 100,
        policy_warmup_steps: int = 5000,
        reward_candidate_times: int = 7,
        reward_active_times: int = 5,
        reward_history_size: int = 64,
        use_baseline: bool = False,
        reward_baseline_momentum: float = 0.95,
        reward_velocity_weight: float = 1.0,
        reward_lattice_weight: float = 1.0,
        reward_size_weight_power: float = 0.0,
        reward_size_weight_max: float = 2.0,
        reward_normalization_eps: float = 1e-6,
        entropy_in_reward: bool = True,
        use_importance_weights: bool = True,
        clip_importance_weights: bool = True,
        weight_clip_min: float = 0.25,
        weight_clip_max: float = 4.0,
        gradient_clip_norm: float = 1.0,
        feature_selection_min_history: int = 32,
        seed: int = 2002,
        device: torch.device | str = "cpu",
        reward_probe_times: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        self.lower_bound = float(lower_bound)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.policy_hidden_depth = int(policy_hidden_depth)
        self.min_concentration = float(min_concentration)
        self.policy_lr = float(policy_lr)
        self.entropy_coef = float(entropy_coef)
        self.policy_update_every = int(policy_update_every)
        self.policy_warmup_steps = int(policy_warmup_steps)
        self.reward_history_size = int(reward_history_size)
        self.use_baseline = bool(use_baseline)
        self.reward_baseline_momentum = float(reward_baseline_momentum)
        self.reward_velocity_weight = float(reward_velocity_weight)
        self.reward_lattice_weight = float(reward_lattice_weight)
        self.reward_size_weight_power = float(reward_size_weight_power)
        self.reward_size_weight_max = float(reward_size_weight_max)
        self.reward_normalization_eps = float(reward_normalization_eps)
        self.entropy_in_reward = bool(entropy_in_reward)
        self.use_importance_weights = bool(use_importance_weights)
        self.clip_importance_weights = bool(clip_importance_weights)
        self.weight_clip_min = float(weight_clip_min)
        self.weight_clip_max = float(weight_clip_max)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.feature_selection_min_history = int(feature_selection_min_history)
        self.seed = int(seed)
        self.device = torch.device(device)

        if self.policy_update_every < 0:
            raise ValueError("policy_update_every must be non-negative.")
        if self.policy_warmup_steps < 0:
            raise ValueError("policy_warmup_steps must be non-negative.")

        if reward_probe_times is not None:
            probe_times = [float(value) for value in reward_probe_times]
        else:
            probe_times = torch.linspace(
                self.lower_bound,
                1.0,
                int(reward_candidate_times),
                dtype=torch.float64,
            ).tolist()
        if not probe_times:
            raise ValueError("At least one reward probe time is required.")
        self.reward_probe_times = torch.tensor(probe_times, dtype=torch.float64)
        self.reward_active_times = min(int(reward_active_times), len(probe_times))
        if self.reward_active_times <= 0:
            raise ValueError("reward_active_times must be positive.")

        self.policy: GraphTimeBetaPolicy | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self._pending_state_dict: dict[str, Any] | None = None
        self._feature_dim: int | None = None
        self._generators: dict[str, torch.Generator] = {}

        self.num_model_steps = 0
        self.reward_baseline = 0.0
        self.reward_baseline_initialized = False
        self.last_reward_mean = 0.0
        self.last_reward_std = 0.0
        self.last_policy_loss = 0.0
        self.last_entropy = 0.0
        self.last_alpha_mean = 0.0
        self.last_beta_mean = 0.0
        self.last_sampled_t_mean = 0.0
        self.last_sampled_t_std = 0.0
        self.last_sampled_t_min = 0.0
        self.last_sampled_t_max = 0.0
        self.last_reward_velocity_mean = 0.0
        self.last_reward_lattice_mean = 0.0
        self.last_reward_size_weight_mean = 1.0
        self.selected_probe_indices = torch.arange(self.reward_active_times, dtype=torch.long)
        self._probe_history: list[torch.Tensor] = []
        self._pending_policy_update = False
        self._probe_cache: dict[str, Any] | None = None

    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def _graph_features(self, batch: Batch | Data) -> torch.Tensor:
        index = batch.batch
        num_graphs = int(batch.num_graphs)
        dtype = batch.pos.dtype

        num_atoms = torch.bincount(index, minlength=num_graphs).to(
            device=batch.pos.device,
            dtype=dtype,
        )
        atom_values = batch.atomic_numbers.to(device=batch.pos.device, dtype=dtype)
        atom_mean = _graphwise_mean(atom_values, index, num_graphs)
        atom_std = _graphwise_std(atom_values, index, num_graphs, atom_mean)
        pos_mean = _graphwise_mean(batch.pos, index, num_graphs)
        pos_std = _graphwise_std(batch.pos, index, num_graphs, pos_mean)
        pos_abs_mean = _graphwise_mean(batch.pos.abs(), index, num_graphs)

        lattice = batch.l.to(dtype=dtype).reshape(num_graphs, -1)
        lattice_abs_mean = lattice.abs().mean(dim=-1, keepdim=True)
        lattice_norm = lattice.norm(dim=-1, keepdim=True)

        features = torch.cat(
            [
                torch.log1p(num_atoms)[:, None],
                atom_mean,
                atom_std,
                pos_mean,
                pos_std,
                pos_abs_mean,
                lattice,
                lattice_abs_mean,
                lattice_norm,
            ],
            dim=-1,
        )
        return features

    def _build_policy(self, feature_dim: int) -> None:
        self._feature_dim = int(feature_dim)
        self.policy = GraphTimeBetaPolicy(
            input_dim=self._feature_dim,
            hidden_dim=self.policy_hidden_dim,
            hidden_depth=self.policy_hidden_depth,
            min_concentration=self.min_concentration,
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.policy_lr,
        )
        if self._pending_state_dict is not None:
            pending = self._pending_state_dict
            self._pending_state_dict = None
            self.load_state_dict(pending)

    def _ensure_policy(self, batch: Batch | Data) -> None:
        if self.policy is not None and self.policy_optimizer is not None:
            return
        feature_dim = int(self._graph_features(batch).shape[-1])
        self._build_policy(feature_dim)

    def _candidate_times(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.reward_probe_times.to(device=device, dtype=dtype)

    def _policy_active(self) -> bool:
        return self.num_model_steps >= self.policy_warmup_steps

    def _default_probe_indices(self) -> torch.Tensor:
        return torch.arange(
            min(self.reward_active_times, self.reward_probe_times.numel()),
            dtype=torch.long,
        )

    def _append_probe_history(self, delta_vector: torch.Tensor) -> None:
        self._probe_history.append(delta_vector.detach().to(device="cpu", dtype=torch.float64))
        if len(self._probe_history) > self.reward_history_size:
            self._probe_history = self._probe_history[-self.reward_history_size :]

    def _graph_size_weights(
        self,
        *,
        batch: Batch | Data,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.reward_size_weight_power <= 0.0:
            return torch.ones(int(batch.num_graphs), device=device, dtype=dtype)

        num_atoms = torch.bincount(batch.batch, minlength=int(batch.num_graphs)).to(
            device=device,
            dtype=dtype,
        ).clamp_min(1.0)
        mean_num_atoms = num_atoms.mean().clamp_min(torch.as_tensor(1.0, device=device, dtype=dtype))
        size_weights = (num_atoms / mean_num_atoms).pow(self.reward_size_weight_power)
        size_weights = size_weights.clamp(max=self.reward_size_weight_max)
        return size_weights / size_weights.mean().clamp_min(torch.as_tensor(1e-12, device=device, dtype=dtype))

    def _select_probe_indices(self) -> torch.Tensor:
        if self.reward_active_times >= self.reward_probe_times.numel():
            return torch.arange(self.reward_probe_times.numel(), dtype=torch.long)
        if len(self._probe_history) < self.feature_selection_min_history:
            return self._default_probe_indices()

        history = torch.stack(self._probe_history, dim=0)
        target = history.mean(dim=1)
        target_centered = target - target.mean()
        target_scale = target_centered.norm().item()
        if target_scale <= 1e-12:
            scores = history.abs().mean(dim=0)
        else:
            scores = []
            for column in history.T:
                column_centered = column - column.mean()
                denom = column_centered.norm() * target_centered.norm()
                if float(denom.item()) <= 1e-12:
                    scores.append(torch.tensor(0.0, dtype=history.dtype))
                else:
                    corr = torch.abs(torch.dot(column_centered, target_centered) / denom)
                    scores.append(corr)
            scores = torch.stack(scores)

        topk = torch.topk(scores, k=self.reward_active_times).indices
        return torch.sort(topk).values.cpu()

    def _prepare_probe_cache(self, *, batch: Batch | Data, model) -> dict[str, Any]:
        candidate_times = self._candidate_times(device=batch.pos.device, dtype=batch.pos.dtype)
        generator = self._generator_for(batch.pos.device)
        prepared_batches = []
        before_velocity_losses = []
        before_lattice_losses = []

        with torch.no_grad():
            for tau in candidate_times.tolist():
                graph_time = torch.full(
                    (int(batch.num_graphs), 1),
                    float(tau),
                    device=batch.pos.device,
                    dtype=batch.pos.dtype,
                )
                lattice_noise = torch.randn(
                    batch.l.shape,
                    device=batch.l.device,
                    dtype=batch.l.dtype,
                    generator=generator,
                )
                velocity_noise = torch.randn(
                    batch.pos.shape,
                    device=batch.pos.device,
                    dtype=batch.pos.dtype,
                    generator=generator,
                )
                position_noise = torch.randn(
                    batch.pos.shape,
                    device=batch.pos.device,
                    dtype=batch.pos.dtype,
                    generator=generator,
                )
                prepared = model.prepare_training_batch(
                    batch=batch,
                    t=graph_time,
                    lattice_noise=lattice_noise,
                    velocity_noise=velocity_noise,
                    position_noise=position_noise,
                )
                _loss, probe_metrics = model.loss_from_prepared(prepared)
                prepared_batches.append(prepared)
                before_velocity_losses.append(probe_metrics["loss_v_graph"])
                before_lattice_losses.append(probe_metrics["loss_l_graph"])

        return {
            "prepared": prepared_batches,
            "before_velocity_losses": torch.stack(before_velocity_losses, dim=1),
            "before_lattice_losses": torch.stack(before_lattice_losses, dim=1),
        }

    def sample(self, batch: Batch | Data) -> TimeSamplerOutput:
        device = batch.pos.device
        dtype = batch.pos.dtype
        num_graphs = int(batch.num_graphs)
        generator = self._generator_for(device)

        if not self._policy_active():
            t = self.lower_bound + (1.0 - self.lower_bound) * torch.rand(
                num_graphs,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            self.last_sampled_t_mean = float(t.detach().mean().item())
            self.last_sampled_t_std = float(t.detach().std(unbiased=False).item())
            self.last_sampled_t_min = float(t.detach().min().item())
            self.last_sampled_t_max = float(t.detach().max().item())
            return TimeSamplerOutput(
                t=t,
                bins=torch.zeros(num_graphs, device=device, dtype=torch.long),
                weights=torch.ones(num_graphs, 1, device=device, dtype=dtype),
                probs=torch.ones(1, device=device, dtype=dtype),
            )

        self._ensure_policy(batch)
        assert self.policy is not None

        features = self._graph_features(batch)
        alpha, beta = self.policy(features)
        dist = torch.distributions.Beta(alpha, beta)
        u = dist.sample().clamp(1e-6, 1.0 - 1e-6)
        log_probs = dist.log_prob(u)
        entropies = dist.entropy()
        t = self.lower_bound + (1.0 - self.lower_bound) * u.to(dtype=dtype)

        if self.use_importance_weights:
            weights = torch.exp(-log_probs.detach())
            if self.clip_importance_weights:
                weights = weights.clamp(self.weight_clip_min, self.weight_clip_max)
        else:
            weights = torch.ones_like(log_probs, dtype=dtype)

        self.last_alpha_mean = float(alpha.detach().mean().item())
        self.last_beta_mean = float(beta.detach().mean().item())
        self.last_sampled_t_mean = float(t.detach().mean().item())
        self.last_sampled_t_std = float(t.detach().std(unbiased=False).item())
        self.last_sampled_t_min = float(t.detach().min().item())
        self.last_sampled_t_max = float(t.detach().max().item())

        return TimeSamplerOutput(
            t=t[:, None].detach(),
            bins=torch.zeros(num_graphs, device=device, dtype=torch.long),
            weights=weights[:, None].to(dtype=dtype),
            probs=torch.ones(1, device=device, dtype=dtype),
            log_probs=log_probs,
            entropies=entropies,
            policy_alpha=alpha.detach(),
            policy_beta=beta.detach(),
        )

    def before_model_update(self, *, batch: Batch | Data, model) -> None:
        self.num_model_steps += 1
        self._pending_policy_update = (
            self._policy_active()
            and self.policy_update_every > 0
            and self.num_model_steps % self.policy_update_every == 0
        )
        self._probe_cache = None

        if not self._pending_policy_update:
            return

        self.selected_probe_indices = self._select_probe_indices()
        self._probe_cache = self._prepare_probe_cache(batch=batch, model=model)

    def after_model_update(
        self,
        *,
        batch: Batch | Data,
        model,
        sampled_time: TimeSamplerOutput,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        del metrics
        if not self._pending_policy_update or self._probe_cache is None:
            return
        if sampled_time.log_probs is None or sampled_time.entropies is None:
            self._probe_cache = None
            return

        assert self.policy is not None
        assert self.policy_optimizer is not None

        prepared_batches = self._probe_cache["prepared"]
        before_velocity_losses = self._probe_cache["before_velocity_losses"]
        before_lattice_losses = self._probe_cache["before_lattice_losses"]
        after_velocity_losses = []
        after_lattice_losses = []

        with torch.no_grad():
            for prepared in prepared_batches:
                _loss, probe_metrics = model.loss_from_prepared(prepared)
                after_velocity_losses.append(probe_metrics["loss_v_graph"])
                after_lattice_losses.append(probe_metrics["loss_l_graph"])

        after_velocity_losses_t = torch.stack(after_velocity_losses, dim=1)
        after_lattice_losses_t = torch.stack(after_lattice_losses, dim=1)

        velocity_delta = before_velocity_losses - after_velocity_losses_t
        lattice_delta = before_lattice_losses - after_lattice_losses_t

        velocity_norm = before_velocity_losses.mean(dim=1, keepdim=True).clamp_min(
            self.reward_normalization_eps
        )
        lattice_norm = before_lattice_losses.mean(dim=1, keepdim=True).clamp_min(
            self.reward_normalization_eps
        )
        combined_delta = (
            self.reward_velocity_weight * (velocity_delta / velocity_norm)
            + self.reward_lattice_weight * (lattice_delta / lattice_norm)
        )

        batch_mean_delta = combined_delta.mean(dim=0)
        self._append_probe_history(batch_mean_delta)

        selected_probe_indices = self.selected_probe_indices.to(device=combined_delta.device)
        selected_velocity_delta = velocity_delta[:, selected_probe_indices]
        selected_lattice_delta = lattice_delta[:, selected_probe_indices]
        selected_velocity_norm = before_velocity_losses[:, selected_probe_indices].mean(
            dim=1,
            keepdim=True,
        ).clamp_min(self.reward_normalization_eps)
        selected_lattice_norm = before_lattice_losses[:, selected_probe_indices].mean(
            dim=1,
            keepdim=True,
        ).clamp_min(self.reward_normalization_eps)
        normalized_velocity_reward = selected_velocity_delta / selected_velocity_norm
        normalized_lattice_reward = selected_lattice_delta / selected_lattice_norm
        selected_combined_delta = (
            self.reward_velocity_weight * normalized_velocity_reward
            + self.reward_lattice_weight * normalized_lattice_reward
        )
        rewards = selected_combined_delta.mean(dim=1)
        size_weights = self._graph_size_weights(
            batch=batch,
            device=rewards.device,
            dtype=rewards.dtype,
        )
        rewards = rewards * size_weights
        reward_mean = float(rewards.mean().item())
        reward_std = float(rewards.std(unbiased=False).item())

        reward_signal = rewards
        if self.use_baseline:
            if not self.reward_baseline_initialized:
                self.reward_baseline = reward_mean
                self.reward_baseline_initialized = True
            else:
                self.reward_baseline = (
                    self.reward_baseline_momentum * self.reward_baseline
                    + (1.0 - self.reward_baseline_momentum) * reward_mean
                )
            reward_signal = rewards - self.reward_baseline
        else:
            self.reward_baseline = 0.0
            self.reward_baseline_initialized = False

        # Inspired by adaptive paper:
        # the policy is updated with a REINFORCE loss based on before/after
        # improvement over probe times. The reference implementation folds the
        # entropy term into the reward; we keep that as the default mode here.
        if self.entropy_in_reward:
            reinforce_reward = reward_signal.detach() + self.entropy_coef * sampled_time.entropies
            policy_loss = -(sampled_time.log_probs * reinforce_reward).mean()
        else:
            policy_loss = -(reward_signal.detach() * sampled_time.log_probs).mean()
            policy_loss = policy_loss - self.entropy_coef * sampled_time.entropies.mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if self.gradient_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.gradient_clip_norm)
        self.policy_optimizer.step()

        self.last_reward_mean = reward_mean
        self.last_reward_std = reward_std
        self.last_policy_loss = float(policy_loss.detach().item())
        self.last_entropy = float(sampled_time.entropies.detach().mean().item())
        self.last_reward_velocity_mean = float(normalized_velocity_reward.mean().item())
        self.last_reward_lattice_mean = float(normalized_lattice_reward.mean().item())
        self.last_reward_size_weight_mean = float(size_weights.mean().item())
        self._probe_cache = None

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "num_model_steps": self.num_model_steps,
            "reward_baseline": self.reward_baseline,
            "reward_baseline_initialized": self.reward_baseline_initialized,
            "selected_probe_indices": self.selected_probe_indices,
            "probe_history": self._probe_history,
            "feature_dim": self._feature_dim,
        }
        if self.policy is not None and self.policy_optimizer is not None:
            state["policy_state_dict"] = self.policy.state_dict()
            state["policy_optimizer_state_dict"] = self.policy_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return

        self.num_model_steps = int(state_dict.get("num_model_steps", 0))
        self.reward_baseline = float(state_dict.get("reward_baseline", 0.0))
        self.reward_baseline_initialized = bool(state_dict.get("reward_baseline_initialized", False))
        selected_probe_indices = state_dict.get("selected_probe_indices")
        if selected_probe_indices is not None:
            self.selected_probe_indices = selected_probe_indices.to(dtype=torch.long, device="cpu")
        probe_history = state_dict.get("probe_history")
        if probe_history is not None:
            self._probe_history = [
                item.to(device="cpu", dtype=torch.float64)
                for item in probe_history
            ]

        if self.policy is None or self.policy_optimizer is None:
            self._pending_state_dict = state_dict
            return

        policy_state = state_dict.get("policy_state_dict")
        optimizer_state = state_dict.get("policy_optimizer_state_dict")
        if policy_state is not None:
            self.policy.load_state_dict(policy_state)
        if optimizer_state is not None:
            self.policy_optimizer.load_state_dict(optimizer_state)

    def diagnostics(self) -> dict[str, float]:
        selected_times = self.reward_probe_times[self.selected_probe_indices].to(dtype=torch.float64)
        return {
            "time_sampler/reward_mean": float(self.last_reward_mean),
            "time_sampler/reward_std": float(self.last_reward_std),
            "time_sampler/policy_loss": float(self.last_policy_loss),
            "time_sampler/policy_entropy": float(self.last_entropy),
            "time_sampler/policy_alpha_mean": float(self.last_alpha_mean),
            "time_sampler/policy_beta_mean": float(self.last_beta_mean),
            "time_sampler/sampled_t_mean": float(self.last_sampled_t_mean),
            "time_sampler/sampled_t_std": float(self.last_sampled_t_std),
            "time_sampler/sampled_t_min": float(self.last_sampled_t_min),
            "time_sampler/sampled_t_max": float(self.last_sampled_t_max),
            "time_sampler/policy_baseline": float(self.reward_baseline),
            "time_sampler/reward_velocity_mean": float(self.last_reward_velocity_mean),
            "time_sampler/reward_lattice_mean": float(self.last_reward_lattice_mean),
            "time_sampler/reward_size_weight_mean": float(self.last_reward_size_weight_mean),
            "time_sampler/policy_active": float(self._policy_active()),
            "time_sampler/policy_selected_t_min": float(selected_times.min().item()),
            "time_sampler/policy_selected_t_max": float(selected_times.max().item()),
        }
