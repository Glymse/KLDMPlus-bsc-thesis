from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kldmPlus.utils.time_sampler import AdaptiveReinforcePaperTimeSampler


class _CountingProbeModel:
    def __init__(self) -> None:
        self.prepare_calls = 0

    def prepare_training_batch(self, *, batch, t, lattice_noise, velocity_noise, position_noise):
        del lattice_noise, velocity_noise, position_noise
        self.prepare_calls += 1
        return SimpleNamespace(num_graphs=int(batch.num_graphs), t=t)

    def loss_from_prepared(self, prepared):
        losses = prepared.t.reshape(-1).new_full((int(prepared.num_graphs),), float(self.prepare_calls))
        return losses.mean(), {"loss_graph": losses}


def test_paper_sampler_default_probe_indices_are_spread_before_history_is_ready() -> None:
    sampler = AdaptiveReinforcePaperTimeSampler(
        reward_active_times=3,
        feature_selection_min_history=4,
    )
    sampler.reward_probe_times = torch.tensor(
        [
            0.001,
            0.002,
            0.004,
            0.008,
            0.015,
            0.030,
            0.060,
            0.100,
            0.180,
            0.320,
            0.550,
            0.800,
            1.000,
        ],
        dtype=torch.float64,
    )

    selected = sampler._default_probe_indices()

    assert selected.tolist() == [3, 6, 9]


def test_paper_sampler_feature_selection_uses_history() -> None:
    sampler = AdaptiveReinforcePaperTimeSampler(
        reward_active_times=1,
        reward_history_size=8,
        feature_selection_min_history=4,
    )
    sampler.reward_probe_times = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
    sampler._feature_history = [
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
    ]
    sampler._target_history = [
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.5], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    ]

    selected = sampler._select_probe_indices()

    assert selected.tolist() == [0]


def test_paper_sampler_separates_feature_and_reward_probe_counts() -> None:
    sampler = AdaptiveReinforcePaperTimeSampler(
        reward_candidate_times=5,
        reward_active_times=2,
        feature_probe_graphs=1,
    )
    sampler.selected_probe_indices = torch.tensor([1, 3], dtype=torch.long)
    batch = SimpleNamespace(
        pos=torch.zeros(2, 3),
        l=torch.zeros(1, 6),
        num_graphs=1,
    )
    model = _CountingProbeModel()

    cache = sampler._prepare_probe_cache(batch=batch, model=model)

    assert len(cache["feature_prepared"]) == 5
    assert len(cache["reward_prepared"]) == 2
    assert model.prepare_calls == 7
