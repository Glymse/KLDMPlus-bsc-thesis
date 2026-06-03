from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch

try:
    from torch_geometric.data import Batch, Data
except ImportError:  # pragma: no cover
    Batch = Data = Any


@dataclass(frozen=True)
class BatchTimes:
    """
    Time views for one PyG batch.

    KLDM uses one diffusion time per material/crystal graph. A PyG batch packs
    many materials together, so the same material time must be reshaped for the
    branches that operate at different levels.

    graph:
        [G, 1], used by the score network.

    lattice:
        [G], same time squeezed, used by the lattice diffusion.

    nodes:
        [N], graph time copied to each atom/node, used by TDM.

    Example:
        graph = [[0.2],
                 [0.8]]

        batch.batch = [0, 0, 0, 1, 1]

        nodes = [0.2, 0.2, 0.2, 0.8, 0.8]

    So nodes are not given independent times. They inherit the time of their
    material graph.
    """

    graph: torch.Tensor
    lattice: torch.Tensor
    nodes: torch.Tensor


@dataclass(frozen=True)
class SamplingTimes:
    """
    Time views for one reverse-sampling step.

    Reverse sampling moves from a noisy time t to a less noisy time t_next:

        t -> t_next, with dt = t - t_next > 0

    now:
        BatchTimes at t.

    next:
        BatchTimes at t_next.

    Algorithm 3 mostly uses now.
    Algorithm 4 uses now for the predictor and next for the corrector.
    """

    step: int
    t: torch.Tensor
    t_next: torch.Tensor
    dt: float
    now: BatchTimes
    next: BatchTimes

    @property
    def t_float(self) -> float:
        return float(self.t.item())

    @property
    def t_next_float(self) -> float:
        return float(self.t_next.item())


def sample_times(
    batch: Data | Batch,
    *,
    lower_bound: float = 1e-3,
) -> torch.Tensor:
    """
    Sample one training time per material graph:

        t ~ Uniform(lower_bound, 1)

    The whole material shares this time. We do not sample separate times per
    atom, because positions, velocities, and lattice should describe one
    consistent noisy crystal state.

    Output shape:
        [G, 1]

    Example:
        For three materials, this could return:

            [[0.14],
             [0.83],
             [0.51]]
    """
    return lower_bound + (1.0 - lower_bound) * torch.rand(
        batch.num_graphs,
        1,
        device=batch.pos.device,
        dtype=batch.pos.dtype,
    )


def make_times(
    batch: Data | Batch,
    t: torch.Tensor | float,
) -> BatchTimes:
    """
    Convert input time to the three shapes used by the KLDM pipeline.

    Accepted input:
        scalar:
            one shared time for all graphs.

        [G]:
            one time per graph.

        [G, 1]:
            one time per graph, already in score-network shape.

    Returned:
        graph   [G, 1]
        lattice [G]
        nodes   [N]

    The important step is the node view:

        nodes = graph[batch.batch].squeeze(-1)

    This copies each material's graph time onto all atoms belonging to that
    material.
    """
    graph = torch.as_tensor(t, device=batch.pos.device, dtype=batch.pos.dtype)

    # Scalar time: use the same time for every material in the batch.
    if graph.ndim == 0:
        graph = graph.expand(batch.num_graphs, 1)

    # Vector time: either one shared value [1], or one value per graph [G].
    elif graph.ndim == 1:
        if graph.numel() == 1:
            graph = graph.expand(batch.num_graphs)[:, None]
        else:
            graph = graph[:, None]

    # Matrix time must already be [G, 1].
    elif graph.ndim != 2 or graph.shape[-1] != 1:
        raise ValueError(
            f"Expected time shape scalar, [G], or [G, 1], got {tuple(graph.shape)}."
        )

    if graph.shape[0] != batch.num_graphs:
        raise ValueError(
            f"Expected {batch.num_graphs} graph times, got {graph.shape[0]}."
        )

    return BatchTimes(
        graph=graph,
        lattice=graph.squeeze(-1),
        nodes=graph[batch.batch].squeeze(-1),
    )


def sampling_grid(
    batch: Data | Batch,
    *,
    n_steps: int,
    t_start: float,
    t_final: float,
    rho: float = 1.0,
) -> torch.Tensor:
    """
    Create the decreasing reverse-sampling grid.

    Sampling is deterministic in time: we start near pure noise and walk toward
    the clean end of the diffusion process.

    Example:
        n_steps = 4

        grid = [1.0, 0.75, 0.50, 0.25, 0.001]

    Each step uses:
        t      = grid[step]
        t_next = grid[step + 1]
        dt     = t - t_next > 0

    rho:
        1.0 gives linear spacing.
        >1 gives more points near t_final.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if not (0.0 < t_final < t_start <= 1.0):
        raise ValueError("Expected 0 < t_final < t_start <= 1.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")

    u = torch.linspace(
        1.0,
        0.0,
        n_steps + 1,
        device=batch.pos.device,
        dtype=batch.pos.dtype,
    )

    return t_final + (t_start - t_final) * u.pow(rho)


def iter_sampling_times(
    batch: Data | Batch,
    grid: torch.Tensor,
) -> Iterator[SamplingTimes]:
    """

    KLDM sampling starts at a noisy time close to 1 and moves toward a cleaner
    time close to 0. We therefore build a decreasing time grid

        grid[0] > grid[1] > ... > grid[K]

    At each step we define
        t      = current noisier time
        t_next = next slightly cleaner time
        dt     = t - t_next > 0

    So dt is stored as a positive backward step size. We simply keep the same
    reverse trajectory but store its magnitude as a positive number, then write
    the update formulas with that convention.

    """
    # Iterate over neighboring pairs in the decreasing grid:
    # grid[0] -> grid[1], grid[1] -> grid[2], ...
    for step in range(grid.numel() - 1):
        # Current/noisier time and next/slightly cleaner time.
        t = grid[step]
        t_next = grid[step + 1]

        # Positive backward step size used by the reverse updates.
        dt = float((t - t_next).item())

        yield SamplingTimes(
            step=step,
            t=t,
            t_next=t_next,
            dt=dt,
            now=make_times(batch, t),
            next=make_times(batch, t_next),
        )
