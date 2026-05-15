# We use ruff
# To format our code!!!
# Remember to write this in paper if relevant.

from __future__ import annotations

import sys
import time
import warnings
import math
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch

try:
    from pymatgen.core import Element, Lattice, Structure
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = None

from kldmPlus.data.transform import ContinuousIntervalLattice
from kldmPlus.diffusionModels.continuous import (
    ContinuousDiffusion,
    ContinuousMattergenVPDiffusion,
    ContinuousVPDiffusion,
)
from kldmPlus.diffusionModels.tdm import TrivialisedDiffusion as TDM
from kldmPlus.scoreNetwork.scoreNetwork import CSPVNet
from kldmPlus.sgdpnp import SGDPnPConfig, sample_kldm_dpnp_sg
from kldmPlus.symmetry import (
    _pcs_state_rank_key,
    build_pyxtal_wyckoff_result,
    initialize_constrained_template_states,
    materialize_pcs_state,
    pcs_projected_objective,
    sample_pcs_step_mala,
    select_requested_template_state,
    select_requested_template_states,
    validate_requested_space_group,
    vanilla_structure_to_model_tensors,
)
from kldmPlus.utils.device import get_default_device
from kldmPlus.utils.time import BatchTimes, iter_sampling_times, make_times, sampling_grid


@dataclass
class PreparedTrainingBatch:
    """
    Fixed noisy training bundle for one KLDM++ loss evaluation.

    Inspired by the adaptive paper:
    the REINFORCE reward must compare before/after model losses on the same
    corruption, so the sampler needs a reusable container for noisy states and
    targets.
    """

    times: BatchTimes
    v_t: torch.Tensor
    f_t: torch.Tensor
    l_t: torch.Tensor
    target_v: torch.Tensor
    target_l: torch.Tensor
    atomic_numbers: torch.Tensor
    node_index: torch.Tensor
    edge_node_index: torch.Tensor
    num_graphs: int
    lattice_representation: str


def _lengths_angles_to_cell_matrix(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a 3x3 lattice matrix from lengths and angles."""
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(eps)

    zeros = torch.zeros_like(a)
    ax = a
    bx = b * cos_gamma
    by = b * sin_gamma
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sq = (c.square() - cx.square() - cy.square()).clamp_min(eps)
    cz = torch.sqrt(cz_sq)

    row_a = torch.stack([ax, zeros, zeros], dim=-1)
    row_b = torch.stack([bx, by, zeros], dim=-1)
    row_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([row_a, row_b, row_c], dim=-2)


def _matrix_log_spd(matrix: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Stable matrix logarithm for symmetric positive definite 3x3 matrices."""
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(eps)
    log_diag = torch.diag_embed(torch.log(eigvals))
    return eigvecs @ log_diag @ eigvecs.transpose(-1, -2)


def _paper_k_from_cell(cell: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Map a cell to the DiffCSP++ invariant 6D k-basis coefficients.

    Following the paper's lattice construction, we form the Gram matrix
    `G = L L^T`, take `S = 0.5 * log(G)`, and then read off the coefficients
    in the six-dimensional symmetric basis:

        S = [[k4 + k5 + k6,      k1,          k2],
             [k1,               -k4 + k5 + k6, k3],
             [k2,                k3,         -2k5 + k6]]
    """
    gram = cell @ cell.transpose(-1, -2)
    s_matrix = 0.5 * _matrix_log_spd(gram, eps=eps)

    s00 = s_matrix[..., 0, 0]
    s11 = s_matrix[..., 1, 1]
    s22 = s_matrix[..., 2, 2]
    k1 = s_matrix[..., 0, 1]
    k2 = s_matrix[..., 0, 2]
    k3 = s_matrix[..., 1, 2]
    k4 = 0.5 * (s00 - s11)
    k5 = (s00 + s11 - 2.0 * s22) / 6.0
    k6 = (s00 + s11 + s22) / 3.0
    return torch.stack([k1, k2, k3, k4, k5, k6], dim=-1)


def _decode_lattice_matrix(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: ContinuousIntervalLattice | None,
) -> torch.Tensor:
    """Decode one or more 6D lattice states into 3x3 cell matrices."""
    if lattice_transform is not None and hasattr(lattice_transform, "invert_to_matrix"):
        matrix = lattice_transform.invert_to_matrix(l=l, num_atoms=num_atoms)
        return matrix.reshape(*l.shape[:-1], 3, 3)

    if lattice_transform is not None:
        lengths, angles = lattice_transform.invert_to_lengths_angles(l=l, num_atoms=num_atoms)
    else:
        lengths = torch.exp(l[..., :3])
        angles = torch.atan(l[..., 3:]) + torch.pi / 2.0

    return _lengths_angles_to_cell_matrix(lengths=lengths, angles=angles)

def _space_group_k_constraint(
    *,
    space_group_number: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the DiffCSP++ k-basis constraint mask/target for one space group.

    The constraints follow Table 1 in "Space Group Constrained Crystal
    Generation" (ICLR 2024), where the 230 space groups are grouped into six
    crystal families and only a subset of the `k` coordinates is fixed.
    """
    if not 1 <= int(space_group_number) <= 230:
        raise ValueError(f"space_group must be in [1, 230], got {space_group_number}.")

    mask = torch.zeros(6, device=device, dtype=dtype)
    target = torch.zeros(6, device=device, dtype=dtype)

    sg = int(space_group_number)
    if 3 <= sg <= 15:
        mask[[0, 2]] = 1.0
    elif 16 <= sg <= 74:
        mask[[0, 1, 2]] = 1.0
    elif 75 <= sg <= 142:
        mask[[0, 1, 2, 3]] = 1.0
    elif 143 <= sg <= 194:
        mask[[0, 1, 2, 3]] = 1.0
        target[0] = -torch.log(torch.tensor(3.0, device=device, dtype=dtype)) / 4.0
    elif 195 <= sg <= 230:
        mask[[0, 1, 2, 3, 4]] = 1.0

    return mask, target


@lru_cache(maxsize=None)
def _cached_space_group_operations(
    space_group_number: int,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Cache non-identity symmetry operations for one space-group number."""
    try:
        from pymatgen.symmetry.groups import SpaceGroup
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sample_CSP_algorithm5 requires pymatgen to decode space-group symmetry operations.",
        ) from exc

    if not 1 <= int(space_group_number) <= 230:
        raise ValueError(f"space_group must be in [1, 230], got {space_group_number}.")

    space_group = SpaceGroup.from_int_number(int(space_group_number))
    ops: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for op in space_group.symmetry_ops:
        rotation = tuple(tuple(float(value) for value in row) for row in op.rotation_matrix.tolist())
        translation = tuple(float(value) for value in op.translation_vector.tolist())
        is_identity = rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) and all(
            abs(value) < 1e-12 for value in translation
        )
        if not is_identity:
            ops.append((rotation, translation))
    return tuple(ops)


def _space_group_operations_as_tensors(
    *,
    space_group_number: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.tensor(rotation, device=device, dtype=dtype),
            torch.tensor(translation, device=device, dtype=dtype),
        )
        for rotation, translation in _cached_space_group_operations(int(space_group_number))
    ]


def _torus_pairwise_distance_sq(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = source.unsqueeze(1) - target.unsqueeze(0)
    delta = delta - torch.round(delta)
    return delta.square().sum(dim=-1)


def _match_cost_matrix(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a one-to-one assignment for a square cost matrix."""
    detached = cost_matrix.detach()
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover
        linear_sum_assignment = None

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(detached.cpu().numpy())
        return (
            torch.as_tensor(row_idx, device=cost_matrix.device, dtype=torch.long),
            torch.as_tensor(col_idx, device=cost_matrix.device, dtype=torch.long),
        )

    remaining_rows = list(range(detached.shape[0]))
    remaining_cols = list(range(detached.shape[1]))
    chosen_rows: list[int] = []
    chosen_cols: list[int] = []
    detached_cpu = detached.cpu()

    while remaining_rows:
        submatrix = detached_cpu[remaining_rows][:, remaining_cols]
        flat_index = int(torch.argmin(submatrix).item())
        sub_rows = submatrix.shape[1]
        row_pos = flat_index // sub_rows
        col_pos = flat_index % sub_rows
        chosen_rows.append(remaining_rows.pop(row_pos))
        chosen_cols.append(remaining_cols.pop(col_pos))

    row_idx = torch.tensor(chosen_rows, device=cost_matrix.device, dtype=torch.long)
    col_idx = torch.tensor(chosen_cols, device=cost_matrix.device, dtype=torch.long)
    order = torch.argsort(row_idx)
    return row_idx[order], col_idx[order]


def _species_matched_torus_energy(
    *,
    current_frac: torch.Tensor,
    transformed_frac: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    energy = current_frac.new_zeros(())
    for atomic_number in torch.unique(atomic_numbers.detach(), sorted=True).tolist():
        species_mask = atomic_numbers == int(atomic_number)
        species_indices = torch.nonzero(species_mask, as_tuple=False).squeeze(-1)
        if species_indices.numel() == 0:
            continue

        cost_matrix = _torus_pairwise_distance_sq(
            transformed_frac[species_indices],
            current_frac[species_indices],
        )
        row_idx, col_idx = _match_cost_matrix(cost_matrix)
        delta = (
            transformed_frac[species_indices[row_idx]]
            - current_frac[species_indices[col_idx]]
        )
        delta = delta - torch.round(delta)
        energy = energy + delta.square().sum()
    return energy


def _atomic_multiset_matches(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape:
        return False
    if left.numel() == 0:
        return True
    left_sorted = torch.sort(left.detach().to(device="cpu", dtype=torch.long)).values
    right_sorted = torch.sort(right.detach().to(device="cpu", dtype=torch.long)).values
    return bool(torch.equal(left_sorted, right_sorted))


def _template_site_shape_signature(state: Any) -> tuple[tuple[int, str, int, int], ...]:
    return tuple(
        sorted(
            (
                int(site.atomic_number),
                str(site.label),
                int(site.multiplicity),
                int(site.dof),
            )
            for site in state.template.site_templates
        )
    )


def _species_label_signature_labels(signature: tuple[tuple[int, str], ...] | None) -> list[str]:
    if not signature:
        return []
    labels: list[str] = []
    for atomic_number, label in signature:
        try:
            symbol = Element.from_Z(int(atomic_number)).symbol if Element is not None else str(int(atomic_number))
        except Exception:
            symbol = str(int(atomic_number))
        labels.append(f"{symbol}@{label}")
    return labels


def _pcs_state_signature_labels(state: Any) -> list[str]:
    signature = getattr(state, "template_species_orbit_signature", None)
    return _species_label_signature_labels(signature)


def _pyxtal_site_shape_signature(result: Any) -> tuple[tuple[int, str, int, int], ...]:
    return tuple(
        sorted(
            (
                int(z),
                str(label),
                int(mult),
                int(dof),
            )
            for z, label, mult, dof in zip(
                result.anchor_atomic_numbers.tolist(),
                result.site_labels,
                result.site_multiplicities.tolist(),
                result.site_dofs.tolist(),
            )
        )
    )


def _site_shape_mismatch_count(
    *,
    expected_signature: tuple[tuple[Any, ...], ...],
    recovered_signature: tuple[tuple[Any, ...], ...],
) -> int:
    expected_counter = Counter(expected_signature)
    recovered_counter = Counter(recovered_signature)
    mismatch = 0
    for key in set(expected_counter) | set(recovered_counter):
        mismatch += abs(int(expected_counter.get(key, 0)) - int(recovered_counter.get(key, 0)))
    return int(mismatch)


def _projected_structure_physical_stats(
    structure: Structure,
    *,
    cutoff: float,
) -> tuple[float, float]:
    try:
        distances = np.asarray(structure.distance_matrix, dtype=float)
    except Exception as exc:
        raise RuntimeError("PCS materialization could not compute pair distances.") from exc

    distances = distances + np.diag(np.full(distances.shape[0], float(cutoff) + 10.0))
    min_pair_distance = float(np.min(distances))
    volume = float(structure.volume)
    return min_pair_distance, volume


def _projected_structure_debug_stats(
    *,
    structure: Structure,
    reference_volume: float | None,
    cutoff: float,
) -> tuple[float, float, float | None]:
    min_pair_distance, volume = _projected_structure_physical_stats(
        structure,
        cutoff=cutoff,
    )
    volume_ratio: float | None = None
    if reference_volume is not None and reference_volume > 0.0:
        volume_ratio = volume / max(float(reference_volume), 1e-8)
    return min_pair_distance, volume, volume_ratio


def _enforce_projected_physical_guards(
    *,
    structure: Structure,
    reference_volume: float | None,
    hard_min_distance: float,
    hard_volume_ratio_min: float,
    hard_volume_ratio_max: float,
) -> None:
    if (
        float(hard_min_distance) <= 0.0
        and float(hard_volume_ratio_min) <= 0.0
        and float(hard_volume_ratio_max) <= 0.0
    ):
        return

    cutoff = max(float(hard_min_distance), 0.0)
    min_pair_distance, volume, volume_ratio = _projected_structure_debug_stats(
        structure=structure,
        reference_volume=reference_volume,
        cutoff=max(cutoff, 1e-6),
    )
    if float(hard_min_distance) > 0.0 and min_pair_distance < float(hard_min_distance):
        raise RuntimeError(
            "PCS materialization violates the hard pair-distance guard "
            f"({min_pair_distance:.4f} < {float(hard_min_distance):.4f})."
        )

    if volume_ratio is None:
        return

    if float(hard_volume_ratio_min) > 0.0 and volume_ratio < float(hard_volume_ratio_min):
        raise RuntimeError(
            "PCS materialization violates the hard lower volume guard "
            f"({volume_ratio:.4f} < {float(hard_volume_ratio_min):.4f}, "
            f"volume={volume:.4f}, reference_volume={float(reference_volume):.4f})."
        )
    if float(hard_volume_ratio_max) > 0.0 and volume_ratio > float(hard_volume_ratio_max):
        raise RuntimeError(
            "PCS materialization violates the hard upper volume guard "
            f"({volume_ratio:.4f} > {float(hard_volume_ratio_max):.4f}, "
            f"volume={volume:.4f}, reference_volume={float(reference_volume):.4f})."
        )


def _symmop_list_to_tensor_ops(
    symmetry_ops: list[Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ops: list[tuple[torch.Tensor, torch.Tensor]] = []
    for op in symmetry_ops:
        rotation = torch.as_tensor(op.rotation_matrix, device=device, dtype=dtype)
        translation = torch.as_tensor(op.translation_vector, device=device, dtype=dtype)
        is_identity = torch.allclose(rotation, torch.eye(3, device=device, dtype=dtype)) and torch.allclose(
            translation,
            torch.zeros(3, device=device, dtype=dtype),
        )
        if is_identity:
            continue
        ops.append((rotation, translation))
    return ops


def _clip_rowwise_norm(x: torch.Tensor, max_norm: float, eps: float = 1e-12) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(x)

    flat = x.reshape(x.shape[0], -1)
    norms = torch.linalg.norm(flat, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / norms.clamp_min(eps), max=1.0)
    clipped = flat * scale
    return clipped.reshape_as(x)


def _direct_torus_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.numel() == 0:
        return float("nan")
    delta = left.detach() - right.detach()
    delta = delta - torch.round(delta)
    return float(torch.sqrt(delta.square().sum(dim=-1).mean()).detach().cpu().item())


def _direct_tensor_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.numel() == 0:
        return float("nan")
    delta = left.detach() - right.detach()
    return float(torch.sqrt(delta.square().mean()).detach().cpu().item())


def _debug_numeric_summary(values: list[float]) -> tuple[int, float, float, float]:
    finite = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if finite.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return int(finite.size), float(finite.mean()), float(finite.max()), float(finite.min())


class ModelKLDM(nn.Module):
    """
    KLDM model

    """

    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 1e-6,
        wrapped_normal_K: int = 3,
        tdm_n_sigmas: int | None = None,
        tdm_compute_sigma_norm: bool = True,
        tdm_velocity_scale: float | None = None,
        tdm_sigma_norm_estimator: str = "quadrature",
        tdm_sigma_norm_density_K: int | None = None,
        tdm_sigma_norm_grid_points: int = 8193,
        tdm_sigma_norm_mc_samples: int = 20000,
        tdm_centered_sigma_norm_correction: bool = False,
        lattice_parameterization: str = "eps",
        lattice_diffusion_type: str = "VP",
        lattice_representation: str = "kldm",
        mattergen_lattice_c: float | None = None,
        mattergen_lattice_nu: float | None = None,
        mattergen_pos_loss_weight: float | None = None,
        mattergen_cell_loss_weight: float | None = None,
        mattergen_pos_loss_reduce: str | None = None,
        *,
        score_network_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        #Load network from our config.
        self.score_network_kwargs = dict(score_network_kwargs)
        self.score_network = CSPVNet(**self.score_network_kwargs)

        self.tdm = TDM(
            eps=eps,
            wrapped_normal_K=wrapped_normal_K,
            n_sigmas=(2000 if self.device.type == "cuda" else 512) if tdm_n_sigmas is None else int(tdm_n_sigmas),
            compute_sigma_norm=tdm_compute_sigma_norm,
            velocity_scale=tdm_velocity_scale,
            sigma_norm_estimator=tdm_sigma_norm_estimator,
            sigma_norm_density_K=tdm_sigma_norm_density_K,
            sigma_norm_grid_points=tdm_sigma_norm_grid_points,
            sigma_norm_mc_samples=tdm_sigma_norm_mc_samples,
            centered_sigma_norm_correction=tdm_centered_sigma_norm_correction,
        )
        self.diffusion_l = self._build_lattice_diffusion(
            lattice_diffusion_type=lattice_diffusion_type,
            eps=eps,
            lattice_parameterization=lattice_parameterization,
            mattergen_lattice_c=mattergen_lattice_c,
            mattergen_lattice_nu=mattergen_lattice_nu,
        )
        self.eps = eps
        self.lattice_parameterization = lattice_parameterization
        self.lattice_diffusion_type = lattice_diffusion_type
        self.lattice_representation = lattice_representation
        self.mattergen_pos_loss_weight = (
            0.1 if mattergen_pos_loss_weight is None else float(mattergen_pos_loss_weight)
        )
        self.mattergen_cell_loss_weight = (
            1.0 if mattergen_cell_loss_weight is None else float(mattergen_cell_loss_weight)
        )
        self.mattergen_pos_loss_reduce = (
            "sum" if mattergen_pos_loss_reduce is None else str(mattergen_pos_loss_reduce)
        )
        if self.mattergen_pos_loss_reduce not in {"sum", "mean"}:
            raise ValueError("mattergen_pos_loss_reduce must be 'sum' or 'mean'.")

    @staticmethod
    def _build_lattice_diffusion(
        *,
        lattice_diffusion_type: str,
        eps: float,
        lattice_parameterization: str,
        mattergen_lattice_c: float | None,
        mattergen_lattice_nu: float | None,
    ) -> ContinuousDiffusion:
        if lattice_diffusion_type == "VP":
            diffusion_cls = ContinuousVPDiffusion
            diffusion_kwargs: dict[str, Any] = {}
        elif lattice_diffusion_type == "mattergenVP":
            diffusion_cls = ContinuousMattergenVPDiffusion
            if mattergen_lattice_c is None or mattergen_lattice_nu is None:
                raise ValueError(
                    "mattergenVP requires mattergen_lattice_c and mattergen_lattice_nu.",
                )
            diffusion_kwargs = {
                "c": float(mattergen_lattice_c),
                "nu": float(mattergen_lattice_nu),
            }
        else:
            raise ValueError(
                "lattice_diffusion_type must be 'VP' or 'mattergenVP'.",
            )

        return diffusion_cls(
            eps=eps,
            parameterization=lattice_parameterization,
            **diffusion_kwargs,
        )

    # ============================================================================
    # ALGORITHM 1
    # ============================================================================

    def algorithm1_training_targets(
        self,
        batch: Data | Batch,
        times: BatchTimes,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """
        Algorithm 1 in KLDM:
        sample noisy variables and score targets.
        """
        index = batch.batch

        # Diffuse lattice, KLDM Alg. 1
        l_t, eps_l = self.diffusion_l.forward_sample(
            t=times.lattice,
            x0=batch.l,
            num_atoms=batch.num_atoms,
        )
        target_l = self.diffusion_l.training_target(
            t=times.lattice,
            x0=batch.l,
            noise=eps_l,
            num_atoms=batch.num_atoms,
        )

        f_t, v_t, epsilon_v, epsilon_r, r_t = self.tdm.sample_noisy_state(
            t=times.nodes,
            f0=batch.pos,
            index=index, # the reason we give the index is because, it has if a batch has 2 crystals with 3 and 2 atoms then index = [0, 0, 0, 1, 1]
                         # THis is used to zero-center velocity noise per graph
        )

        target_v = self.tdm.build_simplified_training_velocity_score(
            t=times.nodes,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )


        return (v_t, f_t, l_t), (target_v, target_l)

    # ============================================================================
    # Loss calculators for algorithm 2
    # ============================================================================

    def mse_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plain MSE, averaged over feature dims.
        """
        loss = F.mse_loss(pred, target, reduction="none")
        return loss.reshape(loss.shape[0], -1).mean(dim=1)

    def mattergen_lattice_mse_6d_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted 6D MSE that matches averaging over the full symmetric 3x3 cell.
        """
        # Code segment inspired from mattergen
        # (mattergen/diffusion/training/field_loss.py:118-152,
        #  mattergen/common/loss.py:35-40).
        #
        # Official MatterGen computes denoising loss on the full `cell` tensor and
        # then averages over all matrix entries. In the KLDM port we store only the
        # six unique entries of the symmetric matrix, so off-diagonal terms need a
        # factor of 2 to match the full 3x3 averaging.
        weights = pred.new_tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        loss = (pred - target).square() * weights
        return loss.sum(dim=-1) / 9.0

    # ============================================================================
    # ALGORITHM 2
    # ============================================================================

    def prepare_training_batch(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        *,
        lattice_noise: torch.Tensor | None = None,
        velocity_noise: torch.Tensor | None = None,
        position_noise: torch.Tensor | None = None,
    ) -> PreparedTrainingBatch:
        """
        Build one fixed noisy training example bundle for Algorithm 2.

        Inspired by the adaptive paper:
        the sampler reward compares before/after improvement at probe times,
        which only makes sense when both evaluations reuse the same corruption.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        index = batch.batch
        times = make_times(batch, t)

        l_t, eps_l = self.diffusion_l.forward_sample(
            t=times.lattice,
            x0=batch.l,
            noise=lattice_noise,
            num_atoms=batch.num_atoms,
        )
        target_l = self.diffusion_l.training_target(
            t=times.lattice,
            x0=batch.l,
            noise=eps_l,
            num_atoms=batch.num_atoms,
        )

        f_t, v_t, _epsilon_v, _epsilon_r, r_t = self.tdm.sample_noisy_state(
            t=times.nodes,
            f0=batch.pos,
            index=index,
            epsilon_v=velocity_noise,
            epsilon_r=position_noise,
        )
        target_v = self.tdm.build_simplified_training_velocity_score(
            t=times.nodes,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )

        return PreparedTrainingBatch(
            times=times,
            v_t=v_t,
            f_t=f_t,
            l_t=l_t,
            target_v=target_v,
            target_l=target_l,
            atomic_numbers=batch.atomic_numbers,
            node_index=index,
            edge_node_index=batch.edge_node_index,
            num_graphs=int(batch.num_graphs),
            lattice_representation=self.lattice_representation,
        )

    def loss_from_prepared(
        self,
        prepared: PreparedTrainingBatch,
        *,
        time_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Evaluate the KLDM++ denoising loss from a fixed noisy bundle.

        This is the loss-only half of Algorithm 2 and pairs with
        `prepare_training_batch(...)` so adaptive samplers can reuse exact
        probe corruptions across before/after reward measurements.
        """
        preds = self.score_network(
            t=prepared.times.graph,
            pos=prepared.f_t,
            v=prepared.v_t,
            h=prepared.atomic_numbers,
            l=prepared.l_t,
            node_index=prepared.node_index,
            edge_node_index=prepared.edge_node_index,
        )

        out_v = preds["v"]
        out_l = preds["l"]

        loss_v_node = self.mse_loss_per_sample(out_v, prepared.target_v)
        if prepared.lattice_representation == "mattergen":
            loss_l_graph = self.mattergen_lattice_mse_6d_per_sample(out_l, prepared.target_l)
        else:
            loss_l_graph = self.mse_loss_per_sample(out_l, prepared.target_l)

        loss_v_sum = torch.zeros(
            prepared.num_graphs,
            device=loss_v_node.device,
            dtype=loss_v_node.dtype,
        )
        loss_v_sum = loss_v_sum.index_add(0, prepared.node_index, loss_v_node)

        counts = torch.bincount(prepared.node_index, minlength=prepared.num_graphs).to(
            device=loss_v_node.device,
            dtype=loss_v_node.dtype,
        ).clamp_min(1.0)

        if prepared.lattice_representation == "mattergen":
            # Official MatterGen uses a weighted summed-field objective:
            # `cell: 1.0`, `pos: 0.1`, with `reduce=sum` over nodes for the
            # position field. The KLDM port still trains the TDM velocity head,
            # but matching the graph-level reduction and branch weighting keeps
            # the overall objective aligned with the MatterGen implementation.
            if self.mattergen_pos_loss_reduce == "sum":
                loss_v_graph = loss_v_sum
            else:
                loss_v_graph = loss_v_sum / counts
            loss_v_weighted_graph = self.mattergen_pos_loss_weight * loss_v_graph
            loss_l_weighted_graph = self.mattergen_cell_loss_weight * loss_l_graph
            loss_graph = (
                loss_v_weighted_graph
                + loss_l_weighted_graph
            )
        else:
            loss_v_graph = loss_v_sum / counts
            loss_v_weighted_graph = loss_v_graph
            loss_l_weighted_graph = loss_l_graph
            loss_graph = loss_v_graph + loss_l_graph

        if time_weight is not None:
            weight = time_weight.reshape(-1).to(device=loss_graph.device, dtype=loss_graph.dtype)
            total_loss = (weight * loss_graph).mean()
        else:
            total_loss = loss_graph.mean()

        metrics = {
            "loss": total_loss.detach(),
            "loss_v": loss_v_graph.mean().detach(),
            "loss_l": loss_l_graph.mean().detach(),
            "loss_v_weighted": loss_v_weighted_graph.mean().detach(),
            "loss_l_weighted": loss_l_weighted_graph.mean().detach(),
            "loss_graph": loss_graph.detach(),
            "loss_v_graph": loss_v_graph.detach(),
            "loss_l_graph": loss_l_graph.detach(),
            "loss_v_weighted_graph": loss_v_weighted_graph.detach(),
            "loss_l_weighted_graph": loss_l_weighted_graph.detach(),
        }
        return total_loss, metrics

    def algorithm2_loss(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        time_weight: torch.Tensor | None = None,
        debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Algorithm 2 in KLDM:
        network prediction + denoising score matching loss.
        """
        del debug
        prepared = self.prepare_training_batch(batch=batch, t=t)
        return self.loss_from_prepared(prepared, time_weight=time_weight)

    def _reverse_lattice_sampling_step(
        self,
        *,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
        num_atoms: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.diffusion_l, ContinuousMattergenVPDiffusion):
            return self.diffusion_l.reverse_step_ancestral(
                t=t,
                x_t=x_t,
                pred=pred,
                dt=dt,
                num_atoms=num_atoms,
            )

        return self.diffusion_l.reverse_step(
            t=t,
            x_t=x_t,
            pred=pred,
            dt=dt,
            num_atoms=num_atoms,
        )

    def _symmetry_guidance_energy(
        self,
        *,
        batch: Batch | Data,
        node_index: torch.Tensor,
        f_t: torch.Tensor,
        l_t: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None,
        operations_by_graph: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(batch, "space_group"):
            raise ValueError("sample_CSP_algorithm5 requires batch.space_group.")

        frac = self.tdm.wrap_positions(f_t)
        coord_energy = frac.new_zeros(())
        lattice_energy = l_t.new_zeros(())

        space_groups = torch.as_tensor(batch.space_group, device=l_t.device, dtype=torch.long).reshape(-1)
        num_atoms_per_graph = torch.as_tensor(batch.num_atoms, device=l_t.device, dtype=torch.long).reshape(-1)

        for graph_idx in range(int(batch.num_graphs)):
            space_group_number = int(space_groups[graph_idx].item())
            if operations_by_graph is None:
                if space_group_number == 1:
                    operations = []
                else:
                    operations = _space_group_operations_as_tensors(
                        space_group_number=space_group_number,
                        device=f_t.device,
                        dtype=f_t.dtype,
                    )
            else:
                operations = operations_by_graph[graph_idx]

            node_mask = node_index == graph_idx
            current_frac = frac[node_mask]
            atomic_numbers = batch.atomic_numbers[node_mask]
            cell = _decode_lattice_matrix(
                l=l_t[graph_idx : graph_idx + 1],
                num_atoms=int(num_atoms_per_graph[graph_idx].item()),
                lattice_transform=lattice_transform,
            ).squeeze(0)
            if not torch.isfinite(cell).all():
                continue
            metric = cell @ cell.transpose(-1, -2)
            if not torch.isfinite(metric).all():
                continue

            graph_coord_energy = frac.new_zeros(())
            graph_lattice_energy = l_t.new_zeros(())

            for rotation, translation in operations:
                transformed = torch.remainder(current_frac @ rotation.transpose(0, 1) + translation, 1.0)
                graph_coord_energy = graph_coord_energy + _species_matched_torus_energy(
                    current_frac=current_frac,
                    transformed_frac=transformed,
                    atomic_numbers=atomic_numbers,
                )

                metric_residual = rotation.transpose(0, 1) @ metric @ rotation - metric
                metric_scale = metric.square().mean().clamp_min(self.eps)
                graph_lattice_energy = graph_lattice_energy + (
                    metric_residual.square().mean() / metric_scale
                )

            normalizer = max(int(current_frac.shape[0]) * len(operations), 1)
            if operations:
                coord_energy = coord_energy + graph_coord_energy / normalizer

            k_current = _paper_k_from_cell(cell, eps=self.eps)
            if torch.isfinite(k_current).all():
                k_mask, k_target = _space_group_k_constraint(
                    space_group_number=space_group_number,
                    device=k_current.device,
                    dtype=k_current.dtype,
                )
                constrained_dims = k_mask.sum().clamp_min(1.0)
                graph_lattice_energy = ((k_current - k_target) * k_mask).square().sum() / constrained_dims
                lattice_energy = lattice_energy + graph_lattice_energy

        return coord_energy, lattice_energy

    def symmetry_guidance_energy(
        self,
        *,
        batch: Batch | Data,
        f_t: torch.Tensor,
        l_t: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None = None,
        operations_by_graph: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> dict[str, torch.Tensor]:
        coord_energy, lattice_energy = self._symmetry_guidance_energy(
            batch=batch,
            node_index=batch.batch,
            f_t=f_t,
            l_t=l_t,
            lattice_transform=lattice_transform,
            operations_by_graph=operations_by_graph,
        )
        total_energy = coord_energy + lattice_energy
        return {
            "coord": coord_energy.detach(),
            "lattice": lattice_energy.detach(),
            "total": total_energy.detach(),
        }

    def _apply_symmetry_guidance(
        self,
        *,
        batch: Batch | Data,
        node_index: torch.Tensor,
        f_t: torch.Tensor,
        l_t: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None,
        coord_scale: float,
        lattice_scale: float,
        coord_grad_clip: float | None = None,
        lattice_grad_clip: float | None = None,
        coord_max_step: float | None = None,
        lattice_max_step: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coord_scale == 0.0 and lattice_scale == 0.0:
            return f_t, l_t

        with torch.enable_grad():
            f_var = f_t.detach().clone().requires_grad_(True)
            l_var = l_t.detach().clone().requires_grad_(True)
            coord_energy, lattice_energy = self._symmetry_guidance_energy(
                batch=batch,
                node_index=node_index,
                f_t=f_var,
                l_t=l_var,
                lattice_transform=lattice_transform,
            )
            total_energy = coord_energy + lattice_energy
            if not total_energy.requires_grad:
                return f_t, l_t

            grad_f, grad_l = torch.autograd.grad(
                outputs=total_energy,
                inputs=(f_var, l_var),
                allow_unused=True,
            )

        with torch.no_grad():
            f_next = f_var
            l_next = l_var
            if grad_f is not None and coord_scale != 0.0:
                grad_f = torch.nan_to_num(grad_f, nan=0.0, posinf=0.0, neginf=0.0)
                if coord_grad_clip is not None:
                    grad_f = _clip_rowwise_norm(grad_f, float(coord_grad_clip))
                coord_update = coord_scale * grad_f
                if coord_max_step is not None:
                    coord_update = _clip_rowwise_norm(coord_update, float(coord_max_step))
                f_next = self.tdm.wrap_displacements(f_var - coord_update)
            if grad_l is not None and lattice_scale != 0.0:
                grad_l = torch.nan_to_num(grad_l, nan=0.0, posinf=0.0, neginf=0.0)
                if lattice_grad_clip is not None:
                    grad_l = _clip_rowwise_norm(grad_l, float(lattice_grad_clip))
                lattice_update = lattice_scale * grad_l
                if lattice_max_step is not None:
                    lattice_update = _clip_rowwise_norm(lattice_update, float(lattice_max_step))
                l_candidate = l_var - lattice_update
                try:
                    num_atoms = torch.as_tensor(batch.num_atoms, device=l_candidate.device, dtype=torch.long).reshape(-1)
                    finite_mask_rows: list[torch.Tensor] = []
                    for graph_idx in range(l_candidate.shape[0]):
                        decoded = _decode_lattice_matrix(
                            l=l_candidate[graph_idx : graph_idx + 1],
                            num_atoms=int(num_atoms[graph_idx].item()),
                            lattice_transform=lattice_transform,
                        ).squeeze(0)
                        lengths = torch.linalg.norm(decoded, dim=-1)
                        graph_ok = (
                            torch.isfinite(decoded).all()
                            and torch.isfinite(lengths).all()
                            and bool((lengths > self.eps).all())
                            and bool((lengths < 100.0).all())
                        )
                        finite_mask_rows.append(torch.tensor(graph_ok, device=l_candidate.device, dtype=torch.bool))
                    finite_mask = torch.stack(finite_mask_rows, dim=0)
                    l_next = torch.where(finite_mask.unsqueeze(-1), l_candidate, l_var)
                except Exception:
                    l_next = l_var

        return f_next.detach(), l_next.detach()

    def sample_CSP_algorithm3(
        self,
        n_steps: int,
        batch: Batch | Data,
        t_start: float = 1.0,
        t_final: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 3 from Appendix H: EM sampling for the CSP model.

        At each time level:
            1. evaluate the network
            2. build the full velocity score
            3. do one exponential-Euler step for (f_t, v_t)
            4. do one reverse diffusion step for l_t
        """
        state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )
        state = self._run_csp_em_reverse_chain(state)

        if state["restore_training"]:
            state["score_network"].train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]

    def sample_CSP_algorithm5(
        self,
        n_steps: int,
        batch: Batch | Data,
        lattice_transform: ContinuousIntervalLattice | None = None,
        t_start: float = 1.0,
        t_final: float = 1e-6,
        coord_scale: float = 2e-3,
        lattice_scale: float = 1e-5,
        guidance_interval: int = 5,
        guidance_start_fraction: float = 0.5,
        coord_grad_clip: float | None = 5.0,
        lattice_grad_clip: float | None = 0.5,
        coord_max_step: float | None = 2e-2,
        lattice_max_step: float | None = 2e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 5: KLDM Algorithm 3 with DPS-style symmetry guidance.

        The score network remains the vanilla prior over (f, l | a). After each
        reverse step we compute a symmetry energy from batch.space_group and use
        autograd to nudge fractional coordinates and lattice features toward the
        requested space-group manifold. The coordinate term still uses
        species-aware torus matching under the space-group operations, while the
        lattice term follows the DiffCSP++ idea of constraining the invariant
        6D `k` coefficients of the lattice family implied by the space group.
        """
        if guidance_interval < 1:
            raise ValueError("guidance_interval must be >= 1.")
        if not 0.0 <= guidance_start_fraction <= 1.0:
            raise ValueError("guidance_start_fraction must be in [0, 1].")

        state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )

        for step_idx, times in enumerate(
            iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]),
            start=1,
        ):
            with torch.no_grad():
                preds_curr = state["score_network"](
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                score_v = state["sampling_tdm"].reconstruct_full_reverse_velocity_score(
                    t=times.now.nodes,
                    v_t=state["v_t"],
                    pred_v=preds_curr["v"],
                    index=state["node_index"],
                )

                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_exp_step(
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    score_v=score_v,
                    index=state["node_index"],
                    dt=times.dt,
                )

                state["l_t"] = self._reverse_lattice_sampling_step(
                    t=times.now.lattice,
                    x_t=state["l_t"],
                    pred=preds_curr["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

            if step_idx % guidance_interval != 0:
                continue
            if (step_idx / max(n_steps, 1)) < guidance_start_fraction:
                continue

            state["f_t"], state["l_t"] = self._apply_symmetry_guidance(
                batch=state["batch"],
                node_index=state["node_index"],
                f_t=state["f_t"],
                l_t=state["l_t"],
                lattice_transform=lattice_transform,
                coord_scale=float(coord_scale),
                lattice_scale=float(lattice_scale),
                coord_grad_clip=coord_grad_clip,
                lattice_grad_clip=lattice_grad_clip,
                coord_max_step=coord_max_step,
                lattice_max_step=lattice_max_step,
            )

        if state["restore_training"]:
            state["score_network"].train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]

    def sample_CSP_algorithm4(
        self,
        n_steps: int,
        batch: Batch | Data,
        t_start: float = 1.0,
        t_final: float = 1e-6,
        tau: float = 0.25,
        n_correction_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 4 from Appendix H, adapted to our internal scaled chart.

        Per step:
            1. evaluate the network at the current time t_n
            2. predictor from t_n to t_{n+1}
            3. evaluate the network again at t_{n+1} on the predicted state
            4. one corrector step at t_{n+1}
            5. one EM step for the lattice branch

        Important:
            - TDM internally uses velocity_scale = 1 / (2*pi)
            - therefore TDM predictor/corrector must use
              reconstruct_full_reverse_velocity_score(...)
            - corrector Langevin noise must use sample_velocity_noise(...)
            - our time grid uses dt = t_n - t_{n+1} > 0 while integrating
              backward, so the predictor position update needs the sign change
              documented in tdm.py to stay equivalent to the paper step
        """
        state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )

        with torch.no_grad():
            for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
                # One predictor-corrector transition in the decreasing grid:
                # times.now is the current/noisier time, and times.next is the
                # predicted/cleaner time used for the second network evaluation.
                # 1. Evaluate the network at the current time level t_n.
                preds_curr = state["score_network"](
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                # 2. Predictor from t_n to t_{n+1}. The TDM helper uses the
                # internally scaled full velocity score and the sign convention
                # corresponding to our positive backward-step dt.
                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_step_predictor(
                    t=times.now.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds_curr["v"],
                    index=state["node_index"],
                    dt=times.dt,
                )

                # Near t = 0 the reconstructed velocity score becomes very
                # stiff because the Gaussian variance term goes to zero.
                # Keep the predictor move, but skip the final corrector/lattice
                # update once the next time level is below 1e-3.
                if times.t_next_float < 1e-3:
                    continue

                preds_next = state["score_network"](
                    t=times.next.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                # 4. Single corrector step at t_{n+1}.
                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_step_corrector(
                    t=times.next.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds_next["v"],
                    dt=times.dt,
                    index=state["node_index"],
                    tau=tau,
                )

                # 5. Lattice branch: KLDM VP uses EM; MatterGen VP uses the
                # source-style ancestral lattice predictor.
                state["l_t"] = self._reverse_lattice_sampling_step(
                    t=times.next.lattice,
                    x_t=state["l_t"],
                    pred=preds_next["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

        if state["restore_training"]:
            state["score_network"].train()

        return state["f_t"], state["v_t"], state["l_t"], state["a_t"]

    def sample_CSP_algorithm6(
        self,
        n_steps: int,
        batch: Batch | Data,
        lattice_transform: ContinuousIntervalLattice | None = None,
        t_start: float = 1.0,
        t_final: float = 1e-6,
        pcs_standardization: str = "conventional",
        pcs_symprec: float = 1e-2,
        pcs_angle_tolerance: float = 5.0,
        pcs_max_templates: int = 256,
        pcs_template_eval_limit: int = 32,
        pcs_optimization_steps: int = 150,
        pcs_learning_rate: float = 5e-2,
        pcs_coord_weight: float = 1.0,
        pcs_lattice_weight: float = 0.25,
        pcs_pairdist_weight: float = 0.0,
        pcs_template_init_pairdist_weight: float | None = None,
        pcs_pairdist_bins: int = 32,
        pcs_pairdist_max_distance: float = 8.0,
        pcs_pairdist_bandwidth: float = 0.25,
        pcs_steric_weight: float = 0.0,
        pcs_steric_min_distance: float = 0.8,
        pcs_volume_weight: float = 0.0,
        pcs_volume_ratio_min: float = 0.0,
        pcs_volume_ratio_max: float = 0.0,
        pcs_k6_weight: float = 0.0,
        pcs_hard_min_distance: float = 0.0,
        pcs_hard_volume_ratio_min: float = 0.0,
        pcs_hard_volume_ratio_max: float = 0.0,
        pcs_freeze_lattice: bool = False,
        pcs_initialization: str = "repair",
        pcs_quick_templates: bool = False,
        pcs_top_k_templates: int = 1,
        pcs_template_prior: Any = None,
        pcs_template_prior_weight: float = 1.0,
        pcs_debug_template_candidates: bool = False,
        pcs_debug_high_prior_templates: bool = False,
        pcs_debug_high_prior_min_score: int = 1,
        pcs_allow_soft_physics_fallback: bool = True,
        pcs_branch_selection_temperature: float = 1.0,
        pcs_oracle_template_orbit_rerank: bool = False,
        pcs_oracle_template_fit_target: bool = False,
        pcs_mala_steps: int = 8,
        pcs_mala_step_size: float = 5e-2,
        pcs_dds_repair: bool = True,
        pcs_dds_n_steps: int = 60,
        pcs_dds_t_final: float = 1e-3,
        pcs_outer_steps: int = 1,
        pcs_outer_eta_start: float = 0.2,
        pcs_outer_eta_end: float = 0.2,
        pcs_outer_eta_k_start: int = 0,
        pcs_outer_eta_rho: float = 1.0,
        pcs_final_projection: bool = True,
        pcs_validate_requested_space_group: bool = True,
        pcs_return_last_pcs_on_validation_failure: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 6: KLDM-adapted DPnP with fixed-template PCS and DDS repair.

        Training stays unchanged on `(f, l, a)`. At test time we:

            1. draw a vanilla KLDM sample
            2. choose one requested-space-group Wyckoff template per graph
            3. run the outer DPnP loop `x_k -> PCS -> x_{k+1/2} -> DDS -> x_{k+1}`
            4. optionally finish with one last manifold materialization

        The discrete template choice is held fixed through the chain. PCS then
        runs a stochastic MALA kernel over the continuous manifold variables
        `(u, k)` defined by that template and the lattice `k`-basis.

        In DPnP form we use

            pi_k(u, k, W) propto exp(L_phys + L_template - E_prox / (2 eta_k^2)),

        where symmetry is enforced implicitly by the Wyckoff manifold map,
        `E_prox` keeps the branch close to the current KLDM sample, and the
        explicit likelihood terms `L_phys` penalize steric overlap and lattice
        scale collapse.
        """
        if pcs_outer_steps < 1:
            raise ValueError("pcs_outer_steps must be >= 1.")

        overall_started_at = time.perf_counter()
        pos_t, v_t, l_t, h_t = self.sample_CSP_algorithm3(
            n_steps=n_steps,
            batch=batch,
            t_start=t_start,
            t_final=t_final,
        )
        vanilla_elapsed_s = time.perf_counter() - overall_started_at
        print(
            f"algorithm6_progress phase=vanilla graphs={int(batch.num_graphs)} "
            f"elapsed_s={vanilla_elapsed_s:.1f}",
            flush=True,
        )
        print(
            f"algorithm6_config freeze_lattice={int(bool(pcs_freeze_lattice))} "
            f"initialization={str(pcs_initialization)} "
            f"final_projection={int(bool(pcs_final_projection))} "
            f"dds_repair={int(bool(pcs_dds_repair))} "
            f"allow_soft_physics_fallback={int(bool(pcs_allow_soft_physics_fallback))} "
            f"branch_selection_temperature={float(pcs_branch_selection_temperature):.6f}",
            flush=True,
        )
        print(
            f"algorithm6_likelihood_config volume_weight={float(pcs_volume_weight):.6f} "
            f"volume_ratio_min={float(pcs_volume_ratio_min):.6f} "
            f"volume_ratio_max={float(pcs_volume_ratio_max):.6f} "
            f"k6_weight={float(pcs_k6_weight):.6f} "
            f"hard_min_distance={float(pcs_hard_min_distance):.6f} "
            f"hard_volume_ratio_min={float(pcs_hard_volume_ratio_min):.6f} "
            f"hard_volume_ratio_max={float(pcs_hard_volume_ratio_max):.6f}",
            flush=True,
        )
        algorithm6_debug_records: dict[str, list[dict[str, Any]]] = {
            "template": [],
            "pcs": [],
            "materialize": [],
            "dds": [],
        }
        pos_vanilla = pos_t.clone()
        v_vanilla = v_t.clone()
        l_vanilla = l_t.clone()
        h_vanilla = h_t.clone()

        if not hasattr(batch, "space_group"):
            raise ValueError("sample_CSP_algorithm6 requires batch.space_group.")

        eta_schedule = self._algorithm6_outer_eta_schedule(
            outer_steps=int(pcs_outer_steps),
            eta_start=float(pcs_outer_eta_start),
            eta_end=float(pcs_outer_eta_end),
            eta_k_start=int(pcs_outer_eta_k_start),
            rho=float(pcs_outer_eta_rho),
            device=pos_t.device,
            dtype=pos_t.dtype,
        )
        print(
            "algorithm6_eta_schedule values="
            + str([round(float(v), 6) for v in eta_schedule.detach().cpu().tolist()]),
            flush=True,
        )

        init_started_at = time.perf_counter()
        chain_states = self._algorithm6_initialize_chain_states(
            batch=batch,
            pos_t=pos_t,
            l_t=l_t,
            h_t=h_t,
            lattice_transform=lattice_transform,
            pcs_standardization=pcs_standardization,
            pcs_symprec=pcs_symprec,
            pcs_angle_tolerance=pcs_angle_tolerance,
            pcs_max_templates=pcs_max_templates,
            pcs_template_eval_limit=pcs_template_eval_limit,
            pcs_optimization_steps=pcs_optimization_steps,
            pcs_learning_rate=pcs_learning_rate,
            pcs_coord_weight=pcs_coord_weight,
            pcs_lattice_weight=pcs_lattice_weight,
            pcs_pairdist_weight=pcs_pairdist_weight,
            pcs_template_init_pairdist_weight=pcs_template_init_pairdist_weight,
            pcs_pairdist_bins=pcs_pairdist_bins,
            pcs_pairdist_max_distance=pcs_pairdist_max_distance,
            pcs_pairdist_bandwidth=pcs_pairdist_bandwidth,
            pcs_steric_weight=pcs_steric_weight,
            pcs_steric_min_distance=pcs_steric_min_distance,
            pcs_volume_weight=pcs_volume_weight,
            pcs_volume_ratio_min=pcs_volume_ratio_min,
            pcs_volume_ratio_max=pcs_volume_ratio_max,
            pcs_k6_weight=pcs_k6_weight,
            pcs_freeze_lattice=pcs_freeze_lattice,
            pcs_initialization=str(pcs_initialization),
            pcs_quick_templates=pcs_quick_templates,
            pcs_top_k_templates=pcs_top_k_templates,
            pcs_template_prior=pcs_template_prior,
            pcs_template_prior_weight=pcs_template_prior_weight,
            pcs_debug_template_candidates=pcs_debug_template_candidates,
            pcs_oracle_template_orbit_rerank=pcs_oracle_template_orbit_rerank,
            pcs_oracle_template_fit_target=pcs_oracle_template_fit_target,
            template_debug_records=algorithm6_debug_records["template"],
        )
        init_elapsed_s = time.perf_counter() - init_started_at
        active_graphs = sum(states is not None for states in chain_states)
        active_branches = sum(len(states) for states in chain_states if states is not None)
        print(
            f"algorithm6_progress phase=template_init active_graphs={active_graphs}/{int(batch.num_graphs)} "
            f"active_branches={active_branches} elapsed_s={init_elapsed_s:.1f}",
            flush=True,
        )

        pos_current = pos_t
        l_current = l_t
        h_current = h_t
        v_current = v_t
        pos_last_pcs = pos_t.clone()
        l_last_pcs = l_t.clone()
        h_last_pcs = h_t.clone()
        v_last_pcs = v_t.clone()
        init_mode = str(pcs_initialization).strip().lower()
        if init_mode in {"constrained", "csp++", "csppp", "manifold"}:
            constrained_started_at = time.perf_counter()
            pos_current, l_current, h_current = self._algorithm6_materialize_chain_states(
                batch=batch,
                pos_t=pos_current,
                l_t=l_current,
                h_t=h_current,
                lattice_transform=lattice_transform,
                chain_states=chain_states,
                pcs_coord_weight=pcs_coord_weight,
                pcs_lattice_weight=pcs_lattice_weight,
                pcs_pairdist_weight=pcs_pairdist_weight,
                pcs_pairdist_bins=pcs_pairdist_bins,
                pcs_pairdist_max_distance=pcs_pairdist_max_distance,
                pcs_pairdist_bandwidth=pcs_pairdist_bandwidth,
                pcs_steric_weight=pcs_steric_weight,
                pcs_steric_min_distance=pcs_steric_min_distance,
                pcs_volume_weight=pcs_volume_weight,
                pcs_volume_ratio_min=pcs_volume_ratio_min,
                pcs_volume_ratio_max=pcs_volume_ratio_max,
                pcs_k6_weight=pcs_k6_weight,
                pcs_hard_min_distance=pcs_hard_min_distance,
                pcs_hard_volume_ratio_min=pcs_hard_volume_ratio_min,
                pcs_hard_volume_ratio_max=pcs_hard_volume_ratio_max,
                pcs_debug_template_candidates=pcs_debug_template_candidates,
                enforce_physical_guards=False,
                fallback_pos_t=pos_vanilla,
                fallback_l_t=l_vanilla,
                fallback_h_t=h_vanilla,
                materialize_debug_records=algorithm6_debug_records["materialize"],
                diagnostic_phase="constrained_init",
                allow_soft_physics_fallback=pcs_allow_soft_physics_fallback,
                branch_selection_temperature=pcs_branch_selection_temperature,
                debug_high_prior_templates=pcs_debug_high_prior_templates,
                debug_high_prior_min_score=pcs_debug_high_prior_min_score,
            )
            v_current = self.tdm.sample_velocity_noise(pos_current, index=batch.batch)
            pos_last_pcs = pos_current.clone()
            l_last_pcs = l_current.clone()
            h_last_pcs = h_current.clone()
            v_last_pcs = v_current.clone()
            print(
                "algorithm6_progress phase=constrained_init_materialize "
                f"elapsed_s={time.perf_counter() - constrained_started_at:.1f}",
                flush=True,
            )

        for outer_idx, eta in enumerate(eta_schedule, start=1):
            pcs_started_at = time.perf_counter()
            chain_states = self._algorithm6_pcs_step(
                batch=batch,
                pos_t=pos_current,
                l_t=l_current,
                h_t=h_current,
                lattice_transform=lattice_transform,
                chain_states=chain_states,
                eta=float(eta.item()),
                pcs_mala_steps=pcs_mala_steps,
                pcs_mala_step_size=pcs_mala_step_size,
                pcs_coord_weight=pcs_coord_weight,
                pcs_lattice_weight=pcs_lattice_weight,
                pcs_pairdist_weight=pcs_pairdist_weight,
                pcs_pairdist_bins=pcs_pairdist_bins,
                pcs_pairdist_max_distance=pcs_pairdist_max_distance,
                pcs_pairdist_bandwidth=pcs_pairdist_bandwidth,
                pcs_steric_weight=pcs_steric_weight,
                pcs_steric_min_distance=pcs_steric_min_distance,
                pcs_volume_weight=pcs_volume_weight,
                pcs_volume_ratio_min=pcs_volume_ratio_min,
                pcs_volume_ratio_max=pcs_volume_ratio_max,
                pcs_k6_weight=pcs_k6_weight,
                pcs_freeze_lattice=pcs_freeze_lattice,
                pcs_debug_records=algorithm6_debug_records["pcs"],
                diagnostic_phase=f"outer_{outer_idx}",
                verbose_pcs_mala=pcs_debug_template_candidates,
            )
            pcs_elapsed_s = time.perf_counter() - pcs_started_at

            materialize_started_at = time.perf_counter()
            pos_projected, l_projected, h_projected = self._algorithm6_materialize_chain_states(
                batch=batch,
                pos_t=pos_current,
                l_t=l_current,
                h_t=h_current,
                lattice_transform=lattice_transform,
                chain_states=chain_states,
                pcs_coord_weight=pcs_coord_weight,
                pcs_lattice_weight=pcs_lattice_weight,
                pcs_pairdist_weight=pcs_pairdist_weight,
                pcs_pairdist_bins=pcs_pairdist_bins,
                pcs_pairdist_max_distance=pcs_pairdist_max_distance,
                pcs_pairdist_bandwidth=pcs_pairdist_bandwidth,
                pcs_steric_weight=pcs_steric_weight,
                pcs_steric_min_distance=pcs_steric_min_distance,
                pcs_volume_weight=pcs_volume_weight,
                pcs_volume_ratio_min=pcs_volume_ratio_min,
                pcs_volume_ratio_max=pcs_volume_ratio_max,
                pcs_k6_weight=pcs_k6_weight,
                pcs_hard_min_distance=pcs_hard_min_distance,
                pcs_hard_volume_ratio_min=pcs_hard_volume_ratio_min,
                pcs_hard_volume_ratio_max=pcs_hard_volume_ratio_max,
                pcs_debug_template_candidates=pcs_debug_template_candidates,
                enforce_physical_guards=not bool(pcs_dds_repair),
                fallback_pos_t=pos_vanilla,
                fallback_l_t=l_vanilla,
                fallback_h_t=h_vanilla,
                materialize_debug_records=algorithm6_debug_records["materialize"],
                diagnostic_phase=f"outer_{outer_idx}",
                allow_soft_physics_fallback=pcs_allow_soft_physics_fallback,
                branch_selection_temperature=pcs_branch_selection_temperature,
                debug_high_prior_templates=pcs_debug_high_prior_templates,
                debug_high_prior_min_score=pcs_debug_high_prior_min_score,
            )
            materialize_elapsed_s = time.perf_counter() - materialize_started_at
            pos_last_pcs = pos_projected.clone()
            l_last_pcs = l_projected.clone()
            h_last_pcs = h_projected.clone()
            v_last_pcs = self.tdm.sample_velocity_noise(pos_last_pcs, index=batch.batch)
            if not pcs_dds_repair:
                algorithm6_debug_records["dds"].append(
                    {
                        "phase": f"outer_{outer_idx}",
                        "status": "disabled",
                        "eta": float(eta.item()),
                        "elapsed_s": 0.0,
                        "t_start": float("nan"),
                        "t_final": float("nan"),
                        "pos_rmse_from_pcs": 0.0,
                        "lattice_rmse_from_pcs": 0.0,
                    }
                )
                pos_current = pos_projected
                l_current = l_projected
                h_current = h_projected
                v_current = self.tdm.sample_velocity_noise(pos_current, index=batch.batch)
                active_graphs = sum(states is not None for states in chain_states)
                active_branches = sum(len(states) for states in chain_states if states is not None)
                print(
                    f"algorithm6_progress phase=outer_step step={outer_idx}/{len(eta_schedule)} "
                    f"eta={float(eta.item()):.5f} active_graphs={active_graphs}/{int(batch.num_graphs)} "
                    f"active_branches={active_branches} pcs_elapsed_s={pcs_elapsed_s:.1f} "
                    f"materialize_elapsed_s={materialize_elapsed_s:.1f} dds_elapsed_s=0.0",
                    flush=True,
                )
                continue

            pos_reference = pos_current
            v_reference = v_current
            l_reference = l_current
            h_reference = h_current
            dds_t_start = self._algorithm6_map_eta_to_kldm_time(
                float(eta.item()),
                num_atoms=batch.num_atoms,
                ref_l=l_projected,
            )
            dds_t_final = float(pcs_dds_t_final)
            if dds_t_start <= dds_t_final:
                algorithm6_debug_records["dds"].append(
                    {
                        "phase": f"outer_{outer_idx}",
                        "status": "skipped",
                        "eta": float(eta.item()),
                        "elapsed_s": 0.0,
                        "t_start": float(dds_t_start),
                        "t_final": float(dds_t_final),
                        "pos_rmse_from_pcs": 0.0,
                        "lattice_rmse_from_pcs": 0.0,
                    }
                )
                pos_current = pos_projected
                l_current = l_projected
                h_current = h_projected
                v_current = self.tdm.sample_velocity_noise(pos_current, index=batch.batch)
                active_graphs = sum(states is not None for states in chain_states)
                active_branches = sum(len(states) for states in chain_states if states is not None)
                print(
                    f"algorithm6_dds_skip step={outer_idx}/{len(eta_schedule)} "
                    f"reason=nonpositive_time_window t_start={dds_t_start:.6f} "
                    f"t_final={dds_t_final:.6f}",
                    flush=True,
                )
                print(
                    f"algorithm6_progress phase=outer_step step={outer_idx}/{len(eta_schedule)} "
                    f"eta={float(eta.item()):.5f} active_graphs={active_graphs}/{int(batch.num_graphs)} "
                    f"active_branches={active_branches} pcs_elapsed_s={pcs_elapsed_s:.1f} "
                    f"materialize_elapsed_s={materialize_elapsed_s:.1f} dds_elapsed_s=0.0",
                    flush=True,
                )
                continue

            dds_started_at = time.perf_counter()
            pos_current, v_current, l_current, h_current = self._algorithm6_dds_repair(
                batch=batch,
                pos_clean=pos_projected,
                l_clean=l_projected,
                h_clean=h_projected,
                n_steps=int(pcs_dds_n_steps),
                t_start=dds_t_start,
                t_final=dds_t_final,
            )
            pos_current, v_current, l_current, h_current = self._algorithm6_restore_inactive_graphs(
                batch=batch,
                chain_states=chain_states,
                pos_reference=pos_reference,
                v_reference=v_reference,
                l_reference=l_reference,
                h_reference=h_reference,
                pos_candidate=pos_current,
                v_candidate=v_current,
                l_candidate=l_current,
                h_candidate=h_current,
            )
            dds_elapsed_s = time.perf_counter() - dds_started_at
            algorithm6_debug_records["dds"].append(
                {
                    "phase": f"outer_{outer_idx}",
                    "status": "ran",
                    "eta": float(eta.item()),
                    "elapsed_s": float(dds_elapsed_s),
                    "t_start": float(dds_t_start),
                    "t_final": float(dds_t_final),
                    "pos_rmse_from_pcs": _direct_torus_rmse(pos_current, pos_projected),
                    "lattice_rmse_from_pcs": _direct_tensor_rmse(l_current, l_projected),
                }
            )
            active_graphs = sum(states is not None for states in chain_states)
            active_branches = sum(len(states) for states in chain_states if states is not None)
            print(
                f"algorithm6_progress phase=outer_step step={outer_idx}/{len(eta_schedule)} "
                f"eta={float(eta.item()):.5f} active_graphs={active_graphs}/{int(batch.num_graphs)} "
                f"active_branches={active_branches} pcs_elapsed_s={pcs_elapsed_s:.1f} "
                f"materialize_elapsed_s={materialize_elapsed_s:.1f} dds_elapsed_s={dds_elapsed_s:.1f}",
                flush=True,
            )

        if pcs_final_projection:
            final_pcs_started_at = time.perf_counter()
            chain_states = self._algorithm6_pcs_step(
                batch=batch,
                pos_t=pos_current,
                l_t=l_current,
                h_t=h_current,
                lattice_transform=lattice_transform,
                chain_states=chain_states,
                eta=float(eta_schedule[-1].item()),
                pcs_mala_steps=pcs_mala_steps,
                pcs_mala_step_size=pcs_mala_step_size,
                pcs_coord_weight=pcs_coord_weight,
                pcs_lattice_weight=pcs_lattice_weight,
                pcs_pairdist_weight=pcs_pairdist_weight,
                pcs_pairdist_bins=pcs_pairdist_bins,
                pcs_pairdist_max_distance=pcs_pairdist_max_distance,
                pcs_pairdist_bandwidth=pcs_pairdist_bandwidth,
                pcs_steric_weight=pcs_steric_weight,
                pcs_steric_min_distance=pcs_steric_min_distance,
                pcs_volume_weight=pcs_volume_weight,
                pcs_volume_ratio_min=pcs_volume_ratio_min,
                pcs_volume_ratio_max=pcs_volume_ratio_max,
                pcs_k6_weight=pcs_k6_weight,
                pcs_freeze_lattice=pcs_freeze_lattice,
                pcs_debug_records=algorithm6_debug_records["pcs"],
                diagnostic_phase="final_projection",
                verbose_pcs_mala=pcs_debug_template_candidates,
            )
            pos_current, l_current, h_current = self._algorithm6_materialize_chain_states(
                batch=batch,
                pos_t=pos_current,
                l_t=l_current,
                h_t=h_current,
                lattice_transform=lattice_transform,
                chain_states=chain_states,
                pcs_coord_weight=pcs_coord_weight,
                pcs_lattice_weight=pcs_lattice_weight,
                pcs_pairdist_weight=pcs_pairdist_weight,
                pcs_pairdist_bins=pcs_pairdist_bins,
                pcs_pairdist_max_distance=pcs_pairdist_max_distance,
                pcs_pairdist_bandwidth=pcs_pairdist_bandwidth,
                pcs_steric_weight=pcs_steric_weight,
                pcs_steric_min_distance=pcs_steric_min_distance,
                pcs_volume_weight=pcs_volume_weight,
                pcs_volume_ratio_min=pcs_volume_ratio_min,
                pcs_volume_ratio_max=pcs_volume_ratio_max,
                pcs_k6_weight=pcs_k6_weight,
                pcs_hard_min_distance=pcs_hard_min_distance,
                pcs_hard_volume_ratio_min=pcs_hard_volume_ratio_min,
                pcs_hard_volume_ratio_max=pcs_hard_volume_ratio_max,
                fallback_pos_t=pos_vanilla,
                fallback_l_t=l_vanilla,
                fallback_h_t=h_vanilla,
                materialize_debug_records=algorithm6_debug_records["materialize"],
                diagnostic_phase="final_projection",
                allow_soft_physics_fallback=pcs_allow_soft_physics_fallback,
                branch_selection_temperature=pcs_branch_selection_temperature,
                debug_high_prior_templates=pcs_debug_high_prior_templates,
                debug_high_prior_min_score=pcs_debug_high_prior_min_score,
            )
            final_pcs_elapsed_s = time.perf_counter() - final_pcs_started_at
            v_current = self.tdm.sample_velocity_noise(pos_current, index=batch.batch)
            pos_last_pcs = pos_current.clone()
            l_last_pcs = l_current.clone()
            h_last_pcs = h_current.clone()
            v_last_pcs = v_current.clone()
            print(
                f"algorithm6_progress phase=final_projection elapsed_s={final_pcs_elapsed_s:.1f}",
                flush=True,
            )

        if pcs_validate_requested_space_group:
            validate_started_at = time.perf_counter()
            pos_current, v_current, l_current, h_current = self._algorithm6_validate_batch_constraints(
                batch=batch,
                pos_t=pos_current,
                v_t=v_current,
                l_t=l_current,
                h_t=h_current,
                pos_reference=(
                    pos_last_pcs if pcs_return_last_pcs_on_validation_failure else pos_vanilla
                ),
                v_reference=(
                    v_last_pcs if pcs_return_last_pcs_on_validation_failure else v_vanilla
                ),
                l_reference=(
                    l_last_pcs if pcs_return_last_pcs_on_validation_failure else l_vanilla
                ),
                h_reference=(
                    h_last_pcs if pcs_return_last_pcs_on_validation_failure else h_vanilla
                ),
                lattice_transform=lattice_transform,
                pcs_symprec=pcs_symprec,
                pcs_angle_tolerance=pcs_angle_tolerance,
                chain_states=chain_states,
                revert_on_failure=pcs_return_last_pcs_on_validation_failure,
            )
            validate_elapsed_s = time.perf_counter() - validate_started_at
            print(
                f"algorithm6_progress phase=validate elapsed_s={validate_elapsed_s:.1f}",
                flush=True,
            )

        total_elapsed_s = time.perf_counter() - overall_started_at
        self._algorithm6_print_debug_summary(
            debug_records=algorithm6_debug_records,
            chain_states=chain_states,
            num_graphs=int(batch.num_graphs),
            oracle_template_orbit_rerank=pcs_oracle_template_orbit_rerank,
            oracle_template_fit_target=pcs_oracle_template_fit_target,
            dds_repair=pcs_dds_repair,
            final_projection=pcs_final_projection,
        )
        print(
            f"algorithm6_progress phase=done total_elapsed_s={total_elapsed_s:.1f}",
            flush=True,
        )

        return pos_current, v_current, l_current, h_current

    def sample_CSP_kldm_dpnp_sg(
        self,
        n_steps: int,
        batch: Batch | Data,
        lattice_transform: ContinuousIntervalLattice | None = None,
        t_start: float = 1.0,
        t_final: float = 1e-6,
        sgdpnp_config: dict[str, Any] | SGDPnPConfig | None = None,
        template_prior: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clean KLDM-DPnP-SG sampler.

        This is the new small architecture:

            KLDM prior sample -> Wyckoff PCS sample -> KLDM DDS -> final Wyckoff PCS.

        The old Algorithm 6 implementation is intentionally left isolated below
        this entry point; new experiments should call this method instead.
        """
        config = (
            sgdpnp_config
            if isinstance(sgdpnp_config, SGDPnPConfig)
            else SGDPnPConfig.from_mapping(sgdpnp_config)
        )
        return sample_kldm_dpnp_sg(
            model=self,
            n_steps=n_steps,
            batch=batch,
            lattice_transform=lattice_transform,
            t_start=t_start,
            t_final=t_final,
            config=config,
            template_prior=template_prior,
        )

    def sample_CSP_algorithm7(
        self,
        n_steps: int,
        batch: Batch | Data,
        lattice_transform: ContinuousIntervalLattice | None = None,
        t_start: float = 1.0,
        t_final: float = 1e-6,
        sgdpnp_config: dict[str, Any] | SGDPnPConfig | None = None,
        template_prior: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample_CSP_kldm_dpnp_sg(
            n_steps=n_steps,
            batch=batch,
            lattice_transform=lattice_transform,
            t_start=t_start,
            t_final=t_final,
            sgdpnp_config=sgdpnp_config,
            template_prior=template_prior,
        )

    def _prepare_csp_sampling(
        self,
        batch: Batch | Data,
        n_steps: int,
        t_start: float,
        t_final: float,
        initial_f: torch.Tensor | None = None,
        initial_v: torch.Tensor | None = None,
        initial_l: torch.Tensor | None = None,
        initial_a: torch.Tensor | None = None,
        initialize_from_clean_state: bool = False,
        initialize_from_dds_anchor: bool = False,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        batch = batch.to(device)

        node_index = batch.batch
        edge_node_index = batch.edge_node_index
        num_graphs = batch.num_graphs

        if initialize_from_clean_state and initialize_from_dds_anchor:
            raise ValueError("Choose at most one of initialize_from_clean_state and initialize_from_dds_anchor.")

        if initialize_from_clean_state or initialize_from_dds_anchor:
            if initial_f is None or initial_l is None:
                raise ValueError("DDS/clean initialization requires initial_f and initial_l.")
            initial_f = initial_f.to(device=device, dtype=batch.pos.dtype)
            initial_l = initial_l.to(device=device, dtype=batch.l.dtype)
            a_t = (
                batch.atomic_numbers
                if initial_a is None
                else initial_a.to(device=device, dtype=batch.atomic_numbers.dtype)
            )
            if initial_f.shape != batch.pos.shape:
                raise ValueError(
                    "initialize_from_clean_state received fractional coordinates with shape "
                    f"{tuple(initial_f.shape)}, expected {tuple(batch.pos.shape)}."
                )
            if initial_l.shape != batch.l.shape:
                raise ValueError(
                    "initialize_from_clean_state received lattice features with shape "
                    f"{tuple(initial_l.shape)}, expected {tuple(batch.l.shape)}."
                )
            if a_t.shape != batch.atomic_numbers.shape:
                raise ValueError(
                    "initialize_from_clean_state received atomic numbers with shape "
                    f"{tuple(a_t.shape)}, expected {tuple(batch.atomic_numbers.shape)}."
                )
            start_times = make_times(batch, float(t_start))
            if initialize_from_dds_anchor:
                t_nodes_internal = self.tdm.T * start_times.nodes
                velocity_noise = self.tdm.sample_velocity_noise(initial_f, index=node_index)
                gaussian_sigma_t = self.tdm.match_dims(self.tdm.gaussian_velocity_sigma(t_nodes_internal), initial_f)
                v_t = gaussian_sigma_t * velocity_noise
                mu_r_t = self.tdm.wrapped_gaussian_mu_r_t(t_nodes_internal, v_t)
                f_t = self.tdm.wrap_displacements(initial_f + mu_r_t)

                if isinstance(self.diffusion_l, ContinuousMattergenVPDiffusion):
                    alpha_t = self.diffusion_l._match_dims(self.diffusion_l.alpha(start_times.lattice), initial_l)
                    mu_vec = self.diffusion_l.prior_mean(num_atoms=batch.num_atoms, ref=initial_l)
                    l_t = alpha_t * initial_l + (1.0 - alpha_t) * mu_vec
                else:
                    alpha_t = self.diffusion_l._match_dims(self.diffusion_l.alpha(start_times.lattice), initial_l)
                    l_t = alpha_t * initial_l
            else:
                f_t, v_t, _epsilon_v, _epsilon_r, _r_t = self.tdm.sample_noisy_state(
                    t=start_times.nodes,
                    f0=initial_f,
                    index=node_index,
                )
                l_t, _eps_l = self.diffusion_l.forward_sample(
                    t=start_times.lattice,
                    x0=initial_l,
                    num_atoms=batch.num_atoms,
                )
        else:
            # Appendix H-style priors, kept in one place so the sampler owns its
            # initial state: f_T ~ U(0, 1) represented in TDM's signed chart,
            # v_T ~ centered N_v(0, I), and l_T follows either the KLDM standard
            # normal prior or the MatterGen atom-count-aware prior.
            f_t = self.tdm.wrap_displacements(torch.rand_like(batch.pos))
            v_t = self.tdm.sample_velocity_noise(f_t, index=node_index)
            l_t = self.diffusion_l.sample_prior(
                x_like=batch.l,
                num_atoms=batch.num_atoms,
            )
            a_t = batch.atomic_numbers

        if initial_v is not None and not initialize_from_clean_state:
            v_t = initial_v.to(device=device, dtype=batch.pos.dtype)

        score_network = self.score_network
        restore_training = score_network.training
        score_network.eval()

        sampling_time_grid = sampling_grid(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
        )

        return {
            "batch": batch,
            "device": device,
            "dtype": batch.pos.dtype,
            "n_steps": n_steps,
            "num_graphs": num_graphs,
            "node_index": node_index,
            "edge_node_index": edge_node_index,
            "sampling_tdm": self.tdm,
            "sampling_diffusion_l": self.diffusion_l,
            "score_network": score_network,
            "restore_training": restore_training,
            "f_t": f_t,
            "v_t": v_t,
            "l_t": l_t,
            "a_t": a_t,
            "sampling_time_grid": sampling_time_grid,
        }

    def _run_csp_em_reverse_chain(self, state: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad():
            """
            Dt is a positive “backward step size”. hence why the sampler uses
            different sign than the appendix algorithms.
            """
            for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
                preds_curr = state["score_network"](
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )

                score_v = state["sampling_tdm"].reconstruct_full_reverse_velocity_score(
                    t=times.now.nodes,
                    v_t=state["v_t"],
                    pred_v=preds_curr["v"],
                    index=state["node_index"],
                )

                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_exp_step(
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    score_v=score_v,
                    index=state["node_index"],
                    dt=times.dt,
                )

                state["l_t"] = self._reverse_lattice_sampling_step(
                    t=times.now.lattice,
                    x_t=state["l_t"],
                    pred=preds_curr["l"],
                    dt=times.dt,
                    num_atoms=state["batch"].num_atoms,
                )

        return state

    def _algorithm6_outer_eta_schedule(
        self,
        *,
        outer_steps: int,
        eta_start: float,
        eta_end: float,
        eta_k_start: int,
        rho: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if outer_steps < 1:
            raise ValueError("outer_steps must be >= 1.")
        if not (0.0 < eta_end <= eta_start <= 1.0):
            raise ValueError("Expected 0 < pcs_outer_eta_end <= pcs_outer_eta_start <= 1.")
        if eta_k_start < 0:
            raise ValueError("pcs_outer_eta_k_start must be >= 0.")
        if rho <= 0.0:
            raise ValueError("pcs_outer_eta_rho must be positive.")

        if outer_steps == 1:
            return torch.tensor([eta_start], device=device, dtype=dtype)

        if eta_k_start > 0:
            warm_count = min(int(eta_k_start), int(outer_steps))
            tail_count = int(outer_steps) - warm_count
            warm = torch.full((warm_count,), float(eta_start), device=device, dtype=dtype)
            if tail_count <= 0:
                return warm
            if tail_count == 1:
                tail = torch.tensor([float(eta_end)], device=device, dtype=dtype)
                return torch.cat([warm, tail], dim=0)
            tail = torch.exp(
                torch.linspace(
                    float(torch.log(torch.tensor(eta_start, dtype=dtype)).item()),
                    float(torch.log(torch.tensor(eta_end, dtype=dtype)).item()),
                    tail_count,
                    device=device,
                    dtype=dtype,
                )
            )
            return torch.cat([warm, tail], dim=0)

        u = torch.linspace(1.0, 0.0, outer_steps, device=device, dtype=dtype)
        return eta_end + (eta_start - eta_end) * u.pow(rho)

    def _algorithm6_map_eta_to_kldm_time(
        self,
        eta: float,
        *,
        num_atoms: torch.Tensor | None = None,
        ref_l: torch.Tensor | None = None,
    ) -> float:
        eta_clamped = float(max(1.0e-5, min(eta, 0.999)))
        device = next(self.parameters()).device
        dtype = torch.get_default_dtype()
        t_internal_grid = torch.linspace(0.0, float(self.tdm.T), 1024, device=device, dtype=dtype)
        sigma_r_grid = self.tdm.wrapped_gaussian_sigma_r_t(t_internal_grid)
        target = torch.tensor(eta_clamped, device=device, dtype=dtype)

        t_unit_grid = t_internal_grid / float(self.tdm.T)
        alpha_grid = self.diffusion_l.alpha(t_unit_grid)
        sigma_grid = self.diffusion_l.sigma(t_unit_grid)
        lattice_width_grid = sigma_grid / alpha_grid.clamp_min(1.0e-6)
        if isinstance(self.diffusion_l, ContinuousMattergenVPDiffusion) and num_atoms is not None:
            ref = ref_l if ref_l is not None else torch.zeros((int(num_atoms.shape[0]), 6), device=device, dtype=dtype)
            _, sigma_n = self.diffusion_l.mu_sigma_n(num_atoms=num_atoms.to(device=device), ref=ref.to(device=device, dtype=dtype))
            lattice_scale = torch.median(sigma_n.to(device=device, dtype=dtype)).clamp_min(1.0e-6)
            lattice_width_grid = lattice_width_grid * lattice_scale

        combined_error = (sigma_r_grid - target).square() + (lattice_width_grid - target).square()
        idx = int(torch.argmin(combined_error).item())
        t_internal = float(t_internal_grid[idx].item())
        return max(1.0e-3, min(t_internal / float(self.tdm.T), 1.0))

    def _algorithm6_initialize_chain_states(
        self,
        *,
        batch: Batch | Data,
        pos_t: torch.Tensor,
        l_t: torch.Tensor,
        h_t: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None,
        pcs_standardization: str,
        pcs_symprec: float,
        pcs_angle_tolerance: float,
        pcs_max_templates: int,
        pcs_template_eval_limit: int,
        pcs_optimization_steps: int,
        pcs_learning_rate: float,
        pcs_coord_weight: float,
        pcs_lattice_weight: float,
        pcs_pairdist_weight: float,
        pcs_template_init_pairdist_weight: float | None,
        pcs_pairdist_bins: int,
        pcs_pairdist_max_distance: float,
        pcs_pairdist_bandwidth: float,
        pcs_steric_weight: float,
        pcs_steric_min_distance: float,
        pcs_volume_weight: float,
        pcs_volume_ratio_min: float,
        pcs_volume_ratio_max: float,
        pcs_k6_weight: float,
        pcs_freeze_lattice: bool,
        pcs_initialization: str,
        pcs_quick_templates: bool,
        pcs_top_k_templates: int,
        pcs_template_prior: Any,
        pcs_template_prior_weight: float,
        pcs_debug_template_candidates: bool,
        pcs_oracle_template_orbit_rerank: bool,
        pcs_oracle_template_fit_target: bool,
        template_debug_records: list[dict[str, Any]] | None = None,
    ) -> list[list[Any] | None]:
        chain_states: list[list[Any] | None] = []
        ptr = batch.ptr.tolist()
        space_groups = torch.as_tensor(batch.space_group, device=pos_t.device, dtype=torch.long).reshape(-1)
        init_mode = str(pcs_initialization).strip().lower()

        for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
            graph_pos = pos_t[start_idx:end_idx]
            graph_h = h_t[start_idx:end_idx]
            graph_l = l_t[graph_idx]
            num_atoms = int(graph_pos.shape[0])
            cell_matrix = _decode_lattice_matrix(
                l=graph_l.view(1, -1),
                num_atoms=num_atoms,
                lattice_transform=lattice_transform,
            ).squeeze(0)
            try:
                oracle_reference_structure = None
                oracle_fit_structure = None
                if pcs_oracle_template_orbit_rerank or pcs_oracle_template_fit_target:
                    target_pos = batch.pos[start_idx:end_idx]
                    target_h = batch.atomic_numbers[start_idx:end_idx]
                    target_l = batch.l[graph_idx]
                    target_cell_matrix = _decode_lattice_matrix(
                        l=target_l.view(1, -1),
                        num_atoms=num_atoms,
                        lattice_transform=lattice_transform,
                    ).squeeze(0)
                    oracle_target_structure = self._algorithm6_graph_structure(
                        frac_coords=target_pos,
                        atomic_numbers=target_h,
                        cell_matrix=target_cell_matrix,
                    )
                    if pcs_oracle_template_orbit_rerank:
                        oracle_reference_structure = oracle_target_structure
                    if pcs_oracle_template_fit_target:
                        oracle_fit_structure = oracle_target_structure
                if init_mode in {"constrained", "csp++", "csppp", "manifold"}:
                    initial_states = initialize_constrained_template_states(
                        reference_frac_coords=graph_pos,
                        atomic_numbers=graph_h,
                        cell_matrix=cell_matrix,
                        space_group_number=int(space_groups[graph_idx].item()),
                        standardization=pcs_standardization,
                        symprec=pcs_symprec,
                        angle_tolerance=pcs_angle_tolerance,
                        max_templates=pcs_max_templates,
                        template_eval_limit=pcs_template_eval_limit,
                        quick_templates=pcs_quick_templates,
                        top_k=max(1, int(pcs_top_k_templates)),
                        template_prior=pcs_template_prior,
                        template_prior_weight=float(pcs_template_prior_weight),
                        debug_template_candidates=pcs_debug_template_candidates,
                        debug_label=f"graph={graph_idx + 1}",
                        freeze_lattice_free_vars=pcs_freeze_lattice,
                    )
                else:
                    initial_states = select_requested_template_states(
                        frac_coords=graph_pos,
                        atomic_numbers=graph_h,
                        cell_matrix=cell_matrix,
                        space_group_number=int(space_groups[graph_idx].item()),
                        standardization=pcs_standardization,
                        symprec=pcs_symprec,
                        angle_tolerance=pcs_angle_tolerance,
                        max_templates=pcs_max_templates,
                        template_eval_limit=pcs_template_eval_limit,
                        optimization_steps=pcs_optimization_steps,
                        learning_rate=pcs_learning_rate,
                        coord_weight=pcs_coord_weight,
                        lattice_weight=pcs_lattice_weight,
                        pairdist_weight=(
                            pcs_pairdist_weight
                            if pcs_template_init_pairdist_weight is None
                            else float(pcs_template_init_pairdist_weight)
                        ),
                        pairdist_bins=pcs_pairdist_bins,
                        pairdist_max_distance=pcs_pairdist_max_distance,
                        pairdist_bandwidth=pcs_pairdist_bandwidth,
                        steric_weight=pcs_steric_weight,
                        steric_min_distance=pcs_steric_min_distance,
                        volume_weight=pcs_volume_weight,
                        volume_ratio_min=pcs_volume_ratio_min,
                        volume_ratio_max=pcs_volume_ratio_max,
                        k6_weight=pcs_k6_weight,
                        freeze_lattice_free_vars=pcs_freeze_lattice,
                        quick_templates=pcs_quick_templates,
                        top_k=max(1, int(pcs_top_k_templates)),
                        template_prior=pcs_template_prior,
                        template_prior_weight=float(pcs_template_prior_weight),
                        debug_template_candidates=pcs_debug_template_candidates,
                        debug_label=f"graph={graph_idx + 1}",
                        oracle_reference_structure=oracle_reference_structure,
                        oracle_fit_structure=oracle_fit_structure,
                    )
                if template_debug_records is not None:
                    for pool_idx, state in enumerate(initial_states, start=1):
                        template_debug_records.append(
                            {
                                "graph": int(graph_idx + 1),
                                "pool_idx": int(pool_idx),
                                "requested_sg": int(space_groups[graph_idx].item()),
                                "template_rank": int(state.template_rank),
                                "prior_score": int(state.prior_score),
                                "prior_bonus": float(state.prior_bonus),
                                "ranking_objective": float(state.ranking_objective),
                                "objective": float(state.objective),
                                "species_orbit_mismatch": int(state.species_orbit_mismatch),
                                "target_repr": (
                                    state.anchor_representation_name
                                    or state.target_representation_name
                                    or "na"
                                ),
                                "signature": _pcs_state_signature_labels(state),
                            }
                        )
                chain_states.append(
                    [
                        replace(
                            state,
                            branch_frac_coords=graph_pos.detach().clone(),
                            branch_atomic_numbers=graph_h.detach().clone(),
                            branch_lattice_features=graph_l.detach().clone(),
                        )
                        for state in initial_states
                    ]
                )
            except Exception as exc:
                warnings.warn(
                    "Algorithm 6 could not initialize a requested-space-group PCS template for one "
                    "graph; keeping vanilla KLDM sampling for that graph. "
                    f"Reason: {type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                chain_states.append(None)

        return chain_states

    def _algorithm6_pcs_step(
        self,
        *,
        batch: Batch | Data,
        pos_t: torch.Tensor,
        l_t: torch.Tensor,
        h_t: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None,
        chain_states: list[Any],
        eta: float,
        pcs_mala_steps: int,
        pcs_mala_step_size: float,
        pcs_coord_weight: float,
        pcs_lattice_weight: float,
        pcs_pairdist_weight: float,
        pcs_pairdist_bins: int,
        pcs_pairdist_max_distance: float,
        pcs_pairdist_bandwidth: float,
        pcs_steric_weight: float,
        pcs_steric_min_distance: float,
        pcs_volume_weight: float,
        pcs_volume_ratio_min: float,
        pcs_volume_ratio_max: float,
        pcs_k6_weight: float,
        pcs_freeze_lattice: bool,
        pcs_debug_records: list[dict[str, Any]] | None = None,
        diagnostic_phase: str = "unknown",
        verbose_pcs_mala: bool = False,
    ) -> list[list[Any] | None]:
        next_states: list[list[Any] | None] = []
        ptr = batch.ptr.tolist()
        for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
            states = chain_states[graph_idx]
            if states is None:
                next_states.append(None)
                continue
            graph_pos = pos_t[start_idx:end_idx]
            graph_h = h_t[start_idx:end_idx]
            graph_l = l_t[graph_idx]
            cell_matrix = _decode_lattice_matrix(
                l=graph_l.view(1, -1),
                num_atoms=int(graph_pos.shape[0]),
                lattice_transform=lattice_transform,
            ).squeeze(0)
            branch_states: list[Any] = []
            for branch_idx, state in enumerate(states, start=1):
                try:
                    previous_branch_pos = (
                        state.branch_frac_coords.to(device=graph_pos.device, dtype=graph_pos.dtype)
                        if state.branch_frac_coords is not None
                        else None
                    )
                    previous_branch_l = (
                        state.branch_lattice_features.to(device=graph_l.device, dtype=graph_l.dtype)
                        if state.branch_lattice_features is not None
                        else None
                    )
                    anchor_coord_rmse = (
                        _direct_torus_rmse(graph_pos, previous_branch_pos)
                        if previous_branch_pos is not None
                        else float("nan")
                    )
                    anchor_lattice_rmse = (
                        _direct_tensor_rmse(graph_l.reshape(-1), previous_branch_l.reshape(-1))
                        if previous_branch_l is not None
                        else float("nan")
                    )
                    # Refresh the proximal anchor from the current full-space iterate
                    # so DDS/KLDM proposals actually influence the next PCS branch step.
                    updated_state = sample_pcs_step_mala(
                        state=state,
                        frac_coords=graph_pos,
                        atomic_numbers=graph_h,
                        cell_matrix=cell_matrix,
                        eta=eta,
                        mala_steps=pcs_mala_steps,
                        mala_step_size=pcs_mala_step_size,
                        coord_weight=pcs_coord_weight,
                        lattice_weight=pcs_lattice_weight,
                        pairdist_weight=pcs_pairdist_weight,
                        pairdist_bins=pcs_pairdist_bins,
                        pairdist_max_distance=pcs_pairdist_max_distance,
                        pairdist_bandwidth=pcs_pairdist_bandwidth,
                        steric_weight=pcs_steric_weight,
                        steric_min_distance=pcs_steric_min_distance,
                        volume_weight=pcs_volume_weight,
                        volume_ratio_min=pcs_volume_ratio_min,
                        volume_ratio_max=pcs_volume_ratio_max,
                        k6_weight=pcs_k6_weight,
                        freeze_lattice_free_vars=pcs_freeze_lattice,
                    )
                    branch_states.append(updated_state)
                    if pcs_debug_records is not None:
                        pcs_debug_records.append(
                            {
                                "phase": str(diagnostic_phase),
                                "graph": int(graph_idx + 1),
                                "branch": int(branch_idx),
                                "target_repr": (
                                    updated_state.anchor_representation_name
                                    or updated_state.target_representation_name
                                    or "na"
                                ),
                                "template_rank": int(updated_state.template_rank),
                                "prior_score": int(updated_state.prior_score),
                                "prior_bonus": float(updated_state.prior_bonus),
                                "signature": _pcs_state_signature_labels(updated_state),
                                "accept_rate": float(updated_state.mala_acceptance_rate),
                                "accept_count": int(updated_state.mala_accept_count),
                                "attempted_steps": int(updated_state.mala_attempted_steps),
                                "energy": float(updated_state.mala_total_energy),
                                "coord_loss": float(updated_state.mala_coord_loss),
                                "lattice_loss": float(updated_state.mala_lattice_loss),
                                "pairdist_loss": float(updated_state.mala_pairdist_loss),
                                "steric_loss": float(updated_state.mala_steric_loss),
                                "volume_loss": float(updated_state.mala_volume_loss),
                                "k6_loss": float(updated_state.mala_k6_loss),
                                "prox_energy": float(updated_state.mala_prox_energy),
                                "likelihood_energy": float(updated_state.mala_likelihood_energy),
                                "anchor_coord_rmse": float(anchor_coord_rmse),
                                "anchor_lattice_rmse": float(anchor_lattice_rmse),
                            }
                        )
                    if verbose_pcs_mala:
                        print(
                            f"algorithm6_pcs_mala graph={graph_idx + 1} branch={branch_idx} "
                            f"phase={diagnostic_phase} "
                            f"anchor_coord_rmse={float(anchor_coord_rmse):.6f} "
                            f"anchor_lattice_rmse={float(anchor_lattice_rmse):.6f} "
                            f"target_repr={updated_state.anchor_representation_name or updated_state.target_representation_name} "
                            f"accept_rate={updated_state.mala_acceptance_rate:.3f} "
                            f"accept={updated_state.mala_accept_count}/{updated_state.mala_attempted_steps} "
                            f"energy={updated_state.mala_total_energy:.6f} "
                            f"coord_loss={updated_state.mala_coord_loss:.6f} "
                            f"lattice_loss={updated_state.mala_lattice_loss:.6f} "
                            f"pairdist_loss={updated_state.mala_pairdist_loss:.6f} "
                            f"steric_loss={updated_state.mala_steric_loss:.6f} "
                            f"volume_loss={updated_state.mala_volume_loss:.6f} "
                            f"k6_loss={updated_state.mala_k6_loss:.6f} "
                            f"prox_energy={updated_state.mala_prox_energy:.6f} "
                            f"likelihood_energy={updated_state.mala_likelihood_energy:.6f}",
                            flush=True,
                        )
                except Exception as exc:
                    warnings.warn(
                        "Algorithm 6 PCS sampling failed for one branch; dropping that branch. "
                        f"Reason: {type(exc).__name__}: {exc}",
                        stacklevel=2,
                    )
            if branch_states:
                branch_states.sort(key=_pcs_state_rank_key)
                next_states.append(branch_states)
            else:
                warnings.warn(
                    "Algorithm 6 PCS sampling failed for all branches of one graph; keeping vanilla KLDM "
                    "sampling for that graph from this point on.",
                    stacklevel=2,
                )
                next_states.append(None)
        return next_states

    def _algorithm6_materialize_chain_states(
        self,
        *,
        batch: Batch | Data,
        pos_t: torch.Tensor,
        l_t: torch.Tensor,
        h_t: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None,
        chain_states: list[list[Any] | None],
        pcs_coord_weight: float,
        pcs_lattice_weight: float,
        pcs_pairdist_weight: float,
        pcs_pairdist_bins: int,
        pcs_pairdist_max_distance: float,
        pcs_pairdist_bandwidth: float,
        pcs_steric_weight: float,
        pcs_steric_min_distance: float,
        pcs_volume_weight: float,
        pcs_volume_ratio_min: float,
        pcs_volume_ratio_max: float,
        pcs_k6_weight: float,
        pcs_hard_min_distance: float,
        pcs_hard_volume_ratio_min: float,
        pcs_hard_volume_ratio_max: float,
        pcs_debug_template_candidates: bool = False,
        enforce_physical_guards: bool = True,
        fallback_pos_t: torch.Tensor | None = None,
        fallback_l_t: torch.Tensor | None = None,
        fallback_h_t: torch.Tensor | None = None,
        materialize_debug_records: list[dict[str, Any]] | None = None,
        diagnostic_phase: str = "unknown",
        allow_soft_physics_fallback: bool = True,
        branch_selection_temperature: float = 1.0,
        debug_high_prior_templates: bool = False,
        debug_high_prior_min_score: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_pos_blocks: list[torch.Tensor] = []
        projected_l_blocks: list[torch.Tensor] = []
        projected_h_blocks: list[torch.Tensor] = []
        ptr = batch.ptr.tolist()

        def _branch_energy(state: Any) -> float:
            energy = float(getattr(state, "ranking_objective", float("inf")))
            if math.isfinite(energy):
                return energy
            return float("inf")

        def _sample_branch_index(entries: list[tuple[Any, ...]], state_pos: int) -> tuple[int, float, float, str]:
            if not entries:
                raise ValueError("Cannot sample from an empty branch list.")
            energies = [_branch_energy(entry[state_pos]) for entry in entries]
            finite_energies = [energy for energy in energies if math.isfinite(energy)]
            if len(entries) == 1:
                return 0, 1.0, energies[0], "single"
            if float(branch_selection_temperature) <= 0.0 or not finite_energies:
                best_idx = min(range(len(entries)), key=lambda idx: energies[idx])
                return best_idx, 1.0, energies[best_idx], "map"
            max_finite = max(finite_energies)
            sanitized = [
                energy if math.isfinite(energy) else max_finite + 1.0e6
                for energy in energies
            ]
            energy_tensor = torch.tensor(
                sanitized,
                device=pos_t.device,
                dtype=torch.float64,
            )
            centered = energy_tensor - torch.min(energy_tensor)
            logits = -centered / max(float(branch_selection_temperature), 1.0e-8)
            probs = torch.softmax(logits, dim=0)
            if not bool(torch.isfinite(probs).all().item()) or float(probs.sum().item()) <= 0.0:
                probs = torch.full_like(probs, 1.0 / float(len(entries)))
            sampled_idx = int(torch.multinomial(probs, 1).item())
            return sampled_idx, float(probs[sampled_idx].item()), energies[sampled_idx], "sample"

        for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
            graph_pos = pos_t[start_idx:end_idx]
            graph_h = h_t[start_idx:end_idx]
            graph_l = l_t[graph_idx]
            fallback_pos = (
                fallback_pos_t[start_idx:end_idx]
                if fallback_pos_t is not None
                else graph_pos
            )
            fallback_h = (
                fallback_h_t[start_idx:end_idx]
                if fallback_h_t is not None
                else graph_h
            )
            fallback_l = (
                fallback_l_t[graph_idx]
                if fallback_l_t is not None
                else graph_l
            )
            cell_matrix = _decode_lattice_matrix(
                l=graph_l.view(1, -1),
                num_atoms=int(graph_pos.shape[0]),
                lattice_transform=lattice_transform,
            ).squeeze(0)
            current_structure = self._algorithm6_graph_structure(
                frac_coords=graph_pos,
                atomic_numbers=graph_h,
                cell_matrix=cell_matrix,
            )
            states = chain_states[graph_idx]
            if states is None:
                projected_pos = fallback_pos
                projected_l = fallback_l.view(1, -1)
                projected_h = fallback_h
            else:
                valid_state_entries: list[
                    tuple[tuple[float, ...], Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                ] = []
                soft_physics_state_entries: list[
                    tuple[tuple[float, ...], Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                ] = []
                for state in states:
                    branch_label = (
                        f"graph={graph_idx + 1} "
                        f"target_repr={state.anchor_representation_name or state.target_representation_name or 'na'} "
                        f"template_rank={int(state.template_rank)} "
                        f"orbit_mismatch={int(state.species_orbit_mismatch)} "
                        f"prior_score={int(state.prior_score)}"
                    )
                    def _print_high_prior_candidate(
                        *,
                        status: str,
                        reason: str = "na",
                        ranking_objective: float = float("nan"),
                        mala_prox: float = float("nan"),
                        mala_like: float = float("nan"),
                        min_pair_distance: float = float("nan"),
                        volume_ratio: float | None = None,
                    ) -> None:
                        if not debug_high_prior_templates:
                            return
                        if int(getattr(state, "prior_score", 0)) < int(debug_high_prior_min_score):
                            return
                        print(
                            "algorithm6_high_prior_candidate "
                            f"graph={graph_idx + 1} phase={diagnostic_phase} "
                            f"template_rank={int(state.template_rank)} "
                            f"prior_score={int(state.prior_score)} "
                            f"prior_bonus={float(getattr(state, 'prior_bonus', 0.0)):.6f} "
                            f"status={status} reason={str(reason).replace(' ', '_')} "
                            f"ranking_objective={float(ranking_objective):.6f} "
                            f"mala_prox={float(mala_prox):.6f} "
                            f"mala_like={float(mala_like):.6f} "
                            f"min_pair_distance={float(min_pair_distance):.6f} "
                            f"volume_ratio={float(volume_ratio) if volume_ratio is not None else float('nan'):.6f} "
                            f"signature={_pcs_state_signature_labels(state)}",
                            flush=True,
                        )
                    try:
                        requested_space_group = int(batch.space_group[graph_idx].item())
                        projection = materialize_pcs_state(
                            state=state,
                            vanilla_reference_structure=current_structure,
                        )
                        if projection.standardized_space_group != requested_space_group:
                            raise RuntimeError(
                                "PCS conventional-space materialization does not satisfy the requested "
                                "space group "
                                f"({projection.standardized_space_group} vs requested "
                                f"{requested_space_group})."
                            )
                        if projection.primitive_space_group != requested_space_group:
                            raise RuntimeError(
                                "PCS primitive reduction does not preserve the requested space group "
                                f"({projection.primitive_space_group} vs requested "
                                f"{requested_space_group})."
                            )
                        candidate_pos, candidate_l, candidate_h = vanilla_structure_to_model_tensors(
                            structure=projection.projected_structure_vanilla,
                            lattice_transform=lattice_transform,
                            device=graph_pos.device,
                            dtype=graph_pos.dtype,
                        )
                        if candidate_pos.shape != graph_pos.shape:
                            raise RuntimeError(
                                "PCS materialization changed the atom count for one graph "
                                f"({candidate_pos.shape[0]} vs expected {graph_pos.shape[0]})."
                            )
                        if not _atomic_multiset_matches(candidate_h, graph_h):
                            raise RuntimeError(
                                "PCS materialization changed the composition for one graph, "
                                "which breaks the KLDM state representation."
                            )
                        rebuilt_cell_matrix = _decode_lattice_matrix(
                            l=candidate_l.view(1, -1),
                            num_atoms=int(candidate_pos.shape[0]),
                            lattice_transform=lattice_transform,
                        ).squeeze(0)
                        rebuilt_structure = self._algorithm6_graph_structure(
                            frac_coords=candidate_pos,
                            atomic_numbers=candidate_h,
                            cell_matrix=rebuilt_cell_matrix,
                        )
                        rebuilt_validation = validate_requested_space_group(
                            structure=rebuilt_structure,
                            requested_space_group=requested_space_group,
                            expected_atomic_numbers=candidate_h,
                            symprec=state.bridge.symprec,
                            angle_tolerance=state.bridge.angle_tolerance,
                        )
                        projected_reference_volume = (
                            state.projected_reference_volume
                            if getattr(state, "projected_reference_volume", None) is not None
                            else state.reference_volume
                        )
                        min_pair_distance, volume, volume_ratio = _projected_structure_debug_stats(
                            structure=rebuilt_structure,
                            reference_volume=projected_reference_volume,
                            cutoff=max(float(pcs_hard_min_distance), 1e-6),
                        )
                        if not rebuilt_validation.requested_space_group_match:
                            raise RuntimeError(
                                "PCS tensor rebuild does not satisfy the requested space group "
                                f"({rebuilt_validation.detected_space_group} vs requested "
                                f"{requested_space_group})."
                            )
                        updated_state = replace(
                            state,
                            ranking_objective=pcs_projected_objective(
                                state=state,
                                frac_coords=candidate_pos,
                                atomic_numbers=candidate_h,
                                cell_matrix=rebuilt_cell_matrix,
                                coord_weight=pcs_coord_weight,
                                lattice_weight=pcs_lattice_weight,
                                pairdist_weight=pcs_pairdist_weight,
                                pairdist_bins=pcs_pairdist_bins,
                                pairdist_max_distance=pcs_pairdist_max_distance,
                                pairdist_bandwidth=pcs_pairdist_bandwidth,
                                steric_weight=pcs_steric_weight,
                                steric_min_distance=pcs_steric_min_distance,
                                volume_weight=pcs_volume_weight,
                                volume_ratio_min=pcs_volume_ratio_min,
                                volume_ratio_max=pcs_volume_ratio_max,
                                k6_weight=pcs_k6_weight,
                            ),
                            branch_frac_coords=candidate_pos.detach().clone(),
                            branch_atomic_numbers=candidate_h.detach().clone(),
                            branch_lattice_features=candidate_l.reshape(-1).detach().clone(),
                        )
                        projected_signature_mismatch = int(state.species_orbit_mismatch)
                        try:
                            pyxtal_result = build_pyxtal_wyckoff_result(
                                rebuilt_structure,
                                symprec=state.bridge.symprec,
                                pyxtal_tol=max(float(state.bridge.symprec), 1e-3),
                            )
                            projected_signature_mismatch = _site_shape_mismatch_count(
                                expected_signature=_template_site_shape_signature(state),
                                recovered_signature=_pyxtal_site_shape_signature(pyxtal_result),
                            )
                        except Exception:
                            projected_signature_mismatch = int(state.species_orbit_mismatch)

                        physical_distance_floor = (
                            float(pcs_hard_min_distance)
                            if float(pcs_hard_min_distance) > 0.0
                            else float(pcs_steric_min_distance)
                        )
                        close_contact_deficit = max(
                            0.0,
                            physical_distance_floor - float(min_pair_distance),
                        )
                        rank_key = (
                            float(projected_signature_mismatch),
                            float(close_contact_deficit),
                            -float(min_pair_distance),
                            *_pcs_state_rank_key(updated_state),
                        )
                        candidate_triplet = (candidate_pos, candidate_l, candidate_h)
                        soft_physics_state_entries.append((rank_key, updated_state, candidate_triplet))
                        if enforce_physical_guards:
                            try:
                                _enforce_projected_physical_guards(
                                    structure=rebuilt_structure,
                                    reference_volume=projected_reference_volume,
                                    hard_min_distance=pcs_hard_min_distance,
                                    hard_volume_ratio_min=pcs_hard_volume_ratio_min,
                                    hard_volume_ratio_max=pcs_hard_volume_ratio_max,
                                )
                            except Exception as guard_exc:
                                if materialize_debug_records is not None:
                                    materialize_debug_records.append(
                                        {
                                            "phase": str(diagnostic_phase),
                                            "graph": int(graph_idx + 1),
                                            "template_rank": int(state.template_rank),
                                            "target_repr": (
                                                state.anchor_representation_name
                                                or state.target_representation_name
                                                or "na"
                                            ),
                                            "prior_score": int(state.prior_score),
                                            "prior_bonus": float(state.prior_bonus),
                                            "signature": _pcs_state_signature_labels(state),
                                            "status": "soft_physics_failed",
                                            "reason": f"{type(guard_exc).__name__}: {guard_exc}",
                                            "standardized_sg": int(projection.standardized_space_group),
                                            "primitive_sg": int(projection.primitive_space_group),
                                            "rebuilt_sg": int(rebuilt_validation.detected_space_group),
                                            "requested_sg": int(requested_space_group),
                                            "volume": float(volume),
                                            "reference_volume": (
                                                float(projected_reference_volume)
                                                if projected_reference_volume is not None
                                                else float("nan")
                                            ),
                                            "volume_ratio": (
                                                float(volume_ratio)
                                                if volume_ratio is not None
                                                else float("nan")
                                            ),
                                            "min_pair_distance": float(min_pair_distance),
                                            "projected_orbit_mismatch": int(projected_signature_mismatch),
                                            "ranking_objective": float(updated_state.ranking_objective),
                                            "mala_prox": float(updated_state.mala_prox_energy),
                                            "mala_like": float(updated_state.mala_likelihood_energy),
                                            "close_contact_deficit": float(close_contact_deficit),
                                        }
                                    )
                                if pcs_debug_template_candidates:
                                    print(
                                        "algorithm6_materialize_candidate "
                                        f"{branch_label} "
                                        f"status=soft_physics_failed reason={type(guard_exc).__name__}:"
                                        f"{str(guard_exc).replace(' ', '_')}",
                                        flush=True,
                                    )
                                _print_high_prior_candidate(
                                    status="soft_physics_failed",
                                    reason=f"{type(guard_exc).__name__}: {guard_exc}",
                                    ranking_objective=float(updated_state.ranking_objective),
                                    mala_prox=float(updated_state.mala_prox_energy),
                                    mala_like=float(updated_state.mala_likelihood_energy),
                                    min_pair_distance=float(min_pair_distance),
                                    volume_ratio=volume_ratio,
                                )
                                continue
                        if materialize_debug_records is not None:
                            materialize_debug_records.append(
                                {
                                    "phase": str(diagnostic_phase),
                                    "graph": int(graph_idx + 1),
                                    "template_rank": int(state.template_rank),
                                    "target_repr": (
                                        state.anchor_representation_name
                                        or state.target_representation_name
                                        or "na"
                                    ),
                                    "prior_score": int(state.prior_score),
                                    "prior_bonus": float(state.prior_bonus),
                                    "signature": _pcs_state_signature_labels(state),
                                    "status": "ok",
                                    "standardized_sg": int(projection.standardized_space_group),
                                    "primitive_sg": int(projection.primitive_space_group),
                                    "rebuilt_sg": int(rebuilt_validation.detected_space_group),
                                    "requested_sg": int(requested_space_group),
                                    "volume": float(volume),
                                    "reference_volume": (
                                        float(projected_reference_volume)
                                        if projected_reference_volume is not None
                                        else float("nan")
                                    ),
                                    "volume_ratio": (
                                        float(volume_ratio)
                                        if volume_ratio is not None
                                        else float("nan")
                                    ),
                                    "min_pair_distance": float(min_pair_distance),
                                    "projected_orbit_mismatch": int(projected_signature_mismatch),
                                    "ranking_objective": float(updated_state.ranking_objective),
                                    "mala_prox": float(updated_state.mala_prox_energy),
                                    "mala_like": float(updated_state.mala_likelihood_energy),
                                    "close_contact_deficit": float(close_contact_deficit),
                                }
                            )
                        if pcs_debug_template_candidates:
                            print(
                                "algorithm6_materialize_candidate "
                                f"{branch_label} "
                                f"standardized_sg={projection.standardized_space_group} "
                                f"primitive_sg={projection.primitive_space_group} "
                                f"rebuilt_sg={rebuilt_validation.detected_space_group} "
                                f"volume={volume:.6f} "
                                f"reference_volume={float(projected_reference_volume) if projected_reference_volume is not None else float('nan'):.6f} "
                                f"volume_ratio={float(volume_ratio) if volume_ratio is not None else float('nan'):.6f} "
                                f"min_pair_distance={min_pair_distance:.6f} "
                                f"projected_orbit_mismatch={int(projected_signature_mismatch)} "
                                f"ranking_objective={float(updated_state.ranking_objective):.6f} "
                                f"mala_prox={float(updated_state.mala_prox_energy):.6f} "
                                f"mala_like={float(updated_state.mala_likelihood_energy):.6f} "
                                "status=ok",
                                flush=True,
                            )
                        _print_high_prior_candidate(
                            status="ok",
                            ranking_objective=float(updated_state.ranking_objective),
                            mala_prox=float(updated_state.mala_prox_energy),
                            mala_like=float(updated_state.mala_likelihood_energy),
                            min_pair_distance=float(min_pair_distance),
                            volume_ratio=volume_ratio,
                        )
                        valid_state_entries.append((rank_key, updated_state, candidate_triplet))
                    except Exception as exc:
                        if materialize_debug_records is not None:
                            materialize_debug_records.append(
                                {
                                    "phase": str(diagnostic_phase),
                                    "graph": int(graph_idx + 1),
                                    "template_rank": int(getattr(state, "template_rank", -1)),
                                    "target_repr": (
                                        getattr(state, "anchor_representation_name", None)
                                        or getattr(state, "target_representation_name", None)
                                        or "na"
                                    ),
                                    "prior_score": int(getattr(state, "prior_score", 0)),
                                    "prior_bonus": float(getattr(state, "prior_bonus", 0.0)),
                                    "signature": _pcs_state_signature_labels(state),
                                    "status": "reject",
                                    "reason": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        if pcs_debug_template_candidates:
                            print(
                                "algorithm6_materialize_candidate "
                                f"{branch_label} "
                                f"status=reject reason={type(exc).__name__}:{str(exc).replace(' ', '_')}",
                                flush=True,
                            )
                        if pcs_debug_template_candidates:
                            warnings.warn(
                                "Algorithm 6 PCS materialization failed for one branch; dropping that branch. "
                                f"Reason: {type(exc).__name__}: {exc}",
                                stacklevel=2,
                            )
                        _print_high_prior_candidate(
                            status="reject",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                if valid_state_entries:
                    valid_state_entries.sort(key=lambda item: item[0])
                    chain_states[graph_idx] = [
                        state for _rank_key, state, _triplet in valid_state_entries
                    ]
                    selected_idx, selection_probability, selection_energy, selection_mode = (
                        _sample_branch_index(valid_state_entries, state_pos=1)
                    )
                    best_state = valid_state_entries[selected_idx][1]
                    projected_pos, projected_l, projected_h = valid_state_entries[selected_idx][2]
                    if materialize_debug_records is not None:
                        materialize_debug_records.append(
                            {
                                "phase": str(diagnostic_phase),
                                "graph": int(graph_idx + 1),
                                "template_rank": int(best_state.template_rank),
                                "target_repr": (
                                    best_state.anchor_representation_name
                                    or best_state.target_representation_name
                                    or "na"
                                ),
                                "status": "selected",
                                "prior_score": int(best_state.prior_score),
                                "prior_bonus": float(best_state.prior_bonus),
                                "signature": _pcs_state_signature_labels(best_state),
                                "num_valid_branches": int(len(valid_state_entries)),
                                "ranking_objective": float(best_state.ranking_objective),
                                "mala_prox": float(best_state.mala_prox_energy),
                                "mala_like": float(best_state.mala_likelihood_energy),
                                "selection_mode": str(selection_mode),
                                "selection_probability": float(selection_probability),
                                "selection_energy": float(selection_energy),
                                "branch_selection_temperature": float(branch_selection_temperature),
                            }
                        )
                    if pcs_debug_template_candidates:
                        print(
                            f"algorithm6_materialize_selected graph={graph_idx + 1} "
                            f"target_repr={best_state.anchor_representation_name or best_state.target_representation_name or 'na'} "
                            f"template_rank={int(best_state.template_rank)} "
                            f"orbit_mismatch={int(best_state.species_orbit_mismatch)} "
                            f"ranking_objective={float(best_state.ranking_objective):.6f} "
                            f"mala_prox={float(best_state.mala_prox_energy):.6f} "
                            f"mala_like={float(best_state.mala_likelihood_energy):.6f} "
                            f"num_valid_branches={len(valid_state_entries)} "
                            f"selection_mode={selection_mode} "
                            f"selection_probability={selection_probability:.6f} "
                            f"selection_energy={selection_energy:.6f}",
                            flush=True,
                        )
                    if debug_high_prior_templates and int(best_state.prior_score) >= int(debug_high_prior_min_score):
                        print(
                            "algorithm6_high_prior_candidate "
                            f"graph={graph_idx + 1} phase={diagnostic_phase} "
                            f"template_rank={int(best_state.template_rank)} "
                            f"prior_score={int(best_state.prior_score)} "
                            f"prior_bonus={float(best_state.prior_bonus):.6f} "
                            f"status=selected reason=selected_by_{selection_mode} "
                            f"ranking_objective={float(best_state.ranking_objective):.6f} "
                            f"mala_prox={float(best_state.mala_prox_energy):.6f} "
                            f"mala_like={float(best_state.mala_likelihood_energy):.6f} "
                            f"selection_probability={float(selection_probability):.6f} "
                            f"selection_energy={float(selection_energy):.6f} "
                            f"signature={_pcs_state_signature_labels(best_state)}",
                            flush=True,
                        )
                elif soft_physics_state_entries:
                    if not allow_soft_physics_fallback:
                        warnings.warn(
                            "Algorithm 6 PCS materialization found SG-valid branches, but all failed hard "
                            "physical guards; soft fallback is disabled, so this graph reverts to fallback.",
                            stacklevel=2,
                        )
                        chain_states[graph_idx] = None
                        projected_pos = fallback_pos
                        projected_l = fallback_l.view(1, -1)
                        projected_h = fallback_h
                        projected_pos_blocks.append(projected_pos)
                        projected_l_blocks.append(projected_l)
                        projected_h_blocks.append(projected_h)
                        continue
                    soft_physics_state_entries.sort(key=lambda item: item[0])
                    chain_states[graph_idx] = [
                        state for _rank_key, state, _triplet in soft_physics_state_entries
                    ]
                    selected_idx, selection_probability, selection_energy, selection_mode = (
                        _sample_branch_index(soft_physics_state_entries, state_pos=1)
                    )
                    best_state = soft_physics_state_entries[selected_idx][1]
                    projected_pos, projected_l, projected_h = soft_physics_state_entries[selected_idx][2]
                    if materialize_debug_records is not None:
                        materialize_debug_records.append(
                            {
                                "phase": str(diagnostic_phase),
                                "graph": int(graph_idx + 1),
                                "template_rank": int(best_state.template_rank),
                                "target_repr": (
                                    best_state.anchor_representation_name
                                    or best_state.target_representation_name
                                    or "na"
                                ),
                                "status": "selected_soft_physics_failed",
                                "prior_score": int(best_state.prior_score),
                                "prior_bonus": float(best_state.prior_bonus),
                                "signature": _pcs_state_signature_labels(best_state),
                                "num_valid_branches": 0,
                                "num_soft_physics_branches": int(len(soft_physics_state_entries)),
                                "ranking_objective": float(best_state.ranking_objective),
                                "mala_prox": float(best_state.mala_prox_energy),
                                "mala_like": float(best_state.mala_likelihood_energy),
                                "selection_mode": str(selection_mode),
                                "selection_probability": float(selection_probability),
                                "selection_energy": float(selection_energy),
                                "branch_selection_temperature": float(branch_selection_temperature),
                            }
                        )
                    warnings.warn(
                        "Algorithm 6 PCS materialization found SG-valid branches, but all failed hard "
                        "physical guards; using a selected SG-valid branch and marking it as "
                        "soft_physics_failed.",
                        stacklevel=2,
                    )
                    if debug_high_prior_templates and int(best_state.prior_score) >= int(debug_high_prior_min_score):
                        print(
                            "algorithm6_high_prior_candidate "
                            f"graph={graph_idx + 1} phase={diagnostic_phase} "
                            f"template_rank={int(best_state.template_rank)} "
                            f"prior_score={int(best_state.prior_score)} "
                            f"prior_bonus={float(best_state.prior_bonus):.6f} "
                            f"status=selected_soft_physics_failed reason=selected_by_{selection_mode} "
                            f"ranking_objective={float(best_state.ranking_objective):.6f} "
                            f"mala_prox={float(best_state.mala_prox_energy):.6f} "
                            f"mala_like={float(best_state.mala_likelihood_energy):.6f} "
                            f"selection_probability={float(selection_probability):.6f} "
                            f"selection_energy={float(selection_energy):.6f} "
                            f"signature={_pcs_state_signature_labels(best_state)}",
                            flush=True,
                        )
                else:
                    warnings.warn(
                        "Algorithm 6 PCS materialization failed for all branches of one graph; keeping "
                        "vanilla KLDM sampling for that graph from this point on.",
                        stacklevel=2,
                    )
                    chain_states[graph_idx] = None
                    projected_pos = fallback_pos
                    projected_l = fallback_l.view(1, -1)
                    projected_h = fallback_h
            projected_pos_blocks.append(projected_pos)
            projected_l_blocks.append(projected_l)
            projected_h_blocks.append(projected_h)

        return (
            torch.cat(projected_pos_blocks, dim=0),
            torch.cat(projected_l_blocks, dim=0),
            torch.cat(projected_h_blocks, dim=0),
        )

    def _algorithm6_print_debug_summary(
        self,
        *,
        debug_records: dict[str, list[dict[str, Any]]],
        chain_states: list[list[Any] | None],
        num_graphs: int,
        oracle_template_orbit_rerank: bool,
        oracle_template_fit_target: bool,
        dds_repair: bool,
        final_projection: bool,
    ) -> None:
        pcs_records = debug_records.get("pcs", [])
        materialize_records = debug_records.get("materialize", [])
        dds_records = debug_records.get("dds", [])
        template_records = debug_records.get("template", [])

        active_graphs = sum(states is not None for states in chain_states)
        active_branches = sum(len(states) for states in chain_states if states is not None)
        dds_status = Counter(str(record.get("status", "unknown")) for record in dds_records)
        dds_ran_records = [record for record in dds_records if record.get("status") == "ran"]
        dds_elapsed_total = sum(float(record.get("elapsed_s", 0.0)) for record in dds_ran_records)
        dds_pos_values = [float(record.get("pos_rmse_from_pcs", float("nan"))) for record in dds_ran_records]
        dds_l_values = [float(record.get("lattice_rmse_from_pcs", float("nan"))) for record in dds_ran_records]
        _, dds_pos_mean, dds_pos_max, _ = _debug_numeric_summary(dds_pos_values)
        _, dds_l_mean, dds_l_max, _ = _debug_numeric_summary(dds_l_values)

        anchor_values = [float(record.get("anchor_coord_rmse", float("nan"))) for record in pcs_records]
        anchor_l_values = [float(record.get("anchor_lattice_rmse", float("nan"))) for record in pcs_records]
        anchor_count, anchor_mean, anchor_max, anchor_min = _debug_numeric_summary(anchor_values)
        anchor_l_count, anchor_l_mean, anchor_l_max, anchor_l_min = _debug_numeric_summary(anchor_l_values)
        finite_anchor = [float(v) for v in anchor_values if np.isfinite(float(v))]
        anchor_positive = sum(abs(v) > 1e-8 for v in finite_anchor)

        prox_values = [float(record.get("prox_energy", float("nan"))) for record in pcs_records]
        like_values = [float(record.get("likelihood_energy", float("nan"))) for record in pcs_records]
        energy_values = [float(record.get("energy", float("nan"))) for record in pcs_records]
        prox_count, prox_mean, prox_max, _ = _debug_numeric_summary(prox_values)
        like_count, like_mean, like_max, _ = _debug_numeric_summary(like_values)
        energy_count, energy_mean, energy_max, _ = _debug_numeric_summary(energy_values)
        prox_nonzero = sum(abs(float(v)) > 1e-12 for v in prox_values if np.isfinite(float(v)))
        like_nonzero = sum(abs(float(v)) > 1e-12 for v in like_values if np.isfinite(float(v)))

        ok_records = [record for record in materialize_records if record.get("status") == "ok"]
        soft_physics_records = [
            record for record in materialize_records if record.get("status") == "soft_physics_failed"
        ]
        reject_records = [record for record in materialize_records if record.get("status") == "reject"]
        selected_records = [
            record
            for record in materialize_records
            if record.get("status") in {"selected", "selected_soft_physics_failed"}
        ]
        selected_soft_physics_records = [
            record
            for record in materialize_records
            if record.get("status") == "selected_soft_physics_failed"
        ]
        selected_by_graph = Counter(int(record.get("graph", -1)) for record in selected_records)
        min_pair_values = [float(record.get("min_pair_distance", float("nan"))) for record in ok_records]
        ranking_values = [float(record.get("ranking_objective", float("nan"))) for record in ok_records]
        min_pair_count, min_pair_mean, min_pair_max, min_pair_min = _debug_numeric_summary(min_pair_values)
        objective_count, objective_mean, objective_max, objective_min = _debug_numeric_summary(ranking_values)
        objective_nonzero = sum(abs(float(v)) > 1e-12 for v in ranking_values if np.isfinite(float(v)))
        sg_ok = sum(
            int(record.get("rebuilt_sg", -1)) == int(record.get("requested_sg", -2))
            for record in ok_records
        )
        template_prior_values = [float(record.get("prior_score", 0.0)) for record in template_records]
        template_prior_count, template_prior_mean, template_prior_max, _ = _debug_numeric_summary(template_prior_values)
        template_prior_nonzero = sum(int(float(v)) > 0 for v in template_prior_values if np.isfinite(float(v)))
        selected_prior_values = [float(record.get("prior_score", 0.0)) for record in selected_records]
        selected_prior_count, selected_prior_mean, selected_prior_max, _ = _debug_numeric_summary(selected_prior_values)
        selected_prior_nonzero = sum(int(float(v)) > 0 for v in selected_prior_values if np.isfinite(float(v)))
        selected_latest_by_graph: dict[int, dict[str, Any]] = {}
        for record in selected_records:
            selected_latest_by_graph[int(record.get("graph", -1))] = record

        dds_active = bool(dds_ran_records) and dds_elapsed_total > 0.0
        pcs_anchor_distance_positive = anchor_positive > 0
        coupling_visible = dds_active and pcs_anchor_distance_positive and prox_count > 0

        print(
            "algorithm6_debug_summary "
            f"active_graphs={active_graphs}/{int(num_graphs)} "
            f"active_branches={active_branches} "
            "anchor_source=current_full_space_iterate "
            f"dds_repair={int(bool(dds_repair))} "
            f"final_projection={int(bool(final_projection))} "
            f"oracle_template_orbit_rerank={int(bool(oracle_template_orbit_rerank))} "
            f"oracle_template_fit_target={int(bool(oracle_template_fit_target))}",
            flush=True,
        )
        print(
            "algorithm6_debug_dds "
            f"records={len(dds_records)} ran={int(dds_status.get('ran', 0))} "
            f"skipped={int(dds_status.get('skipped', 0))} disabled={int(dds_status.get('disabled', 0))} "
            f"elapsed_total_s={dds_elapsed_total:.3f} "
            f"pos_rmse_from_pcs_mean={dds_pos_mean:.6f} pos_rmse_from_pcs_max={dds_pos_max:.6f} "
            f"lattice_rmse_from_pcs_mean={dds_l_mean:.6f} lattice_rmse_from_pcs_max={dds_l_max:.6f}",
            flush=True,
        )
        print(
            "algorithm6_debug_pcs "
            f"branches={len(pcs_records)} "
            f"anchor_coord_count={anchor_count} anchor_coord_mean={anchor_mean:.6f} "
            f"anchor_coord_min={anchor_min:.6f} anchor_coord_max={anchor_max:.6f} "
            f"anchor_coord_positive={anchor_positive}/{len(finite_anchor)} "
            f"anchor_lattice_count={anchor_l_count} anchor_lattice_mean={anchor_l_mean:.6f} "
            f"anchor_lattice_min={anchor_l_min:.6f} anchor_lattice_max={anchor_l_max:.6f} "
            f"energy_count={energy_count} energy_mean={energy_mean:.6f} energy_max={energy_max:.6f} "
            f"prox_finite={prox_count}/{len(pcs_records)} prox_nonzero={prox_nonzero}/{prox_count} "
            f"prox_mean={prox_mean:.6f} prox_max={prox_max:.6f} "
            f"like_finite={like_count}/{len(pcs_records)} like_nonzero={like_nonzero}/{like_count} "
            f"like_mean={like_mean:.6f} like_max={like_max:.6f}",
            flush=True,
        )
        print(
            "algorithm6_debug_materialize "
            f"ok={len(ok_records)} soft_physics_failed={len(soft_physics_records)} "
            f"reject={len(reject_records)} selected={len(selected_records)} "
            f"selected_soft_physics_failed={len(selected_soft_physics_records)} "
            f"selected_graphs={len(selected_by_graph)} "
            f"sg_ok={sg_ok}/{len(ok_records)} "
            f"min_pair_count={min_pair_count} min_pair_mean={min_pair_mean:.6f} "
            f"min_pair_min={min_pair_min:.6f} min_pair_max={min_pair_max:.6f} "
            f"objective_count={objective_count} objective_mean={objective_mean:.6f} "
            f"objective_min={objective_min:.6f} objective_max={objective_max:.6f} "
            f"objective_nonzero={objective_nonzero}/{objective_count}",
            flush=True,
        )
        print(
            "algorithm6_debug_template_prior "
            f"pool_records={len(template_records)} prior_count={template_prior_count} "
            f"prior_nonzero={template_prior_nonzero}/{template_prior_count} "
            f"prior_mean={template_prior_mean:.6f} prior_max={template_prior_max:.6f} "
            f"selected_records={len(selected_records)} selected_prior_count={selected_prior_count} "
            f"selected_prior_nonzero={selected_prior_nonzero}/{selected_prior_count} "
            f"selected_prior_mean={selected_prior_mean:.6f} selected_prior_max={selected_prior_max:.6f}",
            flush=True,
        )
        templates_by_graph: dict[int, list[dict[str, Any]]] = {}
        for record in template_records:
            graph_idx = int(record.get("graph", -1))
            if graph_idx <= 0:
                continue
            templates_by_graph.setdefault(graph_idx, []).append(record)
        for graph_idx in sorted(templates_by_graph):
            top_records = sorted(
                templates_by_graph[graph_idx],
                key=lambda record: (
                    -int(record.get("prior_score", 0)),
                    float(record.get("ranking_objective", float("inf"))),
                    int(record.get("template_rank", 1_000_000)),
                ),
            )[:5]
            rendered = [
                {
                    "rank": int(record.get("template_rank", -1)),
                    "prior": int(record.get("prior_score", 0)),
                    "signature": record.get("signature", []),
                }
                for record in top_records
            ]
            print(
                f"algorithm6_debug_top_prior_templates graph={graph_idx} templates={rendered}",
                flush=True,
            )
        for graph_idx in sorted(key for key in selected_latest_by_graph if key > 0):
            record = selected_latest_by_graph[graph_idx]
            print(
                "algorithm6_debug_selected_template "
                f"graph={graph_idx} phase={record.get('phase', 'na')} "
                f"template_rank={int(record.get('template_rank', -1))} "
                f"prior_score={int(record.get('prior_score', 0))} "
                f"prior_bonus={float(record.get('prior_bonus', 0.0)):.6f} "
                f"mala_prox={float(record.get('mala_prox', float('nan'))):.6f} "
                f"mala_like={float(record.get('mala_like', float('nan'))):.6f} "
                f"selection_mode={record.get('selection_mode', 'na')} "
                f"selection_probability={float(record.get('selection_probability', float('nan'))):.6f} "
                f"selection_energy={float(record.get('selection_energy', float('nan'))):.6f} "
                f"signature={record.get('signature', [])}",
                flush=True,
            )
        print(
            "algorithm6_debug_verdict "
            f"dds_active={int(dds_active)} "
            f"pcs_anchor_distance_positive={int(pcs_anchor_distance_positive)} "
            f"posterior_coupling_visible={int(coupling_visible)}",
            flush=True,
        )

    def _algorithm6_restore_inactive_graphs(
        self,
        *,
        batch: Batch | Data,
        chain_states: list[list[Any] | None],
        pos_reference: torch.Tensor,
        v_reference: torch.Tensor,
        l_reference: torch.Tensor,
        h_reference: torch.Tensor,
        pos_candidate: torch.Tensor,
        v_candidate: torch.Tensor,
        l_candidate: torch.Tensor,
        h_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []
        l_blocks: list[torch.Tensor] = []
        h_blocks: list[torch.Tensor] = []
        ptr = batch.ptr.tolist()

        for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
            if chain_states[graph_idx] is None:
                pos_blocks.append(pos_reference[start_idx:end_idx])
                v_blocks.append(v_reference[start_idx:end_idx])
                l_blocks.append(l_reference[graph_idx].view(1, -1))
                h_blocks.append(h_reference[start_idx:end_idx])
            else:
                pos_blocks.append(pos_candidate[start_idx:end_idx])
                v_blocks.append(v_candidate[start_idx:end_idx])
                l_blocks.append(l_candidate[graph_idx].view(1, -1))
                h_blocks.append(h_candidate[start_idx:end_idx])

        return (
            torch.cat(pos_blocks, dim=0),
            torch.cat(v_blocks, dim=0),
            torch.cat(l_blocks, dim=0),
            torch.cat(h_blocks, dim=0),
        )

    @staticmethod
    def _algorithm6_graph_structure(
        *,
        frac_coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell_matrix: torch.Tensor,
    ):
        if None in (Element, Lattice, Structure):
            raise ImportError("Algorithm 6 structure materialization requires pymatgen.")
        species = [Element.from_Z(int(z)).symbol for z in atomic_numbers.detach().cpu().tolist()]
        return Structure(
            lattice=Lattice(cell_matrix.detach().cpu().numpy()),
            species=species,
            coords=torch.remainder(frac_coords, 1.0).detach().cpu().numpy().tolist(),
            coords_are_cartesian=False,
        ).get_sorted_structure()

    def _algorithm6_validate_batch_constraints(
        self,
        *,
        batch: Batch | Data,
        pos_t: torch.Tensor,
        v_t: torch.Tensor,
        l_t: torch.Tensor,
        h_t: torch.Tensor,
        pos_reference: torch.Tensor,
        v_reference: torch.Tensor,
        l_reference: torch.Tensor,
        h_reference: torch.Tensor,
        lattice_transform: ContinuousIntervalLattice | None,
        pcs_symprec: float,
        pcs_angle_tolerance: float,
        chain_states: list[list[Any] | None],
        revert_on_failure: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr = batch.ptr.tolist()
        requested_space_groups = torch.as_tensor(
            batch.space_group,
            device=pos_t.device,
            dtype=torch.long,
        ).reshape(-1)
        pos_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []
        l_blocks: list[torch.Tensor] = []
        h_blocks: list[torch.Tensor] = []

        for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
            if chain_states[graph_idx] is None:
                pos_blocks.append(pos_t[start_idx:end_idx])
                v_blocks.append(v_t[start_idx:end_idx])
                l_blocks.append(l_t[graph_idx].view(1, -1))
                h_blocks.append(h_t[start_idx:end_idx])
                continue
            graph_pos = pos_t[start_idx:end_idx]
            graph_v = v_t[start_idx:end_idx]
            graph_h = h_t[start_idx:end_idx]
            graph_l = l_t[graph_idx]
            cell_matrix = _decode_lattice_matrix(
                l=graph_l.view(1, -1),
                num_atoms=int(graph_pos.shape[0]),
                lattice_transform=lattice_transform,
            ).squeeze(0)
            structure = self._algorithm6_graph_structure(
                frac_coords=graph_pos,
                atomic_numbers=graph_h,
                cell_matrix=cell_matrix,
            )
            validation = validate_requested_space_group(
                structure=structure,
                requested_space_group=int(requested_space_groups[graph_idx].item()),
                expected_atomic_numbers=graph_h,
                symprec=pcs_symprec,
                angle_tolerance=pcs_angle_tolerance,
            )
            if validation.composition_match and validation.requested_space_group_match:
                pos_blocks.append(graph_pos)
                v_blocks.append(graph_v)
                l_blocks.append(graph_l.view(1, -1))
                h_blocks.append(graph_h)
                continue

            if not validation.composition_match:
                reason = "final composition mismatch"
            else:
                reason = (
                    "detected space group "
                    f"{validation.detected_space_group} does not match requested "
                    f"{validation.requested_space_group}"
                )
            if revert_on_failure:
                warnings.warn(
                    "Algorithm 6 rejected one constrained graph after validation; reverting that graph to "
                    f"its fallback sample. Reason: {reason}.",
                    stacklevel=2,
                )
                chain_states[graph_idx] = None
                pos_blocks.append(pos_reference[start_idx:end_idx])
                v_blocks.append(v_reference[start_idx:end_idx])
                l_blocks.append(l_reference[graph_idx].view(1, -1))
                h_blocks.append(h_reference[start_idx:end_idx])
            else:
                warnings.warn(
                    "Algorithm 6 validation found one constrained graph that does not satisfy the requested "
                    f"space group; keeping the sample and reporting the mismatch. Reason: {reason}.",
                    stacklevel=2,
                )
                pos_blocks.append(graph_pos)
                v_blocks.append(graph_v)
                l_blocks.append(graph_l.view(1, -1))
                h_blocks.append(graph_h)

        return (
            torch.cat(pos_blocks, dim=0),
            torch.cat(v_blocks, dim=0),
            torch.cat(l_blocks, dim=0),
            torch.cat(h_blocks, dim=0),
        )

    def _algorithm6_dds_repair(
        self,
        *,
        batch: Batch | Data,
        pos_clean: torch.Tensor,
        l_clean: torch.Tensor,
        h_clean: torch.Tensor,
        n_steps: int,
        t_start: float,
        t_final: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        repair_state = self._prepare_csp_sampling(
            batch=batch,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
            initial_f=pos_clean,
            initial_l=l_clean,
            initial_a=h_clean,
            initialize_from_dds_anchor=True,
        )
        repair_state = self._run_csp_em_reverse_chain(repair_state)

        if repair_state["restore_training"]:
            repair_state["score_network"].train()

        return repair_state["f_t"], repair_state["v_t"], repair_state["l_t"], repair_state["a_t"]


def main() -> None:
    device = get_default_device()

    from kldmPlus.data import CSPTask, resolve_data_root
    root = resolve_data_root()

    loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )
    batch = next(iter(loader)).to(device)

    model = ModelKLDM(
        device=device,
        score_network_kwargs={
            "hidden_dim": 512,
            "time_dim": 256,
            "num_layers": 6,
            "num_freqs": 128,
            "ln": True,
            "h_dim": 100,
            "smooth": False,
            "pred_v": True,
            "pred_l": True,
            "pred_h": False,
            "zero_cog": True,
        },
    ).to(device)

    pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
        n_steps=1000,
        batch=batch,
    )

    print("Sampled one CSP crystal")
    print("pos shape:", tuple(pos_t.shape))
    print("v shape:", tuple(v_t.shape))
    print("l shape:", tuple(l_t.shape))
    print("h shape:", tuple(h_t.shape))

    print("\nFirst 3 sampled fractional coordinates:")
    print(pos_t[:3])

    print("\nSampled lattice:")
    print(l_t)

if __name__ == "__main__":
    main()
