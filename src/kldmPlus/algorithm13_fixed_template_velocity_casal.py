from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import torch

from kldmPlus.fixed_template_ssvd_project import (
    FixedTemplateProjectionResult,
    ProjectionMetric,
    SSVDProjectionConfig,
    fixed_template_ssvd_project,
)
from kldmPlus.symmetry.k_basis import free_vars_to_k, k_to_cell_matrix, k_to_free_vars
from kldmPlus.symmetry.pcs_projection import PCSTemplateState, _periodic_pairwise_distances, _species_assignment_indices
from kldmPlus.symmetry.wyckoff_templates import (
    expand_wyckoff_template_torch,
    recover_template_free_vars_from_anchor_entries,
)


def wrap_residual(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.remainder(f - z + 0.5, 1.0) - 0.5


def center_velocity(v: torch.Tensor) -> tuple[torch.Tensor, float]:
    centered = v - v.mean(dim=0, keepdim=True)
    mean_norm = float(torch.linalg.norm(v.mean(dim=0)).detach().item())
    return centered, mean_norm


def kinetic_position_update(f: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    return torch.remainder(f - float(dt) * v, 1.0)


@dataclass(frozen=True)
class FixedTemplateVelocityProjection:
    theta: torch.Tensor
    z_frac: torch.Tensor
    z_k: torch.Tensor
    z_l: torch.Tensor
    tau: torch.Tensor
    assignment: torch.Tensor
    objective: float
    coord_loss: float
    lattice_k_loss: float
    physical_lattice_loss: float
    steric_loss: float
    volume_loss: float
    min_pair_distance: float
    raw: FixedTemplateProjectionResult


@dataclass(frozen=True)
class FixedTemplateMaterialization:
    frac_coords: torch.Tensor
    atomic_numbers: torch.Tensor
    k: torch.Tensor
    cell: torch.Tensor


@dataclass(frozen=True)
class TangentProjector:
    J: torch.Tensor
    metric: torch.Tensor | None
    damping: float
    rank: int
    condition_number: float
    gram: torch.Tensor

    def _jt_m(self) -> torch.Tensor:
        if self.metric is None:
            return self.J.transpose(0, 1)
        return self.J.transpose(0, 1) @ self.metric

    def project_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.J.shape[-1] == 0:
            return torch.zeros_like(flat)
        rhs = self._jt_m() @ flat
        eye = torch.eye(
            self.gram.shape[0],
            device=self.gram.device,
            dtype=self.gram.dtype,
        )
        gram_reg = self.gram + float(self.damping) * eye
        try:
            coeff = torch.linalg.solve(gram_reg, rhs)
        except Exception:
            coeff = torch.linalg.pinv(gram_reg) @ rhs
        return self.J @ coeff

    def project(self, v: torch.Tensor) -> torch.Tensor:
        return self.project_flat(v.reshape(-1)).reshape_as(v)

    def residual_norm(self, v: torch.Tensor) -> float:
        projected = self.project(v)
        residual = v - projected
        return float(torch.linalg.norm(residual.reshape(-1)).detach().item())


@dataclass(frozen=True)
class FixedTemplateVelocityConfig:
    projection_metric: ProjectionMetric = field(
        default_factory=lambda: ProjectionMetric(coord_weight=1.0, lattice_weight=0.0)
    )
    projection_config: SSVDProjectionConfig = field(
        default_factory=lambda: SSVDProjectionConfig(
            max_steps=16,
            random_restarts=0,
            freeze_tau=True,
            use_fixed_assignment=True,
            physical_lattice_weight=0.0,
            steric_weight=0.0,
            min_pair_weight=0.0,
            volume_weight=0.0,
            volume_ratio_min=0.0,
            volume_ratio_max=1.0e9,
        )
    )
    projector_damping: float = 1.0e-6
    mean_free_velocity: bool = True
    local_trust_radius: float | None = 0.05
    local_reg_weight: float = 1.0
    no_opt_eps: float = 1.0e-6


def materialize_template(
    theta: torch.Tensor,
    template_state: PCSTemplateState,
    *,
    tau: torch.Tensor | None = None,
) -> FixedTemplateMaterialization:
    expansion = expand_wyckoff_template_torch(
        template=template_state.template,
        free_vars=theta.reshape(-1),
    )
    if tau is None:
        tau = torch.zeros(
            1,
            3,
            device=expansion.frac_coords.device,
            dtype=expansion.frac_coords.dtype,
        )
    frac_coords = torch.remainder(
        expansion.frac_coords + tau.to(device=expansion.frac_coords.device, dtype=expansion.frac_coords.dtype),
        1.0,
    )
    k = free_vars_to_k(
        template_state.lattice_free_vars.to(
            device=frac_coords.device,
            dtype=frac_coords.dtype,
        ),
        template_state.constraint,
    )
    cell = k_to_cell_matrix(k)
    return FixedTemplateMaterialization(
        frac_coords=frac_coords,
        atomic_numbers=expansion.atomic_numbers.to(device=frac_coords.device, dtype=torch.long),
        k=k,
        cell=cell,
    )


def canonicalize_template_theta(
    theta: torch.Tensor,
    template_state: PCSTemplateState,
) -> torch.Tensor:
    theta = theta.reshape(-1)
    if theta.numel() == 0:
        return theta.detach().clone()
    try:
        expansion = expand_wyckoff_template_torch(
            template=template_state.template,
            free_vars=theta.detach().clone(),
            wrap=True,
        )
        anchor_coords = expansion.anchor_coords
        if anchor_coords.ndim != 2 or anchor_coords.shape[0] != len(template_state.template.site_templates):
            return theta.detach().clone()
        anchor_entries = [
            {
                "atomic_number": int(site.atomic_number),
                "label": str(site.label),
                "anchor_frac": torch.remainder(anchor_coords[site_idx], 1.0).detach().cpu().numpy(),
            }
            for site_idx, site in enumerate(template_state.template.site_templates)
        ]
        recovered = recover_template_free_vars_from_anchor_entries(
            template_state.template,
            anchor_entries,
        )
        recovered = recovered.to(device=theta.device, dtype=theta.dtype).reshape(-1)
        if recovered.numel() != theta.numel() or not torch.isfinite(recovered).all():
            return theta.detach().clone()
        return recovered.detach().clone()
    except Exception:
        return theta.detach().clone()


def project_to_fixed_template(
    *,
    f_frac: torch.Tensor,
    atomic_numbers: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    config: FixedTemplateVelocityConfig | None = None,
    canonicalize_theta_result: bool = True,
) -> FixedTemplateVelocityProjection:
    cfg = config or FixedTemplateVelocityConfig()
    result = fixed_template_ssvd_project(
        template_state=template_state,
        y_f=f_frac,
        y_k=target_k,
        y_h=atomic_numbers.to(device=f_frac.device, dtype=torch.long),
        tau0=tau0.to(device=f_frac.device, dtype=f_frac.dtype),
        metric=cfg.projection_metric,
        config=cfg.projection_config,
    )
    theta_raw = result.state.free_vars.detach().clone()
    theta = canonicalize_template_theta(theta_raw, result.state) if canonicalize_theta_result else theta_raw.detach().clone()
    canonical_materialized = materialize_template(theta, result.state, tau=result.tau)
    canonical_state = replace(
        result.state,
        free_vars=theta.detach().clone(),
        anchor_free_vars=theta.detach().clone(),
        anchor_lattice_free_vars=result.state.lattice_free_vars.detach().clone(),
        branch_frac_coords=canonical_materialized.frac_coords.detach().clone(),
        branch_atomic_numbers=canonical_materialized.atomic_numbers.detach().clone(),
        branch_lattice_features=canonical_materialized.cell.detach().reshape(-1).clone(),
    )
    canonical_materialized = materialize_template(theta, canonical_state, tau=result.tau)
    z_k = canonical_materialized.k.detach().clone()
    z_l = canonical_materialized.cell.detach().reshape(-1)
    canonical_result = replace(
        result,
        state=canonical_state,
        frac_coords_chart=canonical_materialized.frac_coords.detach().clone(),
        atomic_numbers_chart=canonical_materialized.atomic_numbers.detach().clone(),
        k=canonical_materialized.k.detach().clone(),
        cell=canonical_materialized.cell.detach().clone(),
    )
    return FixedTemplateVelocityProjection(
        theta=theta,
        z_frac=canonical_materialized.frac_coords.detach().clone(),
        z_k=z_k,
        z_l=z_l,
        tau=result.tau.detach().clone(),
        assignment=result.assignment.detach().clone(),
        objective=float(result.objective),
        coord_loss=float(result.coord_loss),
        lattice_k_loss=float(result.lattice_k_loss),
        physical_lattice_loss=float(result.physical_lattice_loss),
        steric_loss=float(result.steric_loss),
        volume_loss=float(result.volume_loss),
        min_pair_distance=float(result.min_pair_distance),
        raw=canonical_result,
    )


def locked_local_template_state(
    *,
    template_state: PCSTemplateState,
    theta: torch.Tensor | None = None,
    target_k: torch.Tensor | None = None,
    fixed_assignment: torch.Tensor | None = None,
    target_frac: torch.Tensor | None = None,
    target_atomic_numbers: torch.Tensor | None = None,
) -> PCSTemplateState:
    local_theta = (
        template_state.free_vars.detach().clone()
        if theta is None
        else theta.detach().clone().to(
            device=template_state.free_vars.device,
            dtype=template_state.free_vars.dtype,
        )
    ).reshape(-1)
    if target_k is None:
        local_lattice_free = template_state.lattice_free_vars.detach().clone().reshape(-1)
        local_target_k = (
            template_state.target_k.detach().clone()
            if template_state.target_k is not None
            else free_vars_to_k(local_lattice_free, template_state.constraint).detach().clone()
        )
    else:
        local_target_k = target_k.detach().clone().to(
            device=template_state.lattice_free_vars.device,
            dtype=template_state.lattice_free_vars.dtype,
        ).reshape(-1)
        local_lattice_free = k_to_free_vars(local_target_k, template_state.constraint).detach().clone().reshape(-1)
    local_target_cell = k_to_cell_matrix(local_target_k).detach().clone()
    local_assignment = fixed_assignment
    if local_assignment is None:
        local_assignment = template_state.fixed_target_assignment
    if local_assignment is None:
        local_assignment = template_state.anchor_assignment
    local_assignment = (
        None
        if local_assignment is None
        else local_assignment.detach().clone().to(device=local_theta.device, dtype=torch.long).reshape(-1)
    )
    local_target_frac = template_state.target_frac if target_frac is None else target_frac
    local_target_frac = (
        None
        if local_target_frac is None
        else local_target_frac.detach().clone().to(device=local_theta.device, dtype=local_theta.dtype)
    )
    local_target_atomic_numbers = template_state.target_atomic_numbers if target_atomic_numbers is None else target_atomic_numbers
    local_target_atomic_numbers = (
        None
        if local_target_atomic_numbers is None
        else local_target_atomic_numbers.detach().clone().to(device=local_theta.device, dtype=torch.long)
    )
    return replace(
        template_state,
        free_vars=local_theta,
        lattice_free_vars=local_lattice_free,
        target_frac=local_target_frac,
        target_atomic_numbers=local_target_atomic_numbers,
        target_k=local_target_k,
        target_cell=local_target_cell,
        fixed_target_assignment=local_assignment,
        anchor_frac=local_target_frac if local_target_frac is not None else template_state.anchor_frac,
        anchor_atomic_numbers=(
            local_target_atomic_numbers if local_target_atomic_numbers is not None else template_state.anchor_atomic_numbers
        ),
        anchor_k=local_target_k,
        anchor_cell=local_target_cell,
        anchor_assignment=local_assignment if local_assignment is not None else template_state.anchor_assignment,
        anchor_free_vars=local_theta.detach().clone(),
        anchor_lattice_free_vars=local_lattice_free.detach().clone(),
    )


def project_to_fixed_template_local(
    *,
    f_frac: torch.Tensor,
    atomic_numbers: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    theta0: torch.Tensor | None = None,
    fixed_assignment: torch.Tensor | None = None,
    config: FixedTemplateVelocityConfig | None = None,
) -> FixedTemplateVelocityProjection:
    cfg = config or FixedTemplateVelocityConfig()
    local_cfg = replace(
        cfg,
        projection_config=replace(
            cfg.projection_config,
            random_restarts=0,
            use_fixed_assignment=True,
            local_reg_weight=float(cfg.local_reg_weight),
            local_trust_radius=cfg.local_trust_radius,
            local_theta_reference=None if theta0 is None else theta0.detach().clone().reshape(-1),
        ),
    )
    local_state = locked_local_template_state(
        template_state=template_state,
        theta=theta0,
        target_k=target_k,
        fixed_assignment=fixed_assignment,
        target_atomic_numbers=atomic_numbers,
    )
    theta_ref = local_state.free_vars.detach().clone().reshape(-1)
    z_ref = materialize_template(theta_ref, local_state, tau=tau0).frac_coords.detach().clone()
    if float(torch.linalg.norm(wrap_residual(f_frac, z_ref).reshape(-1)).detach().item()) < float(cfg.no_opt_eps):
        assignment = fixed_assignment
        if assignment is None:
            assignment = local_state.fixed_target_assignment
        if assignment is None:
            assignment = local_state.anchor_assignment
        if assignment is None:
            assignment = _species_assignment_indices(
                source_frac=z_ref,
                source_atomic_numbers=atomic_numbers.to(device=f_frac.device, dtype=torch.long),
                target_frac=f_frac,
                target_atomic_numbers=atomic_numbers.to(device=f_frac.device, dtype=torch.long),
            ).to(device=f_frac.device, dtype=torch.long)
        else:
            assignment = assignment.detach().clone().to(device=f_frac.device, dtype=torch.long).reshape(-1)
        cell = k_to_cell_matrix(target_k.detach().clone().reshape(-1)).detach().clone()
        pair_distances = _periodic_pairwise_distances(frac_coords=z_ref, cell_matrix=cell)
        min_pair_distance = float(pair_distances.min().detach().item()) if pair_distances.numel() else float("inf")
        objective = float(torch.linalg.norm(wrap_residual(f_frac, z_ref).reshape(-1)).detach().item() ** 2)
        identity_state = replace(
            local_state,
            objective=objective,
            ranking_objective=objective,
            branch_frac_coords=z_ref.detach().clone(),
            branch_atomic_numbers=atomic_numbers.detach().clone().to(device=f_frac.device, dtype=torch.long),
            branch_lattice_features=cell.detach().reshape(-1).clone(),
        )
        identity_raw = FixedTemplateProjectionResult(
            state=identity_state,
            tau=tau0.detach().clone(),
            objective=objective,
            coord_loss=objective,
            lattice_k_loss=0.0,
            physical_lattice_loss=0.0,
            steric_loss=0.0,
            volume_loss=0.0,
            min_pair_distance=min_pair_distance,
            ssvd_rank=0,
            ssvd_condition_number=float("inf"),
            ssvd_delta_norm=0.0,
            ssvd_steps=0,
            ssvd_line_search_accepts=0,
            ssvd_line_search_failures=0,
            ssvd_clip_count=0,
            ssvd_min_sigma=float("nan"),
            ssvd_max_sigma=float("nan"),
            objective_initial=objective,
            frac_coords_chart=z_ref.detach().clone(),
            atomic_numbers_chart=atomic_numbers.detach().clone().to(device=f_frac.device, dtype=torch.long),
            k=target_k.detach().clone().reshape(-1),
            cell=cell,
            assignment=assignment,
        )
        return FixedTemplateVelocityProjection(
            theta=theta_ref.detach().clone(),
            z_frac=z_ref.detach().clone(),
            z_k=target_k.detach().clone().reshape(-1),
            z_l=cell.detach().reshape(-1).clone(),
            tau=tau0.detach().clone(),
            assignment=assignment.detach().clone(),
            objective=objective,
            coord_loss=objective,
            lattice_k_loss=0.0,
            physical_lattice_loss=0.0,
            steric_loss=0.0,
            volume_loss=0.0,
            min_pair_distance=min_pair_distance,
            raw=identity_raw,
        )
    return project_to_fixed_template(
        f_frac=f_frac,
        atomic_numbers=atomic_numbers,
        template_state=local_state,
        target_k=target_k,
        tau0=tau0,
        config=local_cfg,
        canonicalize_theta_result=False,
    )


def compute_template_jacobian(
    theta: torch.Tensor,
    template_state: PCSTemplateState,
    *,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    theta = theta.reshape(-1)
    materialized = materialize_template(theta, template_state, tau=tau)
    flat_dim = int(materialized.frac_coords.numel())
    if theta.numel() == 0:
        return torch.zeros(
            flat_dim,
            0,
            device=materialized.frac_coords.device,
            dtype=materialized.frac_coords.dtype,
        )

    def _flat_materialization(theta_flat: torch.Tensor) -> torch.Tensor:
        return materialize_template(theta_flat, template_state, tau=tau).frac_coords.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        _flat_materialization,
        theta.detach().clone().requires_grad_(True),
        vectorize=True,
    )
    return jacobian.reshape(flat_dim, theta.numel()).detach()


def finite_difference_jacobian_error(
    *,
    theta: torch.Tensor,
    direction: torch.Tensor,
    epsilon: float,
    template_state: PCSTemplateState,
    tau: torch.Tensor | None = None,
    jacobian: torch.Tensor | None = None,
) -> tuple[float, float]:
    theta = theta.reshape(-1)
    direction = direction.reshape(-1)
    eps = float(epsilon)
    if jacobian is None:
        jacobian = compute_template_jacobian(theta, template_state, tau=tau)
    base = materialize_template(theta, template_state, tau=tau).frac_coords.reshape(-1)
    shifted = materialize_template(theta + eps * direction, template_state, tau=tau).frac_coords.reshape(-1)
    finite_difference = wrap_residual(shifted.reshape_as(base), base.reshape_as(base)).reshape(-1) / eps
    linearized = jacobian @ direction
    abs_error = float(torch.linalg.norm(finite_difference - linearized).detach().item())
    denom = max(float(torch.linalg.norm(finite_difference).detach().item()), 1.0e-12)
    return abs_error, abs_error / denom


def build_fractional_metric(*, num_atoms: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(3 * int(num_atoms), device=device, dtype=dtype)


def build_cartesian_block_metric(cell_matrix: torch.Tensor, *, num_atoms: int) -> torch.Tensor:
    gram = cell_matrix @ cell_matrix.transpose(0, 1)
    eye_n = torch.eye(int(num_atoms), device=gram.device, dtype=gram.dtype)
    return torch.kron(eye_n, gram)


def tangent_projector(
    J: torch.Tensor,
    *,
    metric: torch.Tensor | None = None,
    damping: float = 1.0e-6,
) -> TangentProjector:
    if J.ndim != 2:
        raise ValueError(f"Expected J.ndim == 2, got {J.ndim}.")
    if J.shape[1] == 0:
        gram = torch.zeros(0, 0, device=J.device, dtype=J.dtype)
        return TangentProjector(
            J=J,
            metric=metric,
            damping=float(damping),
            rank=0,
            condition_number=float("inf"),
            gram=gram,
        )

    if metric is None:
        gram = J.transpose(0, 1) @ J
    else:
        gram = J.transpose(0, 1) @ metric @ J

    sigma = torch.linalg.svdvals(gram)
    sigma = sigma[torch.isfinite(sigma)]
    positive = sigma[sigma > max(float(damping), 1.0e-10)]
    rank = int(positive.numel())
    if rank <= 1:
        condition_number = 1.0 if rank == 1 else float("inf")
    else:
        condition_number = float((positive.max() / positive.min()).detach().item())
    return TangentProjector(
        J=J,
        metric=metric,
        damping=float(damping),
        rank=rank,
        condition_number=condition_number,
        gram=gram.detach(),
    )


def tangent_project_velocity(
    v: torch.Tensor,
    *,
    J: torch.Tensor,
    metric: torch.Tensor | None = None,
    damping: float = 1.0e-6,
    mean_free: bool = True,
) -> tuple[torch.Tensor, TangentProjector, float]:
    projector = tangent_projector(J, metric=metric, damping=damping)
    projected = projector.project(v)
    mean_norm_before = 0.0
    if mean_free:
        projected, mean_norm_before = center_velocity(projected)
    return projected, projector, mean_norm_before


def reduced_chart_velocity(
    v: torch.Tensor,
    *,
    J: torch.Tensor,
    metric: torch.Tensor | None = None,
    damping: float = 1.0e-6,
) -> torch.Tensor:
    if J.shape[1] == 0:
        return torch.zeros(0, device=v.device, dtype=v.dtype)
    if metric is None:
        gram = J.transpose(0, 1) @ J
        rhs = J.transpose(0, 1) @ v.reshape(-1)
    else:
        gram = J.transpose(0, 1) @ metric @ J
        rhs = J.transpose(0, 1) @ metric @ v.reshape(-1)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    try:
        return torch.linalg.solve(gram + float(damping) * eye, rhs)
    except Exception:
        return torch.linalg.pinv(gram + float(damping) * eye) @ rhs


def lift_reduced_chart_velocity(
    omega: torch.Tensor,
    *,
    J: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    if omega.numel() == 0:
        return torch.zeros(shape, device=J.device, dtype=J.dtype)
    return (J @ omega.reshape(-1)).reshape(shape)


def apply_full_space_force(
    v: torch.Tensor,
    *,
    residual: torch.Tensor,
    step_size: float,
    mean_free: bool = True,
) -> torch.Tensor:
    updated = v + float(step_size) * residual
    if mean_free:
        updated, _ = center_velocity(updated)
    return updated


def apply_reduced_space_force(
    omega: torch.Tensor,
    *,
    residual: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    return omega + float(step_size) * residual


def projector_idempotence_error(projector: TangentProjector, v: torch.Tensor) -> float:
    once = projector.project(v)
    twice = projector.project(once)
    return float(torch.linalg.norm((twice - once).reshape(-1)).detach().item())


def tangent_residual_after_centering(
    v: torch.Tensor,
    *,
    J: torch.Tensor,
    metric: torch.Tensor | None = None,
    damping: float = 1.0e-6,
) -> float:
    centered, _ = center_velocity(v)
    projector = tangent_projector(J, metric=metric, damping=damping)
    residual = centered - projector.project(centered)
    return float(torch.linalg.norm(residual.reshape(-1)).detach().item())


def site_row_slices(template_state: PCSTemplateState) -> list[slice]:
    start = 0
    slices: list[slice] = []
    for site in template_state.template.site_templates:
        width = int(site.multiplicity) * 3
        slices.append(slice(start, start + width))
        start += width
    return slices


def theta_column_slices(template_state: PCSTemplateState) -> list[slice]:
    start = 0
    slices: list[slice] = []
    for site in template_state.template.site_templates:
        width = int(site.dof)
        slices.append(slice(start, start + width))
        start += width
    return slices


def site_jacobian_block_ranks(
    J: torch.Tensor,
    *,
    template_state: PCSTemplateState,
) -> list[int]:
    row_slices = site_row_slices(template_state)
    col_slices = theta_column_slices(template_state)
    ranks: list[int] = []
    for row_slice, col_slice in zip(row_slices, col_slices, strict=True):
        block = J[row_slice, col_slice]
        if block.numel() == 0:
            ranks.append(0)
            continue
        ranks.append(int(torch.linalg.matrix_rank(block).detach().item()))
    return ranks


def graph_velocity_norm(v: torch.Tensor) -> float:
    return float(torch.linalg.norm(v.reshape(-1)).detach().item())


def mean_norm(v: torch.Tensor) -> float:
    return float(torch.linalg.norm(v.mean(dim=0)).detach().item())
