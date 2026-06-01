from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from kldmPlus.symmetry.k_basis import free_vars_to_k, k_to_cell_matrix
from kldmPlus.symmetry.pcs_projection import (
    PCSTemplateState,
    _periodic_pairwise_distances,
    _species_assignment_indices,
)
from kldmPlus.symmetry.wyckoff_templates import expand_wyckoff_template_torch, sample_random_free_vars


def _wrap_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(delta + 0.5, 1.0) - 0.5


def _cell_lengths_and_angles(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a_vec, b_vec, c_vec = cell.unbind(dim=-2)
    lengths = torch.stack(
        [
            torch.linalg.norm(a_vec),
            torch.linalg.norm(b_vec),
            torch.linalg.norm(c_vec),
        ],
        dim=0,
    ).clamp_min(1.0e-8)
    alpha = torch.acos(
        torch.clamp(torch.dot(b_vec, c_vec) / (lengths[1] * lengths[2]), min=-1.0, max=1.0)
    )
    beta = torch.acos(
        torch.clamp(torch.dot(a_vec, c_vec) / (lengths[0] * lengths[2]), min=-1.0, max=1.0)
    )
    gamma = torch.acos(
        torch.clamp(torch.dot(a_vec, b_vec) / (lengths[0] * lengths[1]), min=-1.0, max=1.0)
    )
    return lengths, torch.stack([alpha, beta, gamma], dim=0)


def _physical_lattice_loss(cell: torch.Tensor, target_cell: torch.Tensor) -> torch.Tensor:
    lengths, angles = _cell_lengths_and_angles(cell)
    target_lengths, target_angles = _cell_lengths_and_angles(target_cell)
    length_loss = (
        torch.sort(torch.log(lengths)).values
        - torch.sort(torch.log(target_lengths.clamp_min(1.0e-8))).values
    ).square().mean()
    angle_loss = (torch.sort(angles).values - torch.sort(target_angles).values).square().mean()
    return length_loss + angle_loss


def _steric_overlap_loss(
    *,
    distances: torch.Tensor,
    min_distance: float,
) -> torch.Tensor:
    if distances.numel() == 0:
        return distances.new_zeros(())
    floor = distances.new_tensor(float(max(min_distance, 0.0)))
    penalties = torch.relu(floor - distances)
    return penalties.square().mean()


def _volume_ratio_loss(
    *,
    cell: torch.Tensor,
    reference_volume: float,
    min_ratio: float,
    max_ratio: float,
) -> torch.Tensor:
    volume = torch.abs(torch.linalg.det(cell)).clamp_min(1.0e-8)
    reference = volume.new_tensor(float(max(reference_volume, 1.0e-8)))
    log_ratio = torch.log(volume / reference)
    penalty = volume.new_zeros(())
    if float(min_ratio) > 0.0:
        lower = volume.new_tensor(float(math.log(max(float(min_ratio), 1.0e-8))))
        penalty = penalty + torch.relu(lower - log_ratio).square()
    if float(max_ratio) > 0.0:
        upper = volume.new_tensor(float(math.log(max(float(max_ratio), 1.0e-8))))
        penalty = penalty + torch.relu(log_ratio - upper).square()
    return penalty


@dataclass(frozen=True)
class ProjectionMetric:
    coord_weight: float = 1.0
    lattice_weight: float = 1.0


@dataclass(frozen=True)
class SSVDProjectionConfig:
    max_steps: int = 16
    svd_damping: float = 1.0e-3
    svd_rank_tol: float = 1.0e-6
    max_delta_norm: float = 0.5
    energy_tol: float = 1.0e-8
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    random_restarts: int = 2
    physical_lattice_weight: float = 1.0
    steric_weight: float = 1.0
    steric_min_distance: float = 0.8
    min_pair_weight: float = 1.0
    min_pair_target: float = 1.0
    volume_weight: float = 1.0
    volume_ratio_min: float = 0.25
    volume_ratio_max: float = 4.0
    freeze_tau: bool = False
    use_fixed_assignment: bool = False
    local_reg_weight: float = 0.0
    local_trust_radius: float | None = None
    local_theta_reference: torch.Tensor | None = None


@dataclass(frozen=True)
class FixedTemplateProjectionResult:
    state: PCSTemplateState
    tau: torch.Tensor
    objective: float
    coord_loss: float
    lattice_k_loss: float
    physical_lattice_loss: float
    steric_loss: float
    volume_loss: float
    min_pair_distance: float
    ssvd_rank: int
    ssvd_condition_number: float
    ssvd_delta_norm: float
    ssvd_steps: int
    ssvd_line_search_accepts: int
    ssvd_line_search_failures: int
    ssvd_clip_count: int
    ssvd_min_sigma: float
    ssvd_max_sigma: float
    objective_initial: float
    frac_coords_chart: torch.Tensor
    atomic_numbers_chart: torch.Tensor
    k: torch.Tensor
    cell: torch.Tensor
    assignment: torch.Tensor


@dataclass(frozen=True)
class _ResidualAux:
    frac_coords: torch.Tensor
    atomic_numbers: torch.Tensor
    matched_target: torch.Tensor
    assignment: torch.Tensor
    coord_delta: torch.Tensor
    coord_loss: torch.Tensor
    lattice_delta: torch.Tensor
    lattice_k_loss: torch.Tensor
    k: torch.Tensor
    cell: torch.Tensor
    min_pair_distance: torch.Tensor
    steric_loss: torch.Tensor
    physical_lattice_loss: torch.Tensor
    volume_loss: torch.Tensor
    free_vars: torch.Tensor
    local_reg_loss: torch.Tensor


def _pack_theta_tau(
    *,
    free_vars: torch.Tensor,
    lattice_free_vars: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            free_vars.reshape(-1),
            lattice_free_vars.reshape(-1),
            tau.reshape(-1),
        ],
        dim=0,
    )


def _unpack_theta_tau(
    *,
    params: torch.Tensor,
    template_state: PCSTemplateState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    free_dim = int(template_state.template.total_free_dims)
    lattice_dim = int(template_state.lattice_free_vars.numel())
    free_vars = params[:free_dim]
    lattice_free_vars = params[free_dim : free_dim + lattice_dim]
    tau = torch.remainder(params[free_dim + lattice_dim : free_dim + lattice_dim + 3], 1.0).reshape(1, 3)
    return free_vars, lattice_free_vars, tau


def _chart_residual(
    *,
    template_state: PCSTemplateState,
    params: torch.Tensor,
    y_f: torch.Tensor,
    y_k: torch.Tensor,
    y_h: torch.Tensor,
    metric: ProjectionMetric,
    config: SSVDProjectionConfig,
) -> tuple[torch.Tensor, _ResidualAux]:
    free_vars, lattice_free_vars, tau = _unpack_theta_tau(params=params, template_state=template_state)
    expansion = expand_wyckoff_template_torch(template=template_state.template, free_vars=free_vars)
    frac_coords = torch.remainder(expansion.frac_coords + tau, 1.0)
    atomic_numbers = expansion.atomic_numbers.to(device=y_h.device, dtype=torch.long)
    fixed_assignment = template_state.fixed_target_assignment
    use_fixed_assignment = False
    if bool(config.use_fixed_assignment) and fixed_assignment is not None:
        fixed_assignment = fixed_assignment.to(device=y_f.device, dtype=torch.long).reshape(-1)
        use_fixed_assignment = (
            int(fixed_assignment.numel()) == int(frac_coords.shape[0])
            and fixed_assignment.numel() > 0
            and int(fixed_assignment.min().detach().item()) >= 0
            and int(fixed_assignment.max().detach().item()) < int(y_f.shape[0])
            and bool(torch.equal(atomic_numbers, y_h[fixed_assignment].to(device=atomic_numbers.device, dtype=torch.long)))
        )
    if use_fixed_assignment:
        assignment = fixed_assignment
    else:
        assignment = _species_assignment_indices(
            source_frac=frac_coords,
            source_atomic_numbers=atomic_numbers,
            target_frac=y_f,
            target_atomic_numbers=y_h,
        ).to(device=y_f.device, dtype=torch.long)
    matched_target = y_f[assignment]
    coord_delta = _wrap_delta(frac_coords - matched_target)
    coord_loss = coord_delta.square().mean() if coord_delta.numel() > 0 else frac_coords.new_zeros(())

    k = free_vars_to_k(lattice_free_vars, template_state.constraint)
    lattice_delta = k - y_k
    lattice_k_loss = lattice_delta.square().mean() if lattice_delta.numel() > 0 else k.new_zeros(())
    cell = k_to_cell_matrix(k)
    target_cell = k_to_cell_matrix(y_k)
    pair_distances = _periodic_pairwise_distances(frac_coords=frac_coords, cell_matrix=cell)
    min_pair_distance = (
        pair_distances.min() if pair_distances.numel() > 0 else frac_coords.new_tensor(float("inf"))
    )
    steric_loss = _steric_overlap_loss(
        distances=pair_distances,
        min_distance=float(config.steric_min_distance),
    )
    physical_lattice_loss = _physical_lattice_loss(cell, target_cell)
    volume_loss = _volume_ratio_loss(
        cell=cell,
        reference_volume=float(torch.abs(torch.linalg.det(target_cell)).detach().item()),
        min_ratio=float(config.volume_ratio_min),
        max_ratio=float(config.volume_ratio_max),
    )
    residual = torch.cat(
        [
            math.sqrt(float(metric.coord_weight)) * coord_delta.reshape(-1),
            math.sqrt(float(metric.lattice_weight)) * lattice_delta.reshape(-1),
        ],
        dim=0,
    )
    theta_delta = _theta_reference_delta(free_vars, config.local_theta_reference)
    local_reg_loss = theta_delta.square().mean() if theta_delta.numel() > 0 else free_vars.new_zeros(())
    aux = _ResidualAux(
        frac_coords=frac_coords,
        atomic_numbers=atomic_numbers,
        matched_target=matched_target,
        assignment=assignment,
        coord_delta=coord_delta,
        coord_loss=coord_loss,
        lattice_delta=lattice_delta,
        lattice_k_loss=lattice_k_loss,
        k=k,
        cell=cell,
        min_pair_distance=min_pair_distance,
        steric_loss=steric_loss,
        physical_lattice_loss=physical_lattice_loss,
        volume_loss=volume_loss,
        free_vars=free_vars,
        local_reg_loss=local_reg_loss,
    )
    return residual, aux


def _projection_energy(
    *,
    residual: torch.Tensor,
    aux: _ResidualAux,
    config: SSVDProjectionConfig,
) -> torch.Tensor:
    base = residual.square().mean() if residual.numel() > 0 else residual.new_zeros(())
    min_pair_penalty = torch.relu(
        aux.min_pair_distance.new_tensor(float(config.min_pair_target)) - aux.min_pair_distance
    ).square()
    return (
        base
        + float(config.physical_lattice_weight) * aux.physical_lattice_loss
        + float(config.steric_weight) * aux.steric_loss
        + float(config.min_pair_weight) * min_pair_penalty
        + float(config.volume_weight) * aux.volume_loss
        + float(config.local_reg_weight) * aux.local_reg_loss
    )


def _theta_reference_delta(
    free_vars: torch.Tensor,
    reference_free_vars: torch.Tensor | None,
) -> torch.Tensor:
    if reference_free_vars is None:
        return free_vars.new_zeros(free_vars.shape)
    ref = reference_free_vars.to(device=free_vars.device, dtype=free_vars.dtype).reshape(-1)
    return _wrap_delta(free_vars.reshape(-1) - ref)


def _clone_params_with_seed(
    *,
    template_state: PCSTemplateState,
    base_tau: torch.Tensor,
    restart_idx: int,
) -> torch.Tensor:
    if restart_idx == 0:
        free_vars = template_state.free_vars.detach().clone()
    else:
        free_vars = sample_random_free_vars(
            template_state.template,
            device=template_state.free_vars.device,
            dtype=template_state.free_vars.dtype,
        ).reshape(-1)
    lattice_free_vars = template_state.lattice_free_vars.detach().clone().reshape(-1)
    return _pack_theta_tau(
        free_vars=free_vars,
        lattice_free_vars=lattice_free_vars,
        tau=base_tau.detach().clone(),
    )


def fixed_template_ssvd_project(
    *,
    template_state: PCSTemplateState,
    y_f: torch.Tensor,
    y_k: torch.Tensor,
    y_h: torch.Tensor,
    tau0: torch.Tensor,
    metric: ProjectionMetric,
    config: SSVDProjectionConfig,
) -> FixedTemplateProjectionResult:
    best_result: FixedTemplateProjectionResult | None = None

    for restart_idx in range(max(1, int(config.random_restarts))):
        params = _clone_params_with_seed(
            template_state=template_state,
            base_tau=tau0,
            restart_idx=restart_idx,
        ).to(device=y_f.device, dtype=y_f.dtype)
        params = params.detach().clone().requires_grad_(True)
        last_rank = 0
        last_cond = float("inf")
        last_delta_norm = 0.0
        last_min_sigma = float("nan")
        last_max_sigma = float("nan")
        step_count = 0
        line_search_accepts = 0
        line_search_failures = 0
        clip_count = 0
        objective_initial = float("inf")

        for _ in range(int(config.max_steps)):
            residual, aux = _chart_residual(
                template_state=template_state,
                params=params,
                y_f=y_f,
                y_k=y_k,
                y_h=y_h,
                metric=metric,
                config=config,
            )
            base_energy = _projection_energy(residual=residual, aux=aux, config=config)
            if not torch.isfinite(base_energy):
                break
            if not math.isfinite(objective_initial):
                objective_initial = float(base_energy.detach().item())

            jacobian = torch.autograd.functional.jacobian(
                lambda p: _chart_residual(
                    template_state=template_state,
                    params=p,
                    y_f=y_f,
                    y_k=y_k,
                    y_h=y_h,
                    metric=metric,
                    config=config,
                )[0].reshape(-1),
                params,
                vectorize=True,
            ).reshape(residual.numel(), params.numel())

            u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
            keep = singular_values > float(config.svd_rank_tol)
            if int(torch.count_nonzero(keep).item()) == 0:
                break
            u = u[:, keep]
            singular_values = singular_values[keep]
            vh = vh[keep, :]
            last_rank = int(singular_values.numel())
            last_cond = float((singular_values.max() / singular_values.min().clamp_min(1.0e-12)).detach().item())
            last_min_sigma = float(singular_values.min().detach().item())
            last_max_sigma = float(singular_values.max().detach().item())
            filt = singular_values / (singular_values.square() + float(config.svd_damping))
            delta = -(vh.transpose(0, 1) @ (filt * (u.transpose(0, 1) @ residual.reshape(-1))))
            if bool(config.freeze_tau):
                free_dim = int(template_state.template.total_free_dims)
                lattice_dim = int(template_state.lattice_free_vars.numel())
                delta = delta.clone()
                delta[free_dim + lattice_dim : free_dim + lattice_dim + 3] = 0.0
            delta_norm = torch.linalg.norm(delta)
            if float(delta_norm.detach().item()) > float(config.max_delta_norm):
                delta = delta * (float(config.max_delta_norm) / float(delta_norm.detach().item()))
                delta_norm = torch.linalg.norm(delta)
                clip_count += 1
            last_delta_norm = float(delta_norm.detach().item())

            accepted = False
            for alpha in config.line_search_alphas:
                proposal = params + float(alpha) * delta
                free_vars_new, lattice_free_vars_new, tau_new = _unpack_theta_tau(
                    params=proposal,
                    template_state=template_state,
                )
                trust_radius = config.local_trust_radius
                if trust_radius is not None and config.local_theta_reference is not None:
                    theta_step = _theta_reference_delta(free_vars_new, config.local_theta_reference)
                    if theta_step.numel() > 0 and float(theta_step.abs().max().detach().item()) > float(trust_radius):
                        continue
                proposal = _pack_theta_tau(
                    free_vars=torch.remainder(free_vars_new, 1.0),
                    lattice_free_vars=lattice_free_vars_new,
                    tau=tau_new,
                ).to(device=params.device, dtype=params.dtype)
                residual_new, aux_new = _chart_residual(
                    template_state=template_state,
                    params=proposal,
                    y_f=y_f,
                    y_k=y_k,
                    y_h=y_h,
                    metric=metric,
                    config=config,
                )
                new_energy = _projection_energy(residual=residual_new, aux=aux_new, config=config)
                if torch.isfinite(new_energy) and float(new_energy.detach().item()) <= float(base_energy.detach().item()):
                    params = proposal.detach().clone().requires_grad_(True)
                    accepted = True
                    line_search_accepts += 1
                    step_count += 1
                    if abs(float(base_energy.detach().item()) - float(new_energy.detach().item())) < float(config.energy_tol):
                        residual = residual_new
                        aux = aux_new
                        base_energy = new_energy
                        break
                    break
            if not accepted:
                line_search_failures += 1
                break

        with torch.no_grad():
            residual, aux = _chart_residual(
                template_state=template_state,
                params=params.detach(),
                y_f=y_f,
                y_k=y_k,
                y_h=y_h,
                metric=metric,
                config=config,
            )
            energy = _projection_energy(residual=residual, aux=aux, config=config)
            if not torch.isfinite(energy):
                continue
            if not math.isfinite(objective_initial):
                objective_initial = float(energy.detach().item())
            free_vars, lattice_free_vars, tau = _unpack_theta_tau(
                params=params.detach(),
                template_state=template_state,
            )
            state = replace(
                template_state,
                free_vars=torch.remainder(free_vars, 1.0).detach().clone(),
                lattice_free_vars=lattice_free_vars.detach().clone(),
                objective=float(energy.detach().item()),
                ranking_objective=float(energy.detach().item()),
            )
            result = FixedTemplateProjectionResult(
                state=state,
                tau=tau.detach().clone(),
                objective=float(energy.detach().item()),
                coord_loss=float(aux.coord_loss.detach().item()),
                lattice_k_loss=float(aux.lattice_k_loss.detach().item()),
                physical_lattice_loss=float(aux.physical_lattice_loss.detach().item()),
                steric_loss=float(aux.steric_loss.detach().item()),
                volume_loss=float(aux.volume_loss.detach().item()),
                min_pair_distance=float(aux.min_pair_distance.detach().item()),
                ssvd_rank=int(last_rank),
                ssvd_condition_number=float(last_cond),
                ssvd_delta_norm=float(last_delta_norm),
                ssvd_steps=int(step_count),
                ssvd_line_search_accepts=int(line_search_accepts),
                ssvd_line_search_failures=int(line_search_failures),
                ssvd_clip_count=int(clip_count),
                ssvd_min_sigma=float(last_min_sigma),
                ssvd_max_sigma=float(last_max_sigma),
                objective_initial=float(objective_initial),
                frac_coords_chart=aux.frac_coords.detach().clone(),
                atomic_numbers_chart=aux.atomic_numbers.detach().clone(),
                k=aux.k.detach().clone(),
                cell=aux.cell.detach().clone(),
                assignment=aux.assignment.detach().clone(),
            )
            if best_result is None or float(result.objective) < float(best_result.objective):
                best_result = result

    if best_result is None:
        raise RuntimeError("fixed_template_ssvd_project_failed")
    return best_result
