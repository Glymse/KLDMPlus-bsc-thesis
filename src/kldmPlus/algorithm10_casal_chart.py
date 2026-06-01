from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import numpy as np
import torch

from kldmPlus.data.transform import (
    cell_lengths_and_angles,
    lattice_feature_components,
)
from kldmPlus.fixed_template_ssvd_project import (
    FixedTemplateProjectionResult,
    ProjectionMetric,
    SSVDProjectionConfig,
    fixed_template_ssvd_project,
)
from kldmPlus.symmetry.k_basis import cell_to_k, k_to_cell_matrix
from kldmPlus.symmetry.frame_bridge import standardize_structure
from kldmPlus.symmetry.pcs_projection import (
    _build_vanilla_structure,
    _centering_translations,
    _expand_target_by_translations,
    _periodic_pairwise_distances,
    _raw_target_in_requested_conventional_frame,
    _requested_centering_symbol,
    _species_assignment_indices,
    _species_orbit_mismatch_count,
    _structure_species_orbit_signature_with_source,
    initialize_constrained_template_states,
    materialize_pcs_state,
    validate_requested_space_group,
)
from kldmPlus.symmetry.wyckoff_templates import flatten_site_signature, requested_conventional_atomic_numbers
from kldmPlus.utils.time import iter_sampling_times


def _lengths_angles_to_cell_matrix(*, lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1.0e-8)

    row0 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    row1 = torch.stack([b * cos_gamma, b * sin_gamma, torch.zeros_like(b)], dim=-1)
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sq = (c.square() - cx.square() - cy.square()).clamp_min(1.0e-12)
    row2 = torch.stack([cx, cy, torch.sqrt(cz_sq)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _decode_lattice_matrix(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    if lattice_transform is not None and hasattr(lattice_transform, "invert_to_matrix"):
        matrix = lattice_transform.invert_to_matrix(l=l, num_atoms=num_atoms)
        return matrix.reshape(*l.shape[:-1], 3, 3)
    if lattice_transform is not None:
        lengths, angles = lattice_transform.invert_to_lengths_angles(l=l, num_atoms=num_atoms)
    else:
        lengths = torch.exp(l[..., :3])
        angles = torch.atan(l[..., 3:]) + torch.pi / 2.0
    return _lengths_angles_to_cell_matrix(lengths=lengths, angles=angles)


def _encode_lattice_matrix(
    *,
    cell_matrix: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    log_lengths, angle_features = lattice_feature_components(
        cell_matrix,
        eps=float(getattr(lattice_transform, "eps", 1.0e-8)),
    )
    if (
        lattice_transform is not None
        and bool(getattr(lattice_transform, "standardize", False))
        and getattr(lattice_transform, "lengths_loc_scale", None) is not None
    ):
        log_lengths, angle_features = lattice_transform._encode_x0_parts(  # noqa: SLF001
            log_lengths=log_lengths,
            angle_features=angle_features,
            num_atoms=int(num_atoms),
        )
        return torch.cat([log_lengths, angle_features], dim=0).reshape(-1)
    features = torch.cat([log_lengths, angle_features], dim=0).reshape(-1)
    if lattice_transform is not None and hasattr(lattice_transform, "standardize_value"):
        features = lattice_transform.standardize_value(features)
    return features


def _invalid_sample_like(
    *,
    pos_ref: torch.Tensor,
    l_ref: torch.Tensor,
    h_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full_like(pos_ref, float("nan")),
        torch.full_like(l_ref, float("nan")),
        torch.full_like(h_ref, -1),
    )


@dataclass(frozen=True)
class Algorithm10Config:
    max_templates: int = 256
    template_eval_limit: int = 256
    template_nmax: int = 20000
    quick_templates: bool = False
    top_k_templates: int = 256
    template_prior_mode: str = "cspml"
    template_prior_weight: float = 1.0

    casal_rho_start: float = 0.5
    casal_rho_end: float = 2.0
    casal_tau_scale: float = 0.25
    casal_mu_eta: float = 0.25
    casal_mu_clip: float = 0.5
    casal_noise_scale: float = 0.0
    casal_coupling_mode: str = "velocity"
    casal_velocity_alpha: float = 1.0e-4
    casal_velocity_sign: float = 1.0
    casal_velocity_controller_mode: str = "impulse"
    casal_force_rho_scale: float = 1.0
    casal_velocity_mean_free: bool = True
    casal_beta_f: float = 0.01
    casal_beta_z: float = 0.025
    casal_velocity_damping: float = 0.0
    casal_velocity_damping_mode: str = "none"
    casal_damping_metric: str = "fractional"
    casal_max_delta_v: float = 0.25
    casal_velocity_placement: str = "corrector"
    casal_coord_weight: float = 1.0
    casal_lattice_weight: float = 0.0
    casal_projection_lattice_weight: float = 0.0
    casal_dual_rule: str = "tau_over_rho"
    casal_dual_enabled: bool = False
    casal_dual_update_policy: str = "projection_success_only"
    casal_residual_mode: str = "wrap_plus_mu"
    casal_damping_residual: str = "geom"
    casal_gauge_mode: str = "full"
    casal_inner_projection_mode: str = "full"
    casal_final_projection_mode: str = "same"
    casal_metric_accept_tol: float = 1.0e-8
    projection_start_mode: str = "fraction"
    casal_require_feasible_initial_projection: bool = True
    casal_require_feasible_projection: bool = True
    projection_interval: int = 10
    projection_start_fraction: float = 0.9
    projection_start_step: int = 0
    projection_min_volume: float = 1.0
    projection_min_lattice_length: float = 0.5
    oracle_wyckoff_debug: bool = False

    projection_coord_weight: float = 1.0
    projection_lattice_weight: float = 1.0
    projection_physical_lattice_weight: float = 2.0
    projection_steric_weight: float = 1.0
    projection_min_pair_weight: float = 1.0
    projection_min_pair_target: float = 1.0
    projection_volume_weight: float = 1.0

    origin_shift_mode: str = "axis_grid"
    origin_shift_values: tuple[float, ...] = (0.0, 0.125, 0.875, 0.25, 0.75, 0.5)
    origin_shift_candidates: tuple[tuple[float, float, float], ...] = ()

    ssvd_max_steps: int = 16
    ssvd_damping: float = 1.0e-3
    ssvd_rank_tol: float = 1.0e-6
    ssvd_max_delta_norm: float = 0.5
    ssvd_energy_tol: float = 1.0e-8
    ssvd_random_restarts: int = 2
    ssvd_line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)

    optimization_steps: int = 120
    learning_rate: float = 5.0e-2
    template_init_restarts: int = 4
    symprec: float = 1.0e-2
    angle_tolerance: float = 5.0
    hard_min_distance: float = 0.8
    hard_volume_ratio_min: float = 0.25
    hard_volume_ratio_max: float = 4.0
    return_best_even_if_invalid: bool = False
    debug_projection_failures: bool = True
    debug_projection_examples: int = 1
    debug: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "Algorithm10Config":
        if not payload:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = sorted(key for key in payload if key not in fields)
        if unknown:
            print(f"algorithm10_config_ignore unknown_keys={unknown}", flush=True)
        values = {key: payload[key] for key in fields if key in payload}
        if "origin_shift_values" in values:
            values["origin_shift_values"] = tuple(float(v) for v in values["origin_shift_values"])
        if "origin_shift_candidates" in values:
            values["origin_shift_candidates"] = tuple(
                tuple(float(component) for component in shift)
                for shift in values["origin_shift_candidates"]
            )
        if "ssvd_line_search_alphas" in values:
            values["ssvd_line_search_alphas"] = tuple(float(v) for v in values["ssvd_line_search_alphas"])
        return cls(**values)

    def projection_metric(self) -> ProjectionMetric:
        return ProjectionMetric(
            coord_weight=float(self.projection_coord_weight),
            lattice_weight=float(self.projection_lattice_weight),
        )

    def ssvd_projection_config(self) -> SSVDProjectionConfig:
        return SSVDProjectionConfig(
            max_steps=int(self.ssvd_max_steps),
            svd_damping=float(self.ssvd_damping),
            svd_rank_tol=float(self.ssvd_rank_tol),
            max_delta_norm=float(self.ssvd_max_delta_norm),
            energy_tol=float(self.ssvd_energy_tol),
            line_search_alphas=tuple(float(v) for v in self.ssvd_line_search_alphas),
            random_restarts=int(self.ssvd_random_restarts),
            physical_lattice_weight=float(self.projection_physical_lattice_weight),
            steric_weight=float(self.projection_steric_weight),
            steric_min_distance=float(self.hard_min_distance),
            min_pair_weight=float(self.projection_min_pair_weight),
            min_pair_target=float(self.projection_min_pair_target),
            volume_weight=float(self.projection_volume_weight),
            volume_ratio_min=float(self.hard_volume_ratio_min),
            volume_ratio_max=float(self.hard_volume_ratio_max),
        )


@dataclass(frozen=True)
class _CasalProjection:
    state: Any
    pos: torch.Tensor
    l: torch.Tensor
    k: torch.Tensor
    h: torch.Tensor
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
    template_labels: str
    prior_score: int
    sg_detected: int | None
    requested_sg_match: bool
    composition_match: bool
    frac_rmse: float
    mean_dist: float
    max_dist: float
    mean_shift: tuple[float, float, float]
    shift_spread: float
    validation_failure: str | None = None
    volume_ratio: float = float("nan")
    volume: float = float("nan")
    target_volume: float = float("nan")
    hard_constraint_satisfied: bool = False
    hard_constraint_violation: float = float("inf")
    projection_elapsed_s: float = float("nan")
    projection_state_count: int = 0
    projection_tau_count: int = 0
    projection_attempt_count: int = 0
    projection_ssvd_fail_count: int = 0
    projection_materialize_fail_count: int = 0


@dataclass
class _CasalGraphState:
    z_pos: torch.Tensor
    z_l: torch.Tensor
    z_k: torch.Tensor
    z_h: torch.Tensor
    mu_f: torch.Tensor
    mu_k: torch.Tensor
    projection: _CasalProjection


def _wrap_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(delta + 0.5, 1.0) - 0.5


def _casal_frac_residual(
    *,
    f: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    residual_mode = str(mode).strip().lower()
    if residual_mode in {"wrap_inside", "inside"}:
        return _wrap_delta(f - z + mu)
    if residual_mode in {"wrap_plus_mu", "plus_mu", "default"}:
        return _wrap_delta(f - z) + mu
    raise ValueError(f"Unsupported casal_residual_mode={mode!r}")


def _clip_norm(tensor: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, bool]:
    bound = float(max_norm)
    if not math.isfinite(bound) or bound <= 0.0:
        return tensor, False
    norm = float(torch.linalg.norm(tensor.reshape(-1)).detach().item())
    if norm <= bound:
        return tensor, False
    scale = bound / max(norm, 1.0e-12)
    return tensor * scale, True


def _velocity_sigma_like(
    *,
    sampling_tdm: Any,
    t_nodes: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    sigma = sampling_tdm.vel_scale * sampling_tdm.gaussian_velocity_sigma(t_nodes)
    return sampling_tdm.match_dims(sigma, ref)


def _project_velocity_parallel(
    *,
    v: torch.Tensor,
    r: torch.Tensor,
    metric: str,
    cell: torch.Tensor | None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if metric == "fractional" or cell is None:
        denom = r.square().sum().clamp_min(eps)
        coeff = (v * r).sum() / denom
        return coeff * r
    if metric == "lattice_metric":
        gram = cell @ cell.T
        grad_r = r @ gram
        denom = (r * grad_r).sum().clamp_min(eps)
        coeff = (v * grad_r).sum() / denom
        return coeff * r
    raise ValueError(f"Unsupported casal_damping_metric={metric!r}")


def _damp_velocity(
    *,
    v: torch.Tensor,
    r_geom: torch.Tensor,
    gamma: float,
    mode: str,
    metric: str,
    cell: torch.Tensor | None,
) -> torch.Tensor:
    gamma_value = float(gamma)
    damping_mode = str(mode).strip().lower()
    if gamma_value == 0.0 or damping_mode in {"none", ""}:
        return torch.zeros_like(v)
    if damping_mode == "global":
        return -gamma_value * v
    if damping_mode in {"residual_parallel", "parallel"}:
        return -gamma_value * _project_velocity_parallel(
            v=v,
            r=r_geom,
            metric=str(metric).strip().lower(),
            cell=cell,
        )
    raise ValueError(f"Unsupported casal_velocity_damping_mode={mode!r}")


def _rho_schedule(config: Algorithm10Config, step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(config.casal_rho_start)
    alpha = float(step_idx) / float(max(total_steps - 1, 1))
    return float(config.casal_rho_start) + alpha * (
        float(config.casal_rho_end) - float(config.casal_rho_start)
    )


def _projection_config_for_mode(
    config: Algorithm10Config,
    *,
    mode: str,
) -> Algorithm10Config:
    projection_mode = str(mode).strip().lower()
    if projection_mode in {"same", "full", ""}:
        return config
    if projection_mode in {"metric_only", "metric", "inner_metric"}:
        return replace(
            config,
            template_prior_weight=0.0,
            projection_physical_lattice_weight=0.0,
            projection_steric_weight=0.0,
            projection_min_pair_weight=0.0,
            projection_volume_weight=0.0,
            hard_min_distance=0.0,
            hard_volume_ratio_min=0.0,
            hard_volume_ratio_max=0.0,
            return_best_even_if_invalid=True,
        )
    raise ValueError(f"Unsupported casal_inner/final_projection_mode={mode!r}")


def _casal_metric_value(
    *,
    x_pos: torch.Tensor,
    x_k: torch.Tensor,
    z_pos: torch.Tensor,
    z_k: torch.Tensor,
    config: Algorithm10Config,
) -> float:
    coord_weight = float(getattr(config, "casal_coord_weight", 1.0))
    lattice_weight = float(getattr(config, "casal_lattice_weight", 0.0))
    coord_term = coord_weight * float(_wrap_delta(x_pos - z_pos).square().sum().detach().item())
    lattice_term = 0.0
    if lattice_weight != 0.0:
        lattice_term = lattice_weight * float((x_k - z_k).square().sum().detach().item())
    return coord_term + lattice_term


def _mean_free_velocity(delta_v: torch.Tensor) -> tuple[torch.Tensor, float]:
    if delta_v.ndim < 2 or int(delta_v.shape[0]) <= 1:
        mean_norm = float(torch.linalg.norm(delta_v.mean(dim=0).reshape(-1)).detach().item()) if delta_v.ndim >= 1 else 0.0
        return delta_v, mean_norm
    mean_shift = delta_v.mean(dim=0, keepdim=True)
    mean_norm = float(torch.linalg.norm(mean_shift.reshape(-1)).detach().item())
    return delta_v - mean_shift, mean_norm


def _velocity_correction(
    *,
    r_force: torch.Tensor,
    dt_abs: float,
    rho: float,
    config: Algorithm10Config,
) -> torch.Tensor:
    controller_mode = str(getattr(config, "casal_velocity_controller_mode", "impulse")).strip().lower()
    scale = float(getattr(config, "casal_velocity_alpha", 0.0)) * float(getattr(config, "casal_coord_weight", 1.0))
    sign = float(getattr(config, "casal_velocity_sign", 1.0))
    if controller_mode in {"impulse", "pd", "velocity_pd", "practical"}:
        return sign * (scale / max(float(dt_abs), 1.0e-12)) * r_force
    if controller_mode in {"force", "tdm_force", "principled"}:
        rho_scale = float(getattr(config, "casal_force_rho_scale", 1.0))
        return sign * (max(float(dt_abs), 1.0e-12) * float(rho) * rho_scale * float(getattr(config, "casal_coord_weight", 1.0))) * r_force
    raise ValueError(f"Unsupported casal_velocity_controller_mode={controller_mode!r}")


def _l_to_k(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    cell = _decode_lattice_matrix(
        l=l.reshape(-1),
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    )
    return cell_to_k(cell.reshape(3, 3), eps=1.0e-8).reshape(-1)


def _k_to_l(
    *,
    k: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    cell = k_to_cell_matrix(k.reshape(-1)).reshape(3, 3)
    return _encode_lattice_matrix(
        cell_matrix=cell,
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    ).reshape(-1)


def _cell_debug_summary(cell: torch.Tensor) -> tuple[float, list[float]]:
    volume = float(torch.det(cell).abs().detach().item())
    lengths = torch.linalg.norm(cell, dim=-1).detach().reshape(-1).tolist()
    return volume, [round(float(value), 4) for value in lengths]


def _projection_lattice_sane(
    *,
    l: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any,
    config: Algorithm10Config,
) -> tuple[bool, str, float, list[float]]:
    try:
        cell = _decode_lattice_matrix(
            l=l.reshape(-1),
            num_atoms=int(num_atoms),
            lattice_transform=lattice_transform,
        )
        volume, lengths = _cell_debug_summary(cell)
        min_length = min(lengths) if lengths else float("inf")
        if not math.isfinite(volume) or volume < float(config.projection_min_volume):
            return False, "volume", volume, lengths
        if not math.isfinite(min_length) or min_length < float(config.projection_min_lattice_length):
            return False, "length", volume, lengths
        return True, "ok", volume, lengths
    except Exception as exc:
        return False, f"decode:{type(exc).__name__}", float("nan"), []


def _fractional_debug_summary(
    *,
    pos: torch.Tensor,
    h: torch.Tensor,
    target_pos: torch.Tensor,
    target_h: torch.Tensor,
) -> tuple[float, float, float, tuple[float, float, float], float]:
    assignment = _species_assignment_indices(
        source_frac=pos,
        source_atomic_numbers=h.to(device=pos.device, dtype=torch.long),
        target_frac=target_pos,
        target_atomic_numbers=target_h.to(device=pos.device, dtype=torch.long),
    ).to(device=target_pos.device, dtype=torch.long)
    matched_target = target_pos[assignment]
    delta = _wrap_delta(pos - matched_target)
    distances = torch.linalg.norm(delta, dim=-1)
    mean_shift = delta.mean(dim=0) if delta.numel() > 0 else delta.new_zeros((3,))
    centered = _wrap_delta(delta - mean_shift.reshape(1, 3))
    shift_spread = float(torch.linalg.norm(centered, dim=-1).mean().detach().item()) if centered.numel() > 0 else 0.0
    frac_rmse = float(torch.sqrt(delta.square().mean()).detach().item()) if delta.numel() > 0 else 0.0
    mean_dist = float(distances.mean().detach().item()) if distances.numel() > 0 else 0.0
    max_dist = float(distances.max().detach().item()) if distances.numel() > 0 else 0.0
    shift_tuple = tuple(float(v) for v in mean_shift.detach().cpu().reshape(-1).tolist())
    return frac_rmse, mean_dist, max_dist, shift_tuple, shift_spread


def _record_projection_reject(
    *,
    reject_reasons: dict[str, int],
    reject_examples: list[str],
    stage: str,
    template_labels: str,
    tau: torch.Tensor,
    exc: Exception,
    max_examples: int,
) -> None:
    exc_type = type(exc).__name__
    key = f"{stage}:{exc_type}"
    reject_reasons[key] = reject_reasons.get(key, 0) + 1
    if len(reject_examples) < max(0, int(max_examples)):
        tau_text = [round(float(value), 4) for value in tau.reshape(-1).detach().cpu().tolist()]
        message = str(exc).replace("\n", " | ")
        if len(message) > 400:
            message = message[:397] + "..."
        reject_examples.append(
            f"stage={stage} template={template_labels} tau={tau_text} type={exc_type} message={message}"
        )


def _hard_constraint_violation(
    *,
    composition_match: bool,
    requested_sg_match: bool,
    min_pair_distance: float,
    volume_ratio: float,
    config: Algorithm10Config,
) -> tuple[bool, float]:
    """Rank approximate chart projections by hard feasibility first.

    CASAL updates the split variable with `z = P_C(...)`. In the crystal chart
    setting, `C` includes exact composition/space-group feasibility plus the
    hard physical guards we require before accepting a projected structure.
    Finite top-k chart search may fail to find a fully feasible point, so we
    rank by hard-constraint violation before using the soft projection distance.
    """
    violation = 0.0
    if not bool(composition_match):
        violation += 1.0e6
    if not bool(requested_sg_match):
        violation += 1.0e5

    threshold = float(config.hard_min_distance)
    min_pair = float(min_pair_distance)
    if not math.isfinite(min_pair):
        violation += 1.0e4
    elif min_pair < threshold:
        violation += (threshold - min_pair) / max(threshold, 1.0e-8)

    ratio = float(volume_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        violation += 1.0e4
    else:
        lower = float(config.hard_volume_ratio_min)
        upper = float(config.hard_volume_ratio_max)
        if lower > 0.0 and ratio < lower:
            violation += math.log(lower / max(ratio, 1.0e-12)) ** 2
        if upper > 0.0 and ratio > upper:
            violation += math.log(max(ratio, 1.0e-12) / upper) ** 2

    return bool(violation <= 1.0e-12), float(violation)


def _projection_selection_key(candidate: _CasalProjection) -> tuple[float, float, float]:
    return (
        0.0 if bool(candidate.hard_constraint_satisfied) else 1.0,
        float(candidate.hard_constraint_violation),
        float(candidate.objective),
    )


def _template_labels(state: Any) -> str:
    return ",".join(
        f"{int(site.atomic_number)}@{site.label}"
        for site in state.template.site_templates
    )


def _origin_shift_candidates(
    *,
    config: Algorithm10Config,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if config.origin_shift_candidates:
        values = config.origin_shift_candidates
    elif str(config.origin_shift_mode).lower() == "diagonal":
        values = tuple((v, v, v) for v in config.origin_shift_values)
    else:
        values = tuple(product(config.origin_shift_values, repeat=3))
    return [torch.tensor(shift, device=device, dtype=dtype).reshape(1, 3) for shift in values]


def _align_projection_to_target_order(
    *,
    pos: torch.Tensor,
    h: torch.Tensor,
    target_pos: torch.Tensor,
    target_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignment = _species_assignment_indices(
        source_frac=pos,
        source_atomic_numbers=h.to(device=pos.device, dtype=torch.long),
        target_frac=target_pos,
        target_atomic_numbers=target_h.to(device=pos.device, dtype=torch.long),
    ).to(device=pos.device, dtype=torch.long)
    inverse = torch.empty_like(assignment)
    inverse[assignment] = torch.arange(int(assignment.numel()), device=assignment.device, dtype=torch.long)
    return pos[inverse], h[inverse]


def _oracle_wyckoff_structures(
    *,
    batch: Any,
    ptr: list[int],
    lattice_transform: Any,
    device: torch.device,
    dtype: torch.dtype,
    config: Algorithm10Config,
) -> list[Any | None]:
    if not bool(config.oracle_wyckoff_debug):
        return [None for _ in range(int(batch.num_graphs))]
    structures: list[Any | None] = []
    for start, end in zip(ptr[:-1], ptr[1:]):
        try:
            target_pos = batch.pos[start:end].to(device=device, dtype=dtype)
            target_h = batch.atomic_numbers[start:end].to(device=device, dtype=torch.long)
            graph_idx = len(structures)
            target_l = batch.l[graph_idx].to(device=device, dtype=dtype).reshape(-1)
            target_cell = _decode_lattice_matrix(
                l=target_l,
                num_atoms=int(end - start),
                lattice_transform=lattice_transform,
            ).to(device=device, dtype=dtype)
            structures.append(
                _build_vanilla_structure(
                    frac_coords=target_pos,
                    atomic_numbers=target_h,
                    cell_matrix=target_cell,
                )
            )
        except Exception:
            structures.append(None)
    return structures


def _build_chart_target(
    *,
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell_matrix: torch.Tensor,
    requested_sg: int,
    symprec: float = 1.0e-2,
    angle_tolerance: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centering_symbol = _requested_centering_symbol(int(requested_sg))
    raw_requested_frac, raw_requested_cell = _raw_target_in_requested_conventional_frame(
        frac_coords=torch.remainder(frac_coords, 1.0),
        cell_matrix=cell_matrix,
        centering_symbol=centering_symbol,
    )
    translations = _centering_translations(
        centering_symbol,
        device=frac_coords.device,
        dtype=frac_coords.dtype,
    )
    use_centering_expansion = int(
        requested_conventional_atomic_numbers(
            atomic_numbers,
            space_group_number=int(requested_sg),
        ).shape[0]
    ) != int(atomic_numbers.shape[0])
    if not use_centering_expansion:
        try:
            target_structure = _build_vanilla_structure(
                frac_coords=torch.remainder(frac_coords, 1.0),
                atomic_numbers=atomic_numbers.to(dtype=torch.long),
                cell_matrix=cell_matrix,
            )
            analyzer, standardized = standardize_structure(
                target_structure,
                standardization="conventional",
                symprec=float(symprec),
                angle_tolerance=float(angle_tolerance),
            )
            if int(analyzer.get_space_group_number()) == int(requested_sg):
                standardized_frac = torch.as_tensor(
                    np.array(standardized.frac_coords, dtype=float, copy=True),
                    device=frac_coords.device,
                    dtype=frac_coords.dtype,
                )
                standardized_atomic_numbers = torch.as_tensor(
                    np.array(standardized.atomic_numbers, dtype=int, copy=True),
                    device=frac_coords.device,
                    dtype=torch.long,
                )
                standardized_cell = torch.as_tensor(
                    np.array(standardized.lattice.matrix, dtype=float, copy=True),
                    device=frac_coords.device,
                    dtype=frac_coords.dtype,
                )
                return (
                    torch.remainder(standardized_frac, 1.0),
                    standardized_atomic_numbers,
                    standardized_cell,
                    cell_to_k(standardized_cell, eps=1.0e-8),
                )
        except Exception:
            pass
    chart_frac, chart_atomic_numbers = _expand_target_by_translations(
        raw_requested_frac,
        atomic_numbers.to(dtype=torch.long),
        translations if use_centering_expansion else None,
    )
    chart_cell = raw_requested_cell
    chart_k = cell_to_k(chart_cell, eps=1.0e-8)
    return chart_frac, chart_atomic_numbers, chart_cell, chart_k


def _rank_projection_states(
    *,
    states: list[Any],
    oracle_reference_structure: Any | None,
    config: Algorithm10Config,
) -> list[Any]:
    if oracle_reference_structure is None:
        return sorted(
            states,
            key=lambda state: (
                -int(state.prior_score),
                int(state.template.total_free_dims),
                int(state.template.total_sites),
                int(state.template.total_atoms),
                tuple(flatten_site_signature(state.template)),
            ),
        )
    try:
        target_signature, _source = _structure_species_orbit_signature_with_source(
            structure=oracle_reference_structure,
            symprec=float(config.symprec),
            angle_tolerance=float(config.angle_tolerance),
        )
    except Exception:
        target_signature = ()
    return sorted(
        states,
        key=lambda state: (
            int(
                _species_orbit_mismatch_count(
                    template_signature=state.template_species_orbit_signature or (),
                    target_signature=target_signature,
                )
            ),
            -int(state.prior_score),
            int(state.template.total_free_dims),
            int(state.template.total_sites),
            int(state.template.total_atoms),
            tuple(flatten_site_signature(state.template)),
        ),
    )


def _materialize_projection(
    *,
    graph_idx: int,
    projection: FixedTemplateProjectionResult,
    target_pos: torch.Tensor,
    target_h: torch.Tensor,
    target_cell: torch.Tensor,
    requested_sg: int,
    lattice_transform: Any,
    config: Algorithm10Config,
) -> _CasalProjection:
    materialized = materialize_pcs_state(
        state=projection.state,
        vanilla_reference_structure=None,
    )
    structure = materialized.projected_structure_vanilla
    pos_out = torch.as_tensor(
        np.array(structure.frac_coords, dtype=float, copy=True),
        device=target_pos.device,
        dtype=target_pos.dtype,
    )
    h_out = torch.as_tensor(
        np.array(structure.atomic_numbers, dtype=int, copy=True),
        device=target_h.device,
        dtype=torch.long,
    )
    cell_out = torch.as_tensor(
        np.array(structure.lattice.matrix, dtype=float, copy=True),
        device=target_pos.device,
        dtype=target_pos.dtype,
    )
    pos_out = torch.remainder(pos_out, 1.0)
    pos_out, h_out = _align_projection_to_target_order(
        pos=pos_out,
        h=h_out,
        target_pos=target_pos,
        target_h=target_h,
    )
    l_out = _encode_lattice_matrix(
        cell_matrix=cell_out,
        num_atoms=int(pos_out.shape[0]),
        lattice_transform=lattice_transform,
    ).to(device=target_pos.device, dtype=target_pos.dtype)
    k_out = cell_to_k(cell_out, eps=1.0e-8).reshape(-1)
    validation_error: str | None = None
    try:
        validation = validate_requested_space_group(
            structure=structure,
            requested_space_group=int(requested_sg),
            expected_atomic_numbers=target_h,
            symprec=float(config.symprec),
            angle_tolerance=float(config.angle_tolerance),
        )
        detected_space_group = validation.detected_space_group
        requested_space_group_match = bool(validation.requested_space_group_match)
        composition_match = bool(validation.composition_match)
    except Exception as exc:
        if not bool(config.return_best_even_if_invalid):
            raise
        validation_error = f"{type(exc).__name__}: {exc}"
        detected_space_group = None
        requested_space_group_match = False
        composition_match = sorted(int(v) for v in h_out.detach().cpu().tolist()) == sorted(
            int(v) for v in target_h.detach().cpu().tolist()
        )
    pair_distances = _periodic_pairwise_distances(frac_coords=pos_out, cell_matrix=cell_out)
    min_pair_distance = float(pair_distances.min().detach().item()) if pair_distances.numel() > 0 else float("inf")
    volume = float(torch.abs(torch.linalg.det(cell_out)).detach().item())
    target_volume = float(torch.abs(torch.linalg.det(target_cell)).detach().item())
    volume_ratio = volume / max(target_volume, 1.0e-8)
    validation_failures: list[str] = []
    if validation_error is not None:
        validation_failures.append(validation_error)
    if not composition_match:
        validation_failures.append("composition_mismatch")
    if not requested_space_group_match:
        validation_failures.append(f"requested_sg_mismatch detected={detected_space_group}")
    if min_pair_distance < float(config.hard_min_distance):
        validation_failures.append(
            f"close_contacts min_pair={min_pair_distance:.4f} threshold={float(config.hard_min_distance):.4f}"
        )
    if (
        volume_ratio < float(config.hard_volume_ratio_min)
        or volume_ratio > float(config.hard_volume_ratio_max)
    ):
        validation_failures.append(
            f"bad_volume_ratio volume={volume:.6f} target_volume={target_volume:.6f}"
        )
    if validation_failures and not bool(config.return_best_even_if_invalid):
        raise RuntimeError(";".join(validation_failures))
    frac_rmse, mean_dist, max_dist, mean_shift, shift_spread = _fractional_debug_summary(
        pos=pos_out,
        h=h_out,
        target_pos=target_pos,
        target_h=target_h,
    )
    hard_satisfied, hard_violation = _hard_constraint_violation(
        composition_match=bool(composition_match),
        requested_sg_match=bool(requested_space_group_match),
        min_pair_distance=float(min_pair_distance),
        volume_ratio=float(volume_ratio),
        config=config,
    )
    del graph_idx
    return _CasalProjection(
        state=projection.state,
        pos=pos_out.detach().clone(),
        l=l_out.reshape(-1).detach().clone(),
        k=k_out.detach().clone(),
        h=h_out.detach().clone(),
        tau=projection.tau.detach().clone(),
        objective=float(projection.objective),
        coord_loss=float(projection.coord_loss),
        lattice_k_loss=float(projection.lattice_k_loss),
        physical_lattice_loss=float(projection.physical_lattice_loss),
        steric_loss=float(projection.steric_loss),
        volume_loss=float(projection.volume_loss),
        min_pair_distance=float(min_pair_distance),
        ssvd_rank=int(projection.ssvd_rank),
        ssvd_condition_number=float(projection.ssvd_condition_number),
        ssvd_delta_norm=float(projection.ssvd_delta_norm),
        template_labels=_template_labels(projection.state),
        prior_score=int(projection.state.prior_score),
        sg_detected=detected_space_group,
        requested_sg_match=bool(requested_space_group_match),
        composition_match=bool(composition_match),
        frac_rmse=float(frac_rmse),
        mean_dist=float(mean_dist),
        max_dist=float(max_dist),
        mean_shift=tuple(float(v) for v in mean_shift),
        shift_spread=float(shift_spread),
        validation_failure=";".join(validation_failures) if validation_failures else None,
        volume_ratio=float(volume_ratio),
        volume=float(volume),
        target_volume=float(target_volume),
        hard_constraint_satisfied=bool(hard_satisfied),
        hard_constraint_violation=float(hard_violation),
    )


def _project_graph_to_chart(
    *,
    graph_idx: int,
    requested_sg: int,
    target_pos: torch.Tensor,
    target_l: torch.Tensor,
    target_h: torch.Tensor,
    lattice_transform: Any,
    config: Algorithm10Config,
    template_prior: Any | None,
    oracle_reference_structure: Any | None = None,
    locked_template_state: Any | None = None,
    locked_tau: torch.Tensor | None = None,
) -> _CasalProjection:
    projection_start_time = time.time()
    device = target_pos.device
    dtype = target_pos.dtype
    target_cell = _decode_lattice_matrix(
        l=target_l.reshape(-1),
        num_atoms=int(target_pos.shape[0]),
        lattice_transform=lattice_transform,
    ).to(device=device, dtype=dtype)
    try:
        chart_frac, chart_h, _chart_cell, chart_k = _build_chart_target(
            frac_coords=target_pos,
            atomic_numbers=target_h,
            cell_matrix=target_cell,
            requested_sg=int(requested_sg),
            symprec=float(config.symprec),
            angle_tolerance=float(config.angle_tolerance),
        )
    except Exception:
        if not bool(config.return_best_even_if_invalid):
            raise
        chart_frac = torch.remainder(target_pos.detach().clone(), 1.0)
        chart_h = target_h.detach().clone().to(dtype=torch.long)
        chart_k = cell_to_k(target_cell.detach(), eps=1.0e-8).reshape(-1)
    metric = config.projection_metric()
    ssvd_config = config.ssvd_projection_config()
    if locked_template_state is not None:
        states = [locked_template_state]
    else:
        states = initialize_constrained_template_states(
            reference_frac_coords=target_pos.detach(),
            atomic_numbers=target_h.detach().to(dtype=torch.long),
            cell_matrix=target_cell.detach(),
            space_group_number=int(requested_sg),
            standardization="conventional",
            symprec=float(config.symprec),
            angle_tolerance=float(config.angle_tolerance),
            max_templates=int(config.max_templates),
            template_eval_limit=int(config.template_eval_limit),
            quick_templates=bool(config.quick_templates),
            top_k=max(1, int(config.template_eval_limit)),
            template_prior=template_prior,
            template_prior_weight=float(config.template_prior_weight),
            debug_template_candidates=False,
            freeze_lattice_free_vars=False,
            oracle_reference_structure=oracle_reference_structure,
            oracle_fit_structure=oracle_reference_structure,
        )
        states = _rank_projection_states(
            states=states,
            oracle_reference_structure=oracle_reference_structure,
            config=config,
        )
    if locked_tau is not None:
        tau_candidates = [locked_tau.detach().clone().to(device=device, dtype=dtype)]
    else:
        tau_candidates = _origin_shift_candidates(config=config, device=device, dtype=dtype)
    best: _CasalProjection | None = None
    reject_reasons: dict[str, int] = {}
    reject_examples: list[str] = []
    projection_attempt_count = 0
    projection_ssvd_fail_count = 0
    projection_materialize_fail_count = 0

    for state in states:
        labels = _template_labels(state)
        for tau in tau_candidates:
            projection_attempt_count += 1
            try:
                projected = fixed_template_ssvd_project(
                    template_state=state,
                    y_f=chart_frac,
                    y_k=chart_k,
                    y_h=chart_h.to(device=device, dtype=torch.long),
                    tau0=tau,
                    metric=metric,
                    config=ssvd_config,
                )
            except Exception as exc:
                _record_projection_reject(
                    reject_reasons=reject_reasons,
                    reject_examples=reject_examples,
                    stage="ssvd",
                    template_labels=labels,
                    tau=tau,
                    exc=exc,
                    max_examples=int(config.debug_projection_examples),
                )
                projection_ssvd_fail_count += 1
                continue
            try:
                candidate = _materialize_projection(
                    graph_idx=graph_idx,
                    projection=projected,
                    target_pos=target_pos,
                    target_h=target_h,
                    target_cell=target_cell,
                    requested_sg=int(requested_sg),
                    lattice_transform=lattice_transform,
                    config=config,
                )
            except Exception as exc:
                _record_projection_reject(
                    reject_reasons=reject_reasons,
                    reject_examples=reject_examples,
                    stage="validate",
                    template_labels=labels,
                    tau=tau,
                    exc=exc,
                    max_examples=int(config.debug_projection_examples),
                )
                projection_materialize_fail_count += 1
                continue
            if best is None or _projection_selection_key(candidate) < _projection_selection_key(best):
                best = candidate

    if best is None:
        reason_text = ",".join(f"{key}:{value}" for key, value in sorted(reject_reasons.items()))
        example_text = " || ".join(reject_examples)
        raise RuntimeError(f"projection_failed rejects={reason_text or 'none'} examples={example_text or 'none'}")

    best = replace(
        best,
        projection_elapsed_s=float(time.time() - projection_start_time),
        projection_state_count=int(len(states)),
        projection_tau_count=int(len(tau_candidates)),
        projection_attempt_count=int(projection_attempt_count),
        projection_ssvd_fail_count=int(projection_ssvd_fail_count),
        projection_materialize_fail_count=int(projection_materialize_fail_count),
    )

    if bool(config.debug):
        print(
            f"algorithm10_project graph={graph_idx + 1} sg={int(requested_sg)} "
            f"template_labels={best.template_labels} "
            f"objective={float(best.objective):.6f} coord={float(best.coord_loss):.6f} "
            f"lattice_k={float(best.lattice_k_loss):.6f} "
            f"physical_lattice={float(best.physical_lattice_loss):.6f} "
            f"steric={float(best.steric_loss):.6f} volume={float(best.volume_loss):.6f} "
            f"min_pair={float(best.min_pair_distance):.4f} "
            f"hard_ok={int(bool(best.hard_constraint_satisfied))} "
            f"hard_violation={float(best.hard_constraint_violation):.6f} "
            f"tau={[round(float(v), 4) for v in best.tau.reshape(-1).tolist()]} "
            f"prior_count={int(best.prior_score)} oracle_wyckoff={int(oracle_reference_structure is not None)} "
            f"sg_detected={best.sg_detected} requested_sg_match={int(bool(best.requested_sg_match))} "
            f"frac_rmse={best.frac_rmse:.6f} mean_dist={best.mean_dist:.6f} max_dist={best.max_dist:.6f} "
            f"mean_shift={[round(v, 4) for v in best.mean_shift]} shift_spread={best.shift_spread:.6f} "
            f"ssvd_rank={int(best.ssvd_rank)} ssvd_cond={float(best.ssvd_condition_number):.6e} "
            f"ssvd_delta={float(best.ssvd_delta_norm):.6f} "
            f"states={int(best.projection_state_count)} taus={int(best.projection_tau_count)} "
            f"attempts={int(best.projection_attempt_count)} "
            f"ssvd_fails={int(best.projection_ssvd_fail_count)} "
            f"materialize_fails={int(best.projection_materialize_fail_count)} "
            f"elapsed_s={float(best.projection_elapsed_s):.2f}",
            flush=True,
        )
    return best


def _init_graph_state(
    *,
    graph_idx: int,
    requested_sg: int,
    pos: torch.Tensor,
    l: torch.Tensor,
    h: torch.Tensor,
    lattice_transform: Any,
    config: Algorithm10Config,
    template_prior: Any | None,
    oracle_reference_structure: Any | None,
) -> _CasalGraphState:
    projection = _project_graph_to_chart(
        graph_idx=graph_idx,
        requested_sg=int(requested_sg),
        target_pos=pos,
        target_l=l,
        target_h=h,
        lattice_transform=lattice_transform,
        config=config,
        template_prior=template_prior,
        oracle_reference_structure=oracle_reference_structure,
    )
    if bool(config.casal_require_feasible_initial_projection) and not bool(projection.hard_constraint_satisfied):
        raise RuntimeError(
            "initial_projection_infeasible "
            f"hard_violation={float(projection.hard_constraint_violation):.6f} "
            f"min_pair={float(projection.min_pair_distance):.4f} "
            f"volume_ratio={float(projection.volume_ratio):.6f} "
            f"requested_sg_match={int(bool(projection.requested_sg_match))} "
            f"composition_match={int(bool(projection.composition_match))} "
            f"validation={projection.validation_failure}",
        )
    return _CasalGraphState(
        z_pos=projection.pos.detach().clone(),
        z_l=projection.l.detach().clone(),
        z_k=projection.k.detach().clone(),
        z_h=projection.h.detach().clone(),
        mu_f=torch.zeros_like(pos),
        mu_k=torch.zeros_like(projection.k.reshape(-1)),
        projection=projection,
    )


def _casal_step_graph(
    *,
    graph_idx: int,
    requested_sg: int,
    x_pos: torch.Tensor,
    x_l: torch.Tensor,
    h: torch.Tensor,
    graph_state: _CasalGraphState,
    rho: float,
    tau_step: float,
    should_project: bool,
    lattice_transform: Any,
    config: Algorithm10Config,
    projection_config: Algorithm10Config | None,
    accept_config: Algorithm10Config | None,
    template_prior: Any | None,
    oracle_reference_structure: Any | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    _CasalProjection | None,
    Exception | None,
    float,
    float,
    float,
    float,
    dict[str, Any],
]:
    if projection_config is None:
        projection_config = config
    if accept_config is None:
        accept_config = config
    num_atoms = int(x_pos.shape[0])
    x_k = _l_to_k(
        l=x_l.reshape(-1),
        num_atoms=num_atoms,
        lattice_transform=lattice_transform,
    ).to(device=x_pos.device, dtype=x_pos.dtype)

    coupling_mode = str(config.casal_coupling_mode).lower()
    coord_weight = float(config.casal_coord_weight)
    lattice_weight = float(config.casal_lattice_weight)
    projection_lattice_weight = float(config.casal_projection_lattice_weight)
    residual_mode = str(getattr(config, "casal_residual_mode", "wrap_plus_mu")).strip().lower()
    residual_geom_f = _wrap_delta(x_pos - graph_state.z_pos)
    residual_f = _casal_frac_residual(
        f=x_pos,
        z=graph_state.z_pos,
        mu=graph_state.mu_f,
        mode=residual_mode,
    )
    residual_k = x_k - graph_state.z_k + graph_state.mu_k
    beta_coord = float(config.casal_beta_f) * coord_weight
    beta_lattice = float(tau_step) * float(rho) * lattice_weight
    beta_projection_coord = float(config.casal_beta_z) * coord_weight
    beta_projection_lattice = float(tau_step) * float(rho) * projection_lattice_weight
    objective_before = float(getattr(graph_state.projection, "objective", float("nan")))
    selected_w_before = getattr(graph_state.projection, "template_labels", None)
    selected_tau_before = (
        float(graph_state.projection.tau.detach().reshape(-1)[0].item())
        if getattr(graph_state.projection, "tau", None) is not None
        else float("nan")
    )
    if coupling_mode in {"direct", "f_k", "fk", "position"}:
        x_pos = torch.remainder(x_pos - beta_coord * residual_f, 1.0)
        if lattice_weight != 0.0:
            x_k = x_k - beta_lattice * residual_k
            x_l = _k_to_l(
                k=x_k,
                num_atoms=num_atoms,
                lattice_transform=lattice_transform,
            ).to(device=x_pos.device, dtype=x_pos.dtype)
    elif coupling_mode in {"velocity", "v", "velocity_coordinate", "velocity_pd", "velocity_damped"}:
        # Coordinate coupling was already inserted as a velocity kick before the
        # KLDM kinetic reverse step. Keep this projection/dual step in the same
        # split geometry, and optionally leave lattice untouched for coordinate-
        # only debugging.
        if lattice_weight != 0.0:
            x_k = x_k - beta_lattice * residual_k
            x_l = _k_to_l(
                k=x_k,
                num_atoms=num_atoms,
                lattice_transform=lattice_transform,
            ).to(device=x_pos.device, dtype=x_pos.dtype)
    elif coupling_mode in {"project_only", "none"}:
        pass
    else:
        raise ValueError(f"Unsupported casal_coupling_mode={config.casal_coupling_mode!r}")

    projection: _CasalProjection | None = None
    projection_error: Exception | None = None
    projection_success = False
    projection_metric_accepted = False
    if should_project:
        gauge_mode = str(getattr(config, "casal_gauge_mode", "full")).strip().lower()
        locked_state = None
        locked_tau = None
        if gauge_mode in {
            "locked",
            "locked_after_initial_projection",
            "locked_w_tau",
            "locked_w_tau_pi",
            "locked_w_tau_assignment",
        }:
            locked_state = getattr(graph_state.projection, "state", None)
            locked_tau = getattr(graph_state.projection, "tau", None)
        elif gauge_mode in {"semi_locked", "semi_locked_w_tau"}:
            locked_state = getattr(graph_state.projection, "state", None)
            locked_tau = getattr(graph_state.projection, "tau", None)
        z_arg_f = torch.remainder(
            graph_state.z_pos
            + beta_projection_coord * _wrap_delta(x_pos + graph_state.mu_f - graph_state.z_pos),
            1.0,
        )
        if projection_lattice_weight == 0.0:
            z_arg_k = graph_state.z_k.detach().clone()
        else:
            z_arg_k = graph_state.z_k - beta_projection_lattice * (
                graph_state.z_k - x_k - graph_state.mu_k
            )
        z_arg_l = _k_to_l(
            k=z_arg_k,
            num_atoms=num_atoms,
            lattice_transform=lattice_transform,
        ).to(device=x_pos.device, dtype=x_pos.dtype)
        try:
            metric_before = _casal_metric_value(
                x_pos=x_pos,
                x_k=x_k,
                z_pos=graph_state.z_pos,
                z_k=graph_state.z_k,
                config=accept_config,
            )
            projection = _project_graph_to_chart(
                graph_idx=graph_idx,
                requested_sg=int(requested_sg),
                target_pos=z_arg_f,
                target_l=z_arg_l,
                target_h=h,
                lattice_transform=lattice_transform,
                config=projection_config,
                template_prior=template_prior,
                oracle_reference_structure=oracle_reference_structure,
                locked_template_state=locked_state,
                locked_tau=locked_tau,
            )
            hard_ok, hard_violation = _hard_constraint_violation(
                composition_match=bool(projection.composition_match),
                requested_sg_match=bool(projection.requested_sg_match),
                min_pair_distance=float(projection.min_pair_distance),
                volume_ratio=float(projection.volume_ratio),
                config=accept_config,
            )
            metric_after = _casal_metric_value(
                x_pos=x_pos,
                x_k=x_k,
                z_pos=projection.pos,
                z_k=projection.k.reshape(-1),
                config=accept_config,
            )
            if metric_after > metric_before + float(getattr(config, "casal_metric_accept_tol", 1.0e-8)):
                raise RuntimeError(
                    "projection_metric_regression "
                    f"before={metric_before:.6f} after={metric_after:.6f}"
                )
            if bool(config.casal_require_feasible_projection) and not bool(hard_ok):
                raise RuntimeError(
                    "projection_infeasible "
                    f"hard_violation={float(hard_violation):.6f} "
                    f"min_pair={float(projection.min_pair_distance):.4f} "
                    f"volume_ratio={float(projection.volume_ratio):.6f} "
                    f"requested_sg_match={int(bool(projection.requested_sg_match))} "
                    f"composition_match={int(bool(projection.composition_match))} "
                    f"validation={projection.validation_failure}",
                )
        except Exception as exc:
            projection_error = exc
        else:
            projection_success = True
            projection_metric_accepted = True
            graph_state.projection = projection
            graph_state.z_pos = projection.pos.detach().clone()
            graph_state.z_l = projection.l.detach().clone()
            graph_state.z_k = projection.k.detach().clone()
            graph_state.z_h = projection.h.detach().clone()

    dual_rule = str(getattr(config, "casal_dual_rule", "tau_over_rho")).strip().lower()
    if dual_rule in {"tau", "plain_tau"}:
        dual_step = float(config.casal_mu_eta) * float(tau_step)
    elif dual_rule in {"tau_over_rho", "tau/rho", "normalized"}:
        dual_step = float(config.casal_mu_eta) * float(tau_step) / max(float(rho), 1.0e-12)
    elif dual_rule in {"beta", "plain_beta"}:
        dual_step = float(config.casal_mu_eta) * float(config.casal_beta_z)
    else:
        raise ValueError(f"Unsupported casal_dual_rule={config.casal_dual_rule!r}")
    dual_enabled = bool(getattr(config, "casal_dual_enabled", True))
    dual_policy = str(getattr(config, "casal_dual_update_policy", "projection_success_only")).strip().lower()
    dual_updated = False
    stale_dual_step = False
    if dual_enabled:
        update_dual = True
        if dual_policy in {"projection_success_only", "success_only"}:
            update_dual = bool(projection_success)
        elif dual_policy in {"every_step", "always"}:
            update_dual = True
        elif dual_policy in {"disabled", "none"}:
            update_dual = False
        else:
            raise ValueError(f"Unsupported casal_dual_update_policy={config.casal_dual_update_policy!r}")
        if update_dual:
            graph_state.mu_f = graph_state.mu_f + dual_step * coord_weight * _wrap_delta(x_pos - graph_state.z_pos)
            graph_state.mu_k = graph_state.mu_k + dual_step * lattice_weight * (x_k - graph_state.z_k)
            dual_updated = True
        elif should_project and not projection_success:
            stale_dual_step = True
    clip = float(config.casal_mu_clip)
    if clip > 0.0:
        graph_state.mu_f = graph_state.mu_f.clamp(min=-clip, max=clip)
        graph_state.mu_k = graph_state.mu_k.clamp(min=-clip, max=clip)

    residual_coord_norm = float(torch.linalg.norm(_wrap_delta(x_pos - graph_state.z_pos).reshape(-1)).detach().item())
    residual_k_norm = float(torch.linalg.norm((x_k - graph_state.z_k).reshape(-1)).detach().item())
    mu_coord_norm = float(torch.linalg.norm(graph_state.mu_f.reshape(-1)).detach().item())
    mu_k_norm = float(torch.linalg.norm(graph_state.mu_k.reshape(-1)).detach().item())
    step_diag = {
        "projection_success": projection_success,
        "projection_metric_accepted": projection_metric_accepted,
        "dual_updated": dual_updated,
        "stale_dual_step": stale_dual_step,
        "projection_objective_before": objective_before,
        "projection_objective_after": float(getattr(projection, "objective", objective_before)) if projection is not None else objective_before,
        "selected_W_before": selected_w_before,
        "selected_W_after": getattr(projection, "template_labels", selected_w_before) if projection is not None else selected_w_before,
        "selected_tau_before": selected_tau_before,
        "selected_tau_after": float(projection.tau.detach().reshape(-1)[0].item()) if projection is not None else selected_tau_before,
        "residual_mode": residual_mode,
        "r_geom_norm": float(torch.linalg.norm(residual_geom_f.reshape(-1)).detach().item()),
        "r_force_norm": float(torch.linalg.norm(residual_f.reshape(-1)).detach().item()),
    }
    return x_pos, x_l, projection, projection_error, residual_coord_norm, residual_k_norm, mu_coord_norm, mu_k_norm, step_diag


def sample_kldm_casal_chart(
    *,
    model: Any,
    n_steps: int,
    batch: Any,
    lattice_transform: Any = None,
    t_start: float = 1.0,
    t_final: float = 1.0e-6,
    config: Algorithm10Config | None = None,
    template_prior: Any | None = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]:
    if config is None:
        config = Algorithm10Config()
    if not hasattr(batch, "space_group"):
        raise ValueError("Algorithm10 requires batch.space_group.")

    started = time.perf_counter()
    state = model._prepare_csp_sampling(
        batch=batch,
        n_steps=n_steps,
        t_start=t_start,
        t_final=t_final,
    )
    batch = state["batch"]
    ptr = batch.ptr.tolist()
    requested_sgs = torch.as_tensor(batch.space_group, device=state["l_t"].device, dtype=torch.long).reshape(-1)
    oracle_structures = _oracle_wyckoff_structures(
        batch=batch,
        ptr=ptr,
        lattice_transform=lattice_transform,
        device=state["l_t"].device,
        dtype=state["l_t"].dtype,
        config=config,
    )
    total_steps = max(1, int(state["sampling_time_grid"].numel()) - 1)
    projection_interval = max(1, int(config.projection_interval))
    projection_start_step = max(
        1,
        int(config.projection_start_step),
        int(math.ceil(float(config.projection_start_fraction) * float(total_steps))),
    )
    projection_start_mode = str(getattr(config, "projection_start_mode", "fraction")).strip().lower()
    inner_projection_mode = str(getattr(config, "casal_inner_projection_mode", "full")).strip().lower()
    final_projection_mode = str(getattr(config, "casal_final_projection_mode", "same")).strip().lower()
    inner_projection_config = _projection_config_for_mode(config, mode=inner_projection_mode)
    final_projection_config = (
        inner_projection_config
        if final_projection_mode in {"same", "inner", "inner_same"}
        else _projection_config_for_mode(config, mode=final_projection_mode)
    )
    graph_states: list[_CasalGraphState | None] = [None for _ in range(int(batch.num_graphs))]
    graph_diagnostics: list[dict[str, Any]] = [
        {
            "graph_idx": int(graph_idx),
            "requested_sg": int(requested_sgs[graph_idx].item()),
            "last_projection_step": 0,
            "last_projection_success": False,
            "last_projection_error": None,
            "num_projection_successes": 0,
            "num_projection_failures": 0,
            "num_init_projection_attempts": 0,
            "num_init_projection_failures": 0,
            "first_feasible_projection_step": 0,
            "xz_residual_coord_last": float("nan"),
            "xz_residual_k_last": float("nan"),
            "mu_norm_coord_last": float("nan"),
            "mu_norm_k_last": float("nan"),
            "velocity_kick_norm_last": float("nan"),
            "damping_norm_last": float("nan"),
            "v_norm_before_last": float("nan"),
            "v_norm_after_last": float("nan"),
            "v_norm_over_sigma_v_last": float("nan"),
            "delta_v_mean_norm_last": float("nan"),
            "v_mean_norm_before_last": float("nan"),
            "v_mean_norm_after_last": float("nan"),
            "impulse_clipped_last": False,
            "num_impulse_clipped_steps": 0,
            "num_velocity_coupling_steps": 0,
            "stale_dual_steps": 0,
            "dual_updated_steps": 0,
            "num_w_switches": 0,
            "num_tau_switches": 0,
            "residual_mode": str(getattr(config, "casal_residual_mode", "wrap_plus_mu")),
            "damping_metric": str(getattr(config, "casal_damping_metric", "fractional")),
            "casal_lattice_weight": float(config.casal_lattice_weight),
            "casal_projection_lattice_weight": float(config.casal_projection_lattice_weight),
            "casal_velocity_controller_mode": str(getattr(config, "casal_velocity_controller_mode", "impulse")),
            "casal_inner_projection_mode": inner_projection_mode,
            "casal_final_projection_mode": final_projection_mode,
            "final_projection_refreshed": False,
        }
        for graph_idx in range(int(batch.num_graphs))
    ]
    if bool(config.debug):
        print(
            f"algorithm10_mode mode=faithful_chart_projection_operator_split "
            f"coupling_mode={config.casal_coupling_mode} "
            f"coord_weight={float(config.casal_coord_weight):.6g} "
            f"lattice_weight={float(config.casal_lattice_weight):.6g} "
            f"projection_lattice_weight={float(config.casal_projection_lattice_weight):.6g} "
            f"velocity_alpha={float(config.casal_velocity_alpha):.6g} "
            f"velocity_sign={float(getattr(config, 'casal_velocity_sign', 1.0)):.6g} "
            f"velocity_controller={str(getattr(config, 'casal_velocity_controller_mode', 'impulse'))} "
            f"beta_f={float(config.casal_beta_f):.6g} "
            f"beta_z={float(config.casal_beta_z):.6g} "
            f"velocity_placement={str(config.casal_velocity_placement)} "
            f"damping_mode={str(config.casal_velocity_damping_mode)} "
            f"residual_mode={str(config.casal_residual_mode)} "
            f"origin_shift_mode={config.origin_shift_mode}",
            flush=True,
        )
        print(
            f"algorithm10_projection_schedule start_step={projection_start_step}/{total_steps} "
            f"interval={projection_interval} min_volume={float(config.projection_min_volume):.4f} "
            f"min_length={float(config.projection_min_lattice_length):.4f} "
            f"oracle_wyckoff={int(bool(config.oracle_wyckoff_debug))} "
            f"start_mode={projection_start_mode} gauge_mode={str(getattr(config, 'casal_gauge_mode', 'full'))} "
            f"inner_projection_mode={inner_projection_mode} final_projection_mode={final_projection_mode}",
            flush=True,
        )

    with torch.no_grad():
        for step_idx, times in enumerate(iter_sampling_times(batch=batch, grid=state["sampling_time_grid"]), start=1):
            rho = _rho_schedule(config, step_idx - 1, total_steps)
            tau_step = max(float(times.dt) * float(config.casal_tau_scale), 0.0)
            velocity_mode = str(config.casal_coupling_mode).lower() in {
                "velocity",
                "v",
                "velocity_coordinate",
                "velocity_pd",
                "velocity_damped",
            }
            velocity_sign = float(getattr(config, "casal_velocity_sign", 1.0))
            velocity_placement = str(getattr(config, "casal_velocity_placement", "corrector")).strip().lower()
            f_step_input = state["f_t"].detach().clone()
            if velocity_mode and velocity_placement == "predictor":
                dt_abs = max(abs(float(times.dt)), 1.0e-12)
                for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                    graph_state = graph_states[graph_idx]
                    if graph_state is None:
                        continue
                    graph_diag = graph_diagnostics[graph_idx]
                    cell_now = _decode_lattice_matrix(
                        l=state["l_t"][graph_idx].reshape(-1),
                        num_atoms=int(end - start),
                        lattice_transform=lattice_transform,
                    ).to(device=state["f_t"].device, dtype=state["f_t"].dtype)
                    r_geom = _wrap_delta(state["f_t"][start:end] - graph_state.z_pos)
                    r_force = _casal_frac_residual(
                        f=state["f_t"][start:end],
                        z=graph_state.z_pos,
                        mu=graph_state.mu_f,
                        mode=str(getattr(config, "casal_residual_mode", "wrap_plus_mu")),
                    )
                    damping_residual_name = str(getattr(config, "casal_damping_residual", "geom")).strip().lower()
                    damping_residual = r_force if damping_residual_name == "force" else r_geom
                    dv_prop = _velocity_correction(
                        r_force=r_force,
                        dt_abs=dt_abs,
                        rho=rho,
                        config=config,
                    )
                    dv_prop_mean_norm = 0.0
                    if bool(getattr(config, "casal_velocity_mean_free", True)):
                        dv_prop, dv_prop_mean_norm = _mean_free_velocity(dv_prop)
                    dv_prop, clipped = _clip_norm(dv_prop, float(getattr(config, "casal_max_delta_v", math.inf)))
                    dv_damp = _damp_velocity(
                        v=state["v_t"][start:end],
                        r_geom=damping_residual,
                        gamma=float(getattr(config, "casal_velocity_damping", 0.0)),
                        mode=str(getattr(config, "casal_velocity_damping_mode", "none")),
                        metric=str(getattr(config, "casal_damping_metric", "fractional")),
                        cell=cell_now,
                    )
                    dv_damp_mean_norm = 0.0
                    if bool(getattr(config, "casal_velocity_mean_free", True)):
                        dv_damp, dv_damp_mean_norm = _mean_free_velocity(dv_damp)
                    sigma_like = _velocity_sigma_like(
                        sampling_tdm=state["sampling_tdm"],
                        t_nodes=times.now.nodes[start:end],
                        ref=state["v_t"][start:end],
                    )
                    v_before = state["v_t"][start:end].detach().clone()
                    state["v_t"][start:end] = state["v_t"][start:end] + dv_prop + dv_damp
                    v_after = state["v_t"][start:end]
                    graph_diag["velocity_kick_norm_last"] = float(torch.linalg.norm(dv_prop.reshape(-1)).detach().item())
                    graph_diag["damping_norm_last"] = float(torch.linalg.norm(dv_damp.reshape(-1)).detach().item())
                    graph_diag["v_norm_before_last"] = float(torch.linalg.norm(v_before.reshape(-1)).detach().item())
                    graph_diag["v_norm_after_last"] = float(torch.linalg.norm(v_after.reshape(-1)).detach().item())
                    graph_diag["v_mean_norm_before_last"] = float(torch.linalg.norm(v_before.mean(dim=0).reshape(-1)).detach().item())
                    graph_diag["v_mean_norm_after_last"] = float(torch.linalg.norm(v_after.mean(dim=0).reshape(-1)).detach().item())
                    graph_diag["v_norm_over_sigma_v_last"] = float(
                        (torch.linalg.norm(v_after.reshape(-1)) / sigma_like.reshape(-1).norm().clamp_min(1.0e-12)).detach().item()
                    )
                    graph_diag["delta_v_mean_norm_last"] = float(max(dv_prop_mean_norm, dv_damp_mean_norm))
                    graph_diag["impulse_clipped_last"] = bool(clipped)
                    graph_diag["num_velocity_coupling_steps"] += 1
                    if bool(clipped):
                        graph_diag["num_impulse_clipped_steps"] += 1
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
            if velocity_mode and velocity_placement == "corrector":
                dt_abs = max(abs(float(times.dt)), 1.0e-12)
                for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                    graph_state = graph_states[graph_idx]
                    if graph_state is None:
                        continue
                    graph_diag = graph_diagnostics[graph_idx]
                    cell_now = _decode_lattice_matrix(
                        l=state["l_t"][graph_idx].reshape(-1),
                        num_atoms=int(end - start),
                        lattice_transform=lattice_transform,
                    ).to(device=state["f_t"].device, dtype=state["f_t"].dtype)
                    f_base = state["f_t"][start:end].detach().clone()
                    v_base = state["v_t"][start:end].detach().clone()
                    r_geom = _wrap_delta(f_base - graph_state.z_pos)
                    r_force = _casal_frac_residual(
                        f=f_base,
                        z=graph_state.z_pos,
                        mu=graph_state.mu_f,
                        mode=str(getattr(config, "casal_residual_mode", "wrap_plus_mu")),
                    )
                    damping_residual_name = str(getattr(config, "casal_damping_residual", "geom")).strip().lower()
                    damping_residual = r_force if damping_residual_name == "force" else r_geom
                    dv_prop = _velocity_correction(
                        r_force=r_force,
                        dt_abs=dt_abs,
                        rho=rho,
                        config=config,
                    )
                    dv_prop_mean_norm = 0.0
                    if bool(getattr(config, "casal_velocity_mean_free", True)):
                        dv_prop, dv_prop_mean_norm = _mean_free_velocity(dv_prop)
                    dv_prop, clipped = _clip_norm(dv_prop, float(getattr(config, "casal_max_delta_v", math.inf)))
                    dv_damp = _damp_velocity(
                        v=v_base,
                        r_geom=damping_residual,
                        gamma=float(getattr(config, "casal_velocity_damping", 0.0)),
                        mode=str(getattr(config, "casal_velocity_damping_mode", "none")),
                        metric=str(getattr(config, "casal_damping_metric", "fractional")),
                        cell=cell_now,
                    )
                    dv_damp_mean_norm = 0.0
                    if bool(getattr(config, "casal_velocity_mean_free", True)):
                        dv_damp, dv_damp_mean_norm = _mean_free_velocity(dv_damp)
                    v_new = v_base + dv_prop + dv_damp
                    f_new = state["sampling_tdm"].wrap_displacements(
                        f_step_input[start:end] - float(times.dt) * v_new,
                    )
                    sigma_like = _velocity_sigma_like(
                        sampling_tdm=state["sampling_tdm"],
                        t_nodes=times.now.nodes[start:end],
                        ref=v_new,
                    )
                    state["v_t"][start:end] = v_new
                    state["f_t"][start:end] = f_new
                    graph_diag["velocity_kick_norm_last"] = float(torch.linalg.norm(dv_prop.reshape(-1)).detach().item())
                    graph_diag["damping_norm_last"] = float(torch.linalg.norm(dv_damp.reshape(-1)).detach().item())
                    graph_diag["v_norm_before_last"] = float(torch.linalg.norm(v_base.reshape(-1)).detach().item())
                    graph_diag["v_norm_after_last"] = float(torch.linalg.norm(v_new.reshape(-1)).detach().item())
                    graph_diag["v_mean_norm_before_last"] = float(torch.linalg.norm(v_base.mean(dim=0).reshape(-1)).detach().item())
                    graph_diag["v_mean_norm_after_last"] = float(torch.linalg.norm(v_new.mean(dim=0).reshape(-1)).detach().item())
                    graph_diag["v_norm_over_sigma_v_last"] = float(
                        (torch.linalg.norm(v_new.reshape(-1)) / sigma_like.reshape(-1).norm().clamp_min(1.0e-12)).detach().item()
                    )
                    graph_diag["delta_v_mean_norm_last"] = float(max(dv_prop_mean_norm, dv_damp_mean_norm))
                    graph_diag["impulse_clipped_last"] = bool(clipped)
                    graph_diag["num_velocity_coupling_steps"] += 1
                    if bool(clipped):
                        graph_diag["num_impulse_clipped_steps"] += 1
            state["l_t"] = model._reverse_lattice_sampling_step(
                t=times.now.lattice,
                x_t=state["l_t"],
                pred=preds_curr["l"],
                dt=times.dt,
                num_atoms=batch.num_atoms,
            )

            residual_coord_norm = 0.0
            residual_k_norm = 0.0
            mu_coord_norm = 0.0
            mu_k_norm = 0.0

            for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                graph_state = graph_states[graph_idx]
                just_initialized = False
                if projection_start_mode in {"dynamic_projectable", "dynamic", "late_dynamic"}:
                    scheduled_projection = (
                        step_idx >= projection_start_step
                        and (step_idx % projection_interval == 0 or step_idx == total_steps)
                    )
                else:
                    scheduled_projection = (
                        step_idx >= projection_start_step
                        and (step_idx % projection_interval == 0 or step_idx == total_steps)
                    )
                lattice_sane, lattice_reason, lattice_volume, lattice_lengths = _projection_lattice_sane(
                    l=state["l_t"][graph_idx].reshape(-1),
                    num_atoms=int(end - start),
                    lattice_transform=lattice_transform,
                    config=config,
                )
                should_project = bool(scheduled_projection and lattice_sane)
                if scheduled_projection and not lattice_sane and bool(config.debug) and bool(config.debug_projection_failures):
                    print(
                        f"algorithm10_projection_defer graph={graph_idx + 1} step={step_idx}/{total_steps} "
                        f"reason={lattice_reason} volume={lattice_volume:.6f} lengths={lattice_lengths}",
                        flush=True,
                    )
                if graph_state is None and should_project:
                    graph_diagnostics[graph_idx]["num_init_projection_attempts"] += 1
                    try:
                        graph_state = _init_graph_state(
                            graph_idx=graph_idx,
                            requested_sg=int(requested_sgs[graph_idx].item()),
                            pos=state["f_t"][start:end],
                            l=state["l_t"][graph_idx].reshape(-1),
                            h=state["a_t"][start:end],
                            lattice_transform=lattice_transform,
                            config=config,
                            template_prior=template_prior,
                            oracle_reference_structure=oracle_structures[graph_idx],
                        )
                        graph_states[graph_idx] = graph_state
                        if not graph_diagnostics[graph_idx]["first_feasible_projection_step"]:
                            graph_diagnostics[graph_idx]["first_feasible_projection_step"] = int(step_idx)
                        just_initialized = True
                    except Exception as exc:
                        graph_diagnostics[graph_idx]["num_init_projection_failures"] += 1
                        if bool(config.debug) and bool(config.debug_projection_failures):
                            print(
                                f"algorithm10_init_retry_skip graph={graph_idx + 1} "
                                f"step={step_idx}/{total_steps} sg={int(requested_sgs[graph_idx].item())} "
                                f"reason={type(exc).__name__} detail={exc}",
                                flush=True,
                            )
                if graph_state is None:
                    continue
                x_pos, x_l, projection, projection_error, r_coord, r_k, m_coord, m_k, step_diag = _casal_step_graph(
                    graph_idx=graph_idx,
                    requested_sg=int(requested_sgs[graph_idx].item()),
                    x_pos=state["f_t"][start:end],
                    x_l=state["l_t"][graph_idx].reshape(-1),
                    h=state["a_t"][start:end],
                    graph_state=graph_state,
                    rho=rho,
                    tau_step=tau_step,
                    should_project=bool(should_project and not just_initialized),
                    lattice_transform=lattice_transform,
                    config=config,
                    projection_config=inner_projection_config,
                    accept_config=config,
                    template_prior=template_prior,
                    oracle_reference_structure=oracle_structures[graph_idx],
                )
                graph_diag = graph_diagnostics[graph_idx]
                graph_diag["xz_residual_coord_last"] = float(r_coord)
                graph_diag["xz_residual_k_last"] = float(r_k)
                graph_diag["mu_norm_coord_last"] = float(m_coord)
                graph_diag["mu_norm_k_last"] = float(m_k)
                graph_diag["residual_mode"] = step_diag.get("residual_mode", graph_diag.get("residual_mode"))
                if bool(step_diag.get("dual_updated", False)):
                    graph_diag["dual_updated_steps"] += 1
                if bool(step_diag.get("stale_dual_step", False)):
                    graph_diag["stale_dual_steps"] += 1
                if step_diag.get("selected_W_before") != step_diag.get("selected_W_after"):
                    graph_diag["num_w_switches"] += 1
                tau_before = step_diag.get("selected_tau_before", float("nan"))
                tau_after = step_diag.get("selected_tau_after", float("nan"))
                if math.isfinite(float(tau_before)) and math.isfinite(float(tau_after)) and abs(float(tau_before) - float(tau_after)) > 1.0e-8:
                    graph_diag["num_tau_switches"] += 1
                graph_diag["projection_objective_before"] = step_diag.get("projection_objective_before", float("nan"))
                graph_diag["projection_objective_after"] = step_diag.get("projection_objective_after", float("nan"))
                graph_diag["r_geom_norm"] = step_diag.get("r_geom_norm", float("nan"))
                graph_diag["r_force_norm"] = step_diag.get("r_force_norm", float("nan"))
                if should_project and not just_initialized:
                    graph_diag["last_projection_step"] = int(step_idx)
                    projection_success = bool(step_diag.get("projection_success", projection is not None and projection_error is None))
                    graph_diag["last_projection_success"] = projection_success
                    if projection_success and projection is not None:
                        graph_diag["num_projection_successes"] += 1
                        graph_diag["last_projection_error"] = None
                    else:
                        graph_diag["num_projection_failures"] += 1
                        graph_diag["last_projection_error"] = None if projection_error is None else f"{type(projection_error).__name__}: {projection_error}"
                state["f_t"][start:end] = x_pos
                state["l_t"][graph_idx] = x_l.reshape_as(state["l_t"][graph_idx])
                if (
                    should_project
                    and not just_initialized
                    and projection is None
                    and projection_error is not None
                    and bool(config.debug)
                    and bool(config.debug_projection_failures)
                ):
                    print(
                        f"algorithm10_project_skip graph={graph_idx + 1} step={step_idx}/{total_steps} "
                        f"reason={type(projection_error).__name__} detail={projection_error}",
                        flush=True,
                    )
                residual_coord_norm += r_coord
                residual_k_norm += r_k
                mu_coord_norm += m_coord
                mu_k_norm += m_k

            if bool(config.debug) and (
                step_idx == total_steps
                or (step_idx >= projection_start_step and step_idx % projection_interval == 0)
            ):
                print(
                    f"algorithm10_step step={step_idx}/{total_steps} rho={rho:.6f} tau={tau_step:.6e} "
                    f"xz_residual_coord={residual_coord_norm:.6f} xz_residual_k={residual_k_norm:.6f} "
                    f"mu_norm_coord={mu_coord_norm:.6f} mu_norm_k={mu_k_norm:.6f}",
                    flush=True,
                )

    pos_parts: list[torch.Tensor] = []
    l_parts: list[torch.Tensor] = []
    h_parts: list[torch.Tensor] = []
    for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        graph_state = graph_states[graph_idx]
        if graph_state is None:
            pos_invalid, l_invalid, h_invalid = _invalid_sample_like(
                pos_ref=state["f_t"][start:end],
                l_ref=state["l_t"][graph_idx].reshape(-1),
                h_ref=state["a_t"][start:end],
            )
            pos_parts.append(pos_invalid)
            l_parts.append(l_invalid.reshape_as(state["l_t"][graph_idx]))
            h_parts.append(h_invalid)
            continue
        final_projection = graph_state.projection
        if final_projection_mode not in {"same", "inner", "inner_same"}:
            try:
                refreshed_projection = _project_graph_to_chart(
                    graph_idx=graph_idx,
                    requested_sg=int(requested_sgs[graph_idx].item()),
                    target_pos=state["f_t"][start:end],
                    target_l=state["l_t"][graph_idx].reshape(-1),
                    target_h=state["a_t"][start:end],
                    lattice_transform=lattice_transform,
                    config=final_projection_config,
                    template_prior=template_prior,
                    oracle_reference_structure=oracle_structures[graph_idx],
                )
            except Exception:
                refreshed_projection = None
            if refreshed_projection is not None:
                final_projection = refreshed_projection
                graph_state.projection = refreshed_projection
                graph_state.z_pos = refreshed_projection.pos.detach().clone()
                graph_state.z_l = refreshed_projection.l.detach().clone()
                graph_state.z_k = refreshed_projection.k.detach().clone()
                graph_state.z_h = refreshed_projection.h.detach().clone()
                graph_diagnostics[graph_idx]["final_projection_refreshed"] = True
        pos_parts.append(final_projection.pos.to(device=state["f_t"].device, dtype=state["f_t"].dtype))
        l_parts.append(
            final_projection.l.to(device=state["l_t"].device, dtype=state["l_t"].dtype).reshape_as(
                state["l_t"][graph_idx]
            )
        )
        h_parts.append(final_projection.h.to(device=state["a_t"].device, dtype=state["a_t"].dtype))

    if state["restore_training"]:
        state["score_network"].train()
    if bool(config.debug):
        elapsed = time.perf_counter() - started
        valid_graphs = sum(1 for item in graph_states if item is not None)
        print(
            f"algorithm10_progress phase=done graphs={int(batch.num_graphs)} "
            f"valid_graphs={valid_graphs}/{int(batch.num_graphs)} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    result = (
        torch.cat(pos_parts, dim=0),
        torch.zeros_like(state["v_t"]),
        torch.stack(l_parts, dim=0),
        torch.cat(h_parts, dim=0),
    )
    if not return_diagnostics:
        return result
    diagnostics: list[dict[str, Any]] = []
    for graph_idx, graph_state in enumerate(graph_states):
        graph_diag = dict(graph_diagnostics[graph_idx])
        projection = None if graph_state is None else graph_state.projection
        if projection is None:
            graph_diag.update(
                {
                    "selected_W": None,
                    "selected_tau": float("nan"),
                    "projection_objective": float("nan"),
                    "hard_constraint_violation": float("nan"),
                    "min_pair_distance": float("nan"),
                    "volume_ratio": float("nan"),
                    "projection_elapsed_s": float("nan"),
                    "projection_state_count": 0,
                    "projection_tau_count": 0,
                    "projection_attempt_count": 0,
                    "projection_ssvd_fail_count": 0,
                    "projection_materialize_fail_count": 0,
                    "projection_objective_before": float("nan"),
                    "projection_objective_after": float("nan"),
                    "r_geom_norm": float("nan"),
                    "r_force_norm": float("nan"),
                }
            )
        else:
            graph_diag.update(
                {
                    "selected_W": projection.template_labels,
                    "selected_tau": float(projection.tau.detach().reshape(-1)[0].item()),
                    "projection_objective": float(projection.objective),
                    "hard_constraint_violation": float(projection.hard_constraint_violation),
                    "min_pair_distance": float(projection.min_pair_distance),
                    "volume_ratio": float(projection.volume_ratio),
                    "projection_elapsed_s": float(projection.projection_elapsed_s),
                    "projection_state_count": int(projection.projection_state_count),
                    "projection_tau_count": int(projection.projection_tau_count),
                    "projection_attempt_count": int(projection.projection_attempt_count),
                    "projection_ssvd_fail_count": int(projection.projection_ssvd_fail_count),
                    "projection_materialize_fail_count": int(projection.projection_materialize_fail_count),
                }
            )
        diagnostics.append(graph_diag)
    return (*result, diagnostics)
