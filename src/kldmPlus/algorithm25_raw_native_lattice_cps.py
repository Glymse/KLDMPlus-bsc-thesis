from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kldmPlus.algorithm25_kldm_pc_cps_lattice import (
    _require_single_graph_lattice_state,
    _restore_l_shape,
    Algorithm25StateUpdateDiagnostics,
    apply_lattice_cps_to_state,
    cps_gamma_weight,
    lattice6_to_matrix,
    matrix_to_lattice6,
    predict_clean_lattice_from_prediction,
    should_project_step,
    spacegroup_to_crystal_family,
)
from kldmPlus.utils.time import iter_sampling_times


@dataclass(frozen=True)
class Algorithm25RawNativeConfig:
    total_steps: int = 1000
    projection_start_frac: float = 0.25
    projection_interval: int = 50
    gamma_min: float = 0.05
    gamma_max: float = 3.0
    gamma_power: float = 2.0
    tau: float = 0.25
    n_correction_steps: int = 1
    use_gate: bool = True
    raw_violation_min: float = 1.0e-8
    raw_violation_max: float = 4.0
    shift_cap_scaled: float = 1.0
    angle_scale_deg: float = 10.0
    use_final_projection: bool = False
    collect_projection_diagnostics: bool = False


@dataclass(frozen=True)
class RawNativeProjectionDiagnostics:
    family: str
    selected_variant: str
    raw_violation_before: float
    raw_violation_after: float
    scaled_shift_norm: float
    gamma: float
    weight: float
    accepted: bool
    skipped_reason: str
    ell0_hat: torch.Tensor
    ell0_raw: torch.Tensor
    ell0_soft: torch.Tensor


@dataclass(frozen=True)
class RawNativeIntervention:
    remaining_step: int
    tau: float
    family: str
    selected_variant: str
    gamma: float
    weight: float
    raw_violation_before: float
    raw_violation_after: float
    scaled_shift_norm: float
    lattice_state_shift_norm: float


@dataclass(frozen=True)
class RawNativeSamplerResult:
    frac_coords: torch.Tensor
    velocity: torch.Tensor
    lattice: torch.Tensor
    atom_types: torch.Tensor
    interventions: tuple[RawNativeIntervention, ...]


def _cell_lengths_angles(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cell_t = torch.as_tensor(cell).reshape(3, 3)
    lengths = torch.linalg.norm(cell_t, dim=-1)

    def _angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        denom = (torch.linalg.norm(u) * torch.linalg.norm(v)).clamp_min(1.0e-12)
        cosine = torch.clamp(torch.dot(u, v) / denom, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cosine))

    alpha = _angle(cell_t[1], cell_t[2])
    beta = _angle(cell_t[0], cell_t[2])
    gamma = _angle(cell_t[0], cell_t[1])
    return lengths, torch.stack([alpha, beta, gamma])


def _lengths_angles_to_cell(lengths: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    lengths = torch.as_tensor(lengths).reshape(3)
    angles = torch.deg2rad(torch.as_tensor(angles_deg, device=lengths.device, dtype=lengths.dtype).reshape(3))
    alpha, beta, gamma = angles.unbind(dim=-1)

    row0 = torch.stack([lengths[0], torch.zeros_like(lengths[0]), torch.zeros_like(lengths[0])])
    row1 = torch.stack([
        lengths[1] * torch.cos(gamma),
        lengths[1] * torch.sin(gamma).clamp_min(1.0e-8),
        torch.zeros_like(lengths[1]),
    ])
    cx = lengths[2] * torch.cos(beta)
    cy = lengths[2] * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma).clamp_min(1.0e-8)
    cz_sq = (lengths[2].square() - cx.square() - cy.square()).clamp_min(1.0e-12)
    row2 = torch.stack([cx, cy, torch.sqrt(cz_sq)])
    return torch.stack([row0, row1, row2], dim=0)


def raw_native_scaled_violation(
    *,
    lengths: torch.Tensor,
    angles_deg: torch.Tensor,
    projected_lengths: torch.Tensor,
    projected_angles_deg: torch.Tensor,
    angle_scale_deg: float = 10.0,
) -> torch.Tensor:
    length_scale = torch.mean(torch.as_tensor(lengths).reshape(3)).abs().clamp_min(1.0e-8)
    angle_scale = torch.as_tensor(float(angle_scale_deg), device=lengths.device, dtype=lengths.dtype).clamp_min(1.0e-8)
    length_term = torch.sum(((projected_lengths - lengths) / length_scale) ** 2)
    angle_term = torch.sum(((projected_angles_deg - angles_deg) / angle_scale) ** 2)
    return length_term + angle_term


def raw_native_project_lengths_angles(
    *,
    lengths: torch.Tensor,
    angles_deg: torch.Tensor,
    space_group_number: int,
    angle_scale_deg: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
    lengths = torch.as_tensor(lengths).reshape(3)
    angles = torch.as_tensor(angles_deg, device=lengths.device, dtype=lengths.dtype).reshape(3)
    family = spacegroup_to_crystal_family(int(space_group_number))
    lattice_family = "hexagonal" if family == "trigonal" else family
    ninety = torch.as_tensor(90.0, device=lengths.device, dtype=lengths.dtype)
    one_twenty = torch.as_tensor(120.0, device=lengths.device, dtype=lengths.dtype)

    if lattice_family == "triclinic":
        proj_lengths, proj_angles, variant = lengths, angles, "triclinic"
    elif lattice_family == "monoclinic":
        candidates: list[tuple[str, torch.Tensor]] = [
            ("unique_alpha", torch.stack([angles[0], ninety, ninety])),
            ("unique_beta", torch.stack([ninety, angles[1], ninety])),
            ("unique_gamma", torch.stack([ninety, ninety, angles[2]])),
        ]
        scored = [
            (
                raw_native_scaled_violation(
                    lengths=lengths,
                    angles_deg=angles,
                    projected_lengths=lengths,
                    projected_angles_deg=candidate_angles,
                    angle_scale_deg=angle_scale_deg,
                ),
                variant_name,
                candidate_angles,
            )
            for variant_name, candidate_angles in candidates
        ]
        violation, variant, proj_angles = min(scored, key=lambda item: float(item[0].detach().item()))
        proj_lengths = lengths
        return proj_lengths, proj_angles, variant, violation
    elif lattice_family == "orthorhombic":
        proj_lengths = lengths
        proj_angles = torch.stack([ninety, ninety, ninety])
        variant = "orthorhombic"
    elif lattice_family == "tetragonal":
        ab = torch.mean(lengths[:2])
        proj_lengths = torch.stack([ab, ab, lengths[2]])
        proj_angles = torch.stack([ninety, ninety, ninety])
        variant = "tetragonal"
    elif lattice_family == "hexagonal":
        ab = torch.mean(lengths[:2])
        proj_lengths = torch.stack([ab, ab, lengths[2]])
        proj_angles = torch.stack([ninety, ninety, one_twenty])
        variant = "hexagonal"
    elif lattice_family == "cubic":
        scale = torch.mean(lengths)
        proj_lengths = torch.stack([scale, scale, scale])
        proj_angles = torch.stack([ninety, ninety, ninety])
        variant = "cubic"
    else:
        raise ValueError(f"Unsupported raw-native lattice family: {lattice_family}")

    violation = raw_native_scaled_violation(
        lengths=lengths,
        angles_deg=angles,
        projected_lengths=proj_lengths,
        projected_angles_deg=proj_angles,
        angle_scale_deg=angle_scale_deg,
    )
    return proj_lengths, proj_angles, variant, violation


def raw_native_project_lattice6(
    ell: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
    angle_scale_deg: float = 10.0,
    shift_cap_scaled: float | None = None,
) -> tuple[torch.Tensor, RawNativeProjectionDiagnostics]:
    ell_t = torch.as_tensor(ell).reshape(-1)
    if ell_t.numel() != 6:
        raise ValueError(f"raw_native_project_lattice6 expects one 6D lattice vector, got {tuple(ell_t.shape)}.")

    cell = lattice6_to_matrix(ell_t, num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    lengths, angles = _cell_lengths_angles(cell)
    proj_lengths, proj_angles, variant, violation = raw_native_project_lengths_angles(
        lengths=lengths,
        angles_deg=angles,
        space_group_number=int(space_group_number),
        angle_scale_deg=float(angle_scale_deg),
    )
    scaled_shift_norm = torch.sqrt(violation.clamp_min(0.0))
    if shift_cap_scaled is not None and float(shift_cap_scaled) > 0.0:
        cap = torch.as_tensor(float(shift_cap_scaled), device=lengths.device, dtype=lengths.dtype)
        if bool(scaled_shift_norm > cap):
            ratio = cap / scaled_shift_norm.clamp_min(1.0e-12)
            proj_lengths = lengths + ratio * (proj_lengths - lengths)
            proj_angles = angles + ratio * (proj_angles - angles)
            violation = raw_native_scaled_violation(
                lengths=lengths,
                angles_deg=angles,
                projected_lengths=proj_lengths,
                projected_angles_deg=proj_angles,
                angle_scale_deg=float(angle_scale_deg),
            )
            scaled_shift_norm = torch.sqrt(violation.clamp_min(0.0))

    projected_cell = _lengths_angles_to_cell(proj_lengths, proj_angles)
    ell_raw = matrix_to_lattice6(projected_cell, num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    diagnostics = RawNativeProjectionDiagnostics(
        family=spacegroup_to_crystal_family(int(space_group_number)),
        selected_variant=variant,
        raw_violation_before=float(violation.detach().item()),
        raw_violation_after=0.0,
        scaled_shift_norm=float(scaled_shift_norm.detach().item()),
        gamma=1.0,
        weight=1.0,
        accepted=True,
        skipped_reason="",
        ell0_hat=ell_t.detach().clone(),
        ell0_raw=ell_raw.detach().clone(),
        ell0_soft=ell_raw.detach().clone(),
    )
    return ell_raw, diagnostics


def raw_native_cps_project_clean_estimate(
    ell0_hat: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
    tau: float,
    gamma_min: float = 0.05,
    gamma_max: float = 3.0,
    gamma_power: float = 2.0,
    projection_start_frac: float = 0.25,
    angle_scale_deg: float = 10.0,
    use_gate: bool = True,
    raw_violation_min: float = 1.0e-8,
    raw_violation_max: float = 4.0,
    shift_cap_scaled: float | None = 1.0,
    compute_diagnostics: bool = True,
) -> tuple[torch.Tensor, RawNativeProjectionDiagnostics]:
    ell0_hat_t = torch.as_tensor(ell0_hat).reshape(-1)
    ell0_raw, hard_diag = raw_native_project_lattice6(
        ell0_hat_t,
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
        space_group_number=int(space_group_number),
        angle_scale_deg=float(angle_scale_deg),
        shift_cap_scaled=shift_cap_scaled,
    )
    raw_violation = float(hard_diag.raw_violation_before)
    accepted = True
    skipped_reason = ""
    if bool(use_gate):
        if raw_violation < float(raw_violation_min):
            accepted = False
            skipped_reason = "below_min_violation"
        elif raw_violation > float(raw_violation_max):
            accepted = False
            skipped_reason = "above_max_violation"

    gamma, weight = cps_gamma_weight(
        tau=float(tau),
        projection_start_frac=float(projection_start_frac),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        gamma_power=float(gamma_power),
    )
    if accepted:
        ell0_soft = ell0_hat_t + float(weight) * (ell0_raw.to(device=ell0_hat_t.device, dtype=ell0_hat_t.dtype) - ell0_hat_t)
        _, after_diag = raw_native_project_lattice6(
            ell0_soft,
            num_atoms=int(num_atoms),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            angle_scale_deg=float(angle_scale_deg),
            shift_cap_scaled=None,
        )
        raw_violation_after = float(after_diag.raw_violation_before)
    else:
        ell0_soft = ell0_hat_t.detach().clone()
        raw_violation_after = raw_violation

    diagnostics = RawNativeProjectionDiagnostics(
        family=hard_diag.family,
        selected_variant=hard_diag.selected_variant,
        raw_violation_before=raw_violation,
        raw_violation_after=raw_violation_after,
        scaled_shift_norm=float(hard_diag.scaled_shift_norm),
        gamma=float(gamma),
        weight=float(weight) if accepted else 0.0,
        accepted=bool(accepted),
        skipped_reason=skipped_reason,
        ell0_hat=ell0_hat_t.detach().clone() if compute_diagnostics else ell0_hat_t.new_empty(0),
        ell0_raw=ell0_raw.detach().clone() if compute_diagnostics else ell0_raw.new_empty(0),
        ell0_soft=ell0_soft.detach().clone() if compute_diagnostics else ell0_soft.new_empty(0),
    )
    return ell0_soft, diagnostics


@torch.no_grad()
def kldm_pc_cps_raw_native_lattice_sampler(
    *,
    model: Any,
    batch: Any,
    lattice_transform: Any | None,
    oracle_space_group: int,
    config: Algorithm25RawNativeConfig = Algorithm25RawNativeConfig(),
    t_start: float = 1.0,
    t_final: float = 1.0e-6,
    debug_label: str | None = None,
    debug_print_every: int | None = None,
) -> RawNativeSamplerResult:
    debug_enabled = bool(debug_label)
    state = model._prepare_csp_sampling(
        batch=batch,
        n_steps=int(config.total_steps),
        t_start=float(t_start),
        t_final=float(t_final),
    )
    _require_single_graph_lattice_state(
        l_t=state["l_t"],
        num_atoms=state["batch"].num_atoms,
        context="kldm_pc_cps_raw_native_lattice_sampler",
    )
    if debug_enabled:
        print(
            f"[algo25-raw-pc] start label={debug_label} total_steps={int(config.total_steps)} "
            f"start_frac={float(config.projection_start_frac):.3f} interval={int(config.projection_interval)}",
            flush=True,
        )

    interventions: list[RawNativeIntervention] = []
    with torch.no_grad():
        for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
            remaining_step = int(config.total_steps) - int(times.step)
            tau = float(remaining_step) / float(max(int(config.total_steps), 1))
            completed_step = int(times.step) + 1
            if debug_enabled and debug_print_every is not None and int(debug_print_every) > 0:
                if completed_step == 1 or completed_step % int(debug_print_every) == 0 or remaining_step <= 1:
                    print(
                        f"[algo25-raw-pc] label={debug_label} progress completed={completed_step}/{int(config.total_steps)} "
                        f"remaining={remaining_step} tau={tau:.3f}",
                        flush=True,
                    )

            if should_project_step(
                remaining_step=remaining_step,
                total_steps=int(config.total_steps),
                projection_start_frac=float(config.projection_start_frac),
                projection_interval=int(config.projection_interval),
            ):
                preds_proj = model.score_network(
                    t=times.now.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )
                ell0_hat = predict_clean_lattice_from_prediction(
                    l_t=state["l_t"],
                    pred_l=preds_proj["l"],
                    t_lattice=times.now.lattice,
                    diffusion_l=state["sampling_diffusion_l"],
                    num_atoms=state["batch"].num_atoms,
                )
                ell0_soft, proj_diag = raw_native_cps_project_clean_estimate(
                    ell0_hat=ell0_hat,
                    num_atoms=int(state["batch"].num_atoms.reshape(-1)[0].item()),
                    lattice_transform=lattice_transform,
                    space_group_number=int(oracle_space_group),
                    tau=float(tau),
                    gamma_min=float(config.gamma_min),
                    gamma_max=float(config.gamma_max),
                    gamma_power=float(config.gamma_power),
                    projection_start_frac=float(config.projection_start_frac),
                    angle_scale_deg=float(config.angle_scale_deg),
                    use_gate=bool(config.use_gate),
                    raw_violation_min=float(config.raw_violation_min),
                    raw_violation_max=float(config.raw_violation_max),
                    shift_cap_scaled=float(config.shift_cap_scaled),
                    compute_diagnostics=bool(config.collect_projection_diagnostics),
                )
                if bool(proj_diag.accepted):
                    state["l_t"], update_diag = apply_lattice_cps_to_state(
                        ell_t=state["l_t"],
                        ell0_hat=ell0_hat,
                        ell0_soft=ell0_soft,
                        t_lattice=times.now.lattice,
                        diffusion_l=state["sampling_diffusion_l"],
                    )
                    interventions.append(
                        RawNativeIntervention(
                            remaining_step=int(remaining_step),
                            tau=float(tau),
                            family=proj_diag.family,
                            selected_variant=proj_diag.selected_variant,
                            gamma=float(proj_diag.gamma),
                            weight=float(proj_diag.weight),
                            raw_violation_before=float(proj_diag.raw_violation_before),
                            raw_violation_after=float(proj_diag.raw_violation_after),
                            scaled_shift_norm=float(proj_diag.scaled_shift_norm),
                            lattice_state_shift_norm=float(update_diag.lattice_state_shift_norm),
                        )
                    )
                    if debug_enabled:
                        print(
                            f"[algo25-raw-pc] label={debug_label} project remaining={remaining_step} "
                            f"family={proj_diag.family} variant={proj_diag.selected_variant} "
                            f"weight={proj_diag.weight:.4f} raw_before={proj_diag.raw_violation_before:.6g} "
                            f"raw_after={proj_diag.raw_violation_after:.6g} shift={update_diag.lattice_state_shift_norm:.6f}",
                            flush=True,
                        )
                elif debug_enabled:
                    print(
                        f"[algo25-raw-pc] label={debug_label} skip remaining={remaining_step} "
                        f"reason={proj_diag.skipped_reason} raw={proj_diag.raw_violation_before:.6g}",
                        flush=True,
                    )

            preds_curr = model.score_network(
                t=times.now.graph,
                pos=state["f_t"],
                v=state["v_t"],
                h=state["a_t"],
                l=state["l_t"],
                node_index=state["node_index"],
                edge_node_index=state["edge_node_index"],
            )
            state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_step_predictor(
                t=times.now.nodes,
                f_t=state["f_t"],
                v_t=state["v_t"],
                pred_v=preds_curr["v"],
                index=state["node_index"],
                dt=times.dt,
            )

            if times.t_next_float < 1.0e-3:
                continue

            preds_next = None
            for _ in range(max(int(config.n_correction_steps), 1)):
                preds_next = model.score_network(
                    t=times.next.graph,
                    pos=state["f_t"],
                    v=state["v_t"],
                    h=state["a_t"],
                    l=state["l_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                )
                state["f_t"], state["v_t"] = state["sampling_tdm"].reverse_step_corrector(
                    t=times.next.nodes,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    pred_v=preds_next["v"],
                    dt=times.dt,
                    index=state["node_index"],
                    tau=float(config.tau),
                )

            state["l_t"] = state["sampling_diffusion_l"].reverse_step(
                t=times.next.lattice,
                x_t=state["l_t"],
                pred=preds_next["l"],
                dt=times.dt,
                num_atoms=state["batch"].num_atoms,
            )

    if debug_enabled:
        print(f"[algo25-raw-pc] done label={debug_label} interventions={len(interventions)}", flush=True)
    return RawNativeSamplerResult(
        frac_coords=state["f_t"].detach().clone(),
        velocity=state["v_t"].detach().clone(),
        lattice=_restore_l_shape(state["l_t"].detach().clone(), ref=state["batch"].l),
        atom_types=state["a_t"].detach().clone(),
        interventions=tuple(interventions),
    )


__all__ = [
    "Algorithm25RawNativeConfig",
    "RawNativeIntervention",
    "RawNativeProjectionDiagnostics",
    "RawNativeSamplerResult",
    "kldm_pc_cps_raw_native_lattice_sampler",
    "raw_native_cps_project_clean_estimate",
    "raw_native_project_lattice6",
    "raw_native_project_lengths_angles",
    "raw_native_scaled_violation",
]
