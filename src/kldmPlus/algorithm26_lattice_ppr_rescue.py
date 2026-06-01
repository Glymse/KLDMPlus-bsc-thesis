from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from kldmPlus.algorithm22_faithful_kldm_cps_csp import wrap01
from kldmPlus.algorithm25_kldm_pc_cps_lattice import (
    _require_single_graph_lattice_state,
    _restore_l_shape,
    cps_lattice_project_clean_estimate,
    k_to_lattice6,
    lattice6_to_matrix,
    predict_clean_lattice_from_prediction,
    should_project_step,
)
from kldmPlus.algorithm25_raw_native_lattice_cps import raw_native_cps_project_clean_estimate
from kldmPlus.utils.time import iter_sampling_times


ProjectionMode = Literal["none", "raw_native", "kspace"]


@dataclass(frozen=True)
class Algorithm26PPRConfig:
    total_steps: int = 800
    projection_mode: ProjectionMode = "raw_native"
    projection_start_frac: float = 0.25
    projection_interval: int = 50
    rho_lattice: float = 0.75
    gamma_min: float = 0.05
    gamma_max: float = 3.0
    gamma_power: float = 2.0
    tau: float = 0.25
    n_correction_steps: int = 1
    preserve_lattice_volume: bool = False
    raw_angle_scale_deg: float = 10.0
    raw_shift_cap_scaled: float = 1.0
    raw_use_gate: bool = False
    raw_violation_min: float = 1.0e-8
    raw_violation_max: float = 4.0
    induced_small_angstrom: float = 0.05
    induced_large_angstrom: float = 0.25
    use_final_ppr: bool = False
    collect_projection_diagnostics: bool = False


@dataclass(frozen=True)
class Algorithm26CleanPrediction:
    f0_hat: torch.Tensor
    ell0_hat: torch.Tensor
    pred_l: torch.Tensor


@dataclass(frozen=True)
class Algorithm26Projection:
    ell0_hat: torch.Tensor
    ell0_projected: torch.Tensor
    ell0_soft: torch.Tensor
    projection_mode: str
    family: str
    weight: float
    accepted: bool
    skipped_reason: str
    clean_lattice_shift_norm: float
    relative_cell_shift: float
    induced_cart_shift_rms: float
    max_length_change_percent: float
    max_angle_change_deg: float
    violation_before: float
    violation_after: float
    zone: str


@dataclass(frozen=True)
class Algorithm26PPRUpdate:
    l_t: torch.Tensor
    eps_old: torch.Tensor
    eps_mix: torch.Tensor
    rho_lattice: float
    lattice_state_shift_norm: float
    projected_mean_shift_norm: float


@dataclass(frozen=True)
class Algorithm26PPRIntervention:
    remaining_step: int
    tau: float
    projection_mode: str
    family: str
    zone: str
    rho_lattice: float
    weight: float
    violation_before: float
    violation_after: float
    induced_cart_shift_rms: float
    relative_cell_shift: float
    lattice_state_shift_norm: float


@dataclass(frozen=True)
class Algorithm26PPRSamplerResult:
    frac_coords: torch.Tensor
    velocity: torch.Tensor
    lattice: torch.Tensor
    atom_types: torch.Tensor
    interventions: tuple[Algorithm26PPRIntervention, ...]


def _debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _as_batch_lattice(l: torch.Tensor) -> torch.Tensor:
    l_t = torch.as_tensor(l)
    if l_t.ndim == 1:
        return l_t.unsqueeze(0)
    return l_t


def _cell_shift_metrics(
    *,
    ell0_hat: torch.Tensor,
    ell0_projected: torch.Tensor,
    f0_hat: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
) -> dict[str, float]:
    cell_hat = lattice6_to_matrix(ell0_hat.reshape(-1), num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    cell_proj = lattice6_to_matrix(ell0_projected.reshape(-1), num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    delta_cell = cell_proj - cell_hat
    relative_cell_shift = float(
        (torch.linalg.norm(delta_cell.reshape(-1)) / torch.linalg.norm(cell_hat.reshape(-1)).clamp_min(1.0e-12)).detach().item()
    )
    induced = torch.as_tensor(f0_hat, device=cell_hat.device, dtype=cell_hat.dtype).reshape(-1, 3) @ delta_cell
    induced_cart_shift_rms = float(torch.sqrt(torch.mean(torch.sum(induced.square(), dim=-1))).detach().item())

    def _angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        denom = (torch.linalg.norm(u) * torch.linalg.norm(v)).clamp_min(1.0e-12)
        return torch.rad2deg(torch.acos(torch.clamp(torch.dot(u, v) / denom, -1.0, 1.0)))

    len_hat = torch.linalg.norm(cell_hat, dim=-1)
    len_proj = torch.linalg.norm(cell_proj, dim=-1)
    ang_hat = torch.stack([_angle(cell_hat[1], cell_hat[2]), _angle(cell_hat[0], cell_hat[2]), _angle(cell_hat[0], cell_hat[1])])
    ang_proj = torch.stack([_angle(cell_proj[1], cell_proj[2]), _angle(cell_proj[0], cell_proj[2]), _angle(cell_proj[0], cell_proj[1])])
    return {
        "relative_cell_shift": relative_cell_shift,
        "induced_cart_shift_rms": induced_cart_shift_rms,
        "max_length_change_percent": float((100.0 * torch.max(torch.abs(len_proj - len_hat) / len_hat.clamp_min(1.0e-12))).detach().item()),
        "max_angle_change_deg": float(torch.max(torch.abs(ang_proj - ang_hat)).detach().item()),
    }


def predict_clean_lattice_and_fractional(
    *,
    model: Any,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    l_t: torch.Tensor,
    atom_types: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    t_lattice: torch.Tensor,
    num_atoms: torch.Tensor,
) -> Algorithm26CleanPrediction:
    preds = model.score_network(
        t=t_graph,
        pos=f_t,
        v=v_t,
        h=atom_types,
        l=l_t,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )
    ell0_hat = predict_clean_lattice_from_prediction(
        l_t=l_t,
        pred_l=preds["l"],
        t_lattice=t_lattice,
        diffusion_l=model.diffusion_l,
        num_atoms=num_atoms,
    ).reshape(-1)
    score_v = model.tdm.reconstruct_full_reverse_velocity_score(
        t=t_nodes,
        v_t=v_t,
        pred_v=preds["v"],
        index=node_index,
    )
    f0_hat = wrap01(f_t + 0.25 * score_v)
    return Algorithm26CleanPrediction(f0_hat=f0_hat.detach().clone(), ell0_hat=ell0_hat.detach().clone(), pred_l=preds["l"].detach().clone())


def project_clean_lattice_for_ppr(
    *,
    ell0_hat: torch.Tensor,
    f0_hat: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
    tau: float,
    config: Algorithm26PPRConfig = Algorithm26PPRConfig(),
) -> Algorithm26Projection:
    mode = str(config.projection_mode)
    if mode in {"none", "no_projection"}:
        ell0_hat_t = torch.as_tensor(ell0_hat).reshape(-1)
        ell0_projected = ell0_hat_t
        ell0_soft = ell0_hat_t
        family = "none"
        weight = 0.0
        accepted = True
        skipped_reason = ""
        violation_before = 0.0
        violation_after = 0.0
    elif mode == "raw_native":
        ell0_soft, diag = raw_native_cps_project_clean_estimate(
            ell0_hat=ell0_hat,
            num_atoms=int(num_atoms),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            tau=float(tau),
            gamma_min=float(config.gamma_min),
            gamma_max=float(config.gamma_max),
            gamma_power=float(config.gamma_power),
            projection_start_frac=float(config.projection_start_frac),
            angle_scale_deg=float(config.raw_angle_scale_deg),
            use_gate=bool(config.raw_use_gate),
            raw_violation_min=float(config.raw_violation_min),
            raw_violation_max=float(config.raw_violation_max),
            shift_cap_scaled=float(config.raw_shift_cap_scaled),
        )
        ell0_projected = diag.ell0_raw if diag.ell0_raw.numel() else ell0_soft
        family = diag.family
        weight = diag.weight
        accepted = diag.accepted
        skipped_reason = diag.skipped_reason
        violation_before = diag.raw_violation_before
        violation_after = diag.raw_violation_after
    elif mode == "kspace":
        ell0_soft, diag = cps_lattice_project_clean_estimate(
            ell0_hat=ell0_hat,
            num_atoms=int(num_atoms),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            tau=float(tau),
            gamma_min=float(config.gamma_min),
            gamma_max=float(config.gamma_max),
            gamma_power=float(config.gamma_power),
            projection_start_frac=float(config.projection_start_frac),
            preserve_lattice_volume=bool(config.preserve_lattice_volume),
        )
        ell0_projected = k_to_lattice6(
            diag.k_projected,
            num_atoms=int(num_atoms),
            lattice_transform=lattice_transform,
        )
        family = diag.family
        weight = diag.weight
        accepted = True
        skipped_reason = ""
        violation_before = diag.k_violation_before
        violation_after = diag.k_violation_after
    else:
        raise ValueError(f"Unknown Algorithm26 projection_mode={mode!r}.")

    shift_metrics = _cell_shift_metrics(
        ell0_hat=ell0_hat,
        ell0_projected=ell0_projected,
        f0_hat=f0_hat,
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    )
    induced = shift_metrics["induced_cart_shift_rms"]
    if induced < float(config.induced_small_angstrom):
        zone = "small"
    elif induced <= float(config.induced_large_angstrom):
        zone = "medium"
    else:
        zone = "large"
    return Algorithm26Projection(
        ell0_hat=ell0_hat.detach().clone().reshape(-1),
        ell0_projected=ell0_projected.detach().clone().reshape(-1),
        ell0_soft=ell0_soft.detach().clone().reshape(-1),
        projection_mode=mode,
        family=family,
        weight=float(weight),
        accepted=bool(accepted),
        skipped_reason=skipped_reason,
        clean_lattice_shift_norm=float(torch.linalg.norm((ell0_projected.reshape(-1) - ell0_hat.reshape(-1))).detach().item()),
        relative_cell_shift=float(shift_metrics["relative_cell_shift"]),
        induced_cart_shift_rms=float(induced),
        max_length_change_percent=float(shift_metrics["max_length_change_percent"]),
        max_angle_change_deg=float(shift_metrics["max_angle_change_deg"]),
        violation_before=float(violation_before),
        violation_after=float(violation_after),
        zone=zone,
    )


def lattice_forward_mean_sigma(
    *,
    diffusion_l: Any,
    t_lattice: torch.Tensor,
    x0: torch.Tensor,
    num_atoms: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0_b = _as_batch_lattice(x0)
    alpha_t = diffusion_l._match_dims(diffusion_l.alpha(t_lattice), x0_b)
    sigma_t = diffusion_l._match_dims(diffusion_l.sigma(t_lattice), x0_b)
    mean_t = alpha_t * x0_b
    return mean_t, sigma_t


def lattice_ppr_renoise_from_clean(
    *,
    ell_t: torch.Tensor,
    ell0_hat: torch.Tensor,
    ell0_projected: torch.Tensor,
    t_lattice: torch.Tensor,
    diffusion_l: Any,
    num_atoms: torch.Tensor | None,
    rho_lattice: float = 0.75,
    noise: torch.Tensor | None = None,
) -> Algorithm26PPRUpdate:
    ell_t_ref = torch.as_tensor(ell_t)
    ell_t_b = _as_batch_lattice(ell_t_ref)
    ell0_hat_b = _as_batch_lattice(torch.as_tensor(ell0_hat, device=ell_t_b.device, dtype=ell_t_b.dtype))
    ell0_proj_b = _as_batch_lattice(torch.as_tensor(ell0_projected, device=ell_t_b.device, dtype=ell_t_b.dtype))
    mean_hat, sigma_t = lattice_forward_mean_sigma(
        diffusion_l=diffusion_l,
        t_lattice=t_lattice,
        x0=ell0_hat_b,
        num_atoms=num_atoms,
    )
    mean_proj, _ = lattice_forward_mean_sigma(
        diffusion_l=diffusion_l,
        t_lattice=t_lattice,
        x0=ell0_proj_b,
        num_atoms=num_atoms,
    )
    eps_old = (ell_t_b - mean_hat) / sigma_t.clamp_min(getattr(diffusion_l, "eps", 1.0e-8))
    if noise is None:
        noise = torch.randn_like(eps_old)
    rho = max(0.0, min(1.0, float(rho_lattice)))
    eps_mix = rho * eps_old + (max(0.0, 1.0 - rho * rho) ** 0.5) * noise.to(device=eps_old.device, dtype=eps_old.dtype)
    ell_new = mean_proj + sigma_t * eps_mix
    return Algorithm26PPRUpdate(
        l_t=_restore_l_shape(ell_new.detach().clone(), ref=ell_t_ref),
        eps_old=eps_old.detach().clone(),
        eps_mix=eps_mix.detach().clone(),
        rho_lattice=float(rho),
        lattice_state_shift_norm=float(torch.linalg.norm((ell_new - ell_t_b).reshape(-1)).detach().item()),
        projected_mean_shift_norm=float(torch.linalg.norm((mean_proj - mean_hat).reshape(-1)).detach().item()),
    )


def apply_lattice_ppr_projection(
    *,
    ell_t: torch.Tensor,
    t_lattice: torch.Tensor,
    diffusion_l: Any,
    num_atoms: torch.Tensor | None,
    projection: Algorithm26Projection,
    rho_lattice: float = 0.75,
    use_soft_projected_clean: bool = True,
    noise: torch.Tensor | None = None,
) -> Algorithm26PPRUpdate:
    target = projection.ell0_soft if bool(use_soft_projected_clean) else projection.ell0_projected
    return lattice_ppr_renoise_from_clean(
        ell_t=ell_t,
        ell0_hat=projection.ell0_hat,
        ell0_projected=target,
        t_lattice=t_lattice,
        diffusion_l=diffusion_l,
        num_atoms=num_atoms,
        rho_lattice=float(rho_lattice),
        noise=noise,
    )


@torch.no_grad()
def kldm_pc_ppr_lattice_sampler(
    *,
    model: Any,
    batch: Any,
    lattice_transform: Any | None,
    oracle_space_group: int,
    config: Algorithm26PPRConfig = Algorithm26PPRConfig(),
    t_start: float = 1.0,
    t_final: float = 1.0e-6,
    debug_label: str | None = None,
    debug_print_every: int | None = None,
) -> Algorithm26PPRSamplerResult:
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
        context="kldm_pc_ppr_lattice_sampler",
    )
    _debug_print(
        debug_enabled,
        f"[algo26-ppr-pc] start label={debug_label} total_steps={int(config.total_steps)} "
        f"mode={config.projection_mode} rho={float(config.rho_lattice):.2f} "
        f"start_frac={float(config.projection_start_frac):.3f} interval={int(config.projection_interval)}",
    )

    interventions: list[Algorithm26PPRIntervention] = []
    with torch.no_grad():
        for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
            remaining_step = int(config.total_steps) - int(times.step)
            tau = float(remaining_step) / float(max(int(config.total_steps), 1))
            completed_step = int(times.step) + 1
            if debug_print_every is not None and int(debug_print_every) > 0:
                if completed_step == 1 or completed_step % int(debug_print_every) == 0 or remaining_step <= 1:
                    _debug_print(
                        debug_enabled,
                        f"[algo26-ppr-pc] label={debug_label} progress completed={completed_step}/{int(config.total_steps)} "
                        f"remaining={remaining_step} tau={tau:.3f}",
                    )

            if should_project_step(
                remaining_step=remaining_step,
                total_steps=int(config.total_steps),
                projection_start_frac=float(config.projection_start_frac),
                projection_interval=int(config.projection_interval),
            ):
                clean = predict_clean_lattice_and_fractional(
                    model=model,
                    f_t=state["f_t"],
                    v_t=state["v_t"],
                    l_t=state["l_t"],
                    atom_types=state["a_t"],
                    node_index=state["node_index"],
                    edge_node_index=state["edge_node_index"],
                    t_graph=times.now.graph,
                    t_nodes=times.now.nodes,
                    t_lattice=times.now.lattice,
                    num_atoms=state["batch"].num_atoms,
                )
                projection = project_clean_lattice_for_ppr(
                    ell0_hat=clean.ell0_hat,
                    f0_hat=clean.f0_hat,
                    num_atoms=int(state["batch"].num_atoms.reshape(-1)[0].item()),
                    lattice_transform=lattice_transform,
                    space_group_number=int(oracle_space_group),
                    tau=float(tau),
                    config=config,
                )
                if projection.accepted:
                    ppr = apply_lattice_ppr_projection(
                        ell_t=state["l_t"],
                        t_lattice=times.now.lattice,
                        diffusion_l=state["sampling_diffusion_l"],
                        num_atoms=state["batch"].num_atoms,
                        projection=projection,
                        rho_lattice=float(config.rho_lattice),
                        use_soft_projected_clean=True,
                    )
                    state["l_t"] = _as_batch_lattice(ppr.l_t)
                    interventions.append(
                        Algorithm26PPRIntervention(
                            remaining_step=int(remaining_step),
                            tau=float(tau),
                            projection_mode=str(config.projection_mode),
                            family=projection.family,
                            zone=projection.zone,
                            rho_lattice=float(ppr.rho_lattice),
                            weight=float(projection.weight),
                            violation_before=float(projection.violation_before),
                            violation_after=float(projection.violation_after),
                            induced_cart_shift_rms=float(projection.induced_cart_shift_rms),
                            relative_cell_shift=float(projection.relative_cell_shift),
                            lattice_state_shift_norm=float(ppr.lattice_state_shift_norm),
                        )
                    )
                    _debug_print(
                        debug_enabled,
                        f"[algo26-ppr-pc] label={debug_label} ppr remaining={remaining_step} "
                        f"mode={config.projection_mode} zone={projection.zone} rho={ppr.rho_lattice:.2f} "
                        f"weight={projection.weight:.4f} shift={ppr.lattice_state_shift_norm:.6f}",
                    )
                else:
                    _debug_print(
                        debug_enabled,
                        f"[algo26-ppr-pc] label={debug_label} skip remaining={remaining_step} "
                        f"reason={projection.skipped_reason}",
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

    _debug_print(debug_enabled, f"[algo26-ppr-pc] done label={debug_label} interventions={len(interventions)}")
    return Algorithm26PPRSamplerResult(
        frac_coords=state["f_t"].detach().clone(),
        velocity=state["v_t"].detach().clone(),
        lattice=_restore_l_shape(state["l_t"].detach().clone(), ref=state["batch"].l),
        atom_types=state["a_t"].detach().clone(),
        interventions=tuple(interventions),
    )


__all__ = [
    "Algorithm26CleanPrediction",
    "Algorithm26PPRConfig",
    "Algorithm26PPRIntervention",
    "Algorithm26PPRSamplerResult",
    "Algorithm26PPRUpdate",
    "Algorithm26Projection",
    "apply_lattice_ppr_projection",
    "kldm_pc_ppr_lattice_sampler",
    "lattice_forward_mean_sigma",
    "lattice_ppr_renoise_from_clean",
    "predict_clean_lattice_and_fractional",
    "project_clean_lattice_for_ppr",
]
