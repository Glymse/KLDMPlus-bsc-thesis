from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kldmPlus.algorithm10_casal_chart import _decode_lattice_matrix, _encode_lattice_matrix
from kldmPlus.symmetry.k_basis import cell_to_k, k_to_cell_matrix, space_group_k_constraint
from kldmPlus.utils.time import iter_sampling_times


def _as_graph_l_batch(l: torch.Tensor) -> torch.Tensor:
    if l.ndim == 1:
        return l.unsqueeze(0)
    return l


def _restore_l_shape(l: torch.Tensor, *, ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim == 1 and l.ndim == 2 and l.shape[0] == 1:
        return l[0]
    return l


def _require_single_graph_lattice_state(*, l_t: torch.Tensor, num_atoms: torch.Tensor, context: str) -> None:
    """Algorithm25 projection is intentionally single-graph until per-graph CPS is added."""
    l_batch = _as_graph_l_batch(torch.as_tensor(l_t))
    num_atoms_t = torch.as_tensor(num_atoms).reshape(-1)
    if l_batch.shape[0] != 1 or num_atoms_t.numel() != 1:
        raise ValueError(
            f"{context} currently supports only batch size 1. "
            f"Got lattice_batch={tuple(l_batch.shape)} num_atoms_shape={tuple(num_atoms_t.shape)}."
        )


def _debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


@dataclass(frozen=True)
class Algorithm25Config:
    total_steps: int = 800
    projection_start_frac: float = 0.5
    projection_interval: int = 50
    gamma_min: float = 0.05
    gamma_max: float = 10.0
    gamma_power: float = 2.0
    use_final_projection: bool = False
    preserve_lattice_volume: bool = False
    tau: float = 0.25
    n_correction_steps: int = 1
    collect_projection_diagnostics: bool = False


@dataclass(frozen=True)
class Algorithm25ProjectionDiagnostics:
    family: str
    lattice_family: str
    gamma: float
    weight: float
    tau: float
    k_hat: torch.Tensor
    k_projected: torch.Tensor
    k_soft: torch.Tensor
    k_violation_before: float
    k_violation_after: float
    ell0_hat: torch.Tensor
    ell0_soft: torch.Tensor


@dataclass(frozen=True)
class Algorithm25StateUpdateDiagnostics:
    alpha_t: float
    lattice_state_shift_norm: float
    ell_t_before: torch.Tensor
    ell_t_after: torch.Tensor


@dataclass(frozen=True)
class Algorithm25Intervention:
    remaining_step: int
    tau: float
    family: str
    lattice_family: str
    gamma: float
    weight: float
    k_violation_before: float
    k_violation_after: float
    lattice_state_shift_norm: float


@dataclass(frozen=True)
class Algorithm25SamplerResult:
    frac_coords: torch.Tensor
    velocity: torch.Tensor
    lattice: torch.Tensor
    atom_types: torch.Tensor
    interventions: tuple[Algorithm25Intervention, ...]


def lattice6_to_matrix(
    ell: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    ell_t = torch.as_tensor(ell)
    return _decode_lattice_matrix(
        l=ell_t.reshape(-1),
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    ).reshape(3, 3)


def matrix_to_lattice6(
    matrix: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    matrix_t = torch.as_tensor(matrix).reshape(3, 3)
    return _encode_lattice_matrix(
        cell_matrix=matrix_t,
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    ).reshape(-1)


def lattice_matrix_to_k(matrix: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    return cell_to_k(torch.as_tensor(matrix).reshape(3, 3), eps=eps).reshape(-1)


def k_to_lattice_matrix(k: torch.Tensor) -> torch.Tensor:
    return k_to_cell_matrix(torch.as_tensor(k).reshape(-1)).reshape(3, 3)


def lattice6_to_k(
    ell: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    return lattice_matrix_to_k(
        lattice6_to_matrix(ell, num_atoms=num_atoms, lattice_transform=lattice_transform),
        eps=eps,
    )


def k_to_lattice6(
    k: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    return matrix_to_lattice6(
        k_to_lattice_matrix(k),
        num_atoms=num_atoms,
        lattice_transform=lattice_transform,
    )


def spacegroup_to_crystal_family(space_group_number: int) -> str:
    sg = int(space_group_number)
    if not 1 <= sg <= 230:
        raise ValueError(f"space_group_number must be in [1, 230], got {sg}.")
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    return "cubic"


def _spacegroup_to_lattice_family(space_group_number: int) -> str:
    family = spacegroup_to_crystal_family(space_group_number)
    return "hexagonal" if family == "trigonal" else family


def project_k_to_spacegroup_family(
    k: torch.Tensor,
    *,
    space_group_number: int,
    preserve_lattice_volume: bool = False,
) -> torch.Tensor:
    k_t = torch.as_tensor(k).reshape(-1)
    constraint = space_group_k_constraint(
        space_group_number=int(space_group_number),
        device=k_t.device,
        dtype=k_t.dtype,
    )
    k_proj = (1.0 - constraint.mask) * k_t + constraint.mask * constraint.target
    if preserve_lattice_volume:
        k_proj[..., 5] = k_t[..., 5]
    return k_proj


def k_family_violation(
    k: torch.Tensor,
    *,
    space_group_number: int,
) -> float:
    k_t = torch.as_tensor(k).reshape(-1)
    constraint = space_group_k_constraint(
        space_group_number=int(space_group_number),
        device=k_t.device,
        dtype=k_t.dtype,
    )
    return float(torch.linalg.norm((constraint.mask * (k_t - constraint.target)).reshape(-1)).detach().item())


def cps_gamma_weight(
    *,
    tau: float,
    projection_start_frac: float = 0.5,
    gamma_min: float = 0.05,
    gamma_max: float = 10.0,
    gamma_power: float = 2.0,
) -> tuple[float, float]:
    start = float(max(projection_start_frac, 1.0e-8))
    u = max(0.0, min(1.0, (start - float(tau)) / start))
    gamma = float(gamma_min) * (float(gamma_max) / float(gamma_min)) ** (u ** float(gamma_power))
    weight = gamma / (1.0 + gamma)
    return float(gamma), float(weight)


def cps_lattice_project_clean_estimate(
    ell0_hat: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
    tau: float,
    gamma_min: float = 0.05,
    gamma_max: float = 10.0,
    gamma_power: float = 2.0,
    projection_start_frac: float = 0.5,
    preserve_lattice_volume: bool = False,
    compute_diagnostics: bool = True,
) -> tuple[torch.Tensor, Algorithm25ProjectionDiagnostics]:
    ell0_hat_raw = torch.as_tensor(ell0_hat)
    if ell0_hat_raw.ndim > 1 and ell0_hat_raw.shape[0] != 1:
        raise ValueError(
            "cps_lattice_project_clean_estimate currently supports only one lattice at a time. "
            f"Got ell0_hat shape={tuple(ell0_hat_raw.shape)}."
        )
    ell0_hat_t = ell0_hat_raw.reshape(-1)
    if ell0_hat_t.numel() != 6:
        raise ValueError(
            "cps_lattice_project_clean_estimate expects one 6D lattice feature vector. "
            f"Got flattened shape={tuple(ell0_hat_t.shape)}."
        )
    k_hat = lattice6_to_k(
        ell0_hat_t,
        num_atoms=num_atoms,
        lattice_transform=lattice_transform,
    )
    k_proj = project_k_to_spacegroup_family(
        k_hat,
        space_group_number=space_group_number,
        preserve_lattice_volume=preserve_lattice_volume,
    )
    gamma, weight = cps_gamma_weight(
        tau=float(tau),
        projection_start_frac=projection_start_frac,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_power=gamma_power,
    )
    k_soft = (1.0 - weight) * k_hat + weight * k_proj
    if preserve_lattice_volume:
        k_soft[..., 5] = k_hat[..., 5]
    ell0_soft = k_to_lattice6(
        k_soft,
        num_atoms=num_atoms,
        lattice_transform=lattice_transform,
    )
    if compute_diagnostics:
        k_violation_before = k_family_violation(k_hat, space_group_number=space_group_number)
        k_violation_after = k_family_violation(k_soft, space_group_number=space_group_number)
    else:
        k_violation_before = float("nan")
        k_violation_after = float("nan")

    diagnostics = Algorithm25ProjectionDiagnostics(
        family=spacegroup_to_crystal_family(space_group_number),
        lattice_family=_spacegroup_to_lattice_family(space_group_number),
        gamma=float(gamma),
        weight=float(weight),
        tau=float(tau),
        k_hat=k_hat.detach().clone() if compute_diagnostics else k_hat.new_empty(0),
        k_projected=k_proj.detach().clone() if compute_diagnostics else k_proj.new_empty(0),
        k_soft=k_soft.detach().clone() if compute_diagnostics else k_soft.new_empty(0),
        k_violation_before=k_violation_before,
        k_violation_after=k_violation_after,
        ell0_hat=ell0_hat_t.detach().clone() if compute_diagnostics else ell0_hat_t.new_empty(0),
        ell0_soft=ell0_soft.detach().clone() if compute_diagnostics else ell0_soft.new_empty(0),
    )
    return ell0_soft, diagnostics


def predict_clean_lattice_from_prediction(
    *,
    l_t: torch.Tensor,
    pred_l: torch.Tensor,
    t_lattice: torch.Tensor,
    diffusion_l: Any,
    num_atoms: torch.Tensor | None,
) -> torch.Tensor:
    l_ref = torch.as_tensor(l_t)
    l_batch = _as_graph_l_batch(l_ref)
    pred_batch = _as_graph_l_batch(torch.as_tensor(pred_l, device=l_batch.device, dtype=l_batch.dtype))

    if getattr(diffusion_l, "parameterization", "eps") == "x0":
        return _restore_l_shape(pred_batch.detach().clone(), ref=l_ref)

    alpha_t = diffusion_l._match_dims(diffusion_l.alpha(t_lattice), l_batch)

    sigma_t = diffusion_l._match_dims(diffusion_l.sigma(t_lattice), l_batch)
    x0_hat = (l_batch - sigma_t * pred_batch) / alpha_t.clamp_min(diffusion_l.eps)

    return _restore_l_shape(x0_hat.detach().clone(), ref=l_ref)


def apply_lattice_cps_to_state(
    *,
    ell_t: torch.Tensor,
    ell0_hat: torch.Tensor,
    ell0_soft: torch.Tensor,
    t_lattice: torch.Tensor,
    diffusion_l: Any,
) -> tuple[torch.Tensor, Algorithm25StateUpdateDiagnostics]:
    ell_t_ref = torch.as_tensor(ell_t)
    ell_t_batch = _as_graph_l_batch(ell_t_ref)
    ell0_hat_batch = _as_graph_l_batch(torch.as_tensor(ell0_hat, device=ell_t_batch.device, dtype=ell_t_batch.dtype))
    ell0_soft_batch = _as_graph_l_batch(torch.as_tensor(ell0_soft, device=ell_t_batch.device, dtype=ell_t_batch.dtype))

    alpha_t = diffusion_l._match_dims(diffusion_l.alpha(t_lattice), ell_t_batch)
    ell_t_cps = ell_t_batch + alpha_t * (ell0_soft_batch - ell0_hat_batch)
    ell_t_out = _restore_l_shape(ell_t_cps.detach().clone(), ref=ell_t_ref)
    diagnostics = Algorithm25StateUpdateDiagnostics(
        alpha_t=float(alpha_t.reshape(-1)[0].detach().item()),
        lattice_state_shift_norm=float(torch.linalg.norm((ell_t_cps - ell_t_batch).reshape(-1)).detach().item()),
        ell_t_before=ell_t_batch.detach().clone(),
        ell_t_after=ell_t_cps.detach().clone(),
    )
    return ell_t_out, diagnostics


def raw_length_angle_family_projection(
    ell: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
) -> torch.Tensor:
    cell = lattice6_to_matrix(ell, num_atoms=num_atoms, lattice_transform=lattice_transform)
    lengths = torch.linalg.norm(cell, dim=1)
    angles = []
    for (u, v) in ((1, 2), (0, 2), (0, 1)):
        angles.append(
            torch.acos(
                torch.clamp(
                    torch.dot(cell[u], cell[v]) / (lengths[u] * lengths[v]).clamp_min(1.0e-8),
                    -1.0,
                    1.0,
                )
            )
        )
    angles = torch.stack(angles)
    family = _spacegroup_to_lattice_family(space_group_number)

    if family == "orthorhombic":
        angles = torch.full_like(angles, torch.pi / 2.0)
    elif family == "tetragonal":
        lengths = torch.stack([lengths[:2].mean(), lengths[:2].mean(), lengths[2]])
        angles = torch.full_like(angles, torch.pi / 2.0)
    elif family == "hexagonal":
        lengths = torch.stack([lengths[:2].mean(), lengths[:2].mean(), lengths[2]])
        angles = torch.stack([
            torch.as_tensor(torch.pi / 2.0, device=angles.device, dtype=angles.dtype),
            torch.as_tensor(torch.pi / 2.0, device=angles.device, dtype=angles.dtype),
            torch.as_tensor(2.0 * torch.pi / 3.0, device=angles.device, dtype=angles.dtype),
        ])
    elif family == "cubic":
        mean_length = lengths.mean()
        lengths = torch.full_like(lengths, mean_length)
        angles = torch.full_like(angles, torch.pi / 2.0)
    elif family == "monoclinic":
        angles = torch.stack([
            torch.as_tensor(torch.pi / 2.0, device=angles.device, dtype=angles.dtype),
            angles[1],
            torch.as_tensor(torch.pi / 2.0, device=angles.device, dtype=angles.dtype),
        ])

    row0 = torch.stack([lengths[0], torch.zeros_like(lengths[0]), torch.zeros_like(lengths[0])])
    row1 = torch.stack([
        lengths[1] * torch.cos(angles[2]),
        lengths[1] * torch.sin(angles[2]).clamp_min(1.0e-8),
        torch.zeros_like(lengths[1]),
    ])
    cx = lengths[2] * torch.cos(angles[1])
    cy = lengths[2] * (torch.cos(angles[0]) - torch.cos(angles[1]) * torch.cos(angles[2])) / torch.sin(angles[2]).clamp_min(1.0e-8)
    cz_sq = (lengths[2].square() - cx.square() - cy.square()).clamp_min(1.0e-12)
    row2 = torch.stack([cx, cy, torch.sqrt(cz_sq)])
    projected_cell = torch.stack([row0, row1, row2], dim=0)
    return matrix_to_lattice6(projected_cell, num_atoms=num_atoms, lattice_transform=lattice_transform)


def should_project_step(
    *,
    remaining_step: int,
    total_steps: int,
    projection_start_frac: float,
    projection_interval: int,
) -> bool:
    if int(remaining_step) <= 0:
        return False
    tau = float(remaining_step) / float(max(int(total_steps), 1))
    return bool(tau <= float(projection_start_frac) and int(remaining_step) % int(projection_interval) == 0)


@torch.no_grad()
def kldm_pc_cps_lattice_sampler(
    *,
    model: Any,
    batch: Any,
    lattice_transform: Any | None,
    oracle_space_group: int,
    config: Algorithm25Config = Algorithm25Config(),
    t_start: float = 1.0,
    t_final: float = 1.0e-6,
    debug_label: str | None = None,
    debug_print_every: int | None = None,
) -> Algorithm25SamplerResult:
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
        context="kldm_pc_cps_lattice_sampler",
    )
    _debug_print(
        debug_enabled,
        f"[algo25-pc] start label={debug_label} total_steps={int(config.total_steps)} "
        f"start_frac={float(config.projection_start_frac):.3f} interval={int(config.projection_interval)}",
    )

    interventions: list[Algorithm25Intervention] = []
    with torch.no_grad():
        for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
            remaining_step = int(config.total_steps) - int(times.step)
            tau = float(remaining_step) / float(max(int(config.total_steps), 1))
            completed_step = int(times.step) + 1
            if debug_print_every is not None and int(debug_print_every) > 0:
                if completed_step == 1 or completed_step % int(debug_print_every) == 0 or remaining_step <= 1:
                    _debug_print(
                        debug_enabled,
                        f"[algo25-pc] label={debug_label} progress completed={completed_step}/{int(config.total_steps)} "
                        f"remaining={remaining_step} tau={tau:.3f}",
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
                ell0_soft, proj_diag = cps_lattice_project_clean_estimate(
                    ell0_hat=ell0_hat,
                    num_atoms=int(state["batch"].num_atoms.reshape(-1)[0].item()),
                    lattice_transform=lattice_transform,
                    space_group_number=int(oracle_space_group),
                    tau=float(tau),
                    gamma_min=float(config.gamma_min),
                    gamma_max=float(config.gamma_max),
                    gamma_power=float(config.gamma_power),
                    projection_start_frac=float(config.projection_start_frac),
                    preserve_lattice_volume=bool(config.preserve_lattice_volume),
                    compute_diagnostics=bool(config.collect_projection_diagnostics),
                )
                state["l_t"], update_diag = apply_lattice_cps_to_state(
                    ell_t=state["l_t"],
                    ell0_hat=ell0_hat,
                    ell0_soft=ell0_soft,
                    t_lattice=times.now.lattice,
                    diffusion_l=state["sampling_diffusion_l"],
                )
                interventions.append(
                    Algorithm25Intervention(
                        remaining_step=int(remaining_step),
                        tau=float(tau),
                        family=proj_diag.family,
                        lattice_family=proj_diag.lattice_family,
                        gamma=float(proj_diag.gamma),
                        weight=float(proj_diag.weight),
                        k_violation_before=float(proj_diag.k_violation_before),
                        k_violation_after=float(proj_diag.k_violation_after),
                        lattice_state_shift_norm=float(update_diag.lattice_state_shift_norm),
                    )
                )
                _debug_print(
                    debug_enabled,
                    f"[algo25-pc] label={debug_label} project remaining={remaining_step} tau={tau:.3f} "
                    f"weight={float(proj_diag.weight):.4f} shift={float(update_diag.lattice_state_shift_norm):.6f}",
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

            if times.t_next_float < 1e-3:
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

        if bool(config.use_final_projection):
            final_preds = model.score_network(
                t=state["sampling_time_grid"][-1].graph,
                pos=state["f_t"],
                v=state["v_t"],
                h=state["a_t"],
                l=state["l_t"],
                node_index=state["node_index"],
                edge_node_index=state["edge_node_index"],
            )
            final_ell0_hat = predict_clean_lattice_from_prediction(
                l_t=state["l_t"],
                pred_l=final_preds["l"],
                t_lattice=state["sampling_time_grid"][-1].lattice,
                diffusion_l=state["sampling_diffusion_l"],
                num_atoms=state["batch"].num_atoms,
            )
            final_ell0_soft, _ = cps_lattice_project_clean_estimate(
                ell0_hat=final_ell0_hat,
                num_atoms=int(state["batch"].num_atoms.reshape(-1)[0].item()),
                lattice_transform=lattice_transform,
                space_group_number=int(oracle_space_group),
                tau=0.0,
                gamma_min=float(config.gamma_min),
                gamma_max=float(config.gamma_max),
                gamma_power=float(config.gamma_power),
                projection_start_frac=float(config.projection_start_frac),
                preserve_lattice_volume=bool(config.preserve_lattice_volume),
                compute_diagnostics=bool(config.collect_projection_diagnostics),
            )
            state["l_t"], _ = apply_lattice_cps_to_state(
                ell_t=state["l_t"],
                ell0_hat=final_ell0_hat,
                ell0_soft=final_ell0_soft,
                t_lattice=state["sampling_time_grid"][-1].lattice,
                diffusion_l=state["sampling_diffusion_l"],
            )
            state["l_t"] = _as_graph_l_batch(state["l_t"])

    _debug_print(
        debug_enabled,
        f"[algo25-pc] done label={debug_label} interventions={len(interventions)}",
    )
    return Algorithm25SamplerResult(
        frac_coords=state["f_t"].detach().clone(),
        velocity=state["v_t"].detach().clone(),
        lattice=_restore_l_shape(state["l_t"].detach().clone(), ref=state["batch"].l),
        atom_types=state["a_t"].detach().clone(),
        interventions=tuple(interventions),
    )


@torch.no_grad()
def kldm_em_cps_lattice_sampler(
    *,
    model: Any,
    batch: Any,
    lattice_transform: Any | None,
    oracle_space_group: int,
    config: Algorithm25Config = Algorithm25Config(total_steps=1000),
    t_start: float = 1.0,
    t_final: float = 1.0e-6,
    debug_label: str | None = None,
    debug_print_every: int | None = None,
) -> Algorithm25SamplerResult:
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
        context="kldm_em_cps_lattice_sampler",
    )
    _debug_print(
        debug_enabled,
        f"[algo25-em] start label={debug_label} total_steps={int(config.total_steps)} "
        f"start_frac={float(config.projection_start_frac):.3f} interval={int(config.projection_interval)}",
    )

    interventions: list[Algorithm25Intervention] = []
    with torch.no_grad():
        for times in iter_sampling_times(batch=state["batch"], grid=state["sampling_time_grid"]):
            remaining_step = int(config.total_steps) - int(times.step)
            tau = float(remaining_step) / float(max(int(config.total_steps), 1))
            completed_step = int(times.step) + 1
            if debug_print_every is not None and int(debug_print_every) > 0:
                if completed_step == 1 or completed_step % int(debug_print_every) == 0 or remaining_step <= 1:
                    _debug_print(
                        debug_enabled,
                        f"[algo25-em] label={debug_label} progress completed={completed_step}/{int(config.total_steps)} "
                        f"remaining={remaining_step} tau={tau:.3f}",
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
                ell0_soft, proj_diag = cps_lattice_project_clean_estimate(
                    ell0_hat=ell0_hat,
                    num_atoms=int(state["batch"].num_atoms.reshape(-1)[0].item()),
                    lattice_transform=lattice_transform,
                    space_group_number=int(oracle_space_group),
                    tau=float(tau),
                    gamma_min=float(config.gamma_min),
                    gamma_max=float(config.gamma_max),
                    gamma_power=float(config.gamma_power),
                    projection_start_frac=float(config.projection_start_frac),
                    preserve_lattice_volume=bool(config.preserve_lattice_volume),
                    compute_diagnostics=bool(config.collect_projection_diagnostics),
                )
                state["l_t"], update_diag = apply_lattice_cps_to_state(
                    ell_t=state["l_t"],
                    ell0_hat=ell0_hat,
                    ell0_soft=ell0_soft,
                    t_lattice=times.now.lattice,
                    diffusion_l=state["sampling_diffusion_l"],
                )
                interventions.append(
                    Algorithm25Intervention(
                        remaining_step=int(remaining_step),
                        tau=float(tau),
                        family=proj_diag.family,
                        lattice_family=proj_diag.lattice_family,
                        gamma=float(proj_diag.gamma),
                        weight=float(proj_diag.weight),
                        k_violation_before=float(proj_diag.k_violation_before),
                        k_violation_after=float(proj_diag.k_violation_after),
                        lattice_state_shift_norm=float(update_diag.lattice_state_shift_norm),
                    )
                )
                _debug_print(
                    debug_enabled,
                    f"[algo25-em] label={debug_label} project remaining={remaining_step} tau={tau:.3f} "
                    f"weight={float(proj_diag.weight):.4f} shift={float(update_diag.lattice_state_shift_norm):.6f}",
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

            state["l_t"] = state["sampling_diffusion_l"].reverse_step(
                t=times.now.lattice,
                x_t=state["l_t"],
                pred=preds_curr["l"],
                dt=times.dt,
                num_atoms=state["batch"].num_atoms,
            )

        if bool(config.use_final_projection):
            final_preds = model.score_network(
                t=state["sampling_time_grid"][-1].graph,
                pos=state["f_t"],
                v=state["v_t"],
                h=state["a_t"],
                l=state["l_t"],
                node_index=state["node_index"],
                edge_node_index=state["edge_node_index"],
            )
            final_ell0_hat = predict_clean_lattice_from_prediction(
                l_t=state["l_t"],
                pred_l=final_preds["l"],
                t_lattice=state["sampling_time_grid"][-1].lattice,
                diffusion_l=state["sampling_diffusion_l"],
                num_atoms=state["batch"].num_atoms,
            )
            final_ell0_soft, _ = cps_lattice_project_clean_estimate(
                ell0_hat=final_ell0_hat,
                num_atoms=int(state["batch"].num_atoms.reshape(-1)[0].item()),
                lattice_transform=lattice_transform,
                space_group_number=int(oracle_space_group),
                tau=0.0,
                gamma_min=float(config.gamma_min),
                gamma_max=float(config.gamma_max),
                gamma_power=float(config.gamma_power),
                projection_start_frac=float(config.projection_start_frac),
                preserve_lattice_volume=bool(config.preserve_lattice_volume),
                compute_diagnostics=bool(config.collect_projection_diagnostics),
            )
            state["l_t"], _ = apply_lattice_cps_to_state(
                ell_t=state["l_t"],
                ell0_hat=final_ell0_hat,
                ell0_soft=final_ell0_soft,
                t_lattice=state["sampling_time_grid"][-1].lattice,
                diffusion_l=state["sampling_diffusion_l"],
            )
            state["l_t"] = _as_graph_l_batch(state["l_t"])

    return Algorithm25SamplerResult(
        frac_coords=state["f_t"].detach().clone(),
        velocity=state["v_t"].detach().clone(),
        lattice=_restore_l_shape(state["l_t"].detach().clone(), ref=state["batch"].l),
        atom_types=state["a_t"].detach().clone(),
        interventions=tuple(interventions),
    )


__all__ = [
    "Algorithm25Config",
    "Algorithm25Intervention",
    "Algorithm25ProjectionDiagnostics",
    "Algorithm25SamplerResult",
    "Algorithm25StateUpdateDiagnostics",
    "apply_lattice_cps_to_state",
    "cps_gamma_weight",
    "cps_lattice_project_clean_estimate",
    "k_family_violation",
    "k_to_lattice6",
    "k_to_lattice_matrix",
    "kldm_em_cps_lattice_sampler",
    "kldm_pc_cps_lattice_sampler",
    "lattice6_to_k",
    "lattice6_to_matrix",
    "lattice_matrix_to_k",
    "matrix_to_lattice6",
    "predict_clean_lattice_from_prediction",
    "project_k_to_spacegroup_family",
    "raw_length_angle_family_projection",
    "should_project_step",
    "spacegroup_to_crystal_family",
]
