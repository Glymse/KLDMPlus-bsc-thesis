from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from kldmPlus.algorithm10_casal_chart import _decode_lattice_matrix, _encode_lattice_matrix
from kldmPlus.symmetry.k_basis import cell_to_k, k_to_cell_matrix, space_group_k_constraint
from kldmPlus.utils.time import iter_sampling_times


@dataclass(frozen=True)
class KSpaceProjectionResult:
    k_input: torch.Tensor
    k_projected: torch.Tensor
    l_projected: torch.Tensor
    cell_input: torch.Tensor
    cell_projected: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    k_residual_before: float
    k_residual_after: float
    volume_before: float
    volume_after: float
    volume_ratio_to_input: float
    min_length_before: float
    min_length_after: float
    finite_ok: bool
    positive_volume_ok: bool
    lengths_ok: bool
    physical_ok: bool


@dataclass(frozen=True)
class KSpaceLatticeCASALState:
    x_k: torch.Tensor
    z_k: torch.Tensor
    mu_k: torch.Tensor
    initialized: bool
    last_projection_ok: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class KSpaceLatticeCASALConfig:
    rho_start: float = 0.5
    rho_end: float = 2.0
    tau_scale: float = 0.025
    mu_eta: float = 1.0
    mu_clip: float = 0.25
    dual_enabled: bool = True
    dual_rule: str = "plain_eta"
    projection_start_fraction: float = 0.0
    projection_start_step: int = 1
    projection_interval: int = 1
    projection_min_lattice_length: float = 0.5
    hard_volume_ratio_min: float = 0.25
    hard_volume_ratio_max: float = 4.0
    projection_target_mode: str = "x_plus_mu"
    projection_guard_mode: str = "none"
    initialize_from_x0: bool = True
    feedback_mode: str = "x"
    return_mode: str = "z"
    debug: bool = False


def _cell_volume(cell: torch.Tensor) -> float:
    return float(torch.abs(torch.linalg.det(cell)).detach().item())


def _cell_min_length(cell: torch.Tensor) -> float:
    return float(torch.linalg.norm(cell, dim=-1).min().detach().item())


def _rho_schedule(config: KSpaceLatticeCASALConfig, step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(config.rho_start)
    alpha = float(step_idx) / float(max(total_steps - 1, 1))
    return float(config.rho_start) + alpha * (float(config.rho_end) - float(config.rho_start))


def _dual_step(config: KSpaceLatticeCASALConfig, *, tau_step: float, rho: float) -> float:
    rule = str(getattr(config, "dual_rule", "plain_eta")).strip().lower()
    if rule in {"plain", "eta", "plain_eta"}:
        return float(config.mu_eta)
    if rule in {"tau_over_rho", "normalized"}:
        return float(config.mu_eta) * float(tau_step) / max(float(rho), 1.0e-12)
    if rule in {"tau", "plain_tau"}:
        return float(config.mu_eta) * float(tau_step)
    if rule in {"beta", "plain_beta"}:
        return float(config.mu_eta) * float(tau_step) * float(rho)
    raise ValueError(f"Unsupported dual_rule={config.dual_rule!r}")


def _projection_target(
    *,
    mode: str,
    x_k_next: torch.Tensor,
    z_k_prev: torch.Tensor,
    mu_k: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    target_mode = str(mode).strip().lower()
    if target_mode in {"x_plus_mu", "cascal", "casal", "direct"}:
        return x_k_next + mu_k
    if target_mode in {"x_only", "primal_only"}:
        return x_k_next
    if target_mode in {"underrelaxed", "z_plus_beta", "relaxed"}:
        return z_k_prev + beta * (x_k_next + mu_k - z_k_prev)
    raise ValueError(f"Unsupported projection_target_mode={mode!r}")


def _projection_accepted(
    *,
    mode: str,
    projection: KSpaceProjectionResult,
) -> bool:
    guard_mode = str(mode).strip().lower()
    if guard_mode in {"none", "off", "disabled", "cascal"}:
        return True
    if guard_mode in {"physical", "safety", "strict"}:
        return bool(projection.physical_ok)
    raise ValueError(f"Unsupported projection_guard_mode={mode!r}")


def l_to_cell(
    l: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    return _decode_lattice_matrix(
        l=torch.as_tensor(l).reshape(-1),
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    ).reshape(3, 3)


def cell_to_l(
    cell: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> torch.Tensor:
    return _encode_lattice_matrix(
        cell_matrix=torch.as_tensor(cell).reshape(3, 3),
        num_atoms=int(num_atoms),
        lattice_transform=lattice_transform,
    ).reshape(-1)


def cell_to_k_paper(cell: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    gram = cell @ cell.transpose(-1, -2)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(eps)
    s_matrix = 0.5 * (eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-1, -2))
    s00 = s_matrix[..., 0, 0]
    s11 = s_matrix[..., 1, 1]
    s22 = s_matrix[..., 2, 2]
    return torch.stack(
        [
            s_matrix[..., 0, 1],
            s_matrix[..., 0, 2],
            s_matrix[..., 1, 2],
            0.5 * (s00 - s11),
            (s00 + s11 - 2.0 * s22) / 6.0,
            (s00 + s11 + s22) / 3.0,
        ],
        dim=-1,
    )


def l_to_k(
    l: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell = l_to_cell(l, num_atoms=num_atoms, lattice_transform=lattice_transform)
    return cell_to_k(cell, eps=1.0e-8).reshape(-1), cell


def k_to_l(
    k: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell = k_to_cell_matrix(torch.as_tensor(k).reshape(-1)).reshape(3, 3)
    return cell_to_l(cell, num_atoms=num_atoms, lattice_transform=lattice_transform), cell


def project_k_to_family(
    *,
    k: torch.Tensor,
    space_group: int,
    num_atoms: int,
    lattice_transform: Any | None,
    volume_ratio_min: float = 0.25,
    volume_ratio_max: float = 4.0,
    min_length: float = 0.5,
) -> KSpaceProjectionResult:
    k_in = torch.as_tensor(k).reshape(-1)
    constraint = space_group_k_constraint(
        space_group_number=int(space_group),
        device=k_in.device,
        dtype=k_in.dtype,
    )
    k_proj = (1.0 - constraint.mask) * k_in + constraint.mask * constraint.target
    cell_in = k_to_cell_matrix(k_in).reshape(3, 3)
    cell_proj = k_to_cell_matrix(k_proj).reshape(3, 3)
    l_proj = cell_to_l(cell_proj, num_atoms=num_atoms, lattice_transform=lattice_transform)
    residual_before = float(torch.linalg.norm((constraint.mask * (k_in - constraint.target)).reshape(-1)).detach().item())
    residual_after = float(torch.linalg.norm((constraint.mask * (k_proj - constraint.target)).reshape(-1)).detach().item())
    volume_before = _cell_volume(cell_in)
    volume_after = _cell_volume(cell_proj)
    ratio = volume_after / max(volume_before, 1.0e-12)
    min_length_before = _cell_min_length(cell_in)
    min_length_after = _cell_min_length(cell_proj)
    finite_ok = bool(torch.isfinite(cell_proj).all() and torch.isfinite(k_proj).all() and torch.isfinite(l_proj).all())
    positive_volume_ok = bool(volume_after > 1.0e-12)
    lengths_ok = bool(min_length_after >= float(min_length))
    volume_ok = bool(float(volume_ratio_min) <= ratio <= float(volume_ratio_max))
    physical_ok = bool(finite_ok and positive_volume_ok and lengths_ok and volume_ok)
    return KSpaceProjectionResult(
        k_input=k_in.detach().clone(),
        k_projected=k_proj.detach().clone(),
        l_projected=l_proj.detach().clone(),
        cell_input=cell_in.detach().clone(),
        cell_projected=cell_proj.detach().clone(),
        mask=constraint.mask.detach().clone(),
        target=constraint.target.detach().clone(),
        k_residual_before=residual_before,
        k_residual_after=residual_after,
        volume_before=volume_before,
        volume_after=volume_after,
        volume_ratio_to_input=ratio,
        min_length_before=min_length_before,
        min_length_after=min_length_after,
        finite_ok=finite_ok,
        positive_volume_ok=positive_volume_ok,
        lengths_ok=lengths_ok,
        physical_ok=physical_ok,
    )


def project_l_to_k_family(
    *,
    l: torch.Tensor,
    space_group: int,
    num_atoms: int,
    lattice_transform: Any | None,
    volume_ratio_min: float = 0.25,
    volume_ratio_max: float = 4.0,
    min_length: float = 0.5,
) -> KSpaceProjectionResult:
    k, _cell = l_to_k(l, num_atoms=num_atoms, lattice_transform=lattice_transform)
    return project_k_to_family(
        k=k,
        space_group=space_group,
        num_atoms=num_atoms,
        lattice_transform=lattice_transform,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
        min_length=min_length,
    )


def kspace_lattice_casal_step(
    *,
    x_k_prev: torch.Tensor,
    x_k_kldm_next: torch.Tensor,
    state: KSpaceLatticeCASALState,
    space_group: int,
    num_atoms: int,
    lattice_transform: Any | None,
    rho: float,
    tau: float,
    mu_clip: float | None,
    dual_enabled: bool,
    dual_rule: str,
    mu_eta: float,
    volume_ratio_min: float,
    volume_ratio_max: float,
    min_length: float,
    projection_target_mode: str = "x_plus_mu",
    projection_guard_mode: str = "none",
) -> tuple[torch.Tensor, KSpaceProjectionResult, KSpaceLatticeCASALState]:
    beta = float(tau) * float(rho)
    residual_prev = x_k_prev - state.z_k + state.mu_k
    x_k_next = x_k_kldm_next - beta * residual_prev
    y = _projection_target(
        mode=projection_target_mode,
        x_k_next=x_k_next,
        z_k_prev=state.z_k,
        mu_k=state.mu_k,
        beta=beta,
    )
    z_proj = project_k_to_family(
        k=y,
        space_group=space_group,
        num_atoms=num_atoms,
        lattice_transform=lattice_transform,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
        min_length=min_length,
    )
    accepted = _projection_accepted(mode=projection_guard_mode, projection=z_proj)
    if not accepted:
        diagnostics = {
            "k_residual_before": float(z_proj.k_residual_before),
            "k_residual_after": float(z_proj.k_residual_after),
            "x_z_residual": float(torch.linalg.norm((x_k_next - state.z_k).reshape(-1)).detach().item()),
            "mu_norm": float(torch.linalg.norm(state.mu_k.reshape(-1)).detach().item()),
            "mu_clip_fraction": 0.0,
            "projection_physical_ok": bool(z_proj.physical_ok),
            "projection_accepted": False,
        }
        next_state = KSpaceLatticeCASALState(
            x_k=x_k_next.detach().clone(),
            z_k=state.z_k.detach().clone(),
            mu_k=state.mu_k.detach().clone(),
            initialized=True,
            last_projection_ok=False,
            diagnostics=diagnostics,
        )
        return x_k_next, z_proj, next_state

    mu_next = state.mu_k.detach().clone()
    mu_clip_fraction = 0.0
    if bool(dual_enabled):
        dual_cfg = KSpaceLatticeCASALConfig(mu_eta=mu_eta, dual_rule=dual_rule)
        dual_step = _dual_step(dual_cfg, tau_step=float(tau), rho=float(rho))
        mu_next = state.mu_k + dual_step * (x_k_next - z_proj.k_projected)
        if mu_clip is not None and float(mu_clip) > 0.0:
            unclipped = mu_next
            mu_next = mu_next.clamp(min=-float(mu_clip), max=float(mu_clip))
            mu_clip_fraction = float((unclipped != mu_next).float().mean().detach().item())
    diagnostics = {
        "k_residual_before": float(z_proj.k_residual_before),
        "k_residual_after": float(z_proj.k_residual_after),
        "x_z_residual": float(torch.linalg.norm((x_k_next - z_proj.k_projected).reshape(-1)).detach().item()),
        "mu_norm": float(torch.linalg.norm(mu_next.reshape(-1)).detach().item()),
        "mu_clip_fraction": mu_clip_fraction,
        "projection_physical_ok": bool(z_proj.physical_ok),
        "projection_accepted": True,
    }
    next_state = KSpaceLatticeCASALState(
        x_k=x_k_next.detach().clone(),
        z_k=z_proj.k_projected.detach().clone(),
        mu_k=mu_next.detach().clone(),
        initialized=True,
        last_projection_ok=True,
        diagnostics=diagnostics,
    )
    return x_k_next, z_proj, next_state


def sample_kldm_lattice_casal_kspace(
    *,
    model: Any,
    batch: Any,
    n_steps: int,
    lattice_transform: Any | None = None,
    t_start: float = 1.0,
    t_final: float = 1.0e-6,
    config: KSpaceLatticeCASALConfig | None = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]:
    if config is None:
        config = KSpaceLatticeCASALConfig()
    if not hasattr(batch, "space_group"):
        raise ValueError("k-space lattice-CASAL requires batch.space_group.")

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
    total_steps = max(1, int(state["sampling_time_grid"].numel()) - 1)
    projection_interval = max(1, int(config.projection_interval))
    projection_start_step = max(
        int(config.projection_start_step),
        int(math.ceil(float(config.projection_start_fraction) * float(total_steps))),
    )

    graph_states: list[KSpaceLatticeCASALState | None] = [None for _ in range(int(batch.num_graphs))]
    diagnostics: list[dict[str, Any]] = [
        {
            "graph_idx": int(graph_idx),
            "requested_sg": int(requested_sgs[graph_idx].item()),
            "num_projection_successes": 0,
            "num_projection_failures": 0,
            "first_casal_step": 0,
            "last_casal_step": 0,
            "k_residual_before_last": float("nan"),
            "k_residual_after_last": float("nan"),
            "x_z_residual_last": float("nan"),
            "mu_norm_last": float("nan"),
            "mu_clip_fraction_last": float("nan"),
            "volume_ratio_last": float("nan"),
            "feedback_mode": str(config.feedback_mode),
            "return_mode": str(config.return_mode),
            "dual_rule": str(config.dual_rule),
            "projection_target_mode": str(config.projection_target_mode),
            "projection_guard_mode": str(config.projection_guard_mode),
        }
        for graph_idx in range(int(batch.num_graphs))
    ]

    def _initialize_graph_state(graph_idx: int, x_k_current: torch.Tensor) -> None:
        num_atoms = int(ptr[graph_idx + 1] - ptr[graph_idx])
        space_group = int(requested_sgs[graph_idx].item())
        diag = diagnostics[graph_idx]
        init_proj = project_k_to_family(
            k=x_k_current,
            space_group=space_group,
            num_atoms=num_atoms,
            lattice_transform=lattice_transform,
            volume_ratio_min=float(config.hard_volume_ratio_min),
            volume_ratio_max=float(config.hard_volume_ratio_max),
            min_length=float(config.projection_min_lattice_length),
        )
        accepted = _projection_accepted(mode=str(config.projection_guard_mode), projection=init_proj)
        if not accepted:
            diag["num_projection_failures"] += 1
            return
        graph_states[graph_idx] = KSpaceLatticeCASALState(
            x_k=x_k_current.detach().clone(),
            z_k=init_proj.k_projected.detach().clone(),
            mu_k=torch.zeros_like(x_k_current),
            initialized=True,
            last_projection_ok=True,
            diagnostics={
                "k_residual_before": float(init_proj.k_residual_before),
                "k_residual_after": float(init_proj.k_residual_after),
                "x_z_residual": float(torch.linalg.norm((x_k_current - init_proj.k_projected).reshape(-1)).detach().item()),
                "mu_norm": 0.0,
                "mu_clip_fraction": 0.0,
                "projection_physical_ok": bool(init_proj.physical_ok),
                "projection_accepted": True,
            },
        )
        diag["num_projection_successes"] += 1

    if bool(config.initialize_from_x0) and projection_start_step <= 1:
        for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            num_atoms = int(end - start)
            x_k0, _ = l_to_k(state["l_t"][graph_idx], num_atoms=num_atoms, lattice_transform=lattice_transform)
            _initialize_graph_state(graph_idx, x_k0)

    with torch.no_grad():
        for step_idx, times in enumerate(iter_sampling_times(batch=batch, grid=state["sampling_time_grid"]), start=1):
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
            l_prev = state["l_t"].detach().clone()
            state["l_t"] = model._reverse_lattice_sampling_step(
                t=times.now.lattice,
                x_t=state["l_t"],
                pred=preds_curr["l"],
                dt=times.dt,
                num_atoms=batch.num_atoms,
            )
            if step_idx < projection_start_step:
                continue
            if (step_idx - projection_start_step) % projection_interval != 0:
                continue

            rho = _rho_schedule(config, step_idx - 1, total_steps)
            tau_step = max(float(times.dt) * float(config.tau_scale), 0.0)
            for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                num_atoms = int(end - start)
                space_group = int(requested_sgs[graph_idx].item())
                x_k_prev, _ = l_to_k(l_prev[graph_idx], num_atoms=num_atoms, lattice_transform=lattice_transform)
                x_k_kldm_next, _ = l_to_k(state["l_t"][graph_idx], num_atoms=num_atoms, lattice_transform=lattice_transform)
                diag = diagnostics[graph_idx]
                if graph_states[graph_idx] is None:
                    _initialize_graph_state(graph_idx, x_k_kldm_next)
                    if graph_states[graph_idx] is None:
                        continue

                current_state = graph_states[graph_idx]
                assert current_state is not None
                x_k_next, z_proj, next_state = kspace_lattice_casal_step(
                    x_k_prev=x_k_prev,
                    x_k_kldm_next=x_k_kldm_next,
                    state=current_state,
                    space_group=space_group,
                    num_atoms=num_atoms,
                    lattice_transform=lattice_transform,
                    rho=rho,
                    tau=tau_step,
                    mu_clip=float(config.mu_clip),
                    dual_enabled=bool(config.dual_enabled),
                    dual_rule=str(config.dual_rule),
                    mu_eta=float(config.mu_eta),
                    volume_ratio_min=float(config.hard_volume_ratio_min),
                    volume_ratio_max=float(config.hard_volume_ratio_max),
                    min_length=float(config.projection_min_lattice_length),
                    projection_target_mode=str(config.projection_target_mode),
                    projection_guard_mode=str(config.projection_guard_mode),
                )
                graph_states[graph_idx] = next_state
                if bool(next_state.diagnostics.get("projection_accepted", False)):
                    diag["num_projection_successes"] += 1
                else:
                    diag["num_projection_failures"] += 1

                if str(config.feedback_mode).strip().lower() in {"z", "projected", "z_feedback"}:
                    feedback_k = next_state.z_k
                else:
                    feedback_k = x_k_next
                feedback_l, _ = k_to_l(
                    feedback_k,
                    num_atoms=num_atoms,
                    lattice_transform=lattice_transform,
                )
                state["l_t"][graph_idx] = feedback_l.reshape_as(state["l_t"][graph_idx])
                diag["last_casal_step"] = int(step_idx)
                diag["k_residual_before_last"] = float(z_proj.k_residual_before)
                diag["k_residual_after_last"] = float(z_proj.k_residual_after)
                diag["x_z_residual_last"] = float(next_state.diagnostics.get("x_z_residual", float("nan")))
                diag["mu_norm_last"] = float(next_state.diagnostics.get("mu_norm", float("nan")))
                diag["mu_clip_fraction_last"] = float(next_state.diagnostics.get("mu_clip_fraction", float("nan")))
                diag["volume_ratio_last"] = float(z_proj.volume_ratio_to_input)

    pos_out = state["f_t"].detach().clone()
    v_out = torch.zeros_like(state["v_t"])
    l_out = state["l_t"].detach().clone()
    h_out = state["a_t"].detach().clone()
    if str(config.return_mode).strip().lower() in {"z", "strict_z", "constrained"}:
        for graph_idx, graph_state in enumerate(graph_states):
            if graph_state is None:
                continue
            num_atoms = int(ptr[graph_idx + 1] - ptr[graph_idx])
            l_graph, _ = k_to_l(
                graph_state.z_k,
                num_atoms=num_atoms,
                lattice_transform=lattice_transform,
            )
            l_out[graph_idx] = l_graph.reshape_as(l_out[graph_idx])

    if bool(config.debug):
        print(
            f"algorithm12_kspace_lattice_casal done elapsed_s={time.perf_counter() - started:.2f} "
            f"graphs={int(batch.num_graphs)}",
            flush=True,
        )

    result = (pos_out, v_out, l_out, h_out)
    if not return_diagnostics:
        return result
    return (*result, diagnostics)
