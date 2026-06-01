from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from kldmPlus.algorithm13_fixed_template_velocity_casal import (
    FixedTemplateVelocityConfig,
    materialize_template,
)
from kldmPlus.algorithm14_kldm_casal_velocity_impulse import center_velocity
from kldmPlus.algorithm16_kldm_wyckoff_ppr_faithful import (
    KLDMForwardRenoiseResult,
    KLDMGraphState,
    KLDMCleanEstimate,
    PPRProjectionResult,
    _as_graph_l_batch,
    _restore_graph_l_shape,
    _score_network_predict,
    kldm_forward_renoise_fv_only,
    project_clean_to_fixed_wyckoff,
)
from kldmPlus.symmetry.pcs_projection import PCSTemplateState
from kldmPlus.utils.time import iter_sampling_times, sampling_grid


ALGORITHM17_MODE = "pfode_kldm_wyckoff_ppr"
ALGORITHM17_IS_FULL_PPR = False
ALGORITHM17_PPR_CLOSENESS = (
    "Optimizes the noisy KLDM fractional/velocity state through a differentiable "
    "PF-ODE clean estimator, then renoises with the KLDM forward kernel. It is "
    "closer to PPR than x0-space CPR, but still keeps lattice fixed and uses a "
    "fixed Wyckoff template."
)


@dataclass(frozen=True)
class PFODEWyckoffPPRConfig:
    mode: str = "FV"
    anchor_mode: str = "soft"
    pf_steps: int = 16
    opt_steps: int = 10
    lr_fv: float = 1.0e-3
    lr_theta: float = 1.0e-2
    lambda_f: float = 100.0
    lambda_v: float = 10.0
    lambda_theta: float = 1.0
    lambda_pair: float = 0.0
    pair_min_distance: float = 0.5
    pair_barrier_sharpness: float = 10.0
    anchor_min_pair_distance_guard: float = 0.5
    rho_f: float = 0.02
    rho_v: float = 0.10
    max_delta_f_rms: float = 0.02
    max_velocity_norm_ratio: float = 1.10
    min_loss_improvement: float = 1.0e-8
    mean_free_threshold: float = 1.0e-6
    lattice_change_threshold: float = 1.0e-8
    return_best_safe_iterate: bool = True
    t_final: float = 1.0e-3


@dataclass(frozen=True)
class PFODECleanEstimate:
    clean: KLDMCleanEstimate
    f_t_star: torch.Tensor
    v_t_star: torch.Tensor


@dataclass(frozen=True)
class PFODEWyckoffPPROptResult:
    config: PFODEWyckoffPPRConfig
    success: bool
    reject_reason: str
    f_star: torch.Tensor
    v_star: torch.Tensor
    theta_star: torch.Tensor
    theta_init: torch.Tensor
    tau: torch.Tensor
    clean_before: KLDMCleanEstimate
    clean_after: KLDMCleanEstimate
    z_init: torch.Tensor
    z_star: torch.Tensor
    loss_before: float
    loss_after: float
    wyckoff_loss_before: float
    wyckoff_loss_after: float
    pair_loss_before: float
    pair_loss_after: float
    delta_f_rms: float
    delta_v_rms: float
    velocity_norm_ratio: float
    mean_free_norm: float
    best_iter: int
    loss_history: tuple[float, ...]
    grad_f_norm: float
    grad_v_norm: float
    grad_theta_norm: float


@dataclass(frozen=True)
class PFODEWyckoffPPRStepResult:
    optimized: PFODEWyckoffPPROptResult
    anchor_f0: torch.Tensor
    anchor_mode: str
    anchor_min_pair_distance: float
    renoise: KLDMForwardRenoiseResult | None
    state_out: KLDMGraphState | None
    accepted: bool
    reject_reason: str


def smooth_wrap_residual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    two_pi = torch.as_tensor(2.0 * math.pi, device=diff.device, dtype=diff.dtype)
    return torch.atan2(torch.sin(two_pi * diff), torch.cos(two_pi * diff)) / two_pi


def center_velocity_tensor(v: torch.Tensor) -> torch.Tensor:
    return v - v.mean(dim=0, keepdim=True)


def _reverse_exp_step_pf_ode(
    *,
    sampling_tdm,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    score_v: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_t = sampling_tdm.wrap_displacements(f_t)
    dt_internal = torch.as_tensor(sampling_tdm.T * dt, device=v_t.device, dtype=v_t.dtype)
    exp_dt = torch.exp(dt_internal)
    expm1_dt = torch.expm1(dt_internal)
    score_scale = torch.as_tensor(sampling_tdm.vel_scale**2, device=v_t.device, dtype=v_t.dtype)
    v_prev = exp_dt * v_t + score_scale * expm1_dt * score_v
    f_prev = sampling_tdm.wrap_displacements(f_t - dt_internal * v_prev)
    return f_prev, v_prev


def differentiable_pfode_clean_estimate(
    *,
    model,
    batch,
    state: KLDMGraphState,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    n_steps: int,
    t_start: float,
    t_final: float = 1.0e-3,
) -> KLDMCleanEstimate:
    score_network = model.score_network
    restore_training = score_network.training
    score_network.eval()
    try:
        grid = sampling_grid(
            batch=batch,
            n_steps=max(int(n_steps), 1),
            t_start=float(t_start),
            t_final=float(t_final),
        )
        f_t = state.f
        v_t = state.v
        # Algorithm 17 optimizes only the coordinate/velocity branch.  The
        # lattice is fixed context from the incoming facitKLDM state.
        l_t = _as_graph_l_batch(state.l)
        h_t = state.h
        actual_steps = 0
        for times in iter_sampling_times(batch=batch, grid=grid):
            preds = _score_network_predict(
                model,
                t_graph=times.now.graph,
                pos=f_t,
                v=v_t,
                h=h_t,
                l=l_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )
            score_v = model.tdm.reconstruct_full_reverse_velocity_score(
                t=times.now.nodes,
                v_t=v_t,
                pred_v=preds["v"],
                index=node_index,
            )
            f_t, v_t = _reverse_exp_step_pf_ode(
                sampling_tdm=model.tdm,
                f_t=f_t,
                v_t=v_t,
                score_v=score_v,
                dt=times.dt,
            )
            actual_steps += 1
        return KLDMCleanEstimate(
            f0_hat=f_t,
            v0_hat=v_t,
            l0_hat=_restore_graph_l_shape(l_t),
            steps=int(actual_steps),
            estimator_mode="differentiable_pf_ode",
        )
    finally:
        if restore_training:
            score_network.train()


def _template_frac(theta: torch.Tensor, template_state: PCSTemplateState, tau: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return materialize_template(theta.reshape(-1), template_state, tau=tau).frac_coords.to(device=ref.device, dtype=ref.dtype)


def _wyckoff_loss(f_hat: torch.Tensor, theta: torch.Tensor, template_state: PCSTemplateState, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = _template_frac(theta, template_state, tau, f_hat)
    residual = smooth_wrap_residual(f_hat, z)
    return torch.mean(residual**2), z


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.reshape(-1) ** 2))


def _pair_distances(frac: torch.Tensor, cell_matrix: torch.Tensor | None) -> torch.Tensor:
    if cell_matrix is None or int(frac.shape[0]) < 2:
        return frac.new_empty((0,))
    cell = cell_matrix.to(device=frac.device, dtype=frac.dtype).reshape(3, 3)
    diff = smooth_wrap_residual(frac[:, None, :], frac[None, :, :])
    cart = torch.matmul(diff, cell)
    dist = torch.linalg.norm(cart, dim=-1)
    i, j = torch.triu_indices(int(frac.shape[0]), int(frac.shape[0]), offset=1, device=frac.device)
    return dist[i, j]


def _pair_barrier(
    frac: torch.Tensor,
    cell_matrix: torch.Tensor | None,
    min_distance: float,
    sharpness: float = 10.0,
) -> torch.Tensor:
    distances = _pair_distances(frac, cell_matrix)
    if distances.numel() == 0:
        return frac.sum() * 0.0
    margin = torch.as_tensor(float(min_distance), device=frac.device, dtype=frac.dtype) - distances
    sharp = torch.as_tensor(max(float(sharpness), 1.0), device=frac.device, dtype=frac.dtype)
    barrier = torch.nn.functional.softplus(sharp * margin) / sharp
    return torch.mean(barrier.square())


def _min_pair_distance(frac: torch.Tensor, cell_matrix: torch.Tensor | None) -> float:
    with torch.no_grad():
        distances = _pair_distances(frac, cell_matrix)
        if distances.numel() == 0:
            return float("inf")
        return float(distances.min().detach().item())


def _safe_velocity_norm_ratio(v_after: torch.Tensor, v_before: torch.Tensor) -> float:
    before = float(torch.linalg.norm(v_before.reshape(-1)).detach().item())
    after = float(torch.linalg.norm(v_after.reshape(-1)).detach().item())
    return float(after / max(before, 1.0e-12))


def optimize_pfode_kldm_wyckoff_ppr(
    *,
    model,
    batch,
    state: KLDMGraphState,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    theta0: torch.Tensor | None,
    fixed_assignment: torch.Tensor | None,
    projection_config: FixedTemplateVelocityConfig,
    config: PFODEWyckoffPPRConfig = PFODEWyckoffPPRConfig(),
    cell_matrix: torch.Tensor | None = None,
) -> PFODEWyckoffPPROptResult:
    for p in model.parameters():
        p.requires_grad_(False)

    clean_before = differentiable_pfode_clean_estimate(
        model=model,
        batch=batch,
        state=state,
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=int(config.pf_steps),
        t_start=float(state.t),
        t_final=float(config.t_final),
    )
    init_proj = project_clean_to_fixed_wyckoff(
        f0_hat=clean_before.f0_hat.detach(),
        atomic_numbers=state.h,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        config=projection_config,
        reference_frac=clean_before.f0_hat.detach(),
    )
    theta_init = init_proj.theta.detach().clone().reshape(-1)
    tau = init_proj.tau.detach().clone().reshape(1, 3)
    template_state_opt = init_proj.raw.state

    u_f = torch.zeros_like(state.f, requires_grad=(str(config.mode).upper() == "FV"))
    u_v = torch.zeros_like(state.v, requires_grad=True)
    theta_var = theta_init.detach().clone().requires_grad_(True)
    params = [u_v, theta_var] if str(config.mode).upper() == "V_ONLY" else [u_f, u_v, theta_var]
    optimizer = torch.optim.Adam(
        [
            {"params": [u_f, u_v], "lr": float(config.lr_fv)},
            {"params": [theta_var], "lr": float(config.lr_theta)},
        ]
    )

    history: list[float] = []
    grad_f_norm = float("nan")
    grad_v_norm = float("nan")
    grad_theta_norm = float("nan")

    def candidate_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if str(config.mode).upper() == "V_ONLY":
            delta_f = torch.zeros_like(state.f)
        else:
            delta_f = float(config.rho_f) * torch.tanh(u_f)
        delta_v = float(config.rho_v) * torch.tanh(u_v)
        f_candidate = torch.remainder(state.f + delta_f, 1.0)
        v_candidate = center_velocity_tensor(state.v + delta_v)
        return f_candidate, v_candidate, delta_f, delta_v

    def objective() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        f_candidate, v_candidate, delta_f, delta_v = candidate_state()
        clean = differentiable_pfode_clean_estimate(
            model=model,
            batch=batch,
            state=KLDMGraphState(
                f=f_candidate,
                v=v_candidate,
                l=state.l,
                h=state.h,
                k=state.k,
                t=state.t,
                dt=state.dt,
                graph_idx0=state.graph_idx0,
            ),
            node_index=node_index,
            edge_node_index=edge_node_index,
            n_steps=int(config.pf_steps),
            t_start=float(state.t),
            t_final=float(config.t_final),
        )
        wyckoff, z = _wyckoff_loss(clean.f0_hat, theta_var, template_state_opt, tau)
        pair_anchor = z if str(config.anchor_mode).lower() == "hard" else clean.f0_hat
        pair_loss = _pair_barrier(
            pair_anchor,
            cell_matrix,
            float(config.pair_min_distance),
            float(config.pair_barrier_sharpness),
        )
        trust_f = torch.mean(delta_f.reshape(-1) ** 2)
        trust_v = torch.mean(delta_v.reshape(-1) ** 2)
        trust_theta = torch.mean((theta_var - theta_init).reshape(-1) ** 2) if theta_init.numel() else theta_var.sum() * 0.0
        loss = (
            wyckoff
            + float(config.lambda_f) * trust_f
            + float(config.lambda_v) * trust_v
            + float(config.lambda_theta) * trust_theta
            + float(config.lambda_pair) * pair_loss
        )
        return loss, {
            "clean_f": clean.f0_hat,
            "z": z,
            "wyckoff": wyckoff,
            "pair_loss": pair_loss,
            "delta_f": delta_f,
            "delta_v": delta_v,
            "f_candidate": f_candidate,
            "v_candidate": v_candidate,
        }

    with torch.enable_grad():
        before_loss, before_aux = objective()
        loss_before = float(before_loss.detach().item())
        wyckoff_before = float(before_aux["wyckoff"].detach().item())
        pair_before = float(before_aux["pair_loss"].detach().item())

        best_payload: dict[str, Any] | None = None
        best_key = (float("inf"),)
        for step_idx in range(max(int(config.opt_steps), 0) + 1):
            loss, aux = objective()
            if step_idx > 0:
                history.append(float(loss.detach().item()))
            delta_f_rms = float(_rms(aux["delta_f"]).detach().item())
            velocity_ratio = _safe_velocity_norm_ratio(aux["v_candidate"], state.v)
            mean_free_norm = float(torch.linalg.norm(aux["v_candidate"].mean(dim=0)).detach().item())
            finite = bool(
                torch.isfinite(loss).all().item()
                and torch.isfinite(aux["f_candidate"]).all().item()
                and torch.isfinite(aux["v_candidate"]).all().item()
                and torch.isfinite(aux["clean_f"]).all().item()
            )
            safe = bool(
                finite
                and delta_f_rms <= float(config.max_delta_f_rms)
                and velocity_ratio <= float(config.max_velocity_norm_ratio)
                and mean_free_norm <= float(config.mean_free_threshold)
            )
            key = (float(loss.detach().item()),)
            if safe and key < best_key:
                best_key = key
                best_payload = {
                    "iter": int(step_idx),
                    "loss": float(loss.detach().item()),
                    "wyckoff": float(aux["wyckoff"].detach().item()),
                    "pair_loss": float(aux["pair_loss"].detach().item()),
                    "f": aux["f_candidate"].detach().clone(),
                    "v": aux["v_candidate"].detach().clone(),
                    "theta": theta_var.detach().clone(),
                    "z": aux["z"].detach().clone(),
                    "clean_f": aux["clean_f"].detach().clone(),
                    "delta_f": aux["delta_f"].detach().clone(),
                    "delta_v": aux["delta_v"].detach().clone(),
                    "delta_f_rms": float(delta_f_rms),
                    "velocity_ratio": float(velocity_ratio),
                    "mean_free_norm": float(mean_free_norm),
                }
            if step_idx == max(int(config.opt_steps), 0):
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_f_norm = 0.0 if u_f.grad is None else float(torch.linalg.norm(u_f.grad.reshape(-1)).detach().item())
            grad_v_norm = 0.0 if u_v.grad is None else float(torch.linalg.norm(u_v.grad.reshape(-1)).detach().item())
            grad_theta_norm = 0.0 if theta_var.grad is None else float(torch.linalg.norm(theta_var.grad.reshape(-1)).detach().item())
            optimizer.step()

    if best_payload is None:
        best_payload = {
            "iter": 0,
            "loss": loss_before,
            "wyckoff": wyckoff_before,
            "pair_loss": pair_before,
            "f": state.f.detach().clone(),
            "v": center_velocity_tensor(state.v).detach().clone(),
            "theta": theta_init.detach().clone(),
            "z": before_aux["z"].detach().clone(),
            "clean_f": clean_before.f0_hat.detach().clone(),
            "delta_f": torch.zeros_like(state.f),
            "delta_v": torch.zeros_like(state.v),
            "delta_f_rms": 0.0,
            "velocity_ratio": _safe_velocity_norm_ratio(center_velocity_tensor(state.v), state.v),
            "mean_free_norm": float(torch.linalg.norm(center_velocity_tensor(state.v).mean(dim=0)).detach().item()),
        }

    clean_after = differentiable_pfode_clean_estimate(
        model=model,
        batch=batch,
        state=KLDMGraphState(
            f=best_payload["f"],
            v=best_payload["v"],
            l=state.l,
            h=state.h,
            k=state.k,
            t=state.t,
            dt=state.dt,
            graph_idx0=state.graph_idx0,
        ),
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=int(config.pf_steps),
        t_start=float(state.t),
        t_final=float(config.t_final),
    )
    improved = bool(float(best_payload["loss"]) < loss_before - float(config.min_loss_improvement))
    reject_reason = "" if improved else "loss_not_improved"
    return PFODEWyckoffPPROptResult(
        config=config,
        success=improved,
        reject_reason=reject_reason,
        f_star=best_payload["f"].detach().clone(),
        v_star=best_payload["v"].detach().clone(),
        theta_star=best_payload["theta"].detach().clone(),
        theta_init=theta_init.detach().clone(),
        tau=tau.detach().clone(),
        clean_before=KLDMCleanEstimate(
            f0_hat=clean_before.f0_hat.detach().clone(),
            v0_hat=clean_before.v0_hat.detach().clone(),
            l0_hat=clean_before.l0_hat.detach().clone(),
            steps=clean_before.steps,
            estimator_mode=clean_before.estimator_mode,
        ),
        clean_after=KLDMCleanEstimate(
            f0_hat=clean_after.f0_hat.detach().clone(),
            v0_hat=clean_after.v0_hat.detach().clone(),
            l0_hat=clean_after.l0_hat.detach().clone(),
            steps=clean_after.steps,
            estimator_mode=clean_after.estimator_mode,
        ),
        z_init=before_aux["z"].detach().clone(),
        z_star=best_payload["z"].detach().clone(),
        loss_before=float(loss_before),
        loss_after=float(best_payload["loss"]),
        wyckoff_loss_before=float(wyckoff_before),
        wyckoff_loss_after=float(best_payload["wyckoff"]),
        pair_loss_before=float(pair_before),
        pair_loss_after=float(best_payload["pair_loss"]),
        delta_f_rms=float(best_payload["delta_f_rms"]),
        delta_v_rms=float(_rms(best_payload["delta_v"]).detach().item()),
        velocity_norm_ratio=float(best_payload["velocity_ratio"]),
        mean_free_norm=float(best_payload["mean_free_norm"]),
        best_iter=int(best_payload["iter"]),
        loss_history=tuple(float(x) for x in history),
        grad_f_norm=float(grad_f_norm),
        grad_v_norm=float(grad_v_norm),
        grad_theta_norm=float(grad_theta_norm),
    )


def pfode_kldm_wyckoff_ppr_step(
    *,
    model,
    batch,
    state: KLDMGraphState,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    theta0: torch.Tensor | None,
    fixed_assignment: torch.Tensor | None,
    projection_config: FixedTemplateVelocityConfig,
    config: PFODEWyckoffPPRConfig = PFODEWyckoffPPRConfig(),
    cell_matrix: torch.Tensor | None = None,
    noise_v: torch.Tensor | None = None,
    noise_r: torch.Tensor | None = None,
) -> PFODEWyckoffPPRStepResult:
    opt = optimize_pfode_kldm_wyckoff_ppr(
        model=model,
        batch=batch,
        state=state,
        node_index=node_index,
        edge_node_index=edge_node_index,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        projection_config=projection_config,
        config=config,
        cell_matrix=cell_matrix,
    )
    if not opt.success:
        return PFODEWyckoffPPRStepResult(
            optimized=opt,
            anchor_f0=opt.clean_after.f0_hat,
            anchor_mode=str(config.anchor_mode),
            anchor_min_pair_distance=float("nan"),
            renoise=None,
            state_out=None,
            accepted=False,
            reject_reason=opt.reject_reason,
        )
    anchor_mode = str(config.anchor_mode).lower()
    if anchor_mode == "hard":
        anchor_f0 = opt.z_star.detach().clone()
    elif anchor_mode == "soft":
        anchor_f0 = opt.clean_after.f0_hat.detach().clone()
    else:
        raise ValueError(f"Unsupported Algorithm 17 anchor_mode={config.anchor_mode!r}")
    anchor_min_pair = _min_pair_distance(anchor_f0, cell_matrix)
    renoise = kldm_forward_renoise_fv_only(
        model=model,
        batch=batch,
        f0=anchor_f0,
        l0=state.l,
        t_graph=torch.as_tensor([float(state.t)], device=state.f.device, dtype=state.f.dtype),
        node_index=node_index,
        v0=None,
        noise_v=noise_v,
        noise_r=noise_r,
        mean_free_velocity=True,
    )
    lattice_changed = float(torch.linalg.norm((renoise.l_t.reshape(-1) - state.l.reshape(-1))).detach().item())
    if not bool(renoise.finite_ok):
        reason = "renoise_nonfinite"
    elif not (
        math.isfinite(anchor_min_pair)
        and anchor_min_pair > float(config.anchor_min_pair_distance_guard)
    ):
        reason = "anchor_min_pair_distance_unsafe"
    elif not (float(renoise.mean_free_norm) <= float(config.mean_free_threshold)):
        reason = "mean_free_norm_too_large"
    elif not (lattice_changed <= float(config.lattice_change_threshold)):
        reason = "lattice_changed"
    else:
        reason = ""
    accepted = reason == ""
    state_out = None
    if accepted:
        state_out = KLDMGraphState(
            f=renoise.f_t.detach().clone(),
            v=renoise.v_t.detach().clone(),
            l=state.l.detach().clone(),
            h=state.h.detach().clone(),
            k=state.k.detach().clone(),
            t=state.t,
            dt=state.dt,
            graph_idx0=state.graph_idx0,
        )
    return PFODEWyckoffPPRStepResult(
        optimized=opt,
        anchor_f0=anchor_f0,
        anchor_mode=anchor_mode,
        anchor_min_pair_distance=float(anchor_min_pair),
        renoise=renoise,
        state_out=state_out,
        accepted=accepted,
        reject_reason=reason,
    )
