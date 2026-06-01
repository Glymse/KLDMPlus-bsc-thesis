from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from kldmPlus.algorithm13_fixed_template_velocity_casal import (
    FixedTemplateVelocityConfig,
    compute_template_jacobian,
    project_to_fixed_template,
    tangent_projector,
    wrap_residual,
)
from kldmPlus.symmetry.pcs_projection import PCSTemplateState


CASAL_THEOREM_APPLIES = False
CASAL_THEOREM_REASON = "kinetic_nonconvex_learned_score_wyckoff_union_no_mixing_time_guarantee"


def wrapdiff(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return wrap_residual(f, z)


def center_velocity(v: torch.Tensor) -> tuple[torch.Tensor, float]:
    centered = v - v.mean(dim=0, keepdim=True)
    mean_norm = float(torch.linalg.norm(v.mean(dim=0)).detach().item())
    return centered, mean_norm


def kinetic_position_update_signed(
    f: torch.Tensor,
    v: torch.Tensor,
    *,
    h_coord: float,
    sign_chi: float,
) -> torch.Tensor:
    return torch.remainder(f + float(sign_chi) * float(h_coord) * v, 1.0)


def velocity_score_prefactor(t: float | torch.Tensor) -> float:
    t_val = float(t) if not isinstance(t, torch.Tensor) else float(t.detach().item())
    return (1.0 - math.exp(-t_val)) / (1.0 + math.exp(-t_val))


def velocity_score_variance(t: float | torch.Tensor, *, eps: float = 1.0e-12) -> float:
    t_val = float(t) if not isinstance(t, torch.Tensor) else float(t.detach().item())
    return max(1.0 - math.exp(-2.0 * t_val), eps)


def reconstruct_simplified_velocity_score(
    wrapped_score: torch.Tensor,
    v: torch.Tensor,
    *,
    t: float | torch.Tensor,
    mean_free: bool = True,
) -> torch.Tensor:
    a_t = velocity_score_prefactor(t)
    sigma_v2 = velocity_score_variance(t)
    score_v = a_t * wrapped_score - v / sigma_v2
    if mean_free:
        score_v, _ = center_velocity(score_v)
    return score_v


def mean_free_jacobian(J: torch.Tensor, *, num_atoms: int) -> torch.Tensor:
    if J.numel() == 0:
        return J.detach().clone()
    J3 = J.reshape(int(num_atoms), 3, J.shape[1])
    J0 = J3 - J3.mean(dim=0, keepdim=True)
    return J0.reshape_as(J)


@dataclass(frozen=True)
class TangentMeanFreeProjection:
    velocity: torch.Tensor
    projector_rank: int
    projector_condition: float
    tangent_residual: float
    mean_norm_before: float
    mean_norm_after: float
    J0: torch.Tensor


def tangent_project_mean_free(
    v: torch.Tensor,
    *,
    J: torch.Tensor,
    metric: torch.Tensor | None = None,
    damping: float = 1.0e-6,
) -> TangentMeanFreeProjection:
    centered_v, mean_before = center_velocity(v)
    J0 = mean_free_jacobian(J, num_atoms=int(v.shape[0]))
    projector = tangent_projector(J0, metric=metric, damping=damping)
    projected = projector.project(centered_v)
    projected, mean_after = center_velocity(projected)
    tangent_residual = float(
        torch.linalg.norm((projected.reshape(-1) - projector.project(projected).reshape(-1))).detach().item()
    )
    return TangentMeanFreeProjection(
        velocity=projected,
        projector_rank=int(projector.rank),
        projector_condition=float(projector.condition_number),
        tangent_residual=tangent_residual,
        mean_norm_before=float(mean_before),
        mean_norm_after=float(mean_after),
        J0=J0.detach().clone(),
    )


def casal_coordinate_residual(
    f: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    *,
    mean_free: bool = True,
) -> tuple[torch.Tensor, float]:
    residual = wrapdiff(f, z) + mu
    mean_norm = 0.0
    if mean_free:
        residual, mean_norm = center_velocity(residual)
    return residual, float(mean_norm)


@dataclass(frozen=True)
class VelocityImpulseConfig:
    sign_chi: float = -1.0
    projector_damping: float = 1.0e-6
    mean_free_velocity: bool = True
    impulse_clip_norm: float | None = None
    kappa_coord: float = 1.0
    projection: FixedTemplateVelocityConfig = field(default_factory=FixedTemplateVelocityConfig)
    relaxed_target_mode: str = "cascal"
    dual_mode: str = "tau_over_rho"
    dual_clip_norm: float | None = None
    tangent_mode: str = "after_move_combined"


@dataclass(frozen=True)
class VelocityImpulseDiagnostics:
    beta: float
    tau_c: float
    rho: float
    h_coord: float
    beta_over_h: float
    kappa_coord: float
    sign_chi: float
    residual_norm: float
    residual_mean_norm: float
    proposal_velocity_norm: float
    proposal_mean_norm: float
    impulse_norm: float
    impulse_scale: float
    impulse_clipped: bool
    corrected_velocity_norm: float
    corrected_mean_norm: float
    proposal_distance_to_old_z: float
    impulse_distance_to_old_z: float
    post_z_distance: float
    dual_norm: float
    dual_mean_norm: float
    dual_step_norm: float
    tangent_rank: int
    tangent_condition: float
    tangent_residual: float
    theorem_applies: bool = CASAL_THEOREM_APPLIES
    theorem_reason: str = CASAL_THEOREM_REASON


@dataclass(frozen=True)
class VelocityImpulseStepOutput:
    f_next: torch.Tensor
    v_next: torch.Tensor
    z_next: torch.Tensor
    mu_next: torch.Tensor
    projection: any
    jacobian: torch.Tensor
    v_tilde: torch.Tensor
    dv: torch.Tensor
    y_relaxed: torch.Tensor
    diagnostics: VelocityImpulseDiagnostics


def clip_tensor_norm(x: torch.Tensor, *, max_norm: float | None) -> tuple[torch.Tensor, bool]:
    if max_norm is None or max_norm <= 0:
        return x, False
    norm = float(torch.linalg.norm(x.reshape(-1)).detach().item())
    if not math.isfinite(norm) or norm <= float(max_norm):
        return x, False
    scale = float(max_norm) / max(norm, 1.0e-12)
    return x * scale, True


def build_relaxed_target(
    *,
    f_next: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    beta: float,
    mode: str,
) -> torch.Tensor:
    mode0 = str(mode).strip().lower()
    if mode0 in {"cascal", "faithful", "relaxed"}:
        q, _ = casal_coordinate_residual(f_next, z, mu, mean_free=True)
        return z + float(beta) * q
    if mode0 in {"direct", "project_f"}:
        return f_next
    if mode0 in {"shortcut", "f_plus_mu"}:
        return f_next + mu
    raise ValueError(f"Unsupported relaxed_target_mode={mode!r}")


def update_dual(
    *,
    mu: torch.Tensor,
    f_next: torch.Tensor,
    z_next: torch.Tensor,
    tau_c: float,
    rho: float,
    mode: str,
    clip_norm: float | None = None,
) -> tuple[torch.Tensor, bool, float]:
    mode0 = str(mode).strip().lower()
    e, _ = casal_coordinate_residual(f_next, z_next, torch.zeros_like(mu), mean_free=True)
    if mode0 in {"none", "off"}:
        mu_next = mu.detach().clone()
        dual_step = torch.zeros_like(mu)
    elif mode0 in {"tau_over_rho", "faithful", "cascal"}:
        dual_step = (float(tau_c) / max(float(rho), 1.0e-12)) * e
        mu_next = mu + dual_step
    elif mode0 in {"tau_times_rho", "aggressive", "wrong"}:
        dual_step = float(tau_c) * float(rho) * e
        mu_next = mu + dual_step
    else:
        raise ValueError(f"Unsupported dual_mode={mode!r}")
    mu_next, _ = center_velocity(mu_next)
    mu_next, clipped = clip_tensor_norm(mu_next, max_norm=clip_norm)
    if clipped:
        mu_next, _ = center_velocity(mu_next)
    return mu_next, clipped, float(torch.linalg.norm(dual_step.reshape(-1)).detach().item())


def build_velocity_impulse(
    *,
    proposal_velocity: torch.Tensor,
    residual: torch.Tensor,
    beta: float,
    h_coord: float,
    sign_chi: float,
    mean_free: bool = True,
    clip_norm: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, bool, float]:
    scale = -float(beta) / (float(sign_chi) * max(float(h_coord), 1.0e-12))
    dv = scale * residual
    dv, clipped = clip_tensor_norm(dv, max_norm=clip_norm)
    v_tilde = proposal_velocity + dv
    mean_norm = 0.0
    if mean_free:
        v_tilde, mean_norm = center_velocity(v_tilde)
    return v_tilde, dv, float(scale), bool(clipped), float(mean_norm)


def kldm_casal_velocity_impulse_step(
    *,
    f: torch.Tensor,
    proposal_velocity: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    target_k: torch.Tensor,
    atomic_numbers: torch.Tensor,
    template_state: PCSTemplateState,
    tau_current: torch.Tensor,
    h_coord: float,
    rho: float,
    beta: float | None = None,
    tau_c: float | None = None,
    metric: torch.Tensor | None = None,
    config: VelocityImpulseConfig | None = None,
) -> VelocityImpulseStepOutput:
    cfg = config or VelocityImpulseConfig()
    h_coord0 = max(float(h_coord), 1.0e-12)
    tau_c0 = float(tau_c) if tau_c is not None else float(cfg.kappa_coord) * h_coord0
    beta0 = float(beta) if beta is not None else tau_c0 * float(rho)
    proposal_velocity0 = proposal_velocity.detach().clone()
    proposal_velocity_norm = float(torch.linalg.norm(proposal_velocity0.reshape(-1)).detach().item())
    proposal_velocity0, proposal_mean_norm = center_velocity(proposal_velocity0)
    residual, residual_mean_norm = casal_coordinate_residual(
        f,
        z,
        mu,
        mean_free=bool(cfg.mean_free_velocity),
    )
    v_tilde, dv, impulse_scale, impulse_clipped, corrected_mean_norm = build_velocity_impulse(
        proposal_velocity=proposal_velocity0,
        residual=residual,
        beta=beta0,
        h_coord=h_coord0,
        sign_chi=cfg.sign_chi,
        mean_free=bool(cfg.mean_free_velocity),
        clip_norm=cfg.impulse_clip_norm,
    )
    proposal_f = kinetic_position_update_signed(
        f,
        proposal_velocity0,
        h_coord=h_coord0,
        sign_chi=cfg.sign_chi,
    )
    f_next = kinetic_position_update_signed(
        f,
        v_tilde,
        h_coord=h_coord0,
        sign_chi=cfg.sign_chi,
    )
    y_relaxed = build_relaxed_target(
        f_next=f_next,
        z=z,
        mu=mu,
        beta=beta0,
        mode=cfg.relaxed_target_mode,
    )
    projection = project_to_fixed_template(
        f_frac=y_relaxed,
        atomic_numbers=atomic_numbers,
        template_state=template_state,
        target_k=target_k,
        tau0=tau_current,
        config=cfg.projection,
    )
    z_next = projection.z_frac.detach().clone()
    mu_next, _dual_clipped, dual_step_norm = update_dual(
        mu=mu,
        f_next=f_next,
        z_next=z_next,
        tau_c=tau_c0,
        rho=rho,
        mode=cfg.dual_mode,
        clip_norm=cfg.dual_clip_norm,
    )
    jacobian = compute_template_jacobian(projection.theta, projection.raw.state, tau=projection.tau)
    if str(cfg.tangent_mode).strip().lower() in {"none", "off"}:
        v_next, _ = center_velocity(v_tilde)
        tangent_rank = 0
        tangent_condition = float("nan")
        tangent_residual = float("nan")
    else:
        tangent = tangent_project_mean_free(
            v_tilde,
            J=jacobian,
            metric=metric,
            damping=cfg.projector_damping,
        )
        v_next = tangent.velocity
        tangent_rank = tangent.projector_rank
        tangent_condition = tangent.projector_condition
        tangent_residual = tangent.tangent_residual
    proposal_distance_to_old_z = float(torch.linalg.norm(wrapdiff(proposal_f, z).reshape(-1)).detach().item())
    impulse_distance_to_old_z = float(torch.linalg.norm(wrapdiff(f_next, z).reshape(-1)).detach().item())
    post_z_distance = float(torch.linalg.norm(wrapdiff(f_next, z_next).reshape(-1)).detach().item())
    diagnostics = VelocityImpulseDiagnostics(
        beta=float(beta0),
        tau_c=float(tau_c0),
        rho=float(rho),
        h_coord=float(h_coord0),
        beta_over_h=float(beta0) / max(float(h_coord0), 1.0e-12),
        kappa_coord=float(cfg.kappa_coord),
        sign_chi=float(cfg.sign_chi),
        residual_norm=float(torch.linalg.norm(residual.reshape(-1)).detach().item()),
        residual_mean_norm=float(residual_mean_norm),
        proposal_velocity_norm=float(proposal_velocity_norm),
        proposal_mean_norm=float(proposal_mean_norm),
        impulse_norm=float(torch.linalg.norm(dv.reshape(-1)).detach().item()),
        impulse_scale=float(impulse_scale),
        impulse_clipped=bool(impulse_clipped),
        corrected_velocity_norm=float(torch.linalg.norm(v_tilde.reshape(-1)).detach().item()),
        corrected_mean_norm=float(corrected_mean_norm),
        proposal_distance_to_old_z=float(proposal_distance_to_old_z),
        impulse_distance_to_old_z=float(impulse_distance_to_old_z),
        post_z_distance=float(post_z_distance),
        dual_norm=float(torch.linalg.norm(mu_next.reshape(-1)).detach().item()),
        dual_mean_norm=float(torch.linalg.norm(mu_next.mean(dim=0)).detach().item()),
        dual_step_norm=float(dual_step_norm),
        tangent_rank=int(tangent_rank),
        tangent_condition=float(tangent_condition),
        tangent_residual=float(tangent_residual),
    )
    return VelocityImpulseStepOutput(
        f_next=f_next,
        v_next=v_next,
        z_next=z_next,
        mu_next=mu_next,
        projection=projection,
        jacobian=jacobian,
        v_tilde=v_tilde,
        dv=dv,
        y_relaxed=y_relaxed,
        diagnostics=diagnostics,
    )
