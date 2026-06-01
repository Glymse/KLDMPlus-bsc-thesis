from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from kldmPlus.algorithm13_fixed_template_velocity_casal import (
    FixedTemplateVelocityConfig,
    compute_template_jacobian,
    project_to_fixed_template,
    project_to_fixed_template_local,
)
from kldmPlus.algorithm14_kldm_casal_velocity_impulse import (
    CASAL_THEOREM_APPLIES,
    CASAL_THEOREM_REASON,
    center_velocity,
    kinetic_position_update_signed,
    tangent_project_mean_free,
    wrapdiff,
)
from kldmPlus.symmetry.pcs_projection import PCSTemplateState


@dataclass(frozen=True)
class PhaseSpaceProjection:
    z_f: torch.Tensor
    z_v: torch.Tensor
    z_k: torch.Tensor
    z_l: torch.Tensor
    tau: torch.Tensor
    theta: torch.Tensor
    jacobian: torch.Tensor
    rank_J: int
    cond_J: float
    d_f: float
    d_v: float
    velocity_survival_ratio: float
    mean_velocity_norm: float
    idempotence_error: float
    projection_success: bool
    projection: any


@dataclass(frozen=True)
class PhaseSpaceCasalConfig:
    sign_chi: float = -1.0
    projector_damping: float = 1.0e-6
    mean_free_velocity: bool = True
    kappa: float = 0.1
    rho_f: float = 1.0
    rho_v: float = 0.0
    dual_mode: str = "no_dual"
    projection: FixedTemplateVelocityConfig = field(default_factory=FixedTemplateVelocityConfig)
    idempotence_tol: float = 1.0e-3
    score_ratio_max: float = 3.0
    velocity_ratio_max: float = 5.0
    min_pair_distance_min: float = 0.5
    volume_ratio_min: float = 0.5
    volume_ratio_max: float = 1.5
    accept_tol: float = 1.0e-8


@dataclass(frozen=True)
class PhaseSpaceResiduals:
    r_f: torch.Tensor
    r_v: torch.Tensor
    r_f_mean_norm: float
    r_v_mean_norm: float


@dataclass(frozen=True)
class PhaseSpaceCasalDiagnostics:
    tau: float
    h: float
    rho_f: float
    rho_v: float
    d_f_before: float
    d_v_before: float
    d_f_after_kldm: float
    d_v_after_kldm: float
    d_f_after_casal: float
    d_v_after_casal: float
    d_f_reduction: float
    d_v_reduction: float
    velocity_survival_ratio: float
    velocity_norm_before: float
    velocity_norm_after: float
    score_norm_before: float
    score_norm_after: float
    score_ratio: float
    projection_success: bool
    idempotence_error: float
    theorem_applies: bool = CASAL_THEOREM_APPLIES
    theorem_reason: str = CASAL_THEOREM_REASON


@dataclass(frozen=True)
class PhaseSpaceCasalStepOutput:
    f_hat: torch.Tensor
    v_hat: torch.Tensor
    f_x: torch.Tensor
    v_x: torch.Tensor
    z_f_next: torch.Tensor
    z_v_next: torch.Tensor
    mu_f_next: torch.Tensor
    mu_v_next: torch.Tensor
    projection_old: PhaseSpaceProjection
    projection_new: PhaseSpaceProjection
    diagnostics: PhaseSpaceCasalDiagnostics
    accepted: bool
    failure_label: str


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(-1)).detach().item())


def _project_velocity(
    v: torch.Tensor,
    *,
    J: torch.Tensor,
    metric: torch.Tensor | None,
    damping: float,
) -> tuple[torch.Tensor, int, float, float, float]:
    centered_v, mean_norm = center_velocity(v)
    tangent = tangent_project_mean_free(
        centered_v,
        J=J,
        metric=metric,
        damping=damping,
    )
    z_v = tangent.velocity
    denom = max(_norm(centered_v), 1.0e-12)
    survival_ratio = _norm(z_v) / denom
    return z_v, tangent.projector_rank, float(tangent.projector_condition), float(tangent.tangent_residual), float(mean_norm)


def project_state_to_phase_space_constraint(
    *,
    f: torch.Tensor,
    v: torch.Tensor,
    atomic_numbers: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    metric: torch.Tensor | None = None,
    config: PhaseSpaceCasalConfig | None = None,
) -> PhaseSpaceProjection:
    cfg = config or PhaseSpaceCasalConfig()
    fixed_assignment = (
        template_state.fixed_target_assignment
        if template_state.fixed_target_assignment is not None
        else template_state.anchor_assignment
    )
    projection = project_to_fixed_template_local(
        f_frac=f,
        atomic_numbers=atomic_numbers,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=template_state.free_vars,
        fixed_assignment=fixed_assignment,
        config=cfg.projection,
    )
    z_f = projection.z_frac.detach().clone()
    jacobian = compute_template_jacobian(projection.theta, projection.raw.state, tau=projection.tau)
    z_v, rank_J, cond_J, _tang_residual, mean_velocity_norm = _project_velocity(
        v,
        J=jacobian,
        metric=metric,
        damping=cfg.projector_damping,
    )
    centered_v, _ = center_velocity(v)
    reproj = project_to_fixed_template_local(
        f_frac=z_f,
        atomic_numbers=atomic_numbers,
        template_state=projection.raw.state,
        target_k=target_k,
        tau0=projection.tau,
        theta0=projection.theta,
        fixed_assignment=projection.assignment,
        config=cfg.projection,
    )
    idempotence_error = _norm(wrapdiff(reproj.z_frac, z_f))
    return PhaseSpaceProjection(
        z_f=z_f,
        z_v=z_v,
        z_k=projection.z_k.detach().clone(),
        z_l=projection.z_l.detach().clone(),
        tau=projection.tau.detach().clone(),
        theta=projection.theta.detach().clone(),
        jacobian=jacobian.detach().clone(),
        rank_J=int(rank_J),
        cond_J=float(cond_J),
        d_f=_norm(wrapdiff(f, z_f)),
        d_v=_norm(centered_v - z_v),
        velocity_survival_ratio=_norm(z_v) / max(_norm(centered_v), 1.0e-12),
        mean_velocity_norm=float(mean_velocity_norm),
        idempotence_error=float(idempotence_error),
        projection_success=bool(torch.isfinite(z_f).all().item() and torch.isfinite(z_v).all().item()),
        projection=projection,
    )


def build_phase_space_residuals(
    *,
    f: torch.Tensor,
    v: torch.Tensor,
    z_f: torch.Tensor,
    z_v: torch.Tensor,
    mu_f: torch.Tensor,
    mu_v: torch.Tensor,
) -> PhaseSpaceResiduals:
    r_f, r_f_mean_norm = center_velocity(wrapdiff(f, z_f) + mu_f)
    r_v, r_v_mean_norm = center_velocity(v - z_v + mu_v)
    return PhaseSpaceResiduals(
        r_f=r_f,
        r_v=r_v,
        r_f_mean_norm=float(r_f_mean_norm),
        r_v_mean_norm=float(r_v_mean_norm),
    )


def _update_duals(
    *,
    mu_f: torch.Tensor,
    mu_v: torch.Tensor,
    f_x: torch.Tensor,
    v_x: torch.Tensor,
    z_f_next: torch.Tensor,
    z_v_next: torch.Tensor,
    tau: float,
    rho_f: float,
    rho_v: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode0 = str(mode).strip().lower()
    if mode0 in {"no_dual", "none", "off"}:
        return torch.zeros_like(mu_f), torch.zeros_like(mu_v)
    e_f, _ = center_velocity(wrapdiff(f_x, z_f_next))
    e_v, _ = center_velocity(v_x - z_v_next)
    mu_f_next = mu_f.detach().clone()
    mu_v_next = mu_v.detach().clone()
    if mode0 in {"mu_f_only", "coord_only"}:
        mu_f_next = mu_f + float(rho_f) * float(tau) * e_f
        mu_v_next = torch.zeros_like(mu_v)
    elif mode0 in {"mu_f_and_mu_v", "both", "phase_space"}:
        mu_f_next = mu_f + float(rho_f) * float(tau) * e_f
        mu_v_next = mu_v + float(rho_v) * float(tau) * e_v
    else:
        raise ValueError(f"Unsupported dual_mode={mode!r}")
    mu_f_next, _ = center_velocity(mu_f_next)
    mu_v_next, _ = center_velocity(mu_v_next)
    return mu_f_next, mu_v_next


def phase_space_casal_step(
    *,
    f: torch.Tensor,
    v: torch.Tensor,
    proposal_velocity: torch.Tensor,
    atomic_numbers: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau_current: torch.Tensor,
    h: float,
    metric: torch.Tensor | None = None,
    mu_f: torch.Tensor | None = None,
    mu_v: torch.Tensor | None = None,
    score_norm_before: float = float("nan"),
    score_norm_after: float = float("nan"),
    config: PhaseSpaceCasalConfig | None = None,
) -> PhaseSpaceCasalStepOutput:
    cfg = config or PhaseSpaceCasalConfig()
    h0 = max(float(h), 1.0e-12)
    tau = float(cfg.kappa) * h0
    mu_f0 = torch.zeros_like(f) if mu_f is None else mu_f.detach().clone()
    mu_v0 = torch.zeros_like(v) if mu_v is None else mu_v.detach().clone()

    projection_old = project_state_to_phase_space_constraint(
        f=f,
        v=v,
        atomic_numbers=atomic_numbers,
        template_state=template_state,
        target_k=target_k,
        tau0=tau_current,
        metric=metric,
        config=cfg,
    )
    v_hat, _ = center_velocity(proposal_velocity)
    f_hat = kinetic_position_update_signed(
        f,
        v_hat,
        h_coord=h0,
        sign_chi=cfg.sign_chi,
    )
    z_hat = project_state_to_phase_space_constraint(
        f=f_hat,
        v=v_hat,
        atomic_numbers=atomic_numbers,
        template_state=projection_old.projection.raw.state,
        target_k=target_k,
        tau0=projection_old.tau,
        metric=metric,
        config=cfg,
    )
    residuals = build_phase_space_residuals(
        f=f,
        v=v,
        z_f=projection_old.z_f,
        z_v=projection_old.z_v,
        mu_f=mu_f0,
        mu_v=mu_v0,
    )
    v_x = v_hat + h0 * tau * float(cfg.rho_f) * residuals.r_f - tau * float(cfg.rho_v) * residuals.r_v
    v_x, _ = center_velocity(v_x)
    f_x = kinetic_position_update_signed(
        f,
        v_x,
        h_coord=h0,
        sign_chi=cfg.sign_chi,
    )
    y_f = projection_old.z_f + tau * float(cfg.rho_f) * center_velocity(wrapdiff(f_x, projection_old.z_f) + mu_f0)[0]
    y_v = projection_old.z_v + tau * float(cfg.rho_v) * center_velocity(v_x + mu_v0 - projection_old.z_v)[0]
    projection_new_coord = project_to_fixed_template_local(
        f_frac=y_f,
        atomic_numbers=atomic_numbers,
        template_state=projection_old.projection.raw.state,
        target_k=target_k,
        tau0=projection_old.tau,
        theta0=projection_old.theta,
        fixed_assignment=projection_old.projection.assignment,
        config=cfg.projection,
    )
    jacobian_new = compute_template_jacobian(
        projection_new_coord.theta,
        projection_new_coord.raw.state,
        tau=projection_new_coord.tau,
    )
    z_v_next, rank_J, cond_J, _tangent_residual, mean_velocity_norm = _project_velocity(
        y_v,
        J=jacobian_new,
        metric=metric,
        damping=cfg.projector_damping,
    )
    projection_new = PhaseSpaceProjection(
        z_f=projection_new_coord.z_frac.detach().clone(),
        z_v=z_v_next.detach().clone(),
        z_k=projection_new_coord.z_k.detach().clone(),
        z_l=projection_new_coord.z_l.detach().clone(),
        tau=projection_new_coord.tau.detach().clone(),
        theta=projection_new_coord.theta.detach().clone(),
        jacobian=jacobian_new.detach().clone(),
        rank_J=int(rank_J),
        cond_J=float(cond_J),
        d_f=_norm(wrapdiff(f_x, projection_new_coord.z_frac)),
        d_v=_norm(center_velocity(v_x)[0] - z_v_next),
        velocity_survival_ratio=_norm(z_v_next) / max(_norm(center_velocity(y_v)[0]), 1.0e-12),
        mean_velocity_norm=float(mean_velocity_norm),
        idempotence_error=_norm(
            wrapdiff(
                project_to_fixed_template_local(
                    f_frac=projection_new_coord.z_frac,
                    atomic_numbers=atomic_numbers,
                    template_state=projection_new_coord.raw.state,
                    target_k=target_k,
                    tau0=projection_new_coord.tau,
                    theta0=projection_new_coord.theta,
                    fixed_assignment=projection_new_coord.assignment,
                    config=cfg.projection,
                ).z_frac,
                projection_new_coord.z_frac,
            )
        ),
        projection_success=bool(torch.isfinite(projection_new_coord.z_frac).all().item() and torch.isfinite(z_v_next).all().item()),
        projection=projection_new_coord,
    )
    mu_f_next, mu_v_next = _update_duals(
        mu_f=mu_f0,
        mu_v=mu_v0,
        f_x=f_x,
        v_x=v_x,
        z_f_next=projection_new.z_f,
        z_v_next=projection_new.z_v,
        tau=tau,
        rho_f=float(cfg.rho_f),
        rho_v=float(cfg.rho_v),
        mode=cfg.dual_mode,
    )
    score_ratio = (
        float(score_norm_after) / max(float(score_norm_before), 1.0e-12)
        if math.isfinite(float(score_norm_before)) and math.isfinite(float(score_norm_after))
        else float("nan")
    )
    accepted = bool(
        projection_new.projection_success
        and projection_new.idempotence_error <= float(cfg.idempotence_tol)
        and projection_new.d_f <= z_hat.d_f + float(cfg.accept_tol)
        and projection_new.d_v <= z_hat.d_v + float(cfg.accept_tol)
        and (not math.isfinite(score_ratio) or score_ratio < float(cfg.score_ratio_max))
    )
    failure_label = ""
    if not projection_new.projection_success:
        failure_label = "COORD_PROJECTION_FAIL"
    elif projection_new.idempotence_error > float(cfg.idempotence_tol):
        failure_label = "Z_BRANCH_SWITCH"
    elif projection_new.d_f > z_hat.d_f + float(cfg.accept_tol):
        failure_label = "DF_NOT_REDUCED"
    elif projection_new.d_v > z_hat.d_v + float(cfg.accept_tol):
        failure_label = "DV_NOT_REDUCED"
    elif math.isfinite(score_ratio) and score_ratio >= float(cfg.score_ratio_max):
        failure_label = "SCORE_OOD"

    diagnostics = PhaseSpaceCasalDiagnostics(
        tau=float(tau),
        h=float(h0),
        rho_f=float(cfg.rho_f),
        rho_v=float(cfg.rho_v),
        d_f_before=float(projection_old.d_f),
        d_v_before=float(projection_old.d_v),
        d_f_after_kldm=float(z_hat.d_f),
        d_v_after_kldm=float(z_hat.d_v),
        d_f_after_casal=float(projection_new.d_f),
        d_v_after_casal=float(projection_new.d_v),
        d_f_reduction=float(z_hat.d_f - projection_new.d_f),
        d_v_reduction=float(z_hat.d_v - projection_new.d_v),
        velocity_survival_ratio=float(projection_new.velocity_survival_ratio),
        velocity_norm_before=float(_norm(center_velocity(v)[0])),
        velocity_norm_after=float(_norm(center_velocity(v_x)[0])),
        score_norm_before=float(score_norm_before),
        score_norm_after=float(score_norm_after),
        score_ratio=float(score_ratio),
        projection_success=bool(projection_new.projection_success),
        idempotence_error=float(projection_new.idempotence_error),
    )
    return PhaseSpaceCasalStepOutput(
        f_hat=f_hat,
        v_hat=v_hat,
        f_x=f_x,
        v_x=v_x,
        z_f_next=projection_new.z_f,
        z_v_next=projection_new.z_v,
        mu_f_next=mu_f_next,
        mu_v_next=mu_v_next,
        projection_old=projection_old,
        projection_new=projection_new,
        diagnostics=diagnostics,
        accepted=accepted,
        failure_label=failure_label,
    )
