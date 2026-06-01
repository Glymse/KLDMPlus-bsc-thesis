from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch

from kldmPlus.algorithm13_fixed_template_velocity_casal import (
    FixedTemplateVelocityConfig,
    compute_template_jacobian,
    materialize_template,
    project_to_fixed_template_local,
    tangent_projector,
    wrap_residual,
)
from kldmPlus.algorithm14_kldm_casal_velocity_impulse import (
    center_velocity,
    tangent_project_mean_free,
)
from kldmPlus.symmetry.pcs_projection import PCSTemplateState
from kldmPlus.utils.time import iter_sampling_times, sampling_grid


PPR_ACTIVE_MODE = "deterministic_clean_project_renoise"
PPR_ACTIVE_MODE_IS_FULL_PPR = False
PPR_ACTIVE_MODE_SPACE = "x0-space"
PPR_ACTIVE_MODE_RENOISE = True
PPR_ACTIVE_MODE_PROPAGATES_THROUGH_DENOISER = False
LEGACY_VELOCITY_PPR_AVAILABLE = True

# Backwards-compatible name for older notebooks. The active mode is not full
# faithful PPR; it is the deterministic x0-space clean-project-renoise path.
PPR_FAITHFUL_MODE = PPR_ACTIVE_MODE
PPR_XT_THROUGH_DENOISER_AVAILABLE = False
PPR_XT_THROUGH_DENOISER_REASON = (
    "The active notebook deliberately uses a non-differentiable x0-space "
    "deterministic clean-project-renoise approximation. Full PPR would optimize "
    "a noisy KLDM state through a differentiable denoiser; that path is reserved "
    "for a future ablation and is not active here."
)
PPR_EXACT_CLEAN_V0_SUPPORTED = False
PPR_EXACT_CLEAN_V0_REASON = (
    "KLDM's exact forward kernel implementation in TrivialisedDiffusion.sample_noisy_state "
    "assumes the clean convention v0 = 0. Exact renoise with nonzero projected clean velocity "
    "is therefore not source-faithful in this repo."
)
PPR_CLEAN_ESTIMATOR_SOURCE = "deterministic_exp_reverse_coords__predictor_lattice"
PPR_CLEAN_ESTIMATOR_REASON = (
    "The clean estimate is produced by a deterministic KLDM reverse-exp/predictor surrogate "
    "driven by the score network. It is an approximate clean surrogate, not an exact "
    "posterior-mean denoiser, not PF-ODE, and not the facit predictor-corrector sampler."
)
PPR_LATTICE_POLICY = "keep_reverse_pc_lattice"
PPR_LATTICE_POLICY_REASON = (
    "The PPR path projects only fractional coordinates. The lattice branch is kept from the "
    "incoming reverse facitKLDM PC sampler state and is not projected."
)
PPR_VELOCITY_OBJECTIVE_REASON = (
    "Legacy velocity-through-denoiser objectives are kept for ablations only. "
    "They are not used by the active deterministic KLDM-Wyckoff CPR notebook."
)
DIFFCSPPP_CLSMP_GUIDED_MLP_PRESENT = False
DIFFCSPPP_CLSMP_GUIDED_MLP_REASON = (
    "No explicit CLSMP-guided MLP implementation is present in this repository. "
    "The notebook uses the same fixed conditions CLSMP would provide in practice: "
    "dataset/batch space_group, composition/atomic_numbers, and a fixed Wyckoff template W."
)


@dataclass(frozen=True)
class KLDMGraphState:
    f: torch.Tensor
    v: torch.Tensor
    l: torch.Tensor
    h: torch.Tensor
    k: torch.Tensor
    t: float
    dt: float
    graph_idx0: int


@dataclass(frozen=True)
class KLDMCleanEstimate:
    f0_hat: torch.Tensor
    v0_hat: torch.Tensor
    l0_hat: torch.Tensor
    steps: int
    estimator_mode: str


@dataclass(frozen=True)
class KLDMForwardRenoiseResult:
    f_t: torch.Tensor
    v_t: torch.Tensor
    l_t: torch.Tensor
    r_t: torch.Tensor
    epsilon_v: torch.Tensor
    epsilon_r: torch.Tensor
    noise_l: torch.Tensor
    finite_ok: bool
    mean_free_norm: float
    epsilon_v_mean_norm: float
    v_t_mean_norm_before_optional_centering: float
    v_t_mean_norm_after: float
    lattice_changed_norm: float
    f_v_kernel_consistency_ok: bool


@dataclass(frozen=True)
class PPRProjectionResult:
    z_f: torch.Tensor
    z_f_raw: torch.Tensor
    theta: torch.Tensor
    tau: torch.Tensor
    assignment: torch.Tensor
    idempotence_error: float
    idempotence_error_raw: float
    projection_success: bool
    branch_changed: bool
    assignment_changed: bool
    objective: float
    alignment_translation: torch.Tensor
    raw: Any


@dataclass(frozen=True)
class KLDMPPRStepResult:
    clean_estimate: KLDMCleanEstimate
    clean_after_renoise: KLDMCleanEstimate
    projection: PPRProjectionResult
    projection_after: PPRProjectionResult
    renoise: KLDMForwardRenoiseResult
    f0_proj: torch.Tensor
    v0_proj: torch.Tensor
    l0_renoise: torch.Tensor
    lattice_policy: str
    d_before: float
    d_after: float
    d_after_to_manifold: float
    d_after_to_initial_anchor: float
    lattice_changed_norm: float
    projection_safe: bool


@dataclass(frozen=True)
class KLDMPPRIterationResult:
    steps: tuple[KLDMPPRStepResult, ...]
    final_state: KLDMGraphState


@dataclass(frozen=True)
class VelocityPPROptimizationResult:
    objective_mode: str
    v_star: torch.Tensor
    theta_delta_star: torch.Tensor
    clean_before: KLDMCleanEstimate
    clean_after: KLDMCleanEstimate
    projection_before: PPRProjectionResult
    projection_after: PPRProjectionResult
    energy_before: float
    energy_after: float
    constraint_energy_before: float
    constraint_energy_after: float
    normal_energy_before: float
    normal_energy_after: float
    velocity_displacement: float
    velocity_norm_before: float
    velocity_norm_after: float
    mean_free_error_before: float
    mean_free_error_after: float
    objective_history: tuple[float, ...]
    best_iter_index: int
    clean_estimator_mode: str


@dataclass(frozen=True)
class VelocityPPRAcceptResult:
    mode: str
    optimized: VelocityPPROptimizationResult
    state_out: KLDMGraphState
    renoise: KLDMForwardRenoiseResult | None
    clean_projected_frac: torch.Tensor | None


@dataclass(frozen=True)
class AssignmentAwareChartAnchor:
    projection: PPRProjectionResult
    template_state: PCSTemplateState
    theta: torch.Tensor
    tau: torch.Tensor
    assignment: torch.Tensor
    z_ref: torch.Tensor
    jacobian: torch.Tensor
    rank_j: int
    condition_jt_j: float


@dataclass(frozen=True)
class NormalResidualEnergy:
    energy_normal: float
    energy_full: float
    rank_j: int
    condition_jt_j: float
    normal_residual_norm: float
    tangent_residual_norm: float


@dataclass(frozen=True)
class AssignmentAwareVelocityPPRConfig:
    objective_mode: str = "gauss_newton_normal"
    opt_steps: int = 6
    opt_lr: float = 1.0e-2
    lambda_v: float = 10.0
    lambda_theta: float = 1.0
    lambda_mf: float = 1.0
    lambda_norm: float = 2.0
    theta_trust_radius: float = 0.05
    projector_damping: float = 1.0e-6
    return_best_valid_iterate: bool = True


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(-1)).detach().item())


def _as_graph_l_batch(l: torch.Tensor) -> torch.Tensor:
    """Normalize lattice features to the score-network graph-batch shape [G, D]."""
    if l.ndim == 1:
        return l.unsqueeze(0)
    if l.ndim == 2:
        return l
    return l.reshape(l.shape[0], -1)


def _restore_graph_l_shape(l: torch.Tensor) -> torch.Tensor:
    """Return single-graph lattice features to the notebook-friendly flat shape [D]."""
    if l.ndim == 2 and l.shape[0] == 1:
        return l[0]
    return l


def _wrap_frac(frac: torch.Tensor) -> torch.Tensor:
    return torch.remainder(frac, 1.0)


def align_projected_translation(
    *,
    z_raw: torch.Tensor,
    reference_frac: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference_frac is None:
        tau = torch.zeros(3, device=z_raw.device, dtype=z_raw.dtype)
        return _wrap_frac(z_raw), tau
    ref = reference_frac.reshape_as(z_raw).to(device=z_raw.device, dtype=z_raw.dtype)
    residual = wrap_residual(ref, z_raw)
    tau = residual.mean(dim=0)
    aligned = _wrap_frac(z_raw + tau.unsqueeze(0))
    raw_err = _norm(wrap_residual(z_raw, ref))
    aligned_err = _norm(wrap_residual(aligned, ref))
    if aligned_err <= raw_err:
        return aligned, tau.detach().clone()
    tau = torch.zeros(3, device=z_raw.device, dtype=z_raw.dtype)
    return _wrap_frac(z_raw), tau


def _score_network_predict(model, *, t_graph, pos, v, h, l, node_index, edge_node_index):
    return model.score_network(
        t=t_graph,
        pos=pos,
        v=v,
        h=h,
        l=l,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )


def _reverse_exp_step_deterministic(
    *,
    sampling_tdm,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    score_v: torch.Tensor,
    index: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_t = sampling_tdm.wrap_displacements(f_t)

    dt_internal = torch.as_tensor(sampling_tdm.T * dt, device=v_t.device, dtype=v_t.dtype)
    exp_dt = torch.exp(dt_internal)
    expm1_dt = torch.expm1(dt_internal)
    score_scale = torch.as_tensor(sampling_tdm.vel_scale**2, device=v_t.device, dtype=v_t.dtype)

    v_prev = exp_dt * v_t + 2.0 * score_scale * expm1_dt * score_v
    f_prev = sampling_tdm.wrap_displacements(f_t - dt_internal * v_prev)
    return f_prev, v_prev


@torch.no_grad()
def deterministic_predictor_clean_estimate(
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
        f_t = state.f.detach().clone()
        v_t = state.v.detach().clone()
        l_t = _as_graph_l_batch(state.l.detach().clone())
        h_t = state.h.detach().clone()
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
            f_t, v_t = _reverse_exp_step_deterministic(
                sampling_tdm=model.tdm,
                f_t=f_t,
                v_t=v_t,
                score_v=score_v,
                index=node_index,
                dt=times.dt,
            )
            if hasattr(model.diffusion_l, "reverse_step_predictor"):
                l_t = model.diffusion_l.reverse_step_predictor(
                    t=times.now.lattice,
                    x_t=l_t,
                    pred=preds["l"],
                    dt=times.dt,
                    num_atoms=batch.num_atoms,
                )
            actual_steps += 1
        return KLDMCleanEstimate(
            f0_hat=f_t.detach().clone(),
            v0_hat=v_t.detach().clone(),
            l0_hat=_restore_graph_l_shape(l_t.detach().clone()),
            steps=int(actual_steps),
            estimator_mode="deterministic_exp_reverse_coords__predictor_lattice",
        )
    finally:
        if restore_training:
            score_network.train()


def differentiable_deterministic_clean_estimate(
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
            f_t, v_t = _reverse_exp_step_deterministic(
                sampling_tdm=model.tdm,
                f_t=f_t,
                v_t=v_t,
                score_v=score_v,
                index=node_index,
                dt=times.dt,
            )
            if hasattr(model.diffusion_l, "reverse_step_predictor"):
                l_t = model.diffusion_l.reverse_step_predictor(
                    t=times.now.lattice,
                    x_t=l_t,
                    pred=preds["l"],
                    dt=times.dt,
                    num_atoms=batch.num_atoms,
                )
            actual_steps += 1
        return KLDMCleanEstimate(
            f0_hat=f_t,
            v0_hat=v_t,
            l0_hat=_restore_graph_l_shape(l_t),
            steps=int(actual_steps),
            estimator_mode="differentiable_exp_reverse_coords__predictor_lattice",
        )
    finally:
        if restore_training:
            score_network.train()


@torch.no_grad()
def facit_pc_clean_estimate(
    *,
    model,
    batch,
    state: KLDMGraphState,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    n_steps: int,
    t_start: float,
    t_final: float = 1.0e-3,
    tau: float = 0.25,
    n_correction_steps: int = 1,
    seed: int | None = None,
) -> KLDMCleanEstimate:
    if seed is not None:
        torch.manual_seed(int(seed))
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
        f_t = state.f.detach().clone()
        v_t = state.v.detach().clone()
        l_t = _as_graph_l_batch(state.l.detach().clone())
        h_t = state.h.detach().clone()
        actual_steps = 0
        for times in iter_sampling_times(batch=batch, grid=grid):
            for _ in range(max(int(n_correction_steps), 1)):
                preds_corr = model.score_network(
                    t=times.now.graph,
                    pos=f_t,
                    v=v_t,
                    h=h_t,
                    l=l_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                )
                f_t, v_t = model.tdm.reverse_step_corrector(
                    t=times.now.nodes,
                    f_t=f_t,
                    v_t=v_t,
                    pred_v=preds_corr["v"],
                    dt=times.dt,
                    index=node_index,
                    tau=float(tau),
                )
                l_t = model.diffusion_l.reverse_step_corrector(
                    t=times.now.lattice,
                    x_t=l_t,
                    pred=preds_corr["l"],
                    tau=float(tau),
                )
            preds_pred = model.score_network(
                t=times.now.graph,
                pos=f_t,
                v=v_t,
                h=h_t,
                l=l_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )
            f_t, v_t = model.tdm.reverse_step_predictor(
                t=times.now.nodes,
                f_t=f_t,
                v_t=v_t,
                pred_v=preds_pred["v"],
                index=node_index,
                dt=times.dt,
            )
            l_t = model.diffusion_l.reverse_step_predictor(
                t=times.now.lattice,
                x_t=l_t,
                pred=preds_pred["l"],
                dt=times.dt,
                num_atoms=batch.num_atoms,
            )
            actual_steps += 1
        return KLDMCleanEstimate(
            f0_hat=f_t.detach().clone(),
            v0_hat=v_t.detach().clone(),
            l0_hat=_restore_graph_l_shape(l_t.detach().clone()),
            steps=int(actual_steps),
            estimator_mode="facit_pc_clean_estimate",
        )
    finally:
        if restore_training:
            score_network.train()


@torch.no_grad()
def facit_pc_best_of_k_clean_estimate(
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
    n_steps: int,
    t_start: float,
    t_final: float = 1.0e-3,
    tau: float = 0.25,
    n_correction_steps: int = 1,
    k: int = 4,
    seed0: int = 0,
) -> KLDMCleanEstimate:
    best: KLDMCleanEstimate | None = None
    best_key: tuple[float, float, float] | None = None
    for idx in range(max(int(k), 1)):
        clean = facit_pc_clean_estimate(
            model=model,
            batch=batch,
            state=state,
            node_index=node_index,
            edge_node_index=edge_node_index,
            n_steps=n_steps,
            t_start=t_start,
            t_final=t_final,
            tau=tau,
            n_correction_steps=n_correction_steps,
            seed=int(seed0) + idx,
        )
        proj = project_clean_to_fixed_wyckoff(
            f0_hat=clean.f0_hat,
            atomic_numbers=state.h,
            template_state=template_state,
            target_k=target_k,
            tau0=tau0,
            theta0=theta0,
            fixed_assignment=fixed_assignment,
            config=projection_config,
            reference_frac=clean.f0_hat,
        )
        min_pair = float(proj.raw.min_pair_distance)
        key = (
            0.0 if bool(proj.projection_success) else 1.0,
            float(-min_pair),
            float(proj.idempotence_error + proj.idempotence_error_raw + proj.objective),
        )
        if best is None or key < best_key:
            best = clean
            best_key = key
    assert best is not None
    return replace(best, estimator_mode=f"{best.estimator_mode}_best_of_{int(k)}")


@torch.no_grad()
def kldm_forward_renoise_exact(
    *,
    model,
    batch,
    f0: torch.Tensor,
    l0: torch.Tensor,
    t_graph: torch.Tensor,
    node_index: torch.Tensor,
    v0: torch.Tensor | None = None,
    noise_v: torch.Tensor | None = None,
    noise_r: torch.Tensor | None = None,
    noise_l: torch.Tensor | None = None,
    mean_free_velocity: bool = True,
    renoise_lattice: bool = True,
) -> KLDMForwardRenoiseResult:
    if v0 is not None and float(torch.linalg.norm(v0.reshape(-1)).detach().item()) > 1.0e-10:
        raise ValueError(PPR_EXACT_CLEAN_V0_REASON)
    if noise_v is None:
        noise_v = torch.randn_like(f0)
    if mean_free_velocity:
        noise_v, _ = center_velocity(noise_v)
    epsilon_v_mean_norm = float(torch.linalg.norm(noise_v.mean(dim=0)).detach().item())
    f_t, v_t, epsilon_v, epsilon_r, r_t = model.tdm.sample_noisy_state(
        t=t_graph,
        f0=f0,
        index=node_index,
        epsilon_v=noise_v,
        epsilon_r=noise_r,
    )
    v_t_mean_norm_before_optional_centering = float(torch.linalg.norm(v_t.mean(dim=0)).detach().item())
    if mean_free_velocity:
        v_t, mean_free_norm = center_velocity(v_t)
    else:
        mean_free_norm = float(torch.linalg.norm(v_t.mean(dim=0)).detach().item())
    v_t_mean_norm_after = float(torch.linalg.norm(v_t.mean(dim=0)).detach().item())
    l0 = _as_graph_l_batch(l0)
    if renoise_lattice:
        l_t, eps_l = model.diffusion_l.forward_sample(
            t=t_graph,
            x0=l0,
            noise=noise_l,
            num_atoms=batch.num_atoms,
        )
    else:
        l_t = l0.detach().clone()
        eps_l = torch.zeros_like(l_t)
    lattice_changed_norm = _norm(_restore_graph_l_shape(l_t.detach()) - _restore_graph_l_shape(l0.detach()))
    f_v_kernel_consistency_ok = bool((not mean_free_velocity) or float(epsilon_v_mean_norm) <= 1.0e-6)
    finite_ok = bool(torch.isfinite(f_t).all().item() and torch.isfinite(v_t).all().item() and torch.isfinite(l_t).all().item())
    return KLDMForwardRenoiseResult(
        f_t=f_t.detach().clone(),
        v_t=v_t.detach().clone(),
        l_t=_restore_graph_l_shape(l_t.detach().clone()),
        r_t=r_t.detach().clone(),
        epsilon_v=epsilon_v.detach().clone(),
        epsilon_r=epsilon_r.detach().clone(),
        noise_l=_restore_graph_l_shape(eps_l.detach().clone()),
        finite_ok=finite_ok,
        mean_free_norm=float(mean_free_norm),
        epsilon_v_mean_norm=float(epsilon_v_mean_norm),
        v_t_mean_norm_before_optional_centering=float(v_t_mean_norm_before_optional_centering),
        v_t_mean_norm_after=float(v_t_mean_norm_after),
        lattice_changed_norm=float(lattice_changed_norm),
        f_v_kernel_consistency_ok=f_v_kernel_consistency_ok,
    )


@torch.no_grad()
def kldm_forward_renoise_fv_only(
    *,
    model,
    batch,
    f0: torch.Tensor,
    l0: torch.Tensor,
    t_graph: torch.Tensor,
    node_index: torch.Tensor,
    v0: torch.Tensor | None = None,
    noise_v: torch.Tensor | None = None,
    noise_r: torch.Tensor | None = None,
    mean_free_velocity: bool = True,
) -> KLDMForwardRenoiseResult:
    """KLDM-native forward renoise for only the fractional/velocity branch.

    The active detKLDM-Wyckoff-CPR path keeps the facit reverse-PC lattice
    unchanged, so this wrapper deliberately disables lattice forward diffusion.
    """
    return kldm_forward_renoise_exact(
        model=model,
        batch=batch,
        f0=f0,
        l0=l0,
        t_graph=t_graph,
        node_index=node_index,
        v0=v0,
        noise_v=noise_v,
        noise_r=noise_r,
        noise_l=None,
        mean_free_velocity=mean_free_velocity,
        renoise_lattice=False,
    )


def project_clean_to_fixed_wyckoff(
    *,
    f0_hat: torch.Tensor,
    atomic_numbers: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    theta0: torch.Tensor | None,
    fixed_assignment: torch.Tensor | None,
    config: FixedTemplateVelocityConfig,
    reference_frac: torch.Tensor | None = None,
) -> PPRProjectionResult:
    projection = project_to_fixed_template_local(
        f_frac=f0_hat,
        atomic_numbers=atomic_numbers,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        config=config,
    )
    z_raw = projection.z_frac.detach().clone()
    z_aligned, tau_align = align_projected_translation(
        z_raw=z_raw,
        reference_frac=reference_frac,
    )
    tau_aligned = torch.remainder(
        projection.tau.detach().clone().reshape(1, 3) + tau_align.detach().clone().reshape(1, 3),
        1.0,
    )
    aligned_state = replace(
        projection.raw.state,
        free_vars=projection.theta.detach().clone(),
        anchor_free_vars=projection.theta.detach().clone(),
        branch_frac_coords=z_aligned.detach().clone(),
        branch_atomic_numbers=projection.raw.atomic_numbers_chart.detach().clone(),
        branch_lattice_features=projection.raw.cell.detach().reshape(-1).clone(),
    )
    aligned_raw = replace(
        projection.raw,
        state=aligned_state,
        tau=tau_aligned.detach().clone(),
        frac_coords_chart=z_aligned.detach().clone(),
    )
    reproj = project_to_fixed_template_local(
        f_frac=z_aligned,
        atomic_numbers=atomic_numbers,
        template_state=aligned_state,
        target_k=target_k,
        tau0=tau_aligned,
        theta0=projection.theta,
        fixed_assignment=projection.assignment,
        config=config,
    )
    reproj_raw = reproj.z_frac.detach().clone()
    reproj_aligned, _ = align_projected_translation(
        z_raw=reproj_raw,
        reference_frac=z_aligned,
    )
    idempotence_error_raw = _norm(wrap_residual(reproj_raw, z_raw))
    idempotence_error = _norm(wrap_residual(reproj_aligned, z_aligned))
    assignment_changed = not torch.equal(reproj.assignment.reshape(-1), projection.assignment.reshape(-1))
    branch_changed = bool(idempotence_error > 1.0e-5)
    objective = float(projection.objective)
    projection_success = bool(
        torch.isfinite(z_aligned).all().item()
        and torch.isfinite(projection.theta).all().item()
        and math.isfinite(objective)
    )
    return PPRProjectionResult(
        z_f=z_aligned.detach().clone(),
        z_f_raw=z_raw,
        theta=projection.theta.detach().clone(),
        tau=tau_aligned.detach().clone(),
        assignment=projection.assignment.detach().clone(),
        idempotence_error=float(idempotence_error),
        idempotence_error_raw=float(idempotence_error_raw),
        projection_success=projection_success,
        branch_changed=branch_changed,
        assignment_changed=assignment_changed,
        objective=objective,
        alignment_translation=tau_align.detach().clone(),
        raw=aligned_raw,
    )


def ppr_projection_is_safe(proj: PPRProjectionResult, *, idempotence_threshold: float = 1.0e-5) -> bool:
    return bool(
        proj.projection_success
        and torch.isfinite(proj.z_f).all().item()
        and math.isfinite(float(proj.objective))
        and float(proj.idempotence_error) < float(idempotence_threshold)
        and not bool(proj.assignment_changed)
        and not bool(proj.branch_changed)
    )


def materialize_fixed_wyckoff(
    *,
    theta: torch.Tensor,
    template_state: PCSTemplateState,
    tau: torch.Tensor,
) -> torch.Tensor:
    return materialize_template(theta.reshape(-1), template_state, tau=tau).frac_coords


def wyckoff_jacobian(
    *,
    theta: torch.Tensor,
    template_state: PCSTemplateState,
    tau: torch.Tensor,
) -> torch.Tensor:
    return compute_template_jacobian(theta.reshape(-1), template_state, tau=tau)


def build_assignment_chart_anchor(projection: PPRProjectionResult) -> AssignmentAwareChartAnchor:
    theta = projection.theta.detach().clone().reshape(-1)
    tau = projection.tau.detach().clone().reshape(1, 3)
    template_state = projection.raw.state
    jacobian = wyckoff_jacobian(theta=theta, template_state=template_state, tau=tau)
    projector = tangent_projector(jacobian, damping=1.0e-6)
    return AssignmentAwareChartAnchor(
        projection=projection,
        template_state=template_state,
        theta=theta,
        tau=tau,
        assignment=projection.assignment.detach().clone(),
        z_ref=projection.z_f.detach().clone(),
        jacobian=jacobian.detach().clone(),
        rank_j=int(projector.rank),
        condition_jt_j=float(projector.condition_number),
    )


def normal_residual_energy(
    *,
    f_hat: torch.Tensor,
    anchor: AssignmentAwareChartAnchor,
    theta_delta: torch.Tensor | None = None,
    damping: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    theta = anchor.theta
    tau = anchor.tau
    template_state = anchor.template_state
    if theta_delta is not None and int(theta_delta.numel()) == int(theta.numel()):
        theta = theta + theta_delta.to(device=theta.device, dtype=theta.dtype)
    z_ref = materialize_fixed_wyckoff(theta=theta, template_state=template_state, tau=tau).to(
        device=f_hat.device,
        dtype=f_hat.dtype,
    )
    jacobian = wyckoff_jacobian(theta=theta, template_state=template_state, tau=tau).to(
        device=f_hat.device,
        dtype=f_hat.dtype,
    )
    residual = wrap_residual(f_hat, z_ref).reshape(-1)
    if jacobian.ndim != 2 or jacobian.shape[1] == 0:
        tangent = torch.zeros_like(residual)
        normal = residual
        rank_j = 0
        condition_jt_j = float("inf")
    else:
        projector = tangent_projector(jacobian, damping=float(damping))
        tangent = projector.project_flat(residual)
        normal = residual - tangent
        rank_j = int(projector.rank)
        condition_jt_j = float(projector.condition_number)
    energy_normal = torch.mean(normal ** 2)
    energy_full = torch.mean(residual ** 2)
    stats = {
        "z_ref": z_ref,
        "jacobian": jacobian,
        "residual": residual,
        "tangent": tangent,
        "normal": normal,
        "energy_normal": energy_normal,
        "energy_full": energy_full,
        "rank_j": float(rank_j),
        "condition_jt_j": float(condition_jt_j),
        "normal_residual_norm": torch.linalg.norm(normal),
        "tangent_residual_norm": torch.linalg.norm(tangent),
    }
    return energy_normal, stats


def velocity_ppr_objective(
    *,
    model,
    batch,
    f_t: torch.Tensor,
    v_candidate: torch.Tensor,
    l_t: torch.Tensor,
    h: torch.Tensor,
    t: float,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    v_reference: torch.Tensor,
    template_state: PCSTemplateState,
    target_k: torch.Tensor,
    tau0: torch.Tensor,
    theta0: torch.Tensor | None,
    fixed_assignment: torch.Tensor | None,
    projection_config: FixedTemplateVelocityConfig,
    clean_estimator_steps: int,
    clean_estimator_t_final: float,
    lambda_v: float = 1.0,
    lambda_mf: float = 1.0,
    lambda_norm: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    clean = differentiable_deterministic_clean_estimate(
        model=model,
        batch=batch,
        state=KLDMGraphState(
            f=f_t,
            v=v_candidate,
            l=l_t,
            h=h,
            k=target_k,
            t=t,
            dt=1.0 / max(int(clean_estimator_steps), 1),
            graph_idx0=0,
        ),
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(t),
        t_final=float(clean_estimator_t_final),
    )
    with torch.no_grad():
        proj = project_clean_to_fixed_wyckoff(
            f0_hat=clean.f0_hat.detach(),
            atomic_numbers=h.detach(),
            template_state=template_state,
            target_k=target_k.detach(),
            tau0=tau0.detach(),
            theta0=None if theta0 is None else theta0.detach(),
            fixed_assignment=None if fixed_assignment is None else fixed_assignment.detach(),
            config=projection_config,
            reference_frac=clean.f0_hat.detach(),
        )
        z_target = proj.z_f.detach()
    energy_clean = torch.mean(wrap_residual(clean.f0_hat, z_target).reshape(-1) ** 2)
    v_ref = v_reference.detach()
    prox_v = torch.mean((v_candidate - v_ref).reshape(-1) ** 2)
    mean_pen = torch.mean(v_candidate.mean(dim=0) ** 2)
    norm_pen = (torch.linalg.norm(v_candidate.reshape(-1)) - torch.linalg.norm(v_ref.reshape(-1))) ** 2
    loss = energy_clean + float(lambda_v) * prox_v + float(lambda_mf) * mean_pen + float(lambda_norm) * norm_pen
    aux = {
        "clean_estimate": clean,
        "projection": proj,
        "energy_clean": float(energy_clean.detach().item()),
        "proximity_v": float(prox_v.detach().item()),
        "mean_free_penalty": float(mean_pen.detach().item()),
        "norm_penalty": float(norm_pen.detach().item()),
    }
    return loss, aux


def optimize_velocity_through_clean_estimator(
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
    clean_estimator_steps: int,
    clean_estimator_t_final: float,
    opt_steps: int = 8,
    opt_lr: float = 5.0e-2,
    lambda_v: float = 1.0,
    lambda_mf: float = 1.0,
    lambda_norm: float = 0.0,
) -> VelocityPPROptimizationResult:
    batch_state = KLDMGraphState(
        f=state.f.detach().clone(),
        v=state.v.detach().clone(),
        l=state.l.detach().clone(),
        h=state.h.detach().clone(),
        k=target_k.detach().clone(),
        t=state.t,
        dt=state.dt,
        graph_idx0=state.graph_idx0,
    )
    clean_before = deterministic_predictor_clean_estimate(
        model=model,
        batch=batch,
        state=batch_state,
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(state.t),
        t_final=float(clean_estimator_t_final),
    )
    proj_before = project_clean_to_fixed_wyckoff(
        f0_hat=clean_before.f0_hat,
        atomic_numbers=state.h,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        config=projection_config,
        reference_frac=clean_before.f0_hat,
    )
    energy_before = _norm(wrap_residual(clean_before.f0_hat, proj_before.z_f)) ** 2
    v_ref = state.v.detach().clone()
    v_var = v_ref.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_var], lr=float(opt_lr))
    history: list[float] = []
    for _ in range(max(int(opt_steps), 1)):
        optimizer.zero_grad()
        v_centered, _ = center_velocity(v_var)
        loss, _aux = velocity_ppr_objective(
            model=model,
            batch=batch,
            f_t=state.f.detach(),
            v_candidate=v_centered,
            l_t=state.l.detach(),
            h=state.h.detach(),
            t=float(state.t),
            node_index=node_index,
            edge_node_index=edge_node_index,
            v_reference=v_ref,
            template_state=template_state,
            target_k=target_k,
            tau0=tau0,
            theta0=theta0,
            fixed_assignment=fixed_assignment,
            projection_config=projection_config,
            clean_estimator_steps=clean_estimator_steps,
            clean_estimator_t_final=clean_estimator_t_final,
            lambda_v=lambda_v,
            lambda_mf=lambda_mf,
            lambda_norm=lambda_norm,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            v_centered, _ = center_velocity(v_var)
            v_var.copy_(v_centered)
            history.append(float(loss.detach().item()))
    v_star, mean_after = center_velocity(v_var.detach().clone())
    clean_after = deterministic_predictor_clean_estimate(
        model=model,
        batch=batch,
        state=replace(batch_state, v=v_star.detach().clone()),
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(state.t),
        t_final=float(clean_estimator_t_final),
    )
    proj_after = project_clean_to_fixed_wyckoff(
        f0_hat=clean_after.f0_hat,
        atomic_numbers=state.h,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        config=projection_config,
        reference_frac=clean_after.f0_hat,
    )
    energy_after = _norm(wrap_residual(clean_after.f0_hat, proj_after.z_f)) ** 2
    _, mean_before = center_velocity(v_ref)
    return VelocityPPROptimizationResult(
        objective_mode="detached_projection",
        v_star=v_star.detach().clone(),
        theta_delta_star=torch.zeros_like(proj_after.theta.detach().clone()),
        clean_before=clean_before,
        clean_after=clean_after,
        projection_before=proj_before,
        projection_after=proj_after,
        energy_before=float(energy_before),
        energy_after=float(energy_after),
        constraint_energy_before=float(energy_before),
        constraint_energy_after=float(energy_after),
        normal_energy_before=float(energy_before),
        normal_energy_after=float(energy_after),
        velocity_displacement=float(torch.linalg.norm((v_star - v_ref).reshape(-1)).detach().item()),
        velocity_norm_before=float(torch.linalg.norm(v_ref.reshape(-1)).detach().item()),
        velocity_norm_after=float(torch.linalg.norm(v_star.reshape(-1)).detach().item()),
        mean_free_error_before=float(mean_before),
        mean_free_error_after=float(mean_after),
        objective_history=tuple(history),
        best_iter_index=max(len(history) - 1, 0),
        clean_estimator_mode=clean_after.estimator_mode,
    )


def assignment_aware_velocity_ppr_objective(
    *,
    model,
    batch,
    f_t: torch.Tensor,
    v_candidate: torch.Tensor,
    l_t: torch.Tensor,
    h: torch.Tensor,
    t: float,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    v_reference: torch.Tensor,
    chart_anchor: AssignmentAwareChartAnchor,
    target_k: torch.Tensor,
    clean_estimator_steps: int,
    clean_estimator_t_final: float,
    objective_mode: str,
    lambda_v: float,
    lambda_theta: float,
    lambda_mf: float,
    lambda_norm: float,
    theta_delta: torch.Tensor | None = None,
    damping: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    clean = differentiable_deterministic_clean_estimate(
        model=model,
        batch=batch,
        state=KLDMGraphState(
            f=f_t,
            v=v_candidate,
            l=l_t,
            h=h,
            k=target_k,
            t=t,
            dt=1.0 / max(int(clean_estimator_steps), 1),
            graph_idx0=0,
        ),
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(t),
        t_final=float(clean_estimator_t_final),
    )
    theta_penalty = torch.zeros((), device=clean.f0_hat.device, dtype=clean.f0_hat.dtype)
    if objective_mode == "detached_projection":
        with torch.no_grad():
            proj = project_clean_to_fixed_wyckoff(
                f0_hat=clean.f0_hat.detach(),
                atomic_numbers=h.detach(),
                template_state=chart_anchor.template_state,
                target_k=target_k.detach(),
                tau0=chart_anchor.tau.detach(),
                theta0=chart_anchor.theta.detach(),
                fixed_assignment=chart_anchor.assignment.detach(),
                config=FixedTemplateVelocityConfig(projector_damping=float(damping)),
                reference_frac=clean.f0_hat.detach(),
            )
            z_target = proj.z_f.detach()
        constraint_energy = torch.mean(wrap_residual(clean.f0_hat, z_target).reshape(-1) ** 2)
        normal_energy, normal_stats = normal_residual_energy(
            f_hat=clean.f0_hat,
            anchor=chart_anchor,
            damping=float(damping),
        )
    elif objective_mode in {"normal_chart", "gauss_newton_normal"}:
        constraint_energy, normal_stats = normal_residual_energy(
            f_hat=clean.f0_hat,
            anchor=chart_anchor,
            damping=float(damping),
        )
        normal_energy = constraint_energy
    elif objective_mode == "joint_v_theta":
        constraint_energy, normal_stats = normal_residual_energy(
            f_hat=clean.f0_hat,
            anchor=chart_anchor,
            theta_delta=theta_delta,
            damping=float(damping),
        )
        normal_energy = constraint_energy
        if theta_delta is not None and int(theta_delta.numel()) > 0:
            theta_penalty = torch.mean(theta_delta.reshape(-1) ** 2)
    else:
        raise ValueError(f"Unsupported objective_mode={objective_mode!r}")
    v_ref = v_reference.detach()
    prox_v = torch.mean((v_candidate - v_ref).reshape(-1) ** 2)
    mean_pen = torch.mean(v_candidate.mean(dim=0) ** 2)
    norm_pen = (torch.linalg.norm(v_candidate.reshape(-1)) - torch.linalg.norm(v_ref.reshape(-1))) ** 2
    loss = (
        constraint_energy
        + float(lambda_v) * prox_v
        + float(lambda_theta) * theta_penalty
        + float(lambda_mf) * mean_pen
        + float(lambda_norm) * norm_pen
    )
    return loss, {
        "clean_estimate": clean,
        "constraint_energy": float(constraint_energy.detach().item()),
        "normal_energy": float(normal_energy.detach().item()),
        "theta_penalty": float(theta_penalty.detach().item()),
        "proximity_v": float(prox_v.detach().item()),
        "mean_free_penalty": float(mean_pen.detach().item()),
        "norm_penalty": float(norm_pen.detach().item()),
        "normal_stats": normal_stats,
    }


def optimize_assignment_aware_velocity_ppr(
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
    clean_estimator_steps: int,
    clean_estimator_t_final: float,
    objective_config: AssignmentAwareVelocityPPRConfig,
    validity_check_fn: Any | None = None,
) -> VelocityPPROptimizationResult:
    batch_state = KLDMGraphState(
        f=state.f.detach().clone(),
        v=state.v.detach().clone(),
        l=state.l.detach().clone(),
        h=state.h.detach().clone(),
        k=target_k.detach().clone(),
        t=state.t,
        dt=state.dt,
        graph_idx0=state.graph_idx0,
    )
    clean_before = deterministic_predictor_clean_estimate(
        model=model,
        batch=batch,
        state=batch_state,
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(state.t),
        t_final=float(clean_estimator_t_final),
    )
    proj_before = project_clean_to_fixed_wyckoff(
        f0_hat=clean_before.f0_hat,
        atomic_numbers=state.h,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        config=projection_config,
        reference_frac=clean_before.f0_hat,
    )
    anchor = build_assignment_chart_anchor(proj_before)
    initial_normal, _ = normal_residual_energy(
        f_hat=clean_before.f0_hat,
        anchor=anchor,
        damping=float(objective_config.projector_damping),
    )
    v_ref = state.v.detach().clone()
    v_var = v_ref.clone().requires_grad_(True)
    theta_var = None
    parameters: list[torch.Tensor] = [v_var]
    if objective_config.objective_mode == "joint_v_theta" and int(anchor.theta.numel()) > 0:
        theta_var = torch.zeros_like(anchor.theta).requires_grad_(True)
        parameters.append(theta_var)
    optimizer = torch.optim.Adam(parameters, lr=float(objective_config.opt_lr))
    history: list[float] = []
    best_idx = 0
    best_loss = float("inf")
    best_v = center_velocity(v_ref)[0]
    best_theta = torch.zeros_like(anchor.theta)
    best_aux: dict[str, Any] | None = None
    best_valid_idx = -1
    best_valid_loss = float("inf")
    best_valid_v: torch.Tensor | None = None
    best_valid_theta: torch.Tensor | None = None
    for step_idx in range(max(int(objective_config.opt_steps), 1)):
        optimizer.zero_grad()
        v_centered, _ = center_velocity(v_var)
        loss, aux = assignment_aware_velocity_ppr_objective(
            model=model,
            batch=batch,
            f_t=state.f.detach(),
            v_candidate=v_centered,
            l_t=state.l.detach(),
            h=state.h.detach(),
            t=float(state.t),
            node_index=node_index,
            edge_node_index=edge_node_index,
            v_reference=v_ref,
            chart_anchor=anchor,
            target_k=target_k,
            clean_estimator_steps=clean_estimator_steps,
            clean_estimator_t_final=clean_estimator_t_final,
            objective_mode=str(objective_config.objective_mode),
            lambda_v=float(objective_config.lambda_v),
            lambda_theta=float(objective_config.lambda_theta),
            lambda_mf=float(objective_config.lambda_mf),
            lambda_norm=float(objective_config.lambda_norm),
            theta_delta=theta_var,
            damping=float(objective_config.projector_damping),
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            v_centered, _ = center_velocity(v_var)
            v_var.copy_(v_centered)
            if theta_var is not None:
                theta_var.clamp_(
                    -float(objective_config.theta_trust_radius),
                    float(objective_config.theta_trust_radius),
                )
        loss_value = float(loss.detach().item())
        history.append(loss_value)
        theta_current = torch.zeros_like(anchor.theta) if theta_var is None else theta_var.detach().clone()
        if loss_value < best_loss:
            best_loss = loss_value
            best_idx = step_idx
            best_v = v_centered.detach().clone()
            best_theta = theta_current.detach().clone()
            best_aux = aux
        if validity_check_fn is not None:
            try:
                is_valid = bool(validity_check_fn(v_centered.detach().clone(), theta_current.detach().clone(), aux))
            except Exception:
                is_valid = False
            if is_valid and loss_value < best_valid_loss:
                best_valid_loss = loss_value
                best_valid_idx = step_idx
                best_valid_v = v_centered.detach().clone()
                best_valid_theta = theta_current.detach().clone()
    if objective_config.return_best_valid_iterate and best_valid_v is not None:
        best_v = best_valid_v
        best_theta = best_valid_theta if best_valid_theta is not None else torch.zeros_like(anchor.theta)
        best_idx = best_valid_idx
    theta_after = anchor.theta if objective_config.objective_mode != "joint_v_theta" else anchor.theta + best_theta
    clean_after = deterministic_predictor_clean_estimate(
        model=model,
        batch=batch,
        state=replace(batch_state, v=best_v.detach().clone()),
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(state.t),
        t_final=float(clean_estimator_t_final),
    )
    proj_after = project_clean_to_fixed_wyckoff(
        f0_hat=clean_after.f0_hat,
        atomic_numbers=state.h,
        template_state=anchor.template_state,
        target_k=target_k,
        tau0=anchor.tau,
        theta0=theta_after,
        fixed_assignment=anchor.assignment,
        config=projection_config,
        reference_frac=clean_after.f0_hat,
    )
    final_normal, _ = normal_residual_energy(
        f_hat=clean_after.f0_hat,
        anchor=anchor if objective_config.objective_mode != "joint_v_theta" else build_assignment_chart_anchor(proj_after),
        damping=float(objective_config.projector_damping),
    )
    _, mean_before = center_velocity(v_ref)
    _, mean_after = center_velocity(best_v)
    return VelocityPPROptimizationResult(
        objective_mode=str(objective_config.objective_mode),
        v_star=best_v.detach().clone(),
        theta_delta_star=best_theta.detach().clone(),
        clean_before=clean_before,
        clean_after=clean_after,
        projection_before=proj_before,
        projection_after=proj_after,
        energy_before=float(best_aux["constraint_energy"] if best_aux is not None else _norm(wrap_residual(clean_before.f0_hat, proj_before.z_f)) ** 2),
        energy_after=float(best_loss if best_loss < float("inf") else _norm(wrap_residual(clean_after.f0_hat, proj_after.z_f)) ** 2),
        constraint_energy_before=float(_norm(wrap_residual(clean_before.f0_hat, proj_before.z_f)) ** 2),
        constraint_energy_after=float(_norm(wrap_residual(clean_after.f0_hat, proj_after.z_f)) ** 2),
        normal_energy_before=float(initial_normal.detach().item()),
        normal_energy_after=float(final_normal.detach().item()),
        velocity_displacement=float(torch.linalg.norm((best_v - v_ref).reshape(-1)).detach().item()),
        velocity_norm_before=float(torch.linalg.norm(v_ref.reshape(-1)).detach().item()),
        velocity_norm_after=float(torch.linalg.norm(best_v.reshape(-1)).detach().item()),
        mean_free_error_before=float(mean_before),
        mean_free_error_after=float(mean_after),
        objective_history=tuple(history),
        best_iter_index=int(best_idx),
        clean_estimator_mode=clean_after.estimator_mode,
    )


def velocity_only_accept(
    *,
    optimized: VelocityPPROptimizationResult,
    state: KLDMGraphState,
    target_k: torch.Tensor,
) -> VelocityPPRAcceptResult:
    return VelocityPPRAcceptResult(
        mode="velocity_only",
        optimized=optimized,
        state_out=KLDMGraphState(
            f=state.f.detach().clone(),
            v=optimized.v_star.detach().clone(),
            l=state.l.detach().clone(),
            h=state.h.detach().clone(),
            k=target_k.detach().clone(),
            t=state.t,
            dt=state.dt,
            graph_idx0=state.graph_idx0,
        ),
        renoise=None,
        clean_projected_frac=None,
    )


def velocity_ppr_renoise_accept(
    *,
    model,
    batch,
    state: KLDMGraphState,
    node_index: torch.Tensor,
    optimized: VelocityPPROptimizationResult,
    target_k: torch.Tensor,
) -> VelocityPPRAcceptResult:
    renoise = kldm_forward_renoise_fv_only(
        model=model,
        batch=batch,
        f0=optimized.projection_after.z_f,
        l0=state.l.detach().clone(),
        t_graph=torch.as_tensor([float(state.t)], device=state.f.device, dtype=state.f.dtype),
        node_index=node_index,
        v0=None,
        mean_free_velocity=True,
    )
    return VelocityPPRAcceptResult(
        mode="velocity_ppr_renoise",
        optimized=optimized,
        state_out=KLDMGraphState(
            f=renoise.f_t.detach().clone(),
            v=renoise.v_t.detach().clone(),
            l=state.l.detach().clone(),
            h=state.h.detach().clone(),
            k=target_k.detach().clone(),
            t=state.t,
            dt=state.dt,
            graph_idx0=state.graph_idx0,
        ),
        renoise=renoise,
        clean_projected_frac=optimized.projection_after.z_f.detach().clone(),
    )


def tangent_project_clean_velocity(
    *,
    v0_hat: torch.Tensor,
    theta: torch.Tensor,
    template_state: PCSTemplateState,
    tau: torch.Tensor,
    damping: float = 1.0e-6,
) -> torch.Tensor:
    from kldmPlus.algorithm13_fixed_template_velocity_casal import compute_template_jacobian

    J = compute_template_jacobian(theta, template_state, tau=tau)
    tangent = tangent_project_mean_free(v0_hat, J=J, metric=None, damping=damping)
    centered, _ = center_velocity(tangent.velocity)
    return centered


def ppr_clean_project_renoise_step(
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
    clean_estimator_steps: int,
    clean_estimator_t_final: float,
    clean_velocity_mode: str = "zero",
    lattice0_source: str = "state",
    clean_estimate_fn: Any | None = None,
) -> KLDMPPRStepResult:
    if clean_velocity_mode != "zero":
        raise ValueError(
            "detKLDM-Wyckoff-CPR supports only clean_velocity_mode='zero'. "
            "Projection/tangent clean velocities are legacy ablations and exact "
            "KLDM renoise with nonzero clean velocity is unsupported."
        )
    if lattice0_source != "state":
        raise ValueError(
            "detKLDM-Wyckoff-CPR keeps the incoming facit reverse-PC lattice fixed; "
            "lattice0_source must be 'state'."
        )
    if clean_estimate_fn is None:
        clean = deterministic_predictor_clean_estimate(
            model=model,
            batch=batch,
            state=state,
            node_index=node_index,
            edge_node_index=edge_node_index,
            n_steps=clean_estimator_steps,
            t_start=float(state.t),
            t_final=float(clean_estimator_t_final),
        )
    else:
        clean = clean_estimate_fn(
            model=model,
            batch=batch,
            state=state,
            node_index=node_index,
            edge_node_index=edge_node_index,
            n_steps=clean_estimator_steps,
            t_start=float(state.t),
            t_final=float(clean_estimator_t_final),
        )
    proj = project_clean_to_fixed_wyckoff(
        f0_hat=clean.f0_hat,
        atomic_numbers=state.h,
        template_state=template_state,
        target_k=target_k,
        tau0=tau0,
        theta0=theta0,
        fixed_assignment=fixed_assignment,
        config=projection_config,
        reference_frac=clean.f0_hat,
    )
    d_before = _norm(wrap_residual(clean.f0_hat, proj.z_f))
    projection_safe = ppr_projection_is_safe(proj)
    v0_proj = torch.zeros_like(clean.v0_hat)
    l0_renoise = state.l.detach().clone()
    renoise = kldm_forward_renoise_fv_only(
        model=model,
        batch=batch,
        f0=proj.z_f,
        l0=l0_renoise,
        t_graph=torch.as_tensor([float(state.t)], device=state.f.device, dtype=state.f.dtype),
        node_index=node_index,
        v0=v0_proj,
        mean_free_velocity=True,
    )
    clean_after = deterministic_predictor_clean_estimate(
        model=model,
        batch=batch,
        state=KLDMGraphState(
            f=renoise.f_t,
            v=renoise.v_t,
            l=state.l,
            h=state.h,
            k=state.k,
            t=state.t,
            dt=state.dt,
            graph_idx0=state.graph_idx0,
        ),
        node_index=node_index,
        edge_node_index=edge_node_index,
        n_steps=clean_estimator_steps,
        t_start=float(state.t),
        t_final=float(clean_estimator_t_final),
    )
    proj_after = project_clean_to_fixed_wyckoff(
        f0_hat=clean_after.f0_hat,
        atomic_numbers=state.h,
        template_state=template_state,
        target_k=target_k,
        tau0=proj.tau,
        theta0=proj.theta,
        fixed_assignment=proj.assignment,
        config=projection_config,
        reference_frac=clean_after.f0_hat,
    )
    d_after_to_initial_anchor = _norm(wrap_residual(clean_after.f0_hat, proj.z_f))
    d_after_to_manifold = _norm(wrap_residual(clean_after.f0_hat, proj_after.z_f))
    d_after = d_after_to_manifold
    lattice_changed_norm = _norm(_restore_graph_l_shape(renoise.l_t) - _restore_graph_l_shape(state.l))
    return KLDMPPRStepResult(
        clean_estimate=clean,
        clean_after_renoise=clean_after,
        projection=proj,
        projection_after=proj_after,
        renoise=renoise,
        f0_proj=proj.z_f.detach().clone(),
        v0_proj=v0_proj.detach().clone(),
        l0_renoise=l0_renoise.detach().clone(),
        lattice_policy=PPR_LATTICE_POLICY,
        d_before=float(d_before),
        d_after=float(d_after),
        d_after_to_manifold=float(d_after_to_manifold),
        d_after_to_initial_anchor=float(d_after_to_initial_anchor),
        lattice_changed_norm=float(lattice_changed_norm),
        projection_safe=projection_safe,
    )


def iterate_ppr_fixed_t(
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
    clean_estimator_steps: int,
    clean_estimator_t_final: float,
    clean_velocity_mode: str = "zero",
    lattice0_source: str = "state",
    clean_estimate_fn: Any | None = None,
    num_iterations: int = 1,
) -> KLDMPPRIterationResult:
    if int(num_iterations) != 1:
        raise ValueError("detKLDM-Wyckoff-CPR active path uses M=1; repeated fixed-t iterations are disabled.")
    current_state = state
    current_tau = tau0.detach().clone()
    current_theta = None if theta0 is None else theta0.detach().clone()
    current_assignment = None if fixed_assignment is None else fixed_assignment.detach().clone()
    current_template_state = template_state
    outputs: list[KLDMPPRStepResult] = []
    for _ in range(1):
        step = ppr_clean_project_renoise_step(
            model=model,
            batch=batch,
            state=current_state,
            node_index=node_index,
            edge_node_index=edge_node_index,
            template_state=current_template_state,
            target_k=target_k,
            tau0=current_tau,
            theta0=current_theta,
            fixed_assignment=current_assignment,
            projection_config=projection_config,
            clean_estimator_steps=clean_estimator_steps,
            clean_estimator_t_final=clean_estimator_t_final,
            clean_velocity_mode=clean_velocity_mode,
            lattice0_source=lattice0_source,
            clean_estimate_fn=clean_estimate_fn,
        )
        outputs.append(step)
        current_state = KLDMGraphState(
            f=step.renoise.f_t.detach().clone(),
            v=step.renoise.v_t.detach().clone(),
            l=current_state.l.detach().clone(),
            h=current_state.h.detach().clone(),
            k=current_state.k.detach().clone(),
            t=current_state.t,
            dt=current_state.dt,
            graph_idx0=current_state.graph_idx0,
        )
        current_tau = step.projection.tau.detach().clone()
        current_theta = step.projection.theta.detach().clone()
        current_assignment = step.projection.assignment.detach().clone()
        current_template_state = step.projection.raw.state
    return KLDMPPRIterationResult(
        steps=tuple(outputs),
        final_state=current_state,
    )
