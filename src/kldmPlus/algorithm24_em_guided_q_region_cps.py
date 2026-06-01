from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
import torch

from kldmPlus.algorithm19_kldm_ppr_diffcsppp import Algorithm19State
from kldmPlus.algorithm22_faithful_kldm_cps_csp import (
    Algorithm22Candidate,
    Algorithm22Config,
    Algorithm22ProjectedCandidate,
    Algorithm22StateUpdateResult,
    clean_fractional_estimate,
    expand_template_to_model_order,
    fit_q_to_template,
    generate_candidates,
    kldm_cps_update_or_renoise,
    predict_clean_f0,
)
from kldmPlus.symmetry.crystalformer_backend import torus_mse, torus_soft_project, wrap01


ALGORITHM24_MODE = "em_guided_q_region_cps"
ALGORITHM24_SHORT_NAME = "Algorithm24-EM-QRegion-KLDM-CPS"
ALGORITHM24_DESCRIPTION = (
    "Treat a cheap facitKLDM EM lookahead endpoint as a noisy structural observation, "
    "explain it under oracle-G Wyckoff templates together with the current KLDM clean "
    "estimate, build a small q-region between q_now and q_EM, and apply weak CPS using "
    "geometry-first / optional CrystalFormer-local ranking."
)


@dataclass(frozen=True)
class Algorithm24EMConfig:
    lookahead_steps: int = 200
    t_final: float = 1.0e-6


@dataclass(frozen=True)
class Algorithm24RegionConfig:
    lambdas: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5)
    perturb_sigmas: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02)
    perturbations_per_lambda: int = 2
    relative_slack_now: float = 0.10
    absolute_slack_now: float = 1.0e-6
    relative_slack_em: float = 0.10
    absolute_slack_em: float = 1.0e-6
    em_residual_gate: float = math.inf
    cross_slack_abs: float = 1.0e-4
    anchor_eps: float = 1.0e-6
    anchor_tiebreak_delta: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class Algorithm24SelectionConfig:
    lambda_em: float = 1.0
    lambda_shift: float = 0.01
    shift_cap: float = 0.02
    prefer_cf_within_anchor_band: bool = True


@dataclass(frozen=True)
class Algorithm24Config:
    algo22: Algorithm22Config = Algorithm22Config()
    em: Algorithm24EMConfig = Algorithm24EMConfig()
    region: Algorithm24RegionConfig = Algorithm24RegionConfig()
    selection: Algorithm24SelectionConfig = Algorithm24SelectionConfig()


@dataclass(frozen=True)
class Algorithm24EMLookaheadResult:
    state_end: Algorithm19State
    f0_em: torch.Tensor
    l0_em: torch.Tensor
    a0_em: torch.Tensor
    t_start: float
    t_final: float
    steps: int


@dataclass(frozen=True)
class Algorithm24TemplateExplanation:
    candidate: Algorithm22Candidate
    q_now: torch.Tensor
    frac_now: torch.Tensor
    residual_now: float
    q_em: torch.Tensor
    frac_em: torch.Tensor
    residual_em: float
    cross_em_to_now: float
    cross_now_to_em: float
    anchor_score_now: float
    passed_em_gate: bool


@dataclass(frozen=True)
class Algorithm24RegionCandidate:
    explanation: Algorithm24TemplateExplanation
    source: str
    q: torch.Tensor
    frac_coords_model_order: torch.Tensor
    candidate_id: str
    lambda_value: float
    sigma_q: float
    radial_index: int
    accepted_by_region: bool
    now_cost: float
    em_cost: float
    now_cost_cap: float
    em_cost_cap: float
    shift_cost: float
    anchor_score: float
    cf_nll: float


@dataclass(frozen=True)
class Algorithm24SelectionEvaluation:
    region_candidate: Algorithm24RegionCandidate
    update: Algorithm22StateUpdateResult
    f0_after: torch.Tensor
    post_geometry_cost: float
    shift_distance: float
    score: float


@dataclass(frozen=True)
class Algorithm24SelectionResult:
    best: Algorithm24SelectionEvaluation
    baseline_now: Algorithm24SelectionEvaluation
    evaluations: tuple[Algorithm24SelectionEvaluation, ...]


def algorithm24_torus_mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torus_mse(left, right).detach().item())


def algorithm24_geometry_cap(
    residual: float,
    *,
    relative_slack: float,
    absolute_slack: float,
) -> float:
    base = max(0.0, float(residual))
    return float(base * (1.0 + float(relative_slack)) + float(absolute_slack))


def algorithm24_shift_distance(
    *,
    f0_projected: torch.Tensor,
    f0_anchor: torch.Tensor,
) -> float:
    return float(torch.sqrt(torus_mse(f0_projected, f0_anchor).clamp_min(0.0)).detach().item())


def _prepare_em_rollout_state(
    *,
    model,
    batch,
    state: Algorithm19State,
    n_steps: int,
    t_start: float,
    t_final: float,
) -> dict[str, Any]:
    prepared = model._prepare_csp_sampling(
        batch=batch,
        n_steps=int(n_steps),
        t_start=float(t_start),
        t_final=float(t_final),
    )
    prepared["f_t"] = state.f.detach().clone().to(device=prepared["f_t"].device, dtype=prepared["f_t"].dtype)
    prepared["v_t"] = state.v.detach().clone().to(device=prepared["v_t"].device, dtype=prepared["v_t"].dtype)
    prepared["l_t"] = state.l.detach().clone().reshape_as(prepared["l_t"]).to(device=prepared["l_t"].device, dtype=prepared["l_t"].dtype)
    prepared["a_t"] = state.atom_types.detach().clone().to(device=prepared["a_t"].device, dtype=prepared["a_t"].dtype)
    prepared["node_index"] = state.node_index.detach().clone().to(device=prepared["node_index"].device, dtype=prepared["node_index"].dtype)
    prepared["edge_node_index"] = state.edge_node_index.detach().clone().to(device=prepared["edge_node_index"].device, dtype=prepared["edge_node_index"].dtype)
    return prepared


def algorithm24_run_facit_em_lookahead(
    *,
    model,
    batch,
    state: Algorithm19State,
    steps: int,
    t_final: float = 1.0e-6,
) -> Algorithm24EMLookaheadResult:
    t_start = float(torch.as_tensor(state.t_graph).reshape(-1)[0].detach().item())
    prepared = _prepare_em_rollout_state(
        model=model,
        batch=batch,
        state=state,
        n_steps=int(steps),
        t_start=float(t_start),
        t_final=float(t_final),
    )
    prepared = model._run_csp_em_reverse_chain(prepared)
    state_end = Algorithm19State(
        f=prepared["f_t"].detach().clone(),
        v=prepared["v_t"].detach().clone(),
        l=prepared["l_t"].reshape(-1).detach().clone(),
        atom_types=prepared["a_t"].detach().clone(),
        node_index=prepared["node_index"].detach().clone(),
        edge_node_index=prepared["edge_node_index"].detach().clone(),
        t_graph=torch.full_like(state.t_graph, float(t_final)),
        t_nodes=torch.full_like(state.t_nodes, float(t_final)),
    )
    return Algorithm24EMLookaheadResult(
        state_end=state_end,
        f0_em=prepared["f_t"].detach().clone(),
        l0_em=prepared["l_t"].reshape(-1).detach().clone(),
        a0_em=prepared["a_t"].detach().clone(),
        t_start=float(t_start),
        t_final=float(t_final),
        steps=int(steps),
    )


def algorithm24_explain_candidate(
    *,
    candidate: Algorithm22Candidate,
    target_now: torch.Tensor,
    target_em: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    config: Algorithm24Config = Algorithm24Config(),
) -> Algorithm24TemplateExplanation:
    projected_now = fit_q_to_template(
        candidate=candidate,
        target_f0=target_now,
        target_atomic_numbers=target_atomic_numbers,
        q_init=candidate.q_init,
        lambda_init=float(config.algo22.lambda_cf_init),
        q_opt_steps=int(config.algo22.q_opt_steps),
        q_lr=float(config.algo22.q_lr),
        grad_clip=float(config.algo22.grad_clip),
    )
    projected_em = fit_q_to_template(
        candidate=candidate,
        target_f0=target_em,
        target_atomic_numbers=target_atomic_numbers,
        q_init=candidate.q_init,
        lambda_init=float(config.algo22.lambda_cf_init),
        q_opt_steps=int(config.algo22.q_opt_steps),
        q_lr=float(config.algo22.q_lr),
        grad_clip=float(config.algo22.grad_clip),
    )
    cross_em_to_now = algorithm24_torus_mse(projected_em.frac_coords_model_order, target_now)
    cross_now_to_em = algorithm24_torus_mse(projected_now.frac_coords_model_order, target_em)
    passed_em_gate = bool(
        float(projected_em.witness) <= float(config.region.em_residual_gate)
        and float(cross_em_to_now) <= float(projected_now.witness) + float(config.region.cross_slack_abs)
    )
    return Algorithm24TemplateExplanation(
        candidate=candidate,
        q_now=projected_now.q_star.detach().clone(),
        frac_now=projected_now.frac_coords_model_order.detach().clone(),
        residual_now=float(projected_now.witness),
        q_em=projected_em.q_star.detach().clone(),
        frac_em=projected_em.frac_coords_model_order.detach().clone(),
        residual_em=float(projected_em.witness),
        cross_em_to_now=float(cross_em_to_now),
        cross_now_to_em=float(cross_now_to_em),
        anchor_score_now=float(projected_now.witness),
        passed_em_gate=bool(passed_em_gate),
    )


def algorithm24_explain_templates(
    *,
    candidates: tuple[Algorithm22Candidate, ...],
    target_now: torch.Tensor,
    target_em: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    config: Algorithm24Config = Algorithm24Config(),
) -> tuple[Algorithm24TemplateExplanation, ...]:
    out: list[Algorithm24TemplateExplanation] = []
    for cand in candidates:
        out.append(
            algorithm24_explain_candidate(
                candidate=cand,
                target_now=target_now,
                target_em=target_em,
                target_atomic_numbers=target_atomic_numbers,
                config=config,
            )
        )
    return tuple(out)


def algorithm24_generate_template_candidates(
    *,
    f0_hat: torch.Tensor,
    state: Algorithm19State,
    space_group: int,
    lattice_transform,
    formula: str | None = None,
    cf_likelihood=None,
    config: Algorithm24Config = Algorithm24Config(),
    source: str = "pyxtal_only",
    debug_label: str | None = None,
) -> tuple[Algorithm22Candidate, ...]:
    return generate_candidates(
        f0_hat=f0_hat,
        state=state,
        space_group=int(space_group),
        source=source,
        lattice_transform=lattice_transform,
        config=config.algo22,
        formula=formula,
        cf_likelihood=cf_likelihood,
        debug_label=debug_label,
    )


def _cf_eval_region(
    *,
    explanation: Algorithm24TemplateExplanation,
    q: torch.Tensor,
    frac_coords_model_order: torch.Tensor,
    lattice_feature: torch.Tensor,
    cf_energy_fn: Callable[[Algorithm22Candidate, torch.Tensor, torch.Tensor], float] | None,
    cf_formula: str | None,
) -> float:
    if cf_energy_fn is None:
        return float("nan")
    proposal_candidate = Algorithm22Candidate(
        source=explanation.candidate.source,
        template=explanation.candidate.template,
        q_init=q.detach().clone(),
        payload=explanation.candidate.payload,
        lattice_matrix=explanation.candidate.lattice_matrix,
        lattice_feature=lattice_feature.detach().clone(),
        atomic_numbers=explanation.candidate.atomic_numbers,
        frac_coords_model_order=frac_coords_model_order.detach().clone(),
        pcs_state=explanation.candidate.pcs_state,
        cf_nll=float("nan"),
        metadata=dict(explanation.candidate.metadata or {}) | {"cf_formula": cf_formula},
    )
    return float(cf_energy_fn(proposal_candidate, q.detach().clone(), lattice_feature.detach().clone()))


def build_q_tube_candidates(
    *,
    q_now: torch.Tensor,
    q_em: torch.Tensor,
    lambdas: tuple[float, ...],
    sigmas: tuple[float, ...],
    n_per_sigma: int,
    seed: int,
) -> tuple[tuple[str, torch.Tensor, float, float, int], ...]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    q_delta = torch.remainder(q_em - q_now + 0.5, 1.0) - 0.5
    out: list[tuple[str, torch.Tensor, float, float, int]] = []
    for lam in tuple(float(v) for v in lambdas):
        q_axis = wrap01(q_now + float(lam) * q_delta)
        for sigma in tuple(float(v) for v in sigmas):
            reps = 1 if abs(float(sigma)) <= 1.0e-15 else int(max(1, n_per_sigma))
            for rep in range(reps):
                if abs(float(sigma)) <= 1.0e-15:
                    q_prop = q_axis
                else:
                    noise = float(sigma) * torch.randn(
                        q_axis.shape,
                        generator=gen,
                        device=q_axis.device,
                        dtype=q_axis.dtype,
                    )
                    q_prop = wrap01(q_axis + noise)
                out.append((f"tube_lam_{lam:.2f}_sigma_{sigma:.4f}_rep_{rep}", q_prop, float(lam), float(sigma), int(rep)))
    return tuple(out)


def score_anchor_candidate(
    *,
    frac_coords_model_order: torch.Tensor,
    q: torch.Tensor,
    q_now: torch.Tensor,
    target_now: torch.Tensor,
    target_em: torch.Tensor,
    r_now: float,
    r_em: float,
    lambda_em: float,
    lambda_shift: float,
    anchor_eps: float,
) -> tuple[float, float, float, float]:
    now_cost = algorithm24_torus_mse(frac_coords_model_order, target_now)
    em_cost = algorithm24_torus_mse(frac_coords_model_order, target_em)
    shift_cost = algorithm24_torus_mse(q, q_now)
    score = (
        float(now_cost) / max(float(anchor_eps), float(r_now) + float(anchor_eps))
        + float(lambda_em) * float(em_cost) / max(float(anchor_eps), float(r_em) + float(anchor_eps))
        + float(lambda_shift) * float(shift_cost)
    )
    return float(now_cost), float(em_cost), float(shift_cost), float(score)


def filter_by_anchor_gates(
    *,
    now_cost: float,
    em_cost: float,
    r_now: float,
    r_em: float,
    rho_now: float,
    rho_em: float,
    absolute_slack_now: float,
    absolute_slack_em: float,
) -> tuple[bool, float, float]:
    now_cap = algorithm24_geometry_cap(
        r_now,
        relative_slack=float(rho_now),
        absolute_slack=float(absolute_slack_now),
    )
    em_cap = algorithm24_geometry_cap(
        r_em,
        relative_slack=float(rho_em),
        absolute_slack=float(absolute_slack_em),
    )
    accepted = bool(float(now_cost) <= float(now_cap) and float(em_cost) <= float(em_cap))
    return bool(accepted), float(now_cap), float(em_cap)


def select_by_anchor_then_cf(
    *,
    region_candidates: tuple[Algorithm24RegionCandidate, ...],
    anchor_delta: float,
    use_cf_tiebreak: bool,
) -> Algorithm24RegionCandidate | None:
    accepted = [cand for cand in region_candidates if bool(cand.accepted_by_region)]
    if not accepted:
        return None
    accepted = sorted(
        accepted,
        key=lambda cand: (
            float(cand.anchor_score),
            float(cand.now_cost),
            float(cand.em_cost),
            float(cand.shift_cost),
            float(cand.lambda_value),
            float(cand.sigma_q),
        ),
    )
    best_anchor = float(accepted[0].anchor_score)
    near = [cand for cand in accepted if float(cand.anchor_score) <= float(best_anchor) + float(anchor_delta)]
    if bool(use_cf_tiebreak):
        finite_cf = [cand for cand in near if np.isfinite(float(cand.cf_nll))]
        if finite_cf:
            finite_cf = sorted(
                finite_cf,
                key=lambda cand: (
                    float(cand.cf_nll),
                    float(cand.anchor_score),
                    float(cand.now_cost),
                    float(cand.em_cost),
                ),
            )
            return finite_cf[0]
    return accepted[0]


def algorithm24_build_q_region(
    *,
    explanation: Algorithm24TemplateExplanation,
    target_now: torch.Tensor,
    target_em: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    lattice_feature: torch.Tensor,
    cf_energy_fn: Callable[[Algorithm22Candidate, torch.Tensor, torch.Tensor], float] | None = None,
    cf_formula: str | None = None,
    config: Algorithm24Config = Algorithm24Config(),
) -> tuple[Algorithm24RegionCandidate, ...]:
    out: list[Algorithm24RegionCandidate] = []
    tube_candidates = build_q_tube_candidates(
        q_now=explanation.q_now,
        q_em=explanation.q_em,
        lambdas=tuple(config.region.lambdas),
        sigmas=tuple(config.region.perturb_sigmas),
        n_per_sigma=int(max(1, config.region.perturbations_per_lambda)),
        seed=int(config.region.seed),
    )
    for candidate_id, q_prop, lam, sigma_q, radial_index in tube_candidates:
            frac_prop = expand_template_to_model_order(
                template=explanation.candidate.template,
                q=q_prop,
                lattice_matrix=explanation.candidate.lattice_matrix,
                target_atomic_numbers=target_atomic_numbers,
                reference_f0=target_now,
                spacegroup=int(explanation.candidate.payload.spacegroup),
            )
            now_cost, em_cost, shift_cost, anchor_score = score_anchor_candidate(
                frac_coords_model_order=frac_prop,
                q=q_prop,
                q_now=explanation.q_now,
                target_now=target_now,
                target_em=target_em,
                r_now=float(explanation.residual_now),
                r_em=float(explanation.residual_em),
                lambda_em=float(config.selection.lambda_em),
                lambda_shift=float(config.selection.lambda_shift),
                anchor_eps=float(config.region.anchor_eps),
            )
            accepted, now_cap, em_cap = filter_by_anchor_gates(
                now_cost=float(now_cost),
                em_cost=float(em_cost),
                r_now=float(explanation.residual_now),
                r_em=float(explanation.residual_em),
                rho_now=float(config.region.relative_slack_now),
                rho_em=float(config.region.relative_slack_em),
                absolute_slack_now=float(config.region.absolute_slack_now),
                absolute_slack_em=float(config.region.absolute_slack_em),
            )
            cf_nll = _cf_eval_region(
                explanation=explanation,
                q=q_prop,
                frac_coords_model_order=frac_prop,
                lattice_feature=lattice_feature,
                cf_energy_fn=cf_energy_fn,
                cf_formula=cf_formula,
            ) if accepted else float("nan")
            out.append(
                Algorithm24RegionCandidate(
                    explanation=explanation,
                    source="q_tube",
                    q=q_prop.detach().clone(),
                    frac_coords_model_order=frac_prop.detach().clone(),
                    candidate_id=str(candidate_id),
                    lambda_value=float(lam),
                    sigma_q=float(sigma_q),
                    radial_index=int(radial_index),
                    accepted_by_region=bool(accepted),
                    now_cost=float(now_cost),
                    em_cost=float(em_cost),
                    now_cost_cap=float(now_cap),
                    em_cost_cap=float(em_cap),
                    shift_cost=float(shift_cost),
                    anchor_score=float(anchor_score),
                    cf_nll=float(cf_nll),
                )
            )
    return tuple(out)


def algorithm24_choose_pre_projection_candidate(
    *,
    region_candidates: tuple[Algorithm24RegionCandidate, ...],
    config: Algorithm24Config = Algorithm24Config(),
) -> Algorithm24RegionCandidate | None:
    return select_by_anchor_then_cf(
        region_candidates=region_candidates,
        anchor_delta=float(config.region.anchor_tiebreak_delta),
        use_cf_tiebreak=bool(config.selection.prefer_cf_within_anchor_band),
    )


def algorithm24_soft_project_clean_estimate(
    *,
    f0_hat: torch.Tensor,
    frac_target: torch.Tensor,
    alpha: float,
    shift_cap: float | None = None,
) -> torch.Tensor:
    f0_pr = torus_soft_project(f0_hat=f0_hat, f0_hard=frac_target, alpha=float(alpha))
    if shift_cap is None:
        return f0_pr
    shift = algorithm24_shift_distance(f0_projected=f0_pr, f0_anchor=f0_hat)
    if shift <= float(shift_cap):
        return f0_pr
    scale = float(shift_cap) / max(1.0e-8, float(shift))
    return torus_soft_project(f0_hat=f0_hat, f0_hard=frac_target, alpha=float(alpha) * scale)


def algorithm24_state_update(
    *,
    model,
    state: Algorithm19State,
    f0_anchor: torch.Tensor,
    frac_target: torch.Tensor,
    alpha: float,
    beta: float = 1.0,
    config: Algorithm24Config = Algorithm24Config(),
) -> Algorithm22StateUpdateResult:
    f0_pr = algorithm24_soft_project_clean_estimate(
        f0_hat=f0_anchor,
        frac_target=frac_target,
        alpha=float(alpha),
        shift_cap=float(config.selection.shift_cap),
    )
    return kldm_cps_update_or_renoise(
        model=model,
        state=state,
        f0_anchor=f0_anchor,
        f0_projected=f0_pr,
        beta=float(beta),
        mode=config.algo22.state_return_mode,
    )


def algorithm24_select_by_post_update_survival(
    *,
    model,
    state: Algorithm19State,
    f0_anchor: torch.Tensor,
    region_candidates: tuple[Algorithm24RegionCandidate, ...],
    alpha: float,
    beta: float = 1.0,
    config: Algorithm24Config = Algorithm24Config(),
) -> Algorithm24SelectionResult:
    accepted = [cand for cand in region_candidates if bool(cand.accepted_by_region)]
    if not accepted:
        raise ValueError("Algorithm24 survival selection requires at least one accepted q-region candidate.")

    evaluations: list[Algorithm24SelectionEvaluation] = []
    for cand in accepted:
        update = algorithm24_state_update(
            model=model,
            state=state,
            f0_anchor=f0_anchor,
            frac_target=cand.frac_coords_model_order,
            alpha=float(alpha),
            beta=float(beta),
            config=config,
        )
        f0_after = predict_clean_f0(
            state=update.state,
            model=model,
            denoiser_variant=config.algo22.denoiser_variant,
            coordinate_score_mode=config.algo22.coordinate_score_mode,
        )
        post_geometry_cost = algorithm24_torus_mse(cand.frac_coords_model_order, f0_after)
        shift_distance = algorithm24_shift_distance(
            f0_projected=update.f0_projected,
            f0_anchor=f0_anchor,
        )
        score = float(post_geometry_cost) + float(config.selection.lambda_shift) * float(shift_distance * shift_distance)
        evaluations.append(
            Algorithm24SelectionEvaluation(
                region_candidate=cand,
                update=update,
                f0_after=f0_after.detach().clone(),
                post_geometry_cost=float(post_geometry_cost),
                shift_distance=float(shift_distance),
                score=float(score),
            )
        )
    evaluations = sorted(
        evaluations,
        key=lambda item: (
            float(item.score),
            float(item.post_geometry_cost),
            float(item.region_candidate.anchor_score),
            float("inf") if not np.isfinite(float(item.region_candidate.cf_nll)) else float(item.region_candidate.cf_nll),
        ),
    )
    baseline = next(
        (
            item for item in evaluations
            if abs(float(item.region_candidate.lambda_value)) <= 1.0e-12 and abs(float(item.region_candidate.sigma_q)) <= 1.0e-12
        ),
        evaluations[0],
    )
    return Algorithm24SelectionResult(
        best=evaluations[0],
        baseline_now=baseline,
        evaluations=tuple(evaluations),
    )


__all__ = [
    "ALGORITHM24_MODE",
    "ALGORITHM24_SHORT_NAME",
    "ALGORITHM24_DESCRIPTION",
    "Algorithm24Config",
    "Algorithm24EMConfig",
    "Algorithm24RegionConfig",
    "Algorithm24SelectionConfig",
    "Algorithm24EMLookaheadResult",
    "Algorithm24TemplateExplanation",
    "Algorithm24RegionCandidate",
    "Algorithm24SelectionEvaluation",
    "Algorithm24SelectionResult",
    "algorithm24_run_facit_em_lookahead",
    "algorithm24_generate_template_candidates",
    "algorithm24_explain_candidate",
    "algorithm24_explain_templates",
    "algorithm24_build_q_region",
    "algorithm24_choose_pre_projection_candidate",
    "algorithm24_select_by_post_update_survival",
    "algorithm24_state_update",
    "algorithm24_soft_project_clean_estimate",
    "algorithm24_geometry_cap",
    "algorithm24_shift_distance",
    "build_q_tube_candidates",
    "score_anchor_candidate",
    "filter_by_anchor_gates",
    "select_by_anchor_then_cf",
]
