from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
import torch

from kldmPlus.algorithm22_faithful_kldm_cps_csp import (
    Algorithm22Candidate,
    Algorithm22Config,
    Algorithm22ProjectedCandidate,
    Algorithm22StateUpdateResult,
    expand_template_to_model_order,
    fit_q_to_template,
    kldm_cps_update_or_renoise,
    predict_clean_f0,
)
from kldmPlus.symmetry.crystalformer_backend import torus_mse, torus_soft_project, wrap01


ALGORITHM23_MODE = "crystalformer_mcmc_q_refinement"
ALGORITHM23_SHORT_NAME = "Algorithm23-CF-MCMC-KLDM-CPS"
ALGORITHM23_DESCRIPTION = (
    "Local CrystalFormer trust-region MCMC around the KLDM geometry-fitted Wyckoff q, "
    "with CrystalFormer used as a proposal generator and KLDM post-update survival "
    "used as the selector, followed by weak CPS projection and velocity-preserving return."
)


@dataclass(frozen=True)
class Algorithm23TrustRegionConfig:
    relative_slack: float = 0.10
    absolute_slack: float = 1.0e-6
    shift_cap: float = 0.02


@dataclass(frozen=True)
class Algorithm23MCMCConfig:
    steps: int = 32
    proposal_sigmas: tuple[float, ...] = (0.005, 0.01, 0.02)
    acceptance_mode: str = "greedy"
    temperature: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class Algorithm23Config:
    algo22: Algorithm22Config = Algorithm22Config()
    trust_region: Algorithm23TrustRegionConfig = Algorithm23TrustRegionConfig()
    mcmc: Algorithm23MCMCConfig = Algorithm23MCMCConfig()


@dataclass(frozen=True)
class Algorithm23MCMCStep:
    step: int
    sigma_q: float
    inside_trust_region: bool
    accepted: bool
    geometry_cost: float
    cf_nll: float
    geometry_cap: float


@dataclass(frozen=True)
class Algorithm23QProposal:
    source: str
    q: torch.Tensor
    frac_coords_model_order: torch.Tensor
    geometry_cost: float
    cf_nll: float
    accepted: bool
    inside_trust_region: bool
    step: int


@dataclass(frozen=True)
class Algorithm23MCMCResult:
    candidate: Algorithm22Candidate
    q_geom: torch.Tensor
    q_best: torch.Tensor
    frac_geom: torch.Tensor
    frac_best: torch.Tensor
    geometry_cost_before: float
    geometry_cost_after: float
    cf_nll_before: float
    cf_nll_after: float
    geometry_cap: float
    acceptance_rate: float
    num_inside_trust: int
    accepted_steps: tuple[Algorithm23MCMCStep, ...]
    proposals: tuple[Algorithm23QProposal, ...]


@dataclass(frozen=True)
class Algorithm23SurvivalEvaluation:
    proposal: Algorithm23QProposal
    update: Algorithm22StateUpdateResult
    f0_after: torch.Tensor
    post_geometry_cost: float
    shift_distance: float
    score: float


@dataclass(frozen=True)
class Algorithm23SurvivalSelectionResult:
    best: Algorithm23SurvivalEvaluation
    baseline: Algorithm23SurvivalEvaluation
    evaluations: tuple[Algorithm23SurvivalEvaluation, ...]


def algorithm23_geometry_cap(
    geometry_cost: float,
    *,
    relative_slack: float = 0.10,
    absolute_slack: float = 1.0e-6,
) -> float:
    base = float(max(0.0, geometry_cost))
    return float(base * (1.0 + float(relative_slack)) + float(absolute_slack))


def algorithm23_geometry_cost(
    *,
    frac_coords_model_order: torch.Tensor,
    target_f0: torch.Tensor,
) -> float:
    return float(torus_mse(frac_coords_model_order, target_f0).detach().item())


def algorithm23_shift_distance(
    *,
    f0_projected: torch.Tensor,
    f0_anchor: torch.Tensor,
) -> float:
    return float(torch.sqrt(torus_mse(f0_projected, f0_anchor).clamp_min(0.0)).detach().item())


def _accept_proposal(
    *,
    current_cf_nll: float,
    proposal_cf_nll: float,
    mode: str,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    accept_mode = str(mode).strip().lower()
    if not np.isfinite(float(proposal_cf_nll)):
        return False
    if not np.isfinite(float(current_cf_nll)):
        return True
    if accept_mode == "greedy":
        return bool(float(proposal_cf_nll) < float(current_cf_nll))
    if accept_mode == "metropolis":
        delta = float(proposal_cf_nll) - float(current_cf_nll)
        if delta <= 0.0:
            return True
        temp = max(1.0e-8, float(temperature))
        prob = math.exp(-delta / temp)
        return bool(rng.uniform() < prob)
    raise ValueError(f"Unsupported Algorithm23 acceptance mode {mode!r}.")


def algorithm23_refine_projected_candidate(
    *,
    candidate: Algorithm22Candidate,
    target_f0: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    lattice_feature: torch.Tensor,
    cf_energy_fn: Callable[[Algorithm22Candidate, torch.Tensor, torch.Tensor], float] | None = None,
    cf_formula: str | None = None,
    config: Algorithm23Config = Algorithm23Config(),
) -> Algorithm23MCMCResult:
    projected_geom = fit_q_to_template(
        candidate=candidate,
        target_f0=target_f0,
        target_atomic_numbers=target_atomic_numbers,
        q_init=candidate.q_init,
        lambda_init=float(config.algo22.lambda_cf_init),
        q_opt_steps=int(config.algo22.q_opt_steps),
        q_lr=float(config.algo22.q_lr),
        grad_clip=float(config.algo22.grad_clip),
    )
    q_geom = projected_geom.q_star.detach().clone()
    frac_geom = projected_geom.frac_coords_model_order.detach().clone()
    geometry_cost_before = algorithm23_geometry_cost(
        frac_coords_model_order=frac_geom,
        target_f0=target_f0,
    )
    geometry_cap = algorithm23_geometry_cap(
        geometry_cost_before,
        relative_slack=float(config.trust_region.relative_slack),
        absolute_slack=float(config.trust_region.absolute_slack),
    )

    def _cf_eval(q: torch.Tensor, frac_coords_model_order: torch.Tensor) -> float:
        if cf_energy_fn is None:
            return float("nan")
        proposal_candidate = Algorithm22Candidate(
            source=candidate.source,
            template=candidate.template,
            q_init=q.detach().clone(),
            payload=candidate.payload,
            lattice_matrix=candidate.lattice_matrix,
            lattice_feature=lattice_feature.detach().clone(),
            atomic_numbers=candidate.atomic_numbers,
            frac_coords_model_order=frac_coords_model_order.detach().clone(),
            pcs_state=candidate.pcs_state,
            cf_nll=float("nan"),
            metadata=dict(candidate.metadata or {}) | {"cf_formula": cf_formula},
        )
        return float(cf_energy_fn(proposal_candidate, q.detach().clone(), lattice_feature.detach().clone()))

    cf_nll_before = _cf_eval(q_geom, frac_geom)
    q_current = q_geom.detach().clone()
    frac_current = frac_geom.detach().clone()
    cf_current = float(cf_nll_before)
    q_best = q_geom.detach().clone()
    frac_best = frac_geom.detach().clone()
    cf_best = float(cf_nll_before)
    cost_best = float(geometry_cost_before)
    accepted_rows: list[Algorithm23MCMCStep] = []
    proposals: list[Algorithm23QProposal] = [
        Algorithm23QProposal(
            source="q_geom",
            q=q_geom.detach().clone(),
            frac_coords_model_order=frac_geom.detach().clone(),
            geometry_cost=float(geometry_cost_before),
            cf_nll=float(cf_nll_before),
            accepted=True,
            inside_trust_region=True,
            step=-1,
        )
    ]
    inside_count = 0
    accept_count = 0
    sigmas = tuple(float(v) for v in config.mcmc.proposal_sigmas)
    rng_np = np.random.default_rng(int(config.mcmc.seed))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(config.mcmc.seed))

    for step_idx in range(int(max(0, config.mcmc.steps))):
        sigma = sigmas[int(rng_np.integers(0, max(1, len(sigmas))))] if sigmas else 0.01
        noise = float(sigma) * torch.randn(
            q_current.shape,
            generator=gen,
            device=q_current.device,
            dtype=q_current.dtype,
        )
        q_prop = wrap01(q_current + noise)
        frac_prop = expand_template_to_model_order(
            template=candidate.template,
            q=q_prop,
            lattice_matrix=candidate.lattice_matrix,
            target_atomic_numbers=target_atomic_numbers,
            reference_f0=target_f0,
            spacegroup=int(candidate.payload.spacegroup),
        )
        cost_prop = algorithm23_geometry_cost(
            frac_coords_model_order=frac_prop,
            target_f0=target_f0,
        )
        inside = bool(cost_prop <= geometry_cap)
        if inside:
            inside_count += 1
        if not inside:
            proposals.append(
                Algorithm23QProposal(
                    source="cf_mcmc_rejected",
                    q=q_prop.detach().clone(),
                    frac_coords_model_order=frac_prop.detach().clone(),
                    geometry_cost=float(cost_prop),
                    cf_nll=float("nan"),
                    accepted=False,
                    inside_trust_region=False,
                    step=int(step_idx),
                )
            )
            accepted_rows.append(
                Algorithm23MCMCStep(
                    step=int(step_idx),
                    sigma_q=float(sigma),
                    inside_trust_region=False,
                    accepted=False,
                    geometry_cost=float(cost_prop),
                    cf_nll=float("nan"),
                    geometry_cap=float(geometry_cap),
                )
            )
            continue
        cf_prop = _cf_eval(q_prop, frac_prop)
        accepted = _accept_proposal(
            current_cf_nll=cf_current,
            proposal_cf_nll=cf_prop,
            mode=config.mcmc.acceptance_mode,
            temperature=float(config.mcmc.temperature),
            rng=rng_np,
        )
        if accepted:
            accept_count += 1
            q_current = q_prop.detach().clone()
            frac_current = frac_prop.detach().clone()
            cf_current = float(cf_prop)
            if np.isfinite(float(cf_prop)) and (
                not np.isfinite(float(cf_best)) or float(cf_prop) < float(cf_best)
            ):
                q_best = q_prop.detach().clone()
                frac_best = frac_prop.detach().clone()
                cf_best = float(cf_prop)
                cost_best = float(cost_prop)
        proposals.append(
            Algorithm23QProposal(
                source="cf_mcmc_accepted" if accepted else "cf_mcmc_inside_rejected",
                q=q_prop.detach().clone(),
                frac_coords_model_order=frac_prop.detach().clone(),
                geometry_cost=float(cost_prop),
                cf_nll=float(cf_prop),
                accepted=bool(accepted),
                inside_trust_region=True,
                step=int(step_idx),
            )
        )
        accepted_rows.append(
            Algorithm23MCMCStep(
                step=int(step_idx),
                sigma_q=float(sigma),
                inside_trust_region=True,
                accepted=bool(accepted),
                geometry_cost=float(cost_prop),
                cf_nll=float(cf_prop),
                geometry_cap=float(geometry_cap),
            )
        )

    return Algorithm23MCMCResult(
        candidate=candidate,
        q_geom=q_geom.detach().clone(),
        q_best=q_best.detach().clone(),
        frac_geom=frac_geom.detach().clone(),
        frac_best=frac_best.detach().clone(),
        geometry_cost_before=float(geometry_cost_before),
        geometry_cost_after=float(cost_best),
        cf_nll_before=float(cf_nll_before),
        cf_nll_after=float(cf_best),
        geometry_cap=float(geometry_cap),
        acceptance_rate=float(accept_count / max(1, int(config.mcmc.steps))),
        num_inside_trust=int(inside_count),
        accepted_steps=tuple(accepted_rows),
        proposals=tuple(proposals),
    )


def algorithm23_soft_project_clean_estimate(
    *,
    f0_hat: torch.Tensor,
    frac_target: torch.Tensor,
    alpha: float,
    shift_cap: float | None = None,
) -> torch.Tensor:
    f0_pr = torus_soft_project(f0_hat=f0_hat, f0_hard=frac_target, alpha=float(alpha))
    if shift_cap is None:
        return f0_pr
    shift = algorithm23_shift_distance(f0_projected=f0_pr, f0_anchor=f0_hat)
    if shift <= float(shift_cap):
        return f0_pr
    scale = float(shift_cap) / max(1.0e-8, float(shift))
    return torus_soft_project(f0_hat=f0_hat, f0_hard=frac_target, alpha=float(alpha) * scale)


def algorithm23_state_update(
    *,
    model,
    state,
    f0_anchor: torch.Tensor,
    frac_target: torch.Tensor,
    alpha: float,
    beta: float = 1.0,
    config: Algorithm23Config = Algorithm23Config(),
) -> Algorithm22StateUpdateResult:
    f0_pr = algorithm23_soft_project_clean_estimate(
        f0_hat=f0_anchor,
        frac_target=frac_target,
        alpha=float(alpha),
        shift_cap=float(config.trust_region.shift_cap),
    )
    return kldm_cps_update_or_renoise(
        model=model,
        state=state,
        f0_anchor=f0_anchor,
        f0_projected=f0_pr,
        beta=float(beta),
        mode=config.algo22.state_return_mode,
    )


def algorithm23_generate_cf_mcmc_q_proposals(
    *,
    candidate: Algorithm22Candidate,
    target_f0: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    lattice_feature: torch.Tensor,
    cf_energy_fn: Callable[[Algorithm22Candidate, torch.Tensor, torch.Tensor], float] | None = None,
    cf_formula: str | None = None,
    config: Algorithm23Config = Algorithm23Config(),
) -> Algorithm23MCMCResult:
    return algorithm23_refine_projected_candidate(
        candidate=candidate,
        target_f0=target_f0,
        target_atomic_numbers=target_atomic_numbers,
        lattice_feature=lattice_feature,
        cf_energy_fn=cf_energy_fn,
        cf_formula=cf_formula,
        config=config,
    )


def algorithm23_select_by_kldm_survival(
    *,
    model,
    state,
    f0_anchor: torch.Tensor,
    proposals: tuple[Algorithm23QProposal, ...],
    alpha: float,
    beta: float = 1.0,
    config: Algorithm23Config = Algorithm23Config(),
    shift_weight: float = 0.0,
) -> Algorithm23SurvivalSelectionResult:
    if not proposals:
        raise ValueError("Algorithm23 survival selection requires at least one proposal.")

    evaluations: list[Algorithm23SurvivalEvaluation] = []
    for proposal in proposals:
        update = algorithm23_state_update(
            model=model,
            state=state,
            f0_anchor=f0_anchor,
            frac_target=proposal.frac_coords_model_order,
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
        post_geometry_cost = algorithm23_geometry_cost(
            frac_coords_model_order=proposal.frac_coords_model_order,
            target_f0=f0_after,
        )
        shift_distance = algorithm23_shift_distance(
            f0_projected=update.f0_projected,
            f0_anchor=f0_anchor,
        )
        score = float(post_geometry_cost) + float(shift_weight) * float(shift_distance * shift_distance)
        evaluations.append(
            Algorithm23SurvivalEvaluation(
                proposal=proposal,
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
            float("inf") if not np.isfinite(float(item.proposal.cf_nll)) else float(item.proposal.cf_nll),
        ),
    )
    baseline = next((item for item in evaluations if item.proposal.source == "q_geom"), evaluations[0])
    best = evaluations[0]
    return Algorithm23SurvivalSelectionResult(
        best=best,
        baseline=baseline,
        evaluations=tuple(evaluations),
    )


def algorithm23_state_update_from_selected_q(
    *,
    selection: Algorithm23SurvivalSelectionResult,
) -> Algorithm22StateUpdateResult:
    return selection.best.update
