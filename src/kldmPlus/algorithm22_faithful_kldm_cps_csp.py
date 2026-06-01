from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import math

import numpy as np
import torch

from kldmPlus.symmetry.crystalformer_backend import (
    Algorithm19State,
    CrystalFormerLikelihood,
    _get_wyckoff_dof_chart as _crystalformer_get_wyckoff_dof_chart,
    build_payload_from_template_q,
    expand_q,
    fit_q_to_clean_prediction,
    model_to_payload,
    payload_to_model,
    predict_clean_f0,
    renoise_from_f0,
    sample_q_from_crystalformer,
    species_match_reorder,
    torus_mse,
    torus_soft_project,
    torus_rmse,
    witness_torus_sin_loss,
    wrap01,
    wrapdiff,
)
from kldmPlus.sample_evaluation.sample_evaluation import (
    build_structure_from_sample,
    decode_lattice_matrix,
    detect_space_group_number,
    validity_structure_reason,
)
from kldmPlus.symmetry.diffcsppp_backend import (
    DiffCSPPPSymmetryPayload,
    attach_payload_reference_chart,
    build_diffcsppp_symmetry_payload,
)
from kldmPlus.symmetry.pcs_projection import (
    PCSTemplateState,
    initialize_constrained_template_states,
    materialize_pcs_state,
    vanilla_structure_to_model_tensors,
)
from kldmPlus.symmetry.wyckoff_templates import WyckoffTemplate


ALGORITHM22_MODE = "faithful_kldm_cps_csp"
ALGORITHM22_SHORT_NAME = "Algorithm22-KLDM-CPS-CSP"
ALGORITHM22_DESCRIPTION = (
    "Finite-candidate KLDM-CPS over exact-composition Wyckoff charts under an oracle "
    "space group, using PyXtal template generation and optional CrystalFormer scoring."
)


@dataclass(frozen=True)
class Algorithm22ScheduleConfig:
    n_pc_steps: int = 800
    projection_interval: int = 50
    p_start: float = 0.625
    gamma_min: float = 1.0
    gamma_max: float = 100.0
    gamma_mid: float = 25.0
    alpha_max: float = 0.25
    beta_default: float = 1.0


@dataclass(frozen=True)
class Algorithm22Config:
    schedule: Algorithm22ScheduleConfig = Algorithm22ScheduleConfig()
    q_opt_steps: int = 50
    q_lr: float = 1.0e-2
    grad_clip: float = 10.0
    lambda_cf_init: float = 0.0
    lambda_lattice: float = 0.0
    lambda_collision: float = 0.0
    collision_min_distance: float = 0.75
    top_branches: int = 1
    eps_rank: float = 1.0e-4
    post_acceptance: bool = True
    max_templates: int = 256
    template_eval_limit: int = 32
    pyxtal_top_k: int = 64
    pyxtal_q_samples_per_template: int = 1
    pyxtal_q_sampling_strategies: tuple[str, ...] = ("pcs_anchor",)
    cf_sample_k: int = 8
    cf_top_p: float = 1.0
    cf_temperature: float = 1.0
    cf_sampler_seed: int = 0
    crystalformer_template_mode: str = "score_only"
    denoiser_variant: str = "minus"
    coordinate_score_mode: str = "direct"
    lattice_projection: bool = False
    state_return_mode: str = "preserve_velocity_shift"


@dataclass(frozen=True)
class Algorithm22SchedulePoint:
    step: int
    progress: float
    project: bool
    gamma: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class Algorithm22Candidate:
    source: str
    template: WyckoffTemplate | None
    q_init: torch.Tensor
    payload: DiffCSPPPSymmetryPayload
    lattice_matrix: torch.Tensor
    lattice_feature: torch.Tensor
    atomic_numbers: torch.Tensor
    frac_coords_model_order: torch.Tensor
    pcs_state: PCSTemplateState | None = None
    cf_nll: float = float("nan")
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Algorithm22ProjectedCandidate:
    candidate: Algorithm22Candidate
    q_star: torch.Tensor
    frac_coords_model_order: torch.Tensor
    witness: float
    witness_rmse: float
    lattice_score: float
    collision_penalty: float
    score_geom: float
    cf_nll: float
    assignment_source_to_model: torch.Tensor
    q_distance_to_init: float


@dataclass(frozen=True)
class Algorithm22BranchResult:
    projected: Algorithm22ProjectedCandidate
    state_candidate: Algorithm19State
    f0_hat_before: torch.Tensor
    f0_hard: torch.Tensor
    f0_projected: torch.Tensor
    f0_hat_after: torch.Tensor
    accepted: bool
    witness_before: float
    witness_after: float
    validity_ok: bool
    validity_reason: str
    min_pair_distance: float | None
    volume: float | None
    max_lattice_length: float | None


@dataclass(frozen=True)
class Algorithm22StateUpdateResult:
    state: Algorithm19State
    mode: str
    beta: float
    delta_clean: torch.Tensor
    f0_anchor: torch.Tensor
    f0_projected: torch.Tensor
    velocity_norm_before: float
    velocity_norm_after: float
    sigma_v_rms: float
    sigma_r_rms: float
    epsilon_r_before: torch.Tensor | None = None
    epsilon_r_after: torch.Tensor | None = None


@dataclass(frozen=True)
class Algorithm22RunResult:
    final_state: Algorithm19State
    accepted_fraction: float
    projection_count: int
    accepted_count: int
    branch_traces: tuple[dict[str, Any], ...]


def algorithm22_piecewise_alpha(progress: float) -> float:
    p = float(progress)
    if p < 0.625:
        return 0.0
    if p < 0.75:
        return 0.10
    if p < 0.875:
        return 0.20
    return 0.25


def algorithm22_piecewise_beta(progress: float) -> float:
    return 0.0 if float(progress) < 0.625 else 1.0


def algorithm22_cps_gamma(progress: float, *, schedule: Algorithm22ScheduleConfig = Algorithm22ScheduleConfig()) -> float:
    p = float(progress)
    if p < float(schedule.p_start):
        return 0.0
    denom = max(1.0e-8, 1.0 - float(schedule.p_start))
    u = min(max((p - float(schedule.p_start)) / denom, 0.0), 1.0)
    return float(schedule.gamma_min) * ((float(schedule.gamma_max) / float(schedule.gamma_min)) ** u)


def algorithm22_cps_alpha(progress: float, *, schedule: Algorithm22ScheduleConfig = Algorithm22ScheduleConfig()) -> float:
    gamma = algorithm22_cps_gamma(progress, schedule=schedule)
    if gamma <= 0.0:
        return 0.0
    return float(schedule.alpha_max) * gamma / (gamma + float(schedule.gamma_mid))


def algorithm22_projection_schedule(
    *,
    n_pc_steps: int = 800,
    projection_interval: int = 50,
    p_start: float = 0.5,
    piecewise: bool = True,
    schedule: Algorithm22ScheduleConfig | None = None,
) -> tuple[Algorithm22SchedulePoint, ...]:
    cfg = schedule or Algorithm22ScheduleConfig(
        n_pc_steps=int(n_pc_steps),
        projection_interval=int(projection_interval),
        p_start=float(p_start),
    )
    out: list[Algorithm22SchedulePoint] = []
    total = max(1, int(cfg.n_pc_steps) - 1)
    for step in range(int(cfg.n_pc_steps)):
        progress = float(step / total)
        project = bool(progress >= float(cfg.p_start) and step % int(cfg.projection_interval) == 0)
        gamma = algorithm22_cps_gamma(progress, schedule=cfg) if project else 0.0
        alpha = algorithm22_piecewise_alpha(progress) if piecewise else algorithm22_cps_alpha(progress, schedule=cfg)
        beta = algorithm22_piecewise_beta(progress) if project else 0.0
        if not project:
            alpha = 0.0
            beta = 0.0
        out.append(
            Algorithm22SchedulePoint(
                step=int(step),
                progress=float(progress),
                project=project,
                gamma=float(gamma),
                alpha=float(alpha),
                beta=float(beta),
            )
        )
    return tuple(out)


def decode_state_cell_matrix(
    *,
    state: Algorithm19State,
    lattice_transform=None,
) -> torch.Tensor:
    return decode_lattice_matrix(
        l=state.l,
        n_atoms=int(state.f.shape[0]),
        lattice_transform=lattice_transform,
    ).to(device=state.f.device, dtype=state.f.dtype)


def build_oracle_diffcsppp_payload_from_structure(
    *,
    standardized_structure,
    requested_spacegroup: int,
    tol: float = 1.0e-2,
) -> DiffCSPPPSymmetryPayload:
    payload = build_diffcsppp_symmetry_payload(standardized_structure, tol=tol)
    if int(payload.spacegroup) != int(requested_spacegroup):
        raise ValueError(
            "Extracted DiffCSP++ payload SG does not match requested oracle SG. "
            f"extracted={int(payload.spacegroup)} oracle={int(requested_spacegroup)}."
        )
    payload = attach_payload_reference_chart(payload, np.asarray(payload.expanded_frac_coords, dtype=float))
    debug_info = dict(payload.debug_info or {})
    debug_info["model_reference_frac_coords"] = np.asarray(standardized_structure.frac_coords, dtype=float).tolist()
    return replace(payload, debug_info=debug_info)


def map_model_to_payload_reference_chart(
    z_model: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    return model_to_payload(f_model=z_model, payload=payload)


def map_payload_reference_chart_to_model_frame(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    return payload_to_model(z_payload=z_payload, payload=payload)


def kldm_clean_fractional_denoiser_Df(
    *,
    model,
    f: torch.Tensor,
    v: torch.Tensor,
    l: torch.Tensor,
    atom_types: torch.Tensor,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    edge_index: torch.Tensor,
    variant: str = "minus",
    coordinate_score_mode: str = "direct",
) -> torch.Tensor:
    state = Algorithm19State(
        f=f,
        v=v,
        l=l,
        atom_types=atom_types,
        node_index=node_index,
        edge_node_index=edge_index,
        t_graph=t_graph,
        t_nodes=t_nodes,
    )
    return predict_clean_f0(
        state=state,
        model=model,
        denoiser_variant=variant,
        coordinate_score_mode=coordinate_score_mode,
    )


def _get_wyckoff_dof_chart(payload: DiffCSPPPSymmetryPayload):
    return _crystalformer_get_wyckoff_dof_chart(payload)


def state_to_structure(
    *,
    state: Algorithm19State,
    lattice_transform=None,
):
    return build_structure_from_sample(
        f=state.f,
        l=state.l,
        a=state.atom_types,
        lattice_transform=lattice_transform,
    )


def clean_fractional_estimate(
    *,
    state: Algorithm19State,
    model,
    config: Algorithm22Config = Algorithm22Config(),
) -> torch.Tensor:
    with torch.no_grad():
        return predict_clean_f0(
            state=state,
            model=model,
            denoiser_variant=config.denoiser_variant,
            coordinate_score_mode=config.coordinate_score_mode,
        )


def kldm_clean_fractional_denoiser(
    *,
    model,
    state: Algorithm19State,
    config: Algorithm22Config = Algorithm22Config(),
) -> torch.Tensor:
    return clean_fractional_estimate(state=state, model=model, config=config)


def clean_lattice_estimate(
    *,
    state: Algorithm19State,
) -> torch.Tensor:
    return state.l.detach().clone()


def kldm_clean_lattice_denoiser(
    *,
    model,
    state: Algorithm19State,
    config: Algorithm22Config = Algorithm22Config(),
) -> torch.Tensor:
    del model, config
    return clean_lattice_estimate(state=state)


def _pairwise_cartesian_collision_penalty(
    *,
    frac_coords: torch.Tensor,
    cell_matrix: torch.Tensor,
    min_distance: float,
) -> float:
    if frac_coords.ndim != 2 or frac_coords.shape[0] <= 1:
        return 0.0
    delta = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
    delta = delta - torch.round(delta)
    cart = torch.einsum("...i,ij->...j", delta, cell_matrix)
    distances = torch.linalg.norm(cart, dim=-1)
    mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    gaps = (float(min_distance) - distances[mask]).clamp_min(0.0)
    return float((gaps.square().sum()).detach().item())


def _lattice_feature_distance(candidate_l: torch.Tensor | None, target_l: torch.Tensor) -> float:
    if candidate_l is None:
        return 0.0
    left = candidate_l.reshape(-1).to(device=target_l.device, dtype=target_l.dtype)
    right = target_l.reshape(-1)
    return float(torch.mean((left - right).square()).detach().item())


def _build_template_payload(
    *,
    template: WyckoffTemplate,
    q: torch.Tensor,
    lattice_matrix: torch.Tensor,
    spacegroup: int | None = None,
) -> DiffCSPPPSymmetryPayload:
    return build_payload_from_template_q(
        template=template,
        q=q.detach().cpu().numpy(),
        lattice_matrix=lattice_matrix.detach().cpu().numpy(),
        spacegroup=spacegroup,
    )


def _special_position_values(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor([0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75], device=device, dtype=dtype)


def sample_template_q_proposals(
    *,
    template: WyckoffTemplate,
    num_samples: int,
    strategy: str = "uniform",
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    q_center: torch.Tensor | None = None,
    perturb_scale: float = 5.0e-2,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cpu") if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype
    dim = int(getattr(template, "total_free_dims", 0))
    count = max(1, int(num_samples))
    if dim <= 0:
        return tuple(torch.zeros((0,), device=device, dtype=dtype) for _ in range(count))

    mode = str(strategy).strip().lower()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    if mode in {"pcs_anchor", "anchor"}:
        if q_center is None:
            center = torch.zeros((dim,), device=device, dtype=dtype)
        else:
            center = torch.remainder(q_center.detach().clone().to(device=device, dtype=dtype).reshape(-1), 1.0)
        return tuple(center.detach().clone() for _ in range(count))
    if mode in {"uniform", "random"}:
        return tuple(torch.rand((dim,), generator=gen, dtype=dtype).to(device=device) for _ in range(count))
    if mode == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(seed))
        return tuple(torch.remainder(engine.draw(1).reshape(-1).to(device=device, dtype=dtype), 1.0) for _ in range(count))
    if mode in {"special", "special_position", "special-position-biased"}:
        base = _special_position_values(device=torch.device("cpu"), dtype=dtype)
        draws: list[torch.Tensor] = []
        for _ in range(count):
            idx = torch.randint(0, int(base.numel()), (dim,), generator=gen)
            q = base[idx]
            noise = perturb_scale * torch.randn((dim,), generator=gen, dtype=dtype)
            draws.append(torch.remainder((q + noise).to(device=device), 1.0))
        return tuple(draws)
    if mode in {"local", "perturb", "small_local_perturbations"}:
        center = torch.zeros((dim,), dtype=dtype) if q_center is None else q_center.detach().clone().reshape(-1).to(dtype=dtype)
        draws: list[torch.Tensor] = []
        for _ in range(count):
            noise = perturb_scale * torch.randn((dim,), generator=gen, dtype=dtype)
            draws.append(torch.remainder((center + noise).to(device=device), 1.0))
        return tuple(draws)
    raise ValueError(f"Unsupported q proposal strategy {strategy!r}.")


def _align_candidate_to_model_order(
    *,
    source_frac: torch.Tensor,
    source_atomic_numbers: torch.Tensor,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    aligned = species_match_reorder(
        source_frac=source_frac,
        source_atomic_numbers=source_atomic_numbers,
        target_frac=target_frac,
        target_atomic_numbers=target_atomic_numbers,
    )
    frac_model_order = torch.as_tensor(
        aligned["aligned_source_in_target_order"],
        device=target_frac.device,
        dtype=target_frac.dtype,
    )
    assignment = torch.as_tensor(aligned["assignment"], device=target_frac.device, dtype=torch.long)
    return frac_model_order, assignment, float(aligned["rmse"])


def expand_template_to_model_order(
    *,
    template: WyckoffTemplate,
    q: torch.Tensor,
    lattice_matrix: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    reference_f0: torch.Tensor | None = None,
    spacegroup: int | None = None,
) -> torch.Tensor:
    payload = _build_template_payload(
        template=template,
        q=q,
        lattice_matrix=lattice_matrix,
        spacegroup=spacegroup,
    )
    device = q.device if reference_f0 is None else reference_f0.device
    dtype = q.dtype if reference_f0 is None else reference_f0.dtype
    z_payload = expand_q(payload=payload, q=q.to(device=device, dtype=dtype))
    if reference_f0 is None:
        reference_f0 = z_payload.detach().clone()
    frac_model_order, _assignment, _rmse = _align_candidate_to_model_order(
        source_frac=z_payload,
        source_atomic_numbers=torch.as_tensor(payload.expanded_atomic_numbers, device=reference_f0.device, dtype=torch.long),
        target_frac=reference_f0,
        target_atomic_numbers=target_atomic_numbers,
    )
    return frac_model_order


def materialize_candidate_from_template(
    *,
    template: WyckoffTemplate,
    q: torch.Tensor,
    state: Algorithm19State,
    lattice_matrix: torch.Tensor,
    lattice_feature: torch.Tensor,
    source: str,
    metadata: dict[str, Any] | None = None,
    cf_nll: float = float("nan"),
    spacegroup: int | None = None,
) -> Algorithm22Candidate:
    payload = _build_template_payload(
        template=template,
        q=q,
        lattice_matrix=lattice_matrix,
        spacegroup=spacegroup,
    )
    frac_model_order = expand_template_to_model_order(
        template=template,
        q=q,
        lattice_matrix=lattice_matrix,
        target_atomic_numbers=state.atom_types,
        reference_f0=state.f,
        spacegroup=spacegroup,
    )
    return Algorithm22Candidate(
        source=str(source),
        template=template,
        q_init=q.detach().clone().to(device=state.f.device, dtype=state.f.dtype),
        payload=payload,
        lattice_matrix=lattice_matrix.detach().clone().to(device=state.f.device, dtype=state.f.dtype),
        lattice_feature=lattice_feature.detach().clone().reshape(-1).to(device=state.f.device, dtype=state.f.dtype),
        atomic_numbers=state.atom_types.detach().clone().to(dtype=torch.long),
        frac_coords_model_order=frac_model_order.detach().clone(),
        pcs_state=None,
        cf_nll=float(cf_nll),
        metadata=dict(metadata or {}),
    )


def oracle_template_projection(
    *,
    state: Algorithm19State,
    model,
    payload: DiffCSPPPSymmetryPayload,
    alpha: float,
    beta: float = 1.0,
    config: Algorithm22Config = Algorithm22Config(),
) -> Algorithm22BranchResult:
    f0_hat = clean_fractional_estimate(state=state, model=model, config=config)
    z_hat = model_to_payload(f_model=f0_hat, payload=payload)
    fit = fit_q_to_clean_prediction(
        z_hat=z_hat,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        q_init=None,
        q_init_mode="random",
        steps=int(config.q_opt_steps),
        lr=float(config.q_lr),
        grad_clip=float(config.grad_clip),
    )
    f0_hard = payload_to_model(z_payload=fit.z_proj_payload, payload=payload)
    f0_pr = torus_soft_project(f0_hat=f0_hat, f0_hard=f0_hard, alpha=float(alpha))
    update = kldm_cps_update_or_renoise(
        model=model,
        state=state,
        f0_anchor=f0_hat,
        f0_projected=f0_pr,
        beta=float(beta),
        mode=config.state_return_mode,
    )
    state_candidate = update.state
    f0_hat_after = clean_fractional_estimate(state=state_candidate, model=model, config=config)
    z_hat_after = model_to_payload(f_model=f0_hat_after, payload=payload)
    fit_after = fit_q_to_clean_prediction(
        z_hat=z_hat_after,
        payload=payload,
        t_nodes=state_candidate.t_nodes,
        lattice_feature=state_candidate.l,
        q_init=fit.q_star.detach().clone(),
        q_init_mode="oracle_structure",
        steps=int(config.q_opt_steps),
        lr=float(config.q_lr),
        grad_clip=float(config.grad_clip),
    )
    before = float(fit.witness_sin)
    after = float(fit_after.witness_sin)
    accepted = bool(after < before) if bool(config.post_acceptance) else True
    valid_ok, valid_reason, min_pair, volume, max_lat = (True, "skipped", None, None, None)
    try:
        structure = build_structure_from_sample(
            f=state_candidate.f,
            l=state_candidate.l,
            a=state_candidate.atom_types,
        )
        valid_ok, valid_reason, min_pair, volume, max_lat = validity_structure_reason(structure)
    except Exception as exc:
        valid_ok = False
        valid_reason = f"structure_failed:{type(exc).__name__}"
    projected = Algorithm22ProjectedCandidate(
        candidate=Algorithm22Candidate(
            source="oracle_template",
            template=payload.debug_info.get("template") if payload.debug_info else None,
            q_init=fit.q_star.detach().clone(),
            payload=payload,
            lattice_matrix=torch.as_tensor(payload.lattice_matrix, device=state.f.device, dtype=state.f.dtype),
            lattice_feature=state.l.detach().clone(),
            atomic_numbers=state.atom_types.detach().clone(),
            frac_coords_model_order=f0_hard.detach().clone(),
            pcs_state=None,
            cf_nll=float("nan"),
            metadata={"mode": "oracle_template"},
        ),
        q_star=fit.q_star.detach().clone(),
        frac_coords_model_order=f0_hard.detach().clone(),
        witness=float(fit.witness_sin),
        witness_rmse=float(fit.witness_rmse_payload),
        lattice_score=0.0,
        collision_penalty=0.0,
        score_geom=float(fit.witness_sin),
        cf_nll=float("nan"),
        assignment_source_to_model=torch.arange(f0_hard.shape[0], device=f0_hard.device, dtype=torch.long),
        q_distance_to_init=0.0,
    )
    return Algorithm22BranchResult(
        projected=projected,
        state_candidate=state_candidate if accepted and valid_ok else state,
        f0_hat_before=f0_hat.detach().clone(),
        f0_hard=f0_hard.detach().clone(),
        f0_projected=f0_pr.detach().clone(),
        f0_hat_after=f0_hat_after.detach().clone(),
        accepted=bool(accepted and valid_ok),
        witness_before=float(before),
        witness_after=float(after),
        validity_ok=bool(valid_ok),
        validity_reason=str(valid_reason),
        min_pair_distance=min_pair,
        volume=volume,
        max_lattice_length=max_lat,
    )


def generate_pyxtal_candidates(
    *,
    f0_hat: torch.Tensor,
    state: Algorithm19State,
    space_group: int,
    lattice_transform,
    top_k: int,
    max_templates: int = 256,
    template_eval_limit: int = 32,
    debug_label: str | None = None,
    formula: str | None = None,
    cf_likelihood: CrystalFormerLikelihood | None = None,
    q_samples_per_template: int = 1,
    q_sampling_strategies: tuple[str, ...] = ("pcs_anchor",),
) -> tuple[Algorithm22Candidate, ...]:
    cell_matrix = decode_state_cell_matrix(state=state, lattice_transform=lattice_transform).to(device=f0_hat.device, dtype=f0_hat.dtype)
    states = initialize_constrained_template_states(
        reference_frac_coords=f0_hat.detach().clone(),
        atomic_numbers=state.atom_types.detach().clone().to(dtype=torch.long),
        cell_matrix=cell_matrix.detach().clone(),
        space_group_number=int(space_group),
        max_templates=int(max_templates),
        template_eval_limit=int(template_eval_limit),
        top_k=int(top_k),
        debug_label=debug_label,
    )
    out: list[Algorithm22Candidate] = []
    for idx, pcs_state in enumerate(states):
        try:
            projection = materialize_pcs_state(
                state=pcs_state,
                vanilla_reference_structure=pcs_state.bridge.vanilla_structure,
            )
            frac_coords, lattice_feature, atomic_numbers = vanilla_structure_to_model_tensors(
                structure=projection.projected_structure_vanilla,
                lattice_transform=lattice_transform,
                device=f0_hat.device,
                dtype=f0_hat.dtype,
            )
            frac_model_order, _assignment, _rmse = _align_candidate_to_model_order(
                source_frac=frac_coords,
                source_atomic_numbers=atomic_numbers,
                target_frac=f0_hat,
                target_atomic_numbers=state.atom_types,
            )
            lattice_matrix = torch.as_tensor(
                np.asarray(projection.projected_structure_vanilla.lattice.matrix, dtype=float),
                device=f0_hat.device,
                dtype=f0_hat.dtype,
            )
            payload = _build_template_payload(
                template=pcs_state.template,
                q=pcs_state.free_vars.detach().clone(),
                lattice_matrix=lattice_matrix,
                spacegroup=int(space_group),
            )
            cf_nll = float("nan")
            if cf_likelihood is not None:
                try:
                    cf_nll = float(
                        cf_likelihood.nll_q(
                            payload=payload,
                            q=np.asarray(pcs_state.free_vars.detach().cpu(), dtype=float),
                            lattice_feature=lattice_feature.detach().cpu(),
                            formula=formula,
                        )
                    )
                except Exception:
                    cf_nll = float("nan")
            base_metadata = {
                "template_rank": int(getattr(pcs_state, "template_rank", idx)),
                "candidate_count": int(getattr(pcs_state, "candidate_count", len(states))),
                "standardized_space_group": int(projection.standardized_space_group) if projection.standardized_space_group is not None else None,
                "primitive_space_group": int(projection.primitive_space_group) if projection.primitive_space_group is not None else None,
            }
            anchor_q = pcs_state.free_vars.detach().clone().to(device=f0_hat.device, dtype=f0_hat.dtype)
            for strategy in tuple(q_sampling_strategies):
                proposals = sample_template_q_proposals(
                    template=pcs_state.template,
                    num_samples=int(q_samples_per_template),
                    strategy=str(strategy),
                    seed=int(10007 * idx + 97),
                    device=f0_hat.device,
                    dtype=f0_hat.dtype,
                    q_center=anchor_q,
                )
                for sample_idx, q_prop in enumerate(proposals):
                    try:
                        cand = materialize_candidate_from_template(
                            template=pcs_state.template,
                            q=q_prop,
                            state=state,
                            lattice_matrix=lattice_matrix,
                            lattice_feature=lattice_feature.reshape(-1),
                            source="pyxtal",
                            metadata={**base_metadata, "q_strategy": str(strategy), "q_strategy_index": int(sample_idx)},
                            cf_nll=cf_nll if sample_idx == 0 and str(strategy).strip().lower() in {"pcs_anchor", "anchor"} else float("nan"),
                            spacegroup=int(space_group),
                        )
                        out.append(replace(cand, pcs_state=pcs_state))
                    except Exception:
                        continue
        except Exception as exc:
            if idx == 0 and not out:
                raise RuntimeError(f"PyXtal candidate materialization failed: {type(exc).__name__}: {exc}") from exc
    return tuple(out)


def augment_candidates_with_crystalformer_q(
    *,
    candidates: tuple[Algorithm22Candidate, ...],
    state: Algorithm19State,
    cf_likelihood: CrystalFormerLikelihood | None,
    formula: str | None,
    K_per_template: int,
    top_p: float = 1.0,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[Algorithm22Candidate, ...]:
    if cf_likelihood is None or not candidates or int(K_per_template) <= 0:
        return candidates
    out: list[Algorithm22Candidate] = []
    for cand_idx, cand in enumerate(candidates):
        out.append(cand)
        try:
            q_samples, cf_nll_values = sample_q_from_crystalformer(
                payload=cand.payload,
                lattice_feature=cand.lattice_feature.reshape(-1),
                formula=formula,
                cf_likelihood=cf_likelihood,
                K=int(K_per_template),
                top_p=float(top_p),
                temperature=float(temperature),
                seed=int(seed + 1000 * cand_idx),
            )
        except Exception:
            continue
        for sample_idx, q_sample in enumerate(q_samples):
            try:
                z_payload = expand_q(payload=cand.payload, q=q_sample.to(device=state.f.device, dtype=state.f.dtype))
                frac_model_order, _assignment, _rmse = _align_candidate_to_model_order(
                    source_frac=z_payload,
                    source_atomic_numbers=torch.as_tensor(cand.payload.expanded_atomic_numbers, device=state.f.device, dtype=torch.long),
                    target_frac=state.f,
                    target_atomic_numbers=state.atom_types,
                )
                out.append(
                    replace(
                        cand,
                        source="pyxtal_cf_q_augmented",
                        q_init=q_sample.detach().clone().to(device=state.f.device, dtype=state.f.dtype),
                        frac_coords_model_order=frac_model_order.detach().clone(),
                        cf_nll=float(cf_nll_values[sample_idx]) if sample_idx < len(cf_nll_values) else float("nan"),
                        metadata=dict(cand.metadata or {}, cf_sample_index=int(sample_idx)),
                    )
                )
            except Exception:
                continue
    return tuple(out)


def fit_q_to_template(
    *,
    candidate: Algorithm22Candidate,
    target_f0: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    q_init: torch.Tensor | None = None,
    lambda_init: float = 0.0,
    q_opt_steps: int = 50,
    q_lr: float = 1.0e-2,
    grad_clip: float = 10.0,
) -> Algorithm22ProjectedCandidate:
    target_f0 = target_f0.detach().clone()
    target_atomic_numbers = target_atomic_numbers.detach().clone()
    payload = candidate.payload
    q_seed = candidate.q_init.detach().clone() if q_init is None else q_init.detach().clone()
    z_seed = expand_q(payload=payload, q=q_seed.to(device=target_f0.device, dtype=target_f0.dtype))
    _aligned_seed, assignment_seed, _rmse_seed = _align_candidate_to_model_order(
        source_frac=z_seed,
        source_atomic_numbers=torch.as_tensor(payload.expanded_atomic_numbers, device=target_f0.device, dtype=torch.long),
        target_frac=target_f0,
        target_atomic_numbers=target_atomic_numbers,
    )
    target_in_source_order = target_f0[assignment_seed]

    q_var = q_seed.to(device=target_f0.device, dtype=target_f0.dtype).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([q_var], lr=float(q_lr))
    for _step_idx in range(int(max(0, q_opt_steps))):
        optimizer.zero_grad(set_to_none=True)
        z_now = expand_q(payload=payload, q=q_var)
        loss = witness_torus_sin_loss(z_now, target_in_source_order)
        if float(lambda_init) > 0.0:
            loss = loss + float(lambda_init) * torus_mse(q_var, q_seed.to(device=q_var.device, dtype=q_var.dtype))
        if torch.isnan(loss) or torch.isinf(loss):
            break
        loss.backward()
        if q_var.grad is not None and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_([q_var], max_norm=float(grad_clip))
        optimizer.step()
        with torch.no_grad():
            q_var.data = torch.remainder(q_var.data, 1.0)

    q_star = torch.remainder(q_var.detach(), 1.0)
    z_star = expand_q(payload=payload, q=q_star)
    frac_model_order, assignment_star, rmse_star = _align_candidate_to_model_order(
        source_frac=z_star,
        source_atomic_numbers=torch.as_tensor(payload.expanded_atomic_numbers, device=target_f0.device, dtype=torch.long),
        target_frac=target_f0,
        target_atomic_numbers=target_atomic_numbers,
    )
    witness = float(witness_torus_sin_loss(frac_model_order, target_f0).detach().item())
    lattice_score = _lattice_feature_distance(candidate.lattice_feature, candidate.lattice_feature)
    collision = _pairwise_cartesian_collision_penalty(
        frac_coords=frac_model_order,
        cell_matrix=candidate.lattice_matrix.to(device=target_f0.device, dtype=target_f0.dtype),
        min_distance=0.75,
    )
    return Algorithm22ProjectedCandidate(
        candidate=candidate,
        q_star=q_star.detach().clone(),
        frac_coords_model_order=frac_model_order.detach().clone(),
        witness=witness,
        witness_rmse=float(rmse_star),
        lattice_score=float(lattice_score),
        collision_penalty=float(collision),
        score_geom=float(witness),
        cf_nll=float(candidate.cf_nll),
        assignment_source_to_model=assignment_star.detach().clone(),
        q_distance_to_init=float(torch.sqrt(torus_mse(q_star, q_seed.to(device=q_star.device, dtype=q_star.dtype)).clamp_min(0.0)).detach().item()),
    )


def torus_distance_squared(
    *,
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    return float(torus_mse(left, right).detach().item())


def project_candidates_to_clean_estimate(
    *,
    candidates: tuple[Algorithm22Candidate, ...],
    target_f0: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
    target_l0: torch.Tensor | None = None,
    config: Algorithm22Config = Algorithm22Config(),
) -> tuple[Algorithm22ProjectedCandidate, ...]:
    target_f0 = target_f0.detach().clone()
    target_atomic_numbers = target_atomic_numbers.detach().clone()
    target_l0 = None if target_l0 is None else target_l0.detach().clone()
    out: list[Algorithm22ProjectedCandidate] = []
    for cand in candidates:
        projected = fit_q_to_template(
            candidate=cand,
            target_f0=target_f0,
            target_atomic_numbers=target_atomic_numbers,
            q_init=cand.q_init,
            lambda_init=float(config.lambda_cf_init),
            q_opt_steps=int(config.q_opt_steps),
            q_lr=float(config.q_lr),
            grad_clip=float(config.grad_clip),
        )
        lattice_score = _lattice_feature_distance(cand.lattice_feature, target_l0) if target_l0 is not None else 0.0
        score_geom = (
            float(projected.witness)
            + float(config.lambda_lattice) * float(lattice_score)
            + float(config.lambda_collision) * float(projected.collision_penalty)
        )
        out.append(
            replace(
                projected,
                lattice_score=float(lattice_score),
                collision_penalty=float(projected.collision_penalty),
                score_geom=float(score_geom),
            )
        )
    return tuple(out)


def geometry_first_select(
    *,
    scored: tuple[Algorithm22ProjectedCandidate, ...],
    top_branches: int = 3,
    eps_rank: float = 1.0e-4,
    use_cf_as_tiebreak: bool = True,
) -> tuple[Algorithm22ProjectedCandidate, ...]:
    if not scored:
        return tuple()

    def _cf_sort_value(value: float) -> float:
        return float("inf") if not np.isfinite(float(value)) else float(value)

    rows = sorted(scored, key=lambda item: (float(item.score_geom), _cf_sort_value(float(item.cf_nll))))
    best_geom = float(rows[0].score_geom)
    pool = [item for item in rows if float(item.score_geom) <= best_geom + float(eps_rank)]
    if use_cf_as_tiebreak:
        pool = sorted(pool, key=lambda item: (_cf_sort_value(float(item.cf_nll)), float(item.score_geom)))
    else:
        pool = sorted(pool, key=lambda item: float(item.score_geom))
    return tuple(pool[: max(1, int(top_branches))])


def rank_projected_candidates_with_rule(
    *,
    scored: tuple[Algorithm22ProjectedCandidate, ...],
    rule: str,
    top_branches: int = 3,
    eps_rank: float = 1.0e-4,
) -> tuple[Algorithm22ProjectedCandidate, ...]:
    if not scored:
        return tuple()

    def _cf_sort_value(value: float) -> float:
        return float("inf") if not np.isfinite(float(value)) else float(value)

    mode = str(rule).strip().lower()
    if mode in {"geometry", "geometry_only"}:
        rows = sorted(scored, key=lambda item: (float(item.witness), float(item.collision_penalty)))
        return tuple(rows[: max(1, int(top_branches))])
    if mode in {"cf", "cf_only", "crystalformer_only"}:
        rows = sorted(scored, key=lambda item: (_cf_sort_value(float(item.cf_nll)), float(item.witness)))
        return tuple(rows[: max(1, int(top_branches))])
    if mode in {"geometry_first", "geometry_first_cf_tiebreak", "geometry+cf"}:
        return geometry_first_select(scored=scored, top_branches=top_branches, eps_rank=eps_rank, use_cf_as_tiebreak=True)
    if mode in {"geometry_lattice", "geometry+lattice"}:
        rows = sorted(scored, key=lambda item: (float(item.witness + item.lattice_score), _cf_sort_value(float(item.cf_nll))))
        return tuple(rows[: max(1, int(top_branches))])
    if mode in {"geometry_collision", "geometry+collision"}:
        rows = sorted(scored, key=lambda item: (float(item.witness + item.collision_penalty), _cf_sort_value(float(item.cf_nll))))
        return tuple(rows[: max(1, int(top_branches))])
    raise ValueError(f"Unsupported ranking rule {rule!r}.")


def current_group_or_candidate_residual(
    *,
    target_f0: torch.Tensor,
    candidates: tuple[Algorithm22ProjectedCandidate, ...],
) -> float:
    if not candidates:
        return float("inf")
    return float(min(float(item.witness) for item in candidates))


def rank_candidates_against_clean_estimate(
    *,
    f0_hat: torch.Tensor,
    state: Algorithm19State,
    candidates: tuple[Algorithm22Candidate, ...],
    config: Algorithm22Config = Algorithm22Config(),
) -> tuple[Algorithm22ProjectedCandidate, ...]:
    target_l0 = clean_lattice_estimate(state=state)
    scored = project_candidates_to_clean_estimate(
        candidates=candidates,
        target_f0=f0_hat,
        target_atomic_numbers=state.atom_types,
        target_l0=target_l0,
        config=config,
    )
    return geometry_first_select(
        scored=scored,
        top_branches=int(config.top_branches),
        eps_rank=float(config.eps_rank),
        use_cf_as_tiebreak=True,
    )


def candidate_post_residual(
    *,
    state: Algorithm19State,
    model,
    candidates: tuple[Algorithm22Candidate, ...],
    config: Algorithm22Config = Algorithm22Config(),
) -> float:
    f0_hat = clean_fractional_estimate(state=state, model=model, config=config)
    ranked = rank_candidates_against_clean_estimate(
        f0_hat=f0_hat,
        state=state,
        candidates=candidates,
        config=config,
    )
    return current_group_or_candidate_residual(target_f0=f0_hat, candidates=ranked)


def tdm_velocity_sigma_at_state(
    *,
    model,
    state: Algorithm19State,
) -> torch.Tensor:
    tau = model.tdm.T * state.t_nodes
    return model.tdm.match_dims(model.tdm.vel_scale * model.tdm.gaussian_velocity_sigma(tau), state.v)


def tdm_position_sigma_at_state(
    *,
    model,
    state: Algorithm19State,
) -> torch.Tensor:
    tau = model.tdm.T * state.t_nodes
    return model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau), state.f)


def tdm_position_mu_at_state(
    *,
    model,
    state: Algorithm19State,
) -> torch.Tensor:
    tau = model.tdm.T * state.t_nodes
    return model.tdm.wrapped_gaussian_mu_r_t(tau, state.v)


def tdm_residual_epsilon_r(
    *,
    model,
    state: Algorithm19State,
    f0_anchor: torch.Tensor,
) -> torch.Tensor:
    mu_r = tdm_position_mu_at_state(model=model, state=state)
    sigma_r = tdm_position_sigma_at_state(model=model, state=state).clamp_min(1.0e-8)
    centered_anchor = wrap01(f0_anchor + mu_r)
    residual = wrapdiff(state.f, centered_anchor)
    return residual / sigma_r


def kldm_cps_update_or_renoise(
    *,
    model,
    state: Algorithm19State,
    f0_anchor: torch.Tensor,
    f0_projected: torch.Tensor,
    beta: float = 1.0,
    mode: str = "preserve_velocity_shift",
) -> Algorithm22StateUpdateResult:
    beta_value = float(beta)
    delta_clean = wrapdiff(f0_projected, f0_anchor)
    mode_key = str(mode).strip().lower()
    sigma_v = tdm_velocity_sigma_at_state(model=model, state=state)
    sigma_r = tdm_position_sigma_at_state(model=model, state=state)
    eps_before = tdm_residual_epsilon_r(model=model, state=state, f0_anchor=f0_anchor)
    velocity_norm_before = float(torch.linalg.norm(state.v).detach().item())

    if mode_key in {"none", "no_correction", "identity"}:
        next_state = replace(state)
        eps_after = eps_before.detach().clone()
    elif mode_key in {"preserve_velocity_shift", "preserve_vt", "shift"}:
        f_new = wrap01(state.f + beta_value * delta_clean)
        next_state = replace(state, f=f_new.detach().clone(), v=state.v.detach().clone())
        eps_after = tdm_residual_epsilon_r(
            model=model,
            state=next_state,
            f0_anchor=wrap01(f0_anchor + beta_value * delta_clean),
        )
    elif mode_key in {"preserve_residual_exact", "residual_exact", "exact"}:
        f0_beta = wrap01(f0_anchor + beta_value * delta_clean)
        mu_r = tdm_position_mu_at_state(model=model, state=state)
        f_new = wrap01(f0_beta + mu_r + sigma_r * eps_before)
        next_state = replace(state, f=f_new.detach().clone(), v=state.v.detach().clone())
        eps_after = tdm_residual_epsilon_r(model=model, state=next_state, f0_anchor=f0_beta)
    elif mode_key in {"resample_tdm_full", "renoise", "full_renoise"}:
        renoised = renoise_from_f0(f0_star=wrap01(f0_anchor + beta_value * delta_clean), state=state, model=model)
        next_state = replace(state, f=renoised.f.detach().clone(), v=renoised.v.detach().clone())
        eps_after = None
    elif mode_key in {"zero_velocity", "v_zero", "set_v_zero"}:
        f_new = wrap01(state.f + beta_value * delta_clean)
        next_state = replace(state, f=f_new.detach().clone(), v=torch.zeros_like(state.v))
        eps_after = tdm_residual_epsilon_r(
            model=model,
            state=next_state,
            f0_anchor=wrap01(f0_anchor + beta_value * delta_clean),
        )
    else:
        raise ValueError(f"Unsupported KLDM-CPS return mode {mode!r}.")

    velocity_norm_after = float(torch.linalg.norm(next_state.v).detach().item())
    return Algorithm22StateUpdateResult(
        state=next_state,
        mode=str(mode),
        beta=float(beta_value),
        delta_clean=delta_clean.detach().clone(),
        f0_anchor=f0_anchor.detach().clone(),
        f0_projected=f0_projected.detach().clone(),
        velocity_norm_before=velocity_norm_before,
        velocity_norm_after=velocity_norm_after,
        sigma_v_rms=float(torch.sqrt(sigma_v.square().mean()).detach().item()),
        sigma_r_rms=float(torch.sqrt(sigma_r.square().mean()).detach().item()),
        epsilon_r_before=eps_before.detach().clone(),
        epsilon_r_after=None if eps_after is None else eps_after.detach().clone(),
    )


def safety_ok(
    *,
    state: Algorithm19State,
    lattice_transform=None,
) -> tuple[bool, str, float | None, float | None, float | None]:
    try:
        structure = state_to_structure(state=state, lattice_transform=lattice_transform)
        return validity_structure_reason(structure)
    except Exception as exc:
        return False, f"structure_failed:{type(exc).__name__}", None, None, None


def branch_survival_step(
    *,
    state: Algorithm19State,
    model,
    projected: Algorithm22ProjectedCandidate,
    alpha: float,
    beta: float,
    candidate_pool: tuple[Algorithm22Candidate, ...],
    config: Algorithm22Config = Algorithm22Config(),
) -> Algorithm22BranchResult:
    f0_hat = clean_fractional_estimate(state=state, model=model, config=config)
    f0_hard = projected.frac_coords_model_order.to(device=f0_hat.device, dtype=f0_hat.dtype)
    f0_pr = torus_soft_project(f0_hat=f0_hat, f0_hard=f0_hard, alpha=float(alpha))
    update = kldm_cps_update_or_renoise(
        model=model,
        state=state,
        f0_anchor=f0_hat,
        f0_projected=f0_pr,
        beta=float(beta),
        mode=config.state_return_mode,
    )
    state_candidate = update.state
    f0_hat_after = clean_fractional_estimate(state=state_candidate, model=model, config=config)
    scored_before = project_candidates_to_clean_estimate(
        candidates=candidate_pool,
        target_f0=f0_hat,
        target_atomic_numbers=state.atom_types,
        target_l0=clean_lattice_estimate(state=state),
        config=config,
    )
    before = current_group_or_candidate_residual(target_f0=f0_hat, candidates=scored_before)
    after = candidate_post_residual(
        state=state_candidate,
        model=model,
        candidates=candidate_pool,
        config=config,
    )
    accepted = bool(after < before) if bool(config.post_acceptance) else True
    valid_ok, valid_reason, min_pair, volume, max_lat = safety_ok(state=state_candidate)
    accepted = bool(accepted and valid_ok)
    return Algorithm22BranchResult(
        projected=projected,
        state_candidate=state_candidate if accepted else state,
        f0_hat_before=f0_hat.detach().clone(),
        f0_hard=f0_hard.detach().clone(),
        f0_projected=f0_pr.detach().clone(),
        f0_hat_after=f0_hat_after.detach().clone(),
        accepted=bool(accepted),
        witness_before=float(before),
        witness_after=float(after),
        validity_ok=bool(valid_ok),
        validity_reason=str(valid_reason),
        min_pair_distance=min_pair,
        volume=volume,
        max_lattice_length=max_lat,
    )


def generate_candidates(
    *,
    f0_hat: torch.Tensor,
    state: Algorithm19State,
    space_group: int,
    source: str,
    lattice_transform,
    config: Algorithm22Config = Algorithm22Config(),
    formula: str | None = None,
    cf_likelihood: CrystalFormerLikelihood | None = None,
    debug_label: str | None = None,
) -> tuple[Algorithm22Candidate, ...]:
    pyxtal_candidates = generate_pyxtal_candidates(
        f0_hat=f0_hat,
        state=state,
        space_group=int(space_group),
        lattice_transform=lattice_transform,
        top_k=int(config.pyxtal_top_k),
        max_templates=int(config.max_templates),
        template_eval_limit=int(config.template_eval_limit),
        debug_label=debug_label,
        formula=formula,
        cf_likelihood=cf_likelihood if source in {"hybrid", "crystalformer_only"} else None,
        q_samples_per_template=int(config.pyxtal_q_samples_per_template),
        q_sampling_strategies=tuple(config.pyxtal_q_sampling_strategies),
    )
    source_key = str(source).strip().lower()
    if source_key == "hybrid":
        source_key = "pyxtal_cf_score"
    elif source_key == "crystalformer_only":
        source_key = "pyxtal_cf_q_augmented"
    if source_key == "pyxtal_only":
        return pyxtal_candidates
    if source_key == "pyxtal_cf_score":
        if str(config.crystalformer_template_mode).strip().lower() in {"score_only", "score", "nll_only"}:
            return tuple(replace(cand, source="pyxtal_cf_score") for cand in pyxtal_candidates)
        return augment_candidates_with_crystalformer_q(
            candidates=pyxtal_candidates,
            state=state,
            cf_likelihood=cf_likelihood,
            formula=formula,
            K_per_template=int(config.cf_sample_k),
            top_p=float(config.cf_top_p),
            temperature=float(config.cf_temperature),
            seed=int(config.cf_sampler_seed),
        )
    if source_key == "pyxtal_cf_q_augmented":
        return augment_candidates_with_crystalformer_q(
            candidates=pyxtal_candidates,
            state=state,
            cf_likelihood=cf_likelihood,
            formula=formula,
            K_per_template=int(max(1, config.cf_sample_k)),
            top_p=float(config.cf_top_p),
            temperature=float(config.cf_temperature),
            seed=int(config.cf_sampler_seed),
        )
    raise ValueError(f"Unsupported candidate source {source!r}.")


def algorithm22b_ranked_kldm_cps_step(
    *,
    state: Algorithm19State,
    model,
    space_group: int,
    lattice_transform,
    candidate_source: str = "pyxtal_cf_score",
    alpha: float = 0.25,
    beta: float = 1.0,
    config: Algorithm22Config = Algorithm22Config(),
    formula: str | None = None,
    cf_likelihood: CrystalFormerLikelihood | None = None,
    debug_label: str | None = None,
) -> tuple[Algorithm22BranchResult | None, tuple[Algorithm22Candidate, ...], tuple[Algorithm22ProjectedCandidate, ...]]:
    f0_hat = clean_fractional_estimate(state=state, model=model, config=config)
    candidates = generate_candidates(
        f0_hat=f0_hat,
        state=state,
        space_group=int(space_group),
        source=candidate_source,
        lattice_transform=lattice_transform,
        config=config,
        formula=formula,
        cf_likelihood=cf_likelihood,
        debug_label=debug_label,
    )
    ranked = rank_candidates_against_clean_estimate(
        f0_hat=f0_hat,
        state=state,
        candidates=candidates,
        config=config,
    )
    if not ranked:
        return None, candidates, ranked
    best: Algorithm22BranchResult | None = None
    for projected in ranked:
        branch = branch_survival_step(
            state=state,
            model=model,
            projected=projected,
            alpha=float(alpha),
            beta=float(beta),
            candidate_pool=candidates,
            config=config,
        )
        if best is None or float(branch.witness_after) < float(best.witness_after):
            best = branch
    return best, candidates, ranked


def algorithm22A_oracle_template_kldm_cps(
    *,
    model,
    initial_state: Algorithm19State,
    oracle_payload: DiffCSPPPSymmetryPayload,
    pc_step_fn,
    decode_final_fn=None,
    config: Algorithm22Config = Algorithm22Config(),
) -> Algorithm22RunResult | Any:
    state = initial_state
    traces: list[dict[str, Any]] = []
    projection_count = 0
    accepted_count = 0
    schedule = algorithm22_projection_schedule(
        n_pc_steps=int(config.schedule.n_pc_steps),
        projection_interval=int(config.schedule.projection_interval),
        p_start=float(config.schedule.p_start),
        piecewise=True,
        schedule=config.schedule,
    )
    for point in schedule:
        state = pc_step_fn(model=model, state=state, step=int(point.step))
        if not bool(point.project):
            continue
        projection_count += 1
        branch = oracle_template_projection(
            state=state,
            model=model,
            payload=oracle_payload,
            alpha=float(point.alpha),
            beta=float(point.beta),
            config=config,
        )
        accepted = bool(branch.accepted)
        if accepted:
            accepted_count += 1
            state = branch.state_candidate
        traces.append(
            {
                "step": int(point.step),
                "progress": float(point.progress),
                "alpha": float(point.alpha),
                "beta": float(point.beta),
                "accepted": bool(accepted),
                "witness_before": float(branch.witness_before),
                "witness_after": float(branch.witness_after),
            }
        )
    result = Algorithm22RunResult(
        final_state=state,
        accepted_fraction=float(accepted_count / max(1, projection_count)),
        projection_count=int(projection_count),
        accepted_count=int(accepted_count),
        branch_traces=tuple(traces),
    )
    return result if decode_final_fn is None else decode_final_fn(model=model, state=state, run_result=result)


def algorithm22B_ranked_kldm_cps(
    *,
    model,
    initial_state: Algorithm19State,
    oracle_space_group: int,
    lattice_transform,
    pc_step_fn,
    decode_final_fn=None,
    candidate_source: str = "pyxtal_cf_score",
    config: Algorithm22Config = Algorithm22Config(),
    formula: str | None = None,
    crystalformer: CrystalFormerLikelihood | None = None,
) -> Algorithm22RunResult | Any:
    state = initial_state
    traces: list[dict[str, Any]] = []
    projection_count = 0
    accepted_count = 0
    schedule = algorithm22_projection_schedule(
        n_pc_steps=int(config.schedule.n_pc_steps),
        projection_interval=int(config.schedule.projection_interval),
        p_start=float(config.schedule.p_start),
        piecewise=True,
        schedule=config.schedule,
    )
    for point in schedule:
        state = pc_step_fn(model=model, state=state, step=int(point.step))
        if not bool(point.project):
            continue
        projection_count += 1
        best, candidates, ranked = algorithm22b_ranked_kldm_cps_step(
            state=state,
            model=model,
            space_group=int(oracle_space_group),
            lattice_transform=lattice_transform,
            candidate_source=candidate_source,
            alpha=float(point.alpha),
            beta=float(point.beta),
            config=config,
            formula=formula,
            cf_likelihood=crystalformer,
            debug_label=f"step={int(point.step)}",
        )
        accepted = bool(best is not None and best.accepted)
        if accepted and best is not None:
            accepted_count += 1
            state = best.state_candidate
        traces.append(
            {
                "step": int(point.step),
                "progress": float(point.progress),
                "alpha": float(point.alpha),
                "beta": float(point.beta),
                "accepted": bool(accepted),
                "num_candidates": int(len(candidates)),
                "num_ranked": int(len(ranked)),
                "best_post": float(best.witness_after) if best is not None else float("inf"),
            }
        )
    result = Algorithm22RunResult(
        final_state=state,
        accepted_fraction=float(accepted_count / max(1, projection_count)),
        projection_count=int(projection_count),
        accepted_count=int(accepted_count),
        branch_traces=tuple(traces),
    )
    return result if decode_final_fn is None else decode_final_fn(model=model, state=state, run_result=result)


def oracle_template_witness(
    *,
    state: Algorithm19State,
    model,
    payload: DiffCSPPPSymmetryPayload,
    config: Algorithm22Config = Algorithm22Config(),
) -> float:
    f0_hat = clean_fractional_estimate(state=state, model=model, config=config)
    z_hat = model_to_payload(f_model=f0_hat, payload=payload)
    fit = fit_q_to_clean_prediction(
        z_hat=z_hat,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        q_init=None,
        q_init_mode="random",
        steps=int(config.q_opt_steps),
        lr=float(config.q_lr),
        grad_clip=float(config.grad_clip),
    )
    return float(fit.witness_sin)


def group_witness_from_candidates(
    *,
    state: Algorithm19State,
    model,
    candidates: tuple[Algorithm22Candidate, ...],
    config: Algorithm22Config = Algorithm22Config(),
) -> float:
    return candidate_post_residual(
        state=state,
        model=model,
        candidates=candidates,
        config=config,
    )


__all__ = [
    "ALGORITHM22_DESCRIPTION",
    "ALGORITHM22_MODE",
    "ALGORITHM22_SHORT_NAME",
    "Algorithm22BranchResult",
    "Algorithm22Candidate",
    "Algorithm22Config",
    "Algorithm22ProjectedCandidate",
    "Algorithm22RunResult",
    "Algorithm22ScheduleConfig",
    "Algorithm22SchedulePoint",
    "Algorithm22StateUpdateResult",
    "algorithm22A_oracle_template_kldm_cps",
    "algorithm22B_ranked_kldm_cps",
    "algorithm22_cps_alpha",
    "algorithm22_cps_gamma",
    "algorithm22_piecewise_alpha",
    "algorithm22_piecewise_beta",
    "algorithm22_projection_schedule",
    "algorithm22b_ranked_kldm_cps_step",
    "_get_wyckoff_dof_chart",
    "branch_survival_step",
    "build_oracle_diffcsppp_payload_from_structure",
    "clean_fractional_estimate",
    "clean_lattice_estimate",
    "current_group_or_candidate_residual",
    "decode_state_cell_matrix",
    "expand_template_to_model_order",
    "fit_q_to_template",
    "generate_candidates",
    "generate_pyxtal_candidates",
    "group_witness_from_candidates",
    "kldm_clean_fractional_denoiser",
    "kldm_clean_fractional_denoiser_Df",
    "kldm_clean_lattice_denoiser",
    "kldm_cps_update_or_renoise",
    "map_model_to_payload_reference_chart",
    "map_payload_reference_chart_to_model_frame",
    "materialize_candidate_from_template",
    "oracle_template_projection",
    "oracle_template_witness",
    "project_candidates_to_clean_estimate",
    "rank_projected_candidates_with_rule",
    "rank_candidates_against_clean_estimate",
    "safety_ok",
    "sample_template_q_proposals",
    "state_to_structure",
    "tdm_position_mu_at_state",
    "tdm_position_sigma_at_state",
    "tdm_residual_epsilon_r",
    "tdm_velocity_sigma_at_state",
    "torus_distance_squared",
    "torus_rmse",
    "wrap01",
    "wrapdiff",
    "witness_torus_sin_loss",
]
