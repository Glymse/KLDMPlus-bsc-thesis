from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from kldmPlus.algorithm25_kldm_pc_cps_lattice import (
    apply_lattice_cps_to_state,
    cps_lattice_project_clean_estimate,
    k_to_lattice6,
    lattice6_to_matrix,
    predict_clean_lattice_from_prediction,
)
from kldmPlus.symmetry.k_basis import space_group_k_constraint


Algorithm27BranchMode = Literal[
    "baseline",
    "denoiser_ppr",
    "denoiser_no_renoise",
    "clean_projection_ppr",
    "renoise_no_projection",
    "cps",
]


@dataclass(frozen=True)
class Algorithm27Config:
    optimizer_steps: int = 10
    optimizer_lr: float = 1.0e-2
    lambda_k: float = 1.0
    lambda_trust: float = 1.0
    lambda_cart: float = 1.0
    lambda_volume: float = 0.05
    rho_lattice: float = 0.75
    gamma_min: float = 0.05
    gamma_max: float = 3.0
    gamma_power: float = 2.0
    projection_start_frac: float = 0.25
    preserve_lattice_volume: bool = False
    max_delta_norm: float = 0.25
    max_induced_cart_shift: float = 0.10
    max_relative_cell_shift: float = 0.05
    max_angle_change_deg: float = 3.0
    early_stop_relative_improvement: float = 0.50
    use_safety_gate: bool = True
    invalid_lattice_penalty: float = 1.0e4
    k_log_eps: float = 1.0e-5


@dataclass(frozen=True)
class Algorithm27CleanPrediction:
    f0_hat: torch.Tensor
    ell0_hat: torch.Tensor
    pred_l: torch.Tensor


@dataclass(frozen=True)
class Algorithm27ProjectionDiagnostics:
    initial_violation: float
    final_violation: float
    objective_initial: float
    objective_final: float
    optimizer_steps_run: int
    accepted: bool
    clipped: bool
    induced_cart_shift_rms: float
    relative_cell_shift: float
    max_angle_change_deg: float
    clean_lattice_shift_norm: float
    noisy_lattice_shift_norm: float
    trust_penalty: float
    volume_penalty: float


@dataclass(frozen=True)
class Algorithm27SensitivityDiagnostics:
    initial_violation: float
    grad_norm: float
    max_abs_grad: float
    best_fd_violation: float
    best_fd_coord: int
    best_fd_delta: float
    best_fd_reduction: float
    invalid_penalty_plateau: bool


@dataclass(frozen=True)
class Algorithm27PPRResult:
    mode: Algorithm27BranchMode
    f_t: torch.Tensor
    v_t: torch.Tensor
    l_t: torch.Tensor
    clean_before: Algorithm27CleanPrediction
    clean_star: Algorithm27CleanPrediction
    diagnostics: Algorithm27ProjectionDiagnostics
    notes: str


def _as_batch_lattice(lattice: torch.Tensor) -> torch.Tensor:
    lattice_t = torch.as_tensor(lattice)
    if lattice_t.ndim == 1:
        return lattice_t.unsqueeze(0)
    return lattice_t


def _restore_l_shape(lattice: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim == 1 and lattice.ndim == 2 and lattice.shape[0] == 1:
        return lattice[0]
    return lattice


def _clean_lattice_from_prediction_grad(
    *,
    l_t: torch.Tensor,
    pred_l: torch.Tensor,
    t_lattice: torch.Tensor,
    diffusion_l: Any,
    num_atoms: torch.Tensor | None,
) -> torch.Tensor:
    l_batch = _as_batch_lattice(l_t)
    pred_batch = _as_batch_lattice(torch.as_tensor(pred_l, device=l_batch.device, dtype=l_batch.dtype))
    if getattr(diffusion_l, "parameterization", "eps") == "x0":
        return pred_batch

    alpha_t = diffusion_l._match_dims(diffusion_l.alpha(t_lattice), l_batch)
    sigma_t = diffusion_l._match_dims(diffusion_l.sigma(t_lattice), l_batch)
    return (l_batch - sigma_t * pred_batch) / alpha_t.clamp_min(diffusion_l.eps)


def predict_clean_state(
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
    detach: bool = True,
) -> Algorithm27CleanPrediction:
    preds = model.score_network(
        t=t_graph,
        pos=f_t,
        v=v_t,
        h=atom_types,
        l=l_t,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )
    if detach:
        ell0_hat = predict_clean_lattice_from_prediction(
            l_t=l_t,
            pred_l=preds["l"],
            t_lattice=t_lattice,
            diffusion_l=model.diffusion_l,
            num_atoms=num_atoms,
        ).reshape(-1)
    else:
        ell0_hat = _clean_lattice_from_prediction_grad(
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
    f0_hat = torch.remainder(f_t + 0.25 * score_v, 1.0)
    if detach:
        return Algorithm27CleanPrediction(
            f0_hat=f0_hat.detach().clone(),
            ell0_hat=ell0_hat.detach().clone(),
            pred_l=preds["l"].detach().clone(),
        )
    return Algorithm27CleanPrediction(f0_hat=f0_hat, ell0_hat=ell0_hat, pred_l=preds["l"])


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


def k_family_violation_tensor(
    ell0: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
    eps: float = 1.0e-5,
    invalid_penalty: float = 1.0e4,
) -> torch.Tensor:
    cell = lattice6_to_matrix(ell0.reshape(-1), num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    k = safe_cell_to_k(cell.reshape(3, 3), eps=float(eps))
    if k is None or not bool(torch.isfinite(k).all()):
        return torch.nan_to_num(ell0.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).square().mean() * 0.0 + torch.as_tensor(
            float(invalid_penalty),
            device=ell0.device,
            dtype=ell0.dtype,
        )
    constraint = space_group_k_constraint(
        space_group_number=int(space_group_number),
        device=k.device,
        dtype=k.dtype,
    )
    diff = constraint.mask * (k - constraint.target)
    return torch.mean(diff.square())


def safe_cell_to_k(cell: torch.Tensor, *, eps: float = 1.0e-5) -> torch.Tensor | None:
    """Stable DiffCSP++ k encoding for optimizer objectives.

    During denoiser-through optimization, intermediate clean lattice predictions
    can be nearly singular or non-finite. The normal `cell_to_k` path then
    raises inside `eigh`, killing the PPR loop before the safety gate can reject
    the candidate. This version adds symmetric jitter and returns `None` for
    invalid inputs so callers can substitute a finite penalty.
    """
    cell_t = torch.as_tensor(cell).reshape(3, 3)
    if not bool(torch.isfinite(cell_t).all()):
        return None
    gram = cell_t @ cell_t.transpose(-1, -2)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    if not bool(torch.isfinite(gram).all()):
        return None
    diag_scale = torch.mean(torch.diagonal(gram).abs()).clamp_min(float(eps))
    eye = torch.eye(3, device=gram.device, dtype=gram.dtype)
    gram = gram + eye * diag_scale * float(eps)
    try:
        gram64 = gram.to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(gram64)
        eigvals = eigvals.clamp_min(float(eps))
        log_diag = torch.diag_embed(torch.log(eigvals))
        s_matrix = (0.5 * (eigvecs @ log_diag @ eigvecs.transpose(-1, -2))).to(dtype=gram.dtype)
    except RuntimeError:
        return None

    s00 = s_matrix[..., 0, 0]
    s11 = s_matrix[..., 1, 1]
    s22 = s_matrix[..., 2, 2]
    k1 = s_matrix[..., 0, 1]
    k2 = s_matrix[..., 0, 2]
    k3 = s_matrix[..., 1, 2]
    k4 = 0.5 * (s00 - s11)
    k5 = (s00 + s11 - 2.0 * s22) / 6.0
    k6 = (s00 + s11 + s22) / 3.0
    return torch.stack([k1, k2, k3, k4, k5, k6], dim=-1).reshape(-1)


def direct_kspace_project_lattice(
    ell0: torch.Tensor,
    *,
    num_atoms: int,
    lattice_transform: Any | None,
    space_group_number: int,
    preserve_lattice_volume: bool = False,
) -> torch.Tensor:
    cell = lattice6_to_matrix(ell0.reshape(-1), num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    k = safe_cell_to_k(cell.reshape(3, 3), eps=1.0e-5)
    if k is None or not bool(torch.isfinite(k).all()):
        return ell0.detach().clone().reshape(-1)
    constraint = space_group_k_constraint(
        space_group_number=int(space_group_number),
        device=k.device,
        dtype=k.dtype,
    )
    k_proj = (1.0 - constraint.mask) * k + constraint.mask * constraint.target
    if preserve_lattice_volume:
        k_proj[-1] = k[-1]
    return k_to_lattice6(k_proj, num_atoms=int(num_atoms), lattice_transform=lattice_transform).reshape(-1)


def lattice_shift_metrics(
    *,
    ell_ref: torch.Tensor,
    ell_new: torch.Tensor,
    f0_hat: torch.Tensor,
    num_atoms: int,
    lattice_transform: Any | None,
) -> dict[str, torch.Tensor]:
    cell_ref = lattice6_to_matrix(ell_ref.reshape(-1), num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    cell_new = lattice6_to_matrix(ell_new.reshape(-1), num_atoms=int(num_atoms), lattice_transform=lattice_transform)
    if not bool(torch.isfinite(cell_ref).all()) or not bool(torch.isfinite(cell_new).all()):
        base = torch.nan_to_num(cell_new, nan=0.0, posinf=0.0, neginf=0.0).square().mean() * 0.0
        penalty = base + torch.as_tensor(1.0e3, device=cell_new.device, dtype=cell_new.dtype)
        return {
            "induced_cart_shift_rms": penalty,
            "relative_cell_shift": penalty,
            "max_angle_change_deg": penalty,
            "volume_penalty": penalty,
        }
    delta_cell = cell_new - cell_ref
    induced = torch.as_tensor(f0_hat, device=cell_ref.device, dtype=cell_ref.dtype).reshape(-1, 3) @ delta_cell
    induced_rms = torch.sqrt(torch.mean(torch.sum(induced.square(), dim=-1)).clamp_min(0.0))
    relative_cell_shift = torch.linalg.norm(delta_cell.reshape(-1)) / torch.linalg.norm(cell_ref.reshape(-1)).clamp_min(1.0e-12)

    def _angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        denom = (torch.linalg.norm(u) * torch.linalg.norm(v)).clamp_min(1.0e-12)
        return torch.rad2deg(torch.acos(torch.clamp(torch.dot(u, v) / denom, -1.0, 1.0)))

    angle_ref = torch.stack([_angle(cell_ref[1], cell_ref[2]), _angle(cell_ref[0], cell_ref[2]), _angle(cell_ref[0], cell_ref[1])])
    angle_new = torch.stack([_angle(cell_new[1], cell_new[2]), _angle(cell_new[0], cell_new[2]), _angle(cell_new[0], cell_new[1])])
    volume_ref = torch.abs(torch.det(cell_ref)).clamp_min(1.0e-12)
    volume_new = torch.abs(torch.det(cell_new)).clamp_min(1.0e-12)
    out = {
        "induced_cart_shift_rms": induced_rms,
        "relative_cell_shift": relative_cell_shift,
        "max_angle_change_deg": torch.max(torch.abs(angle_new - angle_ref)),
        "volume_penalty": torch.square(torch.log(volume_new / volume_ref)),
    }
    if not all(bool(torch.isfinite(value).all()) for value in out.values()):
        base = torch.nan_to_num(cell_new, nan=0.0, posinf=0.0, neginf=0.0).square().mean() * 0.0
        penalty = base + torch.as_tensor(1.0e3, device=cell_new.device, dtype=cell_new.dtype)
        return {
            "induced_cart_shift_rms": penalty,
            "relative_cell_shift": penalty,
            "max_angle_change_deg": penalty,
            "volume_penalty": penalty,
        }
    return out


def _float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _make_diagnostics(
    *,
    initial_violation: torch.Tensor,
    final_violation: torch.Tensor,
    objective_initial: float,
    objective_final: float,
    optimizer_steps_run: int,
    accepted: bool,
    clipped: bool,
    shift_metrics: dict[str, torch.Tensor],
    clean_lattice_shift_norm: torch.Tensor,
    noisy_lattice_shift_norm: torch.Tensor,
    trust_penalty: torch.Tensor,
) -> Algorithm27ProjectionDiagnostics:
    return Algorithm27ProjectionDiagnostics(
        initial_violation=_float(initial_violation),
        final_violation=_float(final_violation),
        objective_initial=float(objective_initial),
        objective_final=float(objective_final),
        optimizer_steps_run=int(optimizer_steps_run),
        accepted=bool(accepted),
        clipped=bool(clipped),
        induced_cart_shift_rms=_float(shift_metrics["induced_cart_shift_rms"]),
        relative_cell_shift=_float(shift_metrics["relative_cell_shift"]),
        max_angle_change_deg=_float(shift_metrics["max_angle_change_deg"]),
        clean_lattice_shift_norm=_float(clean_lattice_shift_norm),
        noisy_lattice_shift_norm=_float(noisy_lattice_shift_norm),
        trust_penalty=_float(trust_penalty),
        volume_penalty=_float(shift_metrics["volume_penalty"]),
    )


def optimize_noisy_lattice_through_denoiser(
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
    lattice_transform: Any | None,
    space_group_number: int,
    config: Algorithm27Config = Algorithm27Config(),
) -> tuple[torch.Tensor, Algorithm27CleanPrediction, Algorithm27CleanPrediction, Algorithm27ProjectionDiagnostics]:
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    l_ref = _as_batch_lattice(torch.as_tensor(l_t)).detach()
    clean_before = predict_clean_state(
        model=model,
        f_t=f_t,
        v_t=v_t,
        l_t=l_ref,
        atom_types=atom_types,
        node_index=node_index,
        edge_node_index=edge_node_index,
        t_graph=t_graph,
        t_nodes=t_nodes,
        t_lattice=t_lattice,
        num_atoms=num_atoms,
        detach=True,
    )
    initial_violation = k_family_violation_tensor(
        clean_before.ell0_hat,
        num_atoms=int(num_atoms.reshape(-1)[0].item()),
        lattice_transform=lattice_transform,
        space_group_number=int(space_group_number),
        eps=float(config.k_log_eps),
        invalid_penalty=float(config.invalid_lattice_penalty),
    )
    _, sigma_t = lattice_forward_mean_sigma(
        diffusion_l=model.diffusion_l,
        t_lattice=t_lattice,
        x0=clean_before.ell0_hat.reshape(1, -1),
        num_atoms=num_atoms,
    )

    l_var = l_ref.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([l_var], lr=float(config.optimizer_lr))
    objective_initial = float("nan")
    objective_final = float("nan")
    steps_run = 0
    for step in range(max(int(config.optimizer_steps), 0)):
        optimizer.zero_grad(set_to_none=True)
        clean_grad = predict_clean_state(
            model=model,
            f_t=f_t.detach(),
            v_t=v_t.detach(),
            l_t=l_var,
            atom_types=atom_types,
            node_index=node_index,
            edge_node_index=edge_node_index,
            t_graph=t_graph,
            t_nodes=t_nodes,
            t_lattice=t_lattice,
            num_atoms=num_atoms,
            detach=False,
        )
        violation = k_family_violation_tensor(
            clean_grad.ell0_hat,
            num_atoms=int(num_atoms.reshape(-1)[0].item()),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            eps=float(config.k_log_eps),
            invalid_penalty=float(config.invalid_lattice_penalty),
        )
        shift = lattice_shift_metrics(
            ell_ref=clean_before.ell0_hat,
            ell_new=clean_grad.ell0_hat,
            f0_hat=clean_before.f0_hat.detach(),
            num_atoms=int(num_atoms.reshape(-1)[0].item()),
            lattice_transform=lattice_transform,
        )
        trust = torch.mean(((l_var - l_ref) / sigma_t.clamp_min(getattr(model.diffusion_l, "eps", 1.0e-8))).square())
        loss = (
            float(config.lambda_k) * violation
            + float(config.lambda_trust) * trust
            + float(config.lambda_cart) * shift["induced_cart_shift_rms"].square()
            + float(config.lambda_volume) * shift["volume_penalty"]
        )
        if step == 0:
            objective_initial = _float(loss)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if not bool(torch.isfinite(l_var).all()):
                l_var.copy_(l_ref)
            step_delta = l_var - l_ref
            step_delta_norm = torch.linalg.norm(step_delta.reshape(-1))
            if float(config.max_delta_norm) > 0.0 and _float(step_delta_norm) > float(config.max_delta_norm):
                l_var.copy_(l_ref + step_delta * (float(config.max_delta_norm) / step_delta_norm.clamp_min(1.0e-12)))
        steps_run = step + 1
        objective_final = _float(loss)
        if _float(violation) <= _float(initial_violation) * (1.0 - float(config.early_stop_relative_improvement)):
            break

    l_star = l_var.detach()
    clipped = False
    delta = l_star - l_ref
    delta_norm = torch.linalg.norm(delta.reshape(-1))
    if float(config.max_delta_norm) > 0.0 and _float(delta_norm) > float(config.max_delta_norm):
        l_star = l_ref + delta * (float(config.max_delta_norm) / delta_norm.clamp_min(1.0e-12))
        clipped = True

    clean_star = predict_clean_state(
        model=model,
        f_t=f_t,
        v_t=v_t,
        l_t=l_star,
        atom_types=atom_types,
        node_index=node_index,
        edge_node_index=edge_node_index,
        t_graph=t_graph,
        t_nodes=t_nodes,
        t_lattice=t_lattice,
        num_atoms=num_atoms,
        detach=True,
    )
    final_violation = k_family_violation_tensor(
        clean_star.ell0_hat,
        num_atoms=int(num_atoms.reshape(-1)[0].item()),
        lattice_transform=lattice_transform,
        space_group_number=int(space_group_number),
        eps=float(config.k_log_eps),
        invalid_penalty=float(config.invalid_lattice_penalty),
    )
    shift_final = lattice_shift_metrics(
        ell_ref=clean_before.ell0_hat,
        ell_new=clean_star.ell0_hat,
        f0_hat=clean_before.f0_hat,
        num_atoms=int(num_atoms.reshape(-1)[0].item()),
        lattice_transform=lattice_transform,
    )
    accepted = (
        _float(final_violation) <= _float(initial_violation)
        and _float(shift_final["induced_cart_shift_rms"]) <= float(config.max_induced_cart_shift)
        and _float(shift_final["relative_cell_shift"]) <= float(config.max_relative_cell_shift)
        and _float(shift_final["max_angle_change_deg"]) <= float(config.max_angle_change_deg)
    )
    trust_final = torch.mean(((l_star - l_ref) / sigma_t.clamp_min(getattr(model.diffusion_l, "eps", 1.0e-8))).square())
    diagnostics = _make_diagnostics(
        initial_violation=initial_violation,
        final_violation=final_violation,
        objective_initial=objective_initial,
        objective_final=objective_final,
        optimizer_steps_run=steps_run,
        accepted=accepted,
        clipped=clipped,
        shift_metrics=shift_final,
        clean_lattice_shift_norm=torch.linalg.norm((clean_star.ell0_hat - clean_before.ell0_hat).reshape(-1)),
        noisy_lattice_shift_norm=torch.linalg.norm((l_star - l_ref).reshape(-1)),
        trust_penalty=trust_final,
    )
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    return _restore_l_shape(l_star, ref=torch.as_tensor(l_t)), clean_before, clean_star, diagnostics


def lattice_objective_sensitivity(
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
    lattice_transform: Any | None,
    space_group_number: int,
    finite_difference_eps: tuple[float, ...] = (1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1),
    config: Algorithm27Config = Algorithm27Config(),
) -> Algorithm27SensitivityDiagnostics:
    """Lightweight check that the denoiser-through-lattice objective is usable.

    PPR only works if `c_G(D_l(F,V,l_t,A,t))` has a usable local signal with
    respect to `l_t`. This diagnostic separates three failure modes:
    zero/flat autograd gradient, no finite-difference direction, and the
    constant invalid-lattice penalty plateau.
    """
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    l_ref = _as_batch_lattice(torch.as_tensor(l_t)).detach()
    l_var = l_ref.detach().clone().requires_grad_(True)
    clean_grad = predict_clean_state(
        model=model,
        f_t=f_t.detach(),
        v_t=v_t.detach(),
        l_t=l_var,
        atom_types=atom_types,
        node_index=node_index,
        edge_node_index=edge_node_index,
        t_graph=t_graph,
        t_nodes=t_nodes,
        t_lattice=t_lattice,
        num_atoms=num_atoms,
        detach=False,
    )
    violation = k_family_violation_tensor(
        clean_grad.ell0_hat,
        num_atoms=int(num_atoms.reshape(-1)[0].item()),
        lattice_transform=lattice_transform,
        space_group_number=int(space_group_number),
        eps=float(config.k_log_eps),
        invalid_penalty=float(config.invalid_lattice_penalty),
    )
    violation.backward()
    grad = l_var.grad.detach().reshape(-1) if l_var.grad is not None else torch.zeros_like(l_var.reshape(-1))
    initial = _float(violation)

    best_value = initial
    best_coord = -1
    best_delta = 0.0
    with torch.no_grad():
        flat_ref = l_ref.reshape(-1)
        for eps in finite_difference_eps:
            for coord in range(int(flat_ref.numel())):
                for sign in (-1.0, 1.0):
                    proposal = flat_ref.clone()
                    proposal[coord] = proposal[coord] + float(sign) * float(eps)
                    clean_prop = predict_clean_state(
                        model=model,
                        f_t=f_t.detach(),
                        v_t=v_t.detach(),
                        l_t=proposal.reshape_as(l_ref),
                        atom_types=atom_types,
                        node_index=node_index,
                        edge_node_index=edge_node_index,
                        t_graph=t_graph,
                        t_nodes=t_nodes,
                        t_lattice=t_lattice,
                        num_atoms=num_atoms,
                        detach=True,
                    )
                    value = _float(
                        k_family_violation_tensor(
                            clean_prop.ell0_hat,
                            num_atoms=int(num_atoms.reshape(-1)[0].item()),
                            lattice_transform=lattice_transform,
                            space_group_number=int(space_group_number),
                            eps=float(config.k_log_eps),
                            invalid_penalty=float(config.invalid_lattice_penalty),
                        )
                    )
                    if value < best_value:
                        best_value = value
                        best_coord = int(coord)
                        best_delta = float(sign) * float(eps)
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    invalid_plateau = bool(initial >= 0.99 * float(config.invalid_lattice_penalty) and _float(torch.linalg.norm(grad)) <= 1.0e-12)
    return Algorithm27SensitivityDiagnostics(
        initial_violation=float(initial),
        grad_norm=_float(torch.linalg.norm(grad)),
        max_abs_grad=_float(torch.max(torch.abs(grad))) if grad.numel() else 0.0,
        best_fd_violation=float(best_value),
        best_fd_coord=int(best_coord),
        best_fd_delta=float(best_delta),
        best_fd_reduction=float(initial - best_value),
        invalid_penalty_plateau=invalid_plateau,
    )


def lattice_ppr_renoise_from_clean(
    *,
    ell_t: torch.Tensor,
    ell0_hat: torch.Tensor,
    ell0_star: torch.Tensor,
    t_lattice: torch.Tensor,
    diffusion_l: Any,
    num_atoms: torch.Tensor | None,
    rho_lattice: float = 0.75,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    ell_t_ref = torch.as_tensor(ell_t)
    ell_t_b = _as_batch_lattice(ell_t_ref)
    ell0_hat_b = _as_batch_lattice(torch.as_tensor(ell0_hat, device=ell_t_b.device, dtype=ell_t_b.dtype))
    ell0_star_b = _as_batch_lattice(torch.as_tensor(ell0_star, device=ell_t_b.device, dtype=ell_t_b.dtype))
    mean_hat, sigma_t = lattice_forward_mean_sigma(
        diffusion_l=diffusion_l,
        t_lattice=t_lattice,
        x0=ell0_hat_b,
        num_atoms=num_atoms,
    )
    mean_star, _ = lattice_forward_mean_sigma(
        diffusion_l=diffusion_l,
        t_lattice=t_lattice,
        x0=ell0_star_b,
        num_atoms=num_atoms,
    )
    eps_old = (ell_t_b - mean_hat) / sigma_t.clamp_min(getattr(diffusion_l, "eps", 1.0e-8))
    if noise is None:
        noise = torch.randn_like(eps_old)
    rho = max(0.0, min(1.0, float(rho_lattice)))
    eps_mix = rho * eps_old + (max(0.0, 1.0 - rho * rho) ** 0.5) * noise.to(device=eps_old.device, dtype=eps_old.dtype)
    ell_new = mean_star + sigma_t * eps_mix
    return _restore_l_shape(ell_new.detach().clone(), ref=ell_t_ref)


def run_algorithm27_branch(
    *,
    mode: Algorithm27BranchMode,
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
    lattice_transform: Any | None,
    space_group_number: int,
    tau: float,
    config: Algorithm27Config = Algorithm27Config(),
) -> Algorithm27PPRResult:
    clean_before = predict_clean_state(
        model=model,
        f_t=f_t,
        v_t=v_t,
        l_t=l_t,
        atom_types=atom_types,
        node_index=node_index,
        edge_node_index=edge_node_index,
        t_graph=t_graph,
        t_nodes=t_nodes,
        t_lattice=t_lattice,
        num_atoms=num_atoms,
        detach=True,
    )
    zero = torch.as_tensor(0.0, device=clean_before.ell0_hat.device, dtype=clean_before.ell0_hat.dtype)
    baseline_diag = _make_diagnostics(
        initial_violation=k_family_violation_tensor(
            clean_before.ell0_hat,
            num_atoms=int(num_atoms.reshape(-1)[0].item()),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            eps=float(config.k_log_eps),
            invalid_penalty=float(config.invalid_lattice_penalty),
        ),
        final_violation=k_family_violation_tensor(
            clean_before.ell0_hat,
            num_atoms=int(num_atoms.reshape(-1)[0].item()),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            eps=float(config.k_log_eps),
            invalid_penalty=float(config.invalid_lattice_penalty),
        ),
        objective_initial=0.0,
        objective_final=0.0,
        optimizer_steps_run=0,
        accepted=True,
        clipped=False,
        shift_metrics={
            "induced_cart_shift_rms": zero,
            "relative_cell_shift": zero,
            "max_angle_change_deg": zero,
            "volume_penalty": zero,
        },
        clean_lattice_shift_norm=zero,
        noisy_lattice_shift_norm=zero,
        trust_penalty=zero,
    )
    if mode == "baseline":
        return Algorithm27PPRResult(
            mode=mode,
            f_t=f_t.detach().clone(),
            v_t=v_t.detach().clone(),
            l_t=l_t.detach().clone(),
            clean_before=clean_before,
            clean_star=clean_before,
            diagnostics=baseline_diag,
            notes="no_projection_no_renoise",
        )
    if mode == "renoise_no_projection":
        l_new = lattice_ppr_renoise_from_clean(
            ell_t=l_t,
            ell0_hat=clean_before.ell0_hat,
            ell0_star=clean_before.ell0_hat,
            t_lattice=t_lattice,
            diffusion_l=model.diffusion_l,
            num_atoms=num_atoms,
            rho_lattice=float(config.rho_lattice),
        )
        return Algorithm27PPRResult(
            mode=mode,
            f_t=f_t.detach().clone(),
            v_t=v_t.detach().clone(),
            l_t=l_new.detach().clone(),
            clean_before=clean_before,
            clean_star=clean_before,
            diagnostics=baseline_diag,
            notes="renoise_from_original_clean_estimate",
        )
    if mode == "clean_projection_ppr":
        ell0_proj = direct_kspace_project_lattice(
            clean_before.ell0_hat,
            num_atoms=int(num_atoms.reshape(-1)[0].item()),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            preserve_lattice_volume=bool(config.preserve_lattice_volume),
        )
        l_new = lattice_ppr_renoise_from_clean(
            ell_t=l_t,
            ell0_hat=clean_before.ell0_hat,
            ell0_star=ell0_proj,
            t_lattice=t_lattice,
            diffusion_l=model.diffusion_l,
            num_atoms=num_atoms,
            rho_lattice=float(config.rho_lattice),
        )
        clean_star = Algorithm27CleanPrediction(f0_hat=clean_before.f0_hat, ell0_hat=ell0_proj.detach().clone(), pred_l=clean_before.pred_l)
        return Algorithm27PPRResult(
            mode=mode,
            f_t=f_t.detach().clone(),
            v_t=v_t.detach().clone(),
            l_t=l_new.detach().clone(),
            clean_before=clean_before,
            clean_star=clean_star,
            diagnostics=baseline_diag,
            notes="direct_clean_kspace_projection_then_renoise",
        )
    if mode == "cps":
        ell0_soft, _ = cps_lattice_project_clean_estimate(
            ell0_hat=clean_before.ell0_hat,
            num_atoms=int(num_atoms.reshape(-1)[0].item()),
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            tau=float(tau),
            gamma_min=float(config.gamma_min),
            gamma_max=float(config.gamma_max),
            gamma_power=float(config.gamma_power),
            projection_start_frac=float(config.projection_start_frac),
            preserve_lattice_volume=bool(config.preserve_lattice_volume),
            compute_diagnostics=False,
        )
        l_new, _ = apply_lattice_cps_to_state(
            ell_t=l_t,
            ell0_hat=clean_before.ell0_hat,
            ell0_soft=ell0_soft,
            t_lattice=t_lattice,
            diffusion_l=model.diffusion_l,
        )
        clean_star = Algorithm27CleanPrediction(f0_hat=clean_before.f0_hat, ell0_hat=ell0_soft.detach().clone(), pred_l=clean_before.pred_l)
        return Algorithm27PPRResult(
            mode=mode,
            f_t=f_t.detach().clone(),
            v_t=v_t.detach().clone(),
            l_t=l_new.detach().clone(),
            clean_before=clean_before,
            clean_star=clean_star,
            diagnostics=baseline_diag,
            notes="cps_clean_projection_state_update",
        )
    if mode in {"denoiser_ppr", "denoiser_no_renoise"}:
        l_star, clean_before_opt, clean_star, diagnostics = optimize_noisy_lattice_through_denoiser(
            model=model,
            f_t=f_t,
            v_t=v_t,
            l_t=l_t,
            atom_types=atom_types,
            node_index=node_index,
            edge_node_index=edge_node_index,
            t_graph=t_graph,
            t_nodes=t_nodes,
            t_lattice=t_lattice,
            num_atoms=num_atoms,
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            config=config,
        )
        if bool(config.use_safety_gate) and not bool(diagnostics.accepted):
            return Algorithm27PPRResult(
                mode=mode,
                f_t=f_t.detach().clone(),
                v_t=v_t.detach().clone(),
                l_t=l_t.detach().clone(),
                clean_before=clean_before_opt,
                clean_star=clean_before_opt,
                diagnostics=diagnostics,
                notes="safety_rejected_lattice_only_projection_fallback_to_input_state",
            )
        if mode == "denoiser_no_renoise":
            l_new = l_star
            notes = "optimized_noisy_lattice_without_renoise"
        else:
            l_new = lattice_ppr_renoise_from_clean(
                ell_t=l_t,
                ell0_hat=clean_before_opt.ell0_hat,
                ell0_star=clean_star.ell0_hat,
                t_lattice=t_lattice,
                diffusion_l=model.diffusion_l,
                num_atoms=num_atoms,
                rho_lattice=float(config.rho_lattice),
            )
            notes = "optimize_through_denoiser_then_lattice_renoise"
        return Algorithm27PPRResult(
            mode=mode,
            f_t=f_t.detach().clone(),
            v_t=v_t.detach().clone(),
            l_t=l_new.detach().clone(),
            clean_before=clean_before_opt,
            clean_star=clean_star,
            diagnostics=diagnostics,
            notes=notes,
        )
    raise ValueError(f"Unknown Algorithm27 branch mode: {mode!r}")


def repeat_algorithm27_kernel(
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
    lattice_transform: Any | None,
    space_group_number: int,
    tau: float,
    repeats: int,
    config: Algorithm27Config = Algorithm27Config(),
) -> tuple[torch.Tensor, tuple[Algorithm27PPRResult, ...]]:
    current_l = l_t.detach().clone()
    results: list[Algorithm27PPRResult] = []
    for _ in range(max(int(repeats), 0)):
        result = run_algorithm27_branch(
            mode="denoiser_ppr",
            model=model,
            f_t=f_t,
            v_t=v_t,
            l_t=current_l,
            atom_types=atom_types,
            node_index=node_index,
            edge_node_index=edge_node_index,
            t_graph=t_graph,
            t_nodes=t_nodes,
            t_lattice=t_lattice,
            num_atoms=num_atoms,
            lattice_transform=lattice_transform,
            space_group_number=int(space_group_number),
            tau=float(tau),
            config=config,
        )
        current_l = result.l_t.detach().clone()
        results.append(result)
    return current_l, tuple(results)


__all__ = [
    "Algorithm27BranchMode",
    "Algorithm27CleanPrediction",
    "Algorithm27Config",
    "Algorithm27PPRResult",
    "Algorithm27ProjectionDiagnostics",
    "direct_kspace_project_lattice",
    "k_family_violation_tensor",
    "lattice_forward_mean_sigma",
    "lattice_ppr_renoise_from_clean",
    "lattice_objective_sensitivity",
    "lattice_shift_metrics",
    "optimize_noisy_lattice_through_denoiser",
    "predict_clean_state",
    "repeat_algorithm27_kernel",
    "run_algorithm27_branch",
]
