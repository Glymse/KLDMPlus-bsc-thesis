from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

from kldmPlus.symmetry.pyxtal_backend import PyXtalWyckoffResult
from kldmPlus.symmetry.diffcsppp_backend import (
    DiffCSPPPSymmetryPayload,
    build_diffcsppp_symmetry_payload,
    oracle_spacegroup_from_task,
    select_template_and_free_vars_from_payload,
)
from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    expand_wyckoff_template_torch,
    extract_wyckoff_templates,
    flatten_site_signature,
    recover_template_free_vars_from_anchor_entries,
)


ALGORITHM18_MODE = "kldm_ppr_fractional_velocity_from_scratch"
ALGORITHM18_SHORT_NAME = "Algorithm18-KLDM-PPR-FromScratch"
ALGORITHM18_IS_FULL_PPR = False
ALGORITHM18_DESCRIPTION = (
    "From-scratch KLDM-compatible PPR for the fractional-coordinate/velocity branch: "
    "optimize xi_r, xi_v through D_f and a fixed SVD Wyckoff normal constraint, "
    "then renoise with KLDM's native forward kernel while keeping the lattice fixed."
)
ALGORITHM18_RELATION_TO_PPR = (
    "Follows PPR's project-through-denoiser, renoise, and repeat structure at fixed t, "
    "adapted to KLDM's torus-plus-velocity state and simplified clean-velocity convention V0=0."
)


def wrap01(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 1.0)


def signed_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + 0.5, 1.0) - 0.5


def torus_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b - torch.round(a - b)


def torus_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torus_diff(a, b).square().mean()


def torus_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torus_mse(a, b).clamp_min(0.0))


def make_anchor_entries(result: PyXtalWyckoffResult) -> list[dict[str, Any]]:
    return [
        {
            "atomic_number": int(result.anchor_atomic_numbers[i]),
            "label": str(result.site_labels[i]),
            "anchor_frac": np.asarray(result.anchor_frac_coords[i], dtype=float),
        }
        for i in range(int(result.anchor_count))
    ]


def pyxtal_result_signature(result: PyXtalWyckoffResult) -> tuple[tuple[int, str], ...]:
    return tuple(
        sorted(
            (int(result.anchor_atomic_numbers[i]), str(result.site_labels[i]))
            for i in range(int(result.anchor_count))
        )
    )


def select_template_by_signature(
    *,
    templates: list[WyckoffTemplate],
    signature: tuple[tuple[int, str], ...],
) -> WyckoffTemplate | None:
    for template in templates:
        if flatten_site_signature(template) == signature:
            return template
    return None


def _debug(enabled: bool, prefix: str, message: str) -> None:
    if enabled:
        print(f"{prefix} {message}", flush=True)


def _vector_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(-1)).detach().item())


def _scatter_center_local(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected x.ndim == 2, got {x.ndim}.")
    if index.ndim != 1 or index.shape[0] != x.shape[0]:
        raise ValueError(
            f"Expected index.shape == ({x.shape[0]},), got {tuple(index.shape)} for x.shape={tuple(x.shape)}."
        )
    num_graphs = int(index.max().item()) + 1 if index.numel() else 0
    if num_graphs <= 0:
        return x
    sums = torch.zeros(num_graphs, x.shape[1], device=x.device, dtype=x.dtype)
    sums.index_add_(0, index, x)
    counts = torch.bincount(index, minlength=num_graphs).to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return x - sums[index] / counts[index].clamp_min(1.0)


def graph_center_velocity(v: torch.Tensor, node_index: torch.Tensor) -> tuple[torch.Tensor, float]:
    centered = _scatter_center_local(v, node_index)
    num_graphs = int(node_index.max().item()) + 1 if node_index.numel() else 0
    if num_graphs <= 0:
        return centered, 0.0
    sums = torch.zeros(num_graphs, v.shape[1], device=v.device, dtype=v.dtype)
    sums.index_add_(0, node_index, centered)
    counts = torch.bincount(node_index, minlength=num_graphs).to(device=v.device, dtype=v.dtype).unsqueeze(-1)
    means = sums / counts.clamp_min(1.0)
    return centered, float(torch.linalg.norm(means).detach().item())


def _as_graph_time_batch(
    t_graph: torch.Tensor,
    *,
    num_graphs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t_graph = torch.as_tensor(t_graph, device=device, dtype=dtype)
    if t_graph.ndim == 0:
        return t_graph.expand(num_graphs, 1)
    if t_graph.ndim == 1:
        if t_graph.numel() == 1:
            return t_graph.expand(num_graphs).unsqueeze(-1)
        if t_graph.shape[0] == num_graphs:
            return t_graph.unsqueeze(-1)
    if t_graph.ndim == 2 and t_graph.shape == (num_graphs, 1):
        return t_graph
    raise ValueError(
        f"Expected t_graph to have shape scalar, [{num_graphs}], or [{num_graphs}, 1], got {tuple(t_graph.shape)}."
    )


def _score_network_predict(
    model,
    *,
    t_graph: torch.Tensor,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    a_t: torch.Tensor,
    l_t: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    l_graph = l_t.unsqueeze(0) if l_t.ndim == 1 else l_t
    t_graph = _as_graph_time_batch(
        t_graph,
        num_graphs=int(l_graph.shape[0]),
        device=f_t.device,
        dtype=f_t.dtype,
    )
    return model.score_network(
        t=t_graph,
        pos=f_t,
        v=v_t,
        h=a_t,
        l=l_graph,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    remaining_rows = list(range(cost.shape[0]))
    remaining_cols = list(range(cost.shape[1]))
    chosen_rows: list[int] = []
    chosen_cols: list[int] = []
    while remaining_rows:
        sub = cost[np.ix_(remaining_rows, remaining_cols)]
        flat = int(np.argmin(sub))
        c = sub.shape[1]
        r_pos = flat // c
        c_pos = flat % c
        chosen_rows.append(remaining_rows.pop(r_pos))
        chosen_cols.append(remaining_cols.pop(c_pos))
    order = np.argsort(np.asarray(chosen_rows))
    return np.asarray(chosen_rows, dtype=int)[order], np.asarray(chosen_cols, dtype=int)[order]


def _species_assignment_template_to_target(
    *,
    template_frac: torch.Tensor,
    template_atomic_numbers: torch.Tensor,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    template_frac_np = wrap01(template_frac).detach().cpu().numpy()
    target_frac_np = wrap01(target_frac).detach().cpu().numpy()
    template_z = template_atomic_numbers.detach().cpu().numpy().astype(int)
    target_z = target_atomic_numbers.detach().cpu().numpy().astype(int)
    if sorted(template_z.tolist()) != sorted(target_z.tolist()):
        raise RuntimeError("Species multiset mismatch in template-to-target assignment.")
    assignment = np.empty(template_frac_np.shape[0], dtype=int)
    for atomic_number in sorted(set(template_z.tolist())):
        src_idx = np.where(template_z == atomic_number)[0]
        dst_idx = np.where(target_z == atomic_number)[0]
        if src_idx.size != dst_idx.size:
            raise RuntimeError(f"Species count mismatch for Z={atomic_number}.")
        src = template_frac_np[src_idx][:, None, :]
        dst = target_frac_np[dst_idx][None, :, :]
        delta = src - dst
        delta = delta - np.round(delta)
        cost = np.sum(delta * delta, axis=-1)
        rows, cols = _hungarian(cost)
        assignment[src_idx[rows]] = dst_idx[cols]
    return torch.as_tensor(assignment, device=template_frac.device, dtype=torch.long)


def _reorder_template_to_target(template_order_frac: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    target_order = torch.empty_like(template_order_frac)
    target_order[assignment] = template_order_frac
    return target_order


def _align_template_translation_to_target(
    *,
    template_order_frac: torch.Tensor,
    assignment: torch.Tensor,
    target_frac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    target_order = _reorder_template_to_target(template_order_frac, assignment)
    tau = torus_diff(wrap01(target_frac), target_order).mean(dim=0)
    aligned_template = wrap01(template_order_frac + tau.unsqueeze(0))
    aligned_target = _reorder_template_to_target(aligned_template, assignment)
    rmse = float(torus_rmse(aligned_target, wrap01(target_frac)).detach().item())
    return aligned_template, tau, rmse


def _orbit_affine_matrix(site, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    basis = torch.as_tensor(site.anchor_basis, device=device, dtype=dtype)
    rotations = torch.as_tensor(site.rotation_matrices, device=device, dtype=dtype)
    if basis.numel() == 0 or basis.shape[-1] == 0 or int(site.dof) == 0:
        return torch.zeros((3 * int(site.multiplicity), 0), device=device, dtype=dtype)
    return torch.cat([(rot @ basis) for rot in rotations], dim=0)


def _svd_left_basis(A: torch.Tensor, tol: float = 1.0e-7) -> tuple[torch.Tensor, int]:
    if A.numel() == 0 or A.shape[1] == 0:
        return A.new_zeros((A.shape[0], 0)), 0
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    threshold = tol * S.max().clamp_min(1.0)
    rank = int((S > threshold).sum().item())
    return U[:, :rank].contiguous(), rank


@dataclass(frozen=True)
class Algorithm18GraphState:
    f: torch.Tensor
    v: torch.Tensor
    l: torch.Tensor
    h: torch.Tensor
    k: torch.Tensor
    t: float
    dt: float
    graph_idx0: int


@dataclass(frozen=True)
class Algorithm18Config:
    repeats: int = 1
    proj_steps: int = 8
    lr: float = 2.0e-2
    lambda_noise: float = 1.0e-2
    optimize_velocity: bool = True
    denoiser_variant: str = "minus"
    gradient_clip_norm: float = 10.0
    mean_free_threshold: float = 1.0e-6
    lattice_change_threshold: float = 1.0e-8
    eps: float = 1.0e-8
    debug: bool = False


@dataclass(frozen=True)
class SVDOrbitConstraint:
    site_index: int
    label: str
    target_indices: torch.Tensor
    reference_template_block: torch.Tensor
    U_tangent: torch.Tensor
    rank: int
    codim: int


@dataclass(frozen=True)
class SVDWyckoffConstraint:
    orbits: tuple[SVDOrbitConstraint, ...]
    codim: int
    reference_target_frac: torch.Tensor

    def value(self, z_clean: torch.Tensor) -> torch.Tensor:
        z01 = wrap01(z_clean)
        total = z_clean.new_zeros(())
        for orbit in self.orbits:
            idx = orbit.target_indices.to(device=z_clean.device, dtype=torch.long)
            z_block = z01[idx].reshape(-1)
            ref_block = orbit.reference_template_block.to(device=z_clean.device, dtype=z_clean.dtype).reshape(-1)
            delta = torus_diff(z_block, ref_block)
            U = orbit.U_tangent.to(device=z_clean.device, dtype=z_clean.dtype)
            if U.numel() == 0:
                residual = delta
            else:
                residual = delta - U @ (U.transpose(0, 1) @ delta)
            total = total + residual.square().sum()
        return total / max(float(self.codim), 1.0)


@dataclass(frozen=True)
class FixedWyckoffFrame:
    template: WyckoffTemplate
    free_vars: torch.Tensor
    assignment: torch.Tensor
    tau: torch.Tensor
    reference_template_order: torch.Tensor
    reference_target_order: torch.Tensor
    atomic_numbers_template_order: torch.Tensor
    atomic_numbers_target_order: torch.Tensor
    constraint: SVDWyckoffConstraint
    reason: str
    debug_info: dict[str, Any]

    def expand_template_order(self, free_vars: torch.Tensor | None = None) -> torch.Tensor:
        local_free = self.free_vars if free_vars is None else free_vars.reshape(-1)
        expansion = expand_wyckoff_template_torch(template=self.template, free_vars=local_free, wrap=True)
        return wrap01(expansion.frac_coords + self.tau.unsqueeze(0))

    def expand_target_order(self, free_vars: torch.Tensor | None = None) -> torch.Tensor:
        template_order = self.expand_template_order(free_vars)
        return _reorder_template_to_target(template_order, self.assignment)

    def hard_project_clean(self, z_clean: torch.Tensor) -> torch.Tensor:
        z01 = wrap01(z_clean)
        projected = z01.clone()
        for orbit in self.constraint.orbits:
            idx = orbit.target_indices.to(device=z_clean.device, dtype=torch.long)
            z_block = z01[idx].reshape(-1)
            ref_block = orbit.reference_template_block.to(device=z_clean.device, dtype=z_clean.dtype).reshape(-1)
            delta = torus_diff(z_block, ref_block)
            U = orbit.U_tangent.to(device=z_clean.device, dtype=z_clean.dtype)
            tangent = z_block.new_zeros(z_block.shape)
            if U.numel() != 0:
                tangent = U @ (U.transpose(0, 1) @ delta)
            block_proj = wrap01(ref_block + tangent).reshape(-1, 3)
            projected[idx] = block_proj
        return projected


@dataclass(frozen=True)
class DenoiserShapeCheck:
    variant: str
    finite: bool
    shape_ok: bool
    min_value: float
    max_value: float


@dataclass(frozen=True)
class OracleSignCheck:
    minus_rmse: float
    plus_rmse: float
    f_t: torch.Tensor
    v_t: torch.Tensor
    r_t: torch.Tensor
    mu_r: torch.Tensor
    sigma_r: torch.Tensor
    s_mu_oracle: torch.Tensor
    s_r_oracle: torch.Tensor
    f0_hat_minus: torch.Tensor
    f0_hat_plus: torch.Tensor


@dataclass(frozen=True)
class ProjectStepMetrics:
    loss_before: float
    loss_after: float
    c_before: float
    c_after_project: float
    xi_r_norm: float
    xi_v_norm: float
    delta_f_rms: float
    delta_v_rms: float
    grad_xi_r_norm: float
    grad_xi_v_norm: float
    mean_free_norm: float
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProjectStepResult:
    success: bool
    reject_reason: str
    f_star: torch.Tensor
    v_star: torch.Tensor
    f0_star: torch.Tensor
    f0_before: torch.Tensor
    metrics: ProjectStepMetrics


@dataclass(frozen=True)
class RenoiseResult:
    f_t: torch.Tensor
    v_t: torch.Tensor
    l_t: torch.Tensor
    r_t: torch.Tensor
    epsilon_v: torch.Tensor
    epsilon_r: torch.Tensor
    finite_ok: bool
    mean_free_norm: float
    epsilon_v_mean_norm: float
    lattice_changed_norm: float


@dataclass(frozen=True)
class KernelIteration:
    repeat_index: int
    project: ProjectStepResult
    renoise: RenoiseResult
    accepted: bool
    reject_reason: str


@dataclass(frozen=True)
class CorrectionResult:
    initial_state: Algorithm18GraphState
    final_state: Algorithm18GraphState | None
    iterations: tuple[KernelIteration, ...]
    accepted: bool
    reject_reason: str


def build_fixed_wyckoff_frame(
    *,
    template: WyckoffTemplate,
    free_vars: torch.Tensor,
    reference_frac_target_order: torch.Tensor,
    atomic_numbers_target_order: torch.Tensor,
    assignment: torch.Tensor | None = None,
    reason: str = "fixed_wyckoff_frame",
    debug: bool = False,
) -> FixedWyckoffFrame:
    expansion = expand_wyckoff_template_torch(template=template, free_vars=free_vars.reshape(-1), wrap=True)
    template_frac = wrap01(expansion.frac_coords)
    template_atomic_numbers = expansion.atomic_numbers.to(device=template_frac.device, dtype=torch.long)
    target_frac = wrap01(reference_frac_target_order).to(device=template_frac.device, dtype=template_frac.dtype)
    target_atomic_numbers = atomic_numbers_target_order.to(device=template_frac.device, dtype=torch.long)

    if assignment is None:
        assignment = _species_assignment_template_to_target(
            template_frac=template_frac,
            template_atomic_numbers=template_atomic_numbers,
            target_frac=target_frac,
            target_atomic_numbers=target_atomic_numbers,
        )
    else:
        assignment = assignment.to(device=template_frac.device, dtype=torch.long).reshape(-1)

    aligned_template, tau, rmse_ref = _align_template_translation_to_target(
        template_order_frac=template_frac,
        assignment=assignment,
        target_frac=target_frac,
    )
    aligned_target = _reorder_template_to_target(aligned_template, assignment)

    _debug(
        debug,
        "[algo18 frame]",
        f"reason={reason} rmse_ref={rmse_ref:.6f} tau={tau.detach().cpu().tolist()} "
        f"n_atoms={int(template.total_atoms)} n_free={int(template.total_free_dims)}",
    )

    orbits: list[SVDOrbitConstraint] = []
    codim = 0
    cursor = 0
    for site_index, site in enumerate(template.site_templates):
        start = cursor
        stop = cursor + int(site.multiplicity)
        U, rank = _svd_left_basis(_orbit_affine_matrix(site, device=template_frac.device, dtype=template_frac.dtype))
        orbit_codim = max(3 * int(site.multiplicity) - int(rank), 0)
        codim += orbit_codim
        orbits.append(
            SVDOrbitConstraint(
                site_index=int(site_index),
                label=str(site.label),
                target_indices=assignment[start:stop].detach().clone(),
                reference_template_block=aligned_template[start:stop].detach().clone(),
                U_tangent=U.detach().clone(),
                rank=int(rank),
                codim=int(orbit_codim),
            )
        )
        cursor = stop

    constraint = SVDWyckoffConstraint(
        orbits=tuple(orbits),
        codim=int(codim),
        reference_target_frac=aligned_target.detach().clone(),
    )

    return FixedWyckoffFrame(
        template=template,
        free_vars=free_vars.detach().clone().reshape(-1),
        assignment=assignment.detach().clone(),
        tau=tau.detach().clone(),
        reference_template_order=aligned_template.detach().clone(),
        reference_target_order=aligned_target.detach().clone(),
        atomic_numbers_template_order=template_atomic_numbers.detach().clone(),
        atomic_numbers_target_order=target_atomic_numbers.detach().clone(),
        constraint=constraint,
        reason=str(reason),
        debug_info={
            "rmse_ref_to_target": float(rmse_ref),
            "tau": tau.detach().cpu().tolist(),
            "codim": int(codim),
            "orbit_count": len(orbits),
        },
    )


def species_align_to_target_order(
    *,
    source_frac: torch.Tensor,
    source_atomic_numbers: torch.Tensor,
    target_frac: torch.Tensor,
    target_atomic_numbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Reorder a source structure into the exact atom order of a target structure.

    This is stricter than species-aware RMSE. It is useful when a structure is
    geometrically matched to a target but may still be in a different atom
    ordering, which would poison downstream fixed-frame constraints.
    """
    assignment = _species_assignment_template_to_target(
        template_frac=source_frac,
        template_atomic_numbers=source_atomic_numbers,
        target_frac=target_frac,
        target_atomic_numbers=target_atomic_numbers,
    )
    aligned_frac = _reorder_template_to_target(wrap01(source_frac), assignment)
    aligned_atomic_numbers = torch.empty_like(target_atomic_numbers)
    aligned_atomic_numbers[assignment] = source_atomic_numbers.to(device=target_atomic_numbers.device, dtype=torch.long)
    rmse = float(torus_rmse(aligned_frac, wrap01(target_frac)).detach().item())
    return aligned_frac, aligned_atomic_numbers, assignment.detach().clone(), rmse


def build_template_and_free_vars_from_pyxtal(
    *,
    pyxtal_result: PyXtalWyckoffResult,
    atomic_numbers_standardized: torch.Tensor,
    max_templates: int = 64,
) -> tuple[WyckoffTemplate, torch.Tensor, list[WyckoffTemplate]]:
    templates = extract_wyckoff_templates(
        space_group_number=int(pyxtal_result.space_group),
        atomic_numbers=atomic_numbers_standardized,
        max_templates=int(max_templates),
        quick=False,
    )
    signature = pyxtal_result_signature(pyxtal_result)
    template = select_template_by_signature(templates=templates, signature=signature)
    if template is None:
        raise RuntimeError(
            f"No template matched GT PyXtal signature for SG={int(pyxtal_result.space_group)} signature={signature!r}."
        )
    free_vars = recover_template_free_vars_from_anchor_entries(template, make_anchor_entries(pyxtal_result))
    return template, free_vars, templates


def build_template_and_free_vars_from_diffcsppp_payload(
    *,
    payload: DiffCSPPPSymmetryPayload,
    atomic_numbers_standardized: torch.Tensor,
    max_templates: int = 64,
) -> tuple[WyckoffTemplate, torch.Tensor, list[WyckoffTemplate]]:
    """Select the occupied Wyckoff template from a DiffCSP++-style symmetry payload."""
    return select_template_and_free_vars_from_payload(
        payload=payload,
        atomic_numbers_standardized=atomic_numbers_standardized,
        max_templates=max_templates,
    )


def build_oracle_diffcsppp_payload_from_structure(
    *,
    standardized_structure,
    requested_spacegroup: int,
    tol: float = 1e-2,
) -> DiffCSPPPSymmetryPayload:
    """Temporary oracle CMPSL path: use the task/requested SG and a DiffCSP++-style payload.

    The payload itself is still extracted from the refined standardized structure,
    but the SG source is explicitly the oracle/task label for now.
    """
    payload = build_diffcsppp_symmetry_payload(standardized_structure, tol=tol)
    oracle_sg = oracle_spacegroup_from_task(requested_spacegroup=int(requested_spacegroup))
    if int(payload.spacegroup) != int(oracle_sg):
        payload = DiffCSPPPSymmetryPayload(
            spacegroup=int(oracle_sg),
            anchor_index=payload.anchor_index,
            wyckoff_ops=payload.wyckoff_ops,
            wyckoff_ops_inv=payload.wyckoff_ops_inv,
            wyckoff_letters=payload.wyckoff_letters,
            atom_types=payload.atom_types,
            anchor_frac_coords=payload.anchor_frac_coords,
            expanded_frac_coords=payload.expanded_frac_coords,
            anchor_atomic_numbers=payload.anchor_atomic_numbers,
            expanded_atomic_numbers=payload.expanded_atomic_numbers,
            lattice_matrix=payload.lattice_matrix,
            standardized_structure=payload.standardized_structure,
            debug_info={
                **(payload.debug_info or {}),
                "oracle_spacegroup": int(oracle_sg),
                "extracted_spacegroup": int(payload.spacegroup),
            },
        )
    return payload


def kldm_mu_r(model, *, t_nodes: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    tau_nodes = model.tdm.T * t_nodes
    return model.tdm.wrapped_gaussian_mu_r_t(tau_nodes, v_t)


def kldm_sigma_r(model, *, t_nodes: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    tau_nodes = model.tdm.T * t_nodes
    return model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau_nodes), ref)


def kldm_sigma_v(model, *, t_nodes: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    tau_nodes = model.tdm.T * t_nodes
    return model.tdm.match_dims(model.tdm.vel_scale * model.tdm.gaussian_velocity_sigma(tau_nodes), ref)


def kldm_clean_fractional_denoiser_Df(
    *,
    model,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    l_t: torch.Tensor,
    a_t: torch.Tensor,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    variant: str = "minus",
) -> torch.Tensor:
    preds = _score_network_predict(
        model,
        t_graph=t_graph,
        f_t=f_t,
        v_t=v_t,
        a_t=a_t,
        l_t=l_t,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )
    u_theta = preds["v"]
    tau_nodes = model.tdm.T * t_nodes
    mu_r = model.tdm.wrapped_gaussian_mu_r_t(tau_nodes, v_t)
    sigma_r = model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau_nodes), f_t)
    sigma_norm = model.tdm.sigma_norm_factor(t=tau_nodes, index=node_index, ref=u_theta)
    score_term = sigma_r.square() * sigma_norm * u_theta
    variant_key = str(variant).lower().strip()
    if variant_key == "minus":
        return model.tdm.wrap_displacements(f_t - mu_r - score_term)
    if variant_key == "plus":
        return model.tdm.wrap_displacements(f_t - mu_r + score_term)
    raise ValueError(f"Unsupported denoiser variant={variant!r}.")


def denoiser_shape_and_finite_sanity(
    *,
    model,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    l_t: torch.Tensor,
    a_t: torch.Tensor,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
) -> tuple[DenoiserShapeCheck, DenoiserShapeCheck]:
    outputs = []
    for variant in ("minus", "plus"):
        f0_hat = kldm_clean_fractional_denoiser_Df(
            model=model,
            f_t=f_t,
            v_t=v_t,
            l_t=l_t,
            a_t=a_t,
            t_graph=t_graph,
            t_nodes=t_nodes,
            node_index=node_index,
            edge_node_index=edge_node_index,
            variant=variant,
        )
        outputs.append(
            DenoiserShapeCheck(
                variant=variant,
                finite=bool(torch.isfinite(f0_hat).all().item()),
                shape_ok=tuple(f0_hat.shape) == tuple(f_t.shape),
                min_value=float(f0_hat.min().detach().item()),
                max_value=float(f0_hat.max().detach().item()),
            )
        )
    return outputs[0], outputs[1]


@torch.no_grad()
def oracle_clean_denoiser_sign_check(
    *,
    model,
    f0: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
) -> OracleSignCheck:
    f_t, v_t, epsilon_v, epsilon_r, r_t = model.tdm.sample_noisy_state(
        t=t_nodes,
        f0=f0,
        index=node_index,
    )
    del epsilon_v, epsilon_r
    tau_nodes = model.tdm.T * t_nodes
    mu_r = model.tdm.wrapped_gaussian_mu_r_t(tau_nodes, v_t)
    sigma_r = model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau_nodes), f_t)
    s_mu_oracle = (r_t - mu_r) / sigma_r.square().clamp_min(model.tdm.eps)
    s_r_oracle = (mu_r - r_t) / sigma_r.square().clamp_min(model.tdm.eps)
    f0_hat_minus = model.tdm.wrap_displacements(f_t - mu_r - sigma_r.square() * s_mu_oracle)
    f0_hat_plus = model.tdm.wrap_displacements(f_t - mu_r + sigma_r.square() * s_r_oracle)
    return OracleSignCheck(
        minus_rmse=float(torus_rmse(f0_hat_minus, signed_wrap(f0)).detach().item()),
        plus_rmse=float(torus_rmse(f0_hat_plus, signed_wrap(f0)).detach().item()),
        f_t=f_t.detach().clone(),
        v_t=v_t.detach().clone(),
        r_t=r_t.detach().clone(),
        mu_r=mu_r.detach().clone(),
        sigma_r=sigma_r.detach().clone(),
        s_mu_oracle=s_mu_oracle.detach().clone(),
        s_r_oracle=s_r_oracle.detach().clone(),
        f0_hat_minus=f0_hat_minus.detach().clone(),
        f0_hat_plus=f0_hat_plus.detach().clone(),
    )


def ppr_noise_parameterization(
    *,
    model,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    xi_r: torch.Tensor,
    xi_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_v = kldm_sigma_v(model, t_nodes=t_nodes, ref=v_t)
    v_var, _ = graph_center_velocity(v_t + sigma_v * xi_v, node_index)
    mu_ref = kldm_mu_r(model, t_nodes=t_nodes, v_t=v_t)
    mu_var = kldm_mu_r(model, t_nodes=t_nodes, v_t=v_var)
    sigma_r = kldm_sigma_r(model, t_nodes=t_nodes, ref=f_t)
    f_var = model.tdm.wrap_displacements(f_t + (mu_var - mu_ref) + sigma_r * xi_r)
    return f_var, v_var, sigma_r, sigma_v


def ppr_objective_gradient_sanity(
    *,
    model,
    frame: FixedWyckoffFrame,
    state: Algorithm18GraphState,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    config: Algorithm18Config = Algorithm18Config(),
) -> dict[str, Any]:
    xi_r = torch.zeros_like(state.f, requires_grad=True)
    xi_v = torch.zeros_like(state.v, requires_grad=True)
    f_var, v_var, _sigma_r, _sigma_v = ppr_noise_parameterization(
        model=model,
        f_t=state.f,
        v_t=state.v,
        t_nodes=t_nodes,
        node_index=node_index,
        xi_r=xi_r,
        xi_v=xi_v,
    )
    f0_hat = kldm_clean_fractional_denoiser_Df(
        model=model,
        f_t=f_var,
        v_t=v_var,
        l_t=state.l,
        a_t=state.h,
        t_graph=t_graph,
        t_nodes=t_nodes,
        node_index=node_index,
        edge_node_index=edge_node_index,
        variant=config.denoiser_variant,
    )
    c_value = frame.constraint.value(f0_hat)
    noise_penalty = xi_r.square().mean() + xi_v.square().mean()
    loss = c_value + float(config.lambda_noise) * noise_penalty
    loss.backward()
    finite_grad = bool(
        xi_r.grad is not None
        and xi_v.grad is not None
        and torch.isfinite(xi_r.grad).all().item()
        and torch.isfinite(xi_v.grad).all().item()
    )
    return {
        "loss": float(loss.detach().item()),
        "c_value": float(c_value.detach().item()),
        "noise_penalty": float(noise_penalty.detach().item()),
        "grad_xi_r_norm": _vector_norm(xi_r.grad if xi_r.grad is not None else torch.zeros_like(xi_r)),
        "grad_xi_v_norm": _vector_norm(xi_v.grad if xi_v.grad is not None else torch.zeros_like(xi_v)),
        "finite_grad": finite_grad,
    }


def ppr_project_step(
    *,
    model,
    frame: FixedWyckoffFrame,
    state: Algorithm18GraphState,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    config: Algorithm18Config = Algorithm18Config(),
    debug_prefix: str = "[algo18 project]",
) -> ProjectStepResult:
    with torch.no_grad():
        f0_before = kldm_clean_fractional_denoiser_Df(
            model=model,
            f_t=state.f,
            v_t=state.v,
            l_t=state.l,
            a_t=state.h,
            t_graph=t_graph,
            t_nodes=t_nodes,
            node_index=node_index,
            edge_node_index=edge_node_index,
            variant=config.denoiser_variant,
        )
        c_before = frame.constraint.value(f0_before)
        loss_before = c_before

    xi_r = torch.zeros_like(state.f, requires_grad=True)
    params: list[torch.Tensor] = [xi_r]
    if config.optimize_velocity:
        xi_v = torch.zeros_like(state.v, requires_grad=True)
        params.append(xi_v)
    else:
        xi_v = torch.zeros_like(state.v)
    opt = torch.optim.Adam(params, lr=float(config.lr))

    history: list[dict[str, Any]] = []
    grad_xi_r_norm = 0.0
    grad_xi_v_norm = 0.0
    mean_free_norm = 0.0

    for step_idx in range(max(int(config.proj_steps), 1)):
        opt.zero_grad()
        f_var, v_var, _sigma_r, _sigma_v = ppr_noise_parameterization(
            model=model,
            f_t=state.f,
            v_t=state.v,
            t_nodes=t_nodes,
            node_index=node_index,
            xi_r=xi_r,
            xi_v=xi_v,
        )
        _, mean_free_norm = graph_center_velocity(v_var, node_index)
        f0_hat = kldm_clean_fractional_denoiser_Df(
            model=model,
            f_t=f_var,
            v_t=v_var,
            l_t=state.l,
            a_t=state.h,
            t_graph=t_graph,
            t_nodes=t_nodes,
            node_index=node_index,
            edge_node_index=edge_node_index,
            variant=config.denoiser_variant,
        )
        c_after = frame.constraint.value(f0_hat)
        noise_penalty = xi_r.square().mean() + xi_v.square().mean()
        loss = c_after + float(config.lambda_noise) * noise_penalty
        loss.backward()
        grad_xi_r_norm = _vector_norm(xi_r.grad if xi_r.grad is not None else torch.zeros_like(xi_r))
        grad_xi_v_norm = _vector_norm(xi_v.grad if isinstance(xi_v, torch.Tensor) and xi_v.grad is not None else torch.zeros_like(state.v))
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(config.gradient_clip_norm))
        opt.step()
        row = {
            "step": int(step_idx),
            "loss": float(loss.detach().item()),
            "c_value": float(c_after.detach().item()),
            "noise_penalty": float(noise_penalty.detach().item()),
            "xi_r_norm": _vector_norm(xi_r.detach()),
            "xi_v_norm": _vector_norm(xi_v.detach()),
            "mean_free_norm": float(mean_free_norm),
        }
        history.append(row)
        _debug(
            config.debug,
            debug_prefix,
            "step={step} loss={loss:.6f} c={c_value:.6f} noise_pen={noise_penalty:.6f} "
            "xi_r_norm={xi_r_norm:.6f} xi_v_norm={xi_v_norm:.6f} mean_free={mean_free_norm:.3e}".format(**row),
        )

    with torch.no_grad():
        f_star, v_star, _sigma_r, _sigma_v = ppr_noise_parameterization(
            model=model,
            f_t=state.f,
            v_t=state.v,
            t_nodes=t_nodes,
            node_index=node_index,
            xi_r=xi_r.detach(),
            xi_v=xi_v.detach(),
        )
        v_star, mean_free_norm = graph_center_velocity(v_star, node_index)
        f0_star = kldm_clean_fractional_denoiser_Df(
            model=model,
            f_t=f_star,
            v_t=v_star,
            l_t=state.l,
            a_t=state.h,
            t_graph=t_graph,
            t_nodes=t_nodes,
            node_index=node_index,
            edge_node_index=edge_node_index,
            variant=config.denoiser_variant,
        )
        c_after_project = frame.constraint.value(f0_star)
        noise_penalty = xi_r.detach().square().mean() + xi_v.detach().square().mean()
        loss_after = c_after_project + float(config.lambda_noise) * noise_penalty
        delta_f_rms = float(torus_rmse(f_star, state.f).detach().item())
        delta_v_rms = float(torch.sqrt(torch.mean((v_star - state.v).square())).detach().item())

    success = bool(torch.isfinite(f_star).all().item() and torch.isfinite(v_star).all().item() and torch.isfinite(f0_star).all().item())
    return ProjectStepResult(
        success=success,
        reject_reason="" if success else "nonfinite_project_state",
        f_star=f_star.detach().clone(),
        v_star=v_star.detach().clone(),
        f0_star=f0_star.detach().clone(),
        f0_before=f0_before.detach().clone(),
        metrics=ProjectStepMetrics(
            loss_before=float(loss_before.detach().item()),
            loss_after=float(loss_after.detach().item()),
            c_before=float(c_before.detach().item()),
            c_after_project=float(c_after_project.detach().item()),
            xi_r_norm=_vector_norm(xi_r.detach()),
            xi_v_norm=_vector_norm(xi_v.detach()),
            delta_f_rms=float(delta_f_rms),
            delta_v_rms=float(delta_v_rms),
            grad_xi_r_norm=float(grad_xi_r_norm),
            grad_xi_v_norm=float(grad_xi_v_norm),
            mean_free_norm=float(mean_free_norm),
            history=tuple(history),
        ),
    )


@torch.no_grad()
def kldm_renoise_from_f0(
    *,
    model,
    f0_star: torch.Tensor,
    l_t: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    epsilon_v: torch.Tensor | None = None,
    epsilon_r: torch.Tensor | None = None,
    debug: bool = False,
    debug_prefix: str = "[algo18 renoise]",
) -> RenoiseResult:
    l_in = l_t.detach().clone()
    if epsilon_v is None:
        epsilon_v = torch.randn_like(f0_star)
    epsilon_v, epsilon_v_mean_norm = graph_center_velocity(epsilon_v, node_index)
    f_t_new, v_t_new, eps_v, eps_r, r_t = model.tdm.sample_noisy_state(
        t=t_nodes,
        f0=f0_star,
        index=node_index,
        epsilon_v=epsilon_v,
        epsilon_r=epsilon_r,
    )
    v_t_new, mean_free_norm = graph_center_velocity(v_t_new, node_index)
    l_out = l_t.detach().clone()
    lattice_changed_norm = _vector_norm(l_out - l_in)
    finite_ok = bool(torch.isfinite(f_t_new).all().item() and torch.isfinite(v_t_new).all().item())
    _debug(
        debug,
        debug_prefix,
        f"finite_ok={finite_ok} mean_free_norm={mean_free_norm:.3e} eps_v_mean_norm={epsilon_v_mean_norm:.3e} lattice_changed={lattice_changed_norm:.3e}",
    )
    return RenoiseResult(
        f_t=f_t_new.detach().clone(),
        v_t=v_t_new.detach().clone(),
        l_t=l_out,
        r_t=r_t.detach().clone(),
        epsilon_v=eps_v.detach().clone(),
        epsilon_r=eps_r.detach().clone(),
        finite_ok=finite_ok,
        mean_free_norm=float(mean_free_norm),
        epsilon_v_mean_norm=float(epsilon_v_mean_norm),
        lattice_changed_norm=float(lattice_changed_norm),
    )


def ppr_kernel_repeated(
    *,
    model,
    frame: FixedWyckoffFrame,
    state: Algorithm18GraphState,
    t_graph: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
    edge_node_index: torch.Tensor,
    config: Algorithm18Config = Algorithm18Config(),
    anchor_mode: str = "soft",
) -> CorrectionResult:
    if anchor_mode not in {"soft", "hard"}:
        raise ValueError(f"Unsupported anchor_mode={anchor_mode!r}.")
    current = Algorithm18GraphState(
        f=state.f.detach().clone(),
        v=state.v.detach().clone(),
        l=state.l.detach().clone(),
        h=state.h.detach().clone(),
        k=state.k.detach().clone(),
        t=state.t,
        dt=state.dt,
        graph_idx0=state.graph_idx0,
    )
    iterations: list[KernelIteration] = []
    reject_reason = ""
    for repeat_idx in range(max(int(config.repeats), 1)):
        _debug(config.debug, "[algo18 kernel]", f"repeat={repeat_idx} start")
        project = ppr_project_step(
            model=model,
            frame=frame,
            state=current,
            t_graph=t_graph,
            t_nodes=t_nodes,
            node_index=node_index,
            edge_node_index=edge_node_index,
            config=config,
            debug_prefix=f"[algo18 project r{repeat_idx}]",
        )
        if not project.success:
            reject_reason = project.reject_reason or "project_failed"
            iterations.append(
                KernelIteration(
                    repeat_index=int(repeat_idx),
                    project=project,
                    renoise=RenoiseResult(
                        f_t=current.f.detach().clone(),
                        v_t=current.v.detach().clone(),
                        l_t=current.l.detach().clone(),
                        r_t=torch.zeros_like(current.f),
                        epsilon_v=torch.zeros_like(current.v),
                        epsilon_r=torch.zeros_like(current.f),
                        finite_ok=False,
                        mean_free_norm=float("nan"),
                        epsilon_v_mean_norm=float("nan"),
                        lattice_changed_norm=float("nan"),
                    ),
                    accepted=False,
                    reject_reason=reject_reason,
                )
            )
            break
        f0_anchor = project.f0_star if anchor_mode == "soft" else frame.hard_project_clean(project.f0_star)
        renoise = kldm_renoise_from_f0(
            model=model,
            f0_star=f0_anchor,
            l_t=current.l,
            t_nodes=t_nodes,
            node_index=node_index,
            debug=config.debug,
            debug_prefix=f"[algo18 renoise r{repeat_idx}]",
        )
        accepted = bool(
            renoise.finite_ok
            and float(renoise.mean_free_norm) <= float(config.mean_free_threshold)
            and float(renoise.lattice_changed_norm) <= float(config.lattice_change_threshold)
        )
        reject = "" if accepted else "renoise_invalid"
        iterations.append(
            KernelIteration(
                repeat_index=int(repeat_idx),
                project=project,
                renoise=renoise,
                accepted=accepted,
                reject_reason=reject,
            )
        )
        if not accepted:
            reject_reason = reject
            break
        current = Algorithm18GraphState(
            f=renoise.f_t.detach().clone(),
            v=renoise.v_t.detach().clone(),
            l=current.l.detach().clone(),
            h=current.h.detach().clone(),
            k=current.k.detach().clone(),
            t=current.t,
            dt=current.dt,
            graph_idx0=current.graph_idx0,
        )
    accepted = len(iterations) == max(int(config.repeats), 1) and all(item.accepted for item in iterations)
    return CorrectionResult(
        initial_state=state,
        final_state=current if accepted else None,
        iterations=tuple(iterations),
        accepted=bool(accepted),
        reject_reason=str(reject_reason),
    )


__all__ = [
    "ALGORITHM18_MODE",
    "ALGORITHM18_SHORT_NAME",
    "ALGORITHM18_IS_FULL_PPR",
    "ALGORITHM18_DESCRIPTION",
    "ALGORITHM18_RELATION_TO_PPR",
    "Algorithm18Config",
    "Algorithm18GraphState",
    "CorrectionResult",
    "DenoiserShapeCheck",
    "FixedWyckoffFrame",
    "KernelIteration",
    "OracleSignCheck",
    "ProjectStepResult",
    "PyXtalWyckoffResult",
    "RenoiseResult",
    "SVDOrbitConstraint",
    "SVDWyckoffConstraint",
    "build_fixed_wyckoff_frame",
    "build_oracle_diffcsppp_payload_from_structure",
    "build_template_and_free_vars_from_diffcsppp_payload",
    "build_template_and_free_vars_from_pyxtal",
    "denoiser_shape_and_finite_sanity",
    "graph_center_velocity",
    "kldm_clean_fractional_denoiser_Df",
    "kldm_mu_r",
    "kldm_renoise_from_f0",
    "kldm_sigma_r",
    "kldm_sigma_v",
    "make_anchor_entries",
    "oracle_clean_denoiser_sign_check",
    "ppr_kernel_repeated",
    "ppr_noise_parameterization",
    "ppr_objective_gradient_sanity",
    "ppr_project_step",
    "pyxtal_result_signature",
    "select_template_by_signature",
    "signed_wrap",
    "torus_diff",
    "torus_mse",
    "torus_rmse",
    "wrap01",
]
