from __future__ import annotations

# Inspired heavily by DiffCSP++

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from kldmPlus.symmetry import (
    DiffCSPPPSymmetryPayload,
    WyckoffDOFChart,
    align_expanded_frac_to_reference_chart_orbit_aware,
    attach_payload_reference_chart,
    build_diffcsppp_symmetry_payload,
    build_wyckoff_dof_chart,
    local_wyckoff_dof_chart_loss,
    oracle_spacegroup_from_task,
    project_payload_to_wyckoff_dof_chart,
)


ALGORITHM19_MODE = "kldm_ppr_diffcsppp"
ALGORITHM19_SHORT_NAME = "Algorithm19-KLDM-PPR-DiffCSPPP"
ALGORITHM19_DESCRIPTION = (
    "Faithful KLDM-PPR for fractional coordinates and velocities using the "
    "DiffCSP++ affine-operator Wyckoff backend."
)
ALGORITHM19_RELATION_TO_PPR = (
    "Implements the predict-project-renoise loop in KLDM phase space: project "
    "through D_f using DiffCSP++ operator constraints, then renoise with the "
    "native KLDM/TDM forward kernel at fixed lattice."
)
WYCKOFF_DOF_CHART_CACHE_VERSION = 2


@dataclass
class Algorithm19State:
    f: torch.Tensor
    v: torch.Tensor
    l: torch.Tensor
    atom_types: torch.Tensor
    node_index: torch.Tensor
    edge_node_index: torch.Tensor
    t_graph: torch.Tensor
    t_nodes: torch.Tensor


@dataclass(frozen=True)
class Algorithm19Config:
    M: int = 1
    proj_steps: int = 8
    lr: float = 1.0e-2
    lambda_noise: float = 1.0e-2
    grad_clip: float = 10.0
    anchor_mode: str = "soft"
    denoiser_variant: str = "minus"
    coordinate_score_mode: str = "direct"
    soft_anchor_tol: float = 1.0e-5
    lambda_q: float = 1.0e-6
    local_projection_tol: float = 5.0e-2
    eps: float = 1.0e-8


@dataclass(frozen=True)
class Algorithm19ProjectResult:
    f_star: torch.Tensor
    v_star: torch.Tensor
    f0_star: torch.Tensor
    logs: tuple[dict[str, Any], ...]
    q_star: torch.Tensor | None = None
    z_proj_payload: torch.Tensor | None = None


@dataclass(frozen=True)
class Algorithm19KernelResult:
    state: Algorithm19State
    logs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Algorithm19SymmetryContext:
    payload: DiffCSPPPSymmetryPayload
    chart: WyckoffDOFChart
    q_ref: torch.Tensor


def wrap01(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 1.0)


def wrapdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b - torch.round(a - b)


def torus_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrapdiff(a, b).square().mean()


def torus_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torus_mse(a, b).clamp_min(0.0))


def torus_sin_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * wrapdiff(a, b)).square().mean()


def _scatter_mean(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected [N, D], got shape {tuple(x.shape)}.")
    if index.ndim != 1 or index.shape[0] != x.shape[0]:
        raise ValueError(f"Index shape {tuple(index.shape)} incompatible with x shape {tuple(x.shape)}.")
    if index.numel() == 0:
        return x.new_zeros((0, x.shape[1]))
    num_graphs = int(index.max().item()) + 1
    sums = torch.zeros(num_graphs, x.shape[1], device=x.device, dtype=x.dtype)
    sums.index_add_(0, index, x)
    counts = torch.bincount(index, minlength=num_graphs).to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return sums / counts.clamp_min(1.0)


def center_velocity(v: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    means = _scatter_mean(v, node_index)
    return v - means[node_index]


def graph_mean_norm(v: torch.Tensor, node_index: torch.Tensor) -> float:
    means = _scatter_mean(v, node_index)
    if means.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(means).detach().item())


def build_oracle_diffcsppp_payload_from_structure(
    *,
    standardized_structure,
    requested_spacegroup: int,
    tol: float = 1.0e-2,
) -> DiffCSPPPSymmetryPayload:
    payload = build_diffcsppp_symmetry_payload(standardized_structure, tol=tol)
    oracle_sg = oracle_spacegroup_from_task(requested_spacegroup=int(requested_spacegroup))
    if int(payload.spacegroup) != int(oracle_sg):
        raise ValueError(
            "Extracted DiffCSP++ payload SG does not match oracle SG. "
            f"extracted={int(payload.spacegroup)} oracle={int(oracle_sg)}. "
            "Cannot reuse operators by replacing only the integer."
        )
    payload = attach_payload_reference_chart(payload, np.asarray(payload.expanded_frac_coords, dtype=float))
    debug_info = dict(payload.debug_info or {})
    debug_info["model_reference_frac_coords"] = np.asarray(standardized_structure.frac_coords, dtype=float).tolist()
    return replace(payload, debug_info=debug_info)


def expand_anchors_to_full(
    anchor_frac: torch.Tensor,
    ops: torch.Tensor,
    anchor_index: torch.Tensor,
) -> torch.Tensor:
    R = ops[:, :3, :3]
    t = ops[:, :3, 3]
    y = anchor_frac[anchor_index]
    f = torch.einsum("nij,nj->ni", R, y) + t
    return wrap01(f)


def _torus_mean(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [M, 3], got {tuple(points.shape)}.")
    if points.shape[0] == 0:
        return points.new_zeros((3,))
    ref = points[0]
    delta = wrapdiff(points, ref.unsqueeze(0))
    return wrap01(ref + delta.mean(dim=0))


def _masked_local_free_mean(delta_y: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    """Average only the declared free chart directions for one Wyckoff site.

    For low-DOF sites we should not estimate a full 3D pseudo-anchor and zero
    frozen coordinates afterward. Instead, estimate only the active local chart
    coordinates and keep frozen directions pinned to the reference anchor.
    """
    out = torch.zeros((3,), device=delta_y.device, dtype=delta_y.dtype)
    active = torch.nonzero(free_mask, as_tuple=False).reshape(-1)
    if delta_y.numel() == 0 or active.numel() == 0:
        return out
    site_delta = delta_y[:, active]
    ref = site_delta[0]
    centered = wrapdiff(site_delta, ref.unsqueeze(0))
    out[active] = ref + centered.mean(dim=0)
    return out


def _local_free_chart_update(
    *,
    delta_z: torch.Tensor,
    rotations: torch.Tensor,
    free_mask: torch.Tensor,
) -> torch.Tensor:
    """Solve the local Wyckoff chart update in the declared free subspace.

    The payload expansion is affine in the anchor coordinates, so around a fixed
    reference anchor the local tangent wrt the free coordinates is given directly
    by the corresponding columns of the orbit rotations. Solving in that tangent
    space is more faithful than pulling back through a full 3D pseudo-inverse and
    masking afterward.
    """
    update = torch.zeros((3,), device=delta_z.device, dtype=delta_z.dtype)
    active = torch.nonzero(free_mask, as_tuple=False).reshape(-1)
    if delta_z.numel() == 0 or active.numel() == 0:
        return update

    jac = rotations[:, :, active].reshape(-1, int(active.numel()))
    rhs = delta_z.reshape(-1, 1)
    if jac.numel() == 0:
        return update
    delta_u = torch.linalg.pinv(jac) @ rhs
    update[active] = delta_u.reshape(-1)
    return update


def _lift_anchor_average_to_full(
    anchor_average_full: torch.Tensor,
    ops: torch.Tensor,
) -> torch.Tensor:
    rotations = ops[:, :3, :3]
    translations = ops[:, :3, 3]
    return wrap01(torch.einsum("nij,nj->ni", rotations, anchor_average_full) + translations)


def _site_ids_from_anchor_index(anchor_index: torch.Tensor) -> torch.Tensor:
    unique_anchor_index = torch.unique(anchor_index, sorted=True)
    return torch.searchsorted(unique_anchor_index, anchor_index)


def _payload_debug_array(payload: DiffCSPPPSymmetryPayload, key: str, *, dtype=None) -> np.ndarray | None:
    debug_info = payload.debug_info or {}
    value = debug_info.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _get_wyckoff_dof_chart(payload: DiffCSPPPSymmetryPayload) -> WyckoffDOFChart:
    debug_info = payload.debug_info or {}
    chart = debug_info.get("wyckoff_dof_chart")
    chart_cache_version = int(debug_info.get("wyckoff_dof_chart_cache_version", -1))
    if isinstance(chart, WyckoffDOFChart) and chart_cache_version == WYCKOFF_DOF_CHART_CACHE_VERSION:
        return chart
    chart = build_wyckoff_dof_chart(payload)
    debug_info["wyckoff_dof_chart"] = chart
    debug_info["wyckoff_dof_q_ref"] = np.asarray(chart.q_ref, dtype=float)
    debug_info["wyckoff_dof_chart_cache_version"] = WYCKOFF_DOF_CHART_CACHE_VERSION
    return chart


def make_algorithm19_symmetry_context(
    payload: DiffCSPPPSymmetryPayload,
    *,
    device: torch.device,
    dtype: torch.dtype,
    q_ref: torch.Tensor | np.ndarray | None = None,
) -> Algorithm19SymmetryContext:
    chart = _get_wyckoff_dof_chart(payload)
    q_ref_t = torch.as_tensor(chart.q_ref if q_ref is None else q_ref, device=device, dtype=dtype).reshape(-1)
    return Algorithm19SymmetryContext(payload=payload, chart=chart, q_ref=q_ref_t)


def initialize_runtime_q_ref_from_payload(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    *,
    q_init: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
    num_iters: int = 3,
) -> torch.Tensor:
    """Runtime-initialize the live chart center from the current sample.

    The chart still carries a bootstrap reference representative, but the
    *active* q_ref used during PPR should track the current clean prediction.
    We therefore run a few local projection refinements and use the resulting
    q_star as the live center for the next optimization.
    """
    chart = _get_wyckoff_dof_chart(payload)
    q_curr = torch.as_tensor(
        chart.q_ref if q_init is None else q_init,
        device=z_payload.device,
        dtype=z_payload.dtype,
    ).reshape(-1)
    if q_curr.numel() == 0:
        return q_curr

    for _ in range(max(int(num_iters), 1)):
        projection = project_payload_to_wyckoff_dof_chart(
            z_payload,
            chart,
            q_ref=q_curr,
            lambda_q=float(lambda_q),
        )
        q_next = projection["q_star"].detach().clone().reshape(-1)
        if torch.linalg.norm(wrapdiff(q_next, q_curr)).detach().item() < 1.0e-7:
            q_curr = q_next
            break
        q_curr = q_next
    return q_curr


def initialize_runtime_q_ref_from_model_frame(
    z_model: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    *,
    q_init: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
    num_iters: int = 3,
) -> torch.Tensor:
    z_payload = map_model_to_payload_reference_chart(z_model, payload)
    return initialize_runtime_q_ref_from_payload(
        z_payload,
        payload,
        q_init=q_init,
        lambda_q=float(lambda_q),
        num_iters=int(num_iters),
    )


def _maybe_to_payload_frame(
    z: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    linear = _payload_debug_array(payload, "model_to_payload_linear", dtype=float)
    tau = _payload_debug_array(payload, "model_to_payload_tau", dtype=float)
    order = _payload_debug_array(payload, "model_to_payload_order", dtype=int)
    model_ref = _payload_debug_array(payload, "model_reference_frac_coords", dtype=float)
    if linear is None or tau is None or order is None:
        return wrap01(z)

    linear_t = torch.as_tensor(linear, device=z.device, dtype=z.dtype)
    order_t = torch.as_tensor(order, device=z.device, dtype=torch.long)
    if model_ref is not None:
        model_ref_t = torch.as_tensor(model_ref, device=z.device, dtype=z.dtype)
        payload_ref_t = torch.as_tensor(
            np.asarray(payload.expanded_frac_coords, dtype=float),
            device=z.device,
            dtype=z.dtype,
        )
        delta_model = wrapdiff(wrap01(z), model_ref_t)
        delta_payload = delta_model[order_t] @ linear_t
        return wrap01(payload_ref_t + delta_payload)

    tau_t = torch.as_tensor(tau, device=z.device, dtype=z.dtype).reshape(1, 3)
    z_payload = wrap01((wrap01(z) - tau_t) @ linear_t)
    return z_payload[order_t]


def _maybe_align_payload_local_chart(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    tau = _payload_debug_array(payload, "payload_reference_tau", dtype=float)
    order = _payload_debug_array(payload, "payload_reference_order", dtype=int)
    if tau is None or order is None:
        alignment = align_expanded_frac_to_reference_chart_orbit_aware(
            payload,
            z_payload.detach().cpu().numpy(),
            expanded_atomic_numbers=np.asarray(payload.expanded_atomic_numbers, dtype=int),
        )
        tau = np.asarray(alignment["tau"], dtype=float)
        order = np.asarray(alignment["reference_order"], dtype=int)

    tau_t = torch.as_tensor(tau, device=z_payload.device, dtype=z_payload.dtype).reshape(1, 3)
    order_t = torch.as_tensor(order, device=z_payload.device, dtype=torch.long)
    return wrap01(z_payload + tau_t)[order_t]


def _maybe_unalign_payload_local_chart(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    tau = _payload_debug_array(payload, "payload_reference_tau", dtype=float)
    order = _payload_debug_array(payload, "payload_reference_order", dtype=int)
    if tau is None or order is None:
        return wrap01(z_payload)

    tau_t = torch.as_tensor(tau, device=z_payload.device, dtype=z_payload.dtype).reshape(1, 3)
    order_t = torch.as_tensor(order, device=z_payload.device, dtype=torch.long)
    z_raw_shifted = torch.zeros_like(z_payload)
    z_raw_shifted[order_t] = z_payload
    return wrap01(z_raw_shifted - tau_t)


def _maybe_from_payload_frame(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    linear = _payload_debug_array(payload, "payload_to_model_linear", dtype=float)
    tau = _payload_debug_array(payload, "payload_to_model_tau", dtype=float)
    assignment = _payload_debug_array(payload, "payload_to_model_order", dtype=int)
    model_ref = _payload_debug_array(payload, "model_reference_frac_coords", dtype=float)
    if linear is None or tau is None or assignment is None:
        return wrap01(z_payload)

    linear_t = torch.as_tensor(linear, device=z_payload.device, dtype=z_payload.dtype)
    assignment_t = torch.as_tensor(assignment, device=z_payload.device, dtype=torch.long)
    if model_ref is not None:
        model_ref_t = torch.as_tensor(model_ref, device=z_payload.device, dtype=z_payload.dtype)
        payload_ref_t = torch.as_tensor(
            np.asarray(payload.expanded_frac_coords, dtype=float),
            device=z_payload.device,
            dtype=z_payload.dtype,
        )
        delta_payload = wrapdiff(wrap01(z_payload), payload_ref_t)
        delta_model = delta_payload @ linear_t
        z_model = wrap01(model_ref_t.clone())
        z_model[assignment_t] = wrap01(model_ref_t[assignment_t] + delta_model)
        return z_model

    tau_t = torch.as_tensor(tau, device=z_payload.device, dtype=z_payload.dtype).reshape(1, 3)
    z_model_scattered = torch.zeros_like(z_payload)
    z_model_scattered[assignment_t] = wrap01(z_payload @ linear_t + tau_t)
    return wrap01(z_model_scattered)


def map_model_to_payload_frame_raw(
    z_model: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    """Public helper: map model-frame fractional coordinates into raw payload frame."""
    return _maybe_to_payload_frame(z_model, payload)


def map_model_to_payload_reference_chart(
    z_model: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    """Map model-frame fractional coordinates into the aligned payload reference chart."""
    return _maybe_align_payload_local_chart(_maybe_to_payload_frame(z_model, payload), payload)


def map_payload_to_model_frame_raw(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    """Map raw payload-frame fractional coordinates back into model frame."""
    return _maybe_from_payload_frame(z_payload, payload)


def map_payload_reference_chart_to_model_frame(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    """Map aligned payload reference-chart coordinates back into model frame."""
    return _maybe_from_payload_frame(_maybe_unalign_payload_local_chart(z_payload, payload), payload)


def map_model_to_payload_frame(
    z_model: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    """Backward-compatible alias for aligned payload reference-chart coordinates."""
    return map_model_to_payload_reference_chart(z_model, payload)


def map_payload_to_model_frame(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
) -> torch.Tensor:
    """Backward-compatible alias for aligned payload reference-chart coordinates."""
    return map_payload_reference_chart_to_model_frame(z_payload, payload)


def project_full_to_anchors(
    z: torch.Tensor,
    ops: torch.Tensor,
    ops_inv: torch.Tensor,
    anchor_index: torch.Tensor,
    *,
    reference_anchor_frac: torch.Tensor | None = None,
    reference_free_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project full fractional coordinates into anchor space.

    When a reference anchor chart is provided, this uses the local branch-aware
    projection described in the Algorithm 19 markdown:
    - expand the reference anchor into the current orbit,
    - measure torus residuals relative to that reference,
    - pull those residuals back with ``ops_inv``,
    - average per site,
    - update only the free coordinates.
    """
    z01 = wrap01(z)
    site_ids = _site_ids_from_anchor_index(anchor_index)
    n_sites = int(site_ids.max().item()) + 1 if site_ids.numel() > 0 else 0

    if reference_anchor_frac is None:
        translations = ops[:, :3, 3]
        pulled_back = torch.einsum("nij,nj->ni", ops_inv, wrapdiff(z01, translations))
        compressed = []
        for site_idx in range(n_sites):
            compressed.append(_torus_mean(pulled_back[site_ids == site_idx]).unsqueeze(0))
        if not compressed:
            return z.new_zeros((0, 3))
        return torch.cat(compressed, dim=0)

    ref_anchor = wrap01(reference_anchor_frac)
    if ref_anchor.ndim != 2 or ref_anchor.shape[-1] != 3:
        raise ValueError(f"Expected reference_anchor_frac with shape [S,3], got {tuple(ref_anchor.shape)}.")
    if ref_anchor.shape[0] != n_sites:
        raise ValueError(f"Expected {n_sites} reference anchors, got {ref_anchor.shape[0]}.")

    if reference_free_mask is None:
        free_mask = torch.ones_like(ref_anchor, dtype=torch.bool)
    else:
        free_mask = reference_free_mask.to(device=z.device, dtype=torch.bool)
        if free_mask.shape != ref_anchor.shape:
            raise ValueError(f"Expected reference_free_mask with shape {tuple(ref_anchor.shape)}, got {tuple(free_mask.shape)}.")

    anchors = torch.zeros_like(ref_anchor)
    rotations = ops[:, :3, :3]
    translations = ops[:, :3, 3]
    for site_idx in range(n_sites):
        mask = site_ids == site_idx
        site_ref = ref_anchor[site_idx]
        site_ref_full = wrap01(torch.einsum("nij,j->ni", rotations[mask], site_ref) + translations[mask])
        delta_z = wrapdiff(z01[mask], site_ref_full)
        delta_mean = _local_free_chart_update(
            delta_z=delta_z,
            rotations=rotations[mask],
            free_mask=free_mask[site_idx],
        )
        anchors[site_idx] = wrap01(site_ref + delta_mean)
    return anchors


def project_full_to_wyckoff_ops(
    z: torch.Tensor,
    ops: torch.Tensor,
    ops_inv: torch.Tensor,
    anchor_index: torch.Tensor,
    *,
    reference_anchor_frac: torch.Tensor | None = None,
    reference_free_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    anchors = project_full_to_anchors(
        z,
        ops,
        ops_inv,
        anchor_index,
        reference_anchor_frac=reference_anchor_frac,
        reference_free_mask=reference_free_mask,
    )
    site_ids = _site_ids_from_anchor_index(anchor_index)
    return expand_anchors_to_full(anchors, ops, site_ids)


def project_full_to_wyckoff_ops_with_payload(
    z: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    *,
    q_ref: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
) -> torch.Tensor:
    z_payload_raw = _maybe_to_payload_frame(z, payload)
    z_payload = _maybe_align_payload_local_chart(z_payload_raw, payload)
    chart = _get_wyckoff_dof_chart(payload)
    projection = project_payload_to_wyckoff_dof_chart(
        z_payload,
        chart,
        q_ref=q_ref,
        lambda_q=float(lambda_q),
    )
    z_proj_payload_raw = _maybe_unalign_payload_local_chart(projection["z_proj"], payload)
    return _maybe_from_payload_frame(z_proj_payload_raw, payload)


def c_w_dof_chart_payload_frame(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    *,
    q_ref: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
) -> dict[str, Any]:
    chart = _get_wyckoff_dof_chart(payload)
    loss, q_star, z_proj, details = local_wyckoff_dof_chart_loss(
        z_payload,
        chart,
        q_ref=q_ref,
        lambda_q=float(lambda_q),
    )
    details["loss"] = loss
    details["q_star"] = q_star
    details["z_proj"] = z_proj
    return details


def c_w_ops_payload_frame(
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    *,
    q_ref: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
) -> torch.Tensor:
    return c_w_dof_chart_payload_frame(
        z_payload,
        payload,
        q_ref=q_ref,
        lambda_q=float(lambda_q),
    )["loss"]


def c_w_ops(
    z: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    *,
    q_ref: torch.Tensor | np.ndarray | None = None,
    lambda_q: float = 1.0e-6,
) -> torch.Tensor:
    z_payload_raw = _maybe_to_payload_frame(z, payload)
    z_payload = _maybe_align_payload_local_chart(z_payload_raw, payload)
    return c_w_ops_payload_frame(
        z_payload,
        payload,
        q_ref=q_ref,
        lambda_q=float(lambda_q),
    )


def coordinate_score_from_model_output(
    *,
    model,
    preds_v: torch.Tensor,
    v_t: torch.Tensor,
    tau: torch.Tensor,
    node_index: torch.Tensor,
    mode: str = "direct",
) -> torch.Tensor:
    """Convert model output into the coordinate wrapped-normal score.

    Current default assumption is that the KLDMPLUS coordinate branch already
    provides the coordinate score target after the repo's existing
    `sigma_norm_factor` normalization. Other modes can be added once the exact
    training target convention is pinned down empirically.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "direct":
        sigma_norm = model.tdm.sigma_norm_factor(t=tau, index=node_index, ref=preds_v)
        return sigma_norm * preds_v
    raise ValueError(f"Unsupported coordinate_score_mode={mode!r}.")


def payload_expand_identity_rmse(payload: DiffCSPPPSymmetryPayload) -> float:
    device = torch.device("cpu")
    dtype = torch.float32
    ops = torch.as_tensor(payload.wyckoff_ops, device=device, dtype=dtype)
    anchor_frac = torch.as_tensor(np.asarray(payload.anchor_frac_coords, dtype=float), device=device, dtype=dtype)
    anchor_index = torch.searchsorted(
        torch.unique(torch.as_tensor(payload.anchor_index, dtype=torch.long), sorted=True),
        torch.as_tensor(payload.anchor_index, dtype=torch.long),
    )
    recon = expand_anchors_to_full(anchor_frac, ops, anchor_index)
    target = torch.as_tensor(np.asarray(payload.expanded_frac_coords, dtype=float), device=device, dtype=dtype)
    return float(torus_rmse(recon, target).item())


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
    preds = model.score_network(
        t=t_graph,
        pos=f,
        v=v,
        h=atom_types,
        l=l.unsqueeze(0) if l.ndim == 1 else l,
        node_index=node_index,
        edge_node_index=edge_index,
    )
    preds_v = preds["v"]
    tau = model.tdm.T * t_nodes
    mu_r = model.tdm.wrapped_gaussian_mu_r_t(tau, v)
    sigma_r = model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau), f)
    s_mu = coordinate_score_from_model_output(
        model=model,
        preds_v=preds_v,
        v_t=v,
        tau=tau,
        node_index=node_index,
        mode=coordinate_score_mode,
    )
    score_part = sigma_r.square() * s_mu
    if str(variant).strip().lower() == "minus":
        f0 = f - mu_r - score_part
    elif str(variant).strip().lower() == "plus":
        f0 = f - mu_r + score_part
    else:
        raise ValueError(f"Unsupported variant={variant!r}.")
    return wrap01(f0)


def kldm_ppr_noise_chart(
    *,
    model,
    f_t: torch.Tensor,
    v_t: torch.Tensor,
    xi_r: torch.Tensor,
    xi_v: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau = model.tdm.T * t_nodes
    sigma_v = model.tdm.match_dims(model.tdm.vel_scale * model.tdm.gaussian_velocity_sigma(tau), xi_v)
    sigma_r = model.tdm.match_dims(model.tdm.wrapped_gaussian_sigma_r_t(tau), xi_r)
    v_var = center_velocity(v_t + sigma_v * xi_v, node_index)
    mu_old = model.tdm.wrapped_gaussian_mu_r_t(tau, v_t)
    mu_new = model.tdm.wrapped_gaussian_mu_r_t(tau, v_var)
    f_var = wrap01(f_t + (mu_new - mu_old) + sigma_r * xi_r)
    return f_var, v_var


def kldm_renoise_from_f0(
    *,
    model,
    f0_star: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.tdm.sample_noisy_state(t=t_nodes, f0=f0_star, index=node_index)


def ppr_project_step_ops(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    config: Algorithm19Config,
    symmetry_context: Algorithm19SymmetryContext | None = None,
) -> Algorithm19ProjectResult:
    if symmetry_context is None:
        symmetry_context = make_algorithm19_symmetry_context(
            payload,
            device=state.f.device,
            dtype=state.f.dtype,
        )
    xi_r = torch.zeros_like(state.f, requires_grad=True)
    xi_v = torch.zeros_like(state.v, requires_grad=True)
    optimizer = torch.optim.Adam([xi_r, xi_v], lr=float(config.lr))
    logs: list[dict[str, Any]] = []
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    old_requires_grad = [bool(p.requires_grad) for p in params]
    was_training = bool(model.training)
    model.eval()
    for p in params:
        p.requires_grad_(False)

    try:
        for step in range(int(config.proj_steps)):
            optimizer.zero_grad(set_to_none=True)
            f_var, v_var = kldm_ppr_noise_chart(
                model=model,
                f_t=state.f,
                v_t=state.v,
                xi_r=xi_r,
                xi_v=xi_v,
                t_nodes=state.t_nodes,
                node_index=state.node_index,
            )
            f0_hat = kldm_clean_fractional_denoiser_Df(
                model=model,
                f=f_var,
                v=v_var,
                l=state.l,
                atom_types=state.atom_types,
                t_graph=state.t_graph,
                t_nodes=state.t_nodes,
                node_index=state.node_index,
                edge_index=state.edge_node_index,
                variant=config.denoiser_variant,
                coordinate_score_mode=config.coordinate_score_mode,
            )
            z_payload = map_model_to_payload_reference_chart(f0_hat, symmetry_context.payload)
            chart_details = c_w_dof_chart_payload_frame(
                z_payload,
                symmetry_context.payload,
                q_ref=symmetry_context.q_ref,
                lambda_q=float(config.lambda_q),
            )
            c = chart_details["loss"]
            prox = xi_r.square().mean() + xi_v.square().mean()
            loss = c + float(config.lambda_noise) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_([xi_r, xi_v], max_norm=float(config.grad_clip))
            optimizer.step()
            logs.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().item()),
                    "c": float(c.detach().item()),
                    "prox": float(prox.detach().item()),
                    "xi_r_norm": float(torch.sqrt(xi_r.detach().square().mean()).item()),
                    "xi_v_norm": float(torch.sqrt(xi_v.detach().square().mean()).item()),
                    "c_dof": float(c.detach().item()),
                    "q_step_norm": float(torch.linalg.norm(chart_details["delta_q"]).detach().item()),
                    "projection_move_payload": float(torus_rmse(z_payload, chart_details["z_proj"]).detach().item()),
                    "projection_move_model": float(
                        torus_rmse(
                            f0_hat,
                            map_payload_reference_chart_to_model_frame(chart_details["z_proj"], symmetry_context.payload),
                        ).detach().item()
                    ),
                    "per_site_c": list(chart_details["per_site_c"]),
                    "per_site_q_step": list(chart_details["per_site_q_step"]),
                    "velocity_mean_norm": graph_mean_norm(v_var.detach(), state.node_index),
                }
            )

        with torch.no_grad():
            f_star, v_star = kldm_ppr_noise_chart(
                model=model,
                f_t=state.f,
                v_t=state.v,
                xi_r=xi_r,
                xi_v=xi_v,
                t_nodes=state.t_nodes,
                node_index=state.node_index,
            )
            f0_star = kldm_clean_fractional_denoiser_Df(
                model=model,
                f=f_star,
                v=v_star,
                l=state.l,
                atom_types=state.atom_types,
                t_graph=state.t_graph,
                t_nodes=state.t_nodes,
                node_index=state.node_index,
                edge_index=state.edge_node_index,
                variant=config.denoiser_variant,
                coordinate_score_mode=config.coordinate_score_mode,
            )
            z_payload_star = map_model_to_payload_reference_chart(f0_star, symmetry_context.payload)
            final_chart_details = c_w_dof_chart_payload_frame(
                z_payload_star,
                symmetry_context.payload,
                q_ref=symmetry_context.q_ref,
                lambda_q=float(config.lambda_q),
            )
    finally:
        for p, req in zip(params, old_requires_grad):
            p.requires_grad_(req)
        if was_training:
            model.train()

    return Algorithm19ProjectResult(
        f_star=f_star.detach().clone(),
        v_star=v_star.detach().clone(),
        f0_star=f0_star.detach().clone(),
        q_star=final_chart_details["q_star"].detach().clone(),
        z_proj_payload=final_chart_details["z_proj"].detach().clone(),
        logs=tuple(logs),
    )


def ppr_kernel_ops(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    config: Algorithm19Config,
) -> Algorithm19KernelResult:
    symmetry_context = make_algorithm19_symmetry_context(
        payload,
        device=state.f.device,
        dtype=state.f.dtype,
    )
    current = Algorithm19State(
        f=state.f.detach().clone(),
        v=state.v.detach().clone(),
        l=state.l.detach().clone(),
        atom_types=state.atom_types.detach().clone(),
        node_index=state.node_index.detach().clone(),
        edge_node_index=state.edge_node_index.detach().clone(),
        t_graph=state.t_graph.detach().clone(),
        t_nodes=state.t_nodes.detach().clone(),
    )
    all_logs: list[dict[str, Any]] = []

    for repeat_idx in range(int(config.M)):
        clean_current = kldm_clean_fractional_denoiser_Df(
            model=model,
            f=current.f,
            v=current.v,
            l=current.l,
            atom_types=current.atom_types,
            t_graph=current.t_graph,
            t_nodes=current.t_nodes,
            node_index=current.node_index,
            edge_index=current.edge_node_index,
            variant=config.denoiser_variant,
            coordinate_score_mode=config.coordinate_score_mode,
        )
        symmetry_context = replace(
            symmetry_context,
            q_ref=initialize_runtime_q_ref_from_model_frame(
                clean_current,
                payload,
                q_init=symmetry_context.q_ref,
                lambda_q=float(config.lambda_q),
            ),
        )
        q_ref_before = symmetry_context.q_ref.detach().clone()
        project = ppr_project_step_ops(
            state=current,
            payload=payload,
            model=model,
            config=config,
            symmetry_context=symmetry_context,
        )
        soft_anchor_constraint = float(
            c_w_ops(
                project.f0_star,
                payload,
                q_ref=symmetry_context.q_ref,
                lambda_q=float(config.lambda_q),
            ).detach().item()
        )
        soft_anchor_feasible = bool(soft_anchor_constraint < float(config.soft_anchor_tol))
        projection_move_model = float(torus_rmse(project.f0_star, map_payload_reference_chart_to_model_frame(project.z_proj_payload, payload)).detach().item()) if project.z_proj_payload is not None else float("inf")
        chart_branch_status = "local" if projection_move_model < float(config.local_projection_tol) else "branch_jump"
        if str(config.anchor_mode).strip().lower() == "soft":
            f0_anchor = project.f0_star
        elif str(config.anchor_mode).strip().lower() == "hard":
            f0_anchor = project_full_to_wyckoff_ops_with_payload(
                project.f0_star,
                payload,
                q_ref=symmetry_context.q_ref,
                lambda_q=float(config.lambda_q),
            )
        else:
            raise ValueError(f"Unsupported anchor_mode={config.anchor_mode!r}.")

        should_update_q_ref = bool(
            project.q_star is not None
            and soft_anchor_feasible
            and projection_move_model < float(config.local_projection_tol)
        )
        if should_update_q_ref and project.q_star is not None:
            symmetry_context = replace(symmetry_context, q_ref=project.q_star.detach().clone())

        f_new, v_new, epsilon_v, epsilon_r, r_t = kldm_renoise_from_f0(
            model=model,
            f0_star=f0_anchor,
            t_nodes=current.t_nodes,
            node_index=current.node_index,
        )
        current = replace(current, f=f_new.detach().clone(), v=v_new.detach().clone(), l=current.l.detach().clone())
        all_logs.append(
            {
                "repeat": int(repeat_idx),
                "project_logs": list(project.logs),
                "velocity_mean_norm": graph_mean_norm(v_new, current.node_index),
                "lattice_changed_norm": 0.0,
                "anchor_mode": str(config.anchor_mode),
                "c_after_anchor": float(
                    c_w_ops(
                        f0_anchor,
                        payload,
                        q_ref=symmetry_context.q_ref,
                        lambda_q=float(config.lambda_q),
                    ).detach().item()
                ),
                "c_anchor_soft": soft_anchor_constraint,
                "soft_anchor_feasible": soft_anchor_feasible,
                "soft_anchor_tol": float(config.soft_anchor_tol),
                "q_step_norm": float(torch.linalg.norm(project.q_star - q_ref_before).detach().item()) if project.q_star is not None else 0.0,
                "projection_move_model": projection_move_model,
                "projection_move_payload": (
                    float(torus_rmse(map_model_to_payload_reference_chart(project.f0_star, payload), project.z_proj_payload).detach().item())
                    if project.z_proj_payload is not None else float("inf")
                ),
                "chart_branch_status": chart_branch_status,
                "q_ref_updated": should_update_q_ref,
                "ppr_faithfulness": (
                    "soft_ppr_feasible" if (soft_anchor_feasible and chart_branch_status == "local")
                    else "soft-PPR-not-yet-faithful"
                ),
                "epsilon_v_rms": float(torch.sqrt(epsilon_v.square().mean()).detach().item()),
                "epsilon_r_rms": float(torch.sqrt(epsilon_r.square().mean()).detach().item()),
                "r_t_rms": float(torch.sqrt(r_t.square().mean()).detach().item()),
            }
        )

    return Algorithm19KernelResult(state=current, logs=tuple(all_logs))


__all__ = [
    "ALGORITHM19_MODE",
    "ALGORITHM19_SHORT_NAME",
    "ALGORITHM19_DESCRIPTION",
    "ALGORITHM19_RELATION_TO_PPR",
    "Algorithm19State",
    "Algorithm19Config",
    "Algorithm19ProjectResult",
    "Algorithm19KernelResult",
    "Algorithm19SymmetryContext",
    "wrap01",
    "wrapdiff",
    "torus_mse",
    "torus_rmse",
    "torus_sin_mse",
    "center_velocity",
    "graph_mean_norm",
    "build_oracle_diffcsppp_payload_from_structure",
    "expand_anchors_to_full",
    "project_full_to_anchors",
    "project_full_to_wyckoff_ops",
    "project_full_to_wyckoff_ops_with_payload",
    "map_model_to_payload_frame_raw",
    "map_model_to_payload_reference_chart",
    "map_model_to_payload_frame",
    "map_payload_to_model_frame_raw",
    "map_payload_reference_chart_to_model_frame",
    "map_payload_to_model_frame",
    "make_algorithm19_symmetry_context",
    "c_w_dof_chart_payload_frame",
    "c_w_ops_payload_frame",
    "c_w_ops",
    "payload_expand_identity_rmse",
    "kldm_clean_fractional_denoiser_Df",
    "kldm_ppr_noise_chart",
    "kldm_renoise_from_f0",
    "ppr_project_step_ops",
    "ppr_kernel_ops",
]
