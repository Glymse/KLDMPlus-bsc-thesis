from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import math
import os
import re
import sys
import gc
from collections.abc import Mapping

import numpy as np
import torch

from kldmPlus.symmetry.diffcsppp_backend import (
    DiffCSPPPSymmetryPayload,
    WyckoffDOFChart,
    align_expanded_frac_to_reference_chart,
    align_expanded_frac_to_reference_chart_orbit_aware,
    attach_payload_reference_chart,
    build_diffcsppp_symmetry_payload,
    build_wyckoff_dof_chart,
    payload_site_slices,
)
from kldmPlus.symmetry.wyckoff_templates import WyckoffTemplate, expand_wyckoff_template_torch


WYCKOFF_DOF_CHART_CACHE_VERSION = 1


ALGORITHM21_MODE = "clean_cf_ppr_kldm"
ALGORITHM21_SHORT_NAME = "Algorithm21-Clean-CF-PPR-KLDM"
ALGORITHM21_DESCRIPTION = (
    "Clean-space KLDM-PPR with q-only Wyckoff projection plus optional "
    "CrystalFormer coordinate likelihood, followed by soft clean anchoring "
    "and native KLDM/TDM renoising."
)


def _algo21_log_path() -> Path | None:
    raw = str(os.environ.get("KLDM_ALGO21_LOG_PATH", "")).strip()
    if not raw:
        return None
    return Path(raw)


def _algo21_log(message: str) -> None:
    text = str(message)
    try:
        print(text)
    except (BrokenPipeError, OSError):
        pass
    path = _algo21_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except Exception:
        pass


def _algo21_trace(name: str, **kwargs: Any) -> None:
    if _algo21_log_path() is None:
        return
    if kwargs:
        suffix = " ".join(f"{key}={value}" for key, value in kwargs.items())
        _algo21_log(f"[trace] {name} start {suffix}")
    else:
        _algo21_log(f"[trace] {name} start")


def _truthy_env(name: str, default: str = "") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Algorithm21Config:
    beta: float = 0.1
    alpha: float = 0.25
    q_opt_steps: int = 50
    q_lr: float = 1.0e-2
    grad_clip: float = 10.0
    q_init_mode: str = "oracle_structure"
    denoiser_variant: str = "minus"
    coordinate_score_mode: str = "direct"
    post_renoise_acceptance: bool = True
    sigma_proj_floor: float = 5.0e-2
    finite_diff_eps: float = 1.0e-3
    t_guide: float = 0.5
    projection_times: tuple[float, ...] = (0.5, 0.4, 0.3, 0.2, 0.1)
    projection_time_tol: float = 2.0e-2
    cf_use_delta: bool = True
    cf_delta_normalize: bool = False
    debug_prints: bool = False
    cf_grad_max_dims: int | None = None
    cf_value_only_after_renoise: bool = False
    cf_sample_k: int = 64
    cf_top_k: int = 3
    cf_rank_eps_abs: float = 1.0e-3
    cf_top_p: float = 1.0
    cf_temperature: float = 1.0
    cf_sampler_seed: int = 0


@dataclass(frozen=True)
class Algorithm21QFitResult:
    q_star: torch.Tensor
    z_proj_payload: torch.Tensor
    near_loss: float
    witness_sin: float
    witness_rmse_payload: float
    cf_nll: float
    cf_nll_start: float
    cf_delta: float
    score_total: float
    logs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Algorithm21StepResult:
    state_before: Algorithm19State
    state_candidate: Algorithm19State
    state_after: Algorithm19State
    accepted: bool
    f0_hat_before: torch.Tensor
    f0_hat_after: torch.Tensor
    f0_hard: torch.Tensor
    f0_star: torch.Tensor
    fit_before: Algorithm21QFitResult
    fit_after: Algorithm21QFitResult
    logs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Algorithm21LocalRerankResult:
    q_center: torch.Tensor
    q_best: torch.Tensor
    z_center_payload: torch.Tensor
    z_best_payload: torch.Tensor
    witness_center: float
    witness_best: float
    cf_nll_center: float
    cf_nll_best: float
    candidate_count: int
    kept_count: int
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Algorithm21RankedQCandidate:
    rank: int
    source_index: int
    q: torch.Tensor
    z_payload: torch.Tensor
    witness_sin: float
    witness_rmse_payload: float
    cf_nll: float
    geometry_kept: bool


@dataclass(frozen=True)
class Algorithm21BranchResult:
    candidate: Algorithm21RankedQCandidate
    step_result: Algorithm21StepResult


def predict_clean_f0(*, state: Algorithm19State, model, denoiser_variant: str = "minus", coordinate_score_mode: str = "direct") -> torch.Tensor:
    return kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state.f,
        v=state.v,
        l=state.l,
        atom_types=state.atom_types,
        t_graph=state.t_graph,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
        edge_index=state.edge_node_index,
        variant=denoiser_variant,
        coordinate_score_mode=coordinate_score_mode,
    )


def model_to_payload(*, f_model: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
    return map_model_to_payload_reference_chart(f_model, payload)


def payload_to_model(*, z_payload: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
    return map_payload_reference_chart_to_model_frame(z_payload, payload)


def expand_q(*, payload: DiffCSPPPSymmetryPayload, q: torch.Tensor) -> torch.Tensor:
    chart = _get_wyckoff_dof_chart(payload)
    return chart.expand_q(q, device=q.device, dtype=q.dtype)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_crystalformer_importable() -> Path:
    root = _repo_root()
    cf_root = root / "src" / "CrystalFormer" / "CrystalFormer-main"
    cf_root_str = str(cf_root)
    if sys.path[:1] != [cf_root_str]:
        try:
            sys.path.remove(cf_root_str)
        except ValueError:
            pass
        sys.path.insert(0, cf_root_str)

    # CrystalFormer is easy to import from the wrong environment if a different
    # package is already present in sys.modules. Force future imports to resolve
    # from the local source tree we vendor in this repo.
    existing = sys.modules.get("crystalformer")
    existing_file = Path(getattr(existing, "__file__", "")).resolve() if existing is not None else None
    if existing_file is not None and cf_root not in existing_file.parents:
        for name in list(sys.modules):
            if name == "crystalformer" or name.startswith("crystalformer."):
                del sys.modules[name]
    return cf_root


def crystalformer_letter_to_number(letter: str) -> int:
    letter = str(letter).strip()
    if len(letter) > 1:
        suffix = "".join(ch for ch in letter if ch.isalpha())
        if suffix:
            letter = suffix
    if "a" <= letter <= "z":
        return ord(letter) - ord("a") + 1
    if letter == "A":
        return 27
    raise ValueError(f"Unsupported CrystalFormer Wyckoff letter {letter!r}.")


def _sort_reduced_sites_for_crystalformer(
    w_numbers: np.ndarray,
    atomic_numbers: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = _sort_reduced_sites_index(w_numbers, xyz)
    return w_numbers[idx], atomic_numbers[idx], (xyz - np.floor(xyz))[idx]


def _sort_reduced_sites_index(
    w_numbers: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    w_temp = np.where(w_numbers > 0, w_numbers, 9999)
    xyz_mod = xyz - np.floor(xyz)
    return np.lexsort((xyz_mod[:, 2], xyz_mod[:, 1], xyz_mod[:, 0], w_temp))


def _build_anchor_xyz_from_q(
    *,
    payload: DiffCSPPPSymmetryPayload,
    q: np.ndarray,
) -> np.ndarray:
    chart = _get_wyckoff_dof_chart(payload)
    q = np.asarray(q, dtype=float).reshape(-1)
    anchors: list[np.ndarray] = []
    for site_idx, dof_slice in enumerate(chart.site_dof_slices):
        basis = np.asarray(chart.site_anchor_bases[site_idx], dtype=float)
        offset = np.asarray(chart.site_anchor_offsets[site_idx], dtype=float)
        q_site = q[dof_slice]
        if basis.shape[1] == 0:
            anchor = offset
        else:
            anchor = q_site.reshape(1, -1) @ basis.T + offset.reshape(1, 3)
            anchor = anchor.reshape(3)
        anchors.append(np.remainder(anchor, 1.0))
    return np.asarray(anchors, dtype=float)


def build_crystalformer_reduced_sequence(
    *,
    payload: DiffCSPPPSymmetryPayload,
    q: np.ndarray | torch.Tensor,
    lattice_feature: np.ndarray | torch.Tensor,
    n_max: int = 21,
) -> dict[str, np.ndarray]:
    q_np = np.asarray(torch.as_tensor(q).detach().cpu(), dtype=float).reshape(-1)
    xyz = _build_anchor_xyz_from_q(payload=payload, q=q_np)
    w = np.asarray([crystalformer_letter_to_number(letter) for letter in payload.wyckoff_letters], dtype=int)
    a = np.asarray(payload.anchor_atomic_numbers, dtype=int)
    w, a, xyz = _sort_reduced_sites_for_crystalformer(w, a, xyz)

    num_sites = int(len(w))
    if num_sites > int(n_max):
        raise ValueError(f"Reduced site count {num_sites} exceeds CrystalFormer n_max={n_max}.")

    w_pad = np.concatenate([w, np.zeros((n_max - num_sites,), dtype=int)], axis=0)
    a_pad = np.concatenate([a, np.zeros((n_max - num_sites,), dtype=int)], axis=0)
    xyz_pad = np.concatenate([xyz, np.full((n_max - num_sites, 3), 1.0e10, dtype=float)], axis=0)
    l = np.asarray(torch.as_tensor(lattice_feature).detach().cpu(), dtype=float).reshape(-1)
    if l.shape[0] != 6:
        raise ValueError(f"Expected lattice feature of length 6, got {l.shape[0]}.")
    return {
        "G": np.asarray(int(payload.spacegroup), dtype=int),
        "L": l,
        "XYZ": xyz_pad,
        "A": a_pad,
        "W": w_pad,
        "num_sites": np.asarray(num_sites, dtype=int),
    }


def crystalformer_reduced_sequence_debug(
    *,
    payload: DiffCSPPPSymmetryPayload,
    q: np.ndarray | torch.Tensor,
    lattice_feature: np.ndarray | torch.Tensor,
    n_max: int = 21,
) -> dict[str, Any]:
    q_np = np.asarray(torch.as_tensor(q).detach().cpu(), dtype=float).reshape(-1)
    xyz_unsorted = _build_anchor_xyz_from_q(payload=payload, q=q_np)
    w_letters = np.asarray([str(letter) for letter in payload.wyckoff_letters], dtype=object)
    w_numbers = np.asarray([crystalformer_letter_to_number(letter) for letter in payload.wyckoff_letters], dtype=int)
    atomic_numbers = np.asarray(payload.anchor_atomic_numbers, dtype=int)

    w_temp = np.where(w_numbers > 0, w_numbers, 9999)
    xyz_mod = xyz_unsorted - np.floor(xyz_unsorted)
    sort_idx = np.lexsort((xyz_mod[:, 2], xyz_mod[:, 1], xyz_mod[:, 0], w_temp))

    seq = build_crystalformer_reduced_sequence(
        payload=payload,
        q=q_np,
        lattice_feature=lattice_feature,
        n_max=n_max,
    )
    cf_expanded = expand_crystalformer_reduced_sequence(
        space_group=int(seq["G"]),
        W=seq["W"],
        XYZ=seq["XYZ"],
    )
    payload_expanded = _get_wyckoff_dof_chart(payload).expand_q(
        torch.as_tensor(q_np, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).detach().cpu().numpy()
    aligned = align_expanded_frac_to_reference_chart(
        payload,
        cf_expanded,
    )

    unsorted_rows = []
    sorted_rows = []
    for site_idx in range(len(w_numbers)):
        unsorted_rows.append({
            "payload_site_index": int(site_idx),
            "wyckoff_label": str(w_letters[site_idx]),
            "wyckoff_number": int(w_numbers[site_idx]),
            "atomic_number": int(atomic_numbers[site_idx]),
            "anchor_xyz": xyz_unsorted[site_idx].tolist(),
        })
    for sort_pos, site_idx in enumerate(sort_idx.tolist()):
        sorted_rows.append({
            "sorted_cf_index": int(sort_pos),
            "payload_site_index": int(site_idx),
            "wyckoff_label": str(w_letters[site_idx]),
            "wyckoff_number": int(w_numbers[site_idx]),
            "atomic_number": int(atomic_numbers[site_idx]),
            "anchor_xyz": xyz_mod[site_idx].tolist(),
        })

    return {
        "unsorted_rows": tuple(unsorted_rows),
        "sorted_rows": tuple(sorted_rows),
        "sort_index": np.asarray(sort_idx, dtype=int),
        "seq": seq,
        "payload_expanded": payload_expanded,
        "cf_expanded": cf_expanded,
        "cf_aligned_to_payload": np.asarray(aligned["aligned_frac_coords"], dtype=float),
        "alignment_rmse": float(aligned["rmse"]),
        "alignment_reference_order": np.asarray(aligned["reference_order"], dtype=int),
    }


def crystalformer_site_representative_search(
    *,
    payload: DiffCSPPPSymmetryPayload,
    q: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    _ensure_crystalformer_importable()
    from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms, wmax_table

    q_np = np.asarray(torch.as_tensor(q).detach().cpu(), dtype=float).reshape(-1)
    chart = _get_wyckoff_dof_chart(payload)
    anchor_xyz = _build_anchor_xyz_from_q(payload=payload, q=q_np)
    payload_expanded = chart.expand_q(
        torch.as_tensor(q_np, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).detach().cpu().numpy()
    site_slices = payload_site_slices(payload)
    g = int(payload.spacegroup)

    rows: list[dict[str, Any]] = []
    candidate_tables: list[dict[str, Any]] = []
    for site_idx, (start, stop) in enumerate(site_slices):
        w_label = str(payload.wyckoff_letters[site_idx])
        w_num = int(crystalformer_letter_to_number(w_label))
        x0 = np.asarray(anchor_xyz[site_idx], dtype=float)
        payload_site = np.asarray(payload_expanded[start:stop], dtype=float)

        w_max = int(wmax_table[g - 1])
        m_max = int(mult_table[g - 1, w_max])
        general_ops = np.asarray(symops[g - 1, w_max, :m_max], dtype=float)
        affine = np.concatenate([x0, np.ones(1, dtype=float)], axis=0)
        candidates = np.remainder((general_ops @ affine).reshape(-1, 3), 1.0)

        candidate_rows: list[dict[str, Any]] = []
        best_rmse = float("inf")
        best_idx = -1
        best_candidate = x0.copy()
        best_cf_site = None
        for cand_idx, cand in enumerate(candidates.tolist()):
            cand_np = np.asarray(cand, dtype=float)
            cf_site = np.asarray(symmetrize_atoms(g, w_num, jnp_or_np(cand_np)), dtype=float)
            aligned = _site_orbit_alignment(cf_site, payload_site)
            rmse = float(aligned["rmse"])
            candidate_rows.append({
                "site_index": int(site_idx),
                "candidate_index": int(cand_idx),
                "candidate_xyz": cand_np.tolist(),
                "site_rmse": rmse,
            })
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = int(cand_idx)
                best_candidate = cand_np
                best_cf_site = np.asarray(aligned["aligned"], dtype=float)

        rows.append({
            "site_index": int(site_idx),
            "wyckoff_label": w_label,
            "wyckoff_number": int(w_num),
            "atomic_number": int(payload.anchor_atomic_numbers[site_idx]),
            "payload_anchor_xyz": x0.tolist(),
            "best_candidate_index": int(best_idx),
            "best_candidate_xyz": best_candidate.tolist(),
            "site_alignment_rmse": float(best_rmse),
            "improves_anchor": bool(best_rmse < 1e-4),
        })
        candidate_tables.append({
            "site_index": int(site_idx),
            "payload_site": payload_site,
            "best_cf_site": best_cf_site,
            "candidates": tuple(candidate_rows),
        })

    return {
        "rows": tuple(rows),
        "candidate_tables": tuple(candidate_tables),
    }


def crystalformer_payload_order_assembly_debug(
    *,
    payload: DiffCSPPPSymmetryPayload,
    q: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    dbg = crystalformer_site_representative_search(payload=payload, q=q)
    chart = _get_wyckoff_dof_chart(payload)
    q_np = np.asarray(torch.as_tensor(q).detach().cpu(), dtype=float).reshape(-1)
    payload_expanded = chart.expand_q(
        torch.as_tensor(q_np, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).detach().cpu().numpy()

    payload_blocks: list[np.ndarray] = []
    cf_blocks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for item in dbg["candidate_tables"]:
        site_idx = int(item["site_index"])
        payload_site = np.asarray(item["payload_site"], dtype=float)
        best_cf_site = np.asarray(item["best_cf_site"], dtype=float)
        payload_blocks.append(payload_site)
        cf_blocks.append(best_cf_site)
        delta = _signed_wrap_numpy(best_cf_site - payload_site)
        rows.append({
            "site_index": site_idx,
            "payload_rows": int(payload_site.shape[0]),
            "site_rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        })

    payload_concat = np.concatenate(payload_blocks, axis=0) if payload_blocks else np.zeros((0, 3), dtype=float)
    cf_concat = np.concatenate(cf_blocks, axis=0) if cf_blocks else np.zeros((0, 3), dtype=float)
    delta_full = _signed_wrap_numpy(cf_concat - payload_concat)
    rmse_full = float(np.sqrt(np.mean(delta_full * delta_full))) if delta_full.size else 0.0

    global_aligned = align_expanded_frac_to_reference_chart(payload, cf_concat)
    return {
        "site_rows": tuple(rows),
        "payload_concat": payload_concat,
        "cf_concat_payload_order": cf_concat,
        "payload_order_rmse": rmse_full,
        "global_aligned_rmse": float(global_aligned["rmse"]),
        "global_aligned": np.asarray(global_aligned["aligned_frac_coords"], dtype=float),
        "payload_expanded": payload_expanded,
    }


def expand_crystalformer_reduced_sequence(
    *,
    space_group: int,
    W: np.ndarray | torch.Tensor,
    XYZ: np.ndarray | torch.Tensor,
) -> np.ndarray:
    _ensure_crystalformer_importable()
    from crystalformer.src.wyckoff import symmetrize_atoms

    W_np = np.asarray(torch.as_tensor(W).detach().cpu(), dtype=int).reshape(-1)
    XYZ_np = np.asarray(torch.as_tensor(XYZ).detach().cpu(), dtype=float).reshape(-1, 3)
    valid = W_np > 0
    coords: list[np.ndarray] = []
    for w_i, x_i in zip(W_np[valid], XYZ_np[valid], strict=False):
        xs = np.asarray(symmetrize_atoms(int(space_group), int(w_i), jnp_or_np(x_i)), dtype=float)
        coords.append(xs)
    if not coords:
        return np.zeros((0, 3), dtype=float)
    out = np.concatenate(coords, axis=0)
    return np.remainder(out, 1.0)


def jnp_or_np(x: np.ndarray):
    try:
        import jax.numpy as jnp

        return jnp.asarray(x)
    except Exception:  # pragma: no cover
        return np.asarray(x)


def _signed_wrap_numpy(value: np.ndarray) -> np.ndarray:
    return np.remainder(np.asarray(value, dtype=float) + 0.5, 1.0) - 0.5


def _torus_pairwise_distance_sq_numpy(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = _signed_wrap_numpy(np.asarray(source, dtype=float)[:, None, :] - np.asarray(target, dtype=float)[None, :, :])
    return np.sum(delta * delta, axis=-1)


def _match_cost_matrix_numpy(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # pragma: no cover
        linear_sum_assignment = None

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return np.asarray(row_idx, dtype=int), np.asarray(col_idx, dtype=int)

    remaining_rows = list(range(cost_matrix.shape[0]))
    remaining_cols = list(range(cost_matrix.shape[1]))
    chosen_rows: list[int] = []
    chosen_cols: list[int] = []
    while remaining_rows:
        submatrix = cost_matrix[np.ix_(remaining_rows, remaining_cols)]
        flat_idx = int(np.argmin(submatrix))
        n_cols = submatrix.shape[1]
        row_pos = flat_idx // n_cols
        col_pos = flat_idx % n_cols
        chosen_rows.append(remaining_rows.pop(row_pos))
        chosen_cols.append(remaining_cols.pop(col_pos))
    order = np.argsort(np.asarray(chosen_rows, dtype=int))
    return np.asarray(chosen_rows, dtype=int)[order], np.asarray(chosen_cols, dtype=int)[order]


def _site_orbit_alignment(cf_site: np.ndarray, payload_site: np.ndarray) -> dict[str, Any]:
    cf_site = np.remainder(np.asarray(cf_site, dtype=float), 1.0)
    payload_site = np.remainder(np.asarray(payload_site, dtype=float), 1.0)
    if cf_site.shape != payload_site.shape:
        raise ValueError(f"Site orbit shapes differ: cf={cf_site.shape}, payload={payload_site.shape}.")
    if cf_site.size == 0:
        return {"aligned": cf_site.copy(), "rmse": 0.0, "order": np.zeros((0,), dtype=int)}

    best_rmse = float("inf")
    best_aligned = cf_site.copy()
    best_order = np.arange(cf_site.shape[0], dtype=int)
    for ref_idx in range(payload_site.shape[0]):
        tau = _signed_wrap_numpy(payload_site[ref_idx] - cf_site[0])
        shifted = np.remainder(cf_site + tau.reshape(1, 3), 1.0)
        cost = _torus_pairwise_distance_sq_numpy(shifted, payload_site)
        row_idx, col_idx = _match_cost_matrix_numpy(cost)
        ordered = shifted[row_idx[np.argsort(col_idx)]]
        delta = _signed_wrap_numpy(ordered - payload_site)
        rmse = float(np.sqrt(np.mean(delta * delta)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_aligned = ordered
            best_order = col_idx
    return {"aligned": best_aligned, "rmse": best_rmse, "order": np.asarray(best_order, dtype=int)}


def build_payload_from_template_q(
    *,
    template: WyckoffTemplate,
    q: np.ndarray | torch.Tensor,
    lattice_matrix: np.ndarray | torch.Tensor,
    spacegroup: int | None = None,
) -> DiffCSPPPSymmetryPayload:
    q_t = torch.as_tensor(q, dtype=torch.get_default_dtype()).reshape(-1)
    expansion = expand_wyckoff_template_torch(template=template, free_vars=q_t, wrap=True)

    anchor_index: list[int] = []
    wyckoff_ops: list[np.ndarray] = []
    wyckoff_ops_inv: list[np.ndarray] = []
    wyckoff_letters: list[str] = []
    anchor_atomic_numbers: list[int] = []
    anchor_dofs: list[int] = []
    anchor_masks: list[tuple[bool, bool, bool]] = []

    cursor = 0
    for site_idx, site in enumerate(template.site_templates):
        for op_idx in range(int(site.multiplicity)):
            affine = np.eye(4, dtype=float)
            affine[:3, :3] = np.asarray(site.rotation_matrices[op_idx], dtype=float)
            affine[:3, 3] = np.asarray(site.translation_vectors[op_idx], dtype=float)
            wyckoff_ops.append(affine)
            wyckoff_ops_inv.append(np.linalg.inv(affine))
            anchor_index.append(int(cursor))
        cursor += int(site.multiplicity)
        wyckoff_letters.append(str(site.label))
        anchor_atomic_numbers.append(int(site.atomic_number))
        anchor_dofs.append(int(site.dof))
        anchor_masks.append(tuple(bool(v) for v in site.free_coordinate_mask))

    expanded_atomic_numbers = np.asarray(expansion.atomic_numbers.detach().cpu(), dtype=int).reshape(-1)
    payload = DiffCSPPPSymmetryPayload(
        spacegroup=int(template.space_group if spacegroup is None else spacegroup),
        anchor_index=np.asarray(anchor_index, dtype=int),
        wyckoff_ops=np.asarray(wyckoff_ops, dtype=float),
        wyckoff_ops_inv=np.asarray(wyckoff_ops_inv, dtype=float),
        wyckoff_letters=tuple(wyckoff_letters),
        atom_types=expanded_atomic_numbers.copy(),
        anchor_frac_coords=np.asarray(expansion.anchor_coords.detach().cpu(), dtype=float),
        expanded_frac_coords=np.asarray(expansion.frac_coords.detach().cpu(), dtype=float),
        anchor_atomic_numbers=np.asarray(anchor_atomic_numbers, dtype=int),
        expanded_atomic_numbers=expanded_atomic_numbers,
        lattice_matrix=np.asarray(torch.as_tensor(lattice_matrix).detach().cpu(), dtype=float).reshape(3, 3),
        anchor_dofs=np.asarray(anchor_dofs, dtype=int),
        anchor_free_coordinate_masks=np.asarray(anchor_masks, dtype=bool),
        standardized_structure=None,
        debug_info={},
    )
    return payload


def species_match_reorder(
    *,
    source_frac: np.ndarray | torch.Tensor,
    source_atomic_numbers: np.ndarray | torch.Tensor,
    target_frac: np.ndarray | torch.Tensor,
    target_atomic_numbers: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    source_frac_np = np.remainder(np.asarray(torch.as_tensor(source_frac).detach().cpu(), dtype=float), 1.0)
    target_frac_np = np.remainder(np.asarray(torch.as_tensor(target_frac).detach().cpu(), dtype=float), 1.0)
    source_z = np.asarray(torch.as_tensor(source_atomic_numbers).detach().cpu(), dtype=int).reshape(-1)
    target_z = np.asarray(torch.as_tensor(target_atomic_numbers).detach().cpu(), dtype=int).reshape(-1)
    if sorted(source_z.tolist()) != sorted(target_z.tolist()):
        raise ValueError("Species multiset mismatch in species_match_reorder.")

    aligned = np.zeros_like(target_frac_np)
    assignment = np.empty((source_frac_np.shape[0],), dtype=int)
    total_sq = 0.0
    total_dims = 0
    for atomic_number in sorted(set(int(v) for v in source_z.tolist())):
        src_idx = np.where(source_z == atomic_number)[0]
        dst_idx = np.where(target_z == atomic_number)[0]
        cost = _torus_pairwise_distance_sq_numpy(source_frac_np[src_idx], target_frac_np[dst_idx])
        row_idx, col_idx = _match_cost_matrix_numpy(cost)
        ordered_src = source_frac_np[src_idx[row_idx]]
        ordered_dst_idx = dst_idx[col_idx]
        aligned[ordered_dst_idx] = ordered_src
        assignment[src_idx[row_idx]] = ordered_dst_idx
        delta = _signed_wrap_numpy(ordered_src - target_frac_np[ordered_dst_idx])
        total_sq += float(np.sum(delta * delta))
        total_dims += int(delta.size)

    rmse = 0.0 if total_dims == 0 else float(np.sqrt(total_sq / total_dims))
    return {
        "aligned_source_in_target_order": aligned,
        "assignment": assignment,
        "rmse": rmse,
    }


def _normalize_crystalformer_param_aliases(tree: Any) -> Any:
    """Handle small checkpoint/source naming drifts across CrystalFormer commits."""
    if isinstance(tree, Mapping):
        out: dict[Any, Any] = {}
        for key, value in tree.items():
            out[key] = _normalize_crystalformer_param_aliases(value)

        # Base transformer compatibility: some checkpoints/source snapshots differ
        # only in the unconditional composition embedding name.
        if "c_embedding_unconditional" in out and "c_embedding_uncond" not in out:
            out["c_embedding_uncond"] = out["c_embedding_unconditional"]
        if "c_embedding_uncond" in out and "c_embedding_unconditional" not in out:
            out["c_embedding_unconditional"] = out["c_embedding_uncond"]
        return out
    if isinstance(tree, (list, tuple)):
        return type(tree)(_normalize_crystalformer_param_aliases(v) for v in tree)
    return tree


def _merge_crystalformer_params(template: Any, loaded: Any) -> Any:
    """Overlay loaded checkpoint params onto a freshly initialized template tree."""
    if isinstance(template, Mapping) and isinstance(loaded, Mapping):
        out: dict[Any, Any] = {}
        all_keys = set(template.keys()) | set(loaded.keys())
        for key in all_keys:
            if key in template and key in loaded:
                out[key] = _merge_crystalformer_params(template[key], loaded[key])
            elif key in loaded:
                out[key] = loaded[key]
            else:
                out[key] = template[key]
        return out
    return loaded


@dataclass
class CrystalFormerLikelihood:
    checkpoint_path: str
    seed: int = 0
    n_max: int = 21
    atom_types: int = 119
    wyck_types: int = 28
    Nf: int = 5
    Kx: int = 16
    Kl: int = 4
    h0_size: int = 256
    transformer_layers: int = 16
    num_heads: int = 8
    key_size: int = 32
    model_size: int = 256
    embed_size: int = 256
    dropout_rate: float = 0.1
    attn_dropout: float = 0.1
    lamb_a: float = 1.0
    lamb_w: float = 1.0
    lamb_l: float = 1.0
    coordinate_only: bool = True
    debug_prints: bool = False
    _jax: Any = field(init=False, repr=False, default=None)
    _jnp: Any = field(init=False, repr=False, default=None)
    _params: Any = field(init=False, repr=False, default=None)
    _logp_fn_raw: Any = field(init=False, repr=False, default=None)
    _logp_fn: Any = field(init=False, repr=False, default=None)
    _safe_mode: bool = field(init=False, repr=False, default=False)
    _nll_cache: dict[Any, dict[str, float]] = field(init=False, repr=False, default_factory=dict)
    _transformer: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        _algo21_trace("CrystalFormerLikelihood.__post_init__", checkpoint=self.checkpoint_path)
        self._safe_mode = (
            str(os.environ.get("KLDM_ALGO21_SAFE_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
            or str(os.environ.get("JAX_DISABLE_JIT", "")).strip().lower() in {"1", "true", "yes", "on"}
        )
        self._load_runtime()

    def _load_runtime(self) -> None:
        _algo21_trace("CrystalFormerLikelihood._load_runtime", checkpoint=self.checkpoint_path)
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=ensure_importable")
        cf_root = _ensure_crystalformer_importable()
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=import_crystalformer")
        import crystalformer

        crystalformer_file = Path(getattr(crystalformer, "__file__", "")).resolve()
        if cf_root not in crystalformer_file.parents:
            raise ImportError(
                "CrystalFormer was imported from an unexpected location: "
                f"{crystalformer_file}. Expected it under {cf_root}."
            )
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=import_jax")
        import jax
        import jax.numpy as jnp
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=import_cf_modules")
        from crystalformer.src.checkpoint import find_ckpt_filename, load_data
        from crystalformer.src.loss import make_loss_fn
        from crystalformer.src.transformer import make_transformer
        from crystalformer.src.wyckoff import mult_table

        self._jax = jax
        self._jnp = jnp
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=make_prng")
        key = jax.random.PRNGKey(int(self.seed))
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=make_transformer")
        init_params, transformer = make_transformer(
            key,
            self.Nf,
            self.Kx,
            self.Kl,
            self.n_max,
            self.h0_size,
            self.transformer_layers,
            self.num_heads,
            self.key_size,
            self.model_size,
            self.embed_size,
            self.atom_types,
            self.wyck_types,
            self.dropout_rate,
            self.attn_dropout,
        )
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=make_loss_fn")
        _, logp_fn = make_loss_fn(
            self.n_max,
            self.atom_types,
            self.wyck_types,
            self.Kx,
            self.Kl,
            transformer,
            self.lamb_a,
            self.lamb_w,
            self.lamb_l,
        )
        self._logp_fn_raw = logp_fn
        self._transformer = transformer
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=find_ckpt_filename")
        ckpt_filename, _ = find_ckpt_filename(self.checkpoint_path)
        if ckpt_filename is None:
            raise FileNotFoundError(f"No CrystalFormer checkpoint found at {self.checkpoint_path!r}.")
        _algo21_log(f"[trace] CrystalFormerLikelihood._load_runtime step=load_data path={ckpt_filename}")
        ckpt = load_data(ckpt_filename)
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=normalize_params")
        ckpt_params = _normalize_crystalformer_param_aliases(ckpt["params"])
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=merge_params")
        self._params = _merge_crystalformer_params(init_params, ckpt_params)
        _algo21_log("[trace] CrystalFormerLikelihood._load_runtime step=jit_logp_fn")
        self._logp_fn = jax.jit(logp_fn, static_argnums=8)
        self._mult_table = mult_table
        if _algo21_log_path() is not None:
            _algo21_log(
                f"[crystalformer] init checkpoint={self.checkpoint_path} coordinate_only={bool(self.coordinate_only)} safe_mode={bool(self._safe_mode)}"
            )

    def _ensure_runtime_loaded(self) -> None:
        _algo21_trace("CrystalFormerLikelihood._ensure_runtime_loaded")
        if self._params is None or self._jax is None or self._jnp is None or self._logp_fn_raw is None:
            _algo21_log("[trace] CrystalFormerLikelihood._ensure_runtime_loaded step=reload")
            self._load_runtime()

    def release_runtime(self) -> None:
        _algo21_trace("CrystalFormerLikelihood.release_runtime")
        if _algo21_log_path() is not None:
            _algo21_log("[crystalformer] release_runtime start")
        try:
            if self._jax is not None:
                try:
                    _algo21_log("[trace] CrystalFormerLikelihood.release_runtime step=jax.clear_caches")
                    self._jax.clear_caches()
                except Exception:
                    pass
        finally:
            _algo21_log("[trace] CrystalFormerLikelihood.release_runtime step=drop_runtime_refs")
            self._jax = None
            self._jnp = None
            self._params = None
            self._logp_fn_raw = None
            self._logp_fn = None
            self._transformer = None
            self._mult_table = None
            _algo21_log("[trace] CrystalFormerLikelihood.release_runtime step=gc.collect")
            gc.collect()
        if _algo21_log_path() is not None:
            _algo21_log("[crystalformer] release_runtime done")

    @staticmethod
    def _formula_to_composition_vector_numpy(formula: str) -> np.ndarray:
        def expand_groups(text: str) -> str:
            pattern = r'[\(\[]([A-Za-z0-9]+)[\)\]](\d*)'
            out = str(text)
            while True:
                match = re.search(pattern, out)
                if not match:
                    break
                inner = match.group(1)
                mult = int(match.group(2)) if match.group(2) else 1
                out = out[:match.start()] + inner * mult + out[match.end():]
            return out

        element_dict = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
            "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
            "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
            "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
            "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
            "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
            "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
            "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
            "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
            "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
            "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
        }
        composition = np.zeros((119,), dtype=np.int32)
        text = expand_groups(formula)
        for symbol, count_str in re.findall(r'([A-Z][a-z]?)(\d*)', text):
            atomic_number = element_dict.get(symbol)
            if atomic_number is None:
                continue
            count = int(count_str) if count_str else 1
            composition[int(atomic_number)] += int(count)
        return composition

    @staticmethod
    def _template_to_composition_vector_numpy(
        *,
        A: np.ndarray,
        multiplicities: np.ndarray,
    ) -> np.ndarray:
        composition = np.zeros((119,), dtype=np.int32)
        a_np = np.asarray(A, dtype=int).reshape(-1)
        mult_np = np.asarray(multiplicities, dtype=int).reshape(-1)
        for atomic_number, multiplicity in zip(a_np.tolist(), mult_np.tolist(), strict=False):
            if int(atomic_number) <= 0 or int(multiplicity) <= 0:
                continue
            if int(atomic_number) >= composition.shape[0]:
                continue
            composition[int(atomic_number)] += int(multiplicity)
        nonzero = composition[composition > 0]
        if nonzero.size > 0:
            gcd = int(np.gcd.reduce(nonzero))
            if gcd > 1:
                composition //= gcd
        return composition

    def _composition_vector(
        self,
        *,
        A: np.ndarray,
        W: np.ndarray,
        G: int,
        formula: str | None,
    ) -> np.ndarray:
        _algo21_trace("CrystalFormerLikelihood._composition_vector", sg=int(G), formula=bool(formula))
        if formula:
            return self._formula_to_composition_vector_numpy(formula)
        multiplicities = np.asarray(self._mult_table[int(G) - 1, np.asarray(W, dtype=int)], dtype=int)
        return self._template_to_composition_vector_numpy(A=np.asarray(A, dtype=int), multiplicities=multiplicities)

    def nll_components_from_reduced_sequence(
        self,
        *,
        seq: Mapping[str, np.ndarray],
        formula: str | None = None,
    ) -> dict[str, float]:
        _algo21_trace(
            "CrystalFormerLikelihood.nll_components_from_reduced_sequence",
            sg=int(seq["G"]),
            num_sites=int(seq["num_sites"]),
        )
        self._ensure_runtime_loaded()
        composition = np.asarray(
            self._composition_vector(A=seq["A"], W=seq["W"], G=int(seq["G"]), formula=formula),
            dtype=np.int32,
        )
        if _algo21_log_path() is not None:
            mode = "formula" if formula else "template"
            nonzero = int(np.count_nonzero(composition))
            _algo21_log(
                f"[crystalformer] conditioning ready sg={int(seq['G'])} num_sites={int(seq['num_sites'])} mode={mode} nonzero_comp={nonzero}"
            )
        key = self._jax.random.PRNGKey(int(self.seed))
        G = self._jnp.asarray([seq["G"]], dtype=self._jnp.int32)
        L = self._jnp.asarray(seq["L"][None, :], dtype=self._jnp.float32)
        XYZ = self._jnp.asarray(seq["XYZ"][None, :, :], dtype=self._jnp.float32)
        A = self._jnp.asarray(seq["A"][None, :], dtype=self._jnp.int32)
        W = self._jnp.asarray(seq["W"][None, :], dtype=self._jnp.int32)
        comp = self._jnp.asarray(composition[None, :], dtype=self._jnp.int32)
        logp_fn = self._logp_fn_raw if self._safe_mode else self._logp_fn
        _algo21_log("[trace] CrystalFormerLikelihood.nll_components_from_reduced_sequence step=logp_fn_call")
        logp_g, logp_w, logp_xyz, logp_a, logp_l = logp_fn(
            self._params,
            key,
            comp,
            G,
            L,
            XYZ,
            A,
            W,
            False,
        )
        _algo21_log("[trace] CrystalFormerLikelihood.nll_components_from_reduced_sequence step=device_get")
        logp_g_np = np.asarray(self._jax.device_get(logp_g))
        logp_w_np = np.asarray(self._jax.device_get(logp_w))
        logp_xyz_np = np.asarray(self._jax.device_get(logp_xyz))
        logp_a_np = np.asarray(self._jax.device_get(logp_a))
        logp_l_np = np.asarray(self._jax.device_get(logp_l))
        out = {
            "logp_g": float(logp_g_np[0]),
            "logp_w": float(logp_w_np[0]),
            "logp_xyz": float(logp_xyz_np[0]),
            "logp_a": float(logp_a_np[0]),
            "logp_l": float(logp_l_np[0]),
        }
        out["nll_q"] = float(-out["logp_xyz"] if self.coordinate_only else -(out["logp_xyz"] + out["logp_l"]))
        return out

    def _nll_cache_key(
        self,
        *,
        seq: Mapping[str, np.ndarray],
        formula: str | None,
    ) -> tuple[Any, ...]:
        num_sites = int(seq["num_sites"])
        return (
            int(seq["G"]),
            str(formula or ""),
            tuple(np.asarray(seq["W"][:num_sites], dtype=int).tolist()),
            tuple(np.asarray(seq["A"][:num_sites], dtype=int).tolist()),
            tuple(np.round(np.asarray(seq["XYZ"][:num_sites], dtype=float).reshape(-1), 8).tolist()),
            tuple(np.round(np.asarray(seq["L"], dtype=float).reshape(-1), 8).tolist()),
            bool(self.coordinate_only),
        )

    def nll_components(
        self,
        *,
        payload: DiffCSPPPSymmetryPayload,
        q: np.ndarray | torch.Tensor,
        lattice_feature: np.ndarray | torch.Tensor,
        formula: str | None = None,
    ) -> dict[str, float]:
        _algo21_trace("CrystalFormerLikelihood.nll_components", sg=int(payload.spacegroup))
        if _algo21_log_path() is not None:
            _algo21_log(f"[crystalformer] nll_components start sg={int(payload.spacegroup)}")
        try:
            seq = build_crystalformer_reduced_sequence(
                payload=payload,
                q=q,
                lattice_feature=lattice_feature,
                n_max=self.n_max,
            )
            if _algo21_log_path() is not None:
                _algo21_log(
                    f"[crystalformer] reduced sequence ready sg={int(seq['G'])} num_sites={int(seq['num_sites'])}"
                )
            cache_key = self._nll_cache_key(seq=seq, formula=formula)
            if cache_key in self._nll_cache:
                out = dict(self._nll_cache[cache_key])
                if _algo21_log_path() is not None:
                    _algo21_log(
                        f"[crystalformer] nll_components cache-hit sg={int(payload.spacegroup)} num_sites={int(np.count_nonzero(seq['A']))} nll_q={float(out['nll_q']):.6g}"
                    )
                return out
            out = self.nll_components_from_reduced_sequence(seq=seq, formula=formula)
            if len(self._nll_cache) > 256:
                self._nll_cache.clear()
            self._nll_cache[cache_key] = dict(out)
            if _algo21_log_path() is not None:
                _algo21_log(
                    f"[crystalformer] nll_components done sg={int(payload.spacegroup)} num_sites={int(np.count_nonzero(seq['A']))} nll_q={float(out['nll_q']):.6g}"
                )
            return out
        finally:
            if self._safe_mode:
                try:
                    self._jax.clear_caches()
                except Exception:
                    pass
                gc.collect()

    def nll_q(
        self,
        *,
        payload: DiffCSPPPSymmetryPayload,
        q: np.ndarray | torch.Tensor,
        lattice_feature: np.ndarray | torch.Tensor,
        formula: str | None = None,
    ) -> float:
        _algo21_trace("CrystalFormerLikelihood.nll_q", sg=int(payload.spacegroup))
        return float(
            self.nll_components(
                payload=payload,
                q=q,
                lattice_feature=lattice_feature,
                formula=formula,
            )["nll_q"]
        )

    def grad_nll_q(
        self,
        *,
        payload: DiffCSPPPSymmetryPayload,
        q: np.ndarray | torch.Tensor,
        lattice_feature: np.ndarray | torch.Tensor,
        formula: str | None = None,
        eps: float = 1.0e-3,
        debug: bool = False,
        max_dims: int | None = None,
    ) -> np.ndarray:
        _algo21_trace("CrystalFormerLikelihood.grad_nll_q", sg=int(payload.spacegroup))
        try:
            return self._grad_nll_q_jax(
                payload=payload,
                q=q,
                lattice_feature=lattice_feature,
                formula=formula,
                debug=debug,
                max_dims=max_dims,
            )
        except Exception:
            if bool(debug) or bool(self.debug_prints):
                _algo21_log("[algo21.cf.grad] direct jax grad failed, falling back to finite differences")
                import traceback
                traceback.print_exc(limit=2)
        return self._grad_nll_q_finite_diff(
            payload=payload,
            q=q,
            lattice_feature=lattice_feature,
            formula=formula,
            eps=eps,
            debug=debug,
            max_dims=max_dims,
        )

    def _grad_nll_q_finite_diff(
        self,
        *,
        payload: DiffCSPPPSymmetryPayload,
        q: np.ndarray | torch.Tensor,
        lattice_feature: np.ndarray | torch.Tensor,
        formula: str | None = None,
        eps: float = 1.0e-3,
        debug: bool = False,
        max_dims: int | None = None,
    ) -> np.ndarray:
        _algo21_trace("CrystalFormerLikelihood._grad_nll_q_finite_diff", sg=int(payload.spacegroup))
        q0 = np.asarray(torch.as_tensor(q).detach().cpu(), dtype=float).reshape(-1)
        grad = np.zeros_like(q0)
        dbg = bool(debug) or bool(self.debug_prints)
        limit = int(q0.shape[0] if max_dims is None else max(0, min(int(max_dims), int(q0.shape[0]))))
        if dbg:
            _algo21_log(f"[algo21.cf.grad] start q_dim={q0.shape[0]} active_dims={limit} eps={float(eps):.3g}")
        for idx in range(limit):
            if dbg:
                _algo21_log(f"[algo21.cf.grad] idx={idx} f_plus start")
            q_plus = q0.copy()
            q_minus = q0.copy()
            q_plus[idx] = np.remainder(q_plus[idx] + float(eps), 1.0)
            q_minus[idx] = np.remainder(q_minus[idx] - float(eps), 1.0)
            f_plus = self.nll_q(payload=payload, q=q_plus, lattice_feature=lattice_feature, formula=formula)
            if dbg:
                _algo21_log(f"[algo21.cf.grad] idx={idx} f_plus done value={float(f_plus):.6g}")
                _algo21_log(f"[algo21.cf.grad] idx={idx} f_minus start")
            f_minus = self.nll_q(payload=payload, q=q_minus, lattice_feature=lattice_feature, formula=formula)
            if dbg:
                _algo21_log(f"[algo21.cf.grad] idx={idx} f_minus done value={float(f_minus):.6g}")
            grad[idx] = (f_plus - f_minus) / (2.0 * float(eps))
            if dbg:
                _algo21_log(f"[algo21.cf.grad] idx={idx} grad={float(grad[idx]):.6g}")
        if dbg and limit < q0.shape[0]:
            _algo21_log(f"[algo21.cf.grad] truncated remaining_dims={int(q0.shape[0] - limit)}")
        if dbg:
            _algo21_log("[algo21.cf.grad] done")
        return grad

    def _grad_nll_q_jax(
        self,
        *,
        payload: DiffCSPPPSymmetryPayload,
        q: np.ndarray | torch.Tensor,
        lattice_feature: np.ndarray | torch.Tensor,
        formula: str | None = None,
        debug: bool = False,
        max_dims: int | None = None,
    ) -> np.ndarray:
        _algo21_trace("CrystalFormerLikelihood._grad_nll_q_jax", sg=int(payload.spacegroup))
        self._ensure_runtime_loaded()
        q0 = np.asarray(torch.as_tensor(q).detach().cpu(), dtype=float).reshape(-1)
        chart = _get_wyckoff_dof_chart(payload)
        xyz_unsorted = _build_anchor_xyz_from_q(payload=payload, q=q0)
        w = np.asarray([crystalformer_letter_to_number(letter) for letter in payload.wyckoff_letters], dtype=int)
        a = np.asarray(payload.anchor_atomic_numbers, dtype=int)
        sort_idx = _sort_reduced_sites_index(w, xyz_unsorted)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(sort_idx.shape[0])
        xyz_sorted = np.remainder(xyz_unsorted, 1.0)[sort_idx]
        w_sorted = w[sort_idx]
        a_sorted = a[sort_idx]
        num_sites = int(len(w_sorted))
        if num_sites > int(self.n_max):
            raise ValueError(f"Reduced site count {num_sites} exceeds CrystalFormer n_max={self.n_max}.")
        composition = self._composition_vector(A=a_sorted, W=w_sorted, G=int(payload.spacegroup), formula=formula)
        key = self._jax.random.PRNGKey(int(self.seed))
        G = self._jnp.asarray([int(payload.spacegroup)], dtype=self._jnp.int32)
        L = self._jnp.asarray(np.asarray(torch.as_tensor(lattice_feature).detach().cpu(), dtype=float).reshape(1, 6), dtype=self._jnp.float32)
        A = self._jnp.asarray(np.concatenate([a_sorted, np.zeros((self.n_max - num_sites,), dtype=int)])[None, :], dtype=self._jnp.int32)
        W = self._jnp.asarray(np.concatenate([w_sorted, np.zeros((self.n_max - num_sites,), dtype=int)])[None, :], dtype=self._jnp.int32)
        comp = self._jnp.asarray(composition[None, :], dtype=self._jnp.int32)
        sentinel = self._jnp.asarray(np.full((self.n_max - num_sites, 3), 1.0e10, dtype=float), dtype=self._jnp.float32)
        dbg = bool(debug) or bool(self.debug_prints)
        limit = int(q0.shape[0] if max_dims is None else max(0, min(int(max_dims), int(q0.shape[0]))))
        active_q_mask = np.zeros_like(q0, dtype=bool)
        active_q_mask[:limit] = True

        def nll_from_xyz_active(xyz_active):
            xyz_full = self._jnp.concatenate([xyz_active, sentinel], axis=0) if num_sites < int(self.n_max) else xyz_active
            XYZ = xyz_full[None, :, :]
            logp_g, logp_w, logp_xyz, logp_a, logp_l = self._logp_fn_raw(
                self._params,
                key,
                comp,
                G,
                L,
                XYZ,
                A,
                W,
                False,
            )
            return -logp_xyz[0] if self.coordinate_only else -(logp_xyz[0] + logp_l[0])

        if dbg:
            _algo21_log(f"[algo21.cf.grad] direct-jax start q_dim={q0.shape[0]} active_dims={limit} num_sites={num_sites}")
        grad_xyz_sorted = np.asarray(self._jax.grad(nll_from_xyz_active)(self._jnp.asarray(xyz_sorted, dtype=self._jnp.float32)))
        grad_xyz_unsorted = grad_xyz_sorted[inv_idx]
        grad_q = np.zeros_like(q0)
        for site_idx, dof_slice in enumerate(chart.site_dof_slices):
            if dof_slice.stop <= dof_slice.start:
                continue
            site_mask = active_q_mask[dof_slice]
            if not np.any(site_mask):
                continue
            basis = np.asarray(chart.site_anchor_bases[site_idx], dtype=float)
            if basis.shape[1] == 0:
                continue
            local_grad = basis.T @ grad_xyz_unsorted[site_idx]
            full_local = grad_q[dof_slice]
            full_local[site_mask] = local_grad[site_mask]
            grad_q[dof_slice] = full_local
        if dbg:
            _algo21_log(f"[algo21.cf.grad] direct-jax done grad_norm={float(np.linalg.norm(grad_q)):.6g}")
        return grad_q

    def sample_q_candidates(
        self,
        *,
        payload: DiffCSPPPSymmetryPayload,
        lattice_feature: np.ndarray | torch.Tensor,
        formula: str | None,
        K: int,
        top_p: float = 1.0,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        self._ensure_runtime_loaded()
        _ensure_crystalformer_importable()
        from crystalformer.src.sample import inference, project_xyz, sample_x

        if int(K) <= 0:
            return [], []

        q_ref = np.asarray(_get_wyckoff_dof_chart(payload).q_ref, dtype=float).reshape(-1)
        seq_ref = build_crystalformer_reduced_sequence(
            payload=payload,
            q=q_ref,
            lattice_feature=lattice_feature,
            n_max=self.n_max,
        )
        num_sites = int(seq_ref["num_sites"])
        if num_sites <= 0:
            return [], []

        composition = np.asarray(
            self._composition_vector(
                A=seq_ref["A"],
                W=seq_ref["W"],
                G=int(seq_ref["G"]),
                formula=formula,
            ),
            dtype=np.int32,
        )
        W_fixed = np.asarray(seq_ref["W"][:num_sites], dtype=int)
        A_fixed = np.asarray(seq_ref["A"][:num_sites], dtype=int)
        sort_idx = _sort_reduced_sites_index(
            W_fixed,
            np.asarray(seq_ref["XYZ"][:num_sites], dtype=float),
        )
        payload_site_for_sorted = sort_idx.astype(int).tolist()

        jax = self._jax
        jnp = self._jnp
        batchsize = int(K)
        key = jax.random.PRNGKey(int(self.seed if seed is None else seed))
        G = jnp.full((batchsize,), int(payload.spacegroup), dtype=jnp.int32)
        # CrystalFormer's `inference(...)` vmaps over `(G, W, A, X, Y, Z)` but
        # keeps `composition` shared across the whole batch via `in_axes=None`.
        # Passing a `(batch, 119)` matrix here triggers shape/broadcast errors
        # inside the transformer because it expects a single `(119,)` vector.
        comp = jnp.asarray(composition, dtype=jnp.int32)
        W = jnp.zeros((batchsize, self.n_max), dtype=jnp.int32)
        A = jnp.zeros((batchsize, self.n_max), dtype=jnp.int32)
        X = jnp.zeros((batchsize, self.n_max), dtype=jnp.float32)
        Y = jnp.zeros((batchsize, self.n_max), dtype=jnp.float32)
        Z = jnp.zeros((batchsize, self.n_max), dtype=jnp.float32)

        for site_idx in range(num_sites):
            w_val = int(W_fixed[site_idx])
            a_val = int(A_fixed[site_idx])
            W = W.at[:, site_idx].set(w_val)
            A = A.at[:, site_idx].set(a_val)

            h_x = inference(self._transformer, self._params, comp, G, W, A, X, Y, Z)[1][:, 5 * site_idx + 2]
            key, x = sample_x(key, h_x, self.Kx, float(top_p), float(temperature), batchsize)
            xyz = jnp.concatenate([x[:, None], jnp.zeros((batchsize, 1)), jnp.zeros((batchsize, 1))], axis=-1)
            xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None), out_axes=0)(G, W[:, site_idx], xyz, 0)
            X = X.at[:, site_idx].set(xyz[:, 0])

            h_y = inference(self._transformer, self._params, comp, G, W, A, X, Y, Z)[1][:, 5 * site_idx + 3]
            key, y = sample_x(key, h_y, self.Kx, float(top_p), float(temperature), batchsize)
            xyz = jnp.concatenate([X[:, site_idx][:, None], y[:, None], jnp.zeros((batchsize, 1))], axis=-1)
            xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None), out_axes=0)(G, W[:, site_idx], xyz, 0)
            Y = Y.at[:, site_idx].set(xyz[:, 1])

            h_z = inference(self._transformer, self._params, comp, G, W, A, X, Y, Z)[1][:, 5 * site_idx + 4]
            key, z = sample_x(key, h_z, self.Kx, float(top_p), float(temperature), batchsize)
            xyz = jnp.concatenate([X[:, site_idx][:, None], Y[:, site_idx][:, None], z[:, None]], axis=-1)
            xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None), out_axes=0)(G, W[:, site_idx], xyz, 0)
            Z = Z.at[:, site_idx].set(xyz[:, 2])

        xyz_samples = np.asarray(
            jax.device_get(jnp.concatenate([X[..., None], Y[..., None], Z[..., None]], axis=-1))
        )[:, :num_sites, :]
        chart = _get_wyckoff_dof_chart(payload)
        q_candidates: list[np.ndarray] = []
        # Do not score every sampled q with CrystalFormer here. That is both
        # expensive and contrary to Algorithm21B's intended logic, where KLDM
        # geometry ranks first and CrystalFormer is only a tie-break inside a
        # small witness-qualified subset.
        cf_nll: list[float] = []

        for batch_idx in range(batchsize):
            q_parts: list[np.ndarray] = []
            for sorted_pos, payload_site_idx in enumerate(payload_site_for_sorted):
                dof_slice = chart.site_dof_slices[int(payload_site_idx)]
                if dof_slice.stop <= dof_slice.start:
                    continue
                basis = np.asarray(chart.site_anchor_bases[int(payload_site_idx)], dtype=float)
                offset = np.asarray(chart.site_anchor_offsets[int(payload_site_idx)], dtype=float)
                anchor = np.asarray(xyz_samples[batch_idx, sorted_pos], dtype=float)
                delta = _signed_wrap_numpy(anchor - offset)
                q_site, *_ = np.linalg.lstsq(basis, delta, rcond=None)
                q_parts.append(np.remainder(np.asarray(q_site, dtype=float), 1.0))
            q_candidate = np.concatenate(q_parts, axis=0) if q_parts else np.zeros((0,), dtype=float)
            q_candidates.append(q_candidate)
            cf_nll.append(float("nan"))

        return q_candidates, cf_nll


def _sigma_proj_weight(t_nodes: torch.Tensor, floor: float) -> float:
    t_mean = float(t_nodes.float().mean().detach().item())
    sigma_proj = max(t_mean, float(floor))
    return 1.0 / (2.0 * sigma_proj * sigma_proj)


def torus_interp(source: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    delta = wrapdiff(target, source)
    return wrap01(source + float(alpha) * delta)


def torus_soft_project(*, f0_hat: torch.Tensor, f0_hard: torch.Tensor, alpha: float) -> torch.Tensor:
    return torus_interp(f0_hat, f0_hard, alpha=float(alpha))


def q_only_clean_cf_fit(
    *,
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    t_nodes: torch.Tensor,
    lattice_feature: torch.Tensor,
    formula: str | None,
    config: Algorithm21Config,
    cf_likelihood: CrystalFormerLikelihood | None = None,
    q_init: torch.Tensor | None = None,
) -> Algorithm21QFitResult:
    z_payload = z_payload.detach().clone()
    lattice_feature = lattice_feature.detach().clone()
    debug = bool(config.debug_prints)
    chart = _get_wyckoff_dof_chart(payload)
    q_raw = _initialize_q_raw(
        chart=chart,
        device=z_payload.device,
        dtype=z_payload.dtype,
        q_init=q_init,
        q_init_mode=config.q_init_mode,
    )
    optimizer = torch.optim.Adam([q_raw], lr=float(config.q_lr))
    weight_near = _sigma_proj_weight(t_nodes, float(config.sigma_proj_floor))
    logs: list[dict[str, Any]] = []
    cf_nll_start = 0.0
    cf_scale = 1.0
    cf_value_enabled = float(config.beta) > 0.0 and cf_likelihood is not None
    cf_grad_enabled = bool(cf_value_enabled and _truthy_env("KLDM_ALGO21_ENABLE_CF_GRAD", "false"))
    if cf_value_enabled:
        if debug:
            _algo21_log(f"[algo21.qfit] init cf_start start beta={float(config.beta):.3g} q_dim={int(q_raw.numel())} z={tuple(z_payload.shape)} lattice={tuple(lattice_feature.shape)}")
        q_start = torch.remainder(q_raw.detach(), 1.0)
        q_start_np = np.asarray(q_start.detach().cpu(), dtype=float)
        cf_nll_start = float(
            cf_likelihood.nll_q(
                payload=payload,
                q=q_start_np,
                lattice_feature=lattice_feature.detach().cpu(),
                formula=formula,
            )
        )
        if debug:
            _algo21_log(f"[algo21.qfit] init cf_start done nll_start={cf_nll_start:.6g}")
        if bool(config.cf_delta_normalize):
            cf_scale = max(abs(cf_nll_start), 1.0e-8)
            if debug:
                _algo21_log(f"[algo21.qfit] init cf_scale={cf_scale:.6g}")
        if debug and not cf_grad_enabled:
            _algo21_log("[algo21.qfit] CF gradient disabled; using witness optimizer with value-only CF diagnostics")

    for step_idx in range(max(int(config.q_opt_steps), 1)):
        if debug:
            _algo21_log(f"[algo21.qfit] step={step_idx} start")
        optimizer.zero_grad(set_to_none=True)
        q = torch.remainder(q_raw, 1.0)
        z_sym = chart.expand_q(q, device=z_payload.device, dtype=z_payload.dtype)
        near_loss = weight_near * witness_torus_sin_loss(z_sym, z_payload)
        near_loss.backward()
        if debug:
            _algo21_log(f"[algo21.qfit] step={step_idx} near_loss={float(near_loss.detach().item()):.6g}")
        cf_nll = 0.0
        cf_delta = 0.0
        cf_term = 0.0
        cf_grad_norm = 0.0
        if cf_grad_enabled and q.numel() > 0:
            q_np = np.asarray(q.detach().cpu(), dtype=float)
            if debug:
                _algo21_log(f"[algo21.qfit] step={step_idx} cf_nll start")
            cf_nll = float(
                cf_likelihood.nll_q(
                    payload=payload,
                    q=q_np,
                    lattice_feature=lattice_feature.detach().cpu(),
                    formula=formula,
                )
            )
            if debug:
                _algo21_log(f"[algo21.qfit] step={step_idx} cf_nll done nll={cf_nll:.6g}")
            cf_delta = float(cf_nll - cf_nll_start)
            cf_term = float(cf_delta / cf_scale) if bool(config.cf_use_delta) else float(cf_nll)
            if debug:
                _algo21_log(f"[algo21.qfit] step={step_idx} cf_delta={cf_delta:.6g} cf_term={cf_term:.6g}")
                _algo21_log(f"[algo21.qfit] step={step_idx} cf_grad start eps={float(config.finite_diff_eps):.3g}")
            cf_grad = cf_likelihood.grad_nll_q(
                payload=payload,
                q=q_np,
                lattice_feature=lattice_feature.detach().cpu(),
                formula=formula,
                eps=float(config.finite_diff_eps),
                debug=debug,
                max_dims=config.cf_grad_max_dims,
            )
            if debug:
                _algo21_log(f"[algo21.qfit] step={step_idx} cf_grad done")
            cf_grad_tensor = torch.as_tensor(cf_grad, device=q_raw.device, dtype=q_raw.dtype)
            if bool(config.cf_use_delta):
                cf_grad_tensor = cf_grad_tensor / float(cf_scale)
            q_raw.grad = q_raw.grad + float(config.beta) * cf_grad_tensor
            cf_grad_norm = float(torch.linalg.norm(cf_grad_tensor).detach().item())
            if debug:
                _algo21_log(f"[algo21.qfit] step={step_idx} cf_grad_norm={cf_grad_norm:.6g}")
        torch.nn.utils.clip_grad_norm_([q_raw], max_norm=float(config.grad_clip))
        optimizer.step()
        if debug:
            _algo21_log(f"[algo21.qfit] step={step_idx} optimizer step done")
        with torch.no_grad():
            q_now = torch.remainder(q_raw, 1.0)
            z_now = chart.expand_q(q_now, device=z_payload.device, dtype=z_payload.dtype)
            witness_now = float(witness_torus_sin_loss(z_now, z_payload).detach().item())
            total_now = float(weight_near * witness_now + float(config.beta) * cf_term)
            q_grad_norm = float(torch.linalg.norm(q_raw.grad.detach()).item()) if q_raw.grad is not None else 0.0
            if debug:
                _algo21_log(f"[algo21.qfit] step={step_idx} done witness={witness_now:.6g} total={total_now:.6g} q_grad_norm={q_grad_norm:.6g}")
            logs.append(
                {
                    "step": int(step_idx),
                    "near_loss": float(near_loss.detach().item()),
                    "witness_sin": witness_now,
                    "cf_nll": float(cf_nll),
                    "cf_nll_start": float(cf_nll_start),
                    "cf_delta": float(cf_delta),
                    "cf_term": float(cf_term),
                    "score_total": total_now,
                    "q_grad_norm": q_grad_norm,
                    "cf_grad_norm": cf_grad_norm,
                }
            )

    with torch.no_grad():
        if debug:
            _algo21_log("[algo21.qfit] finalize start")
        q_star = torch.remainder(q_raw, 1.0).detach().clone()
        z_proj = chart.expand_q(q_star, device=z_payload.device, dtype=z_payload.dtype)
        witness_sin = float(witness_torus_sin_loss(z_proj, z_payload).detach().item())
        cf_nll = 0.0
        cf_delta = 0.0
        cf_term = 0.0
        if cf_value_enabled and q_star.numel() > 0:
            if debug:
                _algo21_log("[algo21.qfit] finalize cf_nll start")
            cf_nll = float(
                cf_likelihood.nll_q(
                    payload=payload,
                    q=np.asarray(q_star.detach().cpu(), dtype=float),
                    lattice_feature=lattice_feature.detach().cpu(),
                    formula=formula,
                )
            )
            if debug:
                _algo21_log(f"[algo21.qfit] finalize cf_nll done nll={cf_nll:.6g}")
            cf_delta = float(cf_nll - cf_nll_start)
            cf_term = float(cf_delta / cf_scale) if bool(config.cf_use_delta) else float(cf_nll)
        near_loss = float(weight_near * witness_sin)
        if debug:
            _algo21_log(f"[algo21.qfit] finalize done witness={witness_sin:.6g} near={near_loss:.6g} delta={cf_delta:.6g} total={float(near_loss + float(config.beta) * cf_term):.6g}")
        return Algorithm21QFitResult(
            q_star=q_star,
            z_proj_payload=z_proj.detach().clone(),
            near_loss=near_loss,
            witness_sin=witness_sin,
            witness_rmse_payload=float(torus_rmse(z_payload, z_proj).detach().item()),
            cf_nll=float(cf_nll),
            cf_nll_start=float(cf_nll_start),
            cf_delta=float(cf_delta),
            score_total=float(near_loss + float(config.beta) * cf_term),
            logs=tuple(logs),
        )


def fit_q_to_clean_prediction(
    *,
    z_hat: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    t_nodes: torch.Tensor,
    lattice_feature: torch.Tensor,
    q_init: torch.Tensor | None = None,
    q_init_mode: str = "random",
    steps: int = 100,
    lr: float = 1.0e-2,
    grad_clip: float = 10.0,
) -> Algorithm21QFitResult:
    config = Algorithm21Config(
        beta=0.0,
        q_opt_steps=int(steps),
        q_lr=float(lr),
        grad_clip=float(grad_clip),
        q_init_mode=str(q_init_mode),
    )
    return q_only_clean_cf_fit(
        z_payload=z_hat,
        payload=payload,
        t_nodes=t_nodes,
        lattice_feature=lattice_feature,
        formula=None,
        config=config,
        cf_likelihood=None,
        q_init=q_init,
    )


def q_only_clean_cf_local_rerank(
    *,
    z_payload: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    lattice_feature: torch.Tensor,
    formula: str | None,
    cf_likelihood: CrystalFormerLikelihood | None,
    q_center: torch.Tensor,
    radius: float = 5.0e-2,
    candidate_count: int = 32,
    keep_tolerance: float = 5.0e-2,
    seed: int = 0,
    debug: bool = False,
) -> Algorithm21LocalRerankResult:
    chart = _get_wyckoff_dof_chart(payload)
    q_center = torch.remainder(torch.as_tensor(q_center).detach().clone().reshape(-1), 1.0)
    z_payload = z_payload.detach().clone()
    lattice_feature = lattice_feature.detach().clone()
    z_center = chart.expand_q(q_center, device=z_payload.device, dtype=z_payload.dtype)
    witness_center = float(witness_torus_sin_loss(z_center, z_payload).detach().item())
    cf_nll_center = float("nan")
    if cf_likelihood is not None and q_center.numel() > 0:
        cf_nll_center = float(
            cf_likelihood.nll_q(
                payload=payload,
                q=np.asarray(q_center.detach().cpu(), dtype=float),
                lattice_feature=lattice_feature.detach().cpu(),
                formula=formula,
            )
        )

    if debug:
        _algo21_log(
            f"[algo21.rerank] start q_dim={int(q_center.numel())} radius={float(radius):.3g} "
            f"candidate_count={int(candidate_count)} keep_tol={float(keep_tolerance):.3g} "
            f"witness_center={float(witness_center):.6g} cf_center={float(cf_nll_center):.6g}"
        )

    if cf_likelihood is None or q_center.numel() == 0 or int(candidate_count) <= 0:
        return Algorithm21LocalRerankResult(
            q_center=q_center.detach().clone(),
            q_best=q_center.detach().clone(),
            z_center_payload=z_center.detach().clone(),
            z_best_payload=z_center.detach().clone(),
            witness_center=float(witness_center),
            witness_best=float(witness_center),
            cf_nll_center=float(cf_nll_center),
            cf_nll_best=float(cf_nll_center),
            candidate_count=0,
            kept_count=0,
            rows=tuple(),
        )

    rng = np.random.default_rng(int(seed))
    q_best = q_center.detach().clone()
    z_best = z_center.detach().clone()
    witness_best = float(witness_center)
    cf_best = float(cf_nll_center)
    rows: list[dict[str, Any]] = []
    kept_count = 0
    witness_cap = float(witness_center * (1.0 + float(keep_tolerance)))

    for cand_idx in range(int(candidate_count)):
        noise = rng.normal(size=q_center.numel())
        norm = float(np.linalg.norm(noise))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            noise = np.zeros_like(noise)
        else:
            noise = noise / norm
        scale = float(radius) * float(rng.uniform(0.0, 1.0) ** (1.0 / max(int(q_center.numel()), 1)))
        q_cand = torch.remainder(
            q_center + torch.as_tensor(scale * noise, device=q_center.device, dtype=q_center.dtype),
            1.0,
        )
        z_cand = chart.expand_q(q_cand, device=z_payload.device, dtype=z_payload.dtype)
        witness_cand = float(witness_torus_sin_loss(z_cand, z_payload).detach().item())
        keep = bool(witness_cand <= witness_cap)
        cf_cand = float("nan")
        if keep:
            kept_count += 1
            cf_cand = float(
                cf_likelihood.nll_q(
                    payload=payload,
                    q=np.asarray(q_cand.detach().cpu(), dtype=float),
                    lattice_feature=lattice_feature.detach().cpu(),
                    formula=formula,
                )
            )
            if cf_cand < cf_best:
                q_best = q_cand.detach().clone()
                z_best = z_cand.detach().clone()
                witness_best = float(witness_cand)
                cf_best = float(cf_cand)
        rows.append(
            {
                "candidate_index": int(cand_idx),
                "step_norm": float(scale),
                "witness": float(witness_cand),
                "keep": bool(keep),
                "cf_nll": float(cf_cand),
            }
        )
        if debug:
            _algo21_log(
                f"[algo21.rerank] cand={int(cand_idx)} step_norm={float(scale):.6g} "
                f"witness={float(witness_cand):.6g} keep={bool(keep)} cf={float(cf_cand):.6g}"
            )

    if debug:
        _algo21_log(
            f"[algo21.rerank] done kept={int(kept_count)}/{int(candidate_count)} "
            f"witness_best={float(witness_best):.6g} cf_best={float(cf_best):.6g}"
        )
    return Algorithm21LocalRerankResult(
        q_center=q_center.detach().clone(),
        q_best=q_best.detach().clone(),
        z_center_payload=z_center.detach().clone(),
        z_best_payload=z_best.detach().clone(),
        witness_center=float(witness_center),
        witness_best=float(witness_best),
        cf_nll_center=float(cf_nll_center),
        cf_nll_best=float(cf_best),
        candidate_count=int(candidate_count),
        kept_count=int(kept_count),
        rows=tuple(rows),
    )


def sample_q_from_crystalformer(
    *,
    payload: DiffCSPPPSymmetryPayload,
    lattice_feature: torch.Tensor,
    formula: str | None,
    cf_likelihood: CrystalFormerLikelihood,
    K: int,
    top_p: float = 1.0,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[float]]:
    q_candidates_np, cf_nll = cf_likelihood.sample_q_candidates(
        payload=payload,
        lattice_feature=lattice_feature.detach().cpu(),
        formula=formula,
        K=int(K),
        top_p=float(top_p),
        temperature=float(temperature),
        seed=int(seed),
    )
    q_candidates = [
        torch.as_tensor(q_i, device=lattice_feature.device, dtype=lattice_feature.dtype)
        for q_i in q_candidates_np
    ]
    return q_candidates, [float(v) for v in cf_nll]


def rank_q_candidates(
    *,
    z_hat: torch.Tensor,
    payload: DiffCSPPPSymmetryPayload,
    q_candidates: list[torch.Tensor],
    cf_nll: list[float] | None = None,
    top_k: int = 3,
    epsilon_abs: float = 1.0e-3,
) -> tuple[Algorithm21RankedQCandidate, ...]:
    chart = _get_wyckoff_dof_chart(payload)
    rows: list[Algorithm21RankedQCandidate] = []
    cf_scores = [float("nan")] * len(q_candidates) if cf_nll is None else [float(v) for v in cf_nll]
    for idx, q in enumerate(q_candidates):
        z_proj = chart.expand_q(q, device=z_hat.device, dtype=z_hat.dtype)
        witness_sin = float(witness_torus_sin_loss(z_proj, z_hat).detach().item())
        rows.append(
            Algorithm21RankedQCandidate(
                rank=-1,
                source_index=int(idx),
                q=q.detach().clone(),
                z_payload=z_proj.detach().clone(),
                witness_sin=witness_sin,
                witness_rmse_payload=float(torus_rmse(z_hat, z_proj).detach().item()),
                cf_nll=float(cf_scores[idx]),
                geometry_kept=False,
            )
        )
    if not rows:
        return tuple()

    def _cf_sort_value(value: float) -> float:
        return float(value) if np.isfinite(float(value)) else float("inf")

    rows.sort(key=lambda item: (float(item.witness_sin), _cf_sort_value(float(item.cf_nll))))
    witness_best = float(rows[0].witness_sin)
    keep_cap = witness_best + float(epsilon_abs)
    kept = [
        item for item in rows
        if float(item.witness_sin) <= keep_cap
    ]
    pool = kept if kept else rows
    pool = sorted(pool, key=lambda item: (_cf_sort_value(float(item.cf_nll)), float(item.witness_sin)))
    ranked: list[Algorithm21RankedQCandidate] = []
    for rank_idx, item in enumerate(pool[: max(int(top_k), 1)]):
        ranked.append(
            replace(
                item,
                rank=int(rank_idx),
                geometry_kept=bool(float(item.witness_sin) <= keep_cap),
            )
        )
    return tuple(ranked)


def score_ranked_q_candidates_with_crystalformer(
    *,
    ranked: tuple[Algorithm21RankedQCandidate, ...],
    payload: DiffCSPPPSymmetryPayload,
    lattice_feature: torch.Tensor,
    formula: str | None,
    cf_likelihood: CrystalFormerLikelihood | None,
) -> tuple[Algorithm21RankedQCandidate, ...]:
    if cf_likelihood is None or not ranked:
        return ranked
    rescored: list[Algorithm21RankedQCandidate] = []
    for item in ranked:
        cf_nll = float(
            cf_likelihood.nll_q(
                payload=payload,
                q=np.asarray(item.q.detach().cpu(), dtype=float),
                lattice_feature=lattice_feature.detach().cpu(),
                formula=formula,
            )
        )
        rescored.append(replace(item, cf_nll=float(cf_nll)))
    rescored.sort(key=lambda item: (float(item.cf_nll), float(item.witness_sin)))
    return tuple(replace(item, rank=int(rank_idx)) for rank_idx, item in enumerate(rescored))


def post_renoise_score(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    q_init: torch.Tensor | None = None,
) -> Algorithm21QFitResult:
    witness_only = Algorithm21Config(beta=0.0)
    f0_hat = kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state.f,
        v=state.v,
        l=state.l,
        atom_types=state.atom_types,
        t_graph=state.t_graph,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
        edge_index=state.edge_node_index,
        variant=witness_only.denoiser_variant,
        coordinate_score_mode=witness_only.coordinate_score_mode,
    )
    z_hat = map_model_to_payload_reference_chart(f0_hat, payload)
    return q_only_clean_cf_fit(
        z_payload=z_hat,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        formula=None,
        config=witness_only,
        cf_likelihood=None,
        q_init=q_init,
    )


def post_renoise_acceptance(
    *,
    state_before: Algorithm19State,
    state_candidate: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    q_init_before: torch.Tensor | None = None,
    q_init_after: torch.Tensor | None = None,
) -> tuple[bool, Algorithm21QFitResult, Algorithm21QFitResult]:
    fit_before = post_renoise_score(
        state=state_before,
        payload=payload,
        model=model,
        q_init=q_init_before,
    )
    fit_after = post_renoise_score(
        state=state_candidate,
        payload=payload,
        model=model,
        q_init=q_init_after,
    )
    return bool(fit_after.witness_sin < fit_before.witness_sin), fit_before, fit_after


def renoise_from_f0(
    *,
    f0_star: torch.Tensor,
    state: Algorithm19State,
    model,
) -> Algorithm19State:
    f_candidate, v_candidate, *_ = kldm_renoise_from_f0(
        model=model,
        f0_star=f0_star,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
    )
    return replace(state, f=f_candidate.detach().clone(), v=v_candidate.detach().clone())


def algorithm21b_project_renoise_step(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    config: Algorithm21Config,
    cf_likelihood: CrystalFormerLikelihood,
    formula: str | None = None,
) -> tuple[Algorithm21StepResult, tuple[Algorithm21RankedQCandidate, ...], tuple[Algorithm21BranchResult, ...]]:
    f0_hat_before = kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state.f,
        v=state.v,
        l=state.l,
        atom_types=state.atom_types,
        t_graph=state.t_graph,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
        edge_index=state.edge_node_index,
        variant=config.denoiser_variant,
        coordinate_score_mode=config.coordinate_score_mode,
    )
    z_hat = map_model_to_payload_reference_chart(f0_hat_before, payload)
    q_candidates, cf_nll = sample_q_from_crystalformer(
        payload=payload,
        lattice_feature=state.l,
        formula=formula,
        cf_likelihood=cf_likelihood,
        K=int(config.cf_sample_k),
        top_p=float(config.cf_top_p),
        temperature=float(config.cf_temperature),
        seed=int(config.cf_sampler_seed),
    )
    ranked = rank_q_candidates(
        z_hat=z_hat,
        payload=payload,
        q_candidates=q_candidates,
        cf_nll=cf_nll,
        top_k=int(config.cf_top_k),
        epsilon_abs=float(config.cf_rank_eps_abs),
    )
    ranked = score_ranked_q_candidates_with_crystalformer(
        ranked=ranked,
        payload=payload,
        lattice_feature=state.l,
        formula=formula,
        cf_likelihood=cf_likelihood,
    )
    if not ranked:
        fallback = algorithm21_project_renoise_step(
            state=state,
            payload=payload,
            model=model,
            config=replace(config, beta=0.0),
            cf_likelihood=None,
            formula=formula,
            q_init=None,
        )
        return fallback, tuple(), tuple()

    branch_results: list[Algorithm21BranchResult] = []
    baseline_fit = post_renoise_score(state=state, payload=payload, model=model)
    baseline_score = float(baseline_fit.witness_sin)
    best_branch: Algorithm21BranchResult | None = None
    best_score = float("inf")

    for candidate in ranked:
        branch_step = algorithm21_project_renoise_step_from_q(
            state=state,
            payload=payload,
            model=model,
            q_star=candidate.q,
            alpha=float(config.alpha),
            post_renoise_accept=bool(config.post_renoise_acceptance),
            denoiser_variant=config.denoiser_variant,
            coordinate_score_mode=config.coordinate_score_mode,
        )
        branch = Algorithm21BranchResult(candidate=candidate, step_result=branch_step)
        branch_results.append(branch)
        score = float(branch_step.fit_after.witness_sin)
        if branch_step.accepted and score < best_score:
            best_branch = branch
            best_score = score

    if best_branch is not None and best_score < baseline_score:
        return best_branch.step_result, ranked, tuple(branch_results)

    fallback = best_branch.step_result if best_branch is not None else branch_results[0].step_result
    fallback = replace(fallback, accepted=False, state_after=state)
    return fallback, ranked, tuple(branch_results)


def algorithm21_project_renoise_step(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    config: Algorithm21Config,
    cf_likelihood: CrystalFormerLikelihood | None = None,
    formula: str | None = None,
    q_init: torch.Tensor | None = None,
) -> Algorithm21StepResult:
    debug = bool(config.debug_prints)
    if debug:
        _algo21_log(
            f"[algo21.step] start beta={float(config.beta):.3g} alpha={float(config.alpha):.3g} "
            f"f={tuple(state.f.shape)} v={tuple(state.v.shape)} l={tuple(state.l.shape)} "
            f"t_graph={tuple(state.t_graph.shape)} t_nodes={tuple(state.t_nodes.shape)}"
        )
        _algo21_log("[algo21.step] clean denoise before start")
    f0_hat_before = kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state.f,
        v=state.v,
        l=state.l,
        atom_types=state.atom_types,
        t_graph=state.t_graph,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
        edge_index=state.edge_node_index,
        variant=config.denoiser_variant,
        coordinate_score_mode=config.coordinate_score_mode,
    )
    if debug:
        _algo21_log(f"[algo21.step] clean denoise before done f0_hat={tuple(f0_hat_before.shape)}")
    z_before = map_model_to_payload_reference_chart(f0_hat_before, payload)
    if debug:
        _algo21_log(f"[algo21.step] map before done z_before={tuple(z_before.shape)}")
        _algo21_log("[algo21.step] fit_before start")
    fit_before = q_only_clean_cf_fit(
        z_payload=z_before,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        formula=formula,
        config=config,
        cf_likelihood=cf_likelihood,
        q_init=q_init,
    )
    if debug:
        _algo21_log(
            f"[algo21.step] fit_before done witness={float(fit_before.witness_sin):.6g} "
            f"cf={float(fit_before.cf_nll):.6g} delta={float(fit_before.cf_delta):.6g} "
            f"score={float(fit_before.score_total):.6g}"
        )
        _algo21_log("[algo21.step] hard/soft anchor build start")
    f0_hard = map_payload_reference_chart_to_model_frame(fit_before.z_proj_payload, payload)
    f0_star = torus_interp(f0_hat_before, f0_hard, alpha=float(config.alpha))
    if debug:
        _algo21_log(
            f"[algo21.step] hard/soft anchor build done f0_hard={tuple(f0_hard.shape)} "
            f"f0_star={tuple(f0_star.shape)}"
        )
        _algo21_log("[algo21.step] renoise start")
    f_candidate, v_candidate, *_ = kldm_renoise_from_f0(
        model=model,
        f0_star=f0_star,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
    )
    if debug:
        _algo21_log(
            f"[algo21.step] renoise done f_candidate={tuple(f_candidate.shape)} "
            f"v_candidate={tuple(v_candidate.shape)}"
        )
    state_candidate = replace(state, f=f_candidate.detach().clone(), v=v_candidate.detach().clone())
    if debug:
        _algo21_log("[algo21.step] clean denoise after start")
    f0_hat_after = kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state_candidate.f,
        v=state_candidate.v,
        l=state_candidate.l,
        atom_types=state_candidate.atom_types,
        t_graph=state_candidate.t_graph,
        t_nodes=state_candidate.t_nodes,
        node_index=state_candidate.node_index,
        edge_index=state_candidate.edge_node_index,
        variant=config.denoiser_variant,
        coordinate_score_mode=config.coordinate_score_mode,
    )
    if debug:
        _algo21_log(f"[algo21.step] clean denoise after done f0_hat={tuple(f0_hat_after.shape)}")
    z_after = map_model_to_payload_reference_chart(f0_hat_after, payload)
    if debug:
        _algo21_log(f"[algo21.step] map after done z_after={tuple(z_after.shape)}")
        _algo21_log("[algo21.step] fit_after start")
    fit_after_config = config
    fit_after_cf = cf_likelihood
    if bool(config.cf_value_only_after_renoise) and float(config.beta) > 0.0:
        fit_after_config = replace(config, beta=0.0)
        fit_after_cf = None
        if debug:
            _algo21_log("[algo21.step] fit_after using witness-only q-fit with value-only CF evaluation")
    fit_after = q_only_clean_cf_fit(
        z_payload=z_after,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        formula=formula,
        config=fit_after_config,
        cf_likelihood=fit_after_cf,
        q_init=fit_before.q_star,
    )
    if bool(config.cf_value_only_after_renoise) and float(config.beta) > 0.0 and cf_likelihood is not None and fit_after.q_star.numel() > 0:
        cf_after_value = float(
            cf_likelihood.nll_q(
                payload=payload,
                q=np.asarray(fit_after.q_star.detach().cpu(), dtype=float),
                lattice_feature=state.l.detach().cpu(),
                formula=formula,
            )
        )
        cf_after_delta = float(cf_after_value - fit_before.cf_nll_start)
        fit_after = replace(
            fit_after,
            cf_nll=cf_after_value,
            cf_nll_start=float(fit_before.cf_nll_start),
            cf_delta=cf_after_delta,
            score_total=float(fit_after.near_loss + float(config.beta) * cf_after_delta),
        )
    if debug:
        _algo21_log(
            f"[algo21.step] fit_after done witness={float(fit_after.witness_sin):.6g} "
            f"cf={float(fit_after.cf_nll):.6g} delta={float(fit_after.cf_delta):.6g} "
            f"score={float(fit_after.score_total):.6g}"
        )
    accepted = True
    if bool(config.post_renoise_acceptance):
        accepted = bool(fit_after.score_total < fit_before.score_total)
    if debug:
        _algo21_log(f"[algo21.step] accept={bool(accepted)}")
    logs = [
        {
            "c_before_witness": float(fit_before.witness_sin),
            "c_after_clean_anchor": float(fit_before.witness_sin),
            "c_after_redenoise": float(fit_after.witness_sin),
            "cf_before": float(fit_before.cf_nll),
            "cf_after": float(fit_after.cf_nll),
            "cf_delta_before": float(fit_before.cf_delta),
            "cf_delta_after": float(fit_after.cf_delta),
            "score_before": float(fit_before.score_total),
            "score_after": float(fit_after.score_total),
            "accepted": bool(accepted),
            "alpha": float(config.alpha),
            "beta": float(config.beta),
        }
    ]
    return Algorithm21StepResult(
        state_before=state,
        state_candidate=state_candidate,
        state_after=state_candidate if accepted else state,
        accepted=bool(accepted),
        f0_hat_before=f0_hat_before.detach().clone(),
        f0_hat_after=f0_hat_after.detach().clone(),
        f0_hard=f0_hard.detach().clone(),
        f0_star=f0_star.detach().clone(),
        fit_before=fit_before,
        fit_after=fit_after,
        logs=tuple(logs),
    )


def algorithm21_project_renoise_step_from_q(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    q_star: torch.Tensor,
    alpha: float = 0.25,
    post_renoise_accept: bool = True,
    denoiser_variant: str = "minus",
    coordinate_score_mode: str = "direct",
) -> Algorithm21StepResult:
    f0_hat_before = predict_clean_f0(
        state=state,
        model=model,
        denoiser_variant=denoiser_variant,
        coordinate_score_mode=coordinate_score_mode,
    )
    z_before = model_to_payload(f_model=f0_hat_before, payload=payload)
    z_proj_payload = expand_q(payload=payload, q=q_star.to(device=z_before.device, dtype=z_before.dtype))
    fit_before = Algorithm21QFitResult(
        q_star=q_star.detach().clone(),
        z_proj_payload=z_proj_payload.detach().clone(),
        near_loss=float(_sigma_proj_weight(state.t_nodes, 5.0e-2) * witness_torus_sin_loss(z_proj_payload, z_before).detach().item()),
        witness_sin=float(witness_torus_sin_loss(z_proj_payload, z_before).detach().item()),
        witness_rmse_payload=float(torus_rmse(z_before, z_proj_payload).detach().item()),
        cf_nll=0.0,
        cf_nll_start=0.0,
        cf_delta=0.0,
        score_total=float(witness_torus_sin_loss(z_proj_payload, z_before).detach().item()),
        logs=tuple(),
    )
    f0_hard = payload_to_model(z_payload=z_proj_payload, payload=payload)
    f0_star = torus_soft_project(f0_hat=f0_hat_before, f0_hard=f0_hard, alpha=float(alpha))
    state_candidate = renoise_from_f0(f0_star=f0_star, state=state, model=model)
    accepted, _fit_before_witness, fit_after = post_renoise_acceptance(
        state_before=state,
        state_candidate=state_candidate,
        payload=payload,
        model=model,
        q_init_before=q_star,
        q_init_after=q_star,
    )
    if not bool(post_renoise_accept):
        accepted = True
    f0_hat_after = predict_clean_f0(
        state=state_candidate,
        model=model,
        denoiser_variant=denoiser_variant,
        coordinate_score_mode=coordinate_score_mode,
    )
    return Algorithm21StepResult(
        state_before=state,
        state_candidate=state_candidate,
        state_after=state_candidate if accepted else state,
        accepted=bool(accepted),
        f0_hat_before=f0_hat_before.detach().clone(),
        f0_hat_after=f0_hat_after.detach().clone(),
        f0_hard=f0_hard.detach().clone(),
        f0_star=f0_star.detach().clone(),
        fit_before=fit_before,
        fit_after=fit_after,
        logs=(
            {
                "accepted": bool(accepted),
                "alpha": float(alpha),
                "witness_before": float(fit_before.witness_sin),
                "witness_after": float(fit_after.witness_sin),
            },
        ),
    )


def algorithm21_project_renoise_step_local_rerank(
    *,
    state: Algorithm19State,
    payload: DiffCSPPPSymmetryPayload,
    model,
    alpha: float,
    rerank_radius: float,
    rerank_candidate_count: int,
    rerank_keep_tolerance: float,
    cf_likelihood: CrystalFormerLikelihood | None = None,
    formula: str | None = None,
    q_init: torch.Tensor | None = None,
    seed: int = 0,
    debug: bool = False,
) -> Algorithm21StepResult:
    if debug:
        _algo21_log(
            f"[algo21.step.rerank] start alpha={float(alpha):.3g} radius={float(rerank_radius):.3g} "
            f"candidates={int(rerank_candidate_count)} keep_tol={float(rerank_keep_tolerance):.3g}"
        )
    pure_config = Algorithm21Config(beta=0.0, alpha=float(alpha), debug_prints=bool(debug))
    f0_hat_before = kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state.f,
        v=state.v,
        l=state.l,
        atom_types=state.atom_types,
        t_graph=state.t_graph,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
        edge_index=state.edge_node_index,
        variant=pure_config.denoiser_variant,
        coordinate_score_mode=pure_config.coordinate_score_mode,
    )
    z_before = map_model_to_payload_reference_chart(f0_hat_before, payload)
    fit_pure = q_only_clean_cf_fit(
        z_payload=z_before,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        formula=formula,
        config=pure_config,
        cf_likelihood=None,
        q_init=q_init,
    )
    rerank = q_only_clean_cf_local_rerank(
        z_payload=z_before,
        payload=payload,
        lattice_feature=state.l,
        formula=formula,
        cf_likelihood=cf_likelihood,
        q_center=fit_pure.q_star,
        radius=float(rerank_radius),
        candidate_count=int(rerank_candidate_count),
        keep_tolerance=float(rerank_keep_tolerance),
        seed=int(seed),
        debug=bool(debug),
    )
    fit_before = replace(
        fit_pure,
        q_star=rerank.q_best.detach().clone(),
        z_proj_payload=rerank.z_best_payload.detach().clone(),
        cf_nll=float(rerank.cf_nll_best),
        cf_nll_start=float(rerank.cf_nll_center),
        cf_delta=float(rerank.cf_nll_best - rerank.cf_nll_center),
        score_total=float(fit_pure.near_loss),
    )
    f0_hard = map_payload_reference_chart_to_model_frame(fit_before.z_proj_payload, payload)
    f0_star = torus_interp(f0_hat_before, f0_hard, alpha=float(alpha))
    f_candidate, v_candidate, *_ = kldm_renoise_from_f0(
        model=model,
        f0_star=f0_star,
        t_nodes=state.t_nodes,
        node_index=state.node_index,
    )
    state_candidate = replace(state, f=f_candidate.detach().clone(), v=v_candidate.detach().clone())
    f0_hat_after = kldm_clean_fractional_denoiser_Df(
        model=model,
        f=state_candidate.f,
        v=state_candidate.v,
        l=state_candidate.l,
        atom_types=state_candidate.atom_types,
        t_graph=state_candidate.t_graph,
        t_nodes=state_candidate.t_nodes,
        node_index=state_candidate.node_index,
        edge_index=state_candidate.edge_node_index,
        variant=pure_config.denoiser_variant,
        coordinate_score_mode=pure_config.coordinate_score_mode,
    )
    z_after = map_model_to_payload_reference_chart(f0_hat_after, payload)
    fit_after_pure = q_only_clean_cf_fit(
        z_payload=z_after,
        payload=payload,
        t_nodes=state.t_nodes,
        lattice_feature=state.l,
        formula=formula,
        config=pure_config,
        cf_likelihood=None,
        q_init=fit_before.q_star,
    )
    cf_after = 0.0
    if cf_likelihood is not None and fit_after_pure.q_star.numel() > 0:
        cf_after = float(
            cf_likelihood.nll_q(
                payload=payload,
                q=np.asarray(fit_after_pure.q_star.detach().cpu(), dtype=float),
                lattice_feature=state.l.detach().cpu(),
                formula=formula,
            )
        )
    fit_after = replace(
        fit_after_pure,
        cf_nll=float(cf_after),
        cf_nll_start=float(rerank.cf_nll_best),
        cf_delta=float(cf_after - rerank.cf_nll_best),
        score_total=float(fit_after_pure.near_loss),
    )
    accepted = bool(fit_after.near_loss < fit_before.near_loss)
    if debug:
        _algo21_log(
            f"[algo21.step.rerank] done accepted={bool(accepted)} "
            f"witness_before={float(fit_before.witness_sin):.6g} witness_after={float(fit_after.witness_sin):.6g} "
            f"cf_before={float(fit_before.cf_nll):.6g} cf_after={float(fit_after.cf_nll):.6g}"
        )
    return Algorithm21StepResult(
        state_before=state,
        state_candidate=state_candidate,
        state_after=state_candidate if accepted else state,
        accepted=bool(accepted),
        f0_hat_before=f0_hat_before.detach().clone(),
        f0_hat_after=f0_hat_after.detach().clone(),
        f0_hard=f0_hard.detach().clone(),
        f0_star=f0_star.detach().clone(),
        fit_before=fit_before,
        fit_after=fit_after,
        logs=(
            {
                "accepted": bool(accepted),
                "alpha": float(alpha),
                "rerank_radius": float(rerank_radius),
                "rerank_candidate_count": int(rerank_candidate_count),
                "rerank_kept_count": int(rerank.kept_count),
                "score_before": float(fit_before.score_total),
                "score_after": float(fit_after.score_total),
                "witness_before": float(fit_before.witness_sin),
                "witness_after": float(fit_after.witness_sin),
                "cf_before": float(fit_before.cf_nll),
                "cf_after": float(fit_after.cf_nll),
            },
        ),
    )


def algorithm21_should_project(t_value: float, config: Algorithm21Config) -> bool:
    t = float(t_value)
    if t > float(config.t_guide):
        return False
    return any(abs(t - float(t_ref)) <= float(config.projection_time_tol) for t_ref in config.projection_times)


__all__ = [
    "ALGORITHM21_MODE",
    "ALGORITHM21_SHORT_NAME",
    "ALGORITHM21_DESCRIPTION",
    "Algorithm21BranchResult",
    "Algorithm21Config",
    "Algorithm21LocalRerankResult",
    "Algorithm21QFitResult",
    "Algorithm21RankedQCandidate",
    "Algorithm21StepResult",
    "CrystalFormerLikelihood",
    "algorithm21b_project_renoise_step",
    "algorithm21_project_renoise_step",
    "algorithm21_project_renoise_step_from_q",
    "algorithm21_project_renoise_step_local_rerank",
    "algorithm21_should_project",
    "build_crystalformer_reduced_sequence",
    "build_payload_from_template_q",
    "crystalformer_payload_order_assembly_debug",
    "crystalformer_site_representative_search",
    "crystalformer_reduced_sequence_debug",
    "expand_crystalformer_reduced_sequence",
    "expand_q",
    "fit_q_to_clean_prediction",
    "model_to_payload",
    "post_renoise_score",
    "post_renoise_acceptance",
    "predict_clean_f0",
    "rank_q_candidates",
    "score_ranked_q_candidates_with_crystalformer",
    "renoise_from_f0",
    "sample_q_from_crystalformer",
    "species_match_reorder",
    "q_only_clean_cf_fit",
    "q_only_clean_cf_local_rerank",
    "payload_to_model",
    "torus_soft_project",
    "torus_interp",
]
@dataclass(frozen=True)
class Algorithm19State:
    f: torch.Tensor
    v: torch.Tensor
    l: torch.Tensor
    atom_types: torch.Tensor
    node_index: torch.Tensor
    edge_node_index: torch.Tensor
    t_graph: torch.Tensor
    t_nodes: torch.Tensor


def wrap01(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 1.0)


def wrapdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b - torch.round(a - b)


def torus_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrapdiff(a, b).square().mean()


def torus_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torus_mse(a, b).clamp_min(0.0))


def witness_torus_sin_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = wrapdiff(source, target)
    return torch.sin(torch.pi * diff).square().mean()


def _initialize_q_raw(
    *,
    chart: WyckoffDOFChart,
    device: torch.device,
    dtype: torch.dtype,
    q_init: torch.Tensor | None = None,
    q_init_mode: str = "random",
) -> torch.Tensor:
    total_dof = int(chart.total_dof)
    if total_dof <= 0:
        return torch.zeros((0,), device=device, dtype=dtype, requires_grad=True)
    mode = str(q_init_mode).strip().lower()
    if q_init is not None and mode in {"oracle_structure", "oracle_q_init", "previous_live_q", "init"}:
        init = torch.as_tensor(q_init, device=device, dtype=dtype).reshape(-1)
    elif mode in {"reference", "chart_ref"}:
        init = torch.as_tensor(chart.q_ref, device=device, dtype=dtype).reshape(-1)
    else:
        init = torch.rand((total_dof,), device=device, dtype=dtype)
    return init.detach().clone().requires_grad_(True)


def _get_wyckoff_dof_chart(payload: DiffCSPPPSymmetryPayload) -> WyckoffDOFChart:
    if payload.debug_info is None:
        object.__setattr__(payload, "debug_info", {})
    debug_info = payload.debug_info
    chart = debug_info.get("wyckoff_dof_chart")
    cache_version = int(debug_info.get("wyckoff_dof_chart_cache_version", -1))
    if isinstance(chart, WyckoffDOFChart) and cache_version == WYCKOFF_DOF_CHART_CACHE_VERSION:
        return chart
    chart = build_wyckoff_dof_chart(payload)
    debug_info["wyckoff_dof_chart"] = chart
    debug_info["wyckoff_dof_q_ref"] = np.asarray(chart.q_ref, dtype=float)
    debug_info["wyckoff_dof_chart_cache_version"] = WYCKOFF_DOF_CHART_CACHE_VERSION
    return chart


def _payload_debug_array(payload: DiffCSPPPSymmetryPayload, key: str, *, dtype=None):
    debug = payload.debug_info or {}
    value = debug.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _maybe_to_payload_frame(z: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
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
        payload_ref_t = torch.as_tensor(np.asarray(payload.expanded_frac_coords, dtype=float), device=z.device, dtype=z.dtype)
        delta_model = wrapdiff(wrap01(z), model_ref_t)
        delta_payload = delta_model[order_t] @ linear_t
        return wrap01(payload_ref_t + delta_payload)
    tau_t = torch.as_tensor(tau, device=z.device, dtype=z.dtype).reshape(1, 3)
    return wrap01((wrap01(z) - tau_t) @ linear_t)[order_t]


def _maybe_align_payload_local_chart(z_payload: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
    tau = _payload_debug_array(payload, "payload_reference_tau", dtype=float)
    order = _payload_debug_array(payload, "payload_reference_order", dtype=int)
    if tau is None or order is None:
        alignment = align_expanded_frac_to_reference_chart_orbit_aware(payload, z_payload.detach().cpu().numpy(), expanded_atomic_numbers=np.asarray(payload.expanded_atomic_numbers, dtype=int))
        tau = np.asarray(alignment["tau"], dtype=float)
        order = np.asarray(alignment["reference_order"], dtype=int)
    tau_t = torch.as_tensor(tau, device=z_payload.device, dtype=z_payload.dtype).reshape(1, 3)
    order_t = torch.as_tensor(order, device=z_payload.device, dtype=torch.long)
    return wrap01(z_payload + tau_t)[order_t]


def _maybe_unalign_payload_local_chart(z_payload: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
    tau = _payload_debug_array(payload, "payload_reference_tau", dtype=float)
    order = _payload_debug_array(payload, "payload_reference_order", dtype=int)
    if tau is None or order is None:
        return wrap01(z_payload)
    tau_t = torch.as_tensor(tau, device=z_payload.device, dtype=z_payload.dtype).reshape(1, 3)
    order_t = torch.as_tensor(order, device=z_payload.device, dtype=torch.long)
    z_raw_shifted = torch.zeros_like(z_payload)
    z_raw_shifted[order_t] = z_payload
    return wrap01(z_raw_shifted - tau_t)


def _maybe_from_payload_frame(z_payload: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
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
        payload_ref_t = torch.as_tensor(np.asarray(payload.expanded_frac_coords, dtype=float), device=z_payload.device, dtype=z_payload.dtype)
        delta_payload = wrapdiff(wrap01(z_payload), payload_ref_t)
        delta_model = delta_payload @ linear_t
        z_model = wrap01(model_ref_t.clone())
        z_model[assignment_t] = wrap01(model_ref_t[assignment_t] + delta_model)
        return z_model
    tau_t = torch.as_tensor(tau, device=z_payload.device, dtype=z_payload.dtype).reshape(1, 3)
    z_model_scattered = torch.zeros_like(z_payload)
    z_model_scattered[assignment_t] = wrap01(z_payload @ linear_t + tau_t)
    return wrap01(z_model_scattered)


def map_model_to_payload_reference_chart(z_model: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
    return _maybe_align_payload_local_chart(_maybe_to_payload_frame(z_model, payload), payload)


def map_payload_reference_chart_to_model_frame(z_payload: torch.Tensor, payload: DiffCSPPPSymmetryPayload) -> torch.Tensor:
    return _maybe_from_payload_frame(_maybe_unalign_payload_local_chart(z_payload, payload), payload)


def coordinate_score_from_model_output(
    *,
    model,
    preds_v: torch.Tensor,
    v_t: torch.Tensor,
    tau: torch.Tensor,
    node_index: torch.Tensor,
    mode: str = "direct",
) -> torch.Tensor:
    del v_t
    if str(mode).strip().lower() == "direct":
        sigma_norm = model.tdm.sigma_norm_factor(t=tau, index=node_index, ref=preds_v)
        return sigma_norm * preds_v
    raise ValueError(f"Unsupported coordinate_score_mode={mode!r}.")


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


def kldm_renoise_from_f0(
    *,
    model,
    f0_star: torch.Tensor,
    t_nodes: torch.Tensor,
    node_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.tdm.sample_noisy_state(t=t_nodes, f0=f0_star, index=node_index)
